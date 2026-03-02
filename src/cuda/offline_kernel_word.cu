/**
 * Word-Based SimplePIR Offline Kernel
 *
 * Computes hint_0 = A × DB in Z_{2^64}, then modswitches to Z_Q and CRT-packs.
 *
 * A: poly_len × db_rows (u64, row-major)
 * DB: db_cols × db_rows_padded (u16, column-major)
 * Output: poly_len × db_cols (u64, row-major), CRT-packed in Z_Q
 *
 * Two code paths:
 *   Tensor cores (SM≥75): 15 cuBLAS int8×int8→int32 GEMMs with bias correction,
 *                          accumulated in uint64 (no carry loss, no signed bug)
 *   SIMT fallback:        1 CUTLASS uint16×uint64→uint64 GEMM
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        abort(); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status_); \
        abort(); \
    } \
} while(0)

// CUTLASS SIMT: uint16 DB × uint64 query → uint64
using CutlassGemmWord = cutlass::gemm::device::Gemm<
    uint16_t,                               // ElementA (DB)
    cutlass::layout::RowMajor,              // LayoutA
    uint64_t,                               // ElementB (A matrix)
    cutlass::layout::ColumnMajor,           // LayoutB
    uint64_t,                               // ElementC (output)
    cutlass::layout::ColumnMajor,           // LayoutC
    uint64_t,                               // ElementAccumulator
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<uint64_t, 1, uint64_t, uint64_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2
>;

// ---------- Decomposition kernels ----------

__global__ void decompose_u16_bytes(
    int8_t* __restrict__ b0,
    int8_t* __restrict__ b1,
    const uint16_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        b0[idx] = (int8_t)((v & 0xFF) ^ 0x80);
        b1[idx] = (int8_t)(((v >> 8) & 0xFF) ^ 0x80);
    }
}

__global__ void decompose_u64_bytes(
    int8_t* __restrict__ b0, int8_t* __restrict__ b1,
    int8_t* __restrict__ b2, int8_t* __restrict__ b3,
    int8_t* __restrict__ b4, int8_t* __restrict__ b5,
    int8_t* __restrict__ b6, int8_t* __restrict__ b7,
    const uint64_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint64_t v = data[idx];
        b0[idx] = (int8_t)((v & 0xFF) ^ 0x80);
        b1[idx] = (int8_t)(((v >> 8) & 0xFF) ^ 0x80);
        b2[idx] = (int8_t)(((v >> 16) & 0xFF) ^ 0x80);
        b3[idx] = (int8_t)(((v >> 24) & 0xFF) ^ 0x80);
        b4[idx] = (int8_t)(((v >> 32) & 0xFF) ^ 0x80);
        b5[idx] = (int8_t)(((v >> 40) & 0xFF) ^ 0x80);
        b6[idx] = (int8_t)(((v >> 48) & 0xFF) ^ 0x80);
        b7[idx] = (int8_t)(((v >> 56) & 0xFF) ^ 0x80);
    }
}

// ---------- Bias correction helpers ----------

// Row sums of centered int8 matrix (CUBLAS_OP_T layout: col-major K_padded × M,
// row m of transposed = data[k + m*stride] for k in [0,K))
__global__ void compute_row_sums_i8(
    int32_t* __restrict__ sums,
    const int8_t* __restrict__ data,
    size_t M, size_t K, size_t stride)
{
    size_t m = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    int32_t s = 0;
    for (size_t k = 0; k < K; k++)
        s += (int32_t)data[k + m * stride];
    sums[m] = s;
}

// Column sums of centered int8 matrix (CUBLAS_OP_N layout: col-major K × N,
// col n = data[k + n*K] for k in [0,K))
__global__ void compute_col_sums_i8(
    int32_t* __restrict__ sums,
    const int8_t* __restrict__ data,
    size_t N, size_t K)
{
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t s = 0;
    for (size_t k = 0; k < K; k++)
        s += (int32_t)data[k + n * K];
    sums[n] = s;
}

// Accumulate one GEMM result into uint64 with bias correction and shift.
// unsigned_dot[m,n] = signed_dot[m,n] + 128*RS[m] + 128*CS[n] + 16384*K
// accum[idx] += unsigned_dot * shift
__global__ void accumulate_shifted_u64(
    uint64_t* __restrict__ accum,
    const int32_t* __restrict__ gemm_out,
    const int32_t* __restrict__ row_sums,
    const int32_t* __restrict__ col_sums,
    uint64_t shift,
    size_t MN, size_t M, size_t K)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= MN) return;
    size_t m = idx % M;  // col-major: m varies fastest
    size_t n = idx / M;
    int64_t corrected = (int64_t)gemm_out[idx]
                        + (int64_t)128 * (int64_t)row_sums[m]
                        + (int64_t)128 * (int64_t)col_sums[n]
                        + (int64_t)16384 * (int64_t)K;
    accum[idx] += (uint64_t)corrected * shift;
}

// ---------- Modswitch u64 → Z_Q with col-major to row-major transpose ----------

// SIMT path: modswitch u64 result to Z_Q
__global__ void modswitch_crt_u64(
    uint64_t* __restrict__ out,        // poly_len * db_cols (row-major)
    const uint64_t* __restrict__ in,   // db_cols * poly_len (col-major from CUTLASS)
    size_t db_cols, size_t poly_len,
    uint64_t q, uint64_t mod0, uint64_t mod1,
    uint64_t inv_n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = db_cols * poly_len;
    if (idx >= total) return;

    size_t col = idx % db_cols;
    size_t row = idx / db_cols;
    size_t cm_idx = col + row * db_cols;  // ColMajor M×N: element(i,j) = data[i + j*M]

    uint64_t val = in[cm_idx];

    __uint128_t prod = (__uint128_t)val * q + (((__uint128_t)1) << 63);
    uint64_t val_q = (uint64_t)(prod >> 64);

    // Multiply by inv_N mod Q: compensates for CDKS packing multiplying by N
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);

    out[row * db_cols + col] = val_q;
}

// ---------- Context ----------

struct GemmSpec {
    int db_b;       // DB byte index (0 or 1)
    int q_b;        // query/A byte index (0..7)
    uint64_t shift; // 256^{db_b + q_b} = 1ULL << (8*(db_b + q_b))
};

static const GemmSpec WORD_GEMM_SPECS[15] = {
    {0, 0, 1ULL},
    {0, 1, 1ULL << 8},
    {1, 0, 1ULL << 8},
    {0, 2, 1ULL << 16},
    {1, 1, 1ULL << 16},
    {0, 3, 1ULL << 24},
    {1, 2, 1ULL << 24},
    {0, 4, 1ULL << 32},
    {1, 3, 1ULL << 32},
    {0, 5, 1ULL << 40},
    {1, 4, 1ULL << 40},
    {0, 6, 1ULL << 48},
    {1, 5, 1ULL << 48},
    {0, 7, 1ULL << 56},
    {1, 6, 1ULL << 56},
};

struct WordOfflineContext {
    bool has_tensor_cores;
    cublasHandle_t cublas_handle;

    // Tensor core path
    int8_t* d_db_bytes[2];      // DB decomposed (centered), each M*K_padded int8
    int8_t* d_A_bytes[8];       // A decomposed (centered), each K*N int8
    int32_t* d_temp;            // Single temp buffer for cuBLAS output, M*N int32
    uint64_t* d_accum_u64;      // uint64 accumulator, M*N
    int32_t* d_db_row_sums[2];  // Row sums of centered DB bytes, M int32 each
    int32_t* d_A_col_sums[8];   // Col sums of centered A bytes, N int32 each

    // SIMT path
    uint16_t* d_db_u16;        // DB as-is
    uint64_t* d_A_u64;         // A as-is
    uint64_t* d_result_u64;    // GEMM output
    void* d_gemm_workspace;

    // Output
    uint64_t* d_out;           // poly_len * db_cols, CRT-packed

    uint32_t db_cols;   // M
    uint32_t poly_len;  // N
    uint32_t db_rows;   // K
    uint32_t db_rows_padded;

    uint64_t modulus;   // Q
    uint64_t mod0;
    uint64_t mod1;
    uint64_t inv_n;     // N^{-1} mod Q, compensates CDKS N-multiplication
};

extern "C" {

void* init_word_offline_context(
    const uint16_t* db,          // db_cols × db_rows_padded, col-major
    const uint64_t* A,           // poly_len × db_rows, row-major (A[i*db_rows+j])
    uint32_t db_rows,
    uint32_t db_rows_padded,
    uint32_t db_cols,
    uint32_t poly_len,
    uint64_t modulus,
    uint64_t mod0,
    uint64_t mod1,
    uint64_t inv_n)
{
    WordOfflineContext* ctx = new WordOfflineContext();
    ctx->db_cols = db_cols;
    ctx->poly_len = poly_len;
    ctx->db_rows = db_rows;
    ctx->db_rows_padded = db_rows_padded;
    ctx->modulus = modulus;
    ctx->mod0 = mod0;
    ctx->mod1 = mod1;
    ctx->inv_n = inv_n;
    ctx->has_tensor_cores = false;
    ctx->cublas_handle = nullptr;
    ctx->d_db_u16 = nullptr;
    ctx->d_A_u64 = nullptr;
    ctx->d_result_u64 = nullptr;
    ctx->d_gemm_workspace = nullptr;
    ctx->d_out = nullptr;
    ctx->d_temp = nullptr;
    ctx->d_accum_u64 = nullptr;
    for (int i = 0; i < 2; i++) { ctx->d_db_bytes[i] = nullptr; ctx->d_db_row_sums[i] = nullptr; }
    for (int i = 0; i < 8; i++) { ctx->d_A_bytes[i] = nullptr; ctx->d_A_col_sums[i] = nullptr; }

    size_t M = db_cols;
    size_t N = poly_len;
    size_t K = db_rows;

    // Detect tensor cores
    {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        int major, minor;
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 75);

        // int8×int8→int32: max safe K before overflow = 2^31 / (127*127) ≈ 133000
        // Databases up to ~32GB (K≈133000) are fine without K-tiling.
        if (ctx->has_tensor_cores && db_rows > 133000) {
            fprintf(stderr, "ERROR: db_rows=%u exceeds max safe K=%d for int8 tensor core accumulation. "
                    "K-tiling not yet implemented.\n", db_rows, 133000);
            delete ctx;
            return nullptr;
        }

        printf("Word offline GEMM (%s): M=%u, N=%u, K=%u\n",
               ctx->has_tensor_cores ? "tensor core" : "CUTLASS SIMT",
               db_cols, poly_len, db_rows);
    }

    // Allocate output
    CUDA_CHECK(cudaMalloc(&ctx->d_out, M * N * sizeof(uint64_t)));

    if (ctx->has_tensor_cores) {
        // Upload DB, decompose to 2 byte slices
        // DB is col-major db_cols × db_rows_padded as u16
        // For GEMM we need DB as RowMajor M×K = db_cols × db_rows
        // col-major db_cols × db_rows_padded with lda=db_rows_padded IS RowMajor with lda=db_rows_padded
        // cuBLAS: OP_T on col-major K×M = row-major M×K
        // So upload as-is, use OP_T with lda=db_rows_padded
        uint16_t* d_db_raw;
        CUDA_CHECK(cudaMalloc(&d_db_raw, M * db_rows_padded * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(d_db_raw, db, M * db_rows_padded * sizeof(uint16_t), cudaMemcpyHostToDevice));

        size_t db_elems = M * db_rows_padded;
        for (int i = 0; i < 2; i++)
            CUDA_CHECK(cudaMalloc(&ctx->d_db_bytes[i], db_elems));
        {
            int threads = 256;
            int blocks = (db_elems + threads - 1) / threads;
            decompose_u16_bytes<<<blocks, threads>>>(ctx->d_db_bytes[0], ctx->d_db_bytes[1], d_db_raw, db_elems);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(d_db_raw));

        // Compute row sums of centered DB bytes (for bias correction)
        for (int i = 0; i < 2; i++) {
            CUDA_CHECK(cudaMalloc(&ctx->d_db_row_sums[i], M * sizeof(int32_t)));
            int t = 256, b = (M + t - 1) / t;
            compute_row_sums_i8<<<b, t>>>(ctx->d_db_row_sums[i], ctx->d_db_bytes[i],
                                          M, K, ctx->db_rows_padded);
            CUDA_CHECK(cudaGetLastError());
        }

        // Upload A (poly_len × db_rows, row-major), decompose to 8 byte slices
        // For GEMM: B is ColMajor K×N = db_rows × poly_len
        // A[i*db_rows+j] is row-major poly_len × db_rows = ColMajor db_rows × poly_len
        // So upload as-is, ldb = db_rows
        uint64_t* d_A_raw;
        size_t A_elems = N * K;
        CUDA_CHECK(cudaMalloc(&d_A_raw, A_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_A_raw, A, A_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        for (int i = 0; i < 8; i++)
            CUDA_CHECK(cudaMalloc(&ctx->d_A_bytes[i], A_elems));
        {
            int threads = 256;
            int blocks = (A_elems + threads - 1) / threads;
            decompose_u64_bytes<<<blocks, threads>>>(
                ctx->d_A_bytes[0], ctx->d_A_bytes[1], ctx->d_A_bytes[2], ctx->d_A_bytes[3],
                ctx->d_A_bytes[4], ctx->d_A_bytes[5], ctx->d_A_bytes[6], ctx->d_A_bytes[7],
                d_A_raw, A_elems);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(d_A_raw));

        // Compute column sums of centered A bytes (for bias correction)
        for (int i = 0; i < 8; i++) {
            CUDA_CHECK(cudaMalloc(&ctx->d_A_col_sums[i], N * sizeof(int32_t)));
            int t = 256, b = (N + t - 1) / t;
            compute_col_sums_i8<<<b, t>>>(ctx->d_A_col_sums[i], ctx->d_A_bytes[i], N, K);
            CUDA_CHECK(cudaGetLastError());
        }

        // Allocate temp buffer (one GEMM output) and uint64 accumulator
        CUDA_CHECK(cudaMalloc(&ctx->d_temp, M * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_accum_u64, M * N * sizeof(uint64_t)));

        CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(ctx->cublas_handle, CUBLAS_TENSOR_OP_MATH));

    } else {
        // SIMT path: keep DB as u16, A as u64
        CUDA_CHECK(cudaMalloc(&ctx->d_db_u16, M * db_rows_padded * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(ctx->d_db_u16, db, M * db_rows_padded * sizeof(uint16_t), cudaMemcpyHostToDevice));

        size_t A_elems = N * K;
        CUDA_CHECK(cudaMalloc(&ctx->d_A_u64, A_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(ctx->d_A_u64, A, A_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx->d_result_u64, M * N * sizeof(uint64_t)));

        // Pre-allocate CUTLASS workspace
        // M=db_cols, N=poly_len, K=db_rows
        // A=DB: uint16 RowMajor lda=db_rows_padded
        // B=A_matrix: uint64 ColMajor ldb=db_rows
        // C=result: uint64 ColMajor ldc=db_cols
        cutlass::gemm::GemmCoord problem_size(M, N, K);
        uint64_t alpha = 1, beta = 0;
        CutlassGemmWord::Arguments args{
            problem_size,
            {ctx->d_db_u16, (int)db_rows_padded},
            {ctx->d_A_u64, (int)K},
            {ctx->d_result_u64, (int)M},
            {ctx->d_result_u64, (int)M},
            {alpha, beta}, 1
        };
        size_t ws = CutlassGemmWord::get_workspace_size(args);
        if (ws > 0)
            CUDA_CHECK(cudaMalloc(&ctx->d_gemm_workspace, ws));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return ctx;
}

int compute_hint_0_word_gpu(void* context, uint64_t* hint_0_out)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return -1;

    size_t M = ctx->db_cols;
    size_t N = ctx->poly_len;
    size_t K = ctx->db_rows;
    size_t total = M * N;

    if (ctx->has_tensor_cores) {
        // Zero the uint64 accumulator
        CUDA_CHECK(cudaMemset(ctx->d_accum_u64, 0, total * sizeof(uint64_t)));

        // 15 cuBLAS int8 GEMMs with bias correction + uint64 accumulation
        // Each GEMM: alpha=1, beta=0 → d_temp, then accumulate with shift
        int32_t alpha_one = 1, beta_zero = 0;
        for (int g = 0; g < 15; g++) {
            const GemmSpec& s = WORD_GEMM_SPECS[g];
            CUBLAS_CHECK(cublasGemmEx(ctx->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)M, (int)N, (int)K,
                &alpha_one,
                ctx->d_db_bytes[s.db_b], CUDA_R_8I, (int)ctx->db_rows_padded,
                ctx->d_A_bytes[s.q_b],   CUDA_R_8I, (int)K,
                &beta_zero,
                ctx->d_temp,             CUDA_R_32I, (int)M,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            accumulate_shifted_u64<<<blocks, threads>>>(
                ctx->d_accum_u64, ctx->d_temp,
                ctx->d_db_row_sums[s.db_b],
                ctx->d_A_col_sums[s.q_b],
                s.shift, total, M, K);
            CUDA_CHECK(cudaGetLastError());
        }

        // Modswitch u64 → Z_Q (reuse SIMT modswitch kernel, same transpose)
        {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            modswitch_crt_u64<<<blocks, threads>>>(
                ctx->d_out, ctx->d_accum_u64,
                M, N, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        // CUTLASS SIMT: single uint16 × uint64 → uint64
        cutlass::gemm::GemmCoord problem_size(M, N, K);
        uint64_t alpha = 1, beta = 0;
        CutlassGemmWord::Arguments args{
            problem_size,
            {ctx->d_db_u16, (int)ctx->db_rows_padded},
            {ctx->d_A_u64, (int)K},
            {ctx->d_result_u64, (int)M},
            {ctx->d_result_u64, (int)M},
            {alpha, beta}, 1
        };
        CutlassGemmWord gemm_op;
        cutlass::Status status = gemm_op.initialize(args, ctx->d_gemm_workspace);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
            return -1;
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
            return -1;
        }

        // Modswitch + multiply by inv_N
        {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            modswitch_crt_u64<<<blocks, threads>>>(
                ctx->d_out, ctx->d_result_u64,
                M, N, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result to host (row-major poly_len × db_cols, CRT-packed)
    CUDA_CHECK(cudaMemcpy(hint_0_out, ctx->d_out, total * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    return 0;
}

void free_word_offline_context(void* context)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return;

    for (int i = 0; i < 2; i++) {
        if (ctx->d_db_bytes[i]) cudaFree(ctx->d_db_bytes[i]);
        if (ctx->d_db_row_sums[i]) cudaFree(ctx->d_db_row_sums[i]);
    }
    for (int i = 0; i < 8; i++) {
        if (ctx->d_A_bytes[i]) cudaFree(ctx->d_A_bytes[i]);
        if (ctx->d_A_col_sums[i]) cudaFree(ctx->d_A_col_sums[i]);
    }
    if (ctx->d_temp) cudaFree(ctx->d_temp);
    if (ctx->d_accum_u64) cudaFree(ctx->d_accum_u64);
    if (ctx->d_db_u16) cudaFree(ctx->d_db_u16);
    if (ctx->d_A_u64) cudaFree(ctx->d_A_u64);
    if (ctx->d_result_u64) cudaFree(ctx->d_result_u64);
    if (ctx->d_gemm_workspace) cudaFree(ctx->d_gemm_workspace);
    if (ctx->d_out) cudaFree(ctx->d_out);
    if (ctx->cublas_handle) cublasDestroy(ctx->cublas_handle);

    delete ctx;
}

} // extern "C"
