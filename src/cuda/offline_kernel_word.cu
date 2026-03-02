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
 *   Tensor cores (SM>=80): 2 CUTLASS uint8 GEMMs (one per DB byte) + fused accumulate
 *   SIMT fallback:         1 CUTLASS uint16×uint64→uint64 GEMM
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
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

// CUTLASS Tensor Core: uint8 × uint8 → int32

// SM80+ (Ampere): wider instruction shape
using CutlassGemmU8TC_Sm80 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3
>;

// SM75 (Turing): u8 IMMA with 8×8×16 instruction shape
using CutlassGemmU8TC_Sm75 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2
>;

// Helper: run the correct TC GEMM based on SM version
static cutlass::Status run_u8tc_gemm(bool is_sm80,
    int M, int wide_N, int K,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    int32_t* C, int ldc)
{
    int32_t alpha = 1, beta = 0;
    cutlass::gemm::GemmCoord problem_size(M, wide_N, K);
    if (is_sm80) {
        CutlassGemmU8TC_Sm80::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm80 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    } else {
        CutlassGemmU8TC_Sm75::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm75 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    }
}

// ---------- Decomposition kernels ----------

__global__ void decompose_u16_bytes(
    uint8_t* __restrict__ b0,
    uint8_t* __restrict__ b1,
    const uint16_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        b0[idx] = (uint8_t)(v & 0xFF);
        b1[idx] = (uint8_t)((v >> 8) & 0xFF);
    }
}

// Decompose u64 → 8 uint8 byte slices, packed contiguously.
// Input: K × N column-major. Output: K × (8*N) column-major with stride K.
__global__ void decompose_u64_bytes_packed(
    uint8_t* __restrict__ out,
    const uint64_t* __restrict__ data,
    size_t K, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    uint64_t v = data[idx];
    out[idx + 0*K*N] = (uint8_t)(v & 0xFF);
    out[idx + 1*K*N] = (uint8_t)((v >> 8) & 0xFF);
    out[idx + 2*K*N] = (uint8_t)((v >> 16) & 0xFF);
    out[idx + 3*K*N] = (uint8_t)((v >> 24) & 0xFF);
    out[idx + 4*K*N] = (uint8_t)((v >> 32) & 0xFF);
    out[idx + 5*K*N] = (uint8_t)((v >> 40) & 0xFF);
    out[idx + 6*K*N] = (uint8_t)((v >> 48) & 0xFF);
    out[idx + 7*K*N] = (uint8_t)((v >> 56) & 0xFF);
}

// ---------- Fused accumulate kernels ----------

// Accumulate 8 specs from db_b=0 GEMM output into uint64 accumulator.
__global__ void accumulate_db0(
    uint64_t* __restrict__ accum,           // M × N col-major
    const int32_t* __restrict__ gemm_out,   // M × (8*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    uint64_t acc = 0;
    acc += (uint64_t)(uint32_t)gemm_out[m + (0*N+n)*M];
    acc += (uint64_t)(uint32_t)gemm_out[m + (1*N+n)*M] << 8;
    acc += (uint64_t)(uint32_t)gemm_out[m + (2*N+n)*M] << 16;
    acc += (uint64_t)(uint32_t)gemm_out[m + (3*N+n)*M] << 24;
    acc += (uint64_t)(uint32_t)gemm_out[m + (4*N+n)*M] << 32;
    acc += (uint64_t)(uint32_t)gemm_out[m + (5*N+n)*M] << 40;
    acc += (uint64_t)(uint32_t)gemm_out[m + (6*N+n)*M] << 48;
    acc += (uint64_t)(uint32_t)gemm_out[m + (7*N+n)*M] << 56;
    accum[idx] = acc;
}

// Accumulate 7 specs from db_b=1, add to existing accumulator.
__global__ void accumulate_db1(
    uint64_t* __restrict__ accum,           // M × N col-major (has db0 partial)
    const int32_t* __restrict__ gemm_out,   // M × (8*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    uint64_t acc = accum[idx];
    acc += (uint64_t)(uint32_t)gemm_out[m + (0*N+n)*M] << 8;
    acc += (uint64_t)(uint32_t)gemm_out[m + (1*N+n)*M] << 16;
    acc += (uint64_t)(uint32_t)gemm_out[m + (2*N+n)*M] << 24;
    acc += (uint64_t)(uint32_t)gemm_out[m + (3*N+n)*M] << 32;
    acc += (uint64_t)(uint32_t)gemm_out[m + (4*N+n)*M] << 40;
    acc += (uint64_t)(uint32_t)gemm_out[m + (5*N+n)*M] << 48;
    acc += (uint64_t)(uint32_t)gemm_out[m + (6*N+n)*M] << 56;
    accum[idx] = acc;
}

// ---------- Modswitch u64 → Z_Q with col-major to row-major transpose ----------

__global__ void modswitch_crt_u64(
    uint64_t* __restrict__ out,        // poly_len * db_cols (row-major)
    const uint64_t* __restrict__ in,   // db_cols * poly_len (col-major)
    size_t db_cols, size_t poly_len,
    uint64_t q, uint64_t mod0, uint64_t mod1,
    uint64_t inv_n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = db_cols * poly_len;
    if (idx >= total) return;

    size_t col = idx % db_cols;
    size_t row = idx / db_cols;
    size_t cm_idx = col + row * db_cols;

    uint64_t val = in[cm_idx];

    __uint128_t prod = (__uint128_t)val * q + (((__uint128_t)1) << 63);
    uint64_t val_q = (uint64_t)(prod >> 64);
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);

    out[row * db_cols + col] = val_q;
}

// ---------- Context ----------

struct WordOfflineContext {
    bool has_tensor_cores;
    bool is_sm80;

    // Tensor core path (CUTLASS uint8, 2 wide GEMMs)
    uint8_t* d_db_bytes[2];        // DB decomposed bytes, M×K_padded
    uint8_t* d_A_bytes_packed;     // A decomposed bytes, K × (8*N) contiguous
    int32_t* d_gemm_out;           // M × (8*N) int32, reused per DB byte
    uint64_t* d_accum_u64;         // M × N uint64 accumulator

    // SIMT path
    uint16_t* d_db_u16;
    uint64_t* d_A_u64;
    uint64_t* d_result_u64;
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
    uint64_t inv_n;
};

extern "C" {

void* init_word_offline_context(
    const uint16_t* db,
    const uint64_t* A,
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
    ctx->d_db_u16 = nullptr;
    ctx->d_A_u64 = nullptr;
    ctx->d_result_u64 = nullptr;
    ctx->d_gemm_workspace = nullptr;
    ctx->d_out = nullptr;
    ctx->d_gemm_out = nullptr;
    ctx->d_accum_u64 = nullptr;
    ctx->d_A_bytes_packed = nullptr;
    for (int i = 0; i < 2; i++) ctx->d_db_bytes[i] = nullptr;

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
        ctx->is_sm80 = (sm >= 80);

        printf("Word offline GEMM (%s): M=%u, N=%u, K=%u\n",
               ctx->has_tensor_cores ? (ctx->is_sm80 ? "CUTLASS TC Sm80" : "CUTLASS TC Sm75") : "CUTLASS SIMT",
               db_cols, poly_len, db_rows);
    }

    // Allocate output
    CUDA_CHECK(cudaMalloc(&ctx->d_out, M * N * sizeof(uint64_t)));

    if (ctx->has_tensor_cores) {
        // Upload DB, decompose to 2 uint8 byte slices
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

        // Upload A, decompose to packed uint8 bytes: K × (8*N)
        size_t A_elems = N * K;
        uint64_t* d_A_raw;
        CUDA_CHECK(cudaMalloc(&d_A_raw, A_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_A_raw, A, A_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx->d_A_bytes_packed, K * 8 * N));
        {
            int threads = 256;
            int blocks = (A_elems + threads - 1) / threads;
            decompose_u64_bytes_packed<<<blocks, threads>>>(ctx->d_A_bytes_packed, d_A_raw, K, N);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaFree(d_A_raw));

        // Allocate GEMM output (reused) and uint64 accumulator
        CUDA_CHECK(cudaMalloc(&ctx->d_gemm_out, M * 8 * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&ctx->d_accum_u64, M * N * sizeof(uint64_t)));

    } else {
        // SIMT path: keep DB as u16, A as u64
        CUDA_CHECK(cudaMalloc(&ctx->d_db_u16, M * db_rows_padded * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemcpy(ctx->d_db_u16, db, M * db_rows_padded * sizeof(uint16_t), cudaMemcpyHostToDevice));

        size_t A_elems = N * K;
        CUDA_CHECK(cudaMalloc(&ctx->d_A_u64, A_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(ctx->d_A_u64, A, A_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&ctx->d_result_u64, M * N * sizeof(uint64_t)));

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
        // GEMM 0: DB[0]^T × A_packed → gemm_out  (M × 8N)
        {
            auto status = run_u8tc_gemm(ctx->is_sm80, M, 8*N, K,
                ctx->d_db_bytes[0], (int)ctx->db_rows_padded,
                ctx->d_A_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)M);
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC GEMM 0 failed\n"); return -1; }
        }
        { int threads = 256, blocks = (total + threads - 1) / threads;
          accumulate_db0<<<blocks, threads>>>(ctx->d_accum_u64, ctx->d_gemm_out, M, N);
          CUDA_CHECK(cudaGetLastError()); }

        // GEMM 1: DB[1]^T × A_packed → gemm_out  (M × 8N, reuse)
        {
            auto status = run_u8tc_gemm(ctx->is_sm80, M, 8*N, K,
                ctx->d_db_bytes[1], (int)ctx->db_rows_padded,
                ctx->d_A_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)M);
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC GEMM 1 failed\n"); return -1; }
        }
        { int threads = 256, blocks = (total + threads - 1) / threads;
          accumulate_db1<<<blocks, threads>>>(ctx->d_accum_u64, ctx->d_gemm_out, M, N);
          CUDA_CHECK(cudaGetLastError()); }

        // Modswitch + transpose: col-major M×N → row-major N×M
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
    }
    if (ctx->d_A_bytes_packed) cudaFree(ctx->d_A_bytes_packed);
    if (ctx->d_gemm_out) cudaFree(ctx->d_gemm_out);
    if (ctx->d_accum_u64) cudaFree(ctx->d_accum_u64);
    if (ctx->d_db_u16) cudaFree(ctx->d_db_u16);
    if (ctx->d_A_u64) cudaFree(ctx->d_A_u64);
    if (ctx->d_result_u64) cudaFree(ctx->d_result_u64);
    if (ctx->d_gemm_workspace) cudaFree(ctx->d_gemm_workspace);
    if (ctx->d_out) cudaFree(ctx->d_out);

    delete ctx;
}

} // extern "C"
