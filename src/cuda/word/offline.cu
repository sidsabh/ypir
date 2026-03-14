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

// Decompose uint16 DB into vertically stacked (2M)×K_padded row-major uint8:
// First M rows = low bytes, next M rows = high bytes
__global__ void decompose_u16_stacked(
    uint8_t* __restrict__ out,          // (2M) × K_padded, row-major
    const uint16_t* __restrict__ data,  // M × K_padded, row-major
    size_t count)                       // M * K_padded
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        out[idx] = (uint8_t)(v & 0xFF);
        out[count + idx] = (uint8_t)((v >> 8) & 0xFF);
    }
}

// Two-pass in-place decomposition:
// Pass 1 (high bytes): read uint16 from buf, write high byte to buf[count + idx]
//   Safe because uint16 input lives in [0, 2*count) bytes and high output in [count, 2*count).
//   For idx < count/2, reads from [2*idx, 2*idx+2) and writes to [count+idx] — no overlap.
//   For idx >= count/2, reads from [2*idx, 2*idx+2) which is in [count, 2*count),
//     but writes to [count+idx] which is >= 1.5*count — past the read region. Safe.
__global__ void decompose_u16_extract_high(
    uint8_t* buf,       // 2*count bytes, first count*sizeof(uint16_t) = 2*count bytes hold uint16 data
    size_t count)       // M * K_padded
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = ((const uint16_t*)buf)[idx];
        buf[count + idx] = (uint8_t)((v >> 8) & 0xFF);
    }
}

// Pass 2 (low bytes): read uint16 from buf, write low byte to buf[idx]
//   After pass 1, high bytes are safely stored in [count, 2*count).
//   This pass overwrites [0, count) which destroys the original uint16 data, but we only need the low byte.
__global__ void decompose_u16_extract_low(
    uint8_t* buf,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = ((const uint16_t*)buf)[idx];
        buf[idx] = (uint8_t)(v & 0xFF);
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

// ---------- Fused stacked accumulate kernels ----------

// Accumulate stacked GEMM output (2M)×(8N) → M×N uint64. (used when full GEMM fits in memory)
// gemm_out is col-major (2M)×(8N), stride 2M.
// Rows [0,M) = db_b0 products, rows [M,2M) = db_b1 products.
// db_b0: 8 terms (shifts 0,8,...,56), db_b1: 7 terms (shifts 8,...,56).
__global__ void accumulate_db16_stacked(
    uint64_t* __restrict__ accum,          // M × N col-major
    const int32_t* __restrict__ gemm_out,  // (2M) × (8*N) col-major, stride 2M
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    size_t stride = 2 * M;
    uint64_t acc = 0;
    #pragma unroll
    for (int q = 0; q < 8; q++)
        acc += (uint64_t)(uint32_t)gemm_out[m + (q*N+n)*stride] << (8*q);
    #pragma unroll
    for (int q = 0; q < 7; q++)
        acc += (uint64_t)(uint32_t)gemm_out[M + m + (q*N+n)*stride] << (8*(q+1));
    accum[idx] = acc;
}

// Tiled accumulate: one A-byte slice at a time.
// gemm_out is col-major (2M)×N, stride 2M. a_byte_idx is which A byte (0..7).
// db_b0 contributes shift = 8*a_byte_idx, db_b1 contributes shift = 8*(a_byte_idx+1).
// If is_first, writes to accum; otherwise adds to existing accum.
__global__ void accumulate_db16_one_byte(
    uint64_t* __restrict__ accum,          // M × N col-major
    const int32_t* __restrict__ gemm_out,  // (2M) × N col-major, stride 2M
    size_t M, size_t N,
    int a_byte_idx, bool is_first)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    size_t stride = 2 * M;
    // db_b0 (low byte of DB) × A_byte[a_byte_idx] → shift by 8*a_byte_idx
    uint64_t acc = (uint64_t)(uint32_t)gemm_out[m + n * stride] << (8 * a_byte_idx);
    // db_b1 (high byte of DB) × A_byte[a_byte_idx] → shift by 8*(a_byte_idx+1)
    if (a_byte_idx < 7)  // db_b1 doesn't contribute at a_byte_idx=7 (would shift by 64)
        acc += (uint64_t)(uint32_t)gemm_out[M + m + n * stride] << (8 * (a_byte_idx + 1));
    if (is_first)
        accum[idx] = acc;
    else
        accum[idx] += acc;
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

    // Tensor core path (CUTLASS uint8, 1 stacked GEMM)
    uint8_t* d_db_stacked;         // DB decomposed+stacked bytes, (2M)×K_padded
    uint8_t* d_A_bytes_packed;     // A decomposed bytes, K × (8*N) contiguous
    int32_t* d_gemm_out;           // (2M) × (8*N) int32
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
    ctx->d_db_stacked = nullptr;

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
        // Upload DB and decompose to vertically stacked uint8 bytes — in-place
        // d_db_stacked is 2*db_elems bytes; raw uint16 data is also 2*db_elems bytes.
        // Upload raw data into d_db_stacked, then two-pass decompose in-place (no extra alloc).
        size_t db_elems = M * db_rows_padded;
        CUDA_CHECK(cudaMalloc(&ctx->d_db_stacked, 2 * db_elems));
        CUDA_CHECK(cudaMemcpy(ctx->d_db_stacked, db, M * db_rows_padded * sizeof(uint16_t), cudaMemcpyHostToDevice));
        {
            int threads = 256;
            int blocks = (db_elems + threads - 1) / threads;
            // Pass 1: extract high bytes to second half (doesn't destroy uint16 data)
            decompose_u16_extract_high<<<blocks, threads>>>(ctx->d_db_stacked, db_elems);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            // Pass 2: extract low bytes, overwriting first half
            decompose_u16_extract_low<<<blocks, threads>>>(ctx->d_db_stacked, db_elems);
            CUDA_CHECK(cudaGetLastError());
        }

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

        // d_gemm_out and d_accum_u64 are allocated lazily in compute_hint_0_word_gpu
        // to avoid peak memory spike during init.

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
        // Lazily allocate accumulator
        if (!ctx->d_accum_u64)
            CUDA_CHECK(cudaMalloc(&ctx->d_accum_u64, M * N * sizeof(uint64_t)));

        // Check if we can fit the full stacked GEMM output
        size_t full_gemm_bytes = 2 * M * 8 * N * sizeof(int32_t);
        size_t tiled_gemm_bytes = 2 * M * N * sizeof(int32_t);  // one A-byte at a time
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

        bool use_tiled_gemm = (free_mem < full_gemm_bytes + (256ULL << 20));

        if (use_tiled_gemm) {
            // Tiled path: 8 GEMMs, one per A-byte slice, d_gemm_out = (2M)×N
            printf("Word offline GEMM: tiled path (%.1f MB free, need %.1f MB full)\n",
                   free_mem / 1e6, full_gemm_bytes / 1e6);
            if (!ctx->d_gemm_out)
                CUDA_CHECK(cudaMalloc(&ctx->d_gemm_out, tiled_gemm_bytes));
            for (int q = 0; q < 8; q++) {
                // B pointer: q-th byte slice of A, starting at offset q*K*N in d_A_bytes_packed
                const uint8_t* A_byte_q = ctx->d_A_bytes_packed + (size_t)q * K * N;
                auto status = run_u8tc_gemm(ctx->is_sm80, 2*M, N, K,
                    ctx->d_db_stacked, (int)ctx->db_rows_padded,
                    A_byte_q, (int)K,
                    ctx->d_gemm_out, (int)(2*M));
                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "CUTLASS TC tiled GEMM (byte %d) failed\n", q);
                    return -1;
                }
                int threads = 256, blocks = (total + threads - 1) / threads;
                accumulate_db16_one_byte<<<blocks, threads>>>(
                    ctx->d_accum_u64, ctx->d_gemm_out, M, N, q, (q == 0));
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            // Fast path: single stacked GEMM, (2M) × (8N) output
            if (!ctx->d_gemm_out)
                CUDA_CHECK(cudaMalloc(&ctx->d_gemm_out, full_gemm_bytes));
            auto status = run_u8tc_gemm(ctx->is_sm80, 2*M, 8*N, K,
                ctx->d_db_stacked, (int)ctx->db_rows_padded,
                ctx->d_A_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)(2*M));
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC stacked GEMM failed\n"); return -1; }
            int threads = 256, blocks = (total + threads - 1) / threads;
            accumulate_db16_stacked<<<blocks, threads>>>(ctx->d_accum_u64, ctx->d_gemm_out, M, N);
            CUDA_CHECK(cudaGetLastError());
        }

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

// Get device pointer to d_out (hint_0 on GPU) without copying to host.
// Returns nullptr if context is null.
uint64_t* ypir_word_offline_get_hint_device_ptr(void* context)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return nullptr;
    return ctx->d_out;
}

// Transfer ownership of d_out to caller (won't be freed by free_word_offline_context).
uint64_t* ypir_word_offline_take_hint_device_ptr(void* context)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return nullptr;
    uint64_t* ptr = ctx->d_out;
    ctx->d_out = nullptr;  // prevent double-free
    return ptr;
}

// Free GEMM-specific buffers that are no longer needed after compute_hint_0_word_gpu().
// Keeps d_db_stacked, d_db_u16, and d_out alive.
void ypir_word_offline_free_gemm_buffers(void* context)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return;

    // Tensor core path
    if (ctx->d_A_bytes_packed) { cudaFree(ctx->d_A_bytes_packed); ctx->d_A_bytes_packed = nullptr; }
    if (ctx->d_gemm_out)       { cudaFree(ctx->d_gemm_out);       ctx->d_gemm_out = nullptr; }
    if (ctx->d_accum_u64)      { cudaFree(ctx->d_accum_u64);      ctx->d_accum_u64 = nullptr; }

    // SIMT path
    if (ctx->d_A_u64)          { cudaFree(ctx->d_A_u64);          ctx->d_A_u64 = nullptr; }
    if (ctx->d_result_u64)     { cudaFree(ctx->d_result_u64);     ctx->d_result_u64 = nullptr; }
    if (ctx->d_gemm_workspace) { cudaFree(ctx->d_gemm_workspace); ctx->d_gemm_workspace = nullptr; }
}

// Transfer ownership of DB device pointers to caller (won't be freed by free_word_offline_context).
void ypir_word_offline_take_db_device_ptrs(void* context, uint8_t** out_db_stacked, uint16_t** out_db_u16)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) { *out_db_stacked = nullptr; *out_db_u16 = nullptr; return; }
    *out_db_stacked = ctx->d_db_stacked;
    *out_db_u16 = ctx->d_db_u16;
    ctx->d_db_stacked = nullptr;
    ctx->d_db_u16 = nullptr;
}

void free_word_offline_context(void* context)
{
    WordOfflineContext* ctx = (WordOfflineContext*)context;
    if (!ctx) return;

    if (ctx->d_db_stacked) cudaFree(ctx->d_db_stacked);
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
