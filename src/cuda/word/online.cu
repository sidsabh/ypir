/**
 * Word-Based SimplePIR Online Kernel
 *
 * Step 1: query × DB in Z_{2^64}, modswitch to Z_Q, CRT-pack.
 *   Tensor cores (SM≥80): 2 CUTLASS uint8 GEMMs (one per DB byte) + fused accumulate
 *   SIMT fallback:        1 CUTLASS uint16×uint64→uint64 GEMM
 *
 * Step 2: LWE packing + mod switch — identical to online_kernel_sp.cu
 *         Streamed: num_streams determined by available GPU memory,
 *         batches processed in chunks of num_streams.
 *
 * query: db_rows u64 (raw, NOT CRT-packed)
 * DB: db_cols × db_rows_padded u16 (column-major)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "common/ntt.cuh"
#include "inspiring/tc_packing.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_ASSERT(stmt) do { \
    cudaError_t err = (stmt);  \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        abort();               \
    }                          \
} while (0)

#define CUDA_ALLOC_AND_COPY(dst, src, size) do { \
    CUDA_ASSERT(cudaMalloc(&(dst), (size)));     \
    CUDA_ASSERT(cudaMemcpy((dst), (src), (size), cudaMemcpyHostToDevice)); \
} while (0)

// ---------- CUTLASS SIMT GEMM: uint16 × uint64 → uint64 ----------
using CutlassGemmWord = cutlass::gemm::device::Gemm<
    uint16_t,                               // ElementA (DB)
    cutlass::layout::RowMajor,
    uint64_t,                               // ElementB (query)
    cutlass::layout::ColumnMajor,
    uint64_t,                               // ElementC
    cutlass::layout::ColumnMajor,
    uint64_t,                               // Accumulator
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<uint64_t, 1, uint64_t, uint64_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2
>;

// ---------- CUTLASS Tensor Core GEMM: uint8 × uint8 → int32 ----------

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
__global__ void word_decompose_u16_stacked(
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

// Query u64 → 8 uint8 byte slices, packed contiguously for wide GEMM.
// Output layout: K × (8*N) column-major with stride K.
// Byte slice i occupies columns [i*N, (i+1)*N).
__global__ void word_decompose_u64_bytes_packed(
    uint8_t* __restrict__ out,  // K × (8*N) contiguous
    const uint64_t* __restrict__ data,
    size_t K, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * N) return;
    // data is K × N column-major: idx = k + n*K
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

// ---------- Fused accumulate + modswitch kernels ----------

// Accumulate stacked GEMM output (2M)×(8N) → M×N uint64, then modswitch.
// gemm_out is col-major (2M)×(8N), stride 2M.
// Rows [0,M) = db_b0 products, rows [M,2M) = db_b1 products.
// db_b0: 8 terms (shifts 0,8,...,56), db_b1: 7 terms (shifts 8,...,56).
__global__ void word_accumulate_stacked_and_modswitch(
    uint64_t* __restrict__ accum,          // M × N col-major (output: modswitched)
    const int32_t* __restrict__ gemm_out,  // (2M) × (8*N) col-major, stride 2M
    size_t M, size_t N,
    uint64_t q, uint64_t inv_n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;
    size_t stride = 2 * M;
    uint64_t acc = 0;
    #pragma unroll
    for (int q_b = 0; q_b < 8; q_b++)
        acc += (uint64_t)(uint32_t)gemm_out[m + (q_b*N+n)*stride] << (8*q_b);
    #pragma unroll
    for (int q_b = 0; q_b < 7; q_b++)
        acc += (uint64_t)(uint32_t)gemm_out[M + m + (q_b*N+n)*stride] << (8*(q_b+1));
    // Modswitch: round(acc * q / 2^64), then multiply by inv_n mod q
    __uint128_t prod = (__uint128_t)acc * q + (((__uint128_t)1) << 63);
    uint64_t val_q = (uint64_t)(prod >> 64);
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);
    accum[idx] = val_q;
}

// ---------- Modswitch ----------

// SIMT path: modswitch u64 to Z_Q (in-place), multiply by inv_N
__global__ void word_modswitch_crt_inplace(
    uint64_t* __restrict__ data,
    size_t count,
    uint64_t q, uint64_t mod0, uint64_t mod1,
    uint64_t inv_n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t val = data[idx];
    __uint128_t prod = (__uint128_t)val * q + (((__uint128_t)1) << 63);
    uint64_t val_q = (uint64_t)(prod >> 64);
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);
    data[idx] = val_q;
}

// ---------- Step 2: Pack LWEs and mod switch ----------
// Streamed variant: batch_idx is passed as a parameter (not blockIdx.y).
// Scratch is per-stream: scratch_idx = output_idx only.

__device__ __forceinline__
void word_write_arbitrary_bits(uint8_t* out, uint64_t val, size_t bit_offs, size_t num_bits) {
    for (size_t i = 0; i < num_bits; i++) {
        size_t abs_bit = bit_offs + i;
        size_t byte_idx = abs_bit >> 3;
        size_t bit_in_byte = abs_bit & 7;
        if ((val >> i) & 1ULL) {
            size_t word_byte = byte_idx & ~3ULL;
            uint32_t* wordp = (uint32_t*)(out + word_byte);
            uint32_t bitpos = (uint32_t)((byte_idx - word_byte) * 8 + bit_in_byte);
            atomicOr(wordp, 1u << bitpos);
        }
    }
}

__host__ __device__ __forceinline__
uint32_t word_ceil_log2_u64(uint64_t x) {
    if (x <= 1) return 0;
    uint64_t y = x - 1;
#if defined(__CUDA_ARCH__)
    return 64u - (uint32_t)__clzll(y);
#else
    return 64u - (uint32_t)__builtin_clzll(y);
#endif
}

__global__ void __launch_bounds__(1024) word_pack_lwes_and_mod_switch(
    uint8_t* d_response_out,           // points to this batch item's response start
    const uint64_t* d_intermediate,    // points to this batch item's intermediate start
    const uint64_t* d_y_constants,
    const uint64_t* d_precomp_res,
    const uint64_t* d_precomp_vals,
    const uint64_t* d_precomp_tables,
    const uint64_t* d_pub_params_row_1s, // points to this batch item's pub params
    uint64_t* d_scratch,               // per-stream scratch: num_outputs * scratch_per_output
    size_t num_outputs,
    size_t db_cols,
    size_t t_exp_left,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t scratch_size_per_output,
    size_t pub_params_size_per_batch,
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    size_t crt_count = params.crt_count;
    size_t convd_len = crt_count * poly_len;
    size_t ell = params.log2_poly_len;
    uint64_t modulus = params.modulus;

    // Cooperative NTT decomposition (used in post-FFT code)
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    // Per-warp decomposition (used in FFT loop)
    size_t warp_id = tid / 32;
    size_t lane_id = tid % 32;
    size_t num_warps = blockDim.x / 32;

    // Scratch layout: working_set | result_ntt | temp_raw | temp_ntt
    // (intermediate buffers removed — fused into registers + ct_odd reuse)
    uint64_t* my_scratch = d_scratch + output_idx * scratch_size_per_output;

    size_t working_set_size = (1 << (ell - 1));
    uint64_t* working_set = my_scratch;
    uint64_t* result_ntt  = working_set + working_set_size * poly_len;
    uint64_t* temp_raw    = result_ntt + 2 * convd_len;
    uint64_t* temp_ntt    = temp_raw + 2 * poly_len;

    // b_values from this batch item's intermediate
    const uint64_t* b_values = d_intermediate + output_idx * poly_len;

    size_t precomp_vals_per_output   = ((1 << ell) - 1) * t_exp_left * poly_len;
    size_t precomp_tables_per_output = ell * poly_len;
    const uint64_t* my_precomp_res    = d_precomp_res    + output_idx * 2 * convd_len;
    const uint64_t* my_precomp_vals   = d_precomp_vals   + output_idx * precomp_vals_per_output;
    const uint64_t* my_precomp_tables = d_precomp_tables + output_idx * precomp_tables_per_output;

    // pub_params already offset to this batch item
    const uint64_t* my_pub_params = d_pub_params_row_1s;

    // response already offset to this batch item + output
    uint8_t* my_response = d_response_out + output_idx * response_bytes_per_output;

    // Zero working set (cooperative, all threads)
    for (size_t i = tid; i < working_set_size * poly_len; i += blockDim.x)
        working_set[i] = 0;
    __syncthreads();

    // ── FFT-style packing (per-warp parallelism) ──
    // Each warp independently processes a subset of tree nodes.
    // Intermediate values (y_times, neg_y_times, w_ginv) stay in registers.
    // ct_odd is reused as ct_sum_1 storage for the automorphism permutation.
    // Only __syncthreads() between levels (not between nodes within a level).

    size_t idx_precomp_base = 0;

    for (size_t cur_ell = 1; cur_ell <= ell; cur_ell++) {
        size_t num_in  = 1 << (ell - cur_ell + 1);
        size_t num_out = num_in >> 1;
        if (num_in == poly_len) num_in = num_out;

        const uint64_t* y     = d_y_constants + (cur_ell - 1) * convd_len;
        const uint64_t* neg_y = d_y_constants + ell * convd_len + (cur_ell - 1) * convd_len;
        size_t pub_param_idx = ell - cur_ell;
        const uint64_t* w = my_pub_params + pub_param_idx * t_exp_left * poly_len;

        for (size_t node = warp_id; node < num_out; node += num_warps) {
            uint64_t* ct_even = working_set + node * poly_len;
            const uint64_t* ginv_ct = my_precomp_vals
                + (idx_precomp_base + node) * t_exp_left * poly_len;

            if (cur_ell > 1) {
                uint64_t* ct_odd = working_set + (num_out + node) * poly_len;

                // Phase 1: y_mult + y_add fused.
                // Write ct_sum_1 into ct_odd (which is consumed and no longer needed).
                for (size_t z = lane_id; z < poly_len; z += 32) {
                    uint64_t ct_val = ct_odd[z];
                    uint64_t ct_lo = ct_val & 0xFFFFFFFF;
                    uint64_t ct_hi = ct_val >> 32;

                    uint64_t y_lo     = y[z];
                    uint64_t y_hi     = y[poly_len + z];
                    uint64_t neg_y_lo = neg_y[z];
                    uint64_t neg_y_hi = neg_y[poly_len + z];

                    uint64_t prod_lo = barrett_raw_u64(ct_lo * y_lo,     params.barrett_cr[0], params.moduli[0]);
                    uint64_t prod_hi = barrett_raw_u64(ct_hi * y_hi,     params.barrett_cr[1], params.moduli[1]);
                    uint64_t y_times = (prod_lo & 0xFFFFFFFF) | ((prod_hi & 0xFFFFFFFF) << 32);

                    prod_lo = barrett_raw_u64(ct_lo * neg_y_lo, params.barrett_cr[0], params.moduli[0]);
                    prod_hi = barrett_raw_u64(ct_hi * neg_y_hi, params.barrett_cr[1], params.moduli[1]);
                    uint64_t neg_y_times = (prod_lo & 0xFFFFFFFF) | ((prod_hi & 0xFFFFFFFF) << 32);

                    uint64_t orig = ct_even[z];
                    ct_even[z] = orig + y_times;
                    ct_odd[z]  = orig + neg_y_times;  // ct_sum_1 stored in ct_odd
                }
                __syncwarp();

                // Phase 2: w_ginv + automorph + reduce fused.
                const uint64_t* table = my_precomp_tables + (ell - cur_ell) * poly_len;
                for (size_t z = lane_id; z < poly_len; z += 32) {
                    // w × ginv inner product
                    uint64_t sum_lo = 0, sum_hi = 0;
                    for (size_t k = 0; k < t_exp_left; k++) {
                        uint64_t w_val = w[k * poly_len + z];
                        uint64_t g_val = ginv_ct[k * poly_len + z];
                        sum_lo += (w_val & 0xFFFFFFFF) * (g_val & 0xFFFFFFFF);
                        sum_hi += (w_val >> 32)         * (g_val >> 32);
                    }
                    uint64_t res_lo = barrett_raw_u64(sum_lo, params.barrett_cr[0], params.moduli[0]);
                    uint64_t res_hi = barrett_raw_u64(sum_hi, params.barrett_cr[1], params.moduli[1]);
                    uint64_t w_ginv_val = (res_lo & 0xFFFFFFFF) | ((res_hi & 0xFFFFFFFF) << 32);

                    // Automorphism: permuted read from ct_odd (= ct_sum_1)
                    size_t src_idx = (size_t)table[z];
                    uint64_t automorph_val = ct_odd[src_idx];

                    // Accumulate + Barrett reduce
                    uint64_t val = ct_even[z] + automorph_val + w_ginv_val;
                    uint64_t lo = barrett_raw_u64(val & 0xFFFFFFFF, params.barrett_cr[0], params.moduli[0]);
                    uint64_t hi = barrett_raw_u64(val >> 32,        params.barrett_cr[1], params.moduli[1]);
                    ct_even[z] = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32);
                }

            } else {
                // cur_ell == 1: only w_ginv + reduce (no y-multiply, no automorphism)
                for (size_t z = lane_id; z < poly_len; z += 32) {
                    uint64_t sum_lo = 0, sum_hi = 0;
                    for (size_t k = 0; k < t_exp_left; k++) {
                        uint64_t w_val = w[k * poly_len + z];
                        uint64_t g_val = ginv_ct[k * poly_len + z];
                        sum_lo += (w_val & 0xFFFFFFFF) * (g_val & 0xFFFFFFFF);
                        sum_hi += (w_val >> 32)         * (g_val >> 32);
                    }
                    uint64_t res_lo = barrett_raw_u64(sum_lo, params.barrett_cr[0], params.moduli[0]);
                    uint64_t res_hi = barrett_raw_u64(sum_hi, params.barrett_cr[1], params.moduli[1]);
                    uint64_t w_ginv_val = (res_lo & 0xFFFFFFFF) | ((res_hi & 0xFFFFFFFF) << 32);

                    uint64_t val = ct_even[z] + w_ginv_val;
                    uint64_t lo = barrett_raw_u64(val & 0xFFFFFFFF, params.barrett_cr[0], params.moduli[0]);
                    uint64_t hi = barrett_raw_u64(val >> 32,        params.barrett_cr[1], params.moduli[1]);
                    ct_even[z] = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32);
                }
            }
        }

        idx_precomp_base += num_out;
        __syncthreads();  // barrier between tree levels
    }

    // Final reduction on working_set[0] (cooperative, all threads)
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = working_set[z];
        uint64_t lo = barrett_raw_u64(val & 0xFFFFFFFF, params.barrett_cr[0], params.moduli[0]);
        uint64_t hi = barrett_raw_u64(val >> 32,        params.barrett_cr[1], params.moduli[1]);
        working_set[z] = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32);
    }
    __syncthreads();

    // Build result
    for (size_t z = tid; z < convd_len; z += blockDim.x)
        result_ntt[z] = my_precomp_res[z];
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = working_set[z];
        result_ntt[convd_len + z]            = val & 0xFFFFFFFF;
        result_ntt[convd_len + poly_len + z] = val >> 32;
    }
    __syncthreads();

    // Add b_values in raw domain
    for (size_t z = tid; z < convd_len; z += blockDim.x) {
        uint32_t crt     = z / poly_len;
        uint64_t mod     = params.moduli[crt];
        uint64_t bcr     = params.barrett_cr[crt];
        temp_ntt[z] = barrett_raw_u64(result_ntt[convd_len + z], bcr, mod);
    }
    __syncthreads();

    if (my_crt < crt_count)
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x     = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t composed = crt_compose_2(x, y_val, &params);

        __uint128_t prod = (__uint128_t)b_values[z] * poly_len;
        uint64_t b_scaled = (uint64_t)(prod % modulus);
        composed = (composed + b_scaled);
        composed -= (modulus * (composed > modulus));

        temp_raw[poly_len + z] = composed;
    }
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        temp_ntt[z]            = barrett_raw_u64(val, params.barrett_cr[0], params.moduli[0]);
        temp_ntt[poly_len + z] = barrett_raw_u64(val, params.barrett_cr[1], params.moduli[1]);
    }
    __syncthreads();

    if (my_crt < crt_count)
        ntt_forward_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    for (size_t z = tid; z < convd_len; z += blockDim.x)
        result_ntt[convd_len + z] = temp_ntt[z];
    __syncthreads();

    // Inverse NTT to raw for mod switch
    for (size_t row = 0; row < 2; row++) {
        for (size_t z = tid; z < convd_len; z += blockDim.x)
            temp_ntt[z] = result_ntt[row * convd_len + z];
        __syncthreads();

        if (my_crt < crt_count)
            ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
        __syncthreads();

        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            uint64_t x     = temp_ntt[z];
            uint64_t y_val = temp_ntt[poly_len + z];
            temp_raw[row * poly_len + z] = crt_compose_2(x, y_val, &params);
        }
        __syncthreads();
    }

    // Mod switch and bit-pack
    size_t q_1_bits = word_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = word_ceil_log2_u64(rlwe_q_prime_1);
    size_t total_sz_bytes = response_bytes_per_output;

    for (size_t i = tid; i < total_sz_bytes; i += blockDim.x)
        my_response[i] = 0;
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, z * q_1_bits, q_1_bits);
    }
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, poly_len * q_1_bits + z * q_2_bits, q_2_bits);
    }
}

// ---------- InspiRING bold_t transpose kernel ----------
// Transposes bold_t from [num_outputs, D, poly_len] to [num_outputs, poly_len, D]
// so that consecutive d-values are stride-1 for the inner product kernel.
__global__ void inspir_transpose_bold_t(
    uint64_t* __restrict__ dst,        // [num_outputs * poly_len * D]
    const uint64_t* __restrict__ src,  // [num_outputs * D * poly_len]
    size_t num_outputs,
    size_t D,
    size_t poly_len)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_outputs * D * poly_len;
    if (idx >= total) return;

    size_t o  = idx / (D * poly_len);
    size_t rem = idx % (D * poly_len);
    size_t d  = rem / poly_len;
    size_t z  = rem % poly_len;

    // src[o][d][z] → dst[o][z][d]
    dst[o * poly_len * D + z * D + d] = src[idx];
}

// ---------- InspiRING FUSED expand + inner product kernel ----------
// Eliminates the separate expand phase entirely. Instead of:
//   expand: write y_all/y_bar_all to global (~96 MB/client)
//   IP: read y_all/y_bar_all from global (~96 MB/client)
// This kernel does on-the-fly permutation gather from y_body (48 KB/client, L2-cached).
// Saves ~192 MB/client of HBM traffic = ~6 GB for 32 clients.
// Also eliminates ~6.8 GB of GPU memory for y_all/y_bar_all buffers.
//
// Block layout: dim3(32, batch_per_block)
//   threadIdx.x → z offset within tile (warp-aligned for coalescing)
//   threadIdx.y → client index within block
//
// Shared memory: bold_t tile (2 × TILE_D × 32 × 8 bytes)
// bold_t loaded cooperatively by all threads, shared across all clients.
// Table lookups cached in L1/L2 (same z across clients).
// y_body gathered via permutation (48 KB/client, fits in L2).

#define FIP_TILE_D 32  // d-values per shared memory tile (16KB smem → 2 blocks/SM)
#define FIP_TILE_Z 32  // z-values per block (= warp size)

__global__ void __launch_bounds__(256)
inspir_fused_expand_ip(
    uint64_t* __restrict__ d_scratch_base,       // contiguous scratch [batch, num_outputs, scratch_per_output]
    const uint64_t* __restrict__ d_bold_t,       // [num_outputs, D, poly_len] — SHARED across all clients
    const uint64_t* __restrict__ d_bold_t_bar,   // same
    const uint64_t* __restrict__ d_bold_t_hat,   // [num_outputs * t_exp_left * poly_len]
    const uint64_t* __restrict__ d_y_body_base,  // [batch, t_exp_left * poly_len] — tiny, L2-cached
    const uint64_t* __restrict__ d_z_body_base,  // [batch, t_exp_left * poly_len]
    const uint32_t* __restrict__ d_tables,       // [num_tables * poly_len] permutation tables
    const uint32_t* __restrict__ d_gen_pows,     // [num_rotations] generator powers
    size_t actual_batch_size,
    size_t num_outputs,
    size_t D,                                     // num_iter * t_exp_left
    size_t poly_len,
    size_t t_exp_left,
    size_t inspir_scratch_per_output,
    size_t y_body_stride,                         // t_exp_left * poly_len (elements per client)
    size_t z_body_stride,                         // same
    size_t scratch_stride,                        // num_outputs * inspir_scratch_per_output
    NTTParams params)
{
    size_t z = blockIdx.x * FIP_TILE_Z + threadIdx.x;
    if (z >= poly_len) return;

    size_t client = blockIdx.y * blockDim.y + threadIdx.y;
    if (client >= actual_batch_size) return;

    size_t o = blockIdx.z;  // output index — parallelized across grid

    // Barrett constants
    uint64_t bcr0 = params.barrett_cr[0];
    uint64_t bcr1 = params.barrett_cr[1];
    uint64_t mod0 = params.moduli[0];
    uint64_t mod1 = params.moduli[1];

    // Per-client data pointers
    const uint64_t* my_y_body = d_y_body_base + client * y_body_stride;
    const uint64_t* my_z_body = d_z_body_base + client * z_body_stride;
    uint64_t* my_scratch      = d_scratch_base + client * scratch_stride;

    // Shared memory: bold_t tile (shared by ALL clients in this block)
    __shared__ uint64_t s_bt[FIP_TILE_D][FIP_TILE_Z];
    __shared__ uint64_t s_btbar[FIP_TILE_D][FIP_TILE_Z];

    uint64_t acc_lo = 0, acc_hi = 0;

    for (size_t d_start = 0; d_start < D; d_start += FIP_TILE_D) {
        size_t tile_len = FIP_TILE_D;
        if (d_start + tile_len > D) tile_len = D - d_start;

        // Cooperative load of bold_t into shared memory.
        // All threads participate for balanced loading.
        {
            size_t tid = threadIdx.y * FIP_TILE_Z + threadIdx.x;
            size_t block_threads = blockDim.y * FIP_TILE_Z;
            for (size_t idx = tid; idx < tile_len * FIP_TILE_Z; idx += block_threads) {
                size_t dd = idx / FIP_TILE_Z;
                size_t zz = idx % FIP_TILE_Z;
                size_t gz = blockIdx.x * FIP_TILE_Z + zz;
                if (gz < poly_len) {
                    s_bt[dd][zz]    = __ldg(&d_bold_t[o * D * poly_len + (d_start + dd) * poly_len + gz]);
                    s_btbar[dd][zz] = __ldg(&d_bold_t_bar[o * D * poly_len + (d_start + dd) * poly_len + gz]);
                }
            }
        }
        __syncthreads();

        // Each client accumulates against shared bold_t.
        // On-the-fly expand: gather y_body via permutation tables.
        // Barrett at tile boundary only (FIP_TILE_D=32, adds 64 products ≤ addition_capacity).
        {
            size_t prev_rot = (size_t)-1;
            uint32_t perm_idx1 = 0, perm_idx2 = 0;
            for (size_t dd = 0; dd < tile_len; dd++) {
                size_t d_idx = d_start + dd;
                size_t rot = d_idx / t_exp_left;
                size_t k   = d_idx - rot * t_exp_left;

                if (rot != prev_rot) {
                    uint32_t gen = __ldg(&d_gen_pows[rot]);
                    uint32_t tidx1 = (gen - 1) / 2;
                    uint32_t tidx2 = (2 * (uint32_t)poly_len - gen - 1) / 2;
                    perm_idx1 = __ldg(&d_tables[(size_t)tidx1 * poly_len + z]);
                    perm_idx2 = __ldg(&d_tables[(size_t)tidx2 * poly_len + z]);
                    prev_rot = rot;
                }

                uint64_t y_val  = __ldg(&my_y_body[k * poly_len + perm_idx1]);
                uint64_t yb_val = __ldg(&my_y_body[k * poly_len + perm_idx2]);

                uint64_t t_val  = s_bt[dd][threadIdx.x];
                uint64_t tb_val = s_btbar[dd][threadIdx.x];

                acc_lo += (uint64_t)(uint32_t)y_val  * (uint32_t)t_val
                        + (uint64_t)(uint32_t)yb_val * (uint32_t)tb_val;
                acc_hi += (uint64_t)(uint32_t)(y_val >> 32)  * (uint32_t)(t_val >> 32)
                        + (uint64_t)(uint32_t)(yb_val >> 32) * (uint32_t)(tb_val >> 32);
            }
        }
        // Barrett reduce once per tile (branch-free inner loop)
        acc_lo = barrett_raw_u64(acc_lo, bcr0, mod0);
        acc_hi = barrett_raw_u64(acc_hi, bcr1, mod1);
        __syncthreads();
    }

    // Add z_body × bold_t_hat (t_exp_left terms)
    uint64_t fin_lo = 0, fin_hi = 0;
    for (size_t k = 0; k < t_exp_left; k++) {
        uint64_t z_val  = __ldg(&my_z_body[k * poly_len + z]);
        uint64_t th_val = __ldg(&d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z]);
        fin_lo += (z_val & 0xFFFFFFFF) * (th_val & 0xFFFFFFFF);
        fin_hi += (z_val >> 32)         * (th_val >> 32);
    }
    fin_lo = barrett_raw_u64(fin_lo, bcr0, mod0);
    fin_hi = barrett_raw_u64(fin_hi, bcr1, mod1);

    acc_lo += fin_lo;
    if (acc_lo >= mod0) acc_lo -= mod0;
    acc_hi += fin_hi;
    if (acc_hi >= mod1) acc_hi -= mod1;

    // Write CRT halves to this client's scratch
    uint64_t* temp_ntt = my_scratch + o * inspir_scratch_per_output;
    temp_ntt[z]            = acc_lo;
    temp_ntt[poly_len + z] = acc_hi;
}

// ---------- InspiRING Multi-Output Fused kernel ----------
// Each thread processes ALL outputs, reading y_body ONCE and reusing across outputs.
// This eliminates num_outputs× redundant y_body scattered reads and table lookups.
// bold_t is read from global memory via __ldg (coalesced per output).
//
// Block: dim3(32, batch_per_block) — 32 z-values × N clients
// Grid: dim3(ceil(poly_len/32), ceil(batch/bpb)) — NO output dimension
// No shared memory needed (bold_t coalesced from global, y_body from L2).
// Templated on NUM_OUTPUTS so compiler can keep accumulators in registers.

template <int NUM_OUTPUTS>
__global__ void __launch_bounds__(256)
inspir_fused_multi_output(
    uint64_t* __restrict__ d_scratch_base,
    const uint64_t* __restrict__ d_bold_t,
    const uint64_t* __restrict__ d_bold_t_bar,
    const uint64_t* __restrict__ d_bold_t_hat,
    const uint64_t* __restrict__ d_y_body_base,
    const uint64_t* __restrict__ d_z_body_base,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    size_t batch_size,
    size_t D,
    size_t poly_len,
    size_t t_exp_left,
    size_t inspir_scratch_per_output,
    size_t y_body_stride,
    size_t z_body_stride,
    size_t scratch_stride,
    NTTParams params)
{
    size_t z = blockIdx.x * 32 + threadIdx.x;
    if (z >= poly_len) return;
    size_t client = blockIdx.y * blockDim.y + threadIdx.y;
    if (client >= batch_size) return;

    uint64_t bcr0 = params.barrett_cr[0];
    uint64_t bcr1 = params.barrett_cr[1];
    uint64_t mod0 = params.moduli[0];
    uint64_t mod1 = params.moduli[1];

    const uint64_t* my_y_body = d_y_body_base + client * y_body_stride;
    const uint64_t* my_z_body = d_z_body_base + client * z_body_stride;
    uint64_t* my_scratch      = d_scratch_base + client * scratch_stride;

    // Compile-time sized accumulators → compiler keeps in registers
    uint64_t acc_lo[NUM_OUTPUTS];
    uint64_t acc_hi[NUM_OUTPUTS];
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        acc_lo[o] = 0;
        acc_hi[o] = 0;
    }

    size_t prev_rot = (size_t)-1;
    uint32_t perm_idx1 = 0, perm_idx2 = 0;
    size_t num_added = 0;

    for (size_t d_idx = 0; d_idx < D; d_idx++) {
        size_t rot = d_idx / t_exp_left;
        size_t k   = d_idx - rot * t_exp_left;

        if (rot != prev_rot) {
            uint32_t gen = __ldg(&d_gen_pows[rot]);
            uint32_t tidx1 = (gen - 1) / 2;
            uint32_t tidx2 = (2 * (uint32_t)poly_len - gen - 1) / 2;
            perm_idx1 = __ldg(&d_tables[(size_t)tidx1 * poly_len + z]);
            perm_idx2 = __ldg(&d_tables[(size_t)tidx2 * poly_len + z]);
            prev_rot = rot;
        }

        // Read y_body ONCE — reuse across ALL outputs
        uint64_t y_val  = __ldg(&my_y_body[k * poly_len + perm_idx1]);
        uint64_t yb_val = __ldg(&my_y_body[k * poly_len + perm_idx2]);
        uint32_t y_lo  = (uint32_t)y_val;
        uint32_t y_hi  = (uint32_t)(y_val >> 32);
        uint32_t yb_lo = (uint32_t)yb_val;
        uint32_t yb_hi = (uint32_t)(yb_val >> 32);

        // Accumulate against ALL outputs (bold_t coalesced from global)
        #pragma unroll
        for (int o = 0; o < NUM_OUTPUTS; o++) {
            size_t bt_idx = o * D * poly_len + d_idx * poly_len + z;
            uint64_t t_val  = __ldg(&d_bold_t[bt_idx]);
            uint64_t tb_val = __ldg(&d_bold_t_bar[bt_idx]);

            acc_lo[o] += (uint64_t)y_lo  * (uint32_t)t_val
                       + (uint64_t)yb_lo * (uint32_t)tb_val;
            acc_hi[o] += (uint64_t)y_hi  * (uint32_t)(t_val >> 32)
                       + (uint64_t)yb_hi * (uint32_t)(tb_val >> 32);
        }

        num_added += 2;
        if (num_added >= 64) {
            #pragma unroll
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                acc_lo[o] = barrett_raw_u64(acc_lo[o], bcr0, mod0);
                acc_hi[o] = barrett_raw_u64(acc_hi[o], bcr1, mod1);
            }
            num_added = 0;
        }
    }

    // Final Barrett
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        acc_lo[o] = barrett_raw_u64(acc_lo[o], bcr0, mod0);
        acc_hi[o] = barrett_raw_u64(acc_hi[o], bcr1, mod1);
    }

    // Add z_body × bold_t_hat
    #pragma unroll
    for (int o = 0; o < NUM_OUTPUTS; o++) {
        uint64_t fin_lo = 0, fin_hi = 0;
        for (size_t k = 0; k < t_exp_left; k++) {
            uint64_t z_val  = __ldg(&my_z_body[k * poly_len + z]);
            uint64_t th_val = __ldg(&d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z]);
            fin_lo += (z_val & 0xFFFFFFFF) * (th_val & 0xFFFFFFFF);
            fin_hi += (z_val >> 32)         * (th_val >> 32);
        }
        fin_lo = barrett_raw_u64(fin_lo, bcr0, mod0);
        fin_hi = barrett_raw_u64(fin_hi, bcr1, mod1);

        acc_lo[o] += fin_lo;
        if (acc_lo[o] >= mod0) acc_lo[o] -= mod0;
        acc_hi[o] += fin_hi;
        if (acc_hi[o] >= mod1) acc_hi[o] -= mod1;

        uint64_t* temp_ntt = my_scratch + o * inspir_scratch_per_output;
        temp_ntt[z]            = acc_lo[o];
        temp_ntt[poly_len + z] = acc_hi[o];
    }
}

// Dispatch helper for multi-output kernel.
// Queries actual register usage via cudaFuncGetAttributes to compute safe batch_per_block.
static void launch_multi_output_kernel(
    size_t num_outputs,
    size_t chunk_size,
    size_t poly_len,
    cudaStream_t stream,
    uint64_t* scratch, const uint64_t* bold_t, const uint64_t* bold_t_bar,
    const uint64_t* bold_t_hat, const uint64_t* y_body, const uint64_t* z_body,
    const uint32_t* tables, const uint32_t* gen_pows,
    size_t batch_size, size_t D, size_t t_exp_left,
    size_t inspir_spo, size_t ybs, size_t zbs, size_t ss, NTTParams params)
{
    size_t batch_per_block;
    if      (num_outputs <= 4)  batch_per_block = 8;
    else if (num_outputs <= 8)  batch_per_block = 4;
    else if (num_outputs <= 16) batch_per_block = 2;
    else                        batch_per_block = 1;
    if (batch_per_block > chunk_size) batch_per_block = chunk_size;

    dim3 block(32, (unsigned)batch_per_block);
    dim3 grid((poly_len + 31) / 32,
              (chunk_size + batch_per_block - 1) / batch_per_block);

    #define LAUNCH_MO(N) \
        inspir_fused_multi_output<N><<<grid, block, 0, stream>>>( \
            scratch, bold_t, bold_t_bar, bold_t_hat, y_body, z_body, \
            tables, gen_pows, batch_size, D, poly_len, t_exp_left, \
            inspir_spo, ybs, zbs, ss, params)

    switch (num_outputs) {
        case 2:  LAUNCH_MO(2);  break;
        case 3:  LAUNCH_MO(3);  break;
        case 4:  LAUNCH_MO(4);  break;
        case 6:  LAUNCH_MO(6);  break;
        case 8:  LAUNCH_MO(8);  break;
        case 12: LAUNCH_MO(12); break;
        case 16: LAUNCH_MO(16); break;
        case 18: LAUNCH_MO(18); break;
        case 24: LAUNCH_MO(24); break;
        case 32: LAUNCH_MO(32); break;
        default:
            fprintf(stderr, "ERROR: unsupported num_outputs=%zu for multi-output kernel\n", num_outputs);
            abort();
    }
    #undef LAUNCH_MO
    CUDA_ASSERT(cudaGetLastError());
}


// ---------- InspiRING post-process kernel ----------
// INTT + CRT compose + add b_values + modswitch + bitpack.
// Reads temp_ntt from scratch (written by inspir_inner_product_tiled).
// Requires b_values from Step1 GEMM (d_intermediate).
//
// Grid:  num_outputs blocks
// Block: 1024 threads (needed for cooperative INTT)

__global__ void __launch_bounds__(1024)
inspir_post_process(
    uint8_t* d_response_out,
    const uint64_t* d_intermediate,     // b_values from Step1
    const uint64_t* d_a_hat,            // per-output, raw domain
    uint64_t* d_scratch,                // per-stream scratch (temp_ntt written by inner product)
    size_t num_outputs,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t inspir_scratch_per_output,
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    uint64_t modulus = params.modulus;

    size_t crt_count = params.crt_count;
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    uint64_t* my_scratch = d_scratch + output_idx * inspir_scratch_per_output;
    uint64_t* temp_ntt = my_scratch;
    uint64_t* temp_raw = temp_ntt + 2 * poly_len;

    const uint64_t* my_a_hat  = d_a_hat + output_idx * poly_len;
    const uint64_t* b_values  = d_intermediate + output_idx * poly_len;
    uint8_t* my_response = d_response_out + output_idx * response_bytes_per_output;

    // INTT both CRT components
    if (my_crt < crt_count)
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    // CRT compose + add b_values → raw domain
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x     = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t sum_raw = crt_compose_2(x, y_val, &params);

        uint64_t final_b = sum_raw + b_values[z];
        final_b -= modulus * (final_b >= modulus);

        temp_raw[poly_len + z] = final_b;
        temp_raw[z]            = my_a_hat[z];
    }
    __syncthreads();

    // Modswitch and bit-pack
    size_t q_1_bits = word_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = word_ceil_log2_u64(rlwe_q_prime_1);

    for (size_t i = tid; i < response_bytes_per_output; i += blockDim.x)
        my_response[i] = 0;
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, z * q_1_bits, q_1_bits);
    }
    __syncthreads();

    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, poly_len * q_1_bits + z * q_2_bits, q_2_bits);
    }
}

// ---------- InspiRING packing kernel (OLD — kept for reference) ----------
// Implements full_packing_with_preprocessing_online (with_rotations):
//   sum = Σ_{i=0..num_iter-1} (y_all[i] · bold_t[i] + y_bar_all[i] · bold_t_bar[i])
//       + z_body · bold_t_hat
//   final_b = INTT(CRT_compose(sum)) + b_values   (no N multiplication)
//   output  = modswitch([a_hat; final_b])

__global__ void __launch_bounds__(1024) word_pack_lwes_inspir_and_mod_switch(
    uint8_t* d_response_out,
    const uint64_t* d_intermediate,     // b_values (Z_Q, one per coefficient per output)
    const uint64_t* d_bold_t,           // per-output precomp, dense condensed
    const uint64_t* d_bold_t_bar,
    const uint64_t* d_bold_t_hat,
    const uint64_t* d_a_hat,            // per-output, raw domain (Z_Q)
    const uint64_t* d_y_all,            // per-client, dense condensed
    const uint64_t* d_y_bar_all,
    const uint64_t* d_z_body,           // per-client
    uint64_t* d_scratch,                // per-stream scratch
    size_t num_iter,                    // poly_len/2 - 1
    size_t t_exp_left,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t inspir_scratch_per_output,
    NTTParams params)
{
    size_t output_idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    uint64_t modulus = params.modulus;

    // Cooperative NTT decomposition
    size_t crt_count = params.crt_count;
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    // Scratch: temp_ntt (2 * poly_len) + temp_raw (2 * poly_len)
    uint64_t* my_scratch = d_scratch + output_idx * inspir_scratch_per_output;
    uint64_t* temp_ntt = my_scratch;
    uint64_t* temp_raw = temp_ntt + 2 * poly_len;

    // Per-output precomp pointers (dense condensed, stride = poly_len)
    size_t bold_t_per_output = num_iter * t_exp_left * poly_len;
    const uint64_t* my_bold_t     = d_bold_t     + output_idx * bold_t_per_output;
    const uint64_t* my_bold_t_bar = d_bold_t_bar + output_idx * bold_t_per_output;
    const uint64_t* my_bold_t_hat = d_bold_t_hat + output_idx * t_exp_left * poly_len;
    const uint64_t* my_a_hat      = d_a_hat      + output_idx * poly_len;

    // b_values from GEMM intermediate (Z_Q values, inv_N=1)
    const uint64_t* b_values = d_intermediate + output_idx * poly_len;

    uint8_t* my_response = d_response_out + output_idx * response_bytes_per_output;

    // Load Barrett constants (will be cached in L1)
    uint64_t bcr0 = params.barrett_cr[0];
    uint64_t bcr1 = params.barrett_cr[1];
    uint64_t mod0 = params.moduli[0];
    uint64_t mod1 = params.moduli[1];

    // Compute addition capacity (match Rust: 1 << (64 - 2*q2_bits - 1))
    size_t q2_bits_0 = word_ceil_log2_u64(mod0);
    size_t q2_bits_1 = word_ceil_log2_u64(mod1);
    size_t max_q2_bits = q2_bits_0 > q2_bits_1 ? q2_bits_0 : q2_bits_1;
    size_t addition_capacity = (size_t)1 << (64 - 2 * max_q2_bits - 1);

    // ── Main loop: inner products in condensed NTT domain ──
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t acc_lo = 0, acc_hi = 0;
        size_t num_added = 0;

        for (size_t i = 0; i < num_iter; i++) {
            size_t base = i * t_exp_left * poly_len;

            // y_all × bold_t inner product (t_exp_left terms)
            for (size_t k = 0; k < t_exp_left; k++) {
                uint64_t y_val = d_y_all[base + k * poly_len + z];
                uint64_t t_val = my_bold_t[base + k * poly_len + z];
                acc_lo += (y_val & 0xFFFFFFFF) * (t_val & 0xFFFFFFFF);
                acc_hi += (y_val >> 32)         * (t_val >> 32);
            }
            num_added += t_exp_left;

            // y_bar_all × bold_t_bar inner product (t_exp_left terms)
            for (size_t k = 0; k < t_exp_left; k++) {
                uint64_t yb_val = d_y_bar_all[base + k * poly_len + z];
                uint64_t tb_val = my_bold_t_bar[base + k * poly_len + z];
                acc_lo += (yb_val & 0xFFFFFFFF) * (tb_val & 0xFFFFFFFF);
                acc_hi += (yb_val >> 32)         * (tb_val >> 32);
            }
            num_added += t_exp_left;

            // Periodic Barrett reduction (match Rust reduction schedule)
            if (num_added >= addition_capacity || i == num_iter - 1) {
                acc_lo = barrett_raw_u64(acc_lo, bcr0, mod0);
                acc_hi = barrett_raw_u64(acc_hi, bcr1, mod1);
                num_added = 0;
            }
        }

        // Final term: z_body × bold_t_hat (t_exp_left terms)
        {
            uint64_t fin_lo = 0, fin_hi = 0;
            for (size_t k = 0; k < t_exp_left; k++) {
                uint64_t z_val  = d_z_body[k * poly_len + z];
                uint64_t th_val = my_bold_t_hat[k * poly_len + z];
                fin_lo += (z_val & 0xFFFFFFFF) * (th_val & 0xFFFFFFFF);
                fin_hi += (z_val >> 32)         * (th_val >> 32);
            }
            fin_lo = barrett_raw_u64(fin_lo, bcr0, mod0);
            fin_hi = barrett_raw_u64(fin_hi, bcr1, mod1);

            // Add final term with Barrett (matches fast_add_into with reduce)
            acc_lo = acc_lo + fin_lo;
            if (acc_lo >= mod0) acc_lo -= mod0;
            acc_hi = acc_hi + fin_hi;
            if (acc_hi >= mod1) acc_hi -= mod1;
        }

        // Store CRT components for INTT
        temp_ntt[z]            = acc_lo;
        temp_ntt[poly_len + z] = acc_hi;
    }
    __syncthreads();

    // ── INTT both CRT components ──
    if (my_crt < crt_count)
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    __syncthreads();

    // ── CRT compose + add b_values → raw domain ──
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x     = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t sum_raw = crt_compose_2(x, y_val, &params);

        // final_b = sum_raw + b_values[z] (mod Q, no N multiplication)
        uint64_t final_b = sum_raw + b_values[z];
        final_b -= modulus * (final_b >= modulus);

        temp_raw[poly_len + z] = final_b;   // body (row 1)
        temp_raw[z]            = my_a_hat[z]; // mask (row 0, a_hat already raw)
    }
    __syncthreads();

    // ── Modswitch and bit-pack (same format as CDKS) ──
    size_t q_1_bits = word_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = word_ceil_log2_u64(rlwe_q_prime_1);

    // Zero response bytes
    for (size_t i = tid; i < response_bytes_per_output; i += blockDim.x)
        my_response[i] = 0;
    __syncthreads();

    // Row 0 (mask/a_hat): rescale Q → q_prime_2
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, z * q_1_bits, q_1_bits);
    }
    __syncthreads();

    // Row 1 (body/final_b): rescale Q → q_prime_1
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        word_write_arbitrary_bits(my_response, val_rescaled, poly_len * q_1_bits + z * q_2_bits, q_2_bits);
    }
}

// ---------- Context ----------

struct WordOnlineContext {
    bool has_tensor_cores;
    bool is_sm80;

    // Tensor core path (CUTLASS uint8, 1 stacked GEMM)
    uint8_t* d_db_stacked;             // DB decomposed+stacked bytes, (2M)×K_padded
    uint8_t* d_query_bytes_packed;     // K × (8*max_N) contiguous, per-batch
    int32_t* d_gemm_out;              // (2M) × (8*max_N) int32

    // SIMT path
    uint16_t* d_db_u16;            // DB as-is, persistent
    uint64_t* d_query_buf;         // Upload buffer for queries
    uint64_t* d_result_u64;        // GEMM output

    // Common
    uint64_t* d_intermediate;      // CRT-packed result, batch × db_cols

    // Packing data (shared, read-only)
    uint64_t* d_y_constants;
    uint64_t* d_precomp_res;
    uint64_t* d_precomp_vals;
    uint64_t* d_precomp_tables;

    // Per-stream scratch (only num_streams allocated, not max_batch_size)
    size_t num_streams;
    cudaStream_t* streams;
    uint64_t** d_scratch_batch;    // [num_streams], each = num_outputs * scratch_per_output u64s

    // Pre-allocated batch buffers (full max_batch_size)
    uint64_t* d_pub_params_all;
    uint8_t*  d_all_responses;

    NTTParams ntt_params;

    size_t db_rows;
    size_t db_rows_padded;
    size_t db_cols;
    size_t num_rlwe_outputs;
    size_t t_exp_left;
    size_t max_batch_size;

    size_t scratch_per_output;
    size_t response_bytes_per_output;
    size_t pub_params_size_per_batch;

    uint64_t modulus;    // Q = moduli[0] * moduli[1]
    uint64_t mod0;
    uint64_t mod1;
    uint64_t inv_n;      // N^{-1} mod Q, compensates CDKS N-multiplication
    uint64_t rlwe_q_prime_1;
    uint64_t rlwe_q_prime_2;

    // InspiRING packing data (per-output precomp, uploaded once)
    bool is_inspir;
    uint64_t* d_bold_t_condensed;      // num_outputs * bold_t_per_output
    uint64_t* d_bold_t_bar_condensed;  // same
    uint64_t* d_bold_t_hat_condensed;  // num_outputs * t_exp_left * poly_len
    uint64_t* d_a_hat;                 // num_outputs * poly_len (raw domain)
    size_t bold_t_per_output;          // num_iter * t_exp_left * poly_len
    size_t num_iter;                   // poly_len/2 - 1
    size_t inspir_scratch_per_output;  // 4 * poly_len (temp_ntt + temp_raw)

    // Per-stream InspiRING client data buffers (contiguous allocations)
    uint64_t* d_z_body_streams;        // num_streams * z_body_per_client
    uint64_t* d_scratch_contiguous;    // num_streams * scratch_bytes (contiguous)
    size_t z_body_per_client;          // t_exp_left * poly_len
    size_t scratch_bytes_per_stream;   // num_rlwe_outputs * inspir_scratch_per_output * 8

    // Expand data (uploaded once during init, used by expand_rotations_double kernel)
    uint32_t* d_tables;                // num_tables * poly_len
    uint32_t* d_gen_pows;              // num_rotations
    size_t num_tables;
    size_t num_rotations;              // poly_len/2 - 1
    uint64_t* d_y_body_streams;        // num_streams * z_body_per_client (expand input)

    // Alternate packing path (tc_packing)
    void* tc_packing_ctx;              // TcPackingContext*, nullptr = use SIMT path
};

// ---------- Expand rotations kernel (fused) ----------
// Generates y_all and y_bar_all from y_body via permutation tables.
// Fuses EXPAND_ROTS rotations per block. y_body[k] is loaded into shared memory
// once per k-iteration and reused across all rotations — eliminates redundant global reads.
#define EXPAND_ROTS 8  // rotations per block

__global__ void expand_rotations_double(
    uint64_t* __restrict__ d_y_all,         // [num_rotations * t_exp_left * poly_len]
    uint64_t* __restrict__ d_y_bar_all,     // same
    const uint64_t* __restrict__ d_y_body,  // [t_exp_left * poly_len] condensed
    const uint32_t* __restrict__ d_tables,  // [num_tables * poly_len]
    const uint32_t* __restrict__ d_gen_pows,// [num_rotations]
    size_t t_exp_left,
    size_t poly_len,
    size_t num_rotations)
{
    extern __shared__ uint64_t s_ybody[];  // poly_len uint64s

    size_t rot_base = blockIdx.x * EXPAND_ROTS;

    for (size_t k = 0; k < t_exp_left; k++) {
        // Load y_body[k] into shared memory (ONE global read, shared across all rotations)
        const uint64_t* src = d_y_body + k * poly_len;
        for (size_t z = threadIdx.x; z < poly_len; z += blockDim.x) {
            s_ybody[z] = src[z];
        }
        __syncthreads();

        // Process EXPAND_ROTS rotations using shared memory gather
        for (size_t r = 0; r < EXPAND_ROTS; r++) {
            size_t rot = rot_base + r;
            if (rot >= num_rotations) break;

            uint32_t t = d_gen_pows[rot];
            uint32_t tidx1 = (t - 1) / 2;
            uint32_t tidx2 = (2 * (uint32_t)poly_len - t - 1) / 2;
            const uint32_t* tab1 = d_tables + (size_t)tidx1 * poly_len;
            const uint32_t* tab2 = d_tables + (size_t)tidx2 * poly_len;

            uint64_t* dst1 = d_y_all + (rot * t_exp_left + k) * poly_len;
            uint64_t* dst2 = d_y_bar_all + (rot * t_exp_left + k) * poly_len;

            for (size_t z = threadIdx.x; z < poly_len; z += blockDim.x) {
                uint32_t idx1 = __ldg(&tab1[z]);
                uint32_t idx2 = __ldg(&tab2[z]);
                dst1[z] = s_ybody[idx1];
                dst2[z] = s_ybody[idx2];
            }
        }
        __syncthreads();
    }
}

// ---------- C API ----------

extern "C" {

void* ypir_word_online_init(
    const uint16_t* db,
    size_t db_rows,
    size_t db_rows_padded,
    size_t db_cols,
    size_t t_exp_left,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    uint32_t poly_len,
    uint32_t crt_count,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    const uint64_t* forward_table,
    const uint64_t* forward_prime_table,
    const uint64_t* inverse_table,
    const uint64_t* inverse_prime_table,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus,
    size_t max_batch_size,
    uint64_t inv_n)
{
    WordOnlineContext* ctx = new WordOnlineContext();
    ctx->tc_packing_ctx  = nullptr;

    ctx->db_rows         = db_rows;
    ctx->db_rows_padded  = db_rows_padded;
    ctx->db_cols         = db_cols;
    ctx->num_rlwe_outputs = db_cols / poly_len;
    ctx->t_exp_left      = t_exp_left;
    ctx->max_batch_size  = max_batch_size;
    ctx->rlwe_q_prime_1  = rlwe_q_prime_1;
    ctx->rlwe_q_prime_2  = rlwe_q_prime_2;
    ctx->mod0            = moduli[0];
    ctx->mod1            = moduli[1];
    ctx->modulus         = moduli[0] * moduli[1];
    ctx->inv_n           = inv_n;

    // Precompute sizes
    size_t ell       = 31 - __builtin_clz(poly_len);
    size_t convd_len = crt_count * poly_len;
    size_t ws_size   = (1 << (ell - 1));
    ctx->scratch_per_output =
        (ws_size * poly_len)
        + (2 * convd_len) + (2 * poly_len) + convd_len;
    ctx->pub_params_size_per_batch = ell * t_exp_left * poly_len;

    size_t q_1_bits = word_ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = word_ceil_log2_u64(rlwe_q_prime_1);
    ctx->response_bytes_per_output = ((q_1_bits + q_2_bits) * poly_len + 7) / 8;

    // NTT params
    ctx->ntt_params.poly_len          = poly_len;
    ctx->ntt_params.log2_poly_len     = ell;
    ctx->ntt_params.crt_count         = crt_count;
    ctx->ntt_params.mod0_inv_mod1     = mod0_inv_mod1;
    ctx->ntt_params.mod1_inv_mod0     = mod1_inv_mod0;
    ctx->ntt_params.barrett_cr_0_modulus = barrett_cr_0_modulus;
    ctx->ntt_params.barrett_cr_1_modulus = barrett_cr_1_modulus;
    ctx->ntt_params.modulus           = ctx->modulus;

    size_t table_size  = crt_count * poly_len * sizeof(uint64_t);
    size_t moduli_size = crt_count * sizeof(uint64_t);

    CUDA_ALLOC_AND_COPY(ctx->ntt_params.moduli,              moduli,              moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.barrett_cr,          barrett_cr,          moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_table,       forward_table,       table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_prime_table, forward_prime_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_table,       inverse_table,       table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_prime_table, inverse_prime_table, table_size);

    // Detect tensor cores
    ctx->has_tensor_cores = false;
    ctx->d_db_stacked = nullptr;
    ctx->d_query_bytes_packed = nullptr;
    ctx->d_gemm_out = nullptr;
    ctx->d_db_u16 = nullptr;
    ctx->d_query_buf = nullptr;
    ctx->d_result_u64 = nullptr;
    ctx->d_intermediate = nullptr;
    ctx->d_y_constants = nullptr;
    ctx->d_precomp_res = nullptr;
    ctx->d_precomp_vals = nullptr;
    ctx->d_precomp_tables = nullptr;
    ctx->num_streams = 0;
    ctx->streams = nullptr;
    ctx->d_scratch_batch = nullptr;
    ctx->d_pub_params_all = nullptr;
    ctx->d_all_responses = nullptr;

    // InspiRING fields (initialized later via init_packing_inspir if needed)
    ctx->is_inspir = false;
    ctx->d_bold_t_condensed = nullptr;
    ctx->d_bold_t_bar_condensed = nullptr;
    ctx->d_bold_t_hat_condensed = nullptr;
    ctx->d_a_hat = nullptr;
    ctx->bold_t_per_output = 0;
    ctx->num_iter = 0;
    ctx->inspir_scratch_per_output = 0;
    ctx->d_z_body_streams = nullptr;
    ctx->z_body_per_client = 0;
    ctx->d_tables = nullptr;
    ctx->d_gen_pows = nullptr;
    ctx->num_tables = 0;
    ctx->num_rotations = 0;
    ctx->d_y_body_streams = nullptr;

    {
        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        int major, minor;
        CUDA_ASSERT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_ASSERT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 75);
        ctx->is_sm80 = (sm >= 80);

        printf("Word online GEMM (%s): M=%zu, K=%zu, max_batch=%zu\n",
               ctx->has_tensor_cores ? (ctx->is_sm80 ? "CUTLASS TC Sm80" : "CUTLASS TC Sm75") : "CUTLASS SIMT",
               db_cols, db_rows, max_batch_size);
    }

    size_t M = db_cols;
    size_t K = db_rows;
    size_t max_N = max_batch_size;

    if (ctx->has_tensor_cores) {
        if (db != nullptr) {
            // Upload + decompose DB to vertically stacked uint8 bytes
            size_t db_elems = M * db_rows_padded;
            uint16_t* d_db_raw;
            CUDA_ALLOC_AND_COPY(d_db_raw, db, db_elems * sizeof(uint16_t));

            CUDA_ASSERT(cudaMalloc(&ctx->d_db_stacked, 2 * db_elems));
            {
                int threads = 256;
                int blocks = (db_elems + threads - 1) / threads;
                word_decompose_u16_stacked<<<blocks, threads>>>(ctx->d_db_stacked, d_db_raw, db_elems);
                CUDA_ASSERT(cudaGetLastError());
            }
            CUDA_ASSERT(cudaFree(d_db_raw));
        }
        // else: DB will be adopted via ypir_word_online_adopt_db

        // Packed query byte buffer: K × (8*max_N) contiguous
        CUDA_ASSERT(cudaMalloc(&ctx->d_query_bytes_packed, K * 8 * max_N));

        // Stacked GEMM output buffer: (2M) × (8*max_N) int32
        CUDA_ASSERT(cudaMalloc(&ctx->d_gemm_out, 2 * M * 8 * max_N * sizeof(int32_t)));

    } else {
        if (db != nullptr) {
            // SIMT: keep DB as u16
            size_t db_size = M * db_rows_padded * sizeof(uint16_t);
            CUDA_ALLOC_AND_COPY(ctx->d_db_u16, db, db_size);
        }
        // else: DB will be adopted via ypir_word_online_adopt_db

        // Pre-allocate query + result buffers (full max_batch)
        CUDA_ASSERT(cudaMalloc(&ctx->d_query_buf, max_N * db_rows_padded * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&ctx->d_result_u64, max_N * M * sizeof(uint64_t)));
    }

    // Common: intermediate (full max_batch — cheap: max_N * M * 8 bytes)
    CUDA_ASSERT(cudaMalloc(&ctx->d_intermediate, max_N * M * sizeof(uint64_t)));

    // Pre-allocate pub_params and response buffers (full max_batch)
    CUDA_ASSERT(cudaMalloc(&ctx->d_pub_params_all,
        max_N * ctx->pub_params_size_per_batch * sizeof(uint64_t)));

    size_t response_bytes_per_batch = ctx->num_rlwe_outputs * ctx->response_bytes_per_output;
    CUDA_ASSERT(cudaMalloc(&ctx->d_all_responses,
        max_N * response_bytes_per_batch));

    // Streaming scratch will be allocated in init_packing, after packing data is uploaded

    CUDA_ASSERT(cudaDeviceSynchronize());
    return ctx;
}

void ypir_word_online_init_packing(
    void* context,
    const uint64_t* y_constants,    size_t y_constants_size,
    const uint64_t* precomp_res,    size_t precomp_res_size,
    const uint64_t* precomp_vals,   size_t precomp_vals_size,
    const uint64_t* precomp_tables, size_t precomp_tables_size)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    CUDA_ALLOC_AND_COPY(ctx->d_y_constants,    y_constants,    y_constants_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_res,    precomp_res,    precomp_res_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_vals,   precomp_vals,   precomp_vals_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_tables, precomp_tables, precomp_tables_size);

    // ── Determine num_streams based on remaining GPU memory ──
    // (done here, AFTER all fixed allocations including packing data)

    size_t per_stream_bytes =
        ctx->num_rlwe_outputs * ctx->scratch_per_output * sizeof(uint64_t);

    size_t free_mem, total_mem;
    CUDA_ASSERT(cudaMemGetInfo(&free_mem, &total_mem));
    size_t usable = (free_mem * 9) / 10;  // use 90%, leave room for CUDA overhead

    size_t num_streams = usable / per_stream_bytes;
    if (num_streams < 1) {
        fprintf(stderr, "ERROR: Not enough GPU memory for even 1 stream. "
                "Required per stream: %.2f MB, free: %.2f MB\n",
                per_stream_bytes / 1e6, free_mem / 1e6);
        abort();
    }
    if (num_streams > ctx->max_batch_size) num_streams = ctx->max_batch_size;
    ctx->num_streams = num_streams;

    printf("GPU: %.1f MB free, per-stream scratch %.2f MB, using %zu parallel streams\n",
        free_mem / 1e6, per_stream_bytes / 1e6, num_streams);

    // Create CUDA streams
    ctx->streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        CUDA_ASSERT(cudaStreamCreate(&ctx->streams[i]));
    }

    // Per-stream scratch allocations
    ctx->d_scratch_batch = new uint64_t*[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_scratch_batch[i], per_stream_bytes));
    }

    CUDA_ASSERT(cudaDeviceSynchronize());
}

void ypir_word_online_init_packing_inspir(
    void* context,
    const uint64_t* bold_t_condensed,      size_t bold_t_size,
    const uint64_t* bold_t_bar_condensed,  size_t bold_t_bar_size,
    const uint64_t* bold_t_hat_condensed,  size_t bold_t_hat_size,
    const uint64_t* a_hat,                 size_t a_hat_size,
    size_t num_iter,      // poly_len/2 - 1
    const uint32_t* tables,                size_t num_tables,
    const uint32_t* gen_pows,              size_t num_rotations)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    ctx->is_inspir = true;
    ctx->num_iter = num_iter;
    size_t poly_len = ctx->ntt_params.poly_len;
    size_t t_exp = ctx->t_exp_left;

    // Per-output precomp sizes (dense condensed, stride = poly_len)
    ctx->bold_t_per_output = num_iter * t_exp * poly_len;
    ctx->inspir_scratch_per_output = 4 * poly_len;  // temp_ntt + temp_raw

    // Per-client data sizes
    ctx->z_body_per_client = t_exp * poly_len;

    // Upload per-output precomp data (original layout: [num_outputs, D, poly_len], z stride-1)
    CUDA_ALLOC_AND_COPY(ctx->d_bold_t_condensed,     bold_t_condensed,     bold_t_size);
    CUDA_ALLOC_AND_COPY(ctx->d_bold_t_bar_condensed, bold_t_bar_condensed, bold_t_bar_size);
    CUDA_ALLOC_AND_COPY(ctx->d_bold_t_hat_condensed, bold_t_hat_condensed, bold_t_hat_size);
    CUDA_ALLOC_AND_COPY(ctx->d_a_hat,                a_hat,                a_hat_size);

    printf("InspiRING precomp uploaded: bold_t %.2f MB, bold_t_bar %.2f MB, bold_t_hat %.2f MB, a_hat %.2f MB\n",
        bold_t_size / 1e6, bold_t_bar_size / 1e6, bold_t_hat_size / 1e6, a_hat_size / 1e6);

    // Upload expand permutation tables and generator powers (once, shared across all clients)
    ctx->num_tables = num_tables;
    ctx->num_rotations = num_rotations;
    CUDA_ALLOC_AND_COPY(ctx->d_tables, tables, num_tables * poly_len * sizeof(uint32_t));
    CUDA_ALLOC_AND_COPY(ctx->d_gen_pows, gen_pows, num_rotations * sizeof(uint32_t));

    printf("Expand tables uploaded: %zu tables × %zu = %.2f MB, gen_pows %zu × 4 = %.2f KB\n",
        num_tables, poly_len,
        num_tables * poly_len * sizeof(uint32_t) / 1e6,
        num_rotations, num_rotations * 4 / 1e3);

    // ── Determine num_streams based on remaining GPU memory ──
    // Fused expand+IP: no y_all/y_bar_all buffers needed (y_body fits in L2)

    size_t per_stream_bytes =
        ctx->z_body_per_client * sizeof(uint64_t) +      // z_body (uploaded)
        ctx->z_body_per_client * sizeof(uint64_t) +      // y_body (uploaded, used by fused kernel)
        ctx->num_rlwe_outputs * ctx->inspir_scratch_per_output * sizeof(uint64_t);  // scratch

    size_t free_mem, total_mem;
    CUDA_ASSERT(cudaMemGetInfo(&free_mem, &total_mem));
    size_t usable = (free_mem * 9) / 10;

    size_t num_streams = usable / per_stream_bytes;
    if (num_streams < 1) {
        fprintf(stderr, "ERROR: Not enough GPU memory for even 1 InspiRING stream. "
                "Required per stream: %.2f MB, free: %.2f MB\n",
                per_stream_bytes / 1e6, free_mem / 1e6);
        abort();
    }
    if (num_streams > ctx->max_batch_size) num_streams = ctx->max_batch_size;
    ctx->num_streams = num_streams;

    printf("InspiRING GPU: %.1f MB free, per-stream %.2f KB, "
           "using %zu parallel streams (fused expand+IP)\n",
        free_mem / 1e6, per_stream_bytes / 1e3, num_streams);

    // Create CUDA streams
    ctx->streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        CUDA_ASSERT(cudaStreamCreate(&ctx->streams[i]));
    }

    // Per-stream client data buffers (fused: no y_all/y_bar_all needed)
    size_t z_body_bytes    = ctx->z_body_per_client * sizeof(uint64_t);
    size_t scratch_bytes   = ctx->num_rlwe_outputs * ctx->inspir_scratch_per_output * sizeof(uint64_t);

    CUDA_ASSERT(cudaMalloc(&ctx->d_z_body_streams,    num_streams * z_body_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_y_body_streams,    num_streams * z_body_bytes));

    // Contiguous scratch allocation (so batched kernel can index by client)
    ctx->scratch_bytes_per_stream = scratch_bytes;
    CUDA_ASSERT(cudaMalloc(&ctx->d_scratch_contiguous, num_streams * scratch_bytes));
    ctx->d_scratch_batch = new uint64_t*[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        ctx->d_scratch_batch[i] = ctx->d_scratch_contiguous + i * (scratch_bytes / sizeof(uint64_t));
    }

    CUDA_ASSERT(cudaDeviceSynchronize());
}

// Variant that takes GPU device pointers directly (from inspir_precomp).
// No cudaMalloc/cudaMemcpy for bold_t data — pointers are adopted.
void ypir_word_online_init_packing_inspir_from_gpu(
    void* context,
    uint64_t* d_bold_t_condensed,
    uint64_t* d_bold_t_bar_condensed,
    uint64_t* d_bold_t_hat_condensed,
    uint64_t* d_a_hat,
    size_t num_iter,
    const uint32_t* tables,     // host
    size_t num_tables,
    const uint32_t* gen_pows,   // host
    size_t num_rotations)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    ctx->is_inspir = true;
    ctx->num_iter = num_iter;
    size_t poly_len = ctx->ntt_params.poly_len;
    size_t t_exp = ctx->t_exp_left;

    ctx->bold_t_per_output = num_iter * t_exp * poly_len;
    ctx->inspir_scratch_per_output = 4 * poly_len;
    ctx->z_body_per_client = t_exp * poly_len;

    // Adopt device pointers directly (ownership transfers from precomp context)
    ctx->d_bold_t_condensed = d_bold_t_condensed;
    ctx->d_bold_t_bar_condensed = d_bold_t_bar_condensed;
    ctx->d_bold_t_hat_condensed = d_bold_t_hat_condensed;
    ctx->d_a_hat = d_a_hat;

    printf("InspiRING precomp adopted from GPU (zero-copy)\n");

    // Upload expand tables (from host)
    ctx->num_tables = num_tables;
    ctx->num_rotations = num_rotations;
    CUDA_ALLOC_AND_COPY(ctx->d_tables, tables, num_tables * poly_len * sizeof(uint32_t));
    CUDA_ALLOC_AND_COPY(ctx->d_gen_pows, gen_pows, num_rotations * sizeof(uint32_t));

    // Allocate per-stream buffers (same as host-upload variant)
    size_t per_stream_bytes =
        ctx->z_body_per_client * sizeof(uint64_t) +
        ctx->z_body_per_client * sizeof(uint64_t) +
        ctx->num_rlwe_outputs * ctx->inspir_scratch_per_output * sizeof(uint64_t);

    size_t free_mem, total_mem;
    CUDA_ASSERT(cudaMemGetInfo(&free_mem, &total_mem));
    size_t usable = (free_mem * 9) / 10;

    size_t num_streams = usable / per_stream_bytes;
    if (num_streams < 1) {
        fprintf(stderr, "ERROR: Not enough GPU memory for InspiRING streams\n");
        abort();
    }
    if (num_streams > ctx->max_batch_size) num_streams = ctx->max_batch_size;
    ctx->num_streams = num_streams;

    printf("InspiRING GPU (from_gpu): using %zu parallel streams\n", num_streams);

    ctx->streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; i++)
        CUDA_ASSERT(cudaStreamCreate(&ctx->streams[i]));

    size_t z_body_bytes = ctx->z_body_per_client * sizeof(uint64_t);
    size_t scratch_bytes = ctx->num_rlwe_outputs * ctx->inspir_scratch_per_output * sizeof(uint64_t);

    CUDA_ASSERT(cudaMalloc(&ctx->d_z_body_streams, num_streams * z_body_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_y_body_streams, num_streams * z_body_bytes));

    ctx->scratch_bytes_per_stream = scratch_bytes;
    CUDA_ASSERT(cudaMalloc(&ctx->d_scratch_contiguous, num_streams * scratch_bytes));
    ctx->d_scratch_batch = new uint64_t*[num_streams];
    for (size_t i = 0; i < num_streams; i++)
        ctx->d_scratch_batch[i] = ctx->d_scratch_contiguous + i * (scratch_bytes / sizeof(uint64_t));

    CUDA_ASSERT(cudaDeviceSynchronize());

    // Optionally initialize tc_packing path (env var toggle)
    const char* tc_env = getenv("YPIR_TC_PACKING");
    if (tc_env && tc_env[0] == '1') {
        size_t q_1_bits = word_ceil_log2_u64(ctx->rlwe_q_prime_2);
        size_t q_2_bits = word_ceil_log2_u64(ctx->rlwe_q_prime_1);
        size_t resp_bpo = ((q_1_bits + q_2_bits) * poly_len + 7) / 8;

        ctx->tc_packing_ctx = tc_packing_init(
            ctx->d_bold_t_condensed,
            ctx->d_bold_t_bar_condensed,
            ctx->d_bold_t_hat_condensed,
            ctx->d_a_hat,
            ctx->d_tables,
            ctx->d_gen_pows,
            ctx->num_iter,
            ctx->t_exp_left,
            poly_len,
            ctx->num_rlwe_outputs,
            ctx->max_batch_size,
            ctx->rlwe_q_prime_1,
            ctx->rlwe_q_prime_2,
            resp_bpo,
            ctx->ntt_params,
            ctx->is_sm80 ? 2 : (ctx->has_tensor_cores ? 1 : 0));
        printf("TcPacking enabled via YPIR_TC_PACKING=1\n");

        // tc_packing_init took ownership and freed bold_t/bold_t_bar — null our pointers
        ctx->d_bold_t_condensed = nullptr;
        ctx->d_bold_t_bar_condensed = nullptr;
    }
}

void ypir_word_online_compute_batch(
    void* context,
    const uint64_t* queries,                  // batch_size × db_rows (raw u64)
    const uint64_t* pack_pub_params_row_1s,
    uint8_t* response_out,
    size_t response_bytes_per_batch,
    size_t batch_size)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx || batch_size == 0) return;
    if (batch_size > ctx->max_batch_size) {
        fprintf(stderr, "batch_size %zu exceeds max_batch_size %zu\n", batch_size, ctx->max_batch_size);
        abort();
    }

    GpuTimer t;

    size_t M = ctx->db_cols;
    size_t K = ctx->db_rows;
    size_t N = batch_size;

    size_t poly_len    = ctx->ntt_params.poly_len;
    size_t num_outputs = ctx->num_rlwe_outputs;

    // ── Step 1: GEMM (full batch, default stream) ──

    t.tic();

    if (ctx->has_tensor_cores) {
        // Upload queries and decompose to packed uint8 bytes
        size_t q_elems = N * K;
        uint64_t* d_query_raw;
        CUDA_ALLOC_AND_COPY(d_query_raw, queries, q_elems * sizeof(uint64_t));
        {
            int threads = 256;
            int blocks = (q_elems + threads - 1) / threads;
            word_decompose_u64_bytes_packed<<<blocks, threads>>>(
                ctx->d_query_bytes_packed, d_query_raw, K, N);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaFree(d_query_raw));

        size_t count = M * N;

        // Stacked GEMM: (2M) × K × K × (8N) → (2M) × (8N)
        {
            auto status = run_u8tc_gemm(ctx->is_sm80, 2*M, 8*N, K,
                ctx->d_db_stacked, (int)ctx->db_rows_padded,
                ctx->d_query_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)(2*M));
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC stacked GEMM failed\n"); abort(); }
        }

        // Accumulate stacked output + modswitch
        {
            int threads = 256, blocks = (count + threads - 1) / threads;
            word_accumulate_stacked_and_modswitch<<<blocks, threads>>>(
                ctx->d_intermediate, ctx->d_gemm_out, M, N,
                ctx->modulus, ctx->inv_n);
            CUDA_ASSERT(cudaGetLastError());
        }

    } else {
        // Upload queries
        size_t q_elems = N * K;
        CUDA_ASSERT(cudaMemcpy(ctx->d_query_buf, queries,
                               q_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // CUTLASS GEMM: uint16 DB × uint64 query → uint64
        cutlass::gemm::GemmCoord problem_size(M, N, K);
        uint64_t alpha = 1, beta = 0;
        CutlassGemmWord::Arguments args{
            problem_size,
            {ctx->d_db_u16, (int)ctx->db_rows_padded},
            {ctx->d_query_buf, (int)K},
            {ctx->d_result_u64, (int)M},
            {ctx->d_result_u64, (int)M},
            {alpha, beta}, 1
        };
        CutlassGemmWord gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
            abort();
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
            abort();
        }

        // Modswitch + multiply by inv_N in-place, then copy to d_intermediate
        {
            size_t count = M * N;
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            word_modswitch_crt_inplace<<<blocks, threads>>>(
                ctx->d_result_u64, count, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaMemcpy(ctx->d_intermediate, ctx->d_result_u64,
                               M * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    // Upload pub_params (full batch, default stream)
    CUDA_ASSERT(cudaMemcpy(ctx->d_pub_params_all, pack_pub_params_row_1s,
                           batch_size * ctx->pub_params_size_per_batch * sizeof(uint64_t),
                           cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaDeviceSynchronize());
    float ms1 = t.toc_ms();

    // ── Step 2: Packing (chunked by num_streams) ──

    t.tic();

    size_t resp_bytes_per_batch = num_outputs * ctx->response_bytes_per_output;
    int threads_pack = 1024;

    for (size_t chunk_start = 0; chunk_start < batch_size; chunk_start += ctx->num_streams) {
        size_t chunk_end = chunk_start + ctx->num_streams;
        if (chunk_end > batch_size) chunk_end = batch_size;

        // Launch one packing kernel per batch item on its own stream
        for (size_t i = chunk_start; i < chunk_end; i++) {
            size_t s = i - chunk_start;  // stream/scratch index
            cudaStream_t stream = ctx->streams[s];

            // Pointers offset to this batch item
            uint8_t* d_resp_i = ctx->d_all_responses + i * resp_bytes_per_batch;
            const uint64_t* d_inter_i = ctx->d_intermediate + i * M;
            const uint64_t* d_pub_i = ctx->d_pub_params_all + i * ctx->pub_params_size_per_batch;

            // Grid: one block per output (1D), on this stream
            word_pack_lwes_and_mod_switch<<<num_outputs, threads_pack, 0, stream>>>(
                d_resp_i,
                d_inter_i,
                ctx->d_y_constants,
                ctx->d_precomp_res,
                ctx->d_precomp_vals,
                ctx->d_precomp_tables,
                d_pub_i,
                ctx->d_scratch_batch[s],
                num_outputs,
                ctx->db_cols,
                ctx->t_exp_left,
                ctx->rlwe_q_prime_1,
                ctx->rlwe_q_prime_2,
                ctx->response_bytes_per_output,
                ctx->scratch_per_output,
                ctx->pub_params_size_per_batch,
                ctx->ntt_params);
            CUDA_ASSERT(cudaGetLastError());
        }

        // Synchronize all streams in this chunk
        for (size_t i = chunk_start; i < chunk_end; i++) {
            size_t s = i - chunk_start;
            CUDA_ASSERT(cudaStreamSynchronize(ctx->streams[s]));
        }
    }

    float ms2 = t.toc_ms();

    printf("Word Step1(%s) %.3f ms, Step2 (%zu streams, %zu chunks) %.3f ms\n",
           ctx->has_tensor_cores ? "TC" : "SIMT", ms1,
           ctx->num_streams, (batch_size + ctx->num_streams - 1) / ctx->num_streams, ms2);

    // Download all responses
    CUDA_ASSERT(cudaMemcpy(response_out, ctx->d_all_responses,
                           batch_size * resp_bytes_per_batch,
                           cudaMemcpyDeviceToHost));
}

// Step 1 only: GEMM + modswitch + inv_N, then copy intermediates to host.
// Used for InspiRING packing (packing done on CPU).
void ypir_word_online_compute_matmul_only(
    void* context,
    const uint64_t* queries,          // batch_size × db_rows (raw u64)
    uint64_t* intermediate_out,       // host buffer: batch_size × db_cols
    size_t batch_size)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx || batch_size == 0) return;
    if (batch_size > ctx->max_batch_size) {
        fprintf(stderr, "batch_size %zu exceeds max_batch_size %zu\n", batch_size, ctx->max_batch_size);
        abort();
    }

    size_t M = ctx->db_cols;
    size_t K = ctx->db_rows;
    size_t N = batch_size;

    // ── Step 1: GEMM (full batch, default stream) ──

    if (ctx->has_tensor_cores) {
        // Upload queries and decompose to packed uint8 bytes
        size_t q_elems = N * K;
        uint64_t* d_query_raw;
        CUDA_ALLOC_AND_COPY(d_query_raw, queries, q_elems * sizeof(uint64_t));
        {
            int threads = 256;
            int blocks = (q_elems + threads - 1) / threads;
            word_decompose_u64_bytes_packed<<<blocks, threads>>>(
                ctx->d_query_bytes_packed, d_query_raw, K, N);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaFree(d_query_raw));

        size_t count = M * N;

        // Stacked GEMM: (2M) × K × K × (8N) → (2M) × (8N)
        {
            auto status = run_u8tc_gemm(ctx->is_sm80, 2*M, 8*N, K,
                ctx->d_db_stacked, (int)ctx->db_rows_padded,
                ctx->d_query_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)(2*M));
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC stacked GEMM failed\n"); abort(); }
        }
        { int threads = 256, blocks = (count + threads - 1) / threads;
          word_accumulate_stacked_and_modswitch<<<blocks, threads>>>(
              ctx->d_intermediate, ctx->d_gemm_out, M, N, ctx->modulus, ctx->inv_n);
          CUDA_ASSERT(cudaGetLastError()); }

    } else {
        // Upload queries
        size_t q_elems = N * K;
        CUDA_ASSERT(cudaMemcpy(ctx->d_query_buf, queries,
                               q_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // CUTLASS GEMM: uint16 DB × uint64 query → uint64
        cutlass::gemm::GemmCoord problem_size(M, N, K);
        uint64_t alpha = 1, beta = 0;
        CutlassGemmWord::Arguments args{
            problem_size,
            {ctx->d_db_u16, (int)ctx->db_rows_padded},
            {ctx->d_query_buf, (int)K},
            {ctx->d_result_u64, (int)M},
            {ctx->d_result_u64, (int)M},
            {alpha, beta}, 1
        };
        CutlassGemmWord gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
            abort();
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
            abort();
        }

        // Modswitch + multiply by inv_N in-place, then copy to d_intermediate
        {
            size_t count = M * N;
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            word_modswitch_crt_inplace<<<blocks, threads>>>(
                ctx->d_result_u64, count, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaMemcpy(ctx->d_intermediate, ctx->d_result_u64,
                               M * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    CUDA_ASSERT(cudaDeviceSynchronize());

    // Download intermediates to host
    CUDA_ASSERT(cudaMemcpy(intermediate_out, ctx->d_intermediate,
                           M * N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

void ypir_word_online_compute_batch_inspir(
    void* context,
    const uint64_t* queries,                  // batch_size × db_rows (raw u64)
    const uint64_t* y_body_condensed,         // batch_size × z_body_per_client (tiny, expand input)
    const uint64_t* z_body_condensed,         // batch_size × z_body_per_client
    uint8_t* response_out,
    size_t response_bytes_per_batch,
    size_t batch_size)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx || batch_size == 0) return;
    if (batch_size > ctx->max_batch_size) {
        fprintf(stderr, "batch_size %zu exceeds max_batch_size %zu\n", batch_size, ctx->max_batch_size);
        abort();
    }

    GpuTimer t;

    size_t M = ctx->db_cols;
    size_t K = ctx->db_rows;
    size_t N = batch_size;

    size_t poly_len    = ctx->ntt_params.poly_len;
    size_t num_outputs = ctx->num_rlwe_outputs;

    // ── Step 1: GEMM (full batch, default stream) ──

    cudaEvent_t step1_start;
    CUDA_ASSERT(cudaEventCreate(&step1_start));
    CUDA_ASSERT(cudaEventRecord(step1_start, 0));
    t.tic();

    if (ctx->has_tensor_cores) {
        size_t q_elems = N * K;
        uint64_t* d_query_raw;
        CUDA_ALLOC_AND_COPY(d_query_raw, queries, q_elems * sizeof(uint64_t));
        {
            int threads = 256;
            int blocks = (q_elems + threads - 1) / threads;
            word_decompose_u64_bytes_packed<<<blocks, threads>>>(
                ctx->d_query_bytes_packed, d_query_raw, K, N);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaFree(d_query_raw));

        size_t count = M * N;

        // Stacked GEMM: (2M) × K × K × (8N) → (2M) × (8N)
        {
            auto status = run_u8tc_gemm(ctx->is_sm80, 2*M, 8*N, K,
                ctx->d_db_stacked, (int)ctx->db_rows_padded,
                ctx->d_query_bytes_packed, (int)K,
                ctx->d_gemm_out, (int)(2*M));
            if (status != cutlass::Status::kSuccess) { fprintf(stderr, "CUTLASS TC stacked GEMM failed\n"); abort(); }
        }
        { int threads = 256, blocks = (count + threads - 1) / threads;
          word_accumulate_stacked_and_modswitch<<<blocks, threads>>>(
              ctx->d_intermediate, ctx->d_gemm_out, M, N, ctx->modulus, ctx->inv_n);
          CUDA_ASSERT(cudaGetLastError()); }

    } else {
        size_t q_elems = N * K;
        CUDA_ASSERT(cudaMemcpy(ctx->d_query_buf, queries,
                               q_elems * sizeof(uint64_t), cudaMemcpyHostToDevice));

        cutlass::gemm::GemmCoord problem_size(M, N, K);
        uint64_t alpha = 1, beta = 0;
        CutlassGemmWord::Arguments args{
            problem_size,
            {ctx->d_db_u16, (int)ctx->db_rows_padded},
            {ctx->d_query_buf, (int)K},
            {ctx->d_result_u64, (int)M},
            {ctx->d_result_u64, (int)M},
            {alpha, beta}, 1
        };
        CutlassGemmWord gemm_op;
        cutlass::Status status = gemm_op.initialize(args, nullptr);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
            abort();
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
            abort();
        }

        {
            size_t count = M * N;
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            word_modswitch_crt_inplace<<<blocks, threads>>>(
                ctx->d_result_u64, count, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaMemcpy(ctx->d_intermediate, ctx->d_result_u64,
                               M * N * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
    }

    // Wait for Step1 to complete before packing
    CUDA_ASSERT(cudaDeviceSynchronize());

    cudaEvent_t step1_timing_end;
    CUDA_ASSERT(cudaEventCreate(&step1_timing_end));
    CUDA_ASSERT(cudaEventRecord(step1_timing_end, 0));

    // ── Step 2: InspiRING packing ──

    size_t resp_bytes_per_batch = num_outputs * ctx->response_bytes_per_output;

    if (ctx->tc_packing_ctx) {
        // ── TC packing path (Phase 1: SIMT kernels via separate context) ──
        // Upload y_body + z_body to GPU
        size_t z_body_bytes = ctx->z_body_per_client * sizeof(uint64_t);
        CUDA_ASSERT(cudaMemcpy(ctx->d_y_body_streams,
            y_body_condensed, batch_size * z_body_bytes, cudaMemcpyHostToDevice));
        CUDA_ASSERT(cudaMemcpy(ctx->d_z_body_streams,
            z_body_condensed, batch_size * z_body_bytes, cudaMemcpyHostToDevice));

        tc_packing_run(
            ctx->tc_packing_ctx,
            ctx->d_intermediate,
            ctx->d_y_body_streams,
            ctx->d_z_body_streams,
            ctx->d_all_responses,
            batch_size);

    } else {
        // ── Original SIMT packing path ──
        size_t z_body_bytes = ctx->z_body_per_client * sizeof(uint64_t);
        size_t D = ctx->num_iter * ctx->t_exp_left;

        for (size_t chunk_start = 0; chunk_start < batch_size; chunk_start += ctx->num_streams) {
            size_t chunk_end = chunk_start + ctx->num_streams;
            if (chunk_end > batch_size) chunk_end = batch_size;
            size_t chunk_size = chunk_end - chunk_start;

            size_t scratch_stride = ctx->num_rlwe_outputs * ctx->inspir_scratch_per_output;

            // Upload y_body + z_body on per-client streams
            for (size_t i = chunk_start; i < chunk_end; i++) {
                size_t s = i - chunk_start;
                cudaStream_t stream = ctx->streams[s];

                uint64_t* d_y_body_s = ctx->d_y_body_streams + s * ctx->z_body_per_client;
                uint64_t* d_z_body_s = ctx->d_z_body_streams + s * ctx->z_body_per_client;

                CUDA_ASSERT(cudaMemcpyAsync(d_y_body_s,
                    y_body_condensed + i * ctx->z_body_per_client,
                    z_body_bytes, cudaMemcpyHostToDevice, stream));
                CUDA_ASSERT(cudaMemcpyAsync(d_z_body_s,
                    z_body_condensed + i * ctx->z_body_per_client,
                    z_body_bytes, cudaMemcpyHostToDevice, stream));
            }

            // Sync all per-client streams to ip_stream
            cudaStream_t ip_stream = ctx->streams[0];
            for (size_t s = 0; s < chunk_size; s++) {
                cudaEvent_t ev;
                CUDA_ASSERT(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
                CUDA_ASSERT(cudaEventRecord(ev, ctx->streams[s]));
                CUDA_ASSERT(cudaStreamWaitEvent(ip_stream, ev, 0));
                CUDA_ASSERT(cudaEventDestroy(ev));
            }

            if (num_outputs > 1) {
                launch_multi_output_kernel(
                    num_outputs, chunk_size, poly_len, ip_stream,
                    ctx->d_scratch_contiguous,
                    ctx->d_bold_t_condensed,
                    ctx->d_bold_t_bar_condensed,
                    ctx->d_bold_t_hat_condensed,
                    ctx->d_y_body_streams,
                    ctx->d_z_body_streams,
                    ctx->d_tables,
                    ctx->d_gen_pows,
                    chunk_size, D, ctx->t_exp_left,
                    ctx->inspir_scratch_per_output,
                    ctx->z_body_per_client,
                    ctx->z_body_per_client,
                    scratch_stride,
                    ctx->ntt_params);
            } else {
                size_t batch_per_block = chunk_size;
                if (batch_per_block > 32) batch_per_block = 32;
                dim3 fip_block(FIP_TILE_Z, batch_per_block);
                dim3 fip_grid((poly_len + FIP_TILE_Z - 1) / FIP_TILE_Z,
                               (chunk_size + batch_per_block - 1) / batch_per_block,
                               num_outputs);

                inspir_fused_expand_ip<<<fip_grid, fip_block, 0, ip_stream>>>(
                    ctx->d_scratch_contiguous,
                    ctx->d_bold_t_condensed,
                    ctx->d_bold_t_bar_condensed,
                    ctx->d_bold_t_hat_condensed,
                    ctx->d_y_body_streams,
                    ctx->d_z_body_streams,
                    ctx->d_tables,
                    ctx->d_gen_pows,
                    chunk_size,
                    num_outputs,
                    D,
                    poly_len,
                    ctx->t_exp_left,
                    ctx->inspir_scratch_per_output,
                    ctx->z_body_per_client,
                    ctx->z_body_per_client,
                    scratch_stride,
                    ctx->ntt_params);
                CUDA_ASSERT(cudaGetLastError());
            }

            // Make all per-client streams wait for batched IP to finish
            {
                cudaEvent_t ip_done;
                CUDA_ASSERT(cudaEventCreateWithFlags(&ip_done, cudaEventDisableTiming));
                CUDA_ASSERT(cudaEventRecord(ip_done, ip_stream));
                for (size_t s = 0; s < chunk_size; s++) {
                    CUDA_ASSERT(cudaStreamWaitEvent(ctx->streams[s], ip_done, 0));
                }
                CUDA_ASSERT(cudaEventDestroy(ip_done));
            }

            // Post-process on per-client streams
            for (size_t i = chunk_start; i < chunk_end; i++) {
                size_t s = i - chunk_start;
                cudaStream_t stream = ctx->streams[s];

                uint8_t* d_resp_i = ctx->d_all_responses + i * resp_bytes_per_batch;
                const uint64_t* d_inter_i = ctx->d_intermediate + i * M;

                inspir_post_process<<<num_outputs, 1024, 0, stream>>>(
                    d_resp_i,
                    d_inter_i,
                    ctx->d_a_hat,
                    ctx->d_scratch_batch[s],
                    num_outputs,
                    ctx->rlwe_q_prime_1,
                    ctx->rlwe_q_prime_2,
                    ctx->response_bytes_per_output,
                    ctx->inspir_scratch_per_output,
                    ctx->ntt_params);
                CUDA_ASSERT(cudaGetLastError());
            }

            for (size_t s = 0; s < chunk_size; s++) {
                CUDA_ASSERT(cudaStreamSynchronize(ctx->streams[s]));
            }
        }
    }

    // Measure total time (from before Step1 to after all packing done)
    float ms_total = t.toc_ms();

    // Measure Step1 time separately
    float ms_step1;
    CUDA_ASSERT(cudaEventElapsedTime(&ms_step1, step1_start, step1_timing_end));

    printf("Word InspiRING Step1(%s)=%.1f ms, total=%.1f ms, packing_overhead=%.1f ms (%zu clients, %s)\n",
           ctx->has_tensor_cores ? "TC" : "SIMT", ms_step1, ms_total,
           ms_total - ms_step1, batch_size,
           ctx->tc_packing_ctx ? "tc_packing" : "simt");

    CUDA_ASSERT(cudaEventDestroy(step1_start));
    CUDA_ASSERT(cudaEventDestroy(step1_timing_end));

    // Download all responses
    CUDA_ASSERT(cudaMemcpy(response_out, ctx->d_all_responses,
                           batch_size * resp_bytes_per_batch,
                           cudaMemcpyDeviceToHost));
}

// Adopt DB device pointers from offline context (avoids second DB upload).
// Frees any existing DB allocations in the online context first.
void ypir_word_online_adopt_db(void* context, uint8_t* d_db_stacked, uint16_t* d_db_u16)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    // Free existing DB allocations if any
    if (ctx->d_db_stacked) { cudaFree(ctx->d_db_stacked); ctx->d_db_stacked = nullptr; }
    if (ctx->d_db_u16)     { cudaFree(ctx->d_db_u16);     ctx->d_db_u16 = nullptr; }

    ctx->d_db_stacked = d_db_stacked;
    ctx->d_db_u16 = d_db_u16;
}

void ypir_word_online_free(void* context)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    if (ctx->d_db_stacked) cudaFree(ctx->d_db_stacked);
    if (ctx->d_query_bytes_packed) cudaFree(ctx->d_query_bytes_packed);
    if (ctx->d_gemm_out) cudaFree(ctx->d_gemm_out);
    if (ctx->d_db_u16) cudaFree(ctx->d_db_u16);
    if (ctx->d_query_buf) cudaFree(ctx->d_query_buf);
    if (ctx->d_result_u64) cudaFree(ctx->d_result_u64);
    if (ctx->d_intermediate) cudaFree(ctx->d_intermediate);

    if (ctx->d_y_constants)    cudaFree(ctx->d_y_constants);
    if (ctx->d_precomp_res)    cudaFree(ctx->d_precomp_res);
    if (ctx->d_precomp_vals)   cudaFree(ctx->d_precomp_vals);
    if (ctx->d_precomp_tables) cudaFree(ctx->d_precomp_tables);

    if (ctx->d_pub_params_all) cudaFree(ctx->d_pub_params_all);
    if (ctx->d_all_responses)  cudaFree(ctx->d_all_responses);

    // InspiRING allocations
    if (ctx->d_bold_t_condensed)     cudaFree(ctx->d_bold_t_condensed);
    if (ctx->d_bold_t_bar_condensed) cudaFree(ctx->d_bold_t_bar_condensed);
    if (ctx->d_bold_t_hat_condensed) cudaFree(ctx->d_bold_t_hat_condensed);
    if (ctx->d_a_hat)                cudaFree(ctx->d_a_hat);
    if (ctx->d_z_body_streams)       cudaFree(ctx->d_z_body_streams);
    if (ctx->d_tables)               cudaFree(ctx->d_tables);
    if (ctx->d_gen_pows)             cudaFree(ctx->d_gen_pows);
    if (ctx->d_y_body_streams)       cudaFree(ctx->d_y_body_streams);

    // Free tc_packing context if initialized
    if (ctx->tc_packing_ctx) tc_packing_free(ctx->tc_packing_ctx);

    // Free contiguous scratch and pointer array
    if (ctx->d_scratch_contiguous) cudaFree(ctx->d_scratch_contiguous);
    if (ctx->d_scratch_batch) delete[] ctx->d_scratch_batch;
    if (ctx->streams) {
        for (size_t i = 0; i < ctx->num_streams; i++) {
            cudaStreamDestroy(ctx->streams[i]);
        }
        delete[] ctx->streams;
    }

    cudaFree(ctx->ntt_params.moduli);
    cudaFree(ctx->ntt_params.barrett_cr);
    cudaFree(ctx->ntt_params.forward_table);
    cudaFree(ctx->ntt_params.forward_prime_table);
    cudaFree(ctx->ntt_params.inverse_table);
    cudaFree(ctx->ntt_params.inverse_prime_table);

    delete ctx;
}

} // extern "C"
