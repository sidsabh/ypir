/**
 * Word-Based SimplePIR Online Kernel
 *
 * Step 1: query × DB in Z_{2^64}, modswitch to Z_Q, CRT-pack.
 *   Tensor cores (SM≥75): 15 cuBLAS int8 GEMMs with 2 accumulators
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
#include <cublas_v2.h>
#include "ntt.cuh"

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

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status_); \
        abort(); \
    } \
} while(0)

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

// ---------- Decomposition kernels ----------

__global__ void word_decompose_u16_bytes(
    int8_t* __restrict__ b0,
    int8_t* __restrict__ b1,
    const uint16_t* __restrict__ data,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint16_t v = data[idx];
        b0[idx] = (int8_t)(v & 0xFF);
        b1[idx] = (int8_t)((v >> 8) & 0xFF);
    }
}

__global__ void word_decompose_u64_bytes(
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
        b0[idx] = (int8_t)(v & 0xFF);
        b1[idx] = (int8_t)((v >> 8) & 0xFF);
        b2[idx] = (int8_t)((v >> 16) & 0xFF);
        b3[idx] = (int8_t)((v >> 24) & 0xFF);
        b4[idx] = (int8_t)((v >> 32) & 0xFF);
        b5[idx] = (int8_t)((v >> 40) & 0xFF);
        b6[idx] = (int8_t)((v >> 48) & 0xFF);
        b7[idx] = (int8_t)((v >> 56) & 0xFF);
    }
}

// ---------- Combine + modswitch + CRT-pack ----------

// Tensor core path: combine lo/hi int32 → u64, modswitch to Z_Q, multiply by inv_N
__global__ void word_combine_modswitch_crt(
    uint64_t* __restrict__ out,        // batch × db_cols (row-major, batch is outer)
    const int32_t* __restrict__ lo,    // col-major M×N = db_cols × batch
    const int32_t* __restrict__ hi,
    size_t count,
    uint64_t q, uint64_t mod0, uint64_t mod1,
    uint64_t inv_n)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t val = (uint64_t)(uint32_t)lo[idx] | ((uint64_t)(uint32_t)hi[idx] << 32);

    __uint128_t prod = (__uint128_t)val * q + (((__uint128_t)1) << 63);
    uint64_t val_q = (uint64_t)(prod >> 64);

    // Multiply by inv_N mod Q: compensates for CDKS packing multiplying by N
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);

    out[idx] = val_q;
}

// SIMT path: modswitch u64 to Z_Q (in-place on col-major output), multiply by inv_N
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

    // Multiply by inv_N mod Q: compensates for CDKS packing multiplying by N
    val_q = (uint64_t)((__uint128_t)val_q * inv_n % q);

    data[idx] = val_q;
}

// ---------- GEMM spec for 15 tensor core GEMMs ----------

struct WordGemmSpec {
    int db_b;
    int q_b;
    int32_t alpha;
    int32_t beta;
    int out;
};

static const WordGemmSpec WORD_ONLINE_SPECS[15] = {
    {0, 0, 1,        0, 0},
    {0, 1, 256,      1, 0},
    {1, 0, 256,      1, 0},
    {0, 2, 65536,    1, 0},
    {1, 1, 65536,    1, 0},
    {0, 3, 16777216, 1, 0},
    {1, 2, 16777216, 1, 0},
    {0, 4, 1,        0, 1},
    {1, 3, 1,        1, 1},
    {0, 5, 256,      1, 1},
    {1, 4, 256,      1, 1},
    {0, 6, 65536,    1, 1},
    {1, 5, 65536,    1, 1},
    {0, 7, 16777216, 1, 1},
    {1, 6, 16777216, 1, 1},
};

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

// ---------- Context ----------

struct WordOnlineContext {
    bool has_tensor_cores;
    cublasHandle_t cublas_handle;

    // Tensor core path
    int8_t* d_db_bytes[2];         // DB decomposed, persistent
    int8_t* d_query_bytes[8];      // Per-batch query byte slices
    int32_t* d_partials[2];        // lo/hi accumulators

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
};

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
    ctx->cublas_handle = nullptr;
    for (int i = 0; i < 2; i++) { ctx->d_db_bytes[i] = nullptr; ctx->d_partials[i] = nullptr; }
    for (int i = 0; i < 8; i++) ctx->d_query_bytes[i] = nullptr;
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

    {
        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        int major, minor;
        CUDA_ASSERT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_ASSERT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 75);

        if (ctx->has_tensor_cores && db_rows > 133000) {
            fprintf(stderr, "ERROR: db_rows=%zu exceeds max safe K for int8 tensor core accumulation.\n", db_rows);
            delete ctx;
            return nullptr;
        }

        printf("Word online GEMM (%s): M=%zu, K=%zu, max_batch=%zu\n",
               ctx->has_tensor_cores ? "tensor core" : "CUTLASS SIMT",
               db_cols, db_rows, max_batch_size);
    }

    size_t M = db_cols;
    size_t K = db_rows;
    size_t max_N = max_batch_size;

    if (ctx->has_tensor_cores) {
        // Upload + decompose DB
        size_t db_elems = M * db_rows_padded;
        uint16_t* d_db_raw;
        CUDA_ALLOC_AND_COPY(d_db_raw, db, db_elems * sizeof(uint16_t));

        for (int i = 0; i < 2; i++)
            CUDA_ASSERT(cudaMalloc(&ctx->d_db_bytes[i], db_elems));
        {
            int threads = 256;
            int blocks = (db_elems + threads - 1) / threads;
            word_decompose_u16_bytes<<<blocks, threads>>>(ctx->d_db_bytes[0], ctx->d_db_bytes[1], d_db_raw, db_elems);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaFree(d_db_raw));

        // Pre-allocate query byte buffers and accumulators (full max_batch)
        size_t q_elems = max_N * db_rows_padded;
        for (int i = 0; i < 8; i++)
            CUDA_ASSERT(cudaMalloc(&ctx->d_query_bytes[i], q_elems));

        for (int i = 0; i < 2; i++)
            CUDA_ASSERT(cudaMalloc(&ctx->d_partials[i], max_N * M * sizeof(int32_t)));

        CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(ctx->cublas_handle, CUBLAS_TENSOR_OP_MATH));

    } else {
        // SIMT: keep DB as u16
        size_t db_size = M * db_rows_padded * sizeof(uint16_t);
        CUDA_ALLOC_AND_COPY(ctx->d_db_u16, db, db_size);

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
        // Upload queries and decompose to bytes
        size_t q_elems = N * K;
        uint64_t* d_query_raw;
        CUDA_ALLOC_AND_COPY(d_query_raw, queries, q_elems * sizeof(uint64_t));
        {
            int threads = 256;
            int blocks = (q_elems + threads - 1) / threads;
            word_decompose_u64_bytes<<<blocks, threads>>>(
                ctx->d_query_bytes[0], ctx->d_query_bytes[1],
                ctx->d_query_bytes[2], ctx->d_query_bytes[3],
                ctx->d_query_bytes[4], ctx->d_query_bytes[5],
                ctx->d_query_bytes[6], ctx->d_query_bytes[7],
                d_query_raw, q_elems);
            CUDA_ASSERT(cudaGetLastError());
        }
        CUDA_ASSERT(cudaFree(d_query_raw));

        // 15 cuBLAS GEMMs
        for (int g = 0; g < 15; g++) {
            const WordGemmSpec& s = WORD_ONLINE_SPECS[g];
            CUBLAS_CHECK(cublasGemmEx(ctx->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                (int)M, (int)N, (int)K,
                &s.alpha,
                ctx->d_db_bytes[s.db_b],    CUDA_R_8I, (int)ctx->db_rows_padded,
                ctx->d_query_bytes[s.q_b],  CUDA_R_8I, (int)K,
                &s.beta,
                ctx->d_partials[s.out],     CUDA_R_32I, (int)M,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

        // Combine + modswitch + multiply by inv_N
        {
            size_t count = M * N;
            int threads = 256;
            int blocks = (count + threads - 1) / threads;
            word_combine_modswitch_crt<<<blocks, threads>>>(
                ctx->d_intermediate, ctx->d_partials[0], ctx->d_partials[1],
                count, ctx->modulus, ctx->mod0, ctx->mod1, ctx->inv_n);
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

void ypir_word_online_free(void* context)
{
    WordOnlineContext* ctx = (WordOnlineContext*)context;
    if (!ctx) return;

    for (int i = 0; i < 2; i++) {
        if (ctx->d_db_bytes[i]) cudaFree(ctx->d_db_bytes[i]);
        if (ctx->d_partials[i]) cudaFree(ctx->d_partials[i]);
    }
    for (int i = 0; i < 8; i++) {
        if (ctx->d_query_bytes[i]) cudaFree(ctx->d_query_bytes[i]);
    }
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

    // Free per-stream scratch and streams
    if (ctx->d_scratch_batch) {
        for (size_t i = 0; i < ctx->num_streams; i++) {
            if (ctx->d_scratch_batch[i]) cudaFree(ctx->d_scratch_batch[i]);
        }
        delete[] ctx->d_scratch_batch;
    }
    if (ctx->streams) {
        for (size_t i = 0; i < ctx->num_streams; i++) {
            cudaStreamDestroy(ctx->streams[i]);
        }
        delete[] ctx->streams;
    }

    if (ctx->cublas_handle) cublasDestroy(ctx->cublas_handle);

    cudaFree(ctx->ntt_params.moduli);
    cudaFree(ctx->ntt_params.barrett_cr);
    cudaFree(ctx->ntt_params.forward_table);
    cudaFree(ctx->ntt_params.forward_prime_table);
    cudaFree(ctx->ntt_params.inverse_table);
    cudaFree(ctx->ntt_params.inverse_prime_table);

    delete ctx;
}

} // extern "C"
