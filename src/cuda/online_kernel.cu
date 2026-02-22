#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "ntt.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status_); \
    } \
} while(0)

static constexpr int COMPRESSION = 4;

typedef uint32_t Elem;

// CUTLASS GEMM: uint8 DB × uint32 query → uint32 output (mod 2^32 accumulation)
using CutlassGemm = cutlass::gemm::device::Gemm<
    uint8_t,                                // ElementA (DB, unpacked)
    cutlass::layout::RowMajor,              // LayoutA
    uint32_t,                               // ElementB (Query)
    cutlass::layout::ColumnMajor,           // LayoutB
    uint32_t,                               // ElementC (Output)
    cutlass::layout::ColumnMajor,           // LayoutC
    uint32_t,                               // ElementAccumulator
    cutlass::arch::OpClassSimt,             // OperatorClass
    cutlass::arch::Sm50,                    // ArchTag
    cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape
    cutlass::gemm::GemmShape<32, 32, 8>,    // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape
    cutlass::epilogue::thread::LinearCombination<uint32_t, 1, uint32_t, uint32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2                                        // Stages
>;

// GPU context for online computation
struct OnlineContext {
    uint8_t* d_db = nullptr;       // Device database (uint8 elements, row-major)
    Elem* d_query = nullptr;       // Device query buffer
    Elem* d_result = nullptr;      // Device result buffer (Step 1 output / a1)
    size_t max_batch_size = 0;     // Maximum supported batch size

    // Phase 2 & 3 data
    uint16_t* d_smaller_db = nullptr; // Expanded smaller DB
    uint64_t* d_query_ntt = nullptr;  // Secondary query in NTT form
    uint64_t* d_hint_acc = nullptr;   // Accumulator for secondary hint (NTT domain)
    uint64_t* d_hint = nullptr;       // Final secondary hint

    // Phase 4 data
    uint64_t* d_query_q2 = nullptr;   // Query for response generation
    uint64_t* d_response = nullptr;   // Final response

    // Phase 5 data
    uint64_t* d_pack_pub_params_full = nullptr; 
    uint64_t* d_rotation_ntt_table = nullptr;

    // Parameters
    NTTParams ntt_params;
    size_t pt_bits = 0;
    uint64_t lwe_modulus;
    uint64_t lwe_q_prime;
    uint64_t rlwe_q_prime_1;
    uint64_t rlwe_q_prime_2;
    size_t special_offs;
    size_t t_exp_left;
    size_t blowup_factor_ceil;

    size_t db_rows = 0;
    size_t db_cols = 0;
    
    size_t smaller_db_rows = 0;
    size_t smaller_db_cols = 0;

    // Packing Data
    uint64_t* d_y_constants = nullptr;
    uint64_t* d_prepacked_lwe = nullptr;
    uint64_t* d_precomp_res = nullptr;
    uint64_t* d_precomp_vals = nullptr;
    uint64_t* d_precomp_tables = nullptr;
    uint64_t* d_fake_pack_pub_params = nullptr;

    // Per-stream parallel execution resources
    size_t num_streams = 0;
    cudaStream_t* streams = nullptr;
    uint16_t** d_modified_rows = nullptr;  // per-stream overlay (blowup rows only)
    uint64_t** d_hint_batch = nullptr;
    uint64_t** d_hint_acc_batch = nullptr;
    uint64_t** d_response_batch = nullptr;
    uint64_t** d_pack_pub_params_full_batch = nullptr;
    uint64_t** d_packed_excess_batch = nullptr;
    uint64_t** d_pack_excess_scratch_batch = nullptr;
    uint64_t** d_pack_lwe_scratch_batch = nullptr;

    // Pre-allocated batch buffers (avoid per-call cudaMalloc)
    uint64_t* d_query_q2_all = nullptr;
    uint64_t* d_pub_params_row_1s_all = nullptr;
    uint8_t* d_all_responses = nullptr;
    size_t response_bytes_per_item = 0;
    size_t pub_params_row_1s_elems = 0;

    // GEMM workspace (pre-allocated, CUTLASS SIMT fallback only)
    void* d_gemm_workspace = nullptr;

    // Tensor core resources (SM >= 72)
    bool has_tensor_cores = false;
    cublasHandle_t cublas_handle = nullptr;
    uint8_t* d_query_bytes[4] = {nullptr, nullptr, nullptr, nullptr};

    // Async transfer resources
    cudaStream_t transfer_stream = nullptr;
    Elem* h_query_pinned = nullptr;
    uint64_t* h_query_q2_pinned = nullptr;
    uint64_t* h_pub_params_pinned = nullptr;
};


// Decompose uint32 query values into 4 uint8 byte slices for tensor core GEMM
__global__ void decompose_query_bytes_kernel(
    uint8_t* __restrict__ q0, uint8_t* __restrict__ q1,
    uint8_t* __restrict__ q2, uint8_t* __restrict__ q3,
    const uint32_t* __restrict__ query, size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t v = query[idx];
        q0[idx] = (uint8_t)(v & 0xFF);
        q1[idx] = (uint8_t)((v >> 8) & 0xFF);
        q2[idx] = (uint8_t)((v >> 16) & 0xFF);
        q3[idx] = (uint8_t)((v >> 24) & 0xFF);
    }
}

// Kernel to rescale intermediate results into per-batch overlay buffer
__global__ void rescale_and_expand_kernel(
    const Elem* __restrict__ intermediate,
    uint16_t* __restrict__ modified_rows,  // output: blowup_factor_ceil × db_cols
    size_t db_cols,
    uint64_t lwe_modulus,
    uint64_t lwe_q_prime,
    int pt_bits,
    size_t blowup_factor_ceil
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= db_cols) return;

    // Rescale
    uint64_t val = (uint64_t)intermediate[idx];

    double d_val = (double)val;
    double d_mod = (double)lwe_modulus;
    double d_qp = (double)lwe_q_prime;

    uint64_t rescaled = (uint64_t)((d_val * d_qp) / d_mod + 0.5);

    for (size_t m = 0; m < blowup_factor_ceil; m++) {
        size_t out_idx = m * db_cols + idx;

        uint16_t val_part = (rescaled >> (m * pt_bits)) & ((1 << pt_bits) - 1);
        modified_rows[out_idx] = val_part;
    }
}

// Secondary hint kernel matching multiply_with_db_ring
// One block per modified row (m = 0..blowup_factor_ceil-1)
__global__ void compute_secondary_hint_kernel(
    uint64_t* __restrict__ hint_out,
    const uint16_t* __restrict__ modified_rows,  // overlay: blowup × db_cols
    const uint64_t* __restrict__ query_ntt,
    uint64_t* __restrict__ sum_global,  // Global memory accumulator
    NTTParams params,
    size_t db_rows_poly,
    size_t blowup_factor_ceil
)
{
    size_t m = blockIdx.x;
    if (m >= blowup_factor_ceil) return;

    size_t tid = threadIdx.x;

    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t convd_len = crt_count * poly_len;

    // Shared memory: only workspace for db_elem during NTT
    // Allocate only 1 * convd_len = 32KB to fit in 48KB limit
    extern __shared__ uint64_t workspace[];

    // Global memory sum for this block
    uint64_t* sum = sum_global + m * convd_len;

    // Thread partitioning for NTT parallelization
    uint32_t threads_per_crt = blockDim.x / crt_count;
    uint32_t my_crt = tid / threads_per_crt;
    uint32_t local_tid = tid % threads_per_crt;

    // Initialize sum to zero
    for (size_t i = tid; i < convd_len; i += blockDim.x) {
        sum[i] = 0;
    }
    __syncthreads();

    // For each row (polynomial chunk)
    for (size_t row = 0; row < db_rows_poly; row++) {
        // Load DB element from overlay buffer
        // Overlay layout: modified_rows[m * db_cols + row * poly_len + z]
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            size_t db_idx = m * (db_rows_poly * poly_len) + row * poly_len + z;
            workspace[z] = (uint64_t)modified_rows[db_idx];
        }
        __syncthreads();

        // Replicate across CRT moduli
        for (size_t crt = 1; crt < crt_count; crt++) {
            for (size_t i = tid; i < poly_len; i += blockDim.x) {
                workspace[crt * poly_len + i] = workspace[i] % params.moduli[crt-1];
            }
        }
        __syncthreads();

        // Forward NTT on workspace (db_elem)
        if (my_crt < crt_count) {
            ntt_forward_kernel_parallel(workspace + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
        }
        __syncthreads();

        // Pointwise multiply with query[row] and accumulate into sum directly
        // query layout: [row][crt][poly_len]
        const uint64_t* query_row = &query_ntt[row * convd_len];
        for (size_t i = tid; i < convd_len; i += blockDim.x) {
            uint32_t crt = i / poly_len;
            uint64_t modulus = params.moduli[crt];
            uint64_t barrett_cr = params.barrett_cr[crt];

            uint64_t a = query_row[i];
            uint64_t b = workspace[i];  // db_elem_ntt result
            uint64_t p = a * b;
            uint64_t reduced = barrett_raw_u64(p, barrett_cr, modulus);

            // Accumulate directly into sum
            sum[i] += reduced;
        }
        __syncthreads();
    }

    // Barrett reduce sum
    for (size_t i = tid; i < convd_len; i += blockDim.x) {
        uint32_t crt = i / poly_len;
        uint64_t modulus = params.moduli[crt];
        uint64_t barrett_cr = params.barrett_cr[crt];
        sum[i] = barrett_raw_u64(sum[i], barrett_cr, modulus);
    }
    __syncthreads();

    // Inverse NTT
    if (my_crt < crt_count) {
        ntt_inverse_kernel_parallel(sum + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    }
    __syncthreads();

    // CRT reconstruction and write to output
    // Output layout: [poly_len][blowup_factor_ceil] (column-major-ish from Rust perspective)
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x = sum[z];
        uint64_t y = sum[poly_len + z];
        uint64_t composed = crt_compose_2(x, y, &params);

        // Write to hint_out[z * out_rows + col] (matching Rust's expectation)
        // But Rust expects: hint_1_combined[inp_idx] where inp_idx = i * blowup_factor_ceil + j
        // And stores at: out_idx = i * out_rows + special_offs + j
        // So we write at: z * blowup_factor_ceil + m
        hint_out[z * blowup_factor_ceil + m] = composed;
    }
}


// Kernel for Response Generation matching fast_batched_dot_product_avx512
// DB is column-major: smaller_db[col * out_rows + row]
// Query is packed u64 values
// helper: warp reduce sum (64-bit)
__inline__ __device__ uint64_t warp_reduce_sum_u64(uint64_t v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void compute_response_kernel(
    uint64_t* __restrict__ response_out,
    const uint16_t* __restrict__ smaller_db,   // row-major [out_rows][db_cols]
    const uint64_t* __restrict__ query_packed, // [db_cols]
    size_t db_cols,
    size_t out_rows,
    const uint16_t* __restrict__ modified_rows, // overlay: blowup rows
    size_t overlay_start,
    size_t overlay_count,
    NTTParams params
) {
    const uint32_t row = blockIdx.x;
    if (row >= out_rows) return;

    const uint32_t lane = threadIdx.x; // 0..255
    const uint32_t warp = lane >> 5;   // 0..7
    const uint32_t wlane = lane & 31;

    // Each thread handles a strided chunk of columns
    // (lane, lane+256, lane+512, ...)
    uint64_t sum_lo = 0;
    uint64_t sum_hi = 0;

    // For overlay rows, read from the per-batch modified_rows buffer
    const uint16_t* db_row;
    if (row >= overlay_start && row < overlay_start + overlay_count) {
        db_row = modified_rows + (size_t)(row - overlay_start) * db_cols;
    } else {
        db_row = smaller_db + (size_t)row * db_cols;
    }

    // Unroll-friendly: db_cols is ~65536, stride=256 => 256 iters
    for (size_t col = lane; col < db_cols; col += blockDim.x) {
        uint64_t q = query_packed[col];
        uint64_t db = (uint64_t)db_row[col];

        uint64_t q_lo = (uint32_t)q;
        uint64_t q_hi = (uint32_t)(q >> 32);

        // Now safe: each thread sums only ~256 terms.
        sum_lo += q_lo * db;
        sum_hi += q_hi * db;
    }

    // Reduce modulo per CRT **before** combining across 256 threads.
    // This keeps totals small and avoids overflow.
    uint64_t lo_part = barrett_raw_u64(sum_lo, params.barrett_cr[0], params.moduli[0]);
    uint64_t hi_part = barrett_raw_u64(sum_hi, params.barrett_cr[1], params.moduli[1]);

    // Warp-reduce within each warp (8 warps total)
    lo_part = warp_reduce_sum_u64(lo_part);
    hi_part = warp_reduce_sum_u64(hi_part);

    // Shared sums per warp
    __shared__ uint64_t warp_lo[8];
    __shared__ uint64_t warp_hi[8];

    if (wlane == 0) {
        warp_lo[warp] = lo_part;
        warp_hi[warp] = hi_part;
    }
    __syncthreads();

    // Final reduce by warp 0
    if (warp == 0) {
        uint64_t lo = (lane < 8) ? warp_lo[wlane] : 0;
        uint64_t hi = (lane < 8) ? warp_hi[wlane] : 0;

        lo = warp_reduce_sum_u64(lo);
        hi = warp_reduce_sum_u64(hi);

        if (lane == 0) {
            // Bring back into [0, mod)
            lo = barrett_raw_u64(lo, params.barrett_cr[0], params.moduli[0]);
            hi = barrett_raw_u64(hi, params.barrett_cr[1], params.moduli[1]);

            uint64_t composed = crt_compose_2(lo, hi, &params);
            uint64_t result = barrett_raw_u64(composed, params.barrett_cr_1_modulus, params.modulus);
            response_out[row] = result;
        }
    }
}

__global__ void precompute_rotation_ntt_table(
    uint64_t* __restrict__ d_rotation_ntt_table, // OUT: blowup * convd_len
    size_t special_offs,
    size_t blowup_factor_ceil,
    NTTParams params
) {
    const size_t tid = threadIdx.x;
    const size_t j = blockIdx.x;

    const size_t poly_len  = params.poly_len;
    const size_t crt_count = params.crt_count;
    const size_t convd_len = crt_count * poly_len;

    if (j >= blowup_factor_ceil) return;

    // partition threads among CRTs for your parallel NTT kernels
    const size_t threads_per_crt = blockDim.x / crt_count;
    const size_t my_crt = tid / threads_per_crt;
    const size_t local_tid = tid % threads_per_crt;

    uint64_t* rot = d_rotation_ntt_table + j * convd_len;
    size_t rot_idx = special_offs + j; // must be < poly_len

    // write basis vector split into CRT lanes
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t raw = (z == rot_idx) ? 1ULL : 0ULL;
        rot[z]            = raw % params.moduli[0];
        rot[poly_len + z] = raw % params.moduli[1];
    }
    __syncthreads();

    if (my_crt < crt_count) {
        ntt_forward_kernel_parallel(rot + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    }
    __syncthreads();
}


__global__ void build_pub_params_full(
    uint64_t* __restrict__ d_pub_params_full,         // OUT: log2_poly_len * (2*t_exp_left*convd_len)
    const uint64_t* __restrict__ d_fake_pub_params,   // IN: same layout as before (row0 already full)
    const uint64_t* __restrict__ d_pub_params_row_1s, // IN: condensed row1s: log2_poly_len*(t_exp_left*poly_len)
    size_t t_exp_left,
    NTTParams params
) {
    const size_t tid = threadIdx.x;
    const size_t key_idx = blockIdx.x;

    const size_t poly_len  = params.poly_len;
    const size_t convd_len = params.crt_count * poly_len;
    const size_t log2_poly_len = params.log2_poly_len;

    if (key_idx >= log2_poly_len) return;

    const size_t pub_param_size_u64 = 2 * t_exp_left * convd_len;
    const size_t row0_size_u64      = t_exp_left * convd_len;
    const size_t row1s_size_u64     = t_exp_left * poly_len;

    const uint64_t* src_key = d_fake_pub_params + key_idx * pub_param_size_u64;
    const uint64_t* src_row1_cond = d_pub_params_row_1s + key_idx * row1s_size_u64;

    uint64_t* dst_key  = d_pub_params_full + key_idx * pub_param_size_u64;
    uint64_t* dst_row0 = dst_key;
    uint64_t* dst_row1 = dst_key + row0_size_u64;

    // copy row0
    for (size_t i = tid; i < row0_size_u64; i += blockDim.x) {
        dst_row0[i] = src_key[i];
    }

    // unpack condensed row1s into CRT lanes
    const size_t total = t_exp_left * poly_len;
    for (size_t idx = tid; idx < total; idx += blockDim.x) {
        const size_t col = idx / poly_len;
        const size_t z   = idx % poly_len;

        uint64_t packed = src_row1_cond[col * poly_len + z];
        uint64_t lo = packed & 0xFFFFFFFFULL;
        uint64_t hi = packed >> 32;

        dst_row1[col * convd_len + z]            = lo; // crt0 lane
        dst_row1[col * convd_len + poly_len + z] = hi; // crt1 lane
    }
}

__global__ void __launch_bounds__(1024, 1) pack_excess(
    uint64_t* d_packed_out,
    const uint64_t* d_hint,
    const uint64_t* d_pub_params_full,     // IN: log2_poly_len * pub_param_size_u64
    const uint64_t* d_rotation_ntt_table,  // IN: blowup * convd_len
    uint64_t* d_scratch,
    size_t special_offs,
    size_t blowup_factor_ceil,
    size_t t_exp_left,
    NTTParams params
) {
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    size_t crt_count = params.crt_count;
    size_t convd_len = crt_count * poly_len;
    size_t log2_poly_len = params.log2_poly_len;
    uint64_t modulus = params.modulus;

    size_t modulus_log2 = 64 - __clzll(modulus);
    size_t bits_per = modulus_log2 / t_exp_left + 1;
    uint64_t mask = (bits_per >= 64) ? ~0ULL : ((1ULL << bits_per) - 1);

    size_t pub_param_size = 2 * t_exp_left * convd_len;

    // Scratch layout (reduced)
    uint64_t* cur_r        = d_scratch;                      // 2*convd_len
    uint64_t* ct_raw       = cur_r + 2 * convd_len;          // 2*poly_len
    uint64_t* ct_auto      = ct_raw + 2 * poly_len;          // 2*poly_len
    uint64_t* ginv_ct      = ct_auto + 2 * poly_len;         // t_exp_left*poly_len
    uint64_t* ginv_ct_ntt  = ginv_ct + t_exp_left * poly_len;// t_exp_left*convd_len
    uint64_t* tau_of_r     = ginv_ct_ntt + t_exp_left * convd_len; // 2*convd_len
    uint64_t* temp_ntt     = tau_of_r + 2 * convd_len;       // convd_len

    // Thread partitioning for NTT
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;

    // zero output
    for (size_t i = tid; i < 2 * convd_len; i += blockDim.x) d_packed_out[i] = 0;
    __syncthreads();

    for (size_t j = 0; j < blowup_factor_ceil; j++) {
        // 1) Load hint into cur_r row0 (negacyclic_perm shift=0), row1=0
        for (size_t k = tid; k < poly_len; k += blockDim.x) {
            uint64_t val;
            if (k == 0) val = d_hint[0 * blowup_factor_ceil + j];
            else {
                uint64_t a_val = d_hint[(poly_len - k) * blowup_factor_ceil + j];
                val = modulus - (a_val % modulus);
                if (val == modulus) val = 0;
            }
            cur_r[k] = val % params.moduli[0];
            cur_r[poly_len + k] = val % params.moduli[1];
            cur_r[convd_len + k] = 0;
            cur_r[convd_len + poly_len + k] = 0;
        }
        __syncthreads();

        // 2) Forward NTT on row0 (row1 is zero)
        if (my_crt < crt_count) {
            ntt_forward_kernel_parallel(cur_r + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
        }
        __syncthreads();

        // 3) pack_single_lwe loop
        for (size_t iter = 0; iter < log2_poly_len; iter++) {
            size_t t = (poly_len / (1 << iter)) + 1;
            const uint64_t* pub_param = d_pub_params_full + iter * pub_param_size;

            // 3a) inverse NTT cur_r -> ct_raw (both rows)
            for (size_t row = 0; row < 2; row++) {
                for (size_t z = tid; z < convd_len; z += blockDim.x) {
                    uint32_t crt = z / poly_len;
                    temp_ntt[z] = barrett_raw_u64(cur_r[row * convd_len + z],
                                                  params.barrett_cr[crt],
                                                  params.moduli[crt]);
                }
                __syncthreads();

                if (my_crt < crt_count) {
                    ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
                }
                __syncthreads();

                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    ct_raw[row * poly_len + z] = crt_compose_2(temp_ntt[z], temp_ntt[poly_len + z], &params);
                }
                __syncthreads();
            }

            // 3b) automorphism ct_raw -> ct_auto
            for (size_t row = 0; row < 2; row++) {
                for (size_t i2 = tid; i2 < poly_len; i2 += blockDim.x) {
                    size_t num = (i2 * t) / poly_len;
                    size_t rem = (i2 * t) % poly_len;
                    uint64_t v = ct_raw[row * poly_len + i2];
                    uint64_t out = (num & 1) ? (modulus - v) : v;
                    if (out == modulus) out = 0;
                    ct_auto[row * poly_len + rem] = out;
                }
            }
            __syncthreads();

            // 3c) gadget invert row0
            for (size_t z = tid; z < poly_len; z += blockDim.x) {
                uint64_t val = ct_auto[z];
                #pragma unroll
                for (size_t k = 0; k < 3; k++) { // you said t_exp_left=3
                    size_t bit_offs = k * bits_per;
                    uint64_t piece = (bit_offs >= 64) ? 0 : ((val >> bit_offs) & mask);
                    ginv_ct[k * poly_len + z] = piece;
                }
            }
            __syncthreads();

            // 3d) NTT ginv_ct into ginv_ct_ntt (k=0..t_exp_left-1)
            // (your old code zeroed and skipped k=0 sometimes; keep consistent with your pub params layout)
            for (size_t k = 0; k < t_exp_left; k++) {
                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    uint64_t v = ginv_ct[k * poly_len + z];
                    ginv_ct_ntt[k * convd_len + z]            = v % params.moduli[0];
                    ginv_ct_ntt[k * convd_len + poly_len + z] = v % params.moduli[1];
                }
                __syncthreads();
                if (my_crt < crt_count) {
                    ntt_forward_kernel_parallel(ginv_ct_ntt + k * convd_len + my_crt * poly_len,
                                               &params, my_crt, local_tid, threads_per_crt);
                }
                __syncthreads();
            }

            // 3e) tau_of_r = pub_param * ginv_ct_ntt (2 rows)
            for (size_t row = 0; row < 2; row++) {
                for (size_t z = tid; z < convd_len; z += blockDim.x) {
                    size_t crt_idx = z / poly_len;
                    uint64_t mod = params.moduli[crt_idx];
                    uint64_t bar = params.barrett_cr[crt_idx];

                    uint64_t acc = 0;
                    #pragma unroll
                    for (size_t k = 0; k < 3; k++) { // t_exp_left=3
                        uint64_t a = pub_param[(row * t_exp_left + k) * convd_len + z];
                        uint64_t b = ginv_ct_ntt[k * convd_len + z];
                        acc += barrett_raw_u64(a * b, bar, mod);
                    }
                    tau_of_r[row * convd_len + z] = barrett_raw_u64(acc, bar, mod);
                }
            }
            __syncthreads();

            // 3f) NTT(ct_auto row1) and add into tau_of_r row1
            for (size_t z = tid; z < poly_len; z += blockDim.x) {
                uint64_t v = ct_auto[poly_len + z];
                temp_ntt[z]            = v % params.moduli[0];
                temp_ntt[poly_len + z] = v % params.moduli[1];
            }
            __syncthreads();
            if (my_crt < crt_count) {
                ntt_forward_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
            }
            __syncthreads();
            for (size_t z = tid; z < convd_len; z += blockDim.x) {
                size_t crt_idx = z / poly_len;
                uint64_t mod = params.moduli[crt_idx];
                tau_of_r[convd_len + z] = (tau_of_r[convd_len + z] + temp_ntt[z]) % mod;
            }
            __syncthreads();

            // 3g) cur_r += tau_of_r
            for (size_t ii = tid; ii < 2 * convd_len; ii += blockDim.x) {
                size_t crt_idx = (ii % convd_len) / poly_len;
                uint64_t mod = params.moduli[crt_idx];
                cur_r[ii] = (cur_r[ii] + tau_of_r[ii]) % mod;
            }
            __syncthreads();
        }

        // 4/5) multiply by precomputed rotation_ntt and accumulate into d_packed_out
        const uint64_t* rot = d_rotation_ntt_table + j * convd_len;

        for (size_t row = 0; row < 2; row++) {
            for (size_t z = tid; z < convd_len; z += blockDim.x) {
                size_t crt_idx = z / poly_len;
                uint64_t mod = params.moduli[crt_idx];
                uint64_t bar = params.barrett_cr[crt_idx];

                uint64_t prod = barrett_raw_u64(cur_r[row * convd_len + z] * rot[z], bar, mod);
                d_packed_out[row * convd_len + z] = (d_packed_out[row * convd_len + z] + prod) % mod;
            }
        }
        __syncthreads();
    }
}



// Helper: write `num_bits` of `val` starting at `bit_offs` into `out`
__device__ __forceinline__
void write_arbitrary_bits(uint8_t* out, uint64_t val, size_t bit_offs, size_t num_bits) {
    for (size_t i = 0; i < num_bits; i++) {
        size_t abs_bit = bit_offs + i;
        size_t byte_idx = abs_bit >> 3;          // /8
        size_t bit_in_byte = abs_bit & 7;        // %8

        if ((val >> i) & 1ULL) {
            size_t word_byte = byte_idx & ~3ULL;                // floor(byte_idx/4)*4
            uint32_t* wordp = (uint32_t*)(out + word_byte);      // aligned
            uint32_t bitpos = (uint32_t)((byte_idx - word_byte) * 8 + bit_in_byte);
            atomicOr(wordp, 1u << bitpos);
        }
    }
}


// ceil(log2(x)) for x>=1; returns 0 for x<=1.
// Identity used: ceil(log2(x)) == bitlen(x-1)
__host__ __device__ __forceinline__
uint32_t ceil_log2_u64(uint64_t x) {
    if (x <= 1) return 0;
    uint64_t y = x - 1;

#if defined(__CUDA_ARCH__)
    // device
    return 64u - (uint32_t)__clzll(y);
#else
    // host (GCC/Clang): __builtin_clzll is undefined for 0, but y>=1 here
    return 64u - (uint32_t)__builtin_clzll(y);
#endif
}

__global__ void __launch_bounds__(1024, 1) pack_lwes_and_mod_switch(
    uint8_t* d_response_out,             // output: final response bytes
    const uint64_t* d_packed_excess,     // from pack_excess kernel: 2 * crt_count * poly_len
    const uint64_t* d_response,          // b_values from step 4: poly_len u64s
    const uint64_t* d_y_constants,       // [y_0..y_{ell-1}, neg_y_0..neg_y_{ell-1}], each crt_count*poly_len
    const uint64_t* d_precomp_res,       // 2 * crt_count * poly_len
    const uint64_t* d_precomp_vals,      // (2^ell - 1) * t_exp_left * poly_len (condensed)
    const uint64_t* d_precomp_tables,    // ell * poly_len
    const uint64_t* d_pub_params_row_1s, // ell * t_exp_left * poly_len (condensed)
    uint64_t* d_scratch,
    size_t t_exp_left,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    NTTParams params
)
{
    size_t tid = threadIdx.x;
    size_t poly_len = params.poly_len;
    size_t crt_count = params.crt_count;
    size_t convd_len = crt_count * poly_len;
    size_t ell = params.log2_poly_len;
    uint64_t modulus = params.modulus;
    
    // Thread partitioning for NTT
    size_t threads_per_crt = blockDim.x / crt_count;
    size_t my_crt = tid / threads_per_crt;
    size_t local_tid = tid % threads_per_crt;
    
    // Scratch layout
    size_t working_set_size = (1 << (ell - 1));
    uint64_t* working_set = d_scratch;
    uint64_t* y_times_ct_odd = working_set + working_set_size * poly_len;
    uint64_t* neg_y_times_ct_odd = y_times_ct_odd + poly_len;
    uint64_t* ct_sum_1 = neg_y_times_ct_odd + poly_len;
    uint64_t* w_times_ginv_ct = ct_sum_1 + poly_len;
    uint64_t* result_ntt = w_times_ginv_ct + poly_len;
    uint64_t* temp_raw = result_ntt + 2 * convd_len;
    uint64_t* temp_ntt = temp_raw + 2 * poly_len;
    
    // Initialize working_set to zero
    for (size_t i = tid; i < working_set_size * poly_len; i += blockDim.x) {
        working_set[i] = 0;
    }
    __syncthreads();
    
    size_t idx_precomp = 0;
    
    // ==================== FFT-style packing ====================
    for (size_t cur_ell = 1; cur_ell <= ell; cur_ell++) {
        size_t num_in = 1 << (ell - cur_ell + 1);
        size_t num_out = num_in >> 1;
        if (num_in == poly_len) {
            num_in = num_out;
        }
        
        const uint64_t* y = d_y_constants + (cur_ell - 1) * convd_len;
        const uint64_t* neg_y = d_y_constants + ell * convd_len + (cur_ell - 1) * convd_len;
        
        for (size_t i = 0; i < num_out; i++) {
            uint64_t* ct_even = working_set + i * poly_len;

            
            if (cur_ell > 1) {
                uint64_t* ct_odd = working_set + (num_out + i) * poly_len;
                
                // scalar_multiply (condensed): y_times_ct_odd = y * ct_odd
                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    uint64_t ct_val = ct_odd[z];
                    uint64_t ct_lo = ct_val & 0xFFFFFFFF;
                    uint64_t ct_hi = ct_val >> 32;
                    
                    uint64_t y_lo = y[z];
                    uint64_t y_hi = y[poly_len + z];
                    uint64_t neg_y_lo = neg_y[z];
                    uint64_t neg_y_hi = neg_y[poly_len + z];
                    
                    uint64_t prod_lo = barrett_raw_u64(ct_lo * y_lo, params.barrett_cr[0], params.moduli[0]);
                    uint64_t prod_hi = barrett_raw_u64(ct_hi * y_hi, params.barrett_cr[1], params.moduli[1]);
                    y_times_ct_odd[z] = (prod_lo & 0xFFFFFFFF) | ((prod_hi & 0xFFFFFFFF) << 32);
                    
                    prod_lo = barrett_raw_u64(ct_lo * neg_y_lo, params.barrett_cr[0], params.moduli[0]);
                    prod_hi = barrett_raw_u64(ct_hi * neg_y_hi, params.barrett_cr[1], params.moduli[1]);
                    neg_y_times_ct_odd[z] = (prod_lo & 0xFFFFFFFF) | ((prod_hi & 0xFFFFFFFF) << 32);
                }
                __syncthreads();
                
                // ct_sum_1 = ct_even + neg_y_times_ct_odd
                // ct_even = ct_even + y_times_ct_odd
                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    ct_sum_1[z] = ct_even[z] + neg_y_times_ct_odd[z];
                    ct_even[z] = ct_even[z] + y_times_ct_odd[z];
                }
                __syncthreads();
            }
            
            // Matrix multiply: w_times_ginv_ct = pub_param * precomp_vals[idx_precomp]
            size_t pub_param_idx = ell - cur_ell;
            const uint64_t* w = d_pub_params_row_1s + pub_param_idx * t_exp_left * poly_len;
            const uint64_t* ginv_ct = d_precomp_vals + idx_precomp * t_exp_left * poly_len;
            idx_precomp++;
            
            for (size_t z = tid; z < poly_len; z += blockDim.x) {
                uint64_t sum_lo = 0;
                uint64_t sum_hi = 0;
                
                for (size_t k = 0; k < t_exp_left; k++) {
                    uint64_t w_val = w[k * poly_len + z];
                    uint64_t g_val = ginv_ct[k * poly_len + z];
                    
                    uint64_t w_lo = w_val & 0xFFFFFFFF;
                    uint64_t w_hi = w_val >> 32;
                    uint64_t g_lo = g_val & 0xFFFFFFFF;
                    uint64_t g_hi = g_val >> 32;
                    
                    sum_lo += w_lo * g_lo;
                    sum_hi += w_hi * g_hi;
                }
                
                uint64_t res_lo = barrett_raw_u64(sum_lo, params.barrett_cr[0], params.moduli[0]);
                uint64_t res_hi = barrett_raw_u64(sum_hi, params.barrett_cr[1], params.moduli[1]);
                w_times_ginv_ct[z] = (res_lo & 0xFFFFFFFF) | ((res_hi & 0xFFFFFFFF) << 32);
            }
            __syncthreads();
            
            if (cur_ell > 1) {
                // Apply automorphism: ct_even[z] += ct_sum_1[table[z]]
                size_t table_idx = ell - cur_ell;
                const uint64_t* table = d_precomp_tables + table_idx * poly_len;
                
                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    size_t src_idx = (size_t)table[z];
                    ct_even[z] += ct_sum_1[src_idx];
                }
                __syncthreads();
            }
            
            // Add w_times_ginv_ct to ct_even
            // bool do_reduce = !(i < num_out / 2 && ((cur_ell - 1) % 5 != 0));
            bool do_reduce = true;
            for (size_t z = tid; z < poly_len; z += blockDim.x) {
                ct_even[z] += w_times_ginv_ct[z];
                if (do_reduce) {
                    uint64_t val = ct_even[z];
                    uint64_t lo = barrett_raw_u64(val & 0xFFFFFFFF, params.barrett_cr[0], params.moduli[0]);
                    uint64_t hi = barrett_raw_u64(val >> 32, params.barrett_cr[1], params.moduli[1]);
                    ct_even[z] = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32);
                }
            }
            __syncthreads();
        }
    }
    
    // ==================== Final reduction on working_set[0] ====================
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = working_set[z];
        uint64_t lo = barrett_raw_u64(val & 0xFFFFFFFF, params.barrett_cr[0], params.moduli[0]);
        uint64_t hi = barrett_raw_u64(val >> 32, params.barrett_cr[1], params.moduli[1]);
        working_set[z] = (lo & 0xFFFFFFFF) | ((hi & 0xFFFFFFFF) << 32);
    }
    __syncthreads();
    
    // ==================== Build result: precomp_res with row 1 = working_set[0] ====================
    for (size_t z = tid; z < convd_len; z += blockDim.x) {
        result_ntt[z] = d_precomp_res[z];
    }
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = working_set[z];
        result_ntt[convd_len + z] = val & 0xFFFFFFFF;
        result_ntt[convd_len + poly_len + z] = val >> 32;
    }
    __syncthreads();
    
    // ==================== Add b_values in raw domain ====================
    // Inverse NTT row 1
    for (size_t z = tid; z < convd_len; z += blockDim.x) {
        uint32_t crt = z / poly_len;
        uint64_t modulus = params.moduli[crt];
        uint64_t barrett_cr = params.barrett_cr[crt];
        temp_ntt[z] = barrett_raw_u64(result_ntt[convd_len + z], barrett_cr, modulus);
    }
    __syncthreads();
    
    if (my_crt < crt_count) {
        ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    }
    __syncthreads();
    
    // CRT compose and add b_values * poly_len
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x = temp_ntt[z];
        uint64_t y_val = temp_ntt[poly_len + z];
        uint64_t composed = crt_compose_2(x, y_val, &params);
        
        // Add b_values[z] * poly_len
        __uint128_t prod = (__uint128_t)d_response[z] * poly_len;
        uint64_t b_scaled = (uint64_t)(prod % modulus);
        composed = (composed + b_scaled);
        composed -= (modulus * (composed > modulus));
        
        temp_raw[poly_len + z] = composed;
    }
    __syncthreads();
    
    // Forward NTT row 1 back (split to CRT first)
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        temp_ntt[z] = barrett_raw_u64(val, params.barrett_cr[0], params.moduli[0]);
        temp_ntt[poly_len + z] = barrett_raw_u64(val, params.barrett_cr[1], params.moduli[1]);
    }
    __syncthreads();
    
    if (my_crt < crt_count) {
        ntt_forward_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    }
    __syncthreads();
    
    for (size_t z = tid; z < convd_len; z += blockDim.x) {
        result_ntt[convd_len + z] = temp_ntt[z];
    }
    __syncthreads();
    
    // ==================== Add pack_excess ====================
    for (size_t z = tid; z < 2 * convd_len; z += blockDim.x) {
        size_t crt_idx = (z % convd_len) / poly_len;
        uint64_t mod = params.moduli[crt_idx];
        result_ntt[z] = (result_ntt[z] + d_packed_excess[z]) % mod;
    }
    __syncthreads();
    
    // ==================== Inverse NTT to raw for mod switch ====================
    for (size_t row = 0; row < 2; row++) {
        for (size_t z = tid; z < convd_len; z += blockDim.x) {
            temp_ntt[z] = result_ntt[row * convd_len + z];
        }
        __syncthreads();
        
        if (my_crt < crt_count) {
            ntt_inverse_kernel_parallel(temp_ntt + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
        }
        __syncthreads();
        
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            uint64_t x = temp_ntt[z];
            uint64_t y_val = temp_ntt[poly_len + z];
            temp_raw[row * poly_len + z] = crt_compose_2(x, y_val, &params);
        }
        __syncthreads();
    }

    // ==================== Mod switch and bit-pack ====================
    // Note: q_1_bits = ceil(log2(q_2)), q_2_bits = ceil(log2(q_1))
    size_t q_1_bits = ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = ceil_log2_u64(rlwe_q_prime_1);
    size_t total_sz_bits = (q_1_bits + q_2_bits) * poly_len;
    size_t total_sz_bytes = (total_sz_bits + 7) / 8;
    
    // Zero output
    for (size_t i = tid; i < total_sz_bytes; i += blockDim.x) {
        d_response_out[i] = 0;
    }
    __syncthreads();
    
    // Row 0: rescale to q_2, write with q_1_bits
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        
        size_t bit_offs = z * q_1_bits;
        write_arbitrary_bits(d_response_out, val_rescaled, bit_offs, q_1_bits);
    }
    __syncthreads();
    
    // Row 1: rescale to q_1, write with q_2_bits
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        
        size_t bit_offs = poly_len * q_1_bits + z * q_2_bits;
        write_arbitrary_bits(d_response_out, val_rescaled, bit_offs, q_2_bits);
    }
}

extern "C" {

// Initialize online computation context
void* ypir_online_init(const Elem* db, size_t db_rows, size_t db_cols,
                       const uint64_t* pseudorandom_query1,
                       const uint16_t* smaller_db, size_t smaller_db_rows,
                       size_t max_batch_size)
{
    OnlineContext* ctx = new OnlineContext();

    ctx->db_rows = db_rows;
    ctx->db_cols = db_cols;
    ctx->smaller_db_rows = smaller_db_rows;
    ctx->smaller_db_cols = db_rows;
    ctx->max_batch_size = max_batch_size;

    // DB is packed uint32 on host: db_rows × (db_cols/4) uint32 = db_rows × db_cols bytes
    // In memory this is identical to db_rows × db_cols uint8 (little-endian)
    const size_t db_bytes = db_rows * db_cols * sizeof(uint8_t);
    const size_t query_bytes = db_cols * max_batch_size * sizeof(Elem);
    const size_t result_bytes = db_rows * max_batch_size * sizeof(Elem);

    CUDA_ASSERT(cudaMalloc(&ctx->d_db, db_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_query, query_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_result, result_bytes));

    // Upload database (reinterpret packed uint32 as uint8)
    CUDA_ASSERT(cudaMemcpy(ctx->d_db, db, db_bytes, cudaMemcpyHostToDevice));

    // Upload pseudorandom query (*2 because CRT )
    size_t query_ntt_size = db_rows*2 * sizeof(uint64_t);
    CUDA_ALLOC_AND_COPY(ctx->d_query_ntt, pseudorandom_query1, query_ntt_size);

    // Allocate and upload smaller_db for Phase 2/3/4
    const size_t smaller_db_bytes = smaller_db_rows * ctx->smaller_db_cols * sizeof(uint16_t);
    CUDA_ALLOC_AND_COPY(ctx->d_smaller_db, smaller_db, smaller_db_bytes);

    // Host-side pointer arrays (GPU allocs deferred to init_ntt/init_packing)
    ctx->d_modified_rows = new uint16_t*[max_batch_size]();
    ctx->d_hint_batch = new uint64_t*[max_batch_size]();
    ctx->d_hint_acc_batch = new uint64_t*[max_batch_size]();
    ctx->d_response_batch = new uint64_t*[max_batch_size]();
    ctx->d_pack_pub_params_full_batch = new uint64_t*[max_batch_size]();
    ctx->d_packed_excess_batch = new uint64_t*[max_batch_size]();
    ctx->d_pack_excess_scratch_batch = new uint64_t*[max_batch_size]();
    ctx->d_pack_lwe_scratch_batch = new uint64_t*[max_batch_size]();

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    // Detect tensor cores (SM >= 72: Turing, Ampere, Hopper, ...)
    {
        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        int major, minor;
        CUDA_ASSERT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_ASSERT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 72);

        if (ctx->has_tensor_cores) {
            CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
            size_t byte_query_size = db_cols * max_batch_size * sizeof(uint8_t);
            for (int i = 0; i < 4; i++) {
                CUDA_ASSERT(cudaMalloc(&ctx->d_query_bytes[i], byte_query_size));
            }
            printf("GPU SM %d.%d: tensor cores enabled for Step 1 (4x cuBLAS int8 GEMM)\n", major, minor);
        } else {
            printf("GPU SM %d.%d: using CUTLASS SIMT for Step 1\n", major, minor);
        }
    }

    // Pre-allocate GEMM workspace for CUTLASS SIMT (fallback for non-tensor-core GPUs)
    if (!ctx->has_tensor_cores) {
        cutlass::gemm::GemmCoord problem_size(db_rows, max_batch_size, db_cols);
        uint32_t alpha = 1, beta = 0;
        CutlassGemm::Arguments args{
            problem_size,
            {ctx->d_db, (int)db_cols},
            {ctx->d_query, (int)db_cols},
            {ctx->d_result, (int)db_rows},
            {ctx->d_result, (int)db_rows},
            {alpha, beta}, 1
        };
        size_t ws = CutlassGemm::get_workspace_size(args);
        if (ws > 0) {
            CUDA_ASSERT(cudaMalloc(&ctx->d_gemm_workspace, ws));
        }
    }

    // Transfer stream for async data uploads
    CUDA_ASSERT(cudaStreamCreate(&ctx->transfer_stream));

    // Pinned host memory for query upload
    CUDA_ASSERT(cudaMallocHost(&ctx->h_query_pinned, db_cols * max_batch_size * sizeof(Elem)));

    return (void*)ctx;
}

// Initialize NTT parameters and allocate Phase 2/3/4 buffers
void ypir_online_init_ntt(
    void* context,
    uint32_t poly_len,
    uint32_t crt_count,
    size_t pt_bits,
    uint64_t lwe_modulus,
    uint64_t lwe_q_prime,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t special_offs,
    size_t t_exp_left,
    size_t blowup_factor_ceil,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    const uint64_t* forward_table,
    const uint64_t* forward_prime_table,
    const uint64_t* inverse_table,
    const uint64_t* inverse_prime_table,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus
) 
{
    OnlineContext* ctx = (OnlineContext*)context;
    if (!ctx) return;

    ctx->pt_bits = pt_bits;
    ctx->lwe_modulus = lwe_modulus;
    ctx->lwe_q_prime = lwe_q_prime;
    ctx->rlwe_q_prime_1 = rlwe_q_prime_1;
    ctx->rlwe_q_prime_2 = rlwe_q_prime_2;
    ctx->special_offs = special_offs;
    ctx->t_exp_left = t_exp_left;
    ctx->blowup_factor_ceil = blowup_factor_ceil;
    
    ctx->ntt_params.poly_len = poly_len;
    ctx->ntt_params.log2_poly_len = 31 - __builtin_clz(poly_len);
    ctx->ntt_params.crt_count = crt_count;
    ctx->ntt_params.mod0_inv_mod1 = mod0_inv_mod1;
    ctx->ntt_params.mod1_inv_mod0 = mod1_inv_mod0;
    ctx->ntt_params.barrett_cr_0_modulus = barrett_cr_0_modulus;
    ctx->ntt_params.barrett_cr_1_modulus = barrett_cr_1_modulus;
    ctx->ntt_params.modulus = moduli[0] * moduli[1]; // Assuming 2 CRT moduli
    
    // Allocate and copy NTT tables
    size_t table_size = crt_count * poly_len * sizeof(uint64_t);
    size_t moduli_size = crt_count * sizeof(uint64_t);
    
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.moduli, moduli, moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.barrett_cr, barrett_cr, moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_table, forward_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_prime_table, forward_prime_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_table, inverse_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_prime_table, inverse_prime_table, table_size);
    
    // More allocations (FPGA)
    size_t convd_len = ctx->ntt_params.crt_count * poly_len;
    
    size_t hint_size = poly_len * ctx->blowup_factor_ceil * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_hint, hint_size));
    size_t sum_size = ctx->blowup_factor_ceil * convd_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_hint_acc, sum_size));

    // Step 4 setup
    size_t query_q2_size = ctx->smaller_db_cols * sizeof(uint64_t); // per batch item
    CUDA_ASSERT(cudaMalloc(&ctx->d_query_q2, query_q2_size));
    size_t response_size = ctx->smaller_db_rows * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_response, response_size));

    // Determine num_streams based on available GPU memory
    size_t modified_rows_bytes = blowup_factor_ceil * ctx->smaller_db_cols * sizeof(uint16_t);
    size_t log2pl = ctx->ntt_params.log2_poly_len;
    size_t pub_params_full_bytes = log2pl * 2 * t_exp_left * convd_len * sizeof(uint64_t);
    size_t packed_excess_bytes = 2 * convd_len * sizeof(uint64_t);
    size_t pack_excess_scratch_bytes = ((2*convd_len) + (2*poly_len) + (2*poly_len) +
        (t_exp_left*poly_len) + (t_exp_left*convd_len) + (2*convd_len) + convd_len) * sizeof(uint64_t);
    size_t working_set_size = (1 << (log2pl - 1));
    size_t pack_lwe_scratch_bytes = ((working_set_size*poly_len) + 4*poly_len +
        (2*convd_len) + (2*poly_len) + convd_len) * sizeof(uint64_t);
    size_t per_stream_bytes = modified_rows_bytes + hint_size + sum_size + response_size +
        pub_params_full_bytes + packed_excess_bytes + pack_excess_scratch_bytes + pack_lwe_scratch_bytes;

    size_t free_mem, total_mem;
    CUDA_ASSERT(cudaMemGetInfo(&free_mem, &total_mem));
    size_t reserved = 128 * 1024 * 1024;  // reserve 128 MB for other allocs
    size_t available = (free_mem > reserved) ? (free_mem - reserved) : 0;

    size_t num_streams = available / per_stream_bytes;
    if (num_streams < 1) num_streams = 1;
    if (num_streams > ctx->max_batch_size) num_streams = ctx->max_batch_size;
    ctx->num_streams = num_streams;

    printf("GPU: %.1f MB free, per-stream %.2f MB, using %zu parallel streams\n",
        free_mem / 1e6, per_stream_bytes / 1e6, num_streams);

    // Create CUDA streams
    ctx->streams = new cudaStream_t[num_streams];
    for (size_t i = 0; i < num_streams; i++) {
        CUDA_ASSERT(cudaStreamCreate(&ctx->streams[i]));
    }

    // Per-stream allocations
    for (size_t i = 0; i < num_streams; i++) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_modified_rows[i], modified_rows_bytes));
        CUDA_ASSERT(cudaMalloc(&ctx->d_hint_batch[i], hint_size));
        CUDA_ASSERT(cudaMalloc(&ctx->d_hint_acc_batch[i], sum_size));
        CUDA_ASSERT(cudaMalloc(&ctx->d_response_batch[i], response_size));
    }

    // Pre-allocate batch upload/download buffers
    ctx->pub_params_row_1s_elems = poly_len * ctx->ntt_params.log2_poly_len * t_exp_left;
    CUDA_ASSERT(cudaMalloc(&ctx->d_query_q2_all,
        ctx->max_batch_size * ctx->smaller_db_cols * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&ctx->d_pub_params_row_1s_all,
        ctx->max_batch_size * ctx->pub_params_row_1s_elems * sizeof(uint64_t)));

    size_t q_1_bits_init = ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits_init = ceil_log2_u64(rlwe_q_prime_1);
    ctx->response_bytes_per_item = ((q_1_bits_init + q_2_bits_init) * poly_len + 7) / 8;
    CUDA_ASSERT(cudaMalloc(&ctx->d_all_responses,
        ctx->max_batch_size * ctx->response_bytes_per_item));

    // Pinned host memory for async batch transfers
    CUDA_ASSERT(cudaMallocHost(&ctx->h_query_q2_pinned,
        ctx->max_batch_size * ctx->smaller_db_cols * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMallocHost(&ctx->h_pub_params_pinned,
        ctx->max_batch_size * ctx->pub_params_row_1s_elems * sizeof(uint64_t)));
}

void ypir_init_packing_data(
    void* context,
    const uint64_t* y_constants, size_t y_constants_size,
    const uint64_t* prepacked_lwe, size_t prepacked_lwe_size,
    const uint64_t* precomp_res, size_t precomp_res_size,
    const uint64_t* precomp_vals, size_t precomp_vals_size,
    const uint64_t* precomp_tables, size_t precomp_tables_size,
    const uint64_t* fake_pack_pub_params, size_t fake_pack_pub_params_size
) 
{
    OnlineContext* ctx = (OnlineContext*)context;

    CUDA_ALLOC_AND_COPY(ctx->d_y_constants, y_constants, y_constants_size);
    CUDA_ALLOC_AND_COPY(ctx->d_prepacked_lwe, prepacked_lwe, prepacked_lwe_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_res, precomp_res, precomp_res_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_vals, precomp_vals, precomp_vals_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_tables, precomp_tables, precomp_tables_size);
    CUDA_ALLOC_AND_COPY(ctx->d_fake_pack_pub_params, fake_pack_pub_params, fake_pack_pub_params_size);


    // sizes
    size_t poly_len      = ctx->ntt_params.poly_len;          // 2048
    size_t crt_count     = ctx->ntt_params.crt_count;         // 2
    size_t convd_len     = crt_count * poly_len;              // 4096
    size_t log2_poly_len = ctx->ntt_params.log2_poly_len;     // 11
    size_t t_exp_left    = ctx->t_exp_left;                   // 3
    size_t blowup        = ctx->blowup_factor_ceil;           // 2

    size_t pub_param_size_u64 = 2 * t_exp_left * convd_len;   // per key_idx

    // 1) full pub params (expanded row1) for the current request/batch-item
    // if pub_params_row_1s is per-query, allocate per-request. If it's constant, allocate once.
    CUDA_ASSERT(cudaMalloc(&ctx->d_pack_pub_params_full,
                        log2_poly_len * pub_param_size_u64 * sizeof(uint64_t)));

    // 2) rotation NTT table (depends only on (special_offs, blowup, params))
    // allocate once (or reallocate when special_offs/blowup changes)
    CUDA_ASSERT(cudaMalloc(&ctx->d_rotation_ntt_table,
                        blowup * convd_len * sizeof(uint64_t)));

    // build it once
    int threads_pack = 256;
    precompute_rotation_ntt_table<<<blowup, threads_pack>>>(
        ctx->d_rotation_ntt_table,
        ctx->special_offs,
        ctx->blowup_factor_ceil,
        ctx->ntt_params
    );
    CUDA_ASSERT(cudaGetLastError());

    // Pre-allocate per-batch scratch buffers
    size_t pack_excess_scratch_sz_u64 =
        (2 * convd_len) +                 // cur_r
        (2 * poly_len) +                  // ct_raw
        (2 * poly_len) +                  // ct_auto
        (t_exp_left * poly_len) +         // ginv_ct
        (t_exp_left * convd_len) +        // ginv_ct_ntt
        (2 * convd_len) +                 // tau_of_r
        convd_len;                        // temp_ntt

    size_t working_set_size = (1 << (log2_poly_len - 1));
    size_t pack_lwe_scratch_sz =
        (working_set_size * poly_len) +   // working_set
        poly_len +                        // y_times_ct_odd
        poly_len +                        // neg_y_times_ct_odd
        poly_len +                        // ct_sum_1
        poly_len +                        // w_times_ginv_ct
        (2 * convd_len) +                 // result_ntt
        (2 * poly_len) +                  // temp_raw
        convd_len;                        // temp_ntt

    size_t packed_excess_sz = 2 * convd_len;

    for (size_t i = 0; i < ctx->num_streams; i++) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_pack_pub_params_full_batch[i],
                    log2_poly_len * pub_param_size_u64 * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&ctx->d_packed_excess_batch[i],
                    packed_excess_sz * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&ctx->d_pack_excess_scratch_batch[i],
                    pack_excess_scratch_sz_u64 * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&ctx->d_pack_lwe_scratch_batch[i],
                    pack_lwe_scratch_sz * sizeof(uint64_t)));
    }
}

// Full batch execution: Step 1 -> Loop(Step 2 -> Step 3 -> Step 4)
void ypir_online_compute_full_batch(
    void* context,
    const Elem* query,             // Step 1 input (batch)
    const uint64_t* query_q2_batch,// Step 4 input (batch)
    const uint64_t* pack_pub_params_row_1s_batch, // Step 5 input (batch)
    size_t batch_size,
    uint8_t* responses // output
) 
{
    OnlineContext* ctx = (OnlineContext*)context;
    GpuTimer t;

    size_t query_bytes = ctx->db_cols * batch_size * sizeof(Elem);
    size_t q2_bytes = batch_size * ctx->smaller_db_cols * sizeof(uint64_t);
    size_t pub_params_bytes = batch_size * ctx->pub_params_row_1s_elems * sizeof(uint64_t);

    // Stage query to pinned memory and async upload on default stream
    memcpy(ctx->h_query_pinned, query, query_bytes);
    CUDA_ASSERT(cudaMemcpyAsync(ctx->d_query, ctx->h_query_pinned, query_bytes,
        cudaMemcpyHostToDevice, 0));

    // Step 1: Matrix multiply DB × query → result
    t.tic();
    if (ctx->has_tensor_cores) {
        // Decompose uint32 query into 4 uint8 byte slices
        size_t total_elems = ctx->db_cols * batch_size;
        int dec_threads = 256;
        int dec_blocks = (total_elems + dec_threads - 1) / dec_threads;
        decompose_query_bytes_kernel<<<dec_blocks, dec_threads>>>(
            ctx->d_query_bytes[0], ctx->d_query_bytes[1],
            ctx->d_query_bytes[2], ctx->d_query_bytes[3],
            ctx->d_query, total_elems);
        CUDA_ASSERT(cudaGetLastError());

        // 4 cuBLAS uint8 GEMMs with alpha/beta folding (tensor cores)
        // D = sum_{g=0..3} (256^g) * DB_u8 * q_byte_g
        int32_t alphas[4] = {1, 256, 65536, 16777216};
        int32_t betas[4] = {0, 1, 1, 1};
        int M = (int)ctx->db_rows;
        int N = (int)batch_size;
        int K = (int)ctx->db_cols;

        for (int g = 0; g < 4; g++) {
            CUBLAS_CHECK(cublasGemmEx(ctx->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alphas[g],
                ctx->d_db,            CUDA_R_8U, K,
                ctx->d_query_bytes[g], CUDA_R_8U, K,
                &betas[g],
                ctx->d_result,        CUDA_R_32I, M,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    } else {
        // CUTLASS SIMT GEMM (fallback for SM < 72)
        cutlass::gemm::GemmCoord problem_size(ctx->db_rows, batch_size, ctx->db_cols);
        uint32_t alpha = 1, beta = 0;
        CutlassGemm::Arguments args{
            problem_size,
            {ctx->d_db, (int)ctx->db_cols},
            {ctx->d_query, (int)ctx->db_cols},
            {ctx->d_result, (int)ctx->db_rows},
            {ctx->d_result, (int)ctx->db_rows},
            {alpha, beta}, 1
        };
        CutlassGemm gemm_op;
        cutlass::Status status = gemm_op.initialize(args, ctx->d_gemm_workspace);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
        }
    }
    CUDA_ASSERT(cudaGetLastError());

    // Async upload q2 and pub_params on transfer stream (overlaps with GEMM)
    memcpy(ctx->h_query_q2_pinned, query_q2_batch, q2_bytes);
    memcpy(ctx->h_pub_params_pinned, pack_pub_params_row_1s_batch, pub_params_bytes);
    CUDA_ASSERT(cudaMemcpyAsync(ctx->d_query_q2_all, ctx->h_query_q2_pinned, q2_bytes,
        cudaMemcpyHostToDevice, ctx->transfer_stream));
    CUDA_ASSERT(cudaMemcpyAsync(ctx->d_pub_params_row_1s_all, ctx->h_pub_params_pinned, pub_params_bytes,
        cudaMemcpyHostToDevice, ctx->transfer_stream));

    // Wait for GEMM + transfers
    CUDA_ASSERT(cudaDeviceSynchronize());
    float ms1 = t.toc_ms();

    // Prepare for parallel loop
    size_t convd_len = ctx->ntt_params.crt_count * ctx->ntt_params.poly_len;
    size_t poly_len = ctx->ntt_params.poly_len;

    // Time the parallel portion
    t.tic();

    // Process batch in chunks of num_streams
    for (size_t chunk_start = 0; chunk_start < batch_size; chunk_start += ctx->num_streams) {
        size_t chunk_end = chunk_start + ctx->num_streams;
        if (chunk_end > batch_size) chunk_end = batch_size;

        // Launch Steps 2-5 for each batch item on its own CUDA stream
        for (size_t i = chunk_start; i < chunk_end; i++) {
            size_t s = i - chunk_start;  // stream/buffer index
            cudaStream_t stream = ctx->streams[s];

            // Step 2: Rescale into per-stream overlay buffer
            Elem* d_intermediate = ctx->d_result + i * ctx->smaller_db_cols;

            int threads_step2 = 1024;
            int blocks_step2 = (ctx->smaller_db_cols + threads_step2 - 1) / threads_step2;

            rescale_and_expand_kernel<<<blocks_step2, threads_step2, 0, stream>>>(
                d_intermediate,
                ctx->d_modified_rows[s],
                ctx->smaller_db_cols,
                ctx->lwe_modulus,
                ctx->lwe_q_prime,
                (int)ctx->pt_bits,
                ctx->blowup_factor_ceil
            );
            CUDA_ASSERT(cudaGetLastError());

            // Step 3: Compute Secondary Hint (reads overlay only)
            size_t shared_mem_step3 = convd_len * sizeof(uint64_t);
            int threads_step3 = 1024;
            compute_secondary_hint_kernel<<<ctx->blowup_factor_ceil, threads_step3, shared_mem_step3, stream>>>(
                ctx->d_hint_batch[s],
                ctx->d_modified_rows[s],
                ctx->d_query_ntt,
                ctx->d_hint_acc_batch[s],
                ctx->ntt_params,
                ctx->smaller_db_cols / ctx->ntt_params.poly_len,
                ctx->blowup_factor_ceil
            );
            CUDA_ASSERT(cudaGetLastError());

            // Step 4: Compute Response (shared d_smaller_db + overlay for modified rows)
            uint64_t* d_q2_current = ctx->d_query_q2_all + i * ctx->smaller_db_cols;

            int threads_step4 = 256;
            dim3 grid(ctx->smaller_db_rows);
            compute_response_kernel<<<grid, threads_step4, 0, stream>>>(
                ctx->d_response_batch[s],
                ctx->d_smaller_db,
                d_q2_current,
                ctx->smaller_db_cols,
                ctx->smaller_db_rows,
                ctx->d_modified_rows[s],
                ctx->special_offs,
                ctx->blowup_factor_ceil,
                ctx->ntt_params
            );
            CUDA_ASSERT(cudaGetLastError());

            // Step 5a: Build pub params and pack excess
            int threads_pack = 1024;

            build_pub_params_full<<<ctx->ntt_params.log2_poly_len, threads_pack, 0, stream>>>(
                ctx->d_pack_pub_params_full_batch[s],
                ctx->d_fake_pack_pub_params,
                ctx->d_pub_params_row_1s_all + i * ctx->pub_params_row_1s_elems,
                ctx->t_exp_left,
                ctx->ntt_params
            );
            CUDA_ASSERT(cudaGetLastError());

            pack_excess<<<1, threads_pack, 0, stream>>>(
                ctx->d_packed_excess_batch[s],
                ctx->d_hint_batch[s],
                ctx->d_pack_pub_params_full_batch[s],
                ctx->d_rotation_ntt_table,
                ctx->d_pack_excess_scratch_batch[s],
                ctx->special_offs,
                ctx->blowup_factor_ceil,
                ctx->t_exp_left,
                ctx->ntt_params
            );
            CUDA_ASSERT(cudaGetLastError());

            // Step 5b: Pack LWEs and mod switch
            uint8_t* d_response_i = ctx->d_all_responses + i * ctx->response_bytes_per_item;

            pack_lwes_and_mod_switch<<<1, threads_pack, 0, stream>>>(
                d_response_i,
                ctx->d_packed_excess_batch[s],
                ctx->d_response_batch[s],
                ctx->d_y_constants,
                ctx->d_precomp_res,
                ctx->d_precomp_vals,
                ctx->d_precomp_tables,
                ctx->d_pub_params_row_1s_all + i * ctx->pub_params_row_1s_elems,
                ctx->d_pack_lwe_scratch_batch[s],
                ctx->t_exp_left,
                ctx->rlwe_q_prime_1,
                ctx->rlwe_q_prime_2,
                ctx->ntt_params
            );
            CUDA_ASSERT(cudaGetLastError());
        }

        // Synchronize all streams in this chunk
        for (size_t i = chunk_start; i < chunk_end; i++) {
            size_t s = i - chunk_start;
            CUDA_ASSERT(cudaStreamSynchronize(ctx->streams[s]));
        }
    }

    float ms_parallel = t.toc_ms();

    printf("Step1 %.3f ms, Steps2-5 (parallel, %zu streams) %.3f ms\n",
        ms1, ctx->num_streams, ms_parallel);

    // Download responses
    CUDA_ASSERT(cudaMemcpy(responses, ctx->d_all_responses,
        batch_size * ctx->response_bytes_per_item, cudaMemcpyDeviceToHost));
}

// Free online computation context
void ypir_online_free(void* context)
{
    OnlineContext* ctx = (OnlineContext*)context;

    CUDA_ASSERT(cudaFree(ctx->d_db));
    CUDA_ASSERT(cudaFree(ctx->d_query));
    CUDA_ASSERT(cudaFree(ctx->d_result));

    CUDA_ASSERT(cudaFree(ctx->d_smaller_db));
    CUDA_ASSERT(cudaFree(ctx->d_query_ntt));
    CUDA_ASSERT(cudaFree(ctx->d_hint_acc));
    CUDA_ASSERT(cudaFree(ctx->d_hint));
    CUDA_ASSERT(cudaFree(ctx->d_query_q2));
    CUDA_ASSERT(cudaFree(ctx->d_response));

    CUDA_ASSERT(cudaFree(ctx->ntt_params.moduli));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.barrett_cr));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_prime_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_prime_table));

    CUDA_ASSERT(cudaFree(ctx->d_y_constants));
    CUDA_ASSERT(cudaFree(ctx->d_prepacked_lwe));
    CUDA_ASSERT(cudaFree(ctx->d_precomp_res));
    CUDA_ASSERT(cudaFree(ctx->d_precomp_vals));
    CUDA_ASSERT(cudaFree(ctx->d_precomp_tables));
    CUDA_ASSERT(cudaFree(ctx->d_fake_pack_pub_params));
    CUDA_ASSERT(cudaFree(ctx->d_pack_pub_params_full));
    CUDA_ASSERT(cudaFree(ctx->d_rotation_ntt_table));

    // Free per-stream resources
    for (size_t i = 0; i < ctx->num_streams; i++) {
        CUDA_ASSERT(cudaFree(ctx->d_modified_rows[i]));
        CUDA_ASSERT(cudaFree(ctx->d_hint_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_hint_acc_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_response_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_pack_pub_params_full_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_packed_excess_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_pack_excess_scratch_batch[i]));
        CUDA_ASSERT(cudaFree(ctx->d_pack_lwe_scratch_batch[i]));
        CUDA_ASSERT(cudaStreamDestroy(ctx->streams[i]));
    }
    delete[] ctx->d_modified_rows;
    delete[] ctx->d_hint_batch;
    delete[] ctx->d_hint_acc_batch;
    delete[] ctx->d_response_batch;
    delete[] ctx->d_pack_pub_params_full_batch;
    delete[] ctx->d_packed_excess_batch;
    delete[] ctx->d_pack_excess_scratch_batch;
    delete[] ctx->d_pack_lwe_scratch_batch;
    delete[] ctx->streams;

    // Free pre-allocated batch buffers
    CUDA_ASSERT(cudaFree(ctx->d_query_q2_all));
    CUDA_ASSERT(cudaFree(ctx->d_pub_params_row_1s_all));
    CUDA_ASSERT(cudaFree(ctx->d_all_responses));
    if (ctx->d_gemm_workspace) CUDA_ASSERT(cudaFree(ctx->d_gemm_workspace));

    // Free tensor core resources
    if (ctx->has_tensor_cores) {
        CUBLAS_CHECK(cublasDestroy(ctx->cublas_handle));
        for (int i = 0; i < 4; i++) {
            CUDA_ASSERT(cudaFree(ctx->d_query_bytes[i]));
        }
    }

    CUDA_ASSERT(cudaStreamDestroy(ctx->transfer_stream));

    // Free pinned host memory
    CUDA_ASSERT(cudaFreeHost(ctx->h_query_pinned));
    CUDA_ASSERT(cudaFreeHost(ctx->h_query_q2_pinned));
    CUDA_ASSERT(cudaFreeHost(ctx->h_pub_params_pinned));

    delete ctx;
}
} // extern "C"
