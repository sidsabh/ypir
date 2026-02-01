#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include "ntt.cuh"

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

#define MAX_SP_BATCH_SIZE 16

// SimplePIR online context
struct SPOnlineContext {
    // Database: column-major, db_cols x db_rows_padded (u16 values)
    uint16_t* d_db = nullptr;
    
    // Step 1 output / Step 2 input (batch)
    uint64_t* d_intermediate = nullptr;  // MAX_SP_BATCH_SIZE * db_cols
    
    // Packing data (from offline precomputation)
    uint64_t* d_y_constants = nullptr;
    uint64_t* d_precomp_res = nullptr;
    uint64_t* d_precomp_vals = nullptr;
    uint64_t* d_precomp_tables = nullptr;
    
    // Scratch space for packing (per output)
    uint64_t* d_scratch = nullptr;
    
    // NTT parameters
    NTTParams ntt_params;
    
    // Dimensions
    size_t db_rows;
    size_t db_rows_padded;
    size_t db_cols;
    size_t num_rlwe_outputs;  // db_cols / poly_len
    size_t t_exp_left;
    
    // Modulus switch parameters
    uint64_t rlwe_q_prime_1;
    uint64_t rlwe_q_prime_2;
};

// ==================== Step 1: Matrix-vector multiply (CRT-packed) ====================
// query is CRT-packed: lo 32 bits = CRT0, hi 32 bits = CRT1


#define SP_TILE_ROWS 512

template<int B>
__global__ void sp_matmul_fused_tiled(
    uint64_t* __restrict__ out,         // B * db_cols
    const uint64_t* __restrict__ query, // B * db_rows_padded (CRT-packed)
    const uint16_t* __restrict__ db,    // db_cols * db_rows_padded
    size_t db_rows,
    size_t db_rows_padded,
    size_t db_cols,
    NTTParams params
) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= db_cols) return;
    
    extern __shared__ uint64_t s_query[]; // B * SP_TILE_ROWS
    
    uint64_t acc_lo[B], acc_hi[B];
    #pragma unroll
    for (int b = 0; b < B; b++) {
        acc_lo[b] = 0;
        acc_hi[b] = 0;
    }
    
    const uint16_t* db_col = db + col * db_rows_padded;
    
    for (size_t tile_start = 0; tile_start < db_rows; tile_start += SP_TILE_ROWS) {
        size_t tile_len = min((size_t)SP_TILE_ROWS, db_rows - tile_start);
        
        // Load query tile for all batches into shared memory
        for (size_t idx = threadIdx.x; idx < B * tile_len; idx += blockDim.x) {
            int b = idx / tile_len;
            size_t r = idx % tile_len;
            s_query[b * SP_TILE_ROWS + r] = query[b * db_rows_padded + tile_start + r];
        }
        __syncthreads();
        
        // Accumulate
        for (size_t r = 0; r < tile_len; r++) {
            uint64_t d = (uint64_t)db_col[tile_start + r];
            #pragma unroll
            for (int b = 0; b < B; b++) {
                uint64_t q = s_query[b * SP_TILE_ROWS + r];
                acc_lo[b] += (q & 0xFFFFFFFF) * d;
                acc_hi[b] += (q >> 32) * d;
            }
        }
        __syncthreads();
    }
    
    // Reduce and write output
    #pragma unroll
    for (int b = 0; b < B; b++) {
        uint64_t lo = barrett_raw_u64(acc_lo[b], params.barrett_cr[0], params.moduli[0]);
        uint64_t hi = barrett_raw_u64(acc_hi[b], params.barrett_cr[1], params.moduli[1]);
        uint64_t composed = crt_compose_2(lo, hi, &params);
        out[b * db_cols + col] = barrett_raw_u64(composed, params.barrett_cr_1_modulus, params.modulus);
    }
}

// Dispatch macro
#define SP_BATCH_LIST(X) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12)

#define SP_LAUNCH_CASE(B) \
    case (B): { \
        size_t shmem = (B) * SP_TILE_ROWS * sizeof(uint64_t); \
        sp_matmul_fused_tiled<(B)><<<blocks, threads, shmem>>>( \
            out, query, db, db_rows, db_rows_padded, db_cols, params); \
        break; \
    }

static inline void launch_sp_matmul(
    uint64_t* out,
    const uint64_t* query,
    const uint16_t* db,
    size_t db_rows,
    size_t db_rows_padded,
    size_t db_cols,
    int batch_size,
    NTTParams params
) {
    int threads = 1024;
    int blocks = (db_cols + threads - 1) / threads;
    switch (batch_size) {
        SP_BATCH_LIST(SP_LAUNCH_CASE)
        default:
            fprintf(stderr, "Unsupported SP batch_size=%d\n", batch_size);
            break;
    }
}


template<int BLOCKSZ>
__global__ void sp_matmul_kernel(
    uint64_t* __restrict__ out,           // OUT: batch_size * db_cols
    const uint64_t* __restrict__ query,   // IN: batch_size * db_rows_padded (CRT-packed)
    const uint16_t* __restrict__ db,      // IN: column-major db_cols x db_rows_padded
    size_t db_rows,
    size_t db_rows_padded,
    size_t db_cols,
    NTTParams params
) {
    size_t col = blockIdx.x;
    size_t batch_idx = blockIdx.y;
    if (col >= db_cols) return;
    
    size_t tid = threadIdx.x;
    const uint16_t* db_col = db + col * db_rows_padded;
    const uint64_t* query_batch = query + batch_idx * db_rows_padded;
    
    // Each thread accumulates partial sums for lo and hi separately
    uint64_t partial_sum_lo = 0;
    uint64_t partial_sum_hi = 0;
    
    for (size_t row = tid; row < db_rows; row += blockDim.x) {
        uint64_t q = query_batch[row];
        uint64_t q_lo = q & 0xFFFFFFFF;
        uint64_t q_hi = q >> 32;
        uint64_t d = (uint64_t)db_col[row];
        
        partial_sum_lo += q_lo * d;
        partial_sum_hi += q_hi * d;
    }
    
    // Reduce within block
    __shared__ uint64_t sdata_lo[BLOCKSZ];
    __shared__ uint64_t sdata_hi[BLOCKSZ];
    sdata_lo[tid] = partial_sum_lo;
    sdata_hi[tid] = partial_sum_hi;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_lo[tid] += sdata_lo[tid + s];
            sdata_hi[tid] += sdata_hi[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // Barrett reduce each CRT component
        uint64_t lo = barrett_raw_u64(sdata_lo[0], params.barrett_cr[0], params.moduli[0]);
        uint64_t hi = barrett_raw_u64(sdata_hi[0], params.barrett_cr[1], params.moduli[1]);
        
        // CRT compose
        uint64_t composed = crt_compose_2(lo, hi, &params);
        
        // Final Barrett reduction
        out[batch_idx * db_cols + col] = barrett_raw_u64(composed, params.barrett_cr_1_modulus, params.modulus);
    }
}


// ==================== Step 2: Pack LWEs and mod switch (parallel over outputs) ====================

__device__ __forceinline__
void write_arbitrary_bits(uint8_t* out, uint64_t val, size_t bit_offs, size_t num_bits) {
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
uint32_t ceil_log2_u64(uint64_t x) {
    if (x <= 1) return 0;
    uint64_t y = x - 1;
#if defined(__CUDA_ARCH__)
    return 64u - (uint32_t)__clzll(y);
#else
    return 64u - (uint32_t)__builtin_clzll(y);
#endif
}

// Grid: (num_rlwe_outputs, batch_size) blocks
// Each block handles one (output, batch) pair
__global__ void sp_pack_lwes_and_mod_switch_batch(
    uint8_t* d_response_out,             // output: batch_size * num_outputs * response_bytes_per_output
    const uint64_t* d_intermediate,      // b_values from step 1: batch_size * db_cols u64s
    const uint64_t* d_y_constants,       // [y_0..y_{ell-1}, neg_y_0..neg_y_{ell-1}], each crt_count*poly_len
    const uint64_t* d_precomp_res,       // num_outputs * 2 * crt_count * poly_len
    const uint64_t* d_precomp_vals,      // num_outputs * (2^ell - 1) * t_exp_left * poly_len (condensed)
    const uint64_t* d_precomp_tables,    // num_outputs * ell * poly_len
    const uint64_t* d_pub_params_row_1s, // batch_size * ell * t_exp_left * poly_len (condensed)
    uint64_t* d_scratch,                 // batch_size * num_outputs * scratch_size_per_output
    size_t num_outputs,
    size_t db_cols,
    size_t t_exp_left,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    size_t scratch_size_per_output,
    size_t pub_params_size_per_batch,
    NTTParams params
)
{
    size_t output_idx = blockIdx.x;
    size_t batch_idx = blockIdx.y;
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
    
    // Per-(batch, output) scratch
    size_t scratch_idx = batch_idx * num_outputs + output_idx;
    uint64_t* my_scratch = d_scratch + scratch_idx * scratch_size_per_output;
    
    // Scratch layout
    size_t working_set_size = (1 << (ell - 1));
    uint64_t* working_set = my_scratch;
    uint64_t* y_times_ct_odd = working_set + working_set_size * poly_len;
    uint64_t* neg_y_times_ct_odd = y_times_ct_odd + poly_len;
    uint64_t* ct_sum_1 = neg_y_times_ct_odd + poly_len;
    uint64_t* w_times_ginv_ct = ct_sum_1 + poly_len;
    uint64_t* result_ntt = w_times_ginv_ct + poly_len;
    uint64_t* temp_raw = result_ntt + 2 * convd_len;
    uint64_t* temp_ntt = temp_raw + 2 * poly_len;
    
    // b_values for this (batch, output)
    const uint64_t* b_values = d_intermediate + batch_idx * db_cols + output_idx * poly_len;
    
    // Precomp data for this output (shared across batch)
    size_t precomp_vals_per_output = ((1 << ell) - 1) * t_exp_left * poly_len;
    size_t precomp_tables_per_output = ell * poly_len;
    const uint64_t* my_precomp_res = d_precomp_res + output_idx * 2 * convd_len;
    const uint64_t* my_precomp_vals = d_precomp_vals + output_idx * precomp_vals_per_output;
    const uint64_t* my_precomp_tables = d_precomp_tables + output_idx * precomp_tables_per_output;
    
    // Pub params for this batch item
    const uint64_t* my_pub_params = d_pub_params_row_1s + batch_idx * pub_params_size_per_batch;
    
    // Output location for this (batch, output)
    size_t response_bytes_per_batch = num_outputs * response_bytes_per_output;
    uint8_t* my_response = d_response_out + batch_idx * response_bytes_per_batch + output_idx * response_bytes_per_output;
    
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
            const uint64_t* w = my_pub_params + pub_param_idx * t_exp_left * poly_len;
            const uint64_t* ginv_ct = my_precomp_vals + idx_precomp * t_exp_left * poly_len;
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
                const uint64_t* table = my_precomp_tables + table_idx * poly_len;
                
                for (size_t z = tid; z < poly_len; z += blockDim.x) {
                    size_t src_idx = (size_t)table[z];
                    ct_even[z] += ct_sum_1[src_idx];
                }
                __syncthreads();
            }
            
            // Add w_times_ginv_ct to ct_even
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
        result_ntt[z] = my_precomp_res[z];
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
        uint64_t mod = params.moduli[crt];
        uint64_t barrett_cr = params.barrett_cr[crt];
        temp_ntt[z] = barrett_raw_u64(result_ntt[convd_len + z], barrett_cr, mod);
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
        __uint128_t prod = (__uint128_t)b_values[z] * poly_len;
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
    
    // ==================== NO pack_excess addition for SimplePIR ====================
    
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
    size_t q_1_bits = ceil_log2_u64(rlwe_q_prime_2);
    size_t q_2_bits = ceil_log2_u64(rlwe_q_prime_1);
    size_t total_sz_bytes = response_bytes_per_output;
    
    // Zero output
    for (size_t i = tid; i < total_sz_bytes; i += blockDim.x) {
        my_response[i] = 0;
    }
    __syncthreads();
    
    // Row 0: rescale to q_2, write with q_1_bits
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_2) / (double)modulus + 0.5);
        
        size_t bit_offs = z * q_1_bits;
        write_arbitrary_bits(my_response, val_rescaled, bit_offs, q_1_bits);
    }
    __syncthreads();
    
    // Row 1: rescale to q_1, write with q_2_bits
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t val = temp_raw[poly_len + z];
        double d_val = (double)val;
        uint64_t val_rescaled = (uint64_t)((d_val * (double)rlwe_q_prime_1) / (double)modulus + 0.5);
        
        size_t bit_offs = poly_len * q_1_bits + z * q_2_bits;
        write_arbitrary_bits(my_response, val_rescaled, bit_offs, q_2_bits);
    }
}


extern "C" {

// Initialize SimplePIR online context
void* ypir_sp_online_init(
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
    uint64_t barrett_cr_1_modulus
) {
    SPOnlineContext* ctx = new SPOnlineContext();
    
    ctx->db_rows = db_rows;
    ctx->db_rows_padded = db_rows_padded;
    ctx->db_cols = db_cols;
    ctx->num_rlwe_outputs = db_cols / poly_len;
    ctx->t_exp_left = t_exp_left;
    ctx->rlwe_q_prime_1 = rlwe_q_prime_1;
    ctx->rlwe_q_prime_2 = rlwe_q_prime_2;
    
    // Set up NTT params
    ctx->ntt_params.poly_len = poly_len;
    ctx->ntt_params.log2_poly_len = 31 - __builtin_clz(poly_len);
    ctx->ntt_params.crt_count = crt_count;
    ctx->ntt_params.mod0_inv_mod1 = mod0_inv_mod1;
    ctx->ntt_params.mod1_inv_mod0 = mod1_inv_mod0;
    ctx->ntt_params.barrett_cr_0_modulus = barrett_cr_0_modulus;
    ctx->ntt_params.barrett_cr_1_modulus = barrett_cr_1_modulus;
    ctx->ntt_params.modulus = moduli[0] * moduli[1];
    
    // Allocate and copy NTT tables
    size_t table_size = crt_count * poly_len * sizeof(uint64_t);
    size_t moduli_size = crt_count * sizeof(uint64_t);
    
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.moduli, moduli, moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.barrett_cr, barrett_cr, moduli_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_table, forward_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.forward_prime_table, forward_prime_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_table, inverse_table, table_size);
    CUDA_ALLOC_AND_COPY(ctx->ntt_params.inverse_prime_table, inverse_prime_table, table_size);
    
    // Upload DB
    size_t db_size = db_cols * db_rows_padded * sizeof(uint16_t);
    CUDA_ALLOC_AND_COPY(ctx->d_db, db, db_size);
    
    // Allocate intermediate buffer (for batch)
    CUDA_ASSERT(cudaMalloc(&ctx->d_intermediate, MAX_SP_BATCH_SIZE * db_cols * sizeof(uint64_t)));
    
    // Allocate scratch for packing (per output)
    size_t ell = ctx->ntt_params.log2_poly_len;
    size_t convd_len = crt_count * poly_len;
    size_t working_set_size = (1 << (ell - 1));
    size_t scratch_size_per_output = 
        (working_set_size * poly_len) +  // working_set
        poly_len +                        // y_times_ct_odd
        poly_len +                        // neg_y_times_ct_odd
        poly_len +                        // ct_sum_1
        poly_len +                        // w_times_ginv_ct
        (2 * convd_len) +                 // result_ntt
        (2 * poly_len) +                  // temp_raw
        convd_len;                        // temp_ntt
    
    CUDA_ASSERT(cudaMalloc(&ctx->d_scratch, ctx->num_rlwe_outputs * scratch_size_per_output * sizeof(uint64_t)));
    
    return ctx;
}

// Upload packing data (from offline precomputation)
void ypir_sp_online_init_packing(
    void* context,
    const uint64_t* y_constants, size_t y_constants_size,
    const uint64_t* precomp_res, size_t precomp_res_size,
    const uint64_t* precomp_vals, size_t precomp_vals_size,
    const uint64_t* precomp_tables, size_t precomp_tables_size
) {
    SPOnlineContext* ctx = (SPOnlineContext*)context;
    if (!ctx) return;
    
    CUDA_ALLOC_AND_COPY(ctx->d_y_constants, y_constants, y_constants_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_res, precomp_res, precomp_res_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_vals, precomp_vals, precomp_vals_size);
    CUDA_ALLOC_AND_COPY(ctx->d_precomp_tables, precomp_tables, precomp_tables_size);
}

// Run full online computation for a batch (fully parallel)
void ypir_sp_online_compute_batch(
    void* context,
    const uint64_t* queries,                  // batch_size * db_rows_padded u64s
    const uint64_t* pack_pub_params_row_1s,   // batch_size * ell * t_exp_left * poly_len (condensed)
    uint8_t* response_out,                    // output: batch_size * num_outputs * response_bytes_per_output
    size_t response_bytes_per_batch,
    size_t batch_size
) {
    SPOnlineContext* ctx = (SPOnlineContext*)context;
    if (!ctx || batch_size == 0) return;

    float ms1=0, ms2=0;
    GpuTimer t;

    
    size_t poly_len = ctx->ntt_params.poly_len;
    size_t crt_count = ctx->ntt_params.crt_count;
    size_t convd_len = crt_count * poly_len;
    size_t ell = ctx->ntt_params.log2_poly_len;
    size_t num_outputs = ctx->num_rlwe_outputs;
    
    // Calculate sizes
    size_t q_1_bits = ceil_log2_u64(ctx->rlwe_q_prime_2);
    size_t q_2_bits = ceil_log2_u64(ctx->rlwe_q_prime_1);
    size_t response_bytes_per_output = ((q_1_bits + q_2_bits) * poly_len + 7) / 8;
    
    size_t working_set_size = (1 << (ell - 1));
    size_t scratch_size_per_output = 
        (working_set_size * poly_len) +
        poly_len + poly_len + poly_len + poly_len +
        (2 * convd_len) + (2 * poly_len) + convd_len;
    
    size_t pub_params_size_per_batch = ell * ctx->t_exp_left * poly_len;
    
    // Upload all queries
    uint64_t* d_queries;
    CUDA_ALLOC_AND_COPY(d_queries, queries, batch_size * ctx->db_rows_padded * sizeof(uint64_t));
    
    // Upload all pub_params
    uint64_t* d_pub_params_batch;
    CUDA_ALLOC_AND_COPY(d_pub_params_batch, pack_pub_params_row_1s, 
                        batch_size * pub_params_size_per_batch * sizeof(uint64_t));
    
    // Allocate device output
    uint8_t* d_response;
    CUDA_ASSERT(cudaMalloc(&d_response, batch_size * response_bytes_per_batch));
    
    // Allocate scratch for all (batch, output) pairs
    uint64_t* d_scratch_batch;
    CUDA_ASSERT(cudaMalloc(&d_scratch_batch, batch_size * num_outputs * scratch_size_per_output * sizeof(uint64_t)));
    
    // Step 1: Matrix-vector multiply
    t.tic();
    if (batch_size >= 6) {
        // Use the Tiled kernel
        // It is compute-bound / optimized for reuse
        launch_sp_matmul(
            ctx->d_intermediate,
            d_queries,
            ctx->d_db,
            ctx->db_rows,
            ctx->db_rows_padded,
            ctx->db_cols,
            (int)batch_size,
            ctx->ntt_params
        );
    } else {
        // Use the Simple kernel
        // It is latency-bound / optimized for low overhead
        dim3 grid_matmul(ctx->db_cols, batch_size);
        const int threads_step1 = 1024;
        sp_matmul_kernel<threads_step1><<<grid_matmul, threads_step1>>>(
            ctx->d_intermediate,
            d_queries,
            ctx->d_db,
            ctx->db_rows,
            ctx->db_rows_padded,
            ctx->db_cols,
            ctx->ntt_params
        );
    }
    CUDA_ASSERT(cudaGetLastError());
    ms1 = t.toc_ms();

    
    // Step 2: Pack all (output, batch) pairs in parallel - 2D grid (num_outputs, batch_size)
    dim3 grid_pack(num_outputs, batch_size);
    int threads_pack = 1024;
    
    t.tic();
    sp_pack_lwes_and_mod_switch_batch<<<grid_pack, threads_pack>>>(
        d_response,
        ctx->d_intermediate,
        ctx->d_y_constants,
        ctx->d_precomp_res,
        ctx->d_precomp_vals,
        ctx->d_precomp_tables,
        d_pub_params_batch,
        d_scratch_batch,
        num_outputs,
        ctx->db_cols,
        ctx->t_exp_left,
        ctx->rlwe_q_prime_1,
        ctx->rlwe_q_prime_2,
        response_bytes_per_output,
        scratch_size_per_output,
        pub_params_size_per_batch,
        ctx->ntt_params
    );
    CUDA_ASSERT(cudaGetLastError());
    ms2 = t.toc_ms();

    printf("Step1 %.3f ms, Step2 %.3f ms\n",
        ms1, ms2);

    CUDA_ASSERT(cudaDeviceSynchronize());
    
    // Download results
    CUDA_ASSERT(cudaMemcpy(response_out, d_response, 
                           batch_size * response_bytes_per_batch, 
                           cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_ASSERT(cudaFree(d_queries));
    CUDA_ASSERT(cudaFree(d_pub_params_batch));
    CUDA_ASSERT(cudaFree(d_scratch_batch));
    CUDA_ASSERT(cudaFree(d_response));
}

// Free SimplePIR online context
void ypir_sp_online_free(void* context) {
    SPOnlineContext* ctx = (SPOnlineContext*)context;
    if (!ctx) return;
    
    CUDA_ASSERT(cudaFree(ctx->d_db));
    CUDA_ASSERT(cudaFree(ctx->d_intermediate));
    CUDA_ASSERT(cudaFree(ctx->d_scratch));
    
    if (ctx->d_y_constants) CUDA_ASSERT(cudaFree(ctx->d_y_constants));
    if (ctx->d_precomp_res) CUDA_ASSERT(cudaFree(ctx->d_precomp_res));
    if (ctx->d_precomp_vals) CUDA_ASSERT(cudaFree(ctx->d_precomp_vals));
    if (ctx->d_precomp_tables) CUDA_ASSERT(cudaFree(ctx->d_precomp_tables));
    
    CUDA_ASSERT(cudaFree(ctx->ntt_params.moduli));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.barrett_cr));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_prime_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_prime_table));
    
    delete ctx;
}

} // extern "C"
