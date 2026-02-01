#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// Include shared NTT infrastructure
#include "ntt.cuh"


// SimplePIR offline context
struct SPOfflineContext {
    // Database: column-major, db_cols x db_rows_padded (u8 values)
    uint16_t* d_db = nullptr;
    
    // Precomputed query in NTT form: db_rows_poly x crt_count x poly_len
    uint64_t* d_query_ntt = nullptr;
    
    // Output hint: poly_len x db_cols
    uint64_t* d_hint_0 = nullptr;
    
    // Global accumulator for kernel: db_cols x convd_len
    uint64_t* d_accum = nullptr;
    
    // NTT parameters
    NTTParams ntt_params;
    
    // Dimensions
    size_t db_rows;        // actual rows
    size_t db_rows_padded; // padded rows
    size_t db_cols;
    size_t db_rows_poly;   // db_rows / poly_len
};

// Kernel: Compute hint_0 = query_ntt * DB for SimplePIR
// One block per column, accumulates over all row polynomials
// Similar to compute_secondary_hint_kernel but simpler (no blowup_factor)
__global__ void compute_hint_0_sp_kernel(
    uint64_t* __restrict__ hint_out,       // OUT: poly_len x db_cols (will transpose on host)
    const uint16_t* __restrict__ db,        // IN: column-major db_cols x db_rows_padded
    const uint64_t* __restrict__ query_ntt,// IN: db_rows_poly x crt_count x poly_len
    uint64_t* __restrict__ accum_global,   // Global accumulator: db_cols x convd_len
    NTTParams params,
    size_t db_cols,
    size_t db_rows_padded,
    size_t db_rows_poly
) {
    size_t col = blockIdx.x;
    if (col >= db_cols) return;
    
    size_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t convd_len = crt_count * poly_len;
    
    // Thread partitioning for NTT
    uint32_t threads_per_crt = blockDim.x / crt_count;
    uint32_t my_crt = tid / threads_per_crt;
    uint32_t local_tid = tid % threads_per_crt;
    
    // Shared memory for DB element during NTT
    extern __shared__ uint64_t workspace[];
    
    // Global accumulator for this column
    uint64_t* accum = accum_global + col * convd_len;
    
    // Initialize accumulator to zero
    for (size_t i = tid; i < convd_len; i += blockDim.x) {
        accum[i] = 0;
    }
    __syncthreads();
    
    // For each row polynomial
    for (size_t row = 0; row < db_rows_poly; row++) {
        // Load DB elements into workspace and replicate across CRT
        // DB layout: column-major, db[col * db_rows_padded + row * poly_len + z]
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            size_t db_idx = col * db_rows_padded + row * poly_len + z;
            workspace[z] = (uint64_t)db[db_idx];
        }
        __syncthreads();
        
        // Replicate to CRT moduli (reduce by each modulus)
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            uint64_t val = workspace[z];
            workspace[z] = barrett_raw_u64(val, params.barrett_cr[0], params.moduli[0]);
            workspace[poly_len + z] = barrett_raw_u64(val, params.barrett_cr[1], params.moduli[1]);
        }
        __syncthreads();
        
        // Forward NTT on workspace (db_elem)
        if (my_crt < crt_count) {
            ntt_forward_kernel_parallel(workspace + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
        }
        __syncthreads();
        
        // Pointwise multiply with query_ntt[row] and accumulate
        // query_ntt layout: [row][crt][poly_len]
        const uint64_t* query_row = &query_ntt[row * convd_len];
        for (size_t i = tid; i < convd_len; i += blockDim.x) {
            uint32_t crt = i / poly_len;
            uint64_t modulus = params.moduli[crt];
            uint64_t barrett_cr = params.barrett_cr[crt];
            
            uint64_t a = query_row[i];
            uint64_t b = workspace[i];
            uint64_t p = a * b;
            uint64_t reduced = barrett_raw_u64(p, barrett_cr, modulus);
            
            // Accumulate
            accum[i] += reduced;
        }
        __syncthreads();
        
        // Periodic reduction to avoid overflow
        if ((row & 255) == 255 || row == db_rows_poly - 1) {
            for (size_t i = tid; i < convd_len; i += blockDim.x) {
                uint32_t crt = i / poly_len;
                uint64_t modulus = params.moduli[crt];
                uint64_t barrett_cr = params.barrett_cr[crt];
                accum[i] = barrett_raw_u64(accum[i], barrett_cr, modulus);
            }
            __syncthreads();
        }
    }
    
    // Final reduction
    for (size_t i = tid; i < convd_len; i += blockDim.x) {
        uint32_t crt = i / poly_len;
        uint64_t modulus = params.moduli[crt];
        uint64_t barrett_cr = params.barrett_cr[crt];
        accum[i] = barrett_raw_u64(accum[i], barrett_cr, modulus);
    }
    __syncthreads();
    
    // Copy to shared for inverse NTT
    for (size_t i = tid; i < convd_len; i += blockDim.x) {
        workspace[i] = accum[i];
    }
    __syncthreads();
    
    // Inverse NTT
    if (my_crt < crt_count) {
        ntt_inverse_kernel_parallel(workspace + my_crt * poly_len, &params, my_crt, local_tid, threads_per_crt);
    }
    __syncthreads();
    
    // CRT compose and write to output
    // Output layout: hint_out[z * db_cols + col] for transposed access
    // But we'll write hint_out[col * poly_len + z] and transpose on host
    for (size_t z = tid; z < poly_len; z += blockDim.x) {
        uint64_t x = workspace[z];
        uint64_t y = workspace[poly_len + z];
        uint64_t composed = crt_compose_2(x, y, &params);
        
        // Write column-major for later transpose
        hint_out[col * poly_len + z] = composed;
    }
}

extern "C" {

// Initialize SimplePIR offline context
void* ypir_sp_offline_init(
    const uint16_t* db,
    size_t db_rows,
    size_t db_rows_padded,
    size_t db_cols,
    const uint64_t* query_ntt,  // Already in NTT form: db_rows_poly x crt_count x poly_len
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
    SPOfflineContext* ctx = new SPOfflineContext();
    
    ctx->db_rows = db_rows;
    ctx->db_rows_padded = db_rows_padded;
    ctx->db_cols = db_cols;
    ctx->db_rows_poly = db_rows / poly_len;
    
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
    
    // Upload DB (column-major: db_cols x db_rows_padded)
    size_t db_size = db_cols * db_rows_padded * sizeof(uint16_t);
    CUDA_ALLOC_AND_COPY(ctx->d_db, db, db_size);
    
    // Upload query_ntt: db_rows_poly x crt_count x poly_len
    size_t query_size = ctx->db_rows_poly * crt_count * poly_len * sizeof(uint64_t);
    CUDA_ALLOC_AND_COPY(ctx->d_query_ntt, query_ntt, query_size);
    
    // Allocate output: poly_len x db_cols (stored as db_cols x poly_len, transpose on host)
    size_t hint_size = poly_len * db_cols * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_hint_0, hint_size));
    
    // Allocate global accumulator: db_cols x convd_len
    size_t convd_len = crt_count * poly_len;
    size_t accum_size = db_cols * convd_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_accum, accum_size));
    
    return ctx;
}

// Compute hint_0 on GPU
void ypir_sp_compute_hint_0(void* context, uint64_t* hint_out) {
    SPOfflineContext* ctx = (SPOfflineContext*)context;
    if (!ctx) return;
    
    uint32_t poly_len = ctx->ntt_params.poly_len;
    uint32_t crt_count = ctx->ntt_params.crt_count;
    uint32_t convd_len = crt_count * poly_len;
    
    // Launch kernel: one block per column
    int threads = 1024;
    int blocks = ctx->db_cols;
    size_t shared_mem = convd_len * sizeof(uint64_t);
    
    compute_hint_0_sp_kernel<<<blocks, threads, shared_mem>>>(
        ctx->d_hint_0,
        ctx->d_db,
        ctx->d_query_ntt,
        ctx->d_accum,
        ctx->ntt_params,
        ctx->db_cols,
        ctx->db_rows_padded,
        ctx->db_rows_poly
    );
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    
    // Download result (db_cols x poly_len, will transpose on host)
    size_t hint_size = poly_len * ctx->db_cols * sizeof(uint64_t);
    CUDA_ASSERT(cudaMemcpy(hint_out, ctx->d_hint_0, hint_size, cudaMemcpyDeviceToHost));
}

// Free SimplePIR offline context
void ypir_sp_offline_free(void* context) {
    SPOfflineContext* ctx = (SPOfflineContext*)context;
    if (!ctx) return;
    
    CUDA_ASSERT(cudaFree(ctx->d_db));
    CUDA_ASSERT(cudaFree(ctx->d_query_ntt));
    CUDA_ASSERT(cudaFree(ctx->d_hint_0));
    CUDA_ASSERT(cudaFree(ctx->d_accum));
    
    CUDA_ASSERT(cudaFree(ctx->ntt_params.moduli));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.barrett_cr));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.forward_prime_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_table));
    CUDA_ASSERT(cudaFree(ctx->ntt_params.inverse_prime_table));
    
    delete ctx;
}

} // extern "C"
