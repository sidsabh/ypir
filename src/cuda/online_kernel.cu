#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "ntt.cuh"

// Force rebuild
// Packed matrix multiplication kernel adapted from SimplePIR
// Achieves 177 GB/s on GTX 1080

#ifndef TILE_COLS
#define TILE_COLS 128 // columns processed per tile
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024  // threads per block
#endif

static constexpr int BASIS = 8;
static constexpr int COMPRESSION = 4;
static constexpr uint32_t MASK = (1u << BASIS) - 1u;

typedef uint32_t Elem;

#define CUDA_ASSERT(stmt) do { \
    cudaError_t err = (stmt);  \
    if (err != cudaSuccess) {  \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        abort();               \
    }                          \
} while (0)

#define MAX_BATCH_SIZE 16 // most number of queries at a time

// GPU context for online computation
struct OnlineContext {
    Elem* d_db = nullptr;          // Device database (primary DB)
    Elem* d_query = nullptr;       // Device query buffer
    Elem* d_result = nullptr;      // Device result buffer (Step 1 output / a1)

    // Phase 2 & 3 data
    uint16_t* d_smaller_db = nullptr; // Expanded smaller DB
    uint64_t* d_query_ntt = nullptr;  // Secondary query in NTT form
    uint64_t* d_hint_acc = nullptr;   // Accumulator for secondary hint (NTT domain)
    uint64_t* d_hint = nullptr;       // Final secondary hint

    // Phase 4 data
    uint64_t* d_query_q2 = nullptr;   // Query for response generation
    uint64_t* d_response = nullptr;   // Final response

    // Parameters
    NTTParams ntt_params;
    size_t pt_bits = 0;
    uint64_t lwe_modulus;
    uint64_t lwe_q_prime;
    size_t special_offs;
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
};

// main idea: each warp computes K dot products
template<int K>
__global__ void matMulVecPackedWarpSpanTileK(Elem * __restrict__ out,
                                              const Elem * __restrict__ a,
                                              const Elem * __restrict__ b,
                                              size_t aRows, size_t aCols, 
                                              size_t numBatches,
                                              size_t startRow, size_t numRows)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const size_t compressedCols = aCols/COMPRESSION;

    // Base row for this warp's K-pack
    const size_t warpPackBaseLocal = (size_t)blockIdx.x * (size_t)(warpsPerBlock * K)
                                     + (size_t)warpId * (size_t)K;
    if (warpPackBaseLocal >= numRows) return;

    // Shared cache for current b tile (COMPRESSION=4)
    extern __shared__ Elem s_b[];
    Elem* s_b0 = s_b + 0 * TILE_COLS * numBatches;
    Elem* s_b1 = s_b + 1 * TILE_COLS * numBatches;
    Elem* s_b2 = s_b + 2 * TILE_COLS * numBatches;
    Elem* s_b3 = s_b + 3 * TILE_COLS * numBatches;

    // K per-thread accumulators
    unsigned long long acc[K][MAX_BATCH_SIZE];
    #pragma unroll
    for (int r=0; r<K; ++r)
    {
        for (int batch=0; batch < numBatches; batch++)
        {
            acc[r][batch] = 0ull;
        }
    }

    // Walk columns in tiles (to fit b in shared)
    for (size_t tileBase = 0; tileBase < compressedCols; tileBase += TILE_COLS) {
        const size_t tileCols = min((size_t)TILE_COLS, compressedCols - tileBase);

        // Cooperative, coalesced load of b tile into shared
        for (size_t c = threadIdx.x; c < tileCols; c += blockDim.x) {
            const size_t j = tileBase + c;
            for (int batch = 0; batch < numBatches; batch++)
            {
                s_b0[c + batch * TILE_COLS] = __ldg(&b[j * COMPRESSION + 0 + aCols*batch]);
                s_b1[c + batch * TILE_COLS] = __ldg(&b[j * COMPRESSION + 1 + aCols*batch]);
                s_b2[c + batch * TILE_COLS] = __ldg(&b[j * COMPRESSION + 2 + aCols*batch]);
                s_b3[c + batch * TILE_COLS] = __ldg(&b[j * COMPRESSION + 3 + aCols*batch]);
            }
        }
        __syncthreads();

        // Each warp processes K rows using this same b tile
        #pragma unroll
        for (int r=0; r<K; ++r) {
            const size_t rowLocal = warpPackBaseLocal + (size_t)r;
            if (rowLocal >= numRows) break;
            const size_t row       = startRow + rowLocal;
            const size_t rowBaseA  = (row * compressedCols) + tileBase;

            for (size_t c = lane; c < tileCols; c += 32) {
                Elem db = __ldg(&a[rowBaseA + c]);
                Elem v0 =  db               & MASK;
                Elem v1 = (db >> BASIS)     & MASK;
                Elem v2 = (db >> (2*BASIS)) & MASK;
                Elem v3 = (db >> (3*BASIS)) & MASK;
                for (int batch = 0; batch < numBatches; batch++)
                {
                    acc[r][batch] += (unsigned long long)v0 * (unsigned long long)s_b0[c + TILE_COLS *batch];
                    acc[r][batch] += (unsigned long long)v1 * (unsigned long long)s_b1[c + TILE_COLS *batch];
                    acc[r][batch] += (unsigned long long)v2 * (unsigned long long)s_b2[c + TILE_COLS *batch];
                    acc[r][batch] += (unsigned long long)v3 * (unsigned long long)s_b3[c + TILE_COLS *batch];
                }
            }
        }
        __syncthreads();
    }

    // Reduce and write K results
    #pragma unroll
    for (int r=0; r<K; ++r) {
        for (int batch = 0; batch < numBatches; batch++)
        {
            const size_t rowLocal = warpPackBaseLocal + (size_t)r;
            if (rowLocal >= numRows) break;
            unsigned long long v = acc[r][batch];
            #pragma unroll
            for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0) out[startRow + rowLocal + numRows*batch] = (Elem)v;
        }
    }
}

// Kernel to rescale intermediate results and update smaller DB
__global__ void rescale_and_expand_kernel(
    const Elem* __restrict__ intermediate,
    uint16_t* __restrict__ smaller_db,
    size_t db_cols,
    uint64_t lwe_modulus,
    uint64_t lwe_q_prime,
    int pt_bits,
    size_t special_offs,
    size_t blowup_factor_ceil,
    size_t out_rows
) 
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= db_cols) return;

    // Rescale
    uint64_t val = (uint64_t)intermediate[idx];
    // rescale: round(val * q_prime / modulus)
    // We use integer arithmetic: (val * q_prime + modulus/2) / modulus
    // Since modulus is large, we might need 128-bit arithmetic or float
    // Using double for simplicity as in Rust code (it uses f64)
    
    double d_val = (double)val;
    double d_mod = (lwe_modulus == 0) ? 18446744073709551616.0 : (double)lwe_modulus;
    double d_qp = (double)lwe_q_prime;
    
    uint64_t rescaled = (uint64_t)((d_val * d_qp) / d_mod + 0.5);
    
    // Expand into smaller_db
    // smaller_db is column-major? 
    // Rust: smaller_db_mut[out_idx] = val_part
    // out_idx = (special_offs + m) * db_cols + j
    // Here idx corresponds to j (column index in intermediate chunk)
    // Wait, intermediate is a chunk of size db_cols?
    // Yes, num_intermediate should be db_cols.
    
    size_t j = idx; // column index
    
    for (size_t m = 0; m < blowup_factor_ceil; m++) {
        size_t out_idx = (special_offs + m) * db_cols + j;
        
        uint16_t val_part = (rescaled >> (m * pt_bits)) & ((1 << pt_bits) - 1);
        smaller_db[out_idx] = val_part;
    }
}

// Secondary hint kernel matching multiply_with_db_ring
// One block per "column" (actually a row in range [special_offs, special_offs + blowup_factor_ceil))
__global__ void compute_secondary_hint_kernel(
    uint64_t* __restrict__ hint_out,
    const uint16_t* __restrict__ smaller_db,
    const uint64_t* __restrict__ query_ntt,
    uint64_t* __restrict__ sum_global,  // Global memory accumulator
    NTTParams params,
    size_t db_cols,
    size_t db_rows_poly,
    size_t special_offs,
    size_t blowup_factor_ceil
) 
{
    // Each block processes one "column" (row in [special_offs, special_offs + blowup_factor_ceil))
    size_t m = blockIdx.x;
    if (m >= blowup_factor_ceil) return;

    size_t col = special_offs + m; // The "column" index we're processing
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
        // Load DB element as polynomial into workspace and replicate across CRT
        // DB layout: db[col * db_rows + row * poly_len + z]
        // where db_rows = db_rows_poly * poly_len
        for (size_t z = tid; z < poly_len; z += blockDim.x) {
            size_t db_idx = col * (db_rows_poly * poly_len) + row * poly_len + z;
            workspace[z] = (uint64_t)smaller_db[db_idx];
        }
        __syncthreads();

        // Replicate across CRT moduli
        for (size_t crt = 1; crt < crt_count; crt++) {
            for (size_t i = tid; i < poly_len; i += blockDim.x) {
                workspace[crt * poly_len + i] = workspace[i];
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
__global__ void compute_response_kernel(
    uint64_t* __restrict__ response_out,
    const uint16_t* __restrict__ smaller_db,
    const uint64_t* __restrict__ query_packed,
    size_t db_cols,
    size_t out_rows,
    NTTParams params
) 
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_rows) return;

    // Two separate accumulators for low and high 32 bits
    uint64_t sum_lo = 0;
    uint64_t sum_hi = 0;

    // Dot product over cols
    for (size_t col = 0; col < db_cols; col++) {
        uint64_t q_val = query_packed[col];
        // Row-major access: smaller_db[row * db_cols + col]
        uint64_t db_val = (uint64_t)smaller_db[row * db_cols + col];

        // Split query into low and high 32 bits
        uint64_t q_lo = q_val & 0xFFFFFFFF;
        uint64_t q_hi = q_val >> 32;

        // Multiply and accumulate (wrapping)
        sum_lo += q_lo * db_val;
        sum_hi += q_hi * db_val;
    }

    // Barrett reduce each sum
    uint64_t lo = barrett_raw_u64(sum_lo, params.barrett_cr[0], params.moduli[0]);
    uint64_t hi = barrett_raw_u64(sum_hi, params.barrett_cr[1], params.moduli[1]);

    // Compose and reduce
    uint64_t composed = crt_compose_2(lo, hi, &params);
    uint64_t result = barrett_raw_u64(composed, params.barrett_cr_1_modulus, params.modulus);

    response_out[row] = result;
}

extern "C" {

// Initialize online computation context
void* ypir_online_init(const Elem* db, size_t db_rows, size_t db_cols,
                       const uint64_t* pseudorandom_query1,
                       const uint16_t* smaller_db, size_t smaller_db_rows)
{
    OnlineContext* ctx = new OnlineContext();

    ctx->db_rows = db_rows;
    ctx->db_cols = db_cols;
    ctx->smaller_db_rows = smaller_db_rows;
    ctx->smaller_db_cols = db_rows;

    // Allocate device memory for Phase 1
    const size_t db_bytes = db_rows * db_cols * (sizeof(Elem)/COMPRESSION);
    const size_t query_bytes = db_cols* MAX_BATCH_SIZE * sizeof(Elem);
    const size_t result_bytes = db_rows * MAX_BATCH_SIZE * sizeof(Elem);

    CUDA_ASSERT(cudaMalloc(&ctx->d_db, db_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_query, query_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_result, result_bytes));

    // Upload database
    CUDA_ASSERT(cudaMemcpy(ctx->d_db, db, db_bytes, cudaMemcpyHostToDevice));

    // Upload pseudorandom query (*2 because CRT )
    size_t query_ntt_size = db_rows*2 * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_query_ntt, query_ntt_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_query_ntt, pseudorandom_query1, query_ntt_size, cudaMemcpyHostToDevice));

    // Allocate and upload smaller_db for Phase 2/3/4
    const size_t smaller_db_bytes = smaller_db_rows * ctx->smaller_db_cols * sizeof(uint16_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_smaller_db, smaller_db_bytes));
    CUDA_ASSERT(cudaMemcpy(ctx->d_smaller_db, smaller_db, smaller_db_bytes, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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
    size_t special_offs,
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
    ctx->special_offs = special_offs;
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

    CUDA_ASSERT(cudaMalloc(&ctx->d_y_constants, y_constants_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_y_constants, y_constants, y_constants_size, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaMalloc(&ctx->d_prepacked_lwe, prepacked_lwe_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_prepacked_lwe, prepacked_lwe, prepacked_lwe_size, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaMalloc(&ctx->d_precomp_res, precomp_res_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_precomp_res, precomp_res, precomp_res_size, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaMalloc(&ctx->d_precomp_vals, precomp_vals_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_precomp_vals, precomp_vals, precomp_vals_size, cudaMemcpyHostToDevice));
    
    CUDA_ASSERT(cudaMalloc(&ctx->d_precomp_tables, precomp_tables_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_precomp_tables, precomp_tables, precomp_tables_size, cudaMemcpyHostToDevice));

    CUDA_ASSERT(cudaMalloc(&ctx->d_fake_pack_pub_params, fake_pack_pub_params_size));
    CUDA_ASSERT(cudaMemcpy(ctx->d_fake_pack_pub_params, fake_pack_pub_params, fake_pack_pub_params_size, cudaMemcpyHostToDevice));
}

// Full batch execution: Step 1 -> Loop(Step 2 -> Step 3 -> Step 4)
int ypir_online_compute_full_batch(
    void* context,
    const Elem* query,             // Step 1 input (batch)
    const uint64_t* query_q2_batch,// Step 4 input (batch)
    size_t batch_size,
    uint64_t* all_hints_out,       // Output: batch_size * hint_size
    uint64_t* all_responses_out   // Output: batch_size * response_size
) 
{
    OnlineContext* ctx = (OnlineContext*)context;

    // --- Step 1: SimplePIR ---
    // Upload query
    size_t query_bytes = ctx->db_cols * batch_size * sizeof(Elem);
    CUDA_ASSERT(cudaMemcpy(ctx->d_query, query, query_bytes, cudaMemcpyHostToDevice));

    // Launch Step 1 kernel
    const int K = 4;
    const int threads = BLOCK_SIZE;
    const int warps_per_block = threads / 32;
    const int blocks_step1 = (ctx->db_rows + warps_per_block * K - 1) / (warps_per_block * K);
    const size_t shared_mem_step1 = batch_size * COMPRESSION * TILE_COLS * sizeof(Elem);

    matMulVecPackedWarpSpanTileK<K><<<blocks_step1, threads, shared_mem_step1>>>(
        ctx->d_result,
        ctx->d_db,
        ctx->d_query,
        ctx->db_rows,
        ctx->db_cols,
        batch_size,
        0,
        ctx->db_rows
    );
    CUDA_ASSERT(cudaGetLastError());
    // No sync needed here, next kernels will wait in stream

    // --- Prepare for Loop ---
    
    // Step 3 setup
    size_t poly_len = ctx->ntt_params.poly_len;
    size_t crt_count = ctx->ntt_params.crt_count;
    size_t convd_len = crt_count * poly_len;
    
    size_t hint_size = poly_len * ctx->blowup_factor_ceil * sizeof(uint64_t);
    if (!ctx->d_hint) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_hint, hint_size));
    }
    size_t sum_size = ctx->blowup_factor_ceil * convd_len * sizeof(uint64_t);
    if (!ctx->d_hint_acc) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_hint_acc, sum_size));
    }
    size_t shared_mem_step3 = convd_len * sizeof(uint64_t);

    // Step 4 setup
    size_t query_q2_size = ctx->smaller_db_cols * sizeof(uint64_t); // per batch item
    if (!ctx->d_query_q2) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_query_q2, query_q2_size));
    }
    size_t response_size = ctx->smaller_db_rows * sizeof(uint64_t);
    if (!ctx->d_response) {
        CUDA_ASSERT(cudaMalloc(&ctx->d_response, response_size));
    }

    // Allocate device buffers for full batch output to avoid host sync inside loop
    uint64_t *d_all_hints, *d_all_responses;
    CUDA_ASSERT(cudaMalloc(&d_all_hints, batch_size * hint_size));
    CUDA_ASSERT(cudaMalloc(&d_all_responses, batch_size * response_size));

    // Upload all query_q2 data at once? No, d_query_q2 is reused. 
    // We can upload it all to a temp buffer or copy chunk by chunk.
    // Copying chunk by chunk involves host->device copy inside loop, which is async but might stall if queue fills.
    // Better to upload all query_q2_batch to a device buffer.
    uint64_t* d_query_q2_batch;
    CUDA_ASSERT(cudaMalloc(&d_query_q2_batch, batch_size * query_q2_size));
    CUDA_ASSERT(cudaMemcpy(d_query_q2_batch, query_q2_batch, batch_size * query_q2_size, cudaMemcpyHostToDevice));

    // --- Loop over batch ---
    for (size_t i = 0; i < batch_size; i++) {
        // 1. Step 2: Update smaller DB
        // Input: ctx->d_result + i * db_cols
        // Output: ctx->d_smaller_db (reused)
        
        Elem* d_intermediate = ctx->d_result + i * ctx->smaller_db_cols;
        
        int threads_step2 = 256;
        int blocks_step2 = (ctx->smaller_db_cols + threads_step2 - 1) / threads_step2;

        rescale_and_expand_kernel<<<blocks_step2, threads_step2>>>(
            d_intermediate,
            ctx->d_smaller_db,
            ctx->smaller_db_cols,
            ctx->lwe_modulus,
            ctx->lwe_q_prime,
            (int)ctx->pt_bits,
            ctx->special_offs,
            ctx->blowup_factor_ceil,
            ctx->smaller_db_rows
        );
        CUDA_ASSERT(cudaGetLastError());

        // 2. Step 3: Compute Secondary Hint
        // Input: ctx->d_smaller_db, ctx->d_query_ntt
        // Output: ctx->d_hint
        
        compute_secondary_hint_kernel<<<ctx->blowup_factor_ceil, 256, shared_mem_step3>>>(
            ctx->d_hint,
            ctx->d_smaller_db,
            ctx->d_query_ntt,
            ctx->d_hint_acc,
            ctx->ntt_params,
            ctx->db_cols,
            ctx->smaller_db_cols / ctx->ntt_params.poly_len,
            ctx->special_offs,
            ctx->blowup_factor_ceil
        );
        CUDA_ASSERT(cudaGetLastError());

        // Copy result to full batch buffer
        CUDA_ASSERT(cudaMemcpyAsync(d_all_hints + i * (hint_size / sizeof(uint64_t)), 
                                    ctx->d_hint, hint_size, cudaMemcpyDeviceToDevice));

        // 3. Step 4: Compute Response
        // Input: ctx->d_smaller_db, d_query_q2_batch + i * ...
        // Output: ctx->d_response
        
        uint64_t* d_q2_current = d_query_q2_batch + i * ctx->smaller_db_cols;
        
        int threads_step4 = 256;
        int blocks_step4 = (ctx->smaller_db_rows + threads_step4 - 1) / threads_step4;

        compute_response_kernel<<<blocks_step4, threads_step4>>>(
            ctx->d_response,
            ctx->d_smaller_db,
            d_q2_current,
            ctx->smaller_db_cols,
            ctx->smaller_db_rows,
            ctx->ntt_params
        );
        CUDA_ASSERT(cudaGetLastError());

        // Copy result to full batch buffer
        CUDA_ASSERT(cudaMemcpyAsync(d_all_responses + i * (response_size / sizeof(uint64_t)), 
                                    ctx->d_response, response_size, cudaMemcpyDeviceToDevice));
    }

    // --- Download Results ---
    CUDA_ASSERT(cudaMemcpy(all_hints_out, d_all_hints, batch_size * hint_size, cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(all_responses_out, d_all_responses, batch_size * response_size, cudaMemcpyDeviceToHost));

    // Cleanup temp buffers
    CUDA_ASSERT(cudaFree(d_all_hints));
    CUDA_ASSERT(cudaFree(d_all_responses));
    CUDA_ASSERT(cudaFree(d_query_q2_batch));

    return 0;
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

    delete ctx;
}
} // extern "C"
