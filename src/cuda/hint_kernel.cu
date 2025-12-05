#include "ntt.cuh"
#include <cuda_runtime.h>

extern "C" {

// Parameters passed to each kernel
struct HintParams {
    uint32_t db_rows;
    uint32_t db_rows_padded;
    uint32_t db_cols;
    uint32_t n;
    uint32_t max_adds;
    uint32_t convd_len;  // crt_count * poly_len
};

// Main kernel: one block per column
// Matches generate_hint_0_ring_cpu in server.rs
__global__ void compute_hint_0_kernel(
    uint64_t* hint_0,
    const uint8_t* db,
    const uint32_t* v_nega_perm_a,
    const NTTParams* ntt_params,
    const HintParams* params
) {
    const uint32_t col = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t poly_len = ntt_params->poly_len;
    const uint32_t crt_count = ntt_params->crt_count;
    const uint32_t convd_len = params->convd_len;
    const uint32_t n = params->n;

    // Shared memory layout:
    // - tmp_col: convd_len elements (accumulator)
    // - pt_ntt: convd_len elements (NTT workspace)
    extern __shared__ uint64_t shared_mem[];
    uint64_t* tmp_col = shared_mem;  // Size: convd_len
    uint64_t* pt_ntt = &shared_mem[convd_len];  // Size: convd_len

    // Initialize tmp_col to zero
    for (uint32_t i = tid; i < convd_len; i += blockDim.x) {
        tmp_col[i] = 0;
    }
    __syncthreads();

    const uint32_t num_chunks = params->db_rows / n;

    // Main loop: for each outer_row (chunk of n elements)
    for (uint32_t outer_row = 0; outer_row < num_chunks; outer_row++) {
        // Load database column chunk and convert to u32
        // Parallel load into pt_ntt (reusing as temp buffer)
        const uint32_t start_idx = col * params->db_rows_padded + outer_row * n;
        for (uint32_t i = tid; i < n; i += blockDim.x) {
            pt_ntt[i] = (uint32_t)db[start_idx + i];
        }
        __syncthreads();

        // Replicate across CRT moduli for NTT
        for (uint32_t crt = 1; crt < crt_count; crt++) {
            for (uint32_t i = tid; i < poly_len; i += blockDim.x) {
                pt_ntt[crt * poly_len + i] = pt_ntt[i];
            }
        }
        __syncthreads();

        // Forward NTT (PARALLELIZED: threads 0-127 do CRT0, 128-255 do CRT1)
        const uint32_t threads_per_crt = blockDim.x / crt_count;
        const uint32_t my_crt = tid / threads_per_crt;
        const uint32_t local_tid = tid % threads_per_crt;
        
        if (my_crt < crt_count) {
            ntt_forward_kernel_parallel(pt_ntt + my_crt * poly_len, ntt_params, my_crt, local_tid, threads_per_crt);
        }
        __syncthreads();

        // Pointwise multiply: result_ntt = v_nega_perm_a[outer_row] * pt_ntt
        const uint32_t* nega_perm_a_ntt = &v_nega_perm_a[outer_row * convd_len];
        for (uint32_t i = tid; i < convd_len; i += blockDim.x) {
            const uint32_t crt = i / poly_len;
            const uint64_t modulus = ntt_params->moduli[crt];
            const uint64_t barrett_cr = ntt_params->barrett_cr[crt];

            const uint64_t a_val = nega_perm_a_ntt[i];
            const uint64_t b_val = pt_ntt[i];
            const uint64_t prod = a_val * b_val;
            pt_ntt[i] = barrett_raw_u64(prod, barrett_cr, modulus);
        }
        __syncthreads();

        // Accumulate into tmp_col
        for (uint32_t i = tid; i < convd_len; i += blockDim.x) {
            tmp_col[i] += pt_ntt[i];
        }
        __syncthreads();

        // Check if we need to INTT and add to hint_0
        const bool should_process = (outer_row % params->max_adds == params->max_adds - 1) ||
                                   (outer_row == num_chunks - 1);

        if (should_process) {
            // Barrett reduce tmp_col to modulus
            for (uint32_t i = tid; i < convd_len; i += blockDim.x) {
                const uint32_t crt = i / poly_len;
                const uint64_t modulus = ntt_params->moduli[crt];
                const uint64_t barrett_cr = ntt_params->barrett_cr[crt];
                tmp_col[i] = barrett_raw_u64(tmp_col[i], barrett_cr, modulus);
            }
            __syncthreads();

            // Inverse NTT (PARALLELIZED: threads 0-127 do CRT0, 128-255 do CRT1)
            if (my_crt < crt_count) {
                ntt_inverse_kernel_parallel(tmp_col + my_crt * poly_len, ntt_params, my_crt, local_tid, threads_per_crt);
            }
            __syncthreads();

            __syncthreads();

            // CRT reconstruction and add to hint_0
            // Matches Convolution::raw() from convolution.rs
            for (uint32_t i = tid; i < n; i += blockDim.x) {
                // CRT compose from two residues
                const uint64_t x = tmp_col[i];  // First CRT residue
                const uint64_t y = tmp_col[poly_len + i];  // Second CRT residue

                // CRT reconstruction: val = x * mod1_inv_mod0 + y * mod0_inv_mod1 mod Q
                uint64_t composed = crt_compose_2(x, y, ntt_params);

                // Signed reduction (matching Convolution::raw lines 85-107)
                int64_t val = (int64_t)composed;
                const uint64_t Q = ntt_params->modulus;
                const int64_t Q_signed = (int64_t)Q;

                // if val > Q/2: val -= Q
                if (val > Q_signed / 2) {
                    val -= Q_signed;
                }

                // if val < 0: bring to positive range mod 2^32
                if (val < 0) {
                    val += (Q_signed / (1LL << 32)) * (1LL << 32);
                    val += 1LL << 32;
                }

                // Final: val % 2^32
                uint32_t final_val = (uint32_t)(val % (1LL << 32));

                // Add to hint_0
                // CPU uses: hint_0[i * db_cols + col]
                const uint32_t hint_idx = i * params->db_cols + col;
                atomicAdd((unsigned long long*)&hint_0[hint_idx], (unsigned long long)final_val);
                // Apply modulo 2^32 (matching CPU logic)
                hint_0[hint_idx] %= (1ULL << 32);
            }
            __syncthreads();

            // Clear tmp_col for next iteration
            for (uint32_t i = tid; i < convd_len; i += blockDim.x) {
                tmp_col[i] = 0;
            }
            __syncthreads();
        }
    }
}

// GPU context structure to hold pre-uploaded data
struct GPUContext {
    uint8_t* d_db;
    uint32_t* d_v_nega_perm_a;
    uint64_t* d_moduli;
    uint64_t* d_barrett_cr;
    uint64_t* d_forward_table;
    uint64_t* d_forward_prime_table;
    uint64_t* d_inverse_table;
    uint64_t* d_inverse_prime_table;
    NTTParams* d_ntt_params;
    HintParams* d_hint_params;

    uint32_t db_rows;
    uint32_t db_rows_padded;
    uint32_t db_cols;
    uint32_t n;
    uint32_t poly_len;
    uint32_t crt_count;
    uint32_t convd_len;
};

// Initialize GPU context and upload database/parameters
void* init_gpu_context(
    const uint8_t* db,
    const uint32_t* v_nega_perm_a,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    const uint64_t* forward_table,
    const uint64_t* forward_prime_table,
    const uint64_t* inverse_table,
    const uint64_t* inverse_prime_table,
    uint32_t db_rows,
    uint32_t db_rows_padded,
    uint32_t db_cols,
    uint32_t n,
    uint32_t poly_len,
    uint32_t crt_count,
    uint32_t max_adds,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus
) {
    GPUContext* ctx = new GPUContext();

    ctx->db_rows = db_rows;
    ctx->db_rows_padded = db_rows_padded;
    ctx->db_cols = db_cols;
    ctx->n = n;
    ctx->poly_len = poly_len;
    ctx->crt_count = crt_count;
    ctx->convd_len = crt_count * poly_len;

    const size_t db_size = db_rows_padded * db_cols * sizeof(uint8_t);
    const size_t nega_size = (db_rows / n) * crt_count * poly_len * sizeof(uint32_t);
    const size_t table_size = crt_count * poly_len * sizeof(uint64_t);

    // Allocate device memory
    cudaMalloc(&ctx->d_db, db_size);
    cudaMalloc(&ctx->d_v_nega_perm_a, nega_size);
    cudaMalloc(&ctx->d_moduli, crt_count * sizeof(uint64_t));
    cudaMalloc(&ctx->d_barrett_cr, crt_count * sizeof(uint64_t));
    cudaMalloc(&ctx->d_forward_table, table_size);
    cudaMalloc(&ctx->d_forward_prime_table, table_size);
    cudaMalloc(&ctx->d_inverse_table, table_size);
    cudaMalloc(&ctx->d_inverse_prime_table, table_size);
    cudaMalloc(&ctx->d_ntt_params, sizeof(NTTParams));
    cudaMalloc(&ctx->d_hint_params, sizeof(HintParams));

    // Copy data to device
    cudaMemcpy(ctx->d_db, db, db_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_v_nega_perm_a, v_nega_perm_a, nega_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_moduli, moduli, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_barrett_cr, barrett_cr, crt_count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_forward_table, forward_table, table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_forward_prime_table, forward_prime_table, table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_inverse_table, inverse_table, table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_inverse_prime_table, inverse_prime_table, table_size, cudaMemcpyHostToDevice);

    // Compute modulus = moduli[0] * moduli[1]
    uint64_t modulus = moduli[0] * moduli[1];

    // Setup NTT params
    NTTParams h_ntt_params;
    h_ntt_params.poly_len = poly_len;
    h_ntt_params.log2_poly_len = __builtin_ctz(poly_len);
    h_ntt_params.crt_count = crt_count;
    h_ntt_params.moduli = ctx->d_moduli;
    h_ntt_params.barrett_cr = ctx->d_barrett_cr;
    h_ntt_params.forward_table = ctx->d_forward_table;
    h_ntt_params.forward_prime_table = ctx->d_forward_prime_table;
    h_ntt_params.inverse_table = ctx->d_inverse_table;
    h_ntt_params.inverse_prime_table = ctx->d_inverse_prime_table;
    h_ntt_params.mod0_inv_mod1 = mod0_inv_mod1;
    h_ntt_params.mod1_inv_mod0 = mod1_inv_mod0;
    h_ntt_params.barrett_cr_0_modulus = barrett_cr_0_modulus;
    h_ntt_params.barrett_cr_1_modulus = barrett_cr_1_modulus;
    h_ntt_params.modulus = modulus;
    cudaMemcpy(ctx->d_ntt_params, &h_ntt_params, sizeof(NTTParams), cudaMemcpyHostToDevice);

    // Setup hint params
    HintParams h_hint_params;
    h_hint_params.db_rows = db_rows;
    h_hint_params.db_rows_padded = db_rows_padded;
    h_hint_params.db_cols = db_cols;
    h_hint_params.n = n;
    h_hint_params.max_adds = max_adds;
    h_hint_params.convd_len = ctx->convd_len;
    cudaMemcpy(ctx->d_hint_params, &h_hint_params, sizeof(HintParams), cudaMemcpyHostToDevice);

    return ctx;
}

// Compute hint_0 using pre-initialized GPU context
int compute_hint_0_cuda(void* context, uint64_t* hint_0) {
    GPUContext* ctx = (GPUContext*)context;

    uint64_t* d_hint_0;
    const size_t hint_size = ctx->n * ctx->db_cols * sizeof(uint64_t);

    cudaMalloc(&d_hint_0, hint_size);
    cudaMemset(d_hint_0, 0, hint_size);

    // Launch kernel: one block per column
    const int threads = 256;
    const int blocks = ctx->db_cols;
    const size_t shared_mem_size = 2 * ctx->convd_len * sizeof(uint64_t);

    compute_hint_0_kernel<<<blocks, threads, shared_mem_size>>>(
        d_hint_0,
        ctx->d_db,
        ctx->d_v_nega_perm_a,
        ctx->d_ntt_params,
        ctx->d_hint_params
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }

    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }

    // Copy result back
    cudaMemcpy(hint_0, d_hint_0, hint_size, cudaMemcpyDeviceToHost);
    cudaFree(d_hint_0);

    return 0;
}

// Free GPU context
void free_gpu_context(void* context) {
    GPUContext* ctx = (GPUContext*)context;

    cudaFree(ctx->d_db);
    cudaFree(ctx->d_v_nega_perm_a);
    cudaFree(ctx->d_moduli);
    cudaFree(ctx->d_barrett_cr);
    cudaFree(ctx->d_forward_table);
    cudaFree(ctx->d_forward_prime_table);
    cudaFree(ctx->d_inverse_table);
    cudaFree(ctx->d_inverse_prime_table);
    cudaFree(ctx->d_ntt_params);
    cudaFree(ctx->d_hint_params);

    delete ctx;
}

} // extern "C"
