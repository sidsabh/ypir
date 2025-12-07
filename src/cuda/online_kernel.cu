#include <cuda_runtime.h>
#include <cstdio>
#include "ntt.cuh"

// Packed matrix multiplication kernel adapted from SimplePIR
// Achieves 177 GB/s on GTX 1080

#ifndef TILE_COLS
#define TILE_COLS 2048  // columns processed per tile
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

// GPU context for online computation
struct OnlineContext {
    Elem* d_db;          // Device database
    Elem* d_query;       // Device query buffer
    Elem* d_result;      // Device result buffer (Step 1 output / a1)
    
    // Step 2 buffers
    Elem* d_a1_packed;   // Transformed a1
    Elem* d_A2t;         // Second layer query matrix (pseudorandom_query_1)
    Elem* d_h1;          // Intermediate result h1
    Elem* d_q2;          // Second layer query vector
    Elem* d_h2;          // Result h2
    
    size_t db_rows;
    size_t db_cols;
    size_t max_batch_size;
    
    // Step 2 dimensions
    size_t A2t_rows;
    size_t A2t_cols;
};

// K rows per warp kernel (from SimplePIR - achieves 177 GB/s)
template<int K>
__global__ void matMulVecPackedWarpSpanTileK(Elem * __restrict__ out,
                                              const Elem * __restrict__ a,
                                              const Elem * __restrict__ b,
                                              size_t aRows, size_t aCols,
                                              size_t startRow, size_t numRows)
{
    const int lane   = threadIdx.x & 31;
    const int warpId = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    // Base row for this warp's K-pack
    const size_t warpPackBaseLocal = (size_t)blockIdx.x * (size_t)(warpsPerBlock * K)
                                     + (size_t)warpId * (size_t)K;
    if (warpPackBaseLocal >= numRows) return;

    // Shared cache for current b tile (COMPRESSION=4)
    extern __shared__ Elem s_b[];
    Elem* s_b0 = s_b + 0 * TILE_COLS;
    Elem* s_b1 = s_b + 1 * TILE_COLS;
    Elem* s_b2 = s_b + 2 * TILE_COLS;
    Elem* s_b3 = s_b + 3 * TILE_COLS;

    // K per-thread accumulators
    unsigned long long acc[K];
    #pragma unroll
    for (int r=0; r<K; ++r) acc[r] = 0ull;

    // Walk columns in tiles (to fit b in shared)
    for (size_t tileBase = 0; tileBase < aCols; tileBase += TILE_COLS) {
        const size_t tileCols = min((size_t)TILE_COLS, aCols - tileBase);

        // Cooperative, coalesced load of b tile into shared
        for (size_t c = threadIdx.x; c < tileCols; c += blockDim.x) {
            const size_t j = tileBase + c;
            s_b0[c] = __ldg(&b[j * COMPRESSION + 0]);
            s_b1[c] = __ldg(&b[j * COMPRESSION + 1]);
            s_b2[c] = __ldg(&b[j * COMPRESSION + 2]);
            s_b3[c] = __ldg(&b[j * COMPRESSION + 3]);
        }
        __syncthreads();

        // Each warp processes K rows using this same b tile
        #pragma unroll
        for (int r=0; r<K; ++r) {
            const size_t rowLocal = warpPackBaseLocal + (size_t)r;
            if (rowLocal >= numRows) break;
            const size_t row       = startRow + rowLocal;
            const size_t rowBaseA  = (row * aCols) + tileBase;

            for (size_t c = lane; c < tileCols; c += 32) {
                Elem db = __ldg(&a[rowBaseA + c]);
                Elem v0 =  db               & MASK;
                Elem v1 = (db >> BASIS)     & MASK;
                Elem v2 = (db >> (2*BASIS)) & MASK;
                Elem v3 = (db >> (3*BASIS)) & MASK;
                acc[r] += (unsigned long long)v0 * (unsigned long long)s_b0[c];
                acc[r] += (unsigned long long)v1 * (unsigned long long)s_b1[c];
                acc[r] += (unsigned long long)v2 * (unsigned long long)s_b2[c];
                acc[r] += (unsigned long long)v3 * (unsigned long long)s_b3[c];
            }
        }
        __syncthreads();
    }

    // Reduce and write K results
    #pragma unroll
    for (int r=0; r<K; ++r) {
        const size_t rowLocal = warpPackBaseLocal + (size_t)r;
        if (rowLocal >= numRows) break;
        unsigned long long v = acc[r];
        #pragma unroll
        for (int off=16; off>0; off>>=1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) out[startRow + rowLocal] = (Elem)v;
    }
}

// Byte-for-byte equivalent of CPU transform
__global__ void transformKernel_safe(const Elem* __restrict__ m,
                                     uint32_t R, uint32_t C,
                                     uint32_t mod_p, uint32_t delta, uint32_t concat,
                                     uint32_t basis, uint32_t d,
                                     Elem* __restrict__ n,
                                     uint32_t nRows, uint32_t nCols)
{
    const uint32_t outIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total  = nRows * nCols;
    if (outIdx >= total) return;

    const uint32_t r    = outIdx / nCols;
    const uint32_t cIdx = outIdx % nCols;

    const uint32_t tmp     = r / delta;
    const uint32_t f       = r % delta;
    const uint32_t i       = tmp % C;
    const uint32_t jstripe = tmp / C;
    const uint32_t stripe  = jstripe % concat;

    uint64_t packed = 0ull;

    for (uint32_t t = 0; t < d; ++t) {
        uint32_t c = cIdx * d + t;
        if (c >= (R / concat)) break;

        uint32_t j = c * concat + stripe;
        if (j >= R) continue;

        uint64_t val = (uint64_t)m[(size_t)j * (size_t)C + (size_t)i];

        for (uint32_t step = 0; step < f; ++step) {
            val /= (uint64_t)mod_p;
        }
        uint32_t digit = (uint32_t)(val % (uint64_t)mod_p);

        packed |= (uint64_t)digit << (basis * t);
    }

    n[(size_t)r * (size_t)nCols + (size_t)cIdx] = (Elem)packed;
}

// C = A × B^T; identical indexing/loop order to CPU gemm
// Adapted for COMPRESSION=4
__global__ void gemmPackedKernel_rowmajor(Elem* __restrict__ out,
                                          const Elem* __restrict__ A,
                                          const Elem* __restrict__ B,
                                          uint32_t nRows, uint32_t nCols,
                                          uint32_t B_r,   uint32_t B_c)
{
    const uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; // row in A/out
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; // row in B (col of out)
    if (i >= nRows || j >= B_r) return;

    uint64_t acc = 0ull;
    for (uint32_t k = 0; k < nCols; ++k) {
        Elem a  = __ldg(&A[(size_t)i * (size_t)nCols + (size_t)k]);

        Elem b0 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 0]);
        Elem b1 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 1]);
        Elem b2 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 2]);
        Elem b3 = __ldg(&B[(size_t)j * (size_t)B_c + (size_t)k * COMPRESSION + 3]);

        Elem v0 =  a               & MASK;
        Elem v1 = (a >> BASIS)     & MASK;
        Elem v2 = (a >> (2*BASIS)) & MASK;
        Elem v3 = (a >> (3*BASIS)) & MASK;

        acc += (uint64_t)v0 * b0 + (uint64_t)v1 * b1 + (uint64_t)v2 * b2 + (uint64_t)v3 * b3;
    }
    out[(size_t)i * (size_t)B_r + (size_t)j] = (Elem)acc;
}

extern "C" {

// Initialize online computation context
void* ypir_online_init(const Elem* db, size_t db_rows, size_t db_cols, size_t max_batch_size,
                       const Elem* A2t, size_t A2t_rows, size_t A2t_cols)
{
    OnlineContext* ctx = new OnlineContext();
    
    ctx->db_rows = db_rows;
    ctx->db_cols = db_cols;
    ctx->max_batch_size = max_batch_size;
    ctx->A2t_rows = A2t_rows;
    ctx->A2t_cols = A2t_cols;
    
    // Allocate device memory
    const size_t db_bytes = db_rows * db_cols * sizeof(Elem);
    const size_t query_bytes = db_cols * COMPRESSION * max_batch_size * sizeof(Elem);
    const size_t result_bytes = db_rows * max_batch_size * sizeof(Elem); // a1
    
    CUDA_ASSERT(cudaMalloc(&ctx->d_db, db_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_query, query_bytes));
    CUDA_ASSERT(cudaMalloc(&ctx->d_result, result_bytes));
    
    // Upload database
    CUDA_ASSERT(cudaMemcpy(ctx->d_db, db, db_bytes, cudaMemcpyHostToDevice));
    
    // Step 2 allocations
    // Calculate dimensions for a1_packed
    // From double_pir_cuda.cu:
    // R = db_rows, C = 1, d = 3 (but we use 4?), basis = 8
    // nRows = C * delta * X
    // nCols = (Cmax + d - 1) / d
    // We need to know delta and X. 
    // In YPIR:
    // blowup_factor corresponds to delta?
    // pt_bits corresponds to basis (8)
    // We need to pass these params or infer them.
    // Assuming standard YPIR params: 
    // blowup_factor ~ 2.0?
    // Let's allocate conservatively or pass params.
    // For now, let's assume max sizes based on db_rows.
    
    // Upload A2t
    const size_t A2t_bytes = A2t_rows * A2t_cols * sizeof(Elem);
    CUDA_ASSERT(cudaMalloc(&ctx->d_A2t, A2t_bytes));
    CUDA_ASSERT(cudaMemcpy(ctx->d_A2t, A2t, A2t_bytes, cudaMemcpyHostToDevice));
    
    // Allocate other Step 2 buffers
    // d_a1_packed: needs to hold transformed a1. 
    // Size is roughly db_rows * blowup_factor?
    // Let's allocate 4x db_rows for safety for now (TODO: precise calculation)
    CUDA_ASSERT(cudaMalloc(&ctx->d_a1_packed, db_rows * 4 * sizeof(Elem)));
    
    // d_h1: a1_packed * A2t -> rows(a1_packed) * cols(A2t)
    // rows(a1_packed) ~ db_rows * blowup_factor
    // cols(A2t) = A2t_cols
    CUDA_ASSERT(cudaMalloc(&ctx->d_h1, db_rows * 4 * A2t_cols * sizeof(Elem)));
    
    // d_q2: max(H1_cols, DB_cols) * COMPRESSION
    CUDA_ASSERT(cudaMalloc(&ctx->d_q2, db_cols * COMPRESSION * sizeof(Elem))); // H1_cols?
    
    // d_a2: H1 * q2 -> H1_rows
    // H1_rows? H1 is the "smaller DB".
    // In YPIR, H1 is computed from a1.
    // Wait, H1 in double_pir_cuda is STATIC.
    // In YPIR, H1 is DYNAMIC (it's a1).
    // So d_a2 = a1_packed * q2?
    // In double_pir_cuda:
    // a2 = H1 * q2 (where H1 is static)
    // h2 = a1_packed * q2 (where a1_packed is dynamic)
    // In YPIR server.rs:
    // secondary_hint = smaller_server * pseudorandom_query_1
    // response = smaller_server * q2
    // smaller_server IS a1_packed (dynamic).
    // So we need:
    // 1. secondary_hint = a1_packed * A2t (Matrix * Matrix) -> gemmPacked
    // 2. response = a1_packed * q2 (Matrix * Vector) -> matMulVecPacked
    
    // So we need buffers for:
    // secondary_hint (d_h1)
    // response (d_h2)
    
    CUDA_ASSERT(cudaMalloc(&ctx->d_h2, db_rows * 4 * sizeof(Elem))); // response

    CUDA_ASSERT(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    
    // Print device info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device: %s\n", prop.name);
    printf("  Global Mem: %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf("  Shared Mem per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    
    return (void*)ctx;
}

// Compute Step 1: DB × query (SimplePIR response)
int ypir_online_compute_step1(void* context, const Elem* queries, Elem* results, 
                               size_t num_queries, size_t start_row, size_t num_rows)
{
    OnlineContext* ctx = (OnlineContext*)context;
    
    if (!ctx || start_row >= ctx->db_rows) return -1;
    if (start_row + num_rows > ctx->db_rows) num_rows = ctx->db_rows - start_row;
    
    // Upload queries
    const size_t query_bytes = ctx->db_cols * COMPRESSION * num_queries * sizeof(Elem);
    CUDA_ASSERT(cudaMemcpy(ctx->d_query, queries, query_bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel for each query
    for (size_t q = 0; q < num_queries; ++q) {
        const Elem* d_query_q = ctx->d_query + q * ctx->db_cols * COMPRESSION;
        Elem* d_result_q = ctx->d_result + q * num_rows;
        
        // Kernel configuration (from SimplePIR)
        const int K = 4;
        const size_t minWarpsNeeded = (num_rows + K - 1) / K;
        const int warpsPerBlock = (int)min((size_t)32, minWarpsNeeded);
        dim3 block(warpsPerBlock * 32);
        const size_t rowsPerBlockLogical = (size_t)warpsPerBlock * (size_t)K;
        dim3 grid((int)((num_rows + rowsPerBlockLogical - 1) / rowsPerBlockLogical));
        const size_t shmemBytes = (size_t)TILE_COLS * (size_t)COMPRESSION * sizeof(Elem);
        
        matMulVecPackedWarpSpanTileK<4><<<grid, block, shmemBytes>>>(
            d_result_q, ctx->d_db, d_query_q, ctx->db_rows, ctx->db_cols, start_row, num_rows);
    }
    
    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    
    // Download results
    const size_t result_bytes = num_rows * num_queries * sizeof(Elem);
    CUDA_ASSERT(cudaMemcpy(results, ctx->d_result, result_bytes, cudaMemcpyDeviceToHost));
    
    return 0;
}

// Free online computation context
void ypir_online_free(void* context)
{
    OnlineContext* ctx = (OnlineContext*)context;
    if (!ctx) return;
    
    if (ctx->d_db) CUDA_ASSERT(cudaFree(ctx->d_db));
    if (ctx->d_query) CUDA_ASSERT(cudaFree(ctx->d_query));
    if (ctx->d_result) CUDA_ASSERT(cudaFree(ctx->d_result));
    
    delete ctx;
}

} // extern "C"
