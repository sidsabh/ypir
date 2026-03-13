/**
 * Tensor Core InspiRING Packing — Header
 *
 * Reformulation: precompute M = Σ_i diag(T_i)·P_i offline, then online is
 * a standard GEMM: result = Y · M.  No expansion, no permutation gathering.
 *
 * M_combined_k[z, z'] = Σ_{i: σ_i(z)=z'} T[it+k, z]   (Term 1)
 *                      + Σ_{i: σ̄_i(z)=z'} T̄[it+k, z]   (Term 2)
 *
 * Online: C = Y_concat · M_all  via byte-decomposed uint8 tensor core GEMM
 *   Y_concat = [y_0 | y_1 | ... | y_{t-1}]  ∈ [B × tN]
 *   M_all ∈ [tN × ρN]   (stacked across gadget k and output o)
 *
 * Then add Term 3: result[c,o,z] += Σ_k T̂_o[k,z] · z_k[c,z]  (pointwise)
 * Then post-process: INTT → CRT compose → add b → modswitch → bitpack
 */

#pragma once

#include <cstdint>
#include <cstddef>

struct NTTParams;  // forward decl from common/ntt.cuh

struct TcPackingContext {
    // Byte-decomposed M_combined (offline precomputed, owned)
    // d_M_bytes[crt][byte_idx] each [K_gemm × N_gemm] uint8, ColumnMajor
    uint8_t* d_M_bytes[2][4];

    // T̂ for Term 3 (condensed uint64, owned copy)
    uint64_t* d_bold_t_hat_condensed;   // [num_outputs × t × poly_len]

    // â for post-process (owned copy)
    uint64_t* d_a_hat;                  // [num_outputs × poly_len]

    // Online scratch (all owned)
    uint8_t* d_A_bytes_buf;             // 8 contiguous byte-slices for A: [2][4][max_B × K_gemm]
    int32_t* d_G;                       // shift accumulator: [max_B × N_gemm] int32
    uint64_t* d_result_crt0;            // CRT0 running total: [max_B × N_gemm] uint64
    uint64_t* d_result_crt1;            // CRT1 running total: [max_B × N_gemm] uint64
    uint64_t* d_scratch;                // post-process scratch: [max_B × num_outputs × 4 × poly_len]

    // Dimensions
    size_t poly_len;                    // N = 2048
    size_t t_exp_left;                  // t = 3
    size_t num_iter;                    // R = N/2 - 1 = 1023
    size_t num_outputs;                 // ρ
    size_t max_batch_size;              // B_max
    size_t K_gemm;                      // t × N
    size_t N_gemm;                      // ρ × N

    // GPU arch: 0 = SIMT (sm<75), 1 = SM75 TC, 2 = SM80+ TC
    int gpu_tier;

    // Post-process params
    size_t response_bytes_per_output;
    uint64_t rlwe_q_prime_1;
    uint64_t rlwe_q_prime_2;

    // Host-side copies of NTT moduli/barrett (device ptrs can't be dereferenced on host)
    uint64_t mod0;
    uint64_t mod1;
    uint64_t barrett_cr0;
    uint64_t barrett_cr1;

    NTTParams ntt_params;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize tensor core packing context.
 * Builds M_combined from bold_t/bold_t_bar + permutation tables on GPU.
 * Takes OWNERSHIP of d_bold_t and d_bold_t_bar — frees them after M build
 * to reduce peak memory (their data is fully absorbed into M).
 * Other input device pointers are borrowed (not freed).
 */
void* tc_packing_init(
    uint64_t* d_bold_t_condensed,
    uint64_t* d_bold_t_bar_condensed,
    const uint64_t* d_bold_t_hat_condensed,
    const uint64_t* d_a_hat,
    const uint32_t* d_tables,
    const uint32_t* d_gen_pows,
    size_t num_iter,
    size_t t_exp_left,
    size_t poly_len,
    size_t num_outputs,
    size_t max_batch_size,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    NTTParams ntt_params,
    int gpu_tier);

/**
 * Run packing for a batch.
 * d_intermediate: Step 1 output (batch × num_outputs × poly_len CRT-condensed u64).
 * d_y_body, d_z_body: per-client condensed keys (on device).
 * d_response_out: output buffer (on device).
 */
void tc_packing_run(
    void* context,
    const uint64_t* d_intermediate,
    const uint64_t* d_y_body_condensed,
    const uint64_t* d_z_body_condensed,
    uint8_t* d_response_out,
    size_t batch_size);

void tc_packing_free(void* context);

#ifdef __cplusplus
}
#endif
