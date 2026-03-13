/// CUDA-accelerated PIR computation.
///
/// Submodules organized by protocol:
/// - `doublepir` — DoublePIR offline hint generation and online computation
/// - `simplepir` — SimplePIR offline hint generation and online computation
/// - `word` — Word-based SimplePIR (Z_{2^64}) offline and online
/// - `inspiring` — InspiRING packing GPU precomputation
/// - `toeplitz` — Toeplitz-based offline hint generation

pub mod doublepir;
pub mod inspiring;
pub mod simplepir;
pub mod toeplitz;
pub mod word;

// Re-export all context types at the cuda module level for convenience.
#[cfg(feature = "cuda")]
pub use doublepir::{OfflineComputeContext, OnlineComputeContext};
#[cfg(feature = "cuda")]
pub use inspiring::InspirPrecompContext;
#[cfg(feature = "cuda")]
pub use simplepir::{SPOfflineContext, SPOnlineContext};
#[cfg(all(feature = "cuda", feature = "toeplitz"))]
pub use toeplitz::ToeplitzContext;
#[cfg(feature = "cuda")]
pub use word::{WordOfflineContext, WordOnlineContext};

/// Flatten spiral-rs NTT tables into contiguous arrays for GPU upload.
#[cfg(feature = "cuda")]
pub(crate) fn flatten_ntt_tables(
    params: &spiral_rs::params::Params,
) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
    let poly_len = params.poly_len;
    let crt_count = params.crt_count;
    let mut forward_table = Vec::with_capacity(crt_count * poly_len);
    let mut forward_prime_table = Vec::with_capacity(crt_count * poly_len);
    let mut inverse_table = Vec::with_capacity(crt_count * poly_len);
    let mut inverse_prime_table = Vec::with_capacity(crt_count * poly_len);
    for i in 0..crt_count {
        forward_table.extend_from_slice(&params.ntt_tables[i][0]);
        forward_prime_table.extend_from_slice(&params.ntt_tables[i][1]);
        inverse_table.extend_from_slice(&params.ntt_tables[i][2]);
        inverse_prime_table.extend_from_slice(&params.ntt_tables[i][3]);
    }
    (forward_table, forward_prime_table, inverse_table, inverse_prime_table)
}
