/// Test Toeplitz matrix-based hint_0 computation vs NTT-based CPU version

use std::time::Instant;
use ypir::server::YServer;
use ypir::lwe::LWEParams;
use ypir::convolution::Convolution;
use ypir::util::test_params;
use rand::Rng;

fn main() {
    println!("Setting up parameters for 1GB database...");
    let params = test_params();

    println!("Generating random database...");
    type T = u8;
    let pt_iter = std::iter::repeat_with(|| rand::thread_rng().gen::<T>());
    let server = YServer::<T>::new(&params, pt_iter, true, false, true);

    let lwe_params = LWEParams::default();
    let conv = Convolution::new(lwe_params.n);

    // CPU NTT version (ground truth)
    println!("\n=== CPU NTT-based hint_0 ===");
    let start_cpu = Instant::now();
    let hint_0_cpu = server.generate_hint_0_ring();
    let cpu_time = start_cpu.elapsed();
    println!("CPU time: {:.3}s", cpu_time.as_secs_f64());

    // GPU Toeplitz version
    #[cfg(all(feature = "cuda", feature = "toeplitz"))]
    {
        println!("\n=== GPU Toeplitz-based hint_0 (with init) ===");
        let start_init = Instant::now();
        let toeplitz_ctx = server.init_toeplitz_context(&lwe_params, &conv);
        let init_time = start_init.elapsed();
        println!("Init time (DB upload + matrix build): {:.3}s", init_time.as_secs_f64());

        let start_compute = Instant::now();
        let hint_0_gpu = server.compute_hint_0_with_toeplitz(&toeplitz_ctx);
        let compute_time = start_compute.elapsed();
        println!("Compute time (cuBLAS gemm): {:.3}s", compute_time.as_secs_f64());
        println!("Total GPU time: {:.3}s", (init_time + compute_time).as_secs_f64());

        // Verify results match
        println!("\n=== Verification ===");
        let mut matches = true;
        let mut mismatch_count = 0;
        for i in 0..hint_0_cpu.len().min(100) {
            if hint_0_cpu[i] != hint_0_gpu[i] {
                if mismatch_count < 10 {
                    println!("Mismatch at index {}: CPU={}, GPU={}", i, hint_0_cpu[i], hint_0_gpu[i]);
                }
                matches = false;
                mismatch_count += 1;
            }
        }

        if matches {
            println!("✓ GPU Toeplitz matches CPU NTT");
            println!("Speedup: {:.2}x", cpu_time.as_secs_f64() / compute_time.as_secs_f64());
        } else {
            println!("✗ Mismatch found ({} / {} checked)", mismatch_count, hint_0_cpu.len().min(100));
        }
    }

    #[cfg(not(all(feature = "cuda", feature = "toeplitz")))]
    {
        println!("\n=== Toeplitz feature not enabled ===");
        println!("Build with: cargo run --release --features \"cuda,toeplitz\" --example test_toeplitz");
    }
}
