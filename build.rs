use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Compile C++ matmul
    println!("cargo:rerun-if-changed=src/matmul.cpp");
    cc::Build::new()
        .cpp(true)
        .file("src/matmul.cpp")
        .flag("-O3")
        .flag("-march=native")
        .flag("-std=c++11")
        .compile("matmul");

    // Compile CUDA if feature is enabled
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if cuda_enabled {
        if Path::new("/usr/local/cuda/bin/nvcc").exists() ||
           Path::new("/opt/cuda/bin/nvcc").exists() ||
           Path::new("/usr/bin/nvcc").exists() {
            compile_cuda();
        } else {
            panic!("CUDA feature enabled but nvcc not found. Please install CUDA toolkit or disable the cuda feature.");
        }
    }
}

fn compile_cuda() {
    println!("cargo:rerun-if-changed=src/cuda/hint_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/online_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/ntt.cuh");

    let out_dir = env::var("OUT_DIR").unwrap();
    let lib_path = PathBuf::from(&out_dir).join("libhint_cuda.so");

    // Detect GPU architecture
    let arch = detect_gpu_arch().unwrap_or("sm_61".to_string());
    println!("cargo:warning=Compiling CUDA code for architecture: {}", arch);

    // Check if toeplitz feature is enabled
    let toeplitz_enabled = env::var("CARGO_FEATURE_TOEPLITZ").is_ok();
    let toeplitz_crt_enabled = env::var("CARGO_FEATURE_TOEPLITZ_CRT").is_ok();

    let arch_flag = format!("-arch={}", arch);
    let mut nvcc_args = vec![
        "-O3",
        arch_flag.as_str(),
        "-Xcompiler", "-fPIC",
        "-shared",
        "src/cuda/hint_kernel.cu",
        "src/cuda/online_kernel.cu",
    ];

    // Add toeplitz kernel - check CRT first since it implies toeplitz
    if toeplitz_crt_enabled {
        println!("cargo:rerun-if-changed=src/cuda/toeplitz_kernel_crt.cu");
        nvcc_args.push("src/cuda/toeplitz_kernel_crt.cu");
        nvcc_args.push("-lcublas");
    } else if toeplitz_enabled {
        println!("cargo:rerun-if-changed=src/cuda/toeplitz_kernel.cu");
        nvcc_args.push("src/cuda/toeplitz_kernel.cu");
    }

    nvcc_args.push("-o");
    nvcc_args.push(lib_path.to_str().unwrap());

    let status = Command::new("nvcc")
        .args(&nvcc_args)
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    // Set up linking
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=hint_cuda");

    // Link CUDA runtime
    if Path::new("/usr/local/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    if Path::new("/opt/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");

    // Link cuBLAS only for CRT variant
    let toeplitz_crt_enabled = env::var("CARGO_FEATURE_TOEPLITZ_CRT").is_ok();
    if toeplitz_crt_enabled {
        println!("cargo:rustc-link-lib=cublas");
    }
}

fn detect_gpu_arch() -> Option<String> {
    // Try to detect GPU compute capability
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if output.status.success() {
        let cap = String::from_utf8_lossy(&output.stdout)
            .trim()
            .lines()
            .next()?
            .replace(".", "");
        Some(format!("sm_{}", cap))
    } else {
        None
    }
}
