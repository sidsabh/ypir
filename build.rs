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
           Path::new("/usr/bin/nvcc").exists() || 
           Path::new("/opt/apps/cuda/12.2/bin/nvcc").exists() {
            compile_cuda();
        } else {
            panic!("CUDA feature enabled but nvcc not found. Please install CUDA toolkit or disable the cuda feature.");
        }
    }
}

fn compile_cuda() {
    // Track all CUDA source files for rebuild detection
    println!("cargo:rerun-if-changed=src/cuda/doublepir/offline.cu");
    println!("cargo:rerun-if-changed=src/cuda/doublepir/online.cu");
    println!("cargo:rerun-if-changed=src/cuda/simplepir/offline.cu");
    println!("cargo:rerun-if-changed=src/cuda/simplepir/online.cu");
    println!("cargo:rerun-if-changed=src/cuda/word/offline.cu");
    println!("cargo:rerun-if-changed=src/cuda/word/online.cu");
    println!("cargo:rerun-if-changed=src/cuda/inspiring/precomp.cu");
    println!("cargo:rerun-if-changed=src/cuda/inspiring/packing.cuh");
    println!("cargo:rerun-if-changed=src/cuda/inspiring/tc_packing.cu");
    println!("cargo:rerun-if-changed=src/cuda/inspiring/tc_packing.cuh");
    println!("cargo:rerun-if-changed=src/cuda/common/ntt.cuh");

    let out_dir = env::var("OUT_DIR").unwrap();
    let lib_path = PathBuf::from(&out_dir).join("libypir_cuda.so");

    // Detect GPU architecture
    let arch = detect_gpu_arch().unwrap_or("sm_61".to_string());
    println!("cargo:warning=Compiling CUDA code for architecture: {}", arch);

    // Check feature flags
    let cutlass_enabled = env::var("CARGO_FEATURE_CUTLASS").is_ok();
    let toeplitz_enabled = env::var("CARGO_FEATURE_TOEPLITZ").is_ok();
    let toeplitz_crt_enabled = env::var("CARGO_FEATURE_TOEPLITZ_CRT").is_ok();

    let arch_flag = format!("-arch={}", arch);

    // CUTLASS is always needed (doublepir/online.cu uses it for DoublePIR Step 1)
    let cutlass_dir = find_cutlass_dir();
    let cutlass_include = format!("-I{}/include", cutlass_dir);
    let cutlass_tools  = format!("-I{}/tools/util/include", cutlass_dir);

    let mut nvcc_args: Vec<String> = vec![
        "-O3".into(),
        arch_flag.clone(),
        "-Xcompiler".into(), "-fPIC".into(),
        "-shared".into(),
        "-std=c++17".into(),
        "--expt-relaxed-constexpr".into(),
        // Include path for shared headers (ntt.cuh, packing.cuh)
        "-Isrc/cuda".into(),
        cutlass_include,
        cutlass_tools,
        "-lcublas".into(),
    ];

    // Core kernels (always compiled)
    nvcc_args.push("src/cuda/doublepir/offline.cu".into());
    nvcc_args.push("src/cuda/doublepir/online.cu".into());
    nvcc_args.push("src/cuda/simplepir/offline.cu".into());
    nvcc_args.push("src/cuda/word/offline.cu".into());
    nvcc_args.push("src/cuda/word/online.cu".into());
    nvcc_args.push("src/cuda/inspiring/precomp.cu".into());
    nvcc_args.push("src/cuda/inspiring/tc_packing.cu".into());

    if cutlass_enabled {
        // SimplePIR CUTLASS variant
        println!("cargo:rerun-if-changed=src/cuda/simplepir/online_cutlass.cu");
        println!("cargo:warning=CUTLASS feature enabled — using simplepir/online_cutlass.cu");
        nvcc_args.push("src/cuda/simplepir/online_cutlass.cu".into());
    } else {
        nvcc_args.push("src/cuda/simplepir/online.cu".into());
    }

    // Toeplitz kernel (optional) - check CRT first since it implies toeplitz
    if toeplitz_crt_enabled {
        println!("cargo:rerun-if-changed=src/cuda/toeplitz/offline_crt.cu");
        nvcc_args.push("src/cuda/toeplitz/offline_crt.cu".into());
    } else if toeplitz_enabled {
        println!("cargo:rerun-if-changed=src/cuda/toeplitz/offline.cu");
        nvcc_args.push("src/cuda/toeplitz/offline.cu".into());
    }

    nvcc_args.push("-o".into());
    nvcc_args.push(lib_path.to_str().unwrap().into());

    let status = Command::new("nvcc")
        .args(&nvcc_args)
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("CUDA compilation failed");
    }

    // Set up linking
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=ypir_cuda");

    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or("/opt/apps/cuda/12.2".to_string());
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);

    // Runtime path so tests/debugger can load it
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir);

    // Link CUDA runtime
    if Path::new("/usr/local/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    if Path::new("/opt/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=cudart");

    // Link cuBLAS (needed for tensor core GEMM in online_kernel.cu)
    println!("cargo:rustc-link-lib=cublas");
}

/// Find CUTLASS header directory.  Checks (in order):
///   1. CUTLASS_DIR env var
///   2. ../cutlass  (relative to cargo manifest dir)
///   3. $HOME/cutlass
///   4. /usr/local/cutlass
fn find_cutlass_dir() -> String {
    if let Ok(dir) = env::var("CUTLASS_DIR") {
        if Path::new(&dir).join("include/cutlass/cutlass.h").exists() {
            return dir;
        }
        println!("cargo:warning=CUTLASS_DIR={} does not contain include/cutlass/cutlass.h, searching fallbacks", dir);
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let candidates = [
        PathBuf::from(&manifest_dir).join("../cutlass"),
        dirs_or_home().join("cutlass"),
        PathBuf::from("/usr/local/cutlass"),
    ];

    for p in &candidates {
        if p.join("include/cutlass/cutlass.h").exists() {
            let resolved = p.canonicalize().unwrap_or_else(|_| p.clone());
            println!("cargo:warning=Found CUTLASS at {}", resolved.display());
            return resolved.to_string_lossy().into_owned();
        }
    }

    panic!(
        "CUTLASS headers not found. Set CUTLASS_DIR env var or clone \
         https://github.com/NVIDIA/cutlass next to this repository."
    );
}

fn dirs_or_home() -> PathBuf {
    env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp"))
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
