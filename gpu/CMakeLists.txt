cmake_minimum_required(VERSION 3.18)
project(pir_bench CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ---------- Find CUTLASS ----------
# Option 1: Set CUTLASS_DIR to your CUTLASS checkout
# Option 2: It will try to find it in common locations
if(NOT DEFINED CUTLASS_DIR)
    # Try common locations
    if(EXISTS "$ENV{HOME}/cutlass")
        set(CUTLASS_DIR "$ENV{HOME}/cutlass")
    elseif(EXISTS "/usr/local/cutlass")
        set(CUTLASS_DIR "/usr/local/cutlass")
    else()
        message(FATAL_ERROR 
            "CUTLASS not found. Clone it and pass -DCUTLASS_DIR=/path/to/cutlass\n"
            "  git clone https://github.com/NVIDIA/cutlass.git\n"
            "  cmake -DCUTLASS_DIR=/path/to/cutlass ..")
    endif()
endif()

message(STATUS "Using CUTLASS from: ${CUTLASS_DIR}")

# ---------- Detect GPU architecture ----------
# Default to sm_80 (A100). Change if needed.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ---------- Build ----------
add_executable(pir_bench pir_bench.cu)

target_include_directories(pir_bench PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

target_compile_options(pir_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(pir_bench PRIVATE CUDA::cudart CUDA::cublas)
