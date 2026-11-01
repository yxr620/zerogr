cmake_minimum_required(VERSION 3.24)
project(ZeroGR LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CCCL_ROOT "" CACHE PATH "Path to an NVIDIA CCCL 3.3.4 checkout")
if(NOT EXISTS "${CCCL_ROOT}/cub/cub/device/device_topk.cuh")
  message(FATAL_ERROR "Set CCCL_ROOT to a CCCL 3.3.4 checkout (DeviceTopK required)")
endif()

add_executable(zerogr src/main.cu src/trie_builder.cc)
target_compile_features(zerogr PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(zerogr BEFORE PRIVATE
  "${CCCL_ROOT}/cub" "${CCCL_ROOT}/thrust" "${CCCL_ROOT}/libcudacxx/include")
target_link_libraries(zerogr PRIVATE CUDA::cudart CUDA::cublas)
