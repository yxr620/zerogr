find_package(CUDAToolkit REQUIRED)

set(CCCL_ROOT "" CACHE PATH "Path to an NVIDIA CCCL 3.3.4 checkout")
foreach(header IN ITEMS
    cub/cub/device/device_topk.cuh
    thrust/thrust/version.h
    libcudacxx/include/cuda/stream)
  if(NOT EXISTS "${CCCL_ROOT}/${header}")
    message(FATAL_ERROR "Set CCCL_ROOT to a complete CCCL 3.3.4 checkout; missing ${header}")
  endif()
endforeach()

add_library(zerogr_cccl INTERFACE)
target_include_directories(zerogr_cccl INTERFACE
  "${CCCL_ROOT}/cub" "${CCCL_ROOT}/thrust" "${CCCL_ROOT}/libcudacxx/include")
