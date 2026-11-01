add_library(zerogr STATIC
  config.cc
  decoder.cu
  trie/trie_builder.cc)
add_library(ZeroGR::zerogr ALIAS zerogr)

target_compile_features(zerogr PUBLIC cxx_std_17 PRIVATE cuda_std_17)
target_include_directories(zerogr
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(zerogr PRIVATE zerogr_cccl CUDA::cudart CUDA::cublas)
