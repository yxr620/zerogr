add_executable(zerogr_synthetic synthetic.cc options.cc synthetic_data.cc)
target_link_libraries(zerogr_synthetic PRIVATE ZeroGR::zerogr)
set_target_properties(zerogr_synthetic PROPERTIES
  OUTPUT_NAME zerogr
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
