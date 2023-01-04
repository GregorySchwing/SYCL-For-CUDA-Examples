cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
# Don't complain about empty CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)

project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

# SYCL installation
if (NOT SYCL_ROOT) 
  message(FATAL_ERROR "No SYCL installation detected")
endif(NOT SYCL_ROOT)

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/14.0.0/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS "-fsycl"
      "-fsycl-targets=nvptx64-nvidia-cuda"
      "-fsycl-unnamed-lambda"
      "-Wno-linker-warnings")

      # Build the SYCL code
add_executable (sycl_bfs bfs.cpp CSRGraphRep.cpp auxFunctions.cpp)
target_compile_features(sycl_bfs PUBLIC cxx_std_17)
target_compile_options(sycl_bfs PUBLIC ${SYCL_FLAGS})
target_link_libraries(sycl_bfs PUBLIC ${SYCL_FLAGS})
target_include_directories(sycl_bfs PUBLIC ${SYCL_INCLUDE_DIR})
target_link_libraries(sycl_bfs PUBLIC ${SYCL_LIB})

