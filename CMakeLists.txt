cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PYTORCH_GROUPED_GEMM LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

# an example executable from nvidia
set(EXE_NAME ${PROJECT_NAME}_EXE)
add_executable(${EXE_NAME} "${CMAKE_SOURCE_DIR}/csrc/gemm_grouped.cu")
set_target_properties(${EXE_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86")
target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
target_include_directories(${EXE_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_compile_features(${EXE_NAME} PUBLIC cxx_std_11)

# my lib
# TODO
