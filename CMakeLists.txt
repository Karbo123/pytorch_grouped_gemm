cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PYTORCH_GROUPED_GEMM LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

# # an example executable from nvidia
# set(EXE_NAME ${PROJECT_NAME}_EXE)
# add_executable(${EXE_NAME} "${CMAKE_SOURCE_DIR}/csrc/gemm_grouped.cu")
# set_target_properties(${EXE_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86")
# target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
# target_include_directories(${EXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
# target_include_directories(${EXE_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
# target_compile_features(${EXE_NAME} PUBLIC cxx_std_11)

# # my exe
# set(MYEXE_NAME ${PROJECT_NAME}_MYEXE)
# add_executable(${MYEXE_NAME} "${CMAKE_SOURCE_DIR}/csrc/torch_grouped_gemm.cu")
# set_target_properties(${MYEXE_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86")
# target_include_directories(${MYEXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
# target_include_directories(${MYEXE_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")
# target_include_directories(${MYEXE_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
# target_compile_features(${MYEXE_NAME} PUBLIC cxx_std_11)

# my lib
find_package(Torch 1.8 REQUIRED)
find_package(Python3 3.6 REQUIRED Interpreter Development)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
add_library(${PROJECT_NAME} SHARED "${CMAKE_SOURCE_DIR}/csrc/main.cu")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;86")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14) 
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})
target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include")
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/third_party/cutlass/include")

# example:
#   cmake .. -DCMAKE_PREFIX_PATH=/data/lei.jiabao/miniconda3/envs/recon180/lib/python3.7/site-packages/torch/share/cmake/Torch
