cmake_minimum_required(VERSION 3.14)
project(trajairnet_onnx_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pass -DONNXRUNTIME_DIR=/path/to/onnxruntime when configuring.
if(NOT DEFINED ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Please set ONNXRUNTIME_DIR, e.g. -DONNXRUNTIME_DIR=/opt/onnxruntime")
endif()

add_executable(onnx_infer onnx_infer.cpp)

target_include_directories(onnx_infer PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_directories(onnx_infer PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(onnx_infer PRIVATE onnxruntime)
