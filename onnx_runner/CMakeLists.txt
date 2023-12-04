cmake_minimum_required(VERSION 3.0)
project(FeatureBooster-OnnxRunner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "Project: FeatureBooster-OnnxRunner")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

# 设置编译模式
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb -DDEBUG")   # debug模式下 gdb相关选项
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")                   # release模式下 gdb相关选项
SET(CMAKE_BUILD_TYPE "Release")  # 选项: Release Debug MinSizeRel RelWithDebInfo

# 设置文件读取目录
SET (THIS_COM "${CMAKE_CURRENT_SOURCE_DIR}")
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake ${CMAKE_CURRENT_SOURCE_DIR}/include/Config.h)

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(LIBPNG_ROOTDIR "libpng root dir")
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

set(ONNXRUNTIME_ROOTDIR "/usr/local")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

include_directories(include)

find_package(JPEG)
find_package(PNG)
find_package(OpenCV 4 QUIET)

add_executable(main
   ./src/main.cpp
   ./src/FeatureBoosterOnnxRunner.cpp)
target_link_libraries(main ${OpenCV_LIBRARIES} onnxruntime)

add_executable(ProviderChecker
   ./src/ProviderChecker.cpp)
target_link_libraries(ProviderChecker onnxruntime)

