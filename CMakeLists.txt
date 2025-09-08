cmake_minimum_required(VERSION 3.4.1)

project(rknn_parallel_deploy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# rknn api
set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)
set(RGA_LIB ${CMAKE_SOURCE_DIR}/lib/librga.so)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)
# include_directories(${CMAKE_SOURCE_DIR}/3rdparty)
include_directories( ${CMAKE_SOURCE_DIR}/include)
include_directories( ${CMAKE_SOURCE_DIR}/src)
include_directories( ${CMAKE_SOURCE_DIR}/include/rga/include)
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
 
)