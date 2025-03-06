cmake_minimum_required(VERSION 3.10)
project(esp)

# Add debug symbols
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg -g")

file(GLOB SOURCES "src/*.cpp")
add_executable(esp ${SOURCES} )
include_directories(esp include)
target_link_libraries(esp uuid)
