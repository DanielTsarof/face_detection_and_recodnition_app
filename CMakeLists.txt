cmake_minimum_required(VERSION 3.0.0)
project(ImgProc VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

file(GLOB lib_src
    "src/*.cpp"
)

add_library(${PROJECT_NAME} STATIC
    ${lib_src}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)

file(GLOB src_files
    "*.cpp"
)

add_executable(runner ${src_files})

target_link_libraries(runner
    ${PROJECT_NAME}
)
