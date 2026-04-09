cmake_minimum_required(VERSION 2.8)
project(falib_demo)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR "/usr/local/opencv/")
#set(OpenCV_DIR "D:/opencv")

include_directories(
#        ${OpenCV_DIR}/include
        ${OpenCV_DIR}/include/opencv4
)

link_directories(
        ${OpenCV_DIR}/lib/
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
set(OpenCV_LIBS opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio)

add_executable(falib_demo main.cpp nx_fa_export.h)
target_link_libraries(falib_demo ${OpenCV_LIBS} libfa_dymc.so)
