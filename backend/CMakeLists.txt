cmake_minimum_required(VERSION 3.16)
project(yolo_tracker_backend CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/homebrew/include
    /opt/homebrew/opt/opencv/include/opencv4
    /opt/homebrew/include/onnxruntime
)

link_directories(
    /opt/homebrew/lib
    /opt/homebrew/opt/opencv/lib
)

find_package(OpenCV REQUIRED)

add_executable(yolo_tracker_backend
    src/main.cpp
    src/yolo_detector.cpp
    src/object_tracker.cpp
    src/logger.cpp
    src/tiling.cpp
)

target_link_libraries(yolo_tracker_backend PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)
