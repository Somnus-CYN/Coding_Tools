cmake_minimum_required(VERSION 3.14)
project(yolo11n_mobile_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TEST "Build desktop test" ON)

# TODO: adjust NCNN include/lib paths for your environment
find_package(ncnn REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(yolo11n_sdk STATIC
    src/yolo11n_detector.cpp
    src/engine_ncnn.cpp
    src/preprocess.cpp
    src/postprocess.cpp
)

set_target_properties(yolo11n_sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(yolo11n_sdk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ncnn_INCLUDE_DIRS}
)

target_link_libraries(yolo11n_sdk PUBLIC ncnn)

if(BUILD_TEST)
    add_executable(yolo11n_test_desktop tests/test_infer_desktop.cpp)
    target_include_directories(yolo11n_test_desktop PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(yolo11n_test_desktop PRIVATE yolo11n_sdk OpenCV::core OpenCV::imgcodecs OpenCV::imgproc yaml-cpp nlohmann_json::nlohmann_json)
endif()

install(TARGETS yolo11n_sdk ARCHIVE DESTINATION lib)
install(FILES include/yolo11n_detector.h DESTINATION include)
