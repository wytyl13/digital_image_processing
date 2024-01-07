cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

project(DIGITAL_IMAGE_PROCESSING)
include_directories(./include)

#add the param
add_compile_options(-Wall -fexec-charset=GBK -std=c++11)

aux_source_directory(./src SRC_DIR_VAR)

add_executable(${PROJECT_NAME} ${SRC_DIR_VAR})


set(OpenCV_DIR C:\\Users\\weiyutao\\opt\\opencv\\build\\install)
find_package(OpenCV REQUIRED)

if((OpenCV_FOUND))
    #add the include path
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


    message("have found opencv include: ${OpenCV_INCLUDE_DIRS}")
    message("have found opencv libs: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "not found opencv!")
endif()