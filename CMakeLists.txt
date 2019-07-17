cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)

project (PhenotyperCV)

find_package( OpenCV REQUIRED )
find_package (Eigen3 REQUIRED NO_MODULE)
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(PhenotyperCV ${SOURCES})
target_link_libraries( PhenotyperCV ${OpenCV_LIBS} )
