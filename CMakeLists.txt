cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS -Wno-narrowing)

project (PhenotyperCV)

find_package( OpenCV REQUIRED )
include_directories(/usr/local/include)

find_library(ZBAR_LIBRARIES NAMES zbar)
find_path(ZBAR_INCLUDE_DIR Decoder.h PATH_SUFFIXES zbar )

include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(include)
file(GLOB SOURCES "src/*.cpp")
add_executable(PhenotyperCV ${SOURCES})
target_link_libraries( PhenotyperCV ${OpenCV_LIBS} ${ZBAR_LIBRARIES})
