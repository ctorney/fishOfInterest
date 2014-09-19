
# Top-Level CmakeLists.txt
cmake_minimum_required(VERSION 2.8)
PROJECT( showTracks )

find_package(OpenCV REQUIRED)
find_package(NetCDF REQUIRED)
find_package(CUDA QUIET)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 )

INCLUDE_DIRECTORIES(${showTracks_SOURCE_DIR}/include ${NETCDF_INCLUDES})

ADD_EXECUTABLE(showTracks main.cpp)

TARGET_LINK_LIBRARIES(showTracks ${OpenCV_LIBS} ${NetCDF_LIBRARIES})
