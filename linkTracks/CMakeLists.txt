
# Top-Level CmakeLists.txt
cmake_minimum_required(VERSION 2.8)
PROJECT( linkTracks )

find_package(NetCDF REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 )

INCLUDE_DIRECTORIES(${linkTracks_SOURCE_DIR}/include ${NETCDF_INCLUDES})

ADD_EXECUTABLE(linkTracks main.cpp)

TARGET_LINK_LIBRARIES(linkTracks ${OpenCV_LIBS} ${NetCDF_LIBRARIES})
