
# Top-Level CmakeLists.txt
cmake_minimum_required(VERSION 2.8)
PROJECT( createPositions )

find_package(OpenCV REQUIRED)
find_package(NetCDF REQUIRED)
find_package(CUDA QUIET)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch sm_20 )

INCLUDE_DIRECTORIES(${createPositions_SOURCE_DIR}/include ${NETCDF_INCLUDES})

ADD_EXECUTABLE(createPositions main.cpp)

TARGET_LINK_LIBRARIES(createPositions ${OpenCV_LIBS} ${NetCDF_LIBRARIES})
