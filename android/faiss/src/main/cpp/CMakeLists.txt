cmake_minimum_required(VERSION 3.10)
project(faiss2)

set(CMAKE_ANDROID_STL_TYPE c++_static)
set(CMAKE_CXX_STANDARD 17)

# include headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/faiss)
include_directories(${CMAKE_SOURCE_DIR}/include/openblas)
include_directories(${CMAKE_SOURCE_DIR}/include/clapack)

add_library(native-lib SHARED
        native-lib.cpp
)

# enable Faiss stats compile-time guards so our #ifdef blocks are active
target_compile_definitions(native-lib PRIVATE
        FAISS_HAS_INDEX_IVF_STATS=1
        FAISS_HAS_HNSW_STATS=1
)

# Import static libraries
add_library(faiss STATIC IMPORTED)
set_target_properties(faiss PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libfaiss.a)

add_library(openblas STATIC IMPORTED)
set_target_properties(openblas PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libopenblas.a)

# CLAPACK libs
add_library(lapack STATIC IMPORTED)
set_target_properties(lapack PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/liblapack.a)

add_library(blas_c STATIC IMPORTED)
set_target_properties(blas_c PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libblas.a)

add_library(f2c STATIC IMPORTED)
set_target_properties(f2c PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libf2c.a)

# Optional: OpenMP support (only if FAISS was built with OpenMP)
# Otherwise delete this section
add_library(omp STATIC IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libomp.a)

target_link_libraries(native-lib
        faiss
        openblas
        lapack
        blas_c
        f2c
        omp
        c++_static
        -lc++abi
        log
        atomic
        m
)
