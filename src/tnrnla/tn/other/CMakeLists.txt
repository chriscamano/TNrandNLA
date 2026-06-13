cmake_minimum_required(VERSION 3.18)
project(ttm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

find_path(MKL_INCLUDE_DIR
  NAMES mkl.h
  HINTS ENV MKLROOT
  PATH_SUFFIXES include
)

find_library(MKL_RT_LIB
  NAMES mkl_rt
  HINTS ENV MKLROOT
  PATH_SUFFIXES lib lib/intel64 lib/intel64_lin
)

if(NOT MKL_INCLUDE_DIR)
  message(FATAL_ERROR "MKL headers not found, set MKLROOT")
endif()

if(NOT MKL_RT_LIB)
  message(FATAL_ERROR "mkl_rt not found, set MKLROOT or adjust library paths")
endif()

pybind11_add_module(ttm ttm.c)

set_source_files_properties(ttm.c PROPERTIES LANGUAGE CXX)

target_include_directories(ttm PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(ttm PRIVATE "${MKL_RT_LIB}")

if(UNIX AND NOT APPLE)
  target_link_libraries(ttm PRIVATE pthread m dl)
endif()

if(DEFINED ENV{MKLROOT})
  set(MKL_LIBDIR "$ENV{MKLROOT}/lib/intel64")
  if(EXISTS "${MKL_LIBDIR}")
    set_target_properties(ttm PROPERTIES BUILD_RPATH "${MKL_LIBDIR}")
  endif()
endif()