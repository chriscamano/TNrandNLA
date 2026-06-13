cmake_minimum_required(VERSION 3.18)
project(tdvp_oracle_prefused LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
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
  message(FATAL_ERROR "MKL headers not found. Set MKLROOT.")
endif()

if(NOT MKL_RT_LIB)
  message(FATAL_ERROR "mkl_rt not found. Set MKLROOT or adjust library paths.")
endif()

pybind11_add_module(tdvp_oracle_prefused tdvp_oracle_prefused.cpp)
set_source_files_properties(tdvp_oracle_prefused.cpp PROPERTIES LANGUAGE CXX)

target_include_directories(tdvp_oracle_prefused PRIVATE "${MKL_INCLUDE_DIR}")
target_link_libraries(tdvp_oracle_prefused PRIVATE "${MKL_RT_LIB}")

target_compile_options(tdvp_oracle_prefused PRIVATE
  -O3
  -DNDEBUG
  -march=native
  -ffast-math
  -fno-math-errno
  -funsafe-math-optimizations
  -funroll-loops
)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_OK OUTPUT IPO_MSG)
if(IPO_OK)
  set_property(TARGET tdvp_oracle_prefused PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(tdvp_oracle_prefused PRIVATE pthread m dl)
endif()

if(DEFINED ENV{MKLROOT})
  set(MKL_LIBDIR "$ENV{MKLROOT}/lib/intel64")
  if(EXISTS "${MKL_LIBDIR}")
    set_target_properties(tdvp_oracle_prefused PROPERTIES
      BUILD_RPATH "${MKL_LIBDIR}"
      INSTALL_RPATH "${MKL_LIBDIR}"
    )
  endif()
endif()