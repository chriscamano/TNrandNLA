cmake_minimum_required(VERSION 3.18)
project(incrementalqr LANGUAGES CXX)

# ------------------------------------------------------------
# Respect active Conda environment
# ------------------------------------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    message(STATUS "Using Conda environment: $ENV{CONDA_PREFIX}")
else()
    message(FATAL_ERROR "CONDA_PREFIX is not set. Activate your conda env first.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to Release if user did not set it
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------
# Python and Pybind11
# ------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ------------------------------------------------------------
# MKL (explicit, from Conda)
# ------------------------------------------------------------
find_path(MKL_INCLUDE_DIR
    NAMES mkl.h mkl_cblas.h mkl_lapacke.h
    PATHS "$ENV{CONDA_PREFIX}/include"
    NO_DEFAULT_PATH
)

find_library(MKL_RT_LIB
    NAMES mkl_rt
    PATHS "$ENV{CONDA_PREFIX}/lib"
    NO_DEFAULT_PATH
)

if(NOT MKL_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find MKL headers in $ENV{CONDA_PREFIX}/include")
endif()

if(NOT MKL_RT_LIB)
    message(FATAL_ERROR "Could not find mkl_rt in $ENV{CONDA_PREFIX}/lib")
endif()

message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")
message(STATUS "MKL runtime: ${MKL_RT_LIB}")

# ------------------------------------------------------------
# Build the Python extension module
# ------------------------------------------------------------
# Make sure incrementalqr.cpp contains PYBIND11_MODULE(libincrementalqr, m)
pybind11_add_module(libincrementalqr incrementalqr.cpp)

target_include_directories(libincrementalqr PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${MKL_INCLUDE_DIR}
)

# Keep RTTI and exceptions ON because pybind11 requires RTTI,
# and your code throws std::runtime_error.
target_compile_definitions(libincrementalqr PRIVATE NDEBUG)

# ------------------------------------------------------------
# Fast compile flags
# ------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(libincrementalqr PRIVATE
        -O3
        -march=native
        -ffast-math
        -fno-math-errno
        -funroll-loops
        -fomit-frame-pointer
        -fvisibility=hidden
    )

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
    if(ipo_ok)
        set_property(TARGET libincrementalqr PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO/LTO not supported: ${ipo_msg}")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Intel")
    target_compile_options(libincrementalqr PRIVATE
        -O3
        -march=native
        -ffast-math
    )
endif()

# ------------------------------------------------------------
# Link against MKL (single entry point) + required system libs
# ------------------------------------------------------------
target_link_libraries(libincrementalqr PRIVATE
    pybind11::module
    ${MKL_RT_LIB}
    m
    dl
    pthread
)

# Strip symbols in Release where supported
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(libincrementalqr PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
endif()

# ------------------------------------------------------------
# Install into the active Python environment
# ------------------------------------------------------------
include(GNUInstallDirs)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE PYTHON_PLATLIB
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Python platlib (install target): ${PYTHON_PLATLIB}")

install(TARGETS libincrementalqr
  LIBRARY DESTINATION "${PYTHON_PLATLIB}"
  RUNTIME DESTINATION "${PYTHON_PLATLIB}"
  ARCHIVE DESTINATION "${PYTHON_PLATLIB}"
)