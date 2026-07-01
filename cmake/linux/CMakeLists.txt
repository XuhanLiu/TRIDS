cmake_minimum_required(VERSION 3.22)
set(TRIDS_VERSION "1.0.0")
# ==================== Conda toolchain (must run before project()) ====================
# - Prefer nvcc from the active env over /usr/local/cuda on PATH
# - Prefer the env-local compiler; base-miniconda gcc breaks linking on Ubuntu
if(DEFINED ENV{CONDA_PREFIX})
    if(EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc")
    endif()

    # Env-local conda gcc when present (CUDA 12.x); otherwise system g++ (CUDA 11.x).
    if(EXISTS "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++")
        set(CMAKE_CXX_COMPILER "$ENV{CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-c++")
    elseif(EXISTS "/usr/bin/g++")
        set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    endif()

    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

    # Help CMake, FindCUDA (PyTorch/Caffe2), and CUDAToolkit locate conda CUDA.
    # Layout varies: cuda.h under $CONDA_PREFIX/include (older) or
    # $CONDA_PREFIX/targets/x86_64-linux/include (CUDA 12.x conda packages).
    if(EXISTS "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cuda.h")
        set(CUDA_PREFIX "$ENV{CONDA_PREFIX}/targets/x86_64-linux")
    elseif(EXISTS "$ENV{CONDA_PREFIX}/include/cuda.h")
        set(CUDA_PREFIX "$ENV{CONDA_PREFIX}")
    endif()
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" "${CUDA_PREFIX}")
endif()

project(trids LANGUAGES CXX CUDA)
# libtorch requires at leat C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include GNUInstallDirs to use standard path variables like CMAKE_INSTALL_BINDIR/LIBDIR/INCLUDEDIR
include(GNUInstallDirs)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -flto=auto")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fopenmp -lcudadevrt -lcufft --use_fast_math -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
add_definitions(-D__CUDA_NO_HALF_CONVERSIONS__)
add_definitions(-D__CUDA_NO_HALF2_OPERATORS__)

# This file lives in cmake/linux/; the TRIDS sources are two levels up.
get_filename_component(TRIDS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(CMAKE_MODULE_PATH ${TRIDS_ROOT}/cmake/modules)


if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CMAKE_COMMAND}" "-E" "env" "--unset=NVCC_PREPEND_FLAGS")
    find_package(Python COMPONENTS Interpreter REQUIRED)
    execute_process(
        # this python code could be indented!!!
        COMMAND "${Python_EXECUTABLE}" -c "
import os, torch
print(torch.utils.cmake_prefix_path)
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
print(os.path.join(os.path.dirname(__import__('nvidia.nvtx').nvtx.__file__), 'include'))
print(os.path.dirname(__import__('nvidia.cudnn').cudnn.__file__))
"
        OUTPUT_VARIABLE TRIDS_PY
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    string(REPLACE "\n" ";" TRIDS_PY "${TRIDS_PY}")
    list(GET TRIDS_PY 0 TORCH_CMAKE_PATH)
    list(GET TRIDS_PY 1 TORCH_LIB_PATH)
    # nvtx3_dir could not be uppercase
    list(GET TRIDS_PY 2 nvtx3_dir)
    list(GET TRIDS_PY 3 CUDNN_BASE_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
    if(CUDNN_BASE_DIR)
        set(CUDNN_INCLUDE_PATH "${CUDNN_BASE_DIR}/include")
        set(CUDNN_LIBRARY_PATH "${CUDNN_BASE_DIR}/lib")
    endif()
endif()

# RPATH (Linux/macOS only)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR};$ORIGIN/../lib/trids${TRIDS_VERSION}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT OFF)
set(USE_CUDSS OFF)
set(CAFFE2_USE_CUFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDAToolkit before Torch so variables are set; Torch/Caffe2 still uses FindCUDA internally.
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenBabel3 REQUIRED)
# CLI11 - header-only command line parser
find_package(CLI11 REQUIRED)
# spdlog - use header-only mode to avoid ABI issues with conda version
find_package(spdlog REQUIRED)

# lib
file(GLOB LIB_SRC ${TRIDS_ROOT}/csrc/*.*)
include_directories(${TRIDS_ROOT}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenBabel3_INCLUDE_DIRS})

add_library(obj OBJECT ${LIB_SRC})
set_property(TARGET obj PROPERTY POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(obj PRIVATE spdlog::spdlog_header_only)
# add_dependencies(obj torch openbabel)

add_library(libtrids SHARED $<TARGET_OBJECTS:obj>)
set_target_properties(libtrids PROPERTIES OUTPUT_NAME trids)
target_link_libraries(libtrids
    ${OpenBabel3_LIBRARIES}
    ${TORCH_LIBRARIES}
    spdlog::spdlog_header_only
)

# main executable (use a different CMake target name to avoid clash with the library)
add_executable(trids_exe "${TRIDS_ROOT}/main.cpp")
set_target_properties(trids_exe PROPERTIES OUTPUT_NAME trids)
target_link_libraries(trids_exe libtrids CLI11::CLI11)

install(TARGETS trids_exe libtrids
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY ${TRIDS_ROOT}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${TRIDS_ROOT}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ==================== Python Bindings (optional) ====================
option(BUILD_PYTHON "Build Python bindings" OFF)

if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)

    message(STATUS "Building Python bindings with pybind11")

    # Python binding source files
    set(PYTHON_BIND_SRC ${TRIDS_ROOT}/csrc/python/api.cpp)
    pybind11_add_module(_core ${PYTHON_BIND_SRC})
    target_link_libraries(_core PRIVATE libtrids "${TORCH_LIB_PATH}/libtorch_python.so")
    set_target_properties(_core PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR};${TORCH_LIB_PATH}"
    )
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "12")
        target_link_options(_core PRIVATE -Wno-stringop-overread)
    endif()

    # Install to the Python package directory
    install(TARGETS _core libtrids DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY ${TRIDS_ROOT}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
