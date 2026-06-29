cmake_minimum_required(VERSION 3.22)

get_filename_component(TRIDS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
get_filename_component(TRIDS_WINDOWS "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

if(NOT WIN32)
    message(FATAL_ERROR "Use cmake/linux for non-Windows builds.")
endif()
if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "CONDA_PREFIX is not set. Run: conda activate trids")
endif()

set(TRIDS_CONDA "$ENV{CONDA_PREFIX}")
file(TO_CMAKE_PATH "${TRIDS_CONDA}" TRIDS_CONDA)

set(CMAKE_CUDA_COMPILER "${TRIDS_CONDA}/Library/bin/nvcc.exe" CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_COMPILER_LAUNCHER "${TRIDS_CONDA}/python.exe" "${TRIDS_WINDOWS}/trids_nvcc_driver.py")

find_program(TRIDS_CL NAMES cl cl.exe)
if(NOT TRIDS_CL)
    find_program(TRIDS_VSWHERE NAMES vswhere.exe HINTS "${TRIDS_CONDA}/Library/bin")
    if(TRIDS_VSWHERE)
        execute_process(
            COMMAND "${TRIDS_VSWHERE}" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            OUTPUT_VARIABLE TRIDS_VS_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(TRIDS_VS_DIR)
            file(TO_CMAKE_PATH "${TRIDS_VS_DIR}" TRIDS_VS_DIR)
            file(GLOB TRIDS_MSVC_VERS LIST_DIRECTORIES true "${TRIDS_VS_DIR}/VC/Tools/MSVC/*")
            list(SORT TRIDS_MSVC_VERS COMPARE NATURAL ORDER DESCENDING)
            if(TRIDS_MSVC_VERS)
                list(GET TRIDS_MSVC_VERS 0 TRIDS_MSVC_ROOT)
                set(TRIDS_CL "${TRIDS_MSVC_ROOT}/bin/Hostx64/x64/cl.exe")
            endif()
        endif()
    endif()
endif()
if(NOT TRIDS_CL OR NOT EXISTS "${TRIDS_CL}")
    message(FATAL_ERROR "MSVC cl.exe not found. Install VS 2022 Build Tools (Desktop C++ workload).")
endif()
file(TO_CMAKE_PATH "${TRIDS_CL}" TRIDS_CL)
set(CMAKE_CXX_COMPILER "${TRIDS_CL}" CACHE FILEPATH "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
get_filename_component(TRIDS_CL_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(ENV{PATH} "${TRIDS_CL_DIR};$ENV{PATH}")

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT "")
set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "")

project(trids LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
include(GNUInstallDirs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /EHsc /bigobj")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3 -lcudadevrt -lcufft --expt-relaxed-constexpr")
add_compile_definitions(
    NOMINMAX _CRT_SECURE_NO_WARNINGS _SILENCE_CXX17_DEPRECATION_WARNINGS SPDLOG_HEADER_ONLY
    __CUDA_NO_HALF_OPERATORS__ __CUDA_NO_HALF_CONVERSIONS__ __CUDA_NO_HALF2_OPERATORS__)
add_compile_options(/wd4624 /wd4267 /wd4244 /wd4251 /wd4275 /wd4996)

set(TORCH_CMAKE_PATH "${TRIDS_CONDA}/Lib/site-packages/torch/share/cmake")
set(TORCH_LIB_PATH "${TRIDS_CONDA}/Lib/site-packages/torch/lib")
set(nvtx3_dir "${TRIDS_CONDA}/Lib/site-packages/nvidia/nvtx/include")
set(CUDNN_BASE_DIR "${TRIDS_CONDA}/Lib/site-packages/nvidia/cudnn")
list(PREPEND CMAKE_PREFIX_PATH "${TRIDS_CONDA}/Library" "${TRIDS_CONDA}/Library/lib/cmake" "${TORCH_CMAKE_PATH}")

if(EXISTS "${nvtx3_dir}/nvtx3")
    set(USE_SYSTEM_NVTX ON CACHE BOOL "" FORCE)
    set(CUDA_INCLUDE_DIRS "${TRIDS_CONDA}/Library/include" "${nvtx3_dir}")
endif()

function(trids_make_import_lib _name _out_var)
    set(_implib "${CMAKE_BINARY_DIR}/${_name}.lib")
    if(NOT EXISTS "${_implib}")
        file(WRITE "${CMAKE_BINARY_DIR}/${_name}.def" "LIBRARY ${_name}\nEXPORTS\n")
        execute_process(COMMAND lib "/DEF:${CMAKE_BINARY_DIR}/${_name}.def" "/OUT:${_implib}" /MACHINE:x64
            RESULT_VARIABLE _ok)
        if(NOT _ok EQUAL 0 OR NOT EXISTS "${_implib}")
            message(FATAL_ERROR "Failed to generate ${_name} import library")
        endif()
    endif()
    set(${_out_var} "${_implib}" PARENT_SCOPE)
endfunction()

function(trids_implib_from_dll _dll _name _out_var)
    set(_implib "${CMAKE_BINARY_DIR}/${_name}.lib")
    set(_def "${CMAKE_BINARY_DIR}/${_name}.def")
    if(NOT EXISTS "${_implib}")
        find_program(TRIDS_DUMPBIN NAMES dumpbin dumpbin.exe)
        find_program(TRIDS_LIB NAMES lib lib.exe)
        if(NOT TRIDS_DUMPBIN OR NOT TRIDS_LIB)
            message(FATAL_ERROR "dumpbin/lib not found. Run build.bat (vcvars64) before cmake configure.")
        endif()
        execute_process(
            COMMAND "${TRIDS_CONDA}/python.exe" -c
                "import re,subprocess,sys; dll,def_path=sys.argv[1:3]; out=subprocess.check_output([sys.argv[3],'/exports',dll],text=True,errors='replace'); names=[m.group(1) for line in out.splitlines() if (m:=re.match(r'^\\s+\\d+\\s+[0-9A-Fa-f]+\\s+[0-9A-Fa-f]+\\s+(\\S+)\\s*$', line))]; f=open(def_path,'w',encoding='ascii'); f.write('LIBRARY libiomp5md\\nEXPORTS\\n'); f.writelines('  %s\\n'%n for n in names); f.close()"
                "${_dll}" "${_def}" "${TRIDS_DUMPBIN}"
            RESULT_VARIABLE _def_ok)
        if(NOT _def_ok EQUAL 0 OR NOT EXISTS "${_def}")
            message(FATAL_ERROR "Failed to generate module definition from ${_dll}")
        endif()
        execute_process(
            COMMAND "${TRIDS_LIB}" "/DEF:${_def}" "/OUT:${_implib}" /MACHINE:x64
            RESULT_VARIABLE _lib_ok)
        if(NOT _lib_ok EQUAL 0 OR NOT EXISTS "${_implib}")
            message(FATAL_ERROR "Failed to generate import library from ${_dll}")
        endif()
    endif()
    set(${_out_var} "${_implib}" PARENT_SCOPE)
endfunction()

if(EXISTS "${CUDNN_BASE_DIR}/include/cudnn.h")
    set(CUDNN_INCLUDE_PATH "${CUDNN_BASE_DIR}/include" CACHE PATH "" FORCE)
    file(GLOB TRIDS_CUDNN_LIB "${CUDNN_BASE_DIR}/lib/cudnn*.lib" "${CUDNN_BASE_DIR}/lib/x64/cudnn*.lib")
    if(TRIDS_CUDNN_LIB)
        list(GET TRIDS_CUDNN_LIB 0 CUDNN_LIBRARY_PATH)
        set(CUDNN_LIBRARY_PATH "${CUDNN_LIBRARY_PATH}" CACHE FILEPATH "" FORCE)
    elseif(EXISTS "${CUDNN_BASE_DIR}/bin/cudnn64_9.dll")
        trids_make_import_lib(cudnn TRIDS_CUDNN_STUB)
        set(CUDNN_LIBRARY "${CMAKE_BINARY_DIR}" CACHE PATH "" FORCE)
        set(CUDNN_LIBRARY_PATH "${TRIDS_CUDNN_STUB}" CACHE FILEPATH "" FORCE)
    endif()
endif()

set(CAFFE2_USE_CUDNN ON)
set(CAFFE2_USE_CUSPARSELT OFF)
set(USE_CUDSS OFF)
set(CAFFE2_USE_CUFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

set(OpenBabel3_INCLUDE_DIRS "${TRIDS_CONDA}/Library/include/openbabel3")
if(NOT TARGET openbabel)
    include("${TRIDS_CONDA}/Library/bin/cmake/openbabel3/OpenBabel3_EXPORTS.cmake")
endif()
set(OpenBabel3_LIBRARIES openbabel)

if(EXISTS "${CUDAToolkit_INCLUDE_DIRS}/../targets/x64")
    include_directories(BEFORE SYSTEM "${CUDAToolkit_INCLUDE_DIRS}/../targets/x64")
elseif(EXISTS "${TRIDS_CONDA}/Library/include/targets/x64")
    include_directories(BEFORE SYSTEM "${TRIDS_CONDA}/Library/include/targets/x64")
endif()

if(NOT TARGET CUDA::nvToolsExt AND EXISTS "${TORCH_LIB_PATH}/nvToolsExt64_1.dll")
    trids_make_import_lib(nvToolsExt64_1 TRIDS_NVTOOLSEXT_LIB)
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_IMPLIB "${TRIDS_NVTOOLSEXT_LIB}"
        IMPORTED_LOCATION "${TORCH_LIB_PATH}/nvToolsExt64_1.dll")
endif()

find_package(CLI11 REQUIRED)
find_package(spdlog REQUIRED)

# Intel OpenMP (libiomp5md.dll from PyTorch), same runtime as PyTorch — not MSVC vcomp/libomp.
# Conda Library/lib/libiomp5md.lib is an import stub for libomp.dll; use PyTorch's Intel DLL instead.
set(TRIDS_IOMP5_DLL "${TORCH_LIB_PATH}/libiomp5md.dll")
if(NOT EXISTS "${TRIDS_IOMP5_DLL}")
    message(FATAL_ERROR "PyTorch Intel OpenMP DLL not found: ${TRIDS_IOMP5_DLL}")
endif()
trids_implib_from_dll("${TRIDS_IOMP5_DLL}" trids_iomp5md TRIDS_IOMP5_LIB)
find_path(TRIDS_OMP_INCLUDE NAMES omp.h HINTS "${TRIDS_CONDA}/Library/include" REQUIRED)
message(STATUS "Intel OpenMP: ${TRIDS_IOMP5_DLL} -> ${TRIDS_IOMP5_LIB}")

set(TRIDS_OPENMP_FLAG "/openmp")
add_library(trids::openmp INTERFACE IMPORTED GLOBAL)
target_include_directories(trids::openmp INTERFACE "${TRIDS_OMP_INCLUDE}")
target_compile_options(trids::openmp INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:${TRIDS_OPENMP_FLAG}>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${TRIDS_OPENMP_FLAG}>")
target_link_libraries(trids::openmp INTERFACE "${TRIDS_IOMP5_LIB}")
target_link_options(trids::openmp INTERFACE
    "/NODEFAULTLIB:vcomp" "/NODEFAULTLIB:vcompd" "/NODEFAULTLIB:libomp")

file(GLOB LIB_SRC "${TRIDS_ROOT}/csrc/*.*")
list(REMOVE_ITEM LIB_SRC "${TRIDS_ROOT}/csrc/log.cpp")
set(TRIDS_LOG_SRC "${TRIDS_ROOT}/csrc/log.cpp")
set(TRIDS_MSVC_COMPAT "${TRIDS_WINDOWS}/msvc_compat.h")
include_directories(${TRIDS_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${OpenBabel3_INCLUDE_DIRS})

add_library(obj OBJECT ${LIB_SRC})
set_property(TARGET obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/FI${TRIDS_MSVC_COMPAT} /wd4804 /wd4819>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>)
set_property(TARGET obj PROPERTY CUDA_COMPILE_OPTIONS_MSVC_DEBUG_INFORMATION_FORMAT "")
set_target_properties(obj PROPERTIES COMPILE_PDB_NAME "")
target_link_libraries(obj PRIVATE spdlog::spdlog_header_only trids::openmp)

add_library(libtrids SHARED $<TARGET_OBJECTS:obj>)
set_target_properties(libtrids PROPERTIES OUTPUT_NAME trids WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_link_libraries(libtrids PRIVATE
    ${OpenBabel3_LIBRARIES} ${TORCH_LIBRARIES} spdlog::spdlog_header_only trids::openmp)

add_executable(trids_exe "${TRIDS_ROOT}/main.cpp" ${TRIDS_LOG_SRC} $<TARGET_OBJECTS:obj>)
target_compile_options(trids_exe PRIVATE /FI${TRIDS_MSVC_COMPAT} /wd4804 /wd4819)
set_target_properties(trids_exe PROPERTIES OUTPUT_NAME trids ARCHIVE_OUTPUT_NAME trids_exe)
set_source_files_properties(${TRIDS_LOG_SRC} TARGET_DIRECTORY trids_exe PROPERTIES COMPILE_DEFINITIONS USING_DYNAMIC_LIBS)
target_link_libraries(trids_exe PRIVATE ${TORCH_LIBRARIES} spdlog::spdlog_header_only CLI11::CLI11 openbabel inchi trids::openmp)

install(TARGETS trids_exe libtrids
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${TRIDS_ROOT}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${TRIDS_ROOT}/lib/" DESTINATION ${CMAKE_INSTALL_LIBDIR})

option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    set(TRIDS_TORCH_PYTHON_LIB "")
    if(EXISTS "${TORCH_LIB_PATH}/torch_python.lib")
        set(TRIDS_TORCH_PYTHON_LIB "${TORCH_LIB_PATH}/torch_python.lib")
    endif()
    pybind11_add_module(_core "${TRIDS_ROOT}/csrc/python/api.cpp" ${TRIDS_LOG_SRC})
    target_compile_options(_core PRIVATE /FI${TRIDS_MSVC_COMPAT} /wd4804 /wd4819)
    set_source_files_properties(${TRIDS_LOG_SRC} TARGET_DIRECTORY _core PROPERTIES COMPILE_DEFINITIONS USING_DYNAMIC_LIBS)
    target_link_libraries(_core PRIVATE
        libtrids
        ${OpenBabel3_LIBRARIES}
        ${TORCH_LIBRARIES}
        ${TRIDS_TORCH_PYTHON_LIB}
        spdlog::spdlog_header_only
        trids::openmp
    )
    install(TARGETS _core libtrids DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY "${TRIDS_ROOT}/lib/" DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
