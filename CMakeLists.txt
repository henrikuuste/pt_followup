cmake_minimum_required(VERSION 3.12)

project(sfml_cuda_project CXX)
include(cmake/StandardProjectSettings.cmake)

# find_package(CUDA REQUIRED)

# include(CheckLanguage)
# check_language(CUDA)
# SET(CUDA_SEPARABLE_COMPILATION ON)
# cuda_select_nvcc_arch_flags(ARCH_FLAGS)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

set(CONAN_EXTRA_REQUIRES ${CONAN_EXTRA_REQUIRES})

include(cmake/Conan.cmake)
run_conan()

add_subdirectory(src)
