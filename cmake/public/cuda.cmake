# ---[ cuda

include_guard(GLOBAL)

# We don't want to statically link cudart, because we rely on it's dynamic linkage in
# python (follow along torch/cuda/__init__.py and usage of cudaGetErrorName).
# Technically, we can link cudart here statically, and link libtorch_python.so
# to a dynamic libcudart.so, but that's just wasteful.
# However, on Windows, if this one gets switched off, the error "cuda: unknown error"
# will be raised when running the following code:
# >>> import torch
# >>> torch.cuda.is_available()
# >>> torch.cuda.current_device()
# More details can be found in the following links.
# https://github.com/pytorch/pytorch/issues/20635
# https://github.com/pytorch/pytorch/issues/17108
if(NOT MSVC)
  set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
endif()

# Enable CUDA language support
if(CUDA_TOOLKIT_ROOT_DIR AND NOT CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
endif()

# CMP0074 - find_package will respect <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)
find_package(CUDAToolkit)

if(NOT CUDAToolkit_FOUND)
  message(WARNING
    "PyTorch: CUDA cannot be found. Depending on whether you are building "
    "PyTorch or a PyTorch dependent library, the next warning / error will "
    "give you more info.")
  set(CAFFE2_USE_CUDA OFF)
  return()
endif()

if(CUDAToolkit_VERSION VERSION_LESS 11.0)
  message(FATAL_ERROR "PyTorch requires CUDA 11.0 or above.")
endif()

# Pass clang as host compiler, which according to the docs
# Must be done before CUDA language is enabled, see
# https://cmake.org/cmake/help/v3.15/variable/CMAKE_CUDA_HOST_COMPILER.html
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
endif()
enable_language(CUDA)
if("X${CMAKE_CUDA_STANDARD}" STREQUAL "X" )
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
endif()
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "PyTorch: CUDA detected: " ${CUDAToolkit_VERSION})
message(STATUS "PyTorch: CUDA nvcc is: " ${CUDAToolkit_NVCC_EXECUTABLE})
message(STATUS "PyTorch: CUDA toolkit directory: " ${CUDAToolkit_ROOT})

# cuda_select_nvcc_arch_flags is required
cmake_policy(SET CMP0146 OLD)
find_package(CUDA)

# ---[ CUDA libraries wrapper

# Create new style imported libraries.
# Several of these libraries have a hardcoded path if CAFFE2_STATIC_LINK_CUDA
# is set. This path is where sane CUDA installations have their static
# libraries installed. This flag should only be used for binary builds, so
# end-users should never have this flag set.

# cudart
add_library(torch::cudart INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA)
  target_link_libraries(torch::cudart INTERFACE CUDA::cudart_static)
else()
  target_link_libraries(torch::cudart INTERFACE CUDA::cudart)
endif()


# cublas
add_library(torch::cublas INTERFACE IMPORTED)
# NOTE: cublas is always linked dynamically
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
  target_link_libraries(torch::cublas INTERFACE CUDA::cublas CUDA::cublasLt CUDA::cudart_static)
else()
  target_link_libraries(torch::cublas INTERFACE CUDA::cublas CUDA::cublasLt)
endif()

# cudnn interface
# static linking is handled by USE_STATIC_CUDNN environment variable
if(CAFFE2_USE_CUDNN)
  if(USE_STATIC_CUDNN)
    set(CUDNN_STATIC ON CACHE BOOL "")
  else()
    set(CUDNN_STATIC OFF CACHE BOOL "")
  endif()

  find_package(CUDNN)

  if(NOT CUDNN_FOUND)
    message(WARNING
      "Cannot find cuDNN library. Turning the option off")
    set(CAFFE2_USE_CUDNN OFF)
  else()
    if(CUDNN_VERSION VERSION_LESS "8.1.0")
      message(FATAL_ERROR "PyTorch requires cuDNN 8.1 and above.")
    endif()
  endif()

  add_library(torch::cudnn INTERFACE IMPORTED)
  target_include_directories(torch::cudnn INTERFACE ${CUDNN_INCLUDE_PATH})
  if(CUDNN_STATIC AND NOT WIN32)
    target_link_options(torch::cudnn INTERFACE
        "-Wl,--exclude-libs,libcudnn_static.a")
  else()
    target_link_libraries(torch::cudnn INTERFACE ${CUDNN_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUDNN is set to 0. Compiling without cuDNN support")
endif()

if(CAFFE2_USE_CUSPARSELT)
  find_package(CUSPARSELT)

  if(NOT CUSPARSELT_FOUND)
    message(WARNING
      "Cannot find cuSPARSELt library. Turning the option off")
    set(CAFFE2_USE_CUSPARSELT OFF)
  else()
    add_library(torch::cusparselt INTERFACE IMPORTED)
    target_include_directories(torch::cusparselt INTERFACE ${CUSPARSELT_INCLUDE_PATH})
    target_link_libraries(torch::cusparselt INTERFACE ${CUSPARSELT_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support")
endif()

if(USE_CUDSS)
  find_package(CUDSS)

  if(NOT CUDSS_FOUND)
    message(WARNING
      "Cannot find CUDSS library. Turning the option off")
    set(USE_CUDSS OFF)
  else()
    add_library(torch::cudss INTERFACE IMPORTED)
    target_include_directories(torch::cudss INTERFACE ${CUDSS_INCLUDE_PATH})
    target_link_libraries(torch::cudss INTERFACE ${CUDSS_LIBRARY_PATH})
  endif()
else()
  message(STATUS "USE_CUDSS is set to 0. Compiling without cuDSS support")
endif()

# cufile
if(CAFFE2_USE_CUFILE)
  add_library(torch::cufile INTERFACE IMPORTED)
  if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
    target_link_libraries(torch::cufile INTERFACE CUDA::cuFile_static CUDA::culibos)
  else()
    target_link_libraries(torch::cufile INTERFACE CUDA::cuFile CUDA::culibos)
  endif()
else()
  message(STATUS "USE_CUFILE is set to 0. Compiling without cuFile support")
endif()

# curand
add_library(torch::curand INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
  target_link_libraries(torch::curand INTERFACE CUDA::curand_static)
else()
  target_link_libraries(torch::curand INTERFACE CUDA::curand)
endif()

# cufft
add_library(torch::cufft INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
  target_link_libraries(torch::cufft INTERFACE CUDA::cufft_static_nocallback)
else()
  target_link_libraries(torch::cufft INTERFACE CUDA::cufft)
endif()

# nvrtc
add_library(torch::nvrtc INTERFACE IMPORTED)
if(CAFFE2_STATIC_LINK_CUDA AND NOT WIN32)
  set_property(
      TARGET torch::nvrtc PROPERTY INTERFACE_LINK_LIBRARIES
      CUDA::nvrtc_static CUDA::cuda_driver)
else()
  set_property(
      TARGET torch::nvrtc PROPERTY INTERFACE_LINK_LIBRARIES
      CUDA::nvrtc CUDA::cuda_driver)
endif()

# Add onnx namespace definition to nvcc
if(ONNX_NAMESPACE)
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  list(APPEND CUDA_NVCC_FLAGS "-DONNX_NAMESPACE=onnx_c2")
endif()

# Don't activate VC env again for Ninja generators with MSVC on Windows if CUDAHOSTCXX is not defined
# by adding --use-local-env.
if(MSVC AND CMAKE_GENERATOR STREQUAL "Ninja" AND NOT DEFINED ENV{CUDAHOSTCXX})
  list(APPEND CUDA_NVCC_FLAGS "--use-local-env")
endif()

# setting nvcc arch flags
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
# CMake 3.18 adds integrated support for architecture selection, but we can't rely on it
set(CMAKE_CUDA_ARCHITECTURES OFF)
list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA}")

# disable some nvcc diagnostic that appears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored
             field_without_dll_interface
             base_class_has_different_dll_interface
             dll_interface_conflict_none_assumed
             dll_interface_conflict_dllexport_assumed
             bad_friend_decl)
  list(APPEND SUPPRESS_WARNING_FLAGS --diag_suppress=${diag})
endforeach()
string(REPLACE ";" "," SUPPRESS_WARNING_FLAGS "${SUPPRESS_WARNING_FLAGS}")
list(APPEND CUDA_NVCC_FLAGS -Xcudafe ${SUPPRESS_WARNING_FLAGS})

set(CUDA_PROPAGATE_HOST_FLAGS_BLOCKLIST "-Werror")
if(MSVC)
  list(APPEND CUDA_NVCC_FLAGS "--Werror" "cross-execution-space-call")
  list(APPEND CUDA_NVCC_FLAGS "--no-host-device-move-forward")
endif()

# Debug and Release symbol support
if(MSVC)
  if(${CAFFE2_USE_MSVC_STATIC_RUNTIME})
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -Xcompiler /MTd")
    string(APPEND CMAKE_CUDA_FLAGS_MINSIZEREL " -Xcompiler /MT")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler /MT")
    string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -Xcompiler /MT")
  else()
    string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -Xcompiler /MDd")
    string(APPEND CMAKE_CUDA_FLAGS_MINSIZEREL " -Xcompiler /MD")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -Xcompiler /MD")
    string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -Xcompiler /MD")
  endif()
  if(CUDA_NVCC_FLAGS MATCHES "Zi")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-FS")
  endif()
elseif(CUDA_DEVICE_DEBUG)
  list(APPEND CUDA_NVCC_FLAGS "-g" "-G")  # -G enables device code debugging symbols
endif()

# Set expt-relaxed-constexpr to suppress Eigen warnings
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

# Set expt-extended-lambda to support lambda on device
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")

foreach(FLAG ${CUDA_NVCC_FLAGS})
  string(FIND "${FLAG}" " " flag_space_position)
  if(NOT flag_space_position EQUAL -1)
    message(FATAL_ERROR "Found spaces in CUDA_NVCC_FLAGS entry '${FLAG}'")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS " ${FLAG}")
endforeach()
