cmake_minimum_required(VERSION 3.21)
project(cuda_hello VERSION 0.1.0 LANGUAGES CXX CUDA)

# Estándar de C++ y CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Tipo de build por defecto
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Ejecutable mínimo de CUDA
add_executable(cuda_hello src/hello.cu)
add_executable(vec_add src/vec_add.cu)
add_executable(saxpy src/saxpy.cu)
add_executable(reduction src/reduction.cu)

# Características de compilación
target_compile_features(cuda_hello PUBLIC cxx_std_17)

# Nota: Deja CMAKE_CUDA_ARCHITECTURES sin especificar para usar defaults del toolkit.
# Puedes fijarlo manualmente si lo deseas (p.ej., 86;89;90 según tu GPU).
