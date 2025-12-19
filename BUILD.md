# Build rápido del proyecto base CUDA

Este proyecto incluye un ejemplo mínimo `src/hello.cu` y un `CMakeLists.txt` para compilar en Windows nativo o en WSL Ubuntu con CUDA 13.x.

## Windows (nativo)

Usa Visual Studio 2022 o CMake desde PowerShell.

Opción A — Generator Visual Studio:

```powershell
mkdir build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
./build/Release/cuda_hello.exe
./build/Release/vec_add.exe
./build/Release/saxpy.exe
./build/Release/reduction.exe
```

Opción B — Generator por defecto (Ninja/NMake, si disponible):

```powershell
mkdir build
cmake -S . -B build
cmake --build build
./build/cuda_hello.exe
./build/vec_add.exe
./build/saxpy.exe
./build/reduction.exe
```

## WSL Ubuntu

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
./build/cuda_hello
./build/vec_add
./build/saxpy
./build/reduction
```

## Alternativa: `nvcc` directo (sin CMake)

Windows PowerShell:

```powershell
nvcc -o hello.exe src/hello.cu
./hello.exe
nvcc -o vec_add.exe src/vec_add.cu
./vec_add.exe
nvcc -o saxpy.exe src/saxpy.cu
./saxpy.exe
nvcc -o reduction.exe src/reduction.cu
./reduction.exe
```

WSL Ubuntu:

```bash
nvcc -o hello src/hello.cu
./hello
nvcc -o vec_add src/vec_add.cu
./vec_add
nvcc -o saxpy src/saxpy.cu
./saxpy
nvcc -o reduction src/reduction.cu
./reduction
```

## (Opcional) Fijar arquitecturas CUDA

Si quieres optimizar para tu GPU específica, puedes editar `CMakeLists.txt` y añadir, por ejemplo:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86) # Ajusta según tu GPU (p.ej., 86, 89, 90)
```

Consulta `nvidia-smi` y la documentación de tu GPU para elegir el número correcto.
