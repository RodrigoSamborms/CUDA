# CUDA

Para instalar, revisa [Instalacion.md](Instalacion.md). Este repo incluye un proyecto base mínimo para probar tu entorno CUDA 13.x.

## Estructura
- [CMakeLists.txt](CMakeLists.txt): definición del proyecto `cuda_hello`.
- [src/hello.cu](src/hello.cu): kernel mínimo y verificación.
- [BUILD.md](BUILD.md): instrucciones de compilación para Windows y WSL.
- [CMakePresets.json](CMakePresets.json): presets para configurar y compilar.
- [.vscode/tasks.json](.vscode/tasks.json): tareas de VS Code para configurar, compilar y ejecutar.

## Prueba rápida (Windows)

> **Importante:** Abre **"Developer PowerShell for VS 2022"** desde el menú Inicio. Esto carga las herramientas de MSVC necesarias para compilar correctamente.

Navega al directorio raíz del proyecto:

```powershell
cd C:\Users\<tu_usuario>\Documents\Programacion\GitHub\CUDA
```

Con Visual Studio 2022:

```powershell
mkdir build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\build\Release\cuda_hello.exe
```

Con generator por defecto:

```powershell
mkdir build
cmake -S . -B build
cmake --build build
.\build\cuda_hello.exe
```

## Prueba rápida (WSL Ubuntu)

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
./build/cuda_hello
```

Si aún se está instalando el Toolkit 13.x, usa estos comandos cuando termine y confirma que `nvcc --version` funciona.
