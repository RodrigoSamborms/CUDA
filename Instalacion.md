# Instalación de CUDA 13.x en Windows 11 (Nativo y WSL Ubuntu)

Esta guía te ayuda a instalar y verificar CUDA Toolkit 13.x tanto en Windows 11 nativo como en WSL (Ubuntu). Incluye requisitos, pasos, verificación y pruebas mínimas.

## Requisitos previos
- GPU NVIDIA compatible con CUDA (consulta el modelo en el Administrador de dispositivos de Windows o en la página del fabricante).
- Windows 11 actualizado (22H2 o superior recomendado).
- Permisos de administrador en el sistema.
- Espacio en disco: ~5–10 GB para herramientas y samples.

Opcional pero recomendado (Windows nativo):
- Visual Studio 2022 o "Build Tools for Visual Studio 2022" con el workload "Desktop development with C++" (CUDA integra toolchains y proyectos con VS).

## Opción A — Windows 11 (nativo)

1) Instalar/Actualizar el driver NVIDIA
- Descarga e instala el último controlador NVIDIA (Game Ready o Studio). La versión más reciente suele incluir soporte WSL también.
- Tras instalar, reinicia. Luego, en PowerShell:

```powershell
nvidia-smi
```

Si ves una tabla con tu GPU y versión del driver, el controlador está OK.

2) Instalar Visual Studio (C++)
- Si usas Visual Studio: instala la carga "Desktop development with C++".
- Alternativa mínima: "Build Tools for Visual Studio 2022" con MSVC, Windows SDK y CMake.

3) Instalar CUDA Toolkit
- Descarga el instalador de CUDA Toolkit 13.x para Windows 11 (x86_64) desde NVIDIA.
- Ejecuta el instalador y elige "Custom" para confirmar que incluye:
  - Driver (si no lo tenías o está desactualizado)
  - CUDA Toolkit
  - Integración con Visual Studio
  - Samples (opcional pero útil)
- El instalador configura PATH automáticamente. Reinicia si te lo pide.

4) Verificar instalación
- Abre PowerShell y comprueba:

```powershell
nvcc --version
```

Deberías ver la versión de `nvcc` (por ejemplo, 13.0 o superior 13.x). También:

```powershell
nvidia-smi
```

5) Probar con un programa mínimo
Crea un archivo `hello.cu` con el siguiente contenido (puedes usar cualquier carpeta de trabajo):

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {}

int main() {
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    std::printf("CUDA OK\n");
    return 0;
}
```

Compila y ejecuta:

> **Importante:** Abre **"Developer PowerShell for VS 2022"** desde el menú Inicio (o activa el entorno en tu terminal PowerShell con `& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64`). Esto carga las herramientas de MSVC necesarias para que `nvcc` pueda vincular correctamente.

```powershell
nvcc -o hello.exe hello.cu
./hello.exe
```

Salida esperada:

```
CUDA OK
```

6) (Opcional) Instalar cuDNN
- Crea una cuenta de NVIDIA Developer, descarga cuDNN para Windows y la versión de CUDA instalada.
- Descomprime y copia el contenido en las carpetas correspondientes del Toolkit (típicamente en `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\`).

7) (Opcional) Nsight Compute/Systems
- Para perfilar y analizar rendimiento, instala Nsight Compute y/o Nsight Systems desde NVIDIA.

8) Desinstalación (Windows)
- Panel de Control / Configuración → Aplicaciones → desinstala "NVIDIA CUDA Toolkit" y, si corresponde, los componentes relacionados. El driver se puede desinstalar aparte si lo necesitas.

## Opción B — WSL Ubuntu (Windows Subsystem for Linux)

Usa esta opción si prefieres un entorno Linux bajo Windows, o si tus dependencias y toolchains están orientadas a Linux.

1) Instalar WSL2 con Ubuntu
En PowerShell (Administrador):

```powershell
wsl --install -d Ubuntu
```

Reinicia si es necesario. Abre Ubuntu y crea tu usuario.

2) Driver NVIDIA con soporte WSL
- Mantén actualizado el driver NVIDIA para Windows (los drivers recientes incluyen soporte WSL2). Verifica luego en WSL:

```bash
nvidia-smi
```

Deberías ver la salida del driver desde WSL.

3) Instalar CUDA Toolkit en Ubuntu (WSL)
- Añade el repositorio oficial de CUDA y luego instala el toolkit (ejemplo para Ubuntu 22.04). Puedes instalar la última 13.x disponible con el metapaquete `cuda-toolkit` o fijar una versión menor específica (por ejemplo, `cuda-toolkit-13-0`):

```bash
sudo apt update
sudo apt install -y wget gnupg
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
# Opción 1: siempre la última estable del repo (recomendado)
sudo apt -y install cuda-toolkit
# Opción 2: versión específica 13.x (ajusta si hay 13.1/13.2, etc.)
# sudo apt -y install cuda-toolkit-13-0
```

- Añade CUDA al PATH (y a la librería) en tu shell:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

4) Verificar en WSL

```bash
nvcc --version
nvidia-smi
```

5) Probar con un programa mínimo en WSL
Crea `hello.cu`:

```cpp
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {}

int main() {
    kernel<<<1,1>>>();
    cudaDeviceSynchronize();
    std::printf("CUDA OK (WSL)\n");
    return 0;
}
```

Compila y ejecuta:

```bash
nvcc -o hello hello.cu
./hello
```

Salida esperada:

```
CUDA OK (WSL)
```

6) VS Code con WSL
- Instala la extensión "Remote - WSL" en VS Code.
- Abre la carpeta en WSL (Command Palette → "Remote-WSL: Open Folder in WSL...").
- Compila desde el terminal de WSL usando `nvcc` y tus herramientas Linux.

7) Desinstalación (WSL)

```bash
sudo apt remove --purge -y cuda-toolkit-12-4 cuda-*
sudo apt autoremove -y
```

(Elimina también `cuda-keyring` si lo deseas.)

## ¿Qué opción elegir?
- Si usas Visual Studio y Windows puro: Opción A (nativo) es más directa.
- Si prefieres Linux, toolchains de Linux o reproducibilidad en Linux: Opción B (WSL) es excelente.
- Puedes tener ambas: driver único en Windows, toolkits separados.

## Problemas comunes
- `nvcc` no se reconoce (Windows): reinicia la sesión o agrega manualmente `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\bin` al PATH del usuario/sistema.
- `nvidia-smi` no aparece: asegúrate de que el driver NVIDIA está correctamente instalado (Windows) o que WSL detecta la GPU (drivers recientes, WSL2 habilitado).
- Error de compilación por Visual Studio no encontrado: instala VS 2022 o Build Tools con MSVC C++.
- CUDA versión/driver incompatibles: usa versiones alineadas (Toolkit 13.x con drivers recientes). La guía oficial de CUDA indica combinaciones compatibles.

---

Con esto tendrás CUDA 13.x operativo en Windows y/o en WSL. Si vas a seguir el plan de aprendizaje, valida que puedes compilar y ejecutar el `hello.cu` antes de continuar.
