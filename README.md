# CUDA

Para instalar, revisa [Instalacion.md](Instalacion.md). Usa los pasos manuales con `nvcc` (sin CMake).

## Prueba rápida manual (Windows)

1) Abre **"Developer PowerShell for VS 2022"** (o en PowerShell normal ejecuta el comando de activación del entorno que se indica en Instalacion.md).
2) Ve a la carpeta donde tengas tu fuente (por ejemplo `ejercicios` si guardaste `hola.cu` allí).

```powershell
cd C:\Users\<tu_usuario>\Documents\Programacion\GitHub\CUDA\ejercicios
nvcc -o hola.exe hola.cu
./hola.exe
```

Deberías ver `CUDA OK`.

## Prueba rápida manual (WSL Ubuntu)

```bash
cd /ruta/donde/este/tu/hello
nvcc -o hello hello.cu
./hello
```
