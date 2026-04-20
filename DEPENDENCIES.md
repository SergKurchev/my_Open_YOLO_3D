# Project Dependencies

This document outlines the strict dependency requirements for the **Open-YOLO-3D** and **ODIN** projects, specifically optimized for environments like Kaggle.

## 1. System Requirements (Compilation Toolchain)

Due to the sensitive nature of C++ extensions like **MinkowskiEngine**, a specific toolchain is required to avoid ABI (Application Binary Interface) mismatches.

- **Operating System**: Linux (tested on Ubuntu 20.04/22.04)
- **Python**: 3.10.x (Mandatory for compatibility with Torch 1.12.1)
- **C++ Compiler**: GCC 9.x / G++ 9.x
- **CUDA Toolkit**: 11.3 (Must match the PyTorch CUDA version)
- **BLAS Library**: OpenBLAS or MKL (Must be ABI-compatible with the compiler)
- **Python Headers**: `python3.10-dev`

> [!IMPORTANT]
> In Kaggle or environments with newer system compilers (GCC 11+), it is highly recommended to use an isolated environment (e.g., via `micromamba`) to provision GCC 9 and CUDA 11.3 manually.

## 2. Core Machine Learning Stack

The project is built on a specific version of PyTorch to ensure stability with legacy extensions.

- **PyTorch**: `1.12.1+cu113`
- **Torchvision**: `0.13.1+cu113`
- **Torchaudio**: `0.12.1+cu113`

Installation Command:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## 3. Specialized Libraries

### MinkowskiEngine
- **Version**: Recommended commit `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`
- **Compilation Flags**: Requires `FORCE_CUDA=1` and explicit BLAS paths if using a custom toolchain.

### Detectron2
- **Version**: 0.6 (or built from source)
- **Dependencies**: `fvcore`, `iopath`, `pycocotools`

## 4. Python Dependencies

The following packages are required for data processing, logging, and training utility:

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `ninja` | latest | Required for building C++ extensions |
| `Cython` | < 3.0 | Required for MinkowskiEngine |
| `yacs` | latest | Configuration management |
| `termcolor` | latest | Logging |
| `tabulate` | latest | Results formatting |
| `imageio` | latest | Image loading |
| `scipy` | latest | Mathematical operations |
| `pandas` | latest | Data management |
| `opencv-python`| latest | Image processing |
| `setuptools` | 69.5.1 | Precision pinning for metadata compatibility |
| `pip` | < 24.1 | Compatibility with certain package metadata |

## 5. Environment Isolation & Troubleshooting (Kaggle Specific)

To successfully compile the C++ extensions in a modern Kaggle environment, the following configuration "tricks" are implemented in the training scripts:

### A. Physical Header Symlinking (The `pyconfig.h` fix)
Kaggle's system Python (3.10) splits headers across architecture-independent and architecture-dependent directories. To ensure the isolated compiler finds everything:
- **Challenge**: `nvcc` often ignores `CPATH` for certain internal includes.
- **Fix**: Physically symlink `/usr/include/x86_64-linux-gnu/python3.10/*` into a subdirectory within the `mamba_env/include`. This forces the compiler to resolve the path `<x86_64-linux-gnu/python3.10/pyconfig.h>` successfully.

### B. Library Priority (The OpenBLAS/gfortran fix)
System libraries in Ubuntu 22.04+ (like OpenBLAS) are built with modern `gfortran` (version 5+), which is incompatible with the older GCC 9 toolchain required by MinkowskiEngine.
- **Challenge**: The linker might pick up system `libopenblas.so` and fail due to missing `_gfortran` symbols.
- **Fix**: Enforce strict library prioritization using three mechanisms simultaneously:
    1. `LD_LIBRARY_PATH`: Points to `mamba_env/lib` first.
    2. `LIBRARY_PATH`: Points to `mamba_env/lib` first.
    3. `LDFLAGS`: Explicitly adds `-L{MAMBA_ROOT}/lib -Wl,-rpath,{MAMBA_ROOT}/lib` to the start of the linking command.

### C. Workaround for NVCC Parser Errors (`__s128` error)
Modern Linux system headers (e.g., in Ubuntu 22.04) use GCC extensions like `__int128` that older `nvcc` versions (like 11.3) cannot parse during their preprocessing phase.
- **Challenge**: Compilation fails with `error: expected initializer before '__s128'` in `linux/types.h`.
- **Fix**: Inject `-D__int128=int` into the `nvcc` flags (via `NVCC_FLAGS` and `CUDAFLAGS`). This tells the internal parser to treat the 128-bit type as a standard `int`. Since MinkowskiEngine does not use 128-bit integers in its CUDA kernels, this is a safe and effective workaround.

### D. Global `CUDA_HOME` Enforcement
PyTorch's `cpp_extension` module often defaults to `/usr/local/cuda`. On Kaggle, this points to CUDA 12.x, which is incompatible with our target `nvcc 11.3`.
- **Challenge**: Mixing CUDA 11.3 compiler with CUDA 12.x headers leads to mysterious parsing and linking errors.
- **Fix**: Forcefully set `os.environ["CUDA_HOME"]` and `os.environ["CUDA_PATH"]` to the `mamba_env` path **globally** within the training script before any build commands are invoked.

### E. Strict Toolchain Pinning
To prevent "version bleeding" from conda-forge, the environment uses specific version pinning:
- `sysroot_linux-64=2.17`: Ensures compatibility with the older GLIBC standards needed by GCC 9.
- `libgcc-ng=9`, `libstdcxx-ng=9`: Prevents the environment from pulling in modern runtime headers (GCC 12/13) that trigger the `__s128` error.

---
*Note: This configuration is essential for training on the Multiview Strawberry Dataset using the provided scripts.*
