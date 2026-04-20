# %% [markdown]
# # Open-YOLO-3D Fine-Tuning on Kaggle
# This notebook clones the Open-YOLO-3D repository, creates a virtual environment, 
# installs dependencies, and runs the `my_train_yolo_3d.py` training script on the Multiview Strawberry Dataset.

# %% [code]
# 1. Сlode the repository
!git clone https://github.com/SergKurchev/my_Open_YOLO_3D.git /kaggle/working/my_Open_YOLO_3D

# %% [code]
# 2. Robust Virtual Environment Creation (Handling Python 3.12+ host & Missing Conda)
import os
import subprocess
import sys
import urllib.request
import tarfile

print(f"Host Python version: {sys.version}")

TARGET_PYTHON_VERSION = "3.10.9"

# ── ALWAYS provision micromamba for compilation toolchain ──────────────────────
# Even if system Python 3.10 exists, we MUST have CUDA 11.3 nvcc for compiling
# extensions against torch 1.12.1+cu113. System CUDA may be 12.x which causes
# a hard version-check failure inside torch.utils.cpp_extension.
MAMBA_BIN = "/kaggle/working/micromamba"
MAMBA_ROOT = "/kaggle/working/mamba_env"

if not os.path.exists(MAMBA_BIN):
    print("Downloading micromamba standalone binary...")
    url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
    urllib.request.urlretrieve(url, "micromamba.tar.bz2")
    with tarfile.open("micromamba.tar.bz2", "r:bz2") as tar:
        tar.extract("bin/micromamba")
    os.rename("bin/micromamba", MAMBA_BIN)
    os.chmod(MAMBA_BIN, 0o755)

if not os.path.exists(os.path.join(MAMBA_ROOT, "bin", "nvcc")) or \
   not os.path.exists(os.path.join(MAMBA_ROOT, "include", "cuda_runtime.h")):
    print("Provisioning gcc-9 + cuda-nvcc 11.3 + openblas via micromamba (needed to compile against torch+cu113)...")
    # Also ensure system python3.10-dev headers are present for the linker/compiler
    subprocess.run(["apt-get", "update"], check=False)
    subprocess.run(["apt-get", "install", "-y", "python3.10-dev"], check=False)
    
    subprocess.run([MAMBA_BIN, "create", "-y", "-p", MAMBA_ROOT,
                    "-c", "conda-forge",
                    "gcc_linux-64=9", "gxx_linux-64=9",
                    "libgcc-ng=9", "libstdcxx-ng=9",
                    "sysroot_linux-64=2.17",
                    "cudatoolkit-dev=11.3.1", "openblas"], check=True)
    print(f"✓ nvcc 11.3 + gcc-9 + openblas available at {MAMBA_ROOT}/bin/")
else:
    print(f"✓ Mamba env with nvcc already exists at {MAMBA_ROOT}")
    # Force add openblas just in case it was created earlier without it
    if not os.path.exists(os.path.join(MAMBA_ROOT, "lib", "libopenblas.so")):
        print("Adding missing openblas to mamba_env...")
        subprocess.run([MAMBA_BIN, "install", "-y", "-p", MAMBA_ROOT,
                        "-c", "conda-forge", "openblas"], check=True)

# ──────────────────────────────────────────────────────────────────────────────
# SELF-HEALING: Locate and symlink everything for PyTorch's strict CUDA_HOME
cuda_h_paths = subprocess.run(["find", MAMBA_ROOT, "-name", "cuda_runtime.h"], capture_output=True, text=True).stdout.strip().split('\n')
cuda_h_path = cuda_h_paths[0] if cuda_h_paths and cuda_h_paths[0] else None

if cuda_h_path:
    actual_include_dir = os.path.dirname(cuda_h_path)
    expected_include_dir = os.path.join(MAMBA_ROOT, "include")
    if actual_include_dir != expected_include_dir:
        os.makedirs(expected_include_dir, exist_ok=True)
        subprocess.run(f"ln -sf {actual_include_dir}/* {expected_include_dir}/", shell=True, check=True)
    
    # Also link lib64 which PyTorch expects
    actual_lib_dir = os.path.join(os.path.dirname(actual_include_dir), "lib")
    expected_lib64_dir = os.path.join(MAMBA_ROOT, "lib64")
    if os.path.exists(actual_lib_dir) and not os.path.exists(expected_lib64_dir):
        subprocess.run(f"ln -sf {actual_lib_dir} {expected_lib64_dir}", shell=True, check=True)
        print(f"\u26a0\ufe0f Symlinked CUDA include and lib64 directories.")

    # NEW: Physical symlink for Python architecture-specific headers (pyconfig.h)
    # This solves the 'x86_64-linux-gnu/python3.10/pyconfig.h' missing error physically.
    target_py_inc = os.path.join(MAMBA_ROOT, "include", "x86_64-linux-gnu", "python3.10")
    if not os.path.exists(target_py_inc):
        os.makedirs(target_py_inc, exist_ok=True)
        # We symlink the entire directory contents to satisfy relative includes
        subprocess.run(f"ln -sf /usr/include/x86_64-linux-gnu/python3.10/* {target_py_inc}/", shell=True, check=False)
        print(f"\u26a0\ufe0f Symlinked system Python arch-headers into mamba_env.")
else:
    print(f"\u274c FATAL: cuda_runtime.h NOT FOUND IN {MAMBA_ROOT}. Compilation will fail!")

# COMPILE_ENV: used for all setup.py / cmake calls that compile CUDA extensions
# IMPORTANT: We set these globally in os.environ to ensure torch/cpp_extension finds our 11.3 toolkit
os.environ["CUDA_HOME"] = MAMBA_ROOT
os.environ["CUDA_PATH"] = MAMBA_ROOT

COMPILE_ENV = os.environ.copy()
COMPILE_ENV["CUDA_HOME"] = MAMBA_ROOT
COMPILE_ENV["CUDA_PATH"] = MAMBA_ROOT
COMPILE_ENV["PATH"] = f"{MAMBA_ROOT}/bin:" + os.environ.get("PATH", "")
COMPILE_ENV["CC"] = f"{MAMBA_ROOT}/bin/x86_64-conda-linux-gnu-gcc"
COMPILE_ENV["CXX"] = f"{MAMBA_ROOT}/bin/x86_64-conda-linux-gnu-g++"

# WORKAROUND for nvcc frontend __int128 parsing error on modern sysroots
# This forces the parser to treat __int128 as a standard int during the pre-processing phase.
COMPILE_ENV["NVCC_FLAGS"] = "-D__int128=int " + os.environ.get("NVCC_FLAGS", "")
# Some build systems use CUDAFLAGS instead
COMPILE_ENV["CUDAFLAGS"] = "-D__int128=int " + os.environ.get("CUDAFLAGS", "")

# FORCE priority for our isolated mamba libraries to avoid system ABI conflicts
# This ensures linker finds mamba's libopenblas and libgfortran.so.4 first.
_mamba_lib = os.path.join(MAMBA_ROOT, "lib")
COMPILE_ENV["LIBRARY_PATH"] = f"{_mamba_lib}:" + os.environ.get("LIBRARY_PATH", "")
COMPILE_ENV["LD_LIBRARY_PATH"] = f"{_mamba_lib}:" + os.environ.get("LD_LIBRARY_PATH", "")

# FIX for pyconfig.h: Include system Python 3.10 dev paths
# Ubuntu/Debian splits headers across these two directories
_py_inc = "/usr/include/python3.10"
_py_inc_arch = "/usr/include/x86_64-linux-gnu/python3.10"
COMPILE_ENV["CPATH"] = f"{_py_inc}:{_py_inc_arch}:" + os.environ.get("CPATH", "")
COMPILE_ENV["C_INCLUDE_PATH"] = f"{_py_inc}:{_py_inc_arch}:" + os.environ.get("C_INCLUDE_PATH", "")
COMPILE_ENV["CPLUS_INCLUDE_PATH"] = f"{_py_inc}:{_py_inc_arch}:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
# ──────────────────────────────────────────────────────────────────────────────

def get_python310_base():
    """Utility to find or provision Python 3.10 on Kaggle."""
    # Level 1: System Check
    print("Searching for Python 3.10 in system...")
    for path in ["/usr/bin/python3.10", "/usr/local/bin/python3.10"]:
        if os.path.exists(path):
            print(f"Found system Python 3.10 at {path}")
            return path

    # Level 2: Conda Check
    PY310_PATH = "/kaggle/working/conda_py310"
    if os.path.exists(os.path.join(PY310_PATH, "bin", "python")):
        return os.path.join(PY310_PATH, "bin", "python")

    conda_bin = "/opt/conda/bin/conda" if os.path.exists("/opt/conda/bin/conda") else "conda"
    try:
        print(f"Attempting to provision Python 3.10 via {conda_bin}...")
        subprocess.run([conda_bin, "create", "-y", "-p", PY310_PATH, f"python={TARGET_PYTHON_VERSION}"],
                       check=True, capture_output=True)
        return os.path.join(PY310_PATH, "bin", "python")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Conda failed or not found. Using mamba_env python...")

    # Level 3: reuse mamba_env python (already provisioned above)
    if os.path.exists(os.path.join(MAMBA_ROOT, "bin", "python")):
        return os.path.join(MAMBA_ROOT, "bin", "python")
    subprocess.run([MAMBA_BIN, "install", "-y", "-p", MAMBA_ROOT,
                    "-c", "conda-forge", f"python={TARGET_PYTHON_VERSION}"], check=True)
    return os.path.join(MAMBA_ROOT, "bin", "python")

# Decide base Python
base_python = sys.executable
if sys.version_info >= (3, 11):
    print("Host Python is 3.11+. We need Python 3.10 for Torch 1.12.1 compatibility.")
    base_python = get_python310_base()
    print(f"Base Python set to: {base_python}")

VENV_PATH = "/kaggle/working/openyolo_env"

if not os.path.exists(VENV_PATH):
    print(f"Creating virtual environment at {VENV_PATH} using {base_python}...")
    subprocess.run([base_python, "-m", "venv", VENV_PATH, "--without-pip"], check=True)

    VENV_PYTHON_TMP = os.path.join(VENV_PATH, "bin", "python")

    print("Downloading get-pip.py for manual pip installation...")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")

    print("Installing and pinning pip < 24.1 + setuptools + wheel...")
    # NOTE: Downgrading pip is REQUIRED for pytorch-lightning 1.7.2 metadata compatibility
    subprocess.run([VENV_PYTHON_TMP, "get-pip.py", "pip<24.1", "setuptools==69.5.1", "wheel"], check=True)

    if os.path.exists("get-pip.py"):
        os.remove("get-pip.py")
    print("Pip and build tools successfully installed in venv!")

# Final absolute paths for subsequent cells
VENV_PYTHON = os.path.join(VENV_PATH, "bin", "python")
VENV_PIP = os.path.join(VENV_PATH, "bin", "pip")

# %% [code]
# 3. Install Dependencies (Following Installation.md strictly)
print("Step 3.1: Installing Torch and specialized base dependencies...")
# CRITICAL: We MUST pin setuptools < 70.0.0 because MinkowskiEngine and PointNet2
# rely on 'pkg_resources' which was removed in newer setuptools.
!{VENV_PIP} install -U "setuptools==69.5.1" wheel packaging cython

# Install Torch and Lightning together to ensure no accidental upgrades
!{VENV_PIP} install torch==1.12.1+cu113 torchvision==0.13.1+cu113 pytorch-lightning==1.7.2 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Pre-compiled torch-scatter
!{VENV_PIP} install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# Detectron2 - clone and install without CUDA compilation
# Only pure-Python modules are needed: detectron2.utils.comm, detectron2.projects.point_rend
!{VENV_PIP} install ninja fvcore iopath pycocotools
!cd /tmp && rm -rf detectron2_src && git clone --depth 1 https://github.com/facebookresearch/detectron2.git detectron2_src
_d2_env = os.environ.copy()
_d2_env["BUILD_NCCL"] = "0"
_d2_env["TORCH_CUDA_ARCH_LIST"] = ""   # empty string = skip all CUDA kernel compilation
_d2_env["FORCE_CUDA"] = "0"
try:
    subprocess.run([VENV_PYTHON, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd="/tmp/detectron2_src", env=_d2_env, check=True)
    print("✓ detectron2 installed via pip (no-CUDA mode)")
except subprocess.CalledProcessError:
    # Fallback: inject repo path directly into venv via .pth file
    # Works because only pure-Python modules are used from detectron2
    import glob
    site_pkgs = glob.glob(os.path.join(VENV_PATH, "lib", "python3.*", "site-packages"))[0]
    pth_file = os.path.join(site_pkgs, "detectron2_repo.pth")
    with open(pth_file, "w") as f:
        f.write("/tmp/detectron2_src\n")
    print(f"✓ detectron2 path injected via {pth_file} (fallback mode)")

# ── DIAGNOSTICS ────────────────────────────────────────────────────────────────
print("=" * 60)
print("DIAGNOSTIC: Compiler & CUDA environment")
print("=" * 60)
print("--- System Default (for reference) ---")
!gcc --version | head -n 1
!g++ --version | head -n 1
print("--- Isolated Mamba Toolchain (used for build) ---")
!{MAMBA_ROOT}/bin/x86_64-conda-linux-gnu-gcc --version | head -n 1
!{MAMBA_ROOT}/bin/x86_64-conda-linux-gnu-g++ --version | head -n 1
print("--- mamba nvcc ---")
!{MAMBA_ROOT}/bin/nvcc --version | grep "release"
print("--- environment check ---")
!{VENV_PYTHON} -c "import torch; print('torch:', torch.__version__); print('CUDA avail:', torch.cuda.is_available()); from torch.utils.cpp_extension import CUDA_HOME; print('CUDA_HOME:', CUDA_HOME)"
!ls -l /usr/include/x86_64-linux-gnu/python3.10/pyconfig.h
print("=" * 60)
# ───────────────────────────────────────────────────────────────────────────────

# MinkowskiEngine
print("Step 3.2: Installing MinkowskiEngine (recursive clone + custom compilation)...")
!cd /kaggle/working && rm -rf MinkowskiEngine && git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
!cd /kaggle/working/MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
_mink_env = COMPILE_ENV.copy()
_mink_env["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6"
_mink_env["MAX_JOBS"] = "4"
# Prioritize mamba lib paths even more strictly in LDFLAGS safely
_mink_env["LDFLAGS"] = f"-L{MAMBA_ROOT}/lib -Wl,-rpath,{MAMBA_ROOT}/lib " + _mink_env.get("LDFLAGS", "")
# Fallback for system gfortran 5 IF it's absolutely needed as a dependency of some other sys-lib
_mink_env["LDFLAGS"] += " -Wl,-rpath-link,/usr/lib/x86_64-linux-gnu"

try:
    # We explicitly use the ABI-compatible openblas that we just installed in mamba_env!
    subprocess.run([
        VENV_PYTHON, "setup.py", "install", 
        "--force_cuda", 
        "--blas=openblas",
        f"--blas_include_dirs={MAMBA_ROOT}/include",
        f"--blas_library_dirs={MAMBA_ROOT}/lib"
    ], cwd="/kaggle/working/MinkowskiEngine", env=_mink_env, check=True)
    print("✓ MinkowskiEngine installed!")
except subprocess.CalledProcessError:
    print("\n\u274c MinkowskiEngine Compilation FAILED! Fetching raw Ninja error logs:")
    subprocess.run("cat /kaggle/working/MinkowskiEngine/build/temp.linux-x86_64-*/build.ninja 2>/dev/null | head -n 20", shell=True)
    print("Trying manual setup.py build_ext to capture raw error:")
    subprocess.run([VENV_PYTHON, "setup.py", "build_ext"],
        cwd="/kaggle/working/MinkowskiEngine", env=_mink_env)
    raise

# ScanNet Segmentator
print("Step 3.3: Compiling ScanNet Segmentator...")
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party && rm -rf ScanNet && git clone https://github.com/ScanNet/ScanNet.git
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/ScanNet/Segmentator && git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
subprocess.run(["make"],
    cwd="/kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/ScanNet/Segmentator",
    env=COMPILE_ENV, check=True)

# pointnet2
print("Step 3.4: Installing pointnet2...")
subprocess.run([VENV_PYTHON, "setup.py", "install"],
    cwd="/kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/pointnet2",
    env=COMPILE_ENV, check=True)

# MM-ecosystem
print("Step 3.5: Installing MM-ecosystem and other utils...")
!{VENV_PIP} install openmim gdown
!{VENV_PIP} install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
!{VENV_PYTHON} -m mim install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
!{VENV_PIP} install mmyolo==0.6.0 mmdet==3.0.0 plyfile supervision open3d pillow==9.1.0 pandas scipy tqdm pyyaml imageio scikit-learn torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Finishing Mask3D and YOLO-World
print("Step 3.6: Finalizing Mask3D and YOLO-World local installs...")
!{VENV_PIP} install black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 pycocotools>=2.0.2 pydot iopath==0.1.7 loguru albumentations volumentations omegaconf==2.0.6 antlr4-python3-runtime==4.8 wrapt transformers tokenizers mmengine==0.7.1
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D && {VENV_PIP} install .
!cd /kaggle/working/my_Open_YOLO_3D/models/YOLO-World && {VENV_PIP} install -e . --no-deps

# %% [code]
# 4. Download Checkpoints Only (skip class-agnostic ScanNet200 masks - not needed for custom dataset)
# Both scripts download the same huge OpenYOLO3D.zip (~several GB).
# We only need checkpoints, so we download once, extract only what we need, and delete the zip.
print("Step 4: Downloading Pretrained Checkpoints (disk-efficient)...")
import subprocess

ZIP_PATH = "/kaggle/working/OpenYOLO3D.zip"
EXTRACT_DIR = "/kaggle/working"

if not os.path.exists("/kaggle/working/my_Open_YOLO_3D/pretrained/checkpoints"):
    print("Downloading OpenYOLO3D.zip via gdown...")
    subprocess.run([VENV_PYTHON, "-m", "gdown", "1FneLaYaClWDO51L9lIvlTQbheh5SfOFD", "-O", ZIP_PATH], check=True)
    # Extract ONLY checkpoints (skip masks to save ~10+ GB of disk)
    os.makedirs("/kaggle/working/my_Open_YOLO_3D/pretrained/checkpoints", exist_ok=True)
    print("Extracting checkpoints only (skipping large ScanNet200 masks)...")
    subprocess.run(["unzip", "-j", ZIP_PATH, "OpenYOLO3D/checkpoints/*", "-d", "/kaggle/working/my_Open_YOLO_3D/pretrained/checkpoints"], check=True)
    if os.path.exists(ZIP_PATH):
        print("Removing zip to free disk space...")
        os.remove(ZIP_PATH)

print(f"Disk usage after setup:")
!df -h /kaggle/working

# 5. Run the training script matching the Evaluation Protocol
import os

# Paths updated according to user feedback
dataset_path = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset"
splits_path = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json"

assert os.path.exists(dataset_path), f"Dataset path not found: {dataset_path}"
assert os.path.exists(splits_path), f"Splits path not found: {splits_path}"

# Running from root of cloned repo
run_cmd = f"cd /kaggle/working/my_Open_YOLO_3D && {VENV_PYTHON} my_train_yolo_3d.py --data_path {dataset_path} --splits_path {splits_path} --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4"
print(f"Executing: {run_cmd}")

# %% [code]
!cd /kaggle/working/my_Open_YOLO_3D && {VENV_PYTHON} my_train_yolo_3d.py --data_path /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset --splits_path /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4
