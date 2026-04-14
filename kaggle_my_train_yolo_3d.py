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
        print("Conda failed or not found. Moving to Micromamba fallback...")

    # Level 3: Micromamba Fallback (Standalone binary)
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
        
    if not os.path.exists(os.path.join(MAMBA_ROOT, "bin", "python")):
        print(f"Creating Python {TARGET_PYTHON_VERSION} environment via micromamba...")
        subprocess.run([MAMBA_BIN, "create", "-y", "-p", MAMBA_ROOT, "-c", "conda-forge", f"python={TARGET_PYTHON_VERSION}"], check=True)
        
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
    # Your core venv logic (as requested)
    subprocess.run([base_python, "-m", "venv", VENV_PATH, "--without-pip"], check=True)
    
    VENV_PYTHON_TMP = os.path.join(VENV_PATH, "bin", "python")
    
    print("Downloading get-pip.py for manual pip installation...")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
    
    print("Installing pip into the virtual environment...")
    subprocess.run([VENV_PYTHON_TMP, "get-pip.py"], check=True)
    
    if os.path.exists("get-pip.py"):
        os.remove("get-pip.py")
    print("Pip successfully installed in venv!")

# Final absolute paths for subsequent cells
VENV_PYTHON = os.path.join(VENV_PATH, "bin", "python")
VENV_PIP = os.path.join(VENV_PATH, "bin", "pip")

# %% [code]
# 3. Install Dependencies (Following Installation.md strictly)
print("Step 3.1: Installing Torch and specialized base dependencies...")
!{VENV_PIP} install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
!{VENV_PIP} install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
!{VENV_PIP} install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

# MinkowskiEngine
print("Step 3.2: Installing MinkowskiEngine (recursive clone + custom compilation)...")
!cd /kaggle/working && git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
!cd /kaggle/working/MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
# Setting arch list for Kaggle GPUs (T4/P100)
!cd /kaggle/working/MinkowskiEngine && export TCNN_CUDA_ARCH_LIST="60;61;70;75;80;86" && {VENV_PYTHON} setup.py install --force_cuda --blas=openblas

# ScanNet Segmentator
print("Step 3.3: Compiling ScanNet Segmentator...")
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party && git clone https://github.com/ScanNet/ScanNet.git
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/ScanNet/Segmentator && git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && make

# pointnet2
print("Step 3.4: Installing pointnet2...")
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/pointnet2 && {VENV_PYTHON} setup.py install

# Finishing Mask3D and YOLO-World
print("Step 3.5: Finalizing Mask3D and YOLO-World local installs...")
!{VENV_PIP} install pytorch-lightning==1.7.2
!{VENV_PIP} install black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 pycocotools>=2.0.2 pydot iopath==0.1.7 loguru albumentations volumentations omegaconf==2.0.6 antlr4-python3-runtime==4.8 wrapt
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D && {VENV_PIP} install .
!cd /kaggle/working/my_Open_YOLO_3D/models/YOLO-World && {VENV_PIP} install -e .

# MM-ecosystem
print("Step 3.6: Installing MM-ecosystem and other utils...")
!{VENV_PIP} install openmim
!{VENV_PYTHON} -m mim install mmcv==2.0.0
!{VENV_PIP} install mmyolo==0.6.0 mmdet==3.0.0 plyfile supervision open3d pillow==9.1.0 pandas scipy tqdm pyyaml imageio scikit-learn

# %% [code]
# 4. Run the training script matching the Evaluation Protocol
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
