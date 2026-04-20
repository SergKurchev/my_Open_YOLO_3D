# %% [markdown]
# # Open-YOLO-3D Modernized Training Pipeline
# This notebook implements the modernized Open-YOLO-3D training and evaluation protocol on Kaggle.
#
# **Features:**
# - **Python 3.12 / Torch 2.3+ / CUDA 12.1**
# - **SpConv 2.x** backend (no MinkowskiEngine required)
# - **2D-to-3D GT Projection**: Automatic consensus voting to generate 3D Ground Truth from 2D masks.
# - **Evaluation Protocol**: Full support for PQ, SQ, RQ, and mAP metrics.

# %% [code]
# 1. Clone Repository and Enter Directory
import os
if not os.path.exists('my_Open_YOLO_3D'):
    print("Cloning repository...")
    os.system('git clone -b modernize_repo https://github.com/SergKurchev/my_Open_YOLO_3D.git')

# Move into the repository directory if we are not already there
if os.path.basename(os.getcwd()) != 'my_Open_YOLO_3D':
    if os.path.exists('my_Open_YOLO_3D'):
        os.chdir('my_Open_YOLO_3D')
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print("ERROR: Repository directory 'my_Open_YOLO_3D' not found!")

# %% [code]
import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    # Ensure models/Mask3D is in PYTHONPATH for the subprocess
    env = os.environ.copy()
    root_dir = os.getcwd()
    mask3d_path = os.path.join(root_dir, "models", "Mask3D")
    env["PYTHONPATH"] = f"{root_dir}:{mask3d_path}:" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, shell=True, check=True, env=env)

# %% [markdown]
# ## 1. Environment Setup (Kaggle System Python)
# We install dependencies into the system environment to avoid ABI conflicts.

# %% [code]
print("--- Setting up environment ---")
# Downgrade NumPy to 1.26.4 (NumPy 2.0.x is incompatible with spconv/cumm JIT)
run_cmd("pip install --quiet 'numpy<2.0.0'")

DEPENDENCIES = [
    "torch==2.3.1+cu121 torchvision==0.18.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121",
    "ninja",
    "pccm",
    "ccimport",
    "cumm-cu121",
    "spconv-cu121",
    "torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html",
    "pytorch-lightning==2.2.1",
    "open3d",
    "mmyolo==0.6.0",
    "mmdet==3.3.0",
    "mmcv==2.1.0",
    "loguru",
    "fire",
    "wandb",
    "imageio",
    "scipy",
    "albumentations",
    "hydra-core",
    "omegaconf",
    "supervision"
]

for dep in DEPENDENCIES:
    run_cmd(f"pip install --quiet {dep}")

# %% [markdown]
# ## 2. Local Package Installation

# %% [code]
print("--- Installing local packages ---")
# Now that we are inside the repo, this will find pyproject.toml
run_cmd("pip install -e .")

# %% [markdown]
# ## 3. Dataset Configuration

# %% [code]
print("--- Preparing Dataset and Evaluation ---")
DATASET_PATH = "/kaggle/input/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset/"
SPLITS_PATH = "/kaggle/input/strawpick-sint-pointnetseg-test/splits.json"

# %% [markdown]
# ## 4. Running the Training

# %% [code]
TRAIN_CMD = (
    f"python train.py "
    f"--data_path {DATASET_PATH} "
    f"--splits_path {SPLITS_PATH} "
    f"--epochs 50 "
    f"--batch_size 1 "
    f"--lr 1e-4 "
    f"--name openyolo3d_strawberry_finetune"
)

if __name__ == "__main__":
    print("--- Starting Open-YOLO-3D Training ---")
    run_cmd(TRAIN_CMD)
