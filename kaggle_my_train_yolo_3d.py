import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Environment Setup (Kaggle System Python)
print("--- Setting up environment ---")
# On Kaggle, we can directly use pip. We'll target CUDA 12.1 compatible wheels.
DEPENDENCIES = [
    "torch==2.3.1+cu121 torchvision==0.18.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121",
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

# 2. Local Package Installation
print("--- Installing local packages ---")
# Install the current directory as a package
run_cmd("pip install -e .")

# 3. Preparation for Evaluation Protocol
print("--- Preparing Dataset and Evaluation ---")
# The dataset path is provided by the user in Kaggle
# /kaggle/input/strawpick-sint-pointnetseg-test/multiview_dataset/
DATASET_PATH = "/kaggle/input/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset/"
SPLITS_PATH = "/kaggle/input/strawpick-sint-pointnetseg-test/splits.json"

# 4. Running the Training
# We use the modernized train.py
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
