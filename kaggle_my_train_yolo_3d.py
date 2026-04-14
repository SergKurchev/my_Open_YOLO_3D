# %% [markdown]
# # Open-YOLO-3D Fine-Tuning on Kaggle
# This notebook clones the Open-YOLO-3D repository, creates a virtual environment, 
# installs dependencies, and runs the `my_train_yolo_3d.py` training script on the Multiview Strawberry Dataset.

# %% [code]
# 1. Сlode the repository
!git clone https://github.com/SergKurchev/my_Open_YOLO_3D.git /kaggle/working/my_Open_YOLO_3D

# %% [code]

# %% [code]
# 3. Upgrade pip and install requirements (assuming conda / environment.yml equivalent for pip)
# In Kaggle it is faster to just install torch and other dependencies via pip inside the venv
!pip install --upgrade pip
!pip install torch torchvision numpy pandas scipy tqdm pyyaml open3d imageio scikit-learn

# %% [code]
# OpenYOLO3D also relies on mmengine and mmyolo
!pip install openmim
!mim install mmengine
!mim install "mmcv>=2.0.0"
!mim install "mmdet>=3.0.0"
!pip install mmyolo supervision

# %% [code]
# 4. Run the training script matching the Evaluation Protocol
import os

# Create splits.json if it doesn't exist out of the box in the dataset
dataset_path = "/kaggle/input/strawpick-sint-pointnetseg-test/multiview_dataset"
assert os.path.exists(dataset_path), "Dataset path not found, please add the dataset to the kernel!"

run_cmd = f"cd /kaggle/working/my_Open_YOLO_3D/OpenYOLO3D && python my_train_yolo_3d.py --data_path {dataset_path} --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4"
print(f"Executing: {run_cmd}")

# %% [code]
!cd /kaggle/working/my_Open_YOLO_3D/OpenYOLO3D && python my_train_yolo_3d.py --data_path /kaggle/input/strawpick-sint-pointnetseg-test/multiview_dataset --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4
