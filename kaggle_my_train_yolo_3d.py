# %% [markdown]
# # Open-YOLO-3D Fine-Tuning on Kaggle
# This notebook clones the Open-YOLO-3D repository, creates a virtual environment, 
# installs dependencies, and runs the `my_train_yolo_3d.py` training script on the Multiview Strawberry Dataset.

# %% [code]
# 1. Сlode the repository
!git clone https://github.com/SergKurchev/my_Open_YOLO_3D.git /kaggle/working/my_Open_YOLO_3D

# %% [code]
# 2. Robust Virtual Environment Creation
import os
import subprocess
import sys
import urllib.request

if not os.path.exists("/kaggle/working/openyolo_env"):
    print("Создаю виртуальное окружение...")
    # Флаг --without-pip предотвращает попытки использовать отсутствующий ensurepip
    subprocess.run([sys.executable, "-m", "venv", "/kaggle/working/openyolo_env", "--without-pip"], check=True)
    
    VENV_PYTHON_TMP = "/kaggle/working/openyolo_env/bin/python"
    
    # Прямая установка pip через официальный скрипт
    print("Загружаю get-pip.py для ручной установки...")
    urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", "get-pip.py")
    
    print("Устанавливаю pip в виртуальное окружение...")
    subprocess.run([VENV_PYTHON_TMP, "get-pip.py"], check=True)
    
    # Чистим за собой
    if os.path.exists("get-pip.py"):
        os.remove("get-pip.py")
    print("Pip успешно установлен!")

# Теперь пути будут корректными
VENV_PYTHON = "/kaggle/working/openyolo_env/bin/python"
VENV_PIP = "/kaggle/working/openyolo_env/bin/pip"
VENV_MIM = "/kaggle/working/openyolo_env/bin/mim"

# %% [code]
# 3. Install Dependencies (Following Installation.md strictly)
print("Installing Torch and specialized dependencies...")
!{VENV_PIP} install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
!{VENV_PIP} install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
!{VENV_PIP} install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

# MinkowskiEngine
print("Installing MinkowskiEngine (compiling from source)...")
!cd /kaggle/working && git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
!cd /kaggle/working/MinkowskiEngine && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
!cd /kaggle/working/MinkowskiEngine && export TCNN_CUDA_ARCH_LIST="60;61;70;75;80;86" && {VENV_PYTHON} setup.py install --force_cuda --blas=openblas

# ScanNet Segmentator
print("Compiling ScanNet Segmentator...")
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party && git clone https://github.com/ScanNet/ScanNet.git
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/ScanNet/Segmentator && git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2 && make

# pointnet2
print("Installing pointnet2...")
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D/third_party/pointnet2 && {VENV_PYTHON} setup.py install

# Finishing Mask3D and YOLO-World
print("Finalizing Mask3D and YOLO-World local installs...")
!{VENV_PIP} install pytorch-lightning==1.7.2
!{VENV_PIP} install black==21.4b2 cloudpickle==3.0.0 future hydra-core==1.0.5 pycocotools>=2.0.2 pydot iopath==0.1.7 loguru albumentations volumentations omegaconf==2.0.6 antlr4-python3-runtime==4.8 wrapt
!cd /kaggle/working/my_Open_YOLO_3D/models/Mask3D && {VENV_PIP} install .
!cd /kaggle/working/my_Open_YOLO_3D/models/YOLO-World && {VENV_PIP} install -e .

# MM-ecosystem
print("Installing MM-ecosystem...")
!{VENV_PIP} install openmim
!{VENV_MIM} install mmcv==2.0.0
!{VENV_PIP} install mmyolo==0.6.0 mmdet==3.0.0 plyfile supervision open3d pillow==9.1.0 pandas scipy tqdm pyyaml imageio scikit-learn

# %% [code]
# 4. Run the training script matching the Evaluation Protocol
import os

# Updated paths provided by user
# Dataset: /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset
# Splits: /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json

dataset_path = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset"
splits_path = "/kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json"

assert os.path.exists(dataset_path), "Dataset path not found!"
assert os.path.exists(splits_path), "Splits path not found!"

run_cmd = f"cd /kaggle/working/my_Open_YOLO_3D && {VENV_PYTHON} my_train_yolo_3d.py --data_path {dataset_path} --splits_path {splits_path} --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4"
print(f"Executing: {run_cmd}")

# %% [code]
!cd /kaggle/working/my_Open_YOLO_3D && {VENV_PYTHON} my_train_yolo_3d.py --data_path /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/multiview_dataset/multiview_dataset --splits_path /kaggle/input/datasets/sergeykurchev/strawpick-sint-pointnetseg-test/splits.json --epochs 100 --batch_size 4 --lr 0.0001 --weight_decay 1e-4
