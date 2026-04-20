import os
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path
from loguru import logger
from openyolo3d.utils.paths import resolve_path, get_project_root

def check_weights(config_path: str = "pretrained/config_scannet200.yaml"):
    """
    Diagnostic script to verify checkpoint existence and version compatibility.
    """
    logger.info(f"🔍 Starting weight verification using config: {config_path}")
    
    # 1. Load Config
    full_config_path = resolve_path(config_path)
    if not os.path.exists(full_config_path):
        logger.error(f"❌ Config file not found at {full_config_path}")
        return
    
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Check 2D weights
    yolo_cfg = resolve_path(config['network2d']['config_path'])
    yolo_weights = resolve_path(config['network2d']['pretrained_path'])
    
    logger.info("Checking 2D Model (YOLO-World)...")
    if os.path.exists(yolo_cfg):
        logger.success(f"✅ 2D Config found: {yolo_cfg}")
    else:
        logger.warning(f"⚠️ 2D Config MISSING: {yolo_cfg}")
        
    if os.path.exists(yolo_weights):
        logger.success(f"✅ 2D Weights found: {yolo_weights}")
        try:
            # Try to load header
            checkpoint = torch.load(yolo_weights, map_location='cpu')
            logger.success("✅ 2D Weights are loadable by PyTorch")
            del checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load 2D Weights: {e}")
    else:
        logger.warning(f"⚠️ 2D Weights MISSING: {yolo_weights}")

    # 3. Check 3D weights
    mask3d_weights = resolve_path(config['network3d']['pretrained_path'])
    logger.info("Checking 3D Model (Mask3D)...")
    if os.path.exists(mask3d_weights):
        logger.success(f"✅ 3D Weights found: {mask3d_weights}")
        try:
            # Try to load header
            checkpoint = torch.load(mask3d_weights, map_location='cpu')
            logger.success("✅ 3D Weights are loadable by PyTorch")
            if 'pytorch-lightning_version' in checkpoint:
                ver = checkpoint['pytorch-lightning_version']
                logger.info(f"ℹ️ Weights were saved with Lightning version: {ver}")
                if pl.__version__.split('.')[0] != ver.split('.')[0]:
                    logger.warning(f"⚠️ Lightning version mismatch! Local: {pl.__version__}, Weights: {ver}")
            del checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load 3D Weights: {e}")
    else:
        logger.warning(f"⚠️ 3D Weights MISSING: {mask3d_weights}")

    # 4. Version Check
    logger.info(f"System Versions:")
    import sys
    logger.info(f"  Python: {sys.version.split(' ')[0]} (Target: 3.12.x)")
    logger.info(f"  PyTorch: {torch.__version__} (Target: 2.3.x+)")
    logger.info(f"  Lightning: {pl.__version__} (Target: 2.3.x+)")
    
    if sys.version_info[:2] != (3, 12):
        logger.warning(f"⚠️ Python version mismatch! Expected 3.12, got {sys.version.split(' ')[0]}")
    
    if torch.__version__.startswith('2.'):
        logger.success("✅ Using PyTorch 2.x. (Target reached)")
    else:
        logger.warning(f"⚠️ PyTorch version is {torch.__version__}, expected 2.3.x+")
    
    logger.info("Verification complete.")

if __name__ == "__main__":
    import fire
    fire.Fire(check_weights)
