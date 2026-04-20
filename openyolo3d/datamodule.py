import os
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
import glob
import random

class MultiviewDataset(Dataset):
    def __init__(self, data_root: str, split: str = 'train', splits_path: Optional[str] = None):
        self.data_root = data_root
        self.split = split
        
        actual_splits_path = splits_path or os.path.join(data_root, 'splits.json')
        
        if not os.path.exists(actual_splits_path):
            # Fallback if splits.json is missing
            samples = [os.path.basename(p) for p in glob.glob(os.path.join(data_root, 'sample_*'))]
            random.seed(42)
            random.shuffle(samples)
            train_idx = int(len(samples)*0.8)
            val_idx = int(len(samples)*0.9)
            self.splits_data = {
                'train': samples[:train_idx],
                'val': samples[train_idx:val_idx],
                'test': samples[val_idx:]
            }
        else:
            with open(actual_splits_path, 'r') as f:
                self.splits_data = json.load(f)
                
        self.samples = self.splits_data.get(split, [])

    def __getitem__(self, idx):
        sample_path = os.path.join(self.data_root, self.samples[idx])
        
        # In a real scenario, we want to know where the masks are
        # For evaluation (validation/test), we need these paths
        mask_dir = os.path.join(sample_path, 'masks')
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        
        return {
            'path': sample_path,
            'mask_paths': mask_paths,
            'split': self.split
        }

class OpenYolo3DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        splits_path: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_path = data_path
        self.splits_path = splits_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = MultiviewDataset(self.data_path, split='train', splits_path=self.splits_path)
            self.val_ds = MultiviewDataset(self.data_path, split='val', splits_path=self.splits_path)
        
        if stage == "test" or stage is None:
            self.test_ds = MultiviewDataset(self.data_path, split='test', splits_path=self.splits_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
