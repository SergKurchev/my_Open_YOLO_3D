import torch
import numpy as np
import os
import json
import imageio
from scipy.spatial.transform import Rotation as R
from loguru import logger
from typing import Dict, List, Tuple

def quaternion_to_matrix(pos: List[float], quat: List[float]) -> np.ndarray:
    """
    Converts position and quaternion [qx, qy, qz, qw] to 4x4 transformation matrix.
    """
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat(quat).as_matrix()
    mat[:3, 3] = pos
    return mat

def get_3d_gt_consensus(
    points: np.ndarray,
    mask_paths: List[str],
    poses: List[np.ndarray],
    intrinsics: np.ndarray,
    image_res: Tuple[int, int],
    depth_res: Tuple[int, int],
    instance_to_category: Dict[int, int],
    depth_scale: float = 1000.0,
    vis_threshold: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to all views, sample labels from 2D masks, and perform consensus voting.
    Returns:
        instance_ids: [N]
        category_ids: [N]
    """
    num_points = points.shape[0]
    num_views = len(mask_paths)
    
    # [num_points, num_views] to store sampled Instance IDs
    # -1 means not projected/not visible/background
    votes = torch.full((num_points, num_views), -1, dtype=torch.int32)
    
    points_h = np.concatenate([points, np.ones((num_points, 1))], axis=-1)
    
    for i in range(num_views):
        mask = imageio.imread(mask_paths[i])
        if len(mask.shape) == 3:
            mask = mask[:, :, 0] # Assume instance ID in first channel
            
        world2cam = np.linalg.inv(poses[i])
        # Project points to camera space
        pts_cam = (world2cam @ points_h.T).T
        
        # Project to screen space
        depths = pts_cam[:, 2]
        valid_depth = depths > 0
        
        pts_2d_h = (intrinsics @ pts_cam.T).T
        pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]
        
        # Check bounds
        x, y = pts_2d[:, 0], pts_2d[:, 1]
        valid_bounds = (x >= 0) & (x < image_res[1]-1) & (y >= 0) & (y < image_res[0]-1)
        
        # Combine masks
        valid_mask = valid_depth & valid_bounds
        
        # Sampling (Nearest neighbor)
        sampled_indices = np.where(valid_mask)[0]
        xi = x[sampled_indices].astype(int)
        yi = y[sampled_indices].astype(int)
        
        instance_ids = mask[yi, xi]
        votes[sampled_indices, i] = torch.from_numpy(instance_ids.astype(np.int32))
        
    # Consensus voting
    final_instance_ids = torch.full((num_points,), -1, dtype=torch.int32)
    
    # For each point, find most frequent ID (excluding -1 and 0 which is usually background)
    for p in range(num_points):
        p_votes = votes[p]
        p_votes = p_votes[p_votes > 0] # Exclude background/missing
        if len(p_votes) > 0:
            final_instance_ids[p] = torch.mode(p_votes).values
            
    # Map to category IDs
    final_category_ids = torch.full((num_points,), 0, dtype=torch.int32)
    for inst_id, cat_id in instance_to_category.items():
        final_category_ids[final_instance_ids == inst_id] = cat_id
        
    return final_instance_ids, final_category_ids

def parse_metadata(metadata_path: str):
    """
    Parses camera parameters and class mappings from the strawberry dataset metadata.
    """
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    # Standard intrinsic for the dataset if not provided per view
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return data
