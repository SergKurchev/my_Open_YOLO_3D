import sys
import os
import torch
from .utils.paths import resolve_path

# Add Mask3D to sys.path to handle its internal absolute imports
# This allows 'import mask3d' to work within the Mask3D sub-package
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mask3d_path = os.path.join(root_dir, "models", "Mask3D")
if mask3d_path not in sys.path:
    sys.path.insert(0, mask3d_path)

from models.Mask3D.mask3d import get_model, prepare_data, map_output_to_pointcloud

class Network_3D():
    def __init__(self, config):
        pretrained_path = resolve_path(config["network3d"]["pretrained_path"])
        self.model = get_model(pretrained_path)
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.model.to(self.device)
    
    def get_class_agnostic_masks(self, pointcloud_file, datatype="point cloud", point2segment=None):
        data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full = prepare_data(pointcloud_file, datatype, self.device)
        with torch.no_grad():
            outputs = self.model(data, raw_coordinates=features, point2segment=[point2segment] if point2segment is not None else None)
        return map_output_to_pointcloud(outputs, inverse_map, point2segment, point2segment_full)
        