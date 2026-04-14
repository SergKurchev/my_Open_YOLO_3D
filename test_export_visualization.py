import os
import sys
import json
import numpy as np
from pathlib import Path

# Add current dir to path to import generate_sample_viewer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generate_sample_viewer import build_pointcloud, build_html

def test_visualization():
    print("Testing visualization export with mock predictions...")
    
    # Path to the dataset
    data_root = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\multiview_dataset_5frames"
    sample_name = "sample_00000"
    sample_path = Path(os.path.join(data_root, sample_name))
    
    if not sample_path.exists():
        print(f"Error: Sample path {sample_path} does not exist.")
        return
        
    cameras = json.loads((sample_path / "cameras.json").read_text())
    color_map = json.loads((sample_path / "color_map.json").read_text())
    
    print(f"Loading point cloud for {sample_name}...")
    # Load point cloud (GT data: 8 columns: x, y, z, r, g, b, inst, cat)
    pts = build_pointcloud(sample_path, cameras, color_map, stride=2, max_points=800000, mode="plant")
    
    # Simulate model prediction (MOCK)
    print("Simulating model predictions...")
    N = len(pts)
    inst_gt = pts[:, 6]
    cat_gt = pts[:, 7]
    
    # We create slightly imperfect predictions
    inst_pred = inst_gt.copy()
    cat_pred = cat_gt.copy()
    
    # Add some errors to demonstrate it's a prediction
    # Randomly change the category of 2% of the points
    error_mask = np.random.rand(N) < 0.02
    cat_pred[error_mask] = np.random.choice([0, 1, 2], size=np.sum(error_mask))
    
    # Stack the predictions to create a 10-column array: 
    # [x, y, z, r, g, b, inst, cat, inst_pred, cat_pred]
    pts_with_pred = np.column_stack([pts, inst_pred.astype(np.float32), cat_pred.astype(np.float32)])
    
    print("Building HTML visualization...")
    html = build_html(pts_with_pred, cameras, color_map, sample_name + "_EPOCH_1")
    
    # Export as required by models_evaluation_protocol.md
    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"epoch_1_{sample_name}_pred.html"
    output_file.write_text(html, encoding="utf-8")
    
    size_mb = output_file.stat().st_size / 1e6
    print(f"\nSuccess! Mock prediction visualization saved to: {output_file} ({size_mb:.1f} MB)")
    print("This file contains the interactive comparison of GT and Predictions.")

if __name__ == "__main__":
    test_visualization()
