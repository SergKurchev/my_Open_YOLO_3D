import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def link_or_copy(src, dst):
    """Try to hardlink to save massive disk space, fallback to copy if unsupported."""
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Convert 20-frame dataset to 5-frame dataset")
    parser.add_argument('--src', type=str, required=True, help="Path to original multiview_dataset")
    parser.add_argument('--dst', type=str, required=True, help="Path to new multiview_dataset_5frames")
    parser.add_argument('--frames', type=int, default=5, help="Number of frames per new sample")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    frames_per_chunk = args.frames

    if not src.exists():
        print(f"Error: Source directory {src} does not exist.")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    splits_path = src / "splits.json"
    if splits_path.exists():
        with open(splits_path, 'r') as f:
            src_splits = json.load(f)
    else:
        # Auto-detect all samples instead if splits doesn't exist
        print("splits.json not found, processing all valid samples...")
        sample_folders = [d.name for d in src.iterdir() if d.is_dir() and d.name.startswith("sample_")]
        src_splits = {"train": sample_folders} # Mock everything to train

    dst_splits = {"train": [], "val": [], "test": []}
    new_sample_idx = 0

    print(f"Converting dataset {src} into {frames_per_chunk}-frame chunks -> {dst}")
    
    for split_name in ["train", "val", "test"]:
        if split_name not in src_splits:
            continue
            
        samples = src_splits[split_name]
        for sample in tqdm(samples, desc=f"Processing {split_name} splits"):
            sample_path = src / sample
            if not sample_path.exists():
                continue

            cam_path = sample_path / "cameras.json"
            if not cam_path.exists():
                continue
                
            with open(cam_path, 'r') as f:
                cameras = json.load(f)

            color_map_path = sample_path / "color_map.json"

            # Sort cameras by frame index
            cameras.sort(key=lambda x: x["frame_index"])

            for i in range(0, len(cameras), frames_per_chunk):
                chunk_cams = cameras[i : i + frames_per_chunk]
                if not chunk_cams: continue

                new_sample_name = f"sample_{new_sample_idx:05d}"
                new_sample_path = dst / new_sample_name
                new_sample_path.mkdir(exist_ok=True)

                dst_splits[split_name].append(new_sample_name)

                # Copy color_map
                if color_map_path.exists():
                    shutil.copy2(color_map_path, new_sample_path / "color_map.json")

                # Subdirs
                for subdir in ["rgb", "depth", "masks", "labels"]:
                    (new_sample_path / subdir).mkdir(exist_ok=True)

                new_cams = []
                for j, cam in enumerate(chunk_cams):
                    orig_f = cam["frame_index"]
                    orig_name = f"{orig_f:05d}"
                    new_name  = f"{j:05d}"

                    # Copy files using hardlink or copy
                    for subdir, ext in [("rgb", ".png"), ("depth", ".npy"), ("masks", ".png"), ("labels", ".txt")]:
                        s_file = sample_path / subdir / (orig_name + ext)
                        if s_file.exists():
                            d_file = new_sample_path / subdir / (new_name + ext)
                            link_or_copy(s_file, d_file)

                    # Update camera JSON for new index
                    new_cam = cam.copy()
                    new_cam["frame_index"] = j
                    new_cams.append(new_cam)

                # Write chunked cameras.json
                with open(new_sample_path / "cameras.json", 'w') as f:
                    json.dump(new_cams, f, indent=4)

                new_sample_idx += 1

    # Write new splits.json
    with open(dst / "splits.json", 'w') as f:
        json.dump(dst_splits, f, indent=4)

    total_old = sum(len(v) for v in src_splits.values())
    total_new = sum(len(v) for v in dst_splits.values())
    print("\n[SUCCESS] Conversion completed!")
    print(f"Old samples (20 frames): {total_old}")
    print(f"New samples ( {frames_per_chunk} frames): {total_new}")
    print(f"Now you can pass --data_path {dst} to your train script.")

if __name__ == "__main__":
    main()
