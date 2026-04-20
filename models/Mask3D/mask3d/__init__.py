import hydra
import torch
from torch_scatter import scatter_mean

import spconv.pytorch as spconv
from .models.mask3d import Mask3D
from .utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(self, x, raw_coordinates=None, point2segment=None):
        return self.model(x, raw_coordinates=raw_coordinates, point2segment=point2segment)
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import numpy as np
import open3d as o3d

# imports for output
from .datasets.scannet200.scannet200_constants import (VALID_CLASS_IDS_20, VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_20, SCANNET_COLOR_MAP_200)

def get_model(checkpoint_path=None, dataset_name = "scannet200"):


    # Initialize the directory with config files
    with initialize(config_path="conf", version_base=None):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    # dataset_name = checkpoint_path.split('/')[-1].split('_')[0]
    if dataset_name == 'scannet200':
        cfg.general.num_targets = 201
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 200
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
    if dataset_name == 'scannet':
        cfg.general.num_targets = 19
        cfg.general.train_mode = False
        cfg.general.eval_on_segments = True
        cfg.general.topk_per_image = 300
        cfg.general.use_dbscan = True
        cfg.general.dbscan_eps = 0.95
        cfg.general.export_threshold = 0.001

        # # data
        cfg.data.num_labels = 20
        cfg.data.test_mode = "test"

        # # model
        cfg.model.num_queries = 150
        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    return mesh

def load_ply(path_2_mesh):
    pcd = o3d.io.read_point_cloud(path_2_mesh)
    return pcd

def load_mesh_or_pc(pointcloud_file, datatype):
    
    if pointcloud_file.split('.')[-1] == 'ply':
        if datatype == "mesh":
            data = load_mesh(pointcloud_file)
        elif datatype == "point cloud":
            data = load_ply(pointcloud_file)
            
        if datatype is None:
            print("DATA TYPE IS NOT SUPPORTED!")
            exit()
    return data

def prepare_data(pointcloud_file, datatype, device, voxel_size=0.02):
    # normalization for point cloud features
    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)
    
    if pointcloud_file.split('.')[-1] == 'ply':
        if datatype == "mesh":
            mesh = load_mesh(pointcloud_file)
            points = np.asarray(mesh.vertices)
            colors = np.asarray(mesh.vertex_colors)
            colors = colors * 255.
        elif datatype == "point cloud":
            pcd = load_ply(pointcloud_file)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
        if datatype is None:
            print("DATA TYPE IS NOT SUPPORTED!")
            exit()
        segments = None
    elif pointcloud_file.split('.')[-1] == 'npy':
        points_data = np.load(pointcloud_file)
        points, colors, normals, segments, labels = (
            points_data[:, :3],
            points_data[:, 3:6],
            points_data[:, 6:9],
            points_data[:, 9],
            points_data[:, 10:12],
        )
        datatype = "mesh"
    else:
        print("FORMAT NOT SUPPORTED")
        exit()

    if datatype == "mesh":
        pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
        colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    # Manual Voxelization (Equivalent to ME.utils.sparse_quantize)
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_map, inverse_map = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    
    sample_coordinates = coords[unique_map]
    sample_features = colors[unique_map]
    
    # spconv needs [N, 4] with batch index in column 0
    batch_idx = np.zeros((len(sample_coordinates), 1), dtype=np.int32)
    coordinates = np.concatenate([batch_idx, sample_coordinates], axis=1)
    
    coordinates = torch.from_numpy(coordinates).int().to(device)
    features = torch.from_numpy(sample_features).float().to(device)
    
    # Calculate spatial shape for spconv
    all_coords = sample_coordinates
    spatial_shape = (all_coords.max(axis=0) - all_coords.min(axis=0) + 1).astype(int).tolist()
    
    data = spconv.SparseConvTensor(
        features=features,
        indices=coordinates,
        spatial_shape=spatial_shape,
        batch_size=1,
    )
    
    point2segment = None
    point2segment_full = None
    if segments is not None:
        point2segment_full = torch.from_numpy(segments).long().to(device)
        point2segment = torch.from_numpy(segments[unique_map]).long().to(device)
        
    return data, points, colors, features, unique_map, inverse_map, point2segment, point2segment_full


def map_output_to_pointcloud(outputs, 
                             inverse_map,
                             point2segment, 
                             point2segment_full):
    
    # parse predictions
    logits = outputs["pred_logits"]
    logits = torch.functional.F.softmax(logits, dim=-1)[..., :-1]
    masks = outputs["pred_masks"]
    # reformat predictions
    logits = logits[0]
    masks = masks[0] if point2segment is None else masks[0][point2segment]

    num_queries = len(logits)
    scores_per_query, topk_indices = logits.flatten(0, 1).topk(
        num_queries, sorted=True
    )

    topk_indices = topk_indices // 200
    masks = masks[:, topk_indices]

    result_pred_mask = (masks > 0).float()
    heatmap = masks.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    score = scores_per_query * mask_scores_per_image
    result_pred_mask = get_full_res_mask(result_pred_mask, inverse_map, point2segment_full[0]) if point2segment_full is not None else result_pred_mask[inverse_map]
    return (result_pred_mask, score)

def get_full_res_mask(mask, inverse_map, point2segment_full):
    mask = mask.detach().cpu()[inverse_map]  # full res
    mask = scatter_mean(mask, point2segment_full, dim=0)  # full res segments
    mask = (mask > 0.5).float()
    mask = mask.detach().cpu()[point2segment_full.cpu()]  # full res points
    return mask

def save_colorized_mesh(mesh, labels_mapped, output_file, colormap='scannet'):
    
    # colorize mesh
    colors = np.zeros((len(mesh.vertices), 3))
    for li in np.unique(labels_mapped):
        if colormap == 'scannet':
            raise ValueError('Not implemented yet')
        elif colormap == 'scannet200':
            v_li = VALID_CLASS_IDS_200[int(li)]
            colors[(labels_mapped == li)[:, 0], :] = SCANNET_COLOR_MAP_200[v_li]
        else:
            raise ValueError('Unknown colormap - not supported')
    
    colors = colors / 255.
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(output_file, mesh)

if __name__ == '__main__':
    
    model = get_model('checkpoints/scannet200/scannet200_benchmark.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load input data
    pointcloud_file = 'data/pcl.ply'
    mesh = load_mesh(pointcloud_file)
    
    # prepare data
    data, points, colors, features, unique_map, inverse_map = prepare_data(mesh, device)
    
    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)
        
    # map output to point cloud
    labels = map_output_to_pointcloud(mesh, outputs, inverse_map)
    
    # save colorized mesh
    save_colorized_mesh(mesh, labels, 'data/pcl_labelled.ply', colormap='scannet200')
    