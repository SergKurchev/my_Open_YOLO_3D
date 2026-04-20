import torch
import hydra
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
from torch.nn import functional as F
from .modules.common import conv
from .position_embedding import PositionEmbeddingCoordsSine
from .modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max
from torch.cuda.amp import autocast

# Minimal emulation of MinkowskiOps for spconv
class MinkowskiOpsEmulation:
    @staticmethod
    def SparseTensor(features, coordinate_manager=None, coordinate_map_key=None, spatial_shape=None, batch_size=None, indices=None, device=None):
        return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

def furthest_point_sample(xyz, npoint):
    # This is usually imported from pointnet2_utils
    # Placeholder or import if available
    from pointnet2.pointnet2_utils import furthest_point_sample as fps
    return fps(xyz, npoint)

class Mask3D(nn.Module):
    def __init__(
        self,
        config,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal,
    ):
        super().__init__()
        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type

        self.backbone = hydra.utils.instantiate(config.backbone)
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.PLANES[-5:]

        self.mask_features_head = conv(
            self.backbone.PLANES[7],
            self.mask_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            D=3,
        )

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(
                mask, p2s, dim=dim
            )[0]
        else:
            assert False, "Scatter function not known"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True,
            )
        else:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)

        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc,
            )
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc,
            )

        self.pooling = spconv.SparseMaxPool3d(
            kernel_size=2, stride=2
        )

        # Decoder initialization (simplified for brevety, assuming similar to original)
        # ... (same as original, but using relu instead of MinkowskiReLU)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def decompose_spconv(self, x):
        """Helper to emulate ME's decomposed_features for spconv"""
        features = x.features
        indices = x.indices
        decomposed = []
        for i in range(x.batch_size):
            mask = indices[:, 0] == i
            decomposed.append(features[mask])
        return decomposed

    def decompose_indices(self, x):
        """Helper to emulate ME's decomposed_coordinates for spconv"""
        indices = x.indices
        decomposed = []
        for i in range(x.batch_size):
            mask = indices[:, 0] == i
            # Remove batch index column
            decomposed.append(indices[mask, 1:])
        return decomposed

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []
        for i in range(len(coords)):
            pos_encodings_pcd.append([[]])
            decomposed = self.decompose_spconv(coords[i])
            for coords_batch in decomposed:
                scene_min = coords_batch.min(dim=0)[0][None, ...]
                scene_max = coords_batch.max(dim=0)[0][None, ...]
                with torch.no_grad():
                    tmp = self.pos_enc(
                        coords_batch[None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )
                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))
        return pos_encodings_pcd

    def forward(
        self, x, point2segment=None, raw_coordinates=None, is_eval=False
    ):
        pcd_features, aux = self.backbone(x)
        batch_size = x.batch_size

        with torch.no_grad():
            # spconv uses indices [B, Z, Y, X] or similar
            # In our case raw_coordinates is features
            coordinates = spconv.SparseConvTensor(
                features=raw_coordinates,
                indices=aux[-1].indices,
                spatial_shape=aux[-1].spatial_shape,
                batch_size=aux[-1].batch_size,
            )

            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))
            coords.reverse()

        pos_encodings_pcd = self.get_pos_encs(coords)
        mask_features = self.mask_features_head(pcd_features)
        
        # Decompose for attention sampling
        decomposed_mask_features = self.decompose_spconv(mask_features)
        
        # ... (rest of forward logic adapted for decomposed spconv)
        # Placeholder for brevity, I will implement full forward in next step if needed
        return {"predictions": "placeholder"} 

class PositionalEncoding3D(nn.Module):
    # Same as original but ensuring numpy/torch compatibility
    def __init__(self, channels):
        super(PositionalEncoding3D, self).__init__()
        self.orig_ch = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))

# Helper classes from original Mask3D (SelfAttentionLayer, CrossAttentionLayer, FFNLayer)
# need to be checked for relu/norm but standard PyTorch is fine.
