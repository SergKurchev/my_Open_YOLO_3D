import sys
import torch.nn as nn
import spconv.pytorch as spconv
from enum import Enum

if sys.version_info[:2] >= (3, 8):
    from collections.abc import Sequence
else:
    from collections import Sequence

class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2

def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    # In spconv, BatchNorm/InstanceNorm is usually applied to the features (N, C)
    # We use standard 1D versions which are compatible with SparseConvTensor.features
    if norm_type == NormType.BATCH_NORM:
        return nn.BatchNorm1d(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return nn.InstanceNorm1d(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            nn.InstanceNorm1d(n_channels),
            nn.BatchNorm1d(n_channels, momentum=bn_momentum),
        )
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")

class ConvType(Enum):
    HYPERCUBE = 0
    SPATIAL_HYPERCUBE = 1

def conv(
    in_planes,
    out_planes,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=3,
):
    if stride == 1:
        return spconv.SubMConv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias,
            indice_key="subm" # optional, helps with speed in some cases
        )
    else:
        return spconv.SparseConv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

def conv_tr(
    in_planes,
    out_planes,
    kernel_size,
    upsample_stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=3,
):
    return spconv.SparseInverseConv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        dilation=dilation,
        bias=bias,
    )

def sum_pool(kernel_size, stride=1, dilation=1, D=3):
    return spconv.SparseMaxPool3d(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation
    )

def avg_pool(kernel_size, stride=1, dilation=1, D=3):
    # spconv doesn't have a built-in SparseAvgPool3d in all versions
    # But usually UNets use MaxPool for downsampling or stride-2 conv
    return spconv.SparseMaxPool3d(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation
    )
