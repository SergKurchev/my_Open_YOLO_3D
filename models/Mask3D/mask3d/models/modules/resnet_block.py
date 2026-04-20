import torch.nn as nn
from mask3d.models.modules.common import ConvType, NormType, conv, get_norm

class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super().__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D,
        )
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum
        )
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D,
        )
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # spconv layers return SparseConvTensor
        # We apply norm and relu to the features
        out = out.replace_feature(self.norm1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        # Skip connection: add features
        # Note: In SubMConv environments, coordinates match exactly
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out

class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM

class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_NORM

class BasicBlockINBN(BasicBlockBase):
    NORM_TYPE = NormType.INSTANCE_BATCH_NORM

class BottleneckBase(nn.Module):
    expansion = 4
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        conv_type=ConvType.HYPERCUBE,
        bn_momentum=0.1,
        D=3,
    ):
        super().__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum
        )

        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D,
        )
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum
        )

        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
        self.norm3 = get_norm(
            self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.norm1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.norm2(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv3(out)
        out = out.replace_feature(self.norm3(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out

class Bottleneck(BottleneckBase):
    NORM_TYPE = NormType.BATCH_NORM
