# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors.float())
        print(type(xs), type(tensor_list.tensors.float()))
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class CustomBackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        #print(backbone)
        self.body = IntermediateLayerGetter(backbone, return_layers={"conv1": 0, "relu1": 1, "bn16": 2, "mx1": 3, "conv2": 4, "relu2": 5, "bn32": 6, "mx2": 7, "conv3": 8, "relu3": 9, "bn64": 10, "mx3": 11, "conv4": 12, "relu4": 13, "bn64_2": 14})
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors.float())
        #print(type(xs), type(tensor_list.tensors.float()))
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class CustomBackboneBaseWithout1D(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        #print(backbone)
        self.body = IntermediateLayerGetter(backbone, return_layers={"conv1": 0, "relu1": 1, "bn16": 2, "mx1": 3, "conv2": 4, "relu2": 5, "bn32": 6, "mx2": 7, "conv3": 8, "relu3": 9, "bn64": 10, "mx3": 11, "conv4": 12, "relu4": 13, "bn64_2": 14, "mx4": 15, "conv5": 16, "relu5": 17, "bn64_3": 18, "mx5": 19})
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors.float())
        #print(type(xs), type(tensor_list.tensors.float()))
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class CustomBackbone(nn.Module):
    def __init__(self, hidden_dim = 256):

        super(CustomBackbone, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (3,3), stride = 1)
        self.relu1 = nn.ReLU()
        self.bn16  = nn.BatchNorm2d(16)

        self.mx1   = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3,3), stride = 1)
        self.relu2 = nn.ReLU()
        self.bn32  = nn.BatchNorm2d(32)

        self.mx2   = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, (3,3), stride = 1)
        self.relu3 = nn.ReLU()
        self.bn64  = nn.BatchNorm2d(64)

        self.mx3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, (30,1), stride = 1)
        self.relu4 = nn.ReLU()
        self.bn64_2  = nn.BatchNorm2d(64)

        #self.conv4 = nn.Conv2d(64, hidden_dim, 1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn16(out)

        out = self.mx1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn32(out)

        out = self.mx2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn64(out)

        out = self.mx3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.bn64_2(out)

        return out

class CustomBackboneWithout1D(nn.Module):
    def __init__(self, hidden_dim = 256):

        super(CustomBackboneWithout1D, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (3,3), stride = 1)
        self.relu1 = nn.ReLU()
        self.bn16  = nn.BatchNorm2d(16)

        self.mx1   = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, (3,3), stride = 1)
        self.relu2 = nn.ReLU()
        self.bn32  = nn.BatchNorm2d(32)

        self.mx2   = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, (3,3), stride = 1)
        self.relu3 = nn.ReLU()
        self.bn64  = nn.BatchNorm2d(64)

        self.mx3   = nn.MaxPool2d(2)  #30, 62

        self.conv4 = nn.Conv2d(64, 64, (3,3), stride = 1)
        self.relu4 = nn.ReLU()
        self.bn64_2  = nn.BatchNorm2d(64)

        self.mx4 = nn.MaxPool2d(2) #15, 31

        self.conv5 = nn.Conv2d(64, 64, (3,3), stride = 1)
        self.relu5 = nn.ReLU()
        self.bn64_3  = nn.BatchNorm2d(64)

        self.mx5 = nn.MaxPool2d(2) #7, 15

        #self.conv4 = nn.Conv2d(64, hidden_dim, 1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn16(out)

        out = self.mx1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn32(out)

        out = self.mx2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn64(out)

        out = self.mx3(out)
        
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.bn64_2(out)

        out = self.mx4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.bn64_3(out)

        out = self.mx5(out)

        return out


class CreateCustomBackbone(CustomBackboneBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = CustomBackbone()
        num_channels = 64
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class CreateCustomBackboneWithout1D(CustomBackboneBaseWithout1D):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = CustomBackboneWithout1D()
        num_channels = 64
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        #backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.custom_backbone == 'True':
        backbone = CreateCustomBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    if args.custom_backbone_1d == 'True':
        backbone = CreateCustomBackboneWithout1D(args.backbone, train_backbone, return_interm_layers, args.dilation)
    if args.custom_backbone != 'True' and args.custom_backbone_1d != 'True':
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
