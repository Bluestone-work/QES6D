import copy

import numpy as np
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1, 1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1, 1, bias=True)

    def forward(self, inputs):
        x = torch.nn.functional.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = torch.nn.functional.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, inputs.size(1), 1, 1)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_se=False,
    ):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()
        self.se = SEBlock(out_channels, out_channels // 16) if use_se else nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            padding_11 = padding - kernel_size // 2
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            self.rbr_1x1 = conv_bn(in_channels, out_channels, 1, stride, padding_11, groups)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for parameter in self.parameters():
            parameter.detach_()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super().__init__()
        self.deploy = deploy
        self.override_groups_map = override_groups_map or {}
        self.use_se = use_se
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes, 3, stride=2, padding=1, deploy=deploy, use_se=use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for block_stride in strides:
            groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(self.in_planes, planes, 3, stride=block_stride, padding=1, groups=groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {layer: 2 for layer in optional_groupwise_layers}
g4_map = {layer: 4 for layer in optional_groupwise_layers}


def create_RepVGG_B1g2(deploy=False):
    return RepVGG([4, 6, 16, 1], num_classes=1000, width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


func_dict = {"RepVGG-B1g2": create_RepVGG_B1g2}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(model, save_path=None, do_copy=True):
    converted = copy.deepcopy(model) if do_copy else model
    for module in converted.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(converted.state_dict(), save_path)
    return converted
