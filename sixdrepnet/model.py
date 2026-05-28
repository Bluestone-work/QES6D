import math

import torch
from torch import nn
import torch.nn.functional as F
from backbone.repvgg import get_RepVGG_func_by_name
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights
import utils

class CoordAtt(nn.Module):
    """
    Coordinate Attention:
      - Pool along H and W separately to encode positional information
      - Produce attention maps a_h and a_w
      - out = x * a_h * a_w
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, channels // reduction)

        # (B,C,H,1) and (B,C,1,W)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        x_h = self.pool_h(x)                        # (B,C,H,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)            # (B,C,H+W,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)               # (B,mid,1,W)

        a_h = torch.sigmoid(self.conv_h(y_h))       # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w))       # (B,C,1,W)

        return x * a_h * a_w

def _infer_last_channel(stage: nn.Module) -> int:
    """Infer output channels of RepVGG stage4 (works for deploy/non-deploy)."""
    last_c = None
    for n, m in stage.named_modules():
        if ("rbr_dense" in n or "rbr_reparam" in n) and isinstance(m, nn.Conv2d):
            last_c = m.out_channels
    if last_c is None:
        # fallback: take the last Conv2d
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                last_c = m.out_channels
    if last_c is None:
        raise RuntimeError("Cannot infer last_channel from RepVGG stage4.")
    return last_c

def _infer_stage_out_channels(stage: nn.Module) -> int:
    """Infer output channels of any RepVGG stage."""
    last_c = None
    for n, m in stage.named_modules():
        if ("rbr_dense" in n or "rbr_reparam" in n) and isinstance(m, nn.Conv2d):
            last_c = m.out_channels
    if last_c is None:
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                last_c = m.out_channels
    if last_c is None:
        raise RuntimeError(f"Cannot infer channels from stage.")
    return last_c

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: lightweight channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ============================================
# 原始 SixDRepNet（未修改版本，已注释）
# ============================================
class SixDRepNet_o(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True):
        super(SixDRepNet_o, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel

        self.linear_reg = nn.Linear(fea_dim, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)

# ============================================
# 改进版 SixDRepNet（带 MLP head + 可选 SE）
# ============================================
class SixDRepNet(nn.Module):
    def __init__(self,
                 backbone_name, backbone_file, deploy,
                 pretrained=True,
                 use_se=False):
        super(SixDRepNet, self).__init__()
        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)
        if pretrained:
            checkpoint = torch.load(backbone_file)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            ckpt = {k.replace('module.', ''): v for k,
                    v in checkpoint.items()}  # strip the names
            backbone.load_state_dict(ckpt)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = backbone.stage0, backbone.stage1, backbone.stage2, backbone.stage3, backbone.stage4
        
        last_channel = 0
        for n, m in self.layer4.named_modules():
            if ('rbr_dense' in n or 'rbr_reparam' in n) and isinstance(m, nn.Conv2d):
                last_channel = m.out_channels

        fea_dim = last_channel
        
        # Optional SE attention before GAP
        self.se = SEBlock(fea_dim, reduction=16) if use_se else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # self.linear_reg = nn.Linear(fea_dim, 6)
        self.linear_reg = nn.Sequential(
            nn.Linear(fea_dim, 256),
            nn.GELU(),              # 或 nn.ReLU(inplace=True)
            nn.Dropout(p=0.2),      # p 可调：0.1~0.5 常见
            nn.Linear(256, 6),
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.se(x)  # SE attention
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)


# class SixDRepNet2(nn.Module):
#     def __init__(self, block, layers, fc_layers=1):
#         self.inplanes = 64
#         super(SixDRepNet2, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7)

#         self.linear_reg = nn.Linear(512*block.expansion,6)
      


#         # Vestigial layer from previous experiments
#         self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)

#         x = self.linear_reg(x)        
#         out = utils.compute_rotation_matrix_from_ortho6d(x)

#         return out

# class SixDRepNet_MultiFuse(nn.Module):
#     """
#     SixDRepNet with multi-level feature fusion (RepVGG backbone):
#       - Extract features from stage2, stage3, stage4
#       - Fuse them in an FPN-like manner
#       - Regression head outputs 6D rotation representation
#       - Convert 6D -> rotation matrix via utils.compute_rotation_matrix_from_ortho6d

#     Output:
#       R_pred: (B,3,3)
#     """
#     def __init__(self,
#                  backbone_name="RepVGG-B1g2",
#                  backbone_file="",
#                  deploy=True,
#                  pretrained=True,
#                  fuse_channels=256,
#                  head_dropout=0.2,
#                  use_mlp_head=True):
#         super().__init__()

#         repvgg_fn = get_RepVGG_func_by_name(backbone_name)
#         backbone = repvgg_fn(deploy)

#         # Load backbone pretrained weights (optional)
#         if pretrained:
#             if not backbone_file:
#                 raise ValueError("pretrained=True requires backbone_file path")
#             checkpoint = torch.load(backbone_file, map_location="cpu")
#             if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#                 checkpoint = checkpoint["state_dict"]
#             ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
#             backbone.load_state_dict(ckpt, strict=False)

#         # Stages
#         self.layer0 = backbone.stage0
#         self.layer1 = backbone.stage1
#         self.layer2 = backbone.stage2
#         self.layer3 = backbone.stage3
#         self.layer4 = backbone.stage4

#         # Channel dims
#         c2 = _infer_stage_out_channels(self.layer2)
#         c3 = _infer_stage_out_channels(self.layer3)
#         c4 = _infer_stage_out_channels(self.layer4)

#         # Lateral 1x1 convs to unify dims
#         self.lat2 = nn.Sequential(
#             nn.Conv2d(c2, fuse_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lat3 = nn.Sequential(
#             nn.Conv2d(c3, fuse_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lat4 = nn.Sequential(
#             nn.Conv2d(c4, fuse_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Smooth conv after fusion
#         self.smooth = nn.Sequential(
#             nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.gap = nn.AdaptiveAvgPool2d(1)

#         # Head
#         if use_mlp_head:
#             self.head = nn.Sequential(
#                 nn.Linear(fuse_channels, fuse_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=head_dropout),
#                 nn.Linear(fuse_channels, 6),
#             )
#         else:
#             self.head = nn.Linear(fuse_channels, 6)

#     def forward(self, x):
#         # RepVGG forward
#         x = self.layer0(x)
#         x = self.layer1(x)
#         f2 = self.layer2(x)    # mid-level feature (good for small faces)
#         f3 = self.layer3(f2)
#         f4 = self.layer4(f3)   # high-level semantic feature

#         # Lateral projections
#         p2 = self.lat2(f2)
#         p3 = self.lat3(f3)
#         p4 = self.lat4(f4)

#         # Upsample to p2 resolution and fuse (simple sum)
#         p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
#         p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")

#         fused = p2 + p3u + p4u
#         fused = self.smooth(fused)

#         g = self.gap(fused).flatten(1)  # (B, fuse_channels)
#         out6d = self.head(g)            # (B, 6)

#         R_pred = utils.compute_rotation_matrix_from_ortho6d(out6d)  # (B,3,3)
#         return R_pred

# class SixDRepNet_MultiFuse_CA(nn.Module):
#     """
#     Multi-level feature fusion + Coordinate Attention (after fusion).
#     Output: rotation matrix (B,3,3)
#     """
#     def __init__(self,
#                  backbone_name="RepVGG-B1g2",
#                  backbone_file="",
#                  deploy=True,
#                  pretrained=True,
#                  fuse_channels=256,
#                  head_dropout=0.2,
#                  use_mlp_head=True,
#                  ca_reduction=32):
#         super().__init__()

#         repvgg_fn = get_RepVGG_func_by_name(backbone_name)
#         backbone = repvgg_fn(deploy)

#         if pretrained:
#             if not backbone_file:
#                 raise ValueError("pretrained=True requires backbone_file path")
#             checkpoint = torch.load(backbone_file, map_location="cpu")
#             if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#                 checkpoint = checkpoint["state_dict"]
#             ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
#             backbone.load_state_dict(ckpt, strict=False)

#         self.layer0 = backbone.stage0
#         self.layer1 = backbone.stage1
#         self.layer2 = backbone.stage2
#         self.layer3 = backbone.stage3
#         self.layer4 = backbone.stage4

#         c2 = _infer_stage_out_channels(self.layer2)
#         c3 = _infer_stage_out_channels(self.layer3)
#         c4 = _infer_stage_out_channels(self.layer4)

#         self.lat2 = nn.Sequential(
#             nn.Conv2d(c2, fuse_channels, 1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lat3 = nn.Sequential(
#             nn.Conv2d(c3, fuse_channels, 1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.lat4 = nn.Sequential(
#             nn.Conv2d(c4, fuse_channels, 1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.smooth = nn.Sequential(
#             nn.Conv2d(fuse_channels, fuse_channels, 3, padding=1, bias=False),
#             nn.BatchNorm2d(fuse_channels),
#             nn.ReLU(inplace=True),
#         )

#         # Coordinate Attention inserted here
#         self.ca = CoordAtt(fuse_channels, reduction=ca_reduction)

#         self.gap = nn.AdaptiveAvgPool2d(1)

#         if use_mlp_head:
#             self.head = nn.Sequential(
#                 nn.Linear(fuse_channels, fuse_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=head_dropout),
#                 nn.Linear(fuse_channels, 6),
#             )
#         else:
#             self.head = nn.Linear(fuse_channels, 6)

#     def forward(self, x):
#         x = self.layer0(x)
#         x = self.layer1(x)
#         f2 = self.layer2(x)
#         f3 = self.layer3(f2)
#         f4 = self.layer4(f3)

#         p2 = self.lat2(f2)
#         p3 = self.lat3(f3)
#         p4 = self.lat4(f4)

#         p3u = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
#         p4u = F.interpolate(p4, size=p2.shape[-2:], mode="nearest")

#         fused = p2 + p3u + p4u
#         fused = self.smooth(fused)

#         # ✅ apply coordinate attention after fusion
#         fused = self.ca(fused)

#         g = self.gap(fused).flatten(1)
#         out6d = self.head(g)

#         R_pred = utils.compute_rotation_matrix_from_ortho6d(out6d)
#         return R_pred


class SixDRepNet_StrongHead(nn.Module):
    """
    Single-scale SixDRepNet (use stage4) + stronger regression head:
      GAP -> MLP(LN+GELU+Dropout) -> 6D -> rotation matrix

    Output: R_pred (B,3,3)
    """
    def __init__(self,
                 backbone_name="RepVGG-B1g2",
                 backbone_file="",
                 deploy=True,
                 pretrained=True,
                 head_hidden=512,
                 dropout=0.2):
        super().__init__()

        repvgg_fn = get_RepVGG_func_by_name(backbone_name)
        backbone = repvgg_fn(deploy)

        if pretrained:
            if not backbone_file:
                raise ValueError("pretrained=True requires backbone_file path")
            checkpoint = torch.load(backbone_file, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            backbone.load_state_dict(ckpt, strict=False)

        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4

        self.gap = nn.AdaptiveAvgPool2d(1)

        fea_dim = _infer_last_channel(self.layer4)

        # LN is on vector, so apply after flatten
        self.head = nn.Sequential(
            nn.Linear(fea_dim, head_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, head_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 6, bias=True),
        )
        self.ln = nn.LayerNorm(fea_dim)

        # (optional) init last layer small to stabilize early training
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x).flatten(1)   # (B, fea_dim)
        x = self.ln(x)
        out6d = self.head(x)         # (B, 6)

        R_pred = utils.compute_rotation_matrix_from_ortho6d(out6d)
        return R_pred

# class SixDRepNet_EffNetV2(nn.Module):
#     def __init__(self,
#                  backbone_name='efficientnet_v2_l',
#                  pretrained=True,
#                  use_se=False,
#                  head_dim=512,
#                  dropout=0.2,
#                  geo_dim=128,
#                  use_geometry_refine=True,
#                  head_style='baseline'):
#         super().__init__()
#         if backbone_name == 'efficientnet_v2_s':
#             try:
#                 backbone = efficientnet_v2_s(
#                     weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
#                 )
#             except NameError:
#                 backbone = efficientnet_v2_s(pretrained=pretrained)
#             fea_dim = 1280
#         elif backbone_name == 'efficientnet_v2_m':
#             try:
#                 backbone = efficientnet_v2_m(
#                     weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
#                 )
#             except NameError:
#                 backbone = efficientnet_v2_m(pretrained=pretrained)
#             fea_dim = 1280
#         elif backbone_name == 'efficientnet_v2_l':
#             try:
#                 backbone = efficientnet_v2_l(
#                     weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
#                 )
#             except NameError:
#                 backbone = efficientnet_v2_l(pretrained=pretrained)
#             fea_dim = 1280
#         else:
#             raise ValueError(f"Unsupported backbone_name: {backbone_name}")

#         self.backbone = backbone.features
#         self.se = SEBlock(fea_dim, reduction=16) if use_se else nn.Identity()
#         self.gap = nn.AdaptiveAvgPool2d(1)

#         self.use_geometry_refine = use_geometry_refine
#         self.head_style = str(head_style).lower()

#         if self.head_style == 'bn_relu6':
#             self.pose_proj = nn.Sequential(
#                 nn.Linear(fea_dim, head_dim),
#                 nn.BatchNorm1d(head_dim),
#                 nn.ReLU6(inplace=True),
#                 nn.Dropout(p=dropout),
#             )
#         elif self.head_style == 'baseline':
#             self.pose_proj = nn.Sequential(
#                 nn.Linear(fea_dim, head_dim),
#                 nn.GELU(),
#                 nn.Dropout(p=dropout),
#             )
#         else:
#             raise ValueError(f"Unsupported head_style: {head_style}")
#         self.rot_head = nn.Linear(head_dim, 6)

#         if self.use_geometry_refine:
#             self.geo_proj = nn.Sequential(
#                 nn.Linear(fea_dim, head_dim),
#                 nn.GELU(),
#                 nn.Dropout(p=dropout),
#                 nn.Linear(head_dim, geo_dim),
#             )
#             self.geo_to_feat = nn.Sequential(
#                 nn.Linear(geo_dim, head_dim),
#                 nn.GELU(),
#                 nn.Dropout(p=dropout),
#             )
#             self.refiner = TRGStyleRefiner(feat_dim=head_dim, dropout=dropout)

#             # 稳定训练：初始时先接近 identity refine（delta≈0）
#             nn.init.normal_(self.refiner.delta_head.weight, mean=0.0, std=1e-3)
#             nn.init.zeros_(self.refiner.delta_head.bias)

#     def forward(self, x, return_aux=False):
#         x = self.backbone(x)
#         x = self.se(x)
#         x = self.gap(x)
#         x = torch.flatten(x, 1)

#         pose_feat = self.pose_proj(x)
#         rot6d_init = self.rot_head(pose_feat)

#         if self.use_geometry_refine:
#             geo_latent = self.geo_proj(x)
#             geo_feat = self.geo_to_feat(geo_latent)
#             rot6d_refined = self.refiner(pose_feat, geo_feat, rot6d_init)
#         else:
#             geo_latent = None
#             rot6d_refined = rot6d_init

#         R_pred = utils.compute_rotation_matrix_from_ortho6d(rot6d_refined)

#         if return_aux:
#             aux = {
#                 "R_pred": R_pred,
#                 "R_init": utils.compute_rotation_matrix_from_ortho6d(rot6d_init),
#                 "rot6d_init": rot6d_init,
#                 "rot6d_refined": rot6d_refined,
#             }
#             if geo_latent is not None:
#                 aux["geo_latent"] = geo_latent
#             return aux

#         return R_pred

class SixDRepNet_EffNetV2(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_v2_l',
                 pretrained=True,
                 use_se=False,
                 head_dim=512,
                 dropout=0.2):
        super().__init__()
        if backbone_name == 'efficientnet_v2_s':
            try:
                backbone = efficientnet_v2_s(
                    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_s(pretrained=pretrained)
            fea_dim = 1280
        elif backbone_name == 'efficientnet_v2_m':
            try:
                backbone = efficientnet_v2_m(
                    weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_m(pretrained=pretrained)
            fea_dim = 1280
        elif backbone_name == 'efficientnet_v2_l':
            try:
                backbone = efficientnet_v2_l(
                    weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_l(pretrained=pretrained)
            fea_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")
        self.backbone = backbone.features
        self.se = SEBlock(fea_dim, reduction=16) if use_se else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear_reg = nn.Sequential(
            nn.Linear(fea_dim, head_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_dim, 6),
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.se(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return utils.compute_rotation_matrix_from_ortho6d(x)

class TRGStyleRefiner(nn.Module):
    def __init__(self, feat_dim=256, dropout=0.2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(feat_dim, 6)

    def forward(self, pose_feat, geo_feat, rot6d_init):
        fused = torch.cat([pose_feat, geo_feat], dim=1)
        refined_feat = self.fuse(fused)
        delta_rot = self.delta_head(refined_feat)
        return rot6d_init + delta_rot

class SixDRepNet_EffNetV2_Advanced(nn.Module):
    def __init__(self,
                 backbone_name='efficientnet_v2_m',
                 pretrained=True,
                 use_se=False,
                 head_dim=256,
                 geo_dim=128,
                 dropout=0.2):
        super().__init__()

        if backbone_name == 'efficientnet_v2_s':
            try:
                backbone = efficientnet_v2_s(
                    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_s(pretrained=pretrained)
            fea_dim = 1280

        elif backbone_name == 'efficientnet_v2_m':
            try:
                backbone = efficientnet_v2_m(
                    weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_m(pretrained=pretrained)
            fea_dim = 1280

        elif backbone_name == 'efficientnet_v2_l':
            try:
                backbone = efficientnet_v2_l(
                    weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
                )
            except NameError:
                backbone = efficientnet_v2_l(pretrained=pretrained)
            fea_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.backbone = backbone.features
        self.se = SEBlock(fea_dim, reduction=16) if use_se else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # pose branch
        self.pose_proj = nn.Sequential(
            nn.Linear(fea_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rot_head = nn.Linear(head_dim, 6)

        # geometry branch
        self.geo_proj = nn.Sequential(
            nn.Linear(fea_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, geo_dim),
        )

        self.geo_to_feat = nn.Sequential(
            nn.Linear(geo_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # refinement
        self.refiner = TRGStyleRefiner(feat_dim=head_dim, dropout=dropout)

    def forward(self, x, return_aux=False):
        x = self.backbone(x)
        x = self.se(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)

        pose_feat = self.pose_proj(x)
        rot6d_init = self.rot_head(pose_feat)

        geo_latent = self.geo_proj(x)
        geo_feat = self.geo_to_feat(geo_latent)

        rot6d_refined = self.refiner(pose_feat, geo_feat, rot6d_init)

        R_pred = utils.compute_rotation_matrix_from_ortho6d(rot6d_refined)

        if return_aux:
            R_init = utils.compute_rotation_matrix_from_ortho6d(rot6d_init)
            return {
                "R_pred": R_pred,
                "R_init": R_init,
                "rot6d_init": rot6d_init,
                "rot6d_refined": rot6d_refined,
                "geo_latent": geo_latent,
            }

        return R_pred