"""
半监督训练脚本（论文主训练路径）

当前策略：
1. 支持验证集，并按 val_avg_mae 选择 best_model
2. 支持 skip_pseudo_only_batch，避免纯伪标签 batch 主导训练
3. 支持 face-level pseudo 数据过滤参数
4. 支持 EMA teacher + weak/strong classroom consistency
5. 保存 history.json / config.json
6. 兼容旧格式 batch
"""

import os
import argparse
import random
import json
import importlib.util
import sys
import copy
from datetime import datetime

omp_num_threads = os.environ.get('OMP_NUM_THREADS')
if omp_num_threads is not None:
    try:
        if int(omp_num_threads) <= 0:
            os.environ['OMP_NUM_THREADS'] = '1'
    except ValueError:
        os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

try:
    from QES6D.model import (
        SixDRepNet, SixDRepNet_o, SixDRepNet_StrongHead,
        SixDRepNet_EffNetV2, SixDRepNet_EffNetV2_Advanced,
    )
    from QES6D.semi_supervised_datasets import create_semi_supervised_dataloaders
    from QES6D.loss import GeodesicPlusAxisLoss, RobustEulerAxisLoss
    import QES6D.utils as utils
except ImportError:
    from model import (
        SixDRepNet, SixDRepNet_o, SixDRepNet_StrongHead,
        SixDRepNet_EffNetV2, SixDRepNet_EffNetV2_Advanced,
    )
    from semi_supervised_datasets import create_semi_supervised_dataloaders
    from loss import GeodesicPlusAxisLoss, RobustEulerAxisLoss
    import utils


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_multi_path_args(values):
    """将 argparse 的多次/多值输入统一展开为扁平路径列表。"""
    if values is None:
        return []

    if isinstance(values, str):
        values = [values]

    results = []
    stack = list(values)
    while stack:
        item = stack.pop(0)
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
            continue

        text = str(item).strip()
        if not text:
            continue

        for part in text.replace('\n', ',').split(','):
            part = part.strip()
            if part:
                results.append(part)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-supervised training')

    # 数据路径
    parser.add_argument('--labeled_data_dir', type=str, required=True,
                        help='Labeled dataset npz or root')
    parser.add_argument('--labeled_filename_list', type=str, required=True,
                        help='Labeled dataset filename list')
    parser.add_argument('--pseudo_label_file', '--pseudo_labels_file',
                        dest='pseudo_label_file', nargs='+', action='append', required=True,
                        help='One or more pseudo label files (.npz). Can be passed multiple times')
    parser.add_argument('--pseudo_data_root', '--unlabeled_data_dir',
                        dest='pseudo_data_root', nargs='+', action='append', required=True,
                        help='One or more pseudo label data roots. Can be passed multiple times')

    # 验证集
    parser.add_argument('--val_data_dir', type=str, default='',
                        help='Validation dataset path')
    parser.add_argument('--val_filename_list', type=str, default='',
                        help='Validation filename list')
    parser.add_argument('--selection_metric', type=str, default='val_avg_mae',
                        choices=['val_avg_mae'],
                        help='Metric used to select best model')

    # 伪标签过滤参数（给新的 *_faces.npz 用）
    parser.add_argument('--consistency_threshold_deg', type=float, default=None,
                        help='Max consistency score (deg) for pseudo samples')
    parser.add_argument('--max_model_consistency_deg', type=float, default=None,
                        help='Static max cross-model disagreement (deg) for pseudo samples')
    parser.add_argument('--min_det_confidence', type=float, default=0.0,
                        help='Minimum detector confidence for pseudo samples')
    parser.add_argument('--min_track_length', type=int, default=0,
                        help='Minimum track length for pseudo samples')
    parser.add_argument('--min_blur_score', type=float, default=None,
                        help='Static minimum blur score (Laplacian variance) for pseudo samples')
    parser.add_argument('--disable_require_use_for_training', action='store_true',
                        help='Do not require use_for_training=True in pseudo labels')
    parser.add_argument('--enable_pseudo_curriculum', action='store_true',
                        help='Enable face-size / blur / consensus curriculum for pseudo samples')
    parser.add_argument('--curriculum_ramp_epochs', type=int, default=20,
                        help='Epochs used to relax curriculum thresholds')
    parser.add_argument('--curriculum_face_size_start', type=int, default=72,
                        help='Initial strict minimum face size for pseudo samples')
    parser.add_argument('--curriculum_face_size_end', type=int, default=40,
                        help='Final minimum face size for pseudo samples')
    parser.add_argument('--curriculum_blur_start', type=float, default=120.0,
                        help='Initial strict minimum blur score for pseudo samples')
    parser.add_argument('--curriculum_blur_end', type=float, default=40.0,
                        help='Final minimum blur score for pseudo samples')
    parser.add_argument('--curriculum_model_consistency_start', type=float, default=3.5,
                        help='Initial strict max cross-model disagreement for pseudo samples')
    parser.add_argument('--curriculum_model_consistency_end', type=float, default=6.0,
                        help='Final max cross-model disagreement for pseudo samples')

    # 模型
    parser.add_argument('--pretrained_model', type=str,
                        default='/root/autodl-tmp/output/qes_paper_runs/effnetv2_full_semi_seed/best_model.pth',
                        help='Pretrained model path')
    parser.add_argument('--backbone_file', type=str,
                        default='/root/6DRepNet/sixdrepnet/RepVGG-B1g2-train.pth',
                        help='Backbone weights')
    parser.add_argument('--model_variant', type=str, default='strong_head',
                        choices=['sixdrepnet', 'sixdrepnet_o', 'strong_head', 'effnetv2', 'convnextv2', 'effnetv2_advanced'],
                        help='Model variant from model.py')
    parser.add_argument('--use_se', action='store_true',
                        help='Enable SE block in SixDRepNet / EffNetV2 when supported')
    parser.add_argument('--effnet_backbone', type=str, default='efficientnet_v2_l',
                        choices=['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
                        help='Backbone used when model_variant=effnetv2')
    parser.add_argument('--effnet_geo_dim', type=int, default=128,
                        help='Geometry latent dim used when model_variant=effnetv2')
    parser.add_argument('--effnet_head_style', type=str, default='baseline', choices=['baseline', 'bn_relu6'],
                        help='Head style used when model_variant=effnetv2')
    parser.add_argument('--disable_effnet_geometry_refine', action='store_true',
                        help='Disable geometry latent + refinement in SixDRepNet_EffNetV2')
    parser.add_argument('--disable_torchvision_pretrained', action='store_true',
                        help='Disable torchvision ImageNet pretrained weights download for EffNet/ConvNeXt backbones')
    parser.add_argument('--convnext_backbone', type=str, default='convnextv2_tiny',
                        help='Backbone used when model_variant=convnextv2')
    parser.add_argument('--exp_setting', type=str, default='custom',
                        choices=['custom', 'sup_only', 'pseudo', 'full_semi'],
                        help='Convenience preset for paper experiments')

    # 半监督参数
    parser.add_argument('--pseudo_label_ratio', type=float, default=0.25,
                        help='Ratio of pseudo labels to labeled data')
    parser.add_argument('--confidence_threshold', type=float, default=0.90,
                        help='Confidence threshold for pseudo labels')
    parser.add_argument('--pseudo_loss_weight', '--unlabeled_weight',
                        dest='pseudo_loss_weight', type=float, default=0,
                        help='Weight for pseudo label loss')
    parser.add_argument('--loss_type', type=str, default='geodesic_plus_axis',
                        choices=['geodesic_plus_axis', 'robust_euler'],
                        help='Supervised/pseudo rotation loss type')
    parser.add_argument('--axis_loss_lambda', type=float, default=0.2,
                        help='Axis loss lambda used when loss_type=geodesic_plus_axis')
    parser.add_argument('--axis_weights', type=str, default='1.0,1.0,1.0',
                        help='Axis weights [yaw,pitch,roll] for robust_euler')
    parser.add_argument('--robust_base_loss', type=str, default='huber',
                        choices=['l1', 'huber', 'adaptive_huber'],
                        help='Base loss used by robust_euler')
    parser.add_argument('--huber_delta_deg', type=float, default=4.0,
                        help='Huber delta (degrees) for robust_euler')
    parser.add_argument('--adaptive_gamma', type=float, default=1.5,
                        help='Gamma for adaptive_huber in robust_euler')
    parser.add_argument('--angle_bin_edges', type=str, default='30,60',
                        help='Absolute-angle bin edges (deg) for robust_euler, e.g. 30,60')
    parser.add_argument('--angle_bin_weights', type=str, default='1.0,1.3,1.7',
                        help='Bin weights for robust_euler, length must be len(edges)+1')
    parser.add_argument('--disable_angle_bin_weight', action='store_true',
                        help='Disable angle-bin weighting in robust_euler')
    parser.add_argument('--consistency_weight', type=float, default=0.0,
                        help='Compatibility arg, currently unused in this script')
    parser.add_argument('--equivariance_weight', type=float, default=0.0,
                        help='Deprecated compatibility arg. In-plane rotation equivariance is disabled.')
    parser.add_argument('--equivariance_max_deg', type=float, default=20.0,
                        help='Deprecated compatibility arg. In-plane rotation equivariance is disabled.')
    parser.add_argument('--equivariance_min_deg', type=float, default=0.0,
                        help='Deprecated compatibility arg. In-plane rotation equivariance is disabled.')
    parser.add_argument('--equivariance_sign', type=int, default=1, choices=[-1, 1],
                        help='Deprecated compatibility arg. In-plane rotation equivariance is disabled.')
    parser.add_argument('--equivariance_detach_base', action='store_true',
                        help='Deprecated compatibility arg. In-plane rotation equivariance is disabled.')
    parser.add_argument('--classroom_consistency_weight', type=float, default=0.2,
                        help='Weight for weak/strong classroom consistency on pseudo samples')
    parser.add_argument('--classroom_aug_prob', type=float, default=0.4,
                        help='Probability of strong classroom degradation on pseudo samples')
    parser.add_argument('--teacher_ema_decay', type=float, default=0.999,
                        help='EMA decay for teacher model on pseudo samples')
    parser.add_argument('--disable_teacher_ema', action='store_true',
                        help='Disable EMA teacher and use online weak predictions as pseudo targets')
    parser.add_argument('--disable_eval_with_ema', action='store_true',
                        help='Disable validation/export using EMA weights when teacher EMA is enabled')
    parser.add_argument('--multi_face_mode', action='store_true',
                        help='Multi-face mode for pseudo labels')
    parser.add_argument('--skip_pseudo_only_batch', action='store_true',
                        help='Skip batch if it contains only pseudo samples')
    parser.add_argument('--supervised_only', action='store_true',
                        help='Use only labeled data inside this script for controlled ablation')
    parser.add_argument('--disable_weighted_sampler', action='store_true',
                        help='Disable weighted pseudo sampling and use ordinary shuffle')

    # 可选 refine / enhance
    parser.add_argument('--use_yolo_refine', action='store_true',
                        help='Use YOLO to refine pseudo face crops during training')
    parser.add_argument('--yolo_version', type=str, default='yolov8', choices=['yolov8', 'yolov11'],
                        help='YOLO version for auto face detector weights selection')
    parser.add_argument('--yolo_size', type=str, default='x', choices=['n', 'x'],
                        help='Preferred YOLO model size when auto-selecting weights')
    parser.add_argument('--strict_yolo_version', action='store_true',
                        help='Strict YOLO version/size: no fallback to other local weights')
    parser.add_argument('--yolo_weights', '--yolo_model', dest='yolo_weights', type=str,
                        default='',
                        help='YOLO face model path for crop refinement (overrides --yolo_version auto selection)')
    parser.add_argument('--face_det_threshold', type=float, default=0.5,
                        help='Face detection threshold for YOLO refinement')
    parser.add_argument('--min_face_size', type=int, default=40,
                        help='Minimum face size for pseudo samples')
    parser.add_argument('--expand_bbox_ratio', type=float, default=0.2,
                        help='Expand ratio for pseudo face bbox before crop')
    parser.add_argument('--use_gfpgan', action='store_true',
                        help='Use GFPGAN enhancement for low-resolution pseudo faces')
    parser.add_argument('--gfpgan_model', type=str, default=None,
                        help='GFPGAN model path (auto-detect if omitted)')
    parser.add_argument('--gfpgan_low_conf_only', action='store_true',
                        help='Apply GFPGAN only on low-confidence pseudo labels')
    parser.add_argument('--gfpgan_low_conf_threshold', type=float, default=0.85,
                        help='Confidence threshold for low-confidence GFPGAN gating')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', '--num_epochs', dest='epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup epochs for pseudo label loss')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                        choices=['cosine', 'multistep', 'none'],
                        help='LR scheduler type. Use multistep to match train.py more closely')
    parser.add_argument('--scheduler_milestones', type=str, default='10,20',
                        help='Comma-separated milestones for multistep scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                        help='Gamma for multistep scheduler')

    # 其他
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', '--output_path', dest='output_dir', type=str,
                        default='/root/autodl-tmp/output/semi_supervised')
    parser.add_argument('--save_freq', type=int, default=25)
    parser.add_argument('--save_best_model', action='store_true',
                        help='Save best model according to validation metric')
    parser.add_argument('--print_freq', type=int, default=75)
    parser.add_argument('--timestamp_subdir', action='store_true',
                        help='Append timestamp subdir inside output_dir')
    parser.add_argument('--resume_checkpoint', type=str, default='',
                        help='Resume training from a checkpoint/best/final model file')
    parser.add_argument('--resume_history_path', type=str, default='',
                        help='History json path to load and continue appending')

    args = parser.parse_args()
    args.yolo_weights = utils.resolve_yolo_weights(
        args.yolo_weights,
        args.yolo_version,
        prefer_size=args.yolo_size,
        strict_version=args.strict_yolo_version
    )
    args.yolo_model = args.yolo_weights
    if args.use_yolo_refine:
        print(f"[YOLO] version={args.yolo_version}, size={args.yolo_size}, strict={args.strict_yolo_version}, resolved_weights={args.yolo_weights}")
    args.pseudo_label_file = parse_multi_path_args(args.pseudo_label_file)
    args.pseudo_data_root = parse_multi_path_args(args.pseudo_data_root)
    return args


def geodesic_distance_batch(R1, R2, eps=1e-7):
    R_rel = torch.bmm(R1, R2.transpose(1, 2))
    tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos = (tr - 1.0) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def compute_pseudo_loss_weight(epoch, warmup_epochs, max_weight):
    if warmup_epochs <= 0:
        return max_weight
    if epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return max_weight
        return max_weight * ((epoch - 1) / (warmup_epochs - 1))
    return max_weight


def parse_int_list(text, default=None):
    default = [10, 20] if default is None else default
    if not text:
        return default

    values = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError:
            continue
    return values if values else default


def parse_float_list(text, default=None):
    default = [] if default is None else default
    if text is None:
        return list(default)

    values = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values if values else list(default)


def interpolate_curriculum_value(epoch, ramp_epochs, start_value, end_value):
    if start_value is None or end_value is None:
        return None
    ramp_epochs = max(int(ramp_epochs), 1)
    progress = min(max((int(epoch) - 1) / float(max(ramp_epochs - 1, 1)), 0.0), 1.0)
    return float(start_value) + (float(end_value) - float(start_value)) * progress


def compute_curriculum_thresholds(args, epoch):
    if args.supervised_only or (not args.enable_pseudo_curriculum):
        return {
            'curriculum_min_face_size': None,
            'curriculum_min_blur_score': None,
            'curriculum_max_model_consistency_deg': None,
        }

    face_size = int(round(interpolate_curriculum_value(
        epoch, args.curriculum_ramp_epochs,
        args.curriculum_face_size_start, args.curriculum_face_size_end
    )))
    blur_score = interpolate_curriculum_value(
        epoch, args.curriculum_ramp_epochs,
        args.curriculum_blur_start, args.curriculum_blur_end
    )
    model_consistency = interpolate_curriculum_value(
        epoch, args.curriculum_ramp_epochs,
        args.curriculum_model_consistency_start, args.curriculum_model_consistency_end
    )

    return {
        'curriculum_min_face_size': face_size,
        'curriculum_min_blur_score': blur_score,
        'curriculum_max_model_consistency_deg': model_consistency,
    }


def wrap_abs_diff_deg_torch(a, b):
    return torch.abs((a - b + 180.0) % 360.0 - 180.0)


def apply_strong_classroom_aug_tensor(images, prob=0.4):
    if images is None or images.numel() == 0:
        return images

    x = images.clone()
    batch_size, _, height, width = x.shape
    device = x.device

    for i in range(batch_size):
        if torch.rand(1, device=device).item() > float(prob):
            continue

        if torch.rand(1, device=device).item() < 0.40:
            scale = float(torch.empty(1, device=device).uniform_(0.5, 0.85).item())
            h2 = max(16, int(height * scale))
            w2 = max(16, int(width * scale))
            xi = F.interpolate(x[i:i+1], size=(h2, w2), mode='bilinear', align_corners=False)
            x[i:i+1] = F.interpolate(xi, size=(height, width), mode='bilinear', align_corners=False)

        if torch.rand(1, device=device).item() < 0.35:
            kernel = 5 if torch.rand(1, device=device).item() < 0.5 else 3
            x[i:i+1] = F.avg_pool2d(x[i:i+1], kernel_size=kernel, stride=1, padding=kernel // 2)

        if torch.rand(1, device=device).item() < 0.35:
            alpha = float(torch.empty(1, device=device).uniform_(0.55, 0.95).item())
            beta = float(torch.empty(1, device=device).uniform_(-0.15, 0.05).item())
            x[i] = torch.clamp(x[i] * alpha + beta, -3.0, 3.0)

        if torch.rand(1, device=device).item() < 0.25:
            noise_std = float(torch.empty(1, device=device).uniform_(0.01, 0.035).item())
            x[i] = torch.clamp(x[i] + torch.randn_like(x[i]) * noise_std, -3.0, 3.0)

        if torch.rand(1, device=device).item() < 0.30:
            occ_h = max(8, int(height * float(torch.empty(1, device=device).uniform_(0.12, 0.28).item())))
            occ_w = max(8, int(width * float(torch.empty(1, device=device).uniform_(0.12, 0.28).item())))
            y0 = int(torch.randint(0, max(1, height - occ_h + 1), (1,), device=device).item())
            x0 = int(torch.randint(0, max(1, width - occ_w + 1), (1,), device=device).item())
            fill = x[i].mean()
            x[i, :, y0:y0 + occ_h, x0:x0 + occ_w] = fill

    return x


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key, param in model_state.items():
        ema_param = ema_state[key]
        if torch.is_floating_point(ema_param):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        else:
            ema_param.copy_(param)


def build_val_loader(val_data_dir, val_filename_list, batch_size, num_workers):
    from torchvision import transforms
    from torch.utils.data import DataLoader
    try:
        from QES6D.semi_supervised_datasets import FlatNPZPoseDataset
    except ImportError:
        from semi_supervised_datasets import FlatNPZPoseDataset

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 解析并修复验证列表路径（兼容 biwi_val.txt / val.txt 命名差异）
    val_filename_list = os.path.expanduser(str(val_filename_list))
    if not os.path.isfile(val_filename_list):
        requested_name = os.path.basename(val_filename_list)
        alt_names = [requested_name]
        if requested_name.startswith('biwi_'):
            alt_names.append(requested_name[len('biwi_'):])
        if 'val' in requested_name.lower() and 'val.txt' not in alt_names:
            alt_names.append('val.txt')

        search_dirs = []
        if os.path.dirname(val_filename_list):
            search_dirs.append(os.path.dirname(val_filename_list))

        val_data_dir_expanded = os.path.expanduser(str(val_data_dir))
        if os.path.isdir(val_data_dir_expanded):
            search_dirs.append(val_data_dir_expanded)
            search_dirs.append(os.path.join(val_data_dir_expanded, 'data_split_lists'))

        val_parent = os.path.dirname(val_data_dir_expanded)
        if val_parent:
            search_dirs.append(os.path.join(val_parent, 'data_split_lists'))

        project_root = os.path.dirname(__file__)
        search_dirs.append(os.path.join(project_root, 'data_split_lists'))

        dedup_dirs = []
        for d in search_dirs:
            if d and d not in dedup_dirs and os.path.isdir(d):
                dedup_dirs.append(d)

        resolved_path = None
        for d in dedup_dirs:
            for name in alt_names:
                candidate = os.path.join(d, name)
                if os.path.isfile(candidate):
                    resolved_path = candidate
                    break
            if resolved_path is not None:
                break

        if resolved_path is None:
            available_txt = []
            for d in dedup_dirs:
                try:
                    txts = sorted([x for x in os.listdir(d) if x.endswith('.txt')])
                except Exception:
                    txts = []
                if txts:
                    available_txt.append(f"{d}: {txts}")

            msg = (
                f"Validation filename list not found: {val_filename_list}. "
                f"Tried names={alt_names} in dirs={dedup_dirs}."
            )
            if available_txt:
                msg += " Available .txt files -> " + " | ".join(available_txt)
            raise FileNotFoundError(msg)

        print(f"[VAL] val_filename_list not found, fallback to: {resolved_path}")
        val_filename_list = resolved_path

    # 优先识别你自己的扁平 npz
    use_flat_npz = False
    if os.path.isfile(val_data_dir) and val_data_dir.endswith('.npz'):
        try:
            pack = np.load(val_data_dir, allow_pickle=True)
            if 'image' in pack and 'pose' in pack:
                use_flat_npz = True
        except Exception:
            use_flat_npz = False

    if use_flat_npz:
        print("Using FlatNPZPoseDataset for validation data")
        val_dataset = FlatNPZPoseDataset(
            val_data_dir,
            val_filename_list,
            transformations,
            train_mode=False
        )
    else:
        # 仅从项目内加载，避免误导入外部同名 datasets 包
        project_root = os.path.dirname(__file__)
        candidate_paths = [
            os.path.join(project_root, 'QES6D', 'datasets.py'),
            os.path.join(project_root, 'datasets.py'),
        ]
        datasets_py = next((p for p in candidate_paths if os.path.isfile(p)), None)
        if datasets_py is None:
            raise FileNotFoundError(
                f"Validation fallback dataset file not found. Tried: {candidate_paths}"
            )

        spec = importlib.util.spec_from_file_location('QES6D_local_datasets', datasets_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {datasets_py}")
        datasets_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datasets_module)

        if hasattr(datasets_module, 'BIWI'):
            dataset_cls = datasets_module.BIWI
        else:
            raise ImportError(f"No BIWI dataset class found in {datasets_py}")

        try:
            val_dataset = dataset_cls(
                val_data_dir,
                val_filename_list,
                transformations,
                train_mode=False
            )
        except TypeError:
            val_dataset = dataset_cls(
                val_data_dir,
                val_filename_list,
                transformations
            )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


def build_train_loader_with_curriculum(args, epoch):
    curriculum = compute_curriculum_thresholds(args, epoch)
    train_loader, dataset_stats = create_semi_supervised_dataloaders(
        labeled_data_dir=args.labeled_data_dir,
        labeled_filename_list=args.labeled_filename_list,
        pseudo_label_file=args.pseudo_label_file,
        pseudo_data_root=args.pseudo_data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pseudo_label_ratio=args.pseudo_label_ratio,
        confidence_threshold=args.confidence_threshold,
        multi_face_mode=args.multi_face_mode,
        use_yolo_refine=args.use_yolo_refine,
        yolo_model_path=args.yolo_model,
        face_det_threshold=args.face_det_threshold,
        min_face_size=args.min_face_size,
        expand_bbox_ratio=args.expand_bbox_ratio,
        use_gfpgan=args.use_gfpgan,
        gfpgan_model_path=args.gfpgan_model,
        gfpgan_low_conf_only=args.gfpgan_low_conf_only,
        gfpgan_low_conf_threshold=args.gfpgan_low_conf_threshold,
        consistency_threshold_deg=args.consistency_threshold_deg,
        max_model_consistency_deg=args.max_model_consistency_deg,
        min_det_confidence=args.min_det_confidence,
        min_track_length=args.min_track_length,
        require_use_for_training=(not args.disable_require_use_for_training),
        include_pseudo=(not args.supervised_only),
        use_weighted_sampler=(not args.disable_weighted_sampler),
        seed=args.seed + max(int(epoch) - 1, 0),
        min_blur_score=args.min_blur_score,
        curriculum_min_face_size=curriculum['curriculum_min_face_size'],
        curriculum_min_blur_score=curriculum['curriculum_min_blur_score'],
        curriculum_max_model_consistency_deg=curriculum['curriculum_max_model_consistency_deg'],
    )
    dataset_stats['epoch'] = int(epoch)
    dataset_stats.update(curriculum)
    return train_loader, dataset_stats

def validate_one_epoch(model, val_loader, device):
    model.eval()

    total = 0
    yaw_err = 0.0
    pitch_err = 0.0
    roll_err = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                images, _, cont_labels, _ = batch
            elif len(batch) == 5:
                images, _, cont_labels, _, _ = batch
            else:
                raise ValueError(f"Unexpected val batch format: {len(batch)}")

            images = images.to(device, non_blocking=True)
            cont_labels = cont_labels.to(device)

            y_gt = cont_labels[:, 0].float() * 180.0 / np.pi
            p_gt = cont_labels[:, 1].float() * 180.0 / np.pi
            r_gt = cont_labels[:, 2].float() * 180.0 / np.pi

            R_pred = model(images)
            euler_deg = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180.0 / np.pi
            p_pred = euler_deg[:, 0]
            y_pred = euler_deg[:, 1]
            r_pred = euler_deg[:, 2]

            dy = wrap_abs_diff_deg_torch(y_pred, y_gt)
            dp = wrap_abs_diff_deg_torch(p_pred, p_gt)
            dr = wrap_abs_diff_deg_torch(r_pred, r_gt)

            yaw_err += float(dy.sum().item())
            pitch_err += float(dp.sum().item())
            roll_err += float(dr.sum().item())
            total += int(images.shape[0])

    if total == 0:
        raise RuntimeError('Validation loader produced zero samples.')

    mae_yaw = yaw_err / total
    mae_pitch = pitch_err / total
    mae_roll = roll_err / total
    mae_avg = (mae_yaw + mae_pitch + mae_roll) / 3.0

    return {
        'yaw_mae': mae_yaw,
        'pitch_mae': mae_pitch,
        'roll_mae': mae_roll,
        'avg_mae': mae_avg,
        'n_samples': total
    }


def build_checkpoint_payload(model, ema_model, optimizer, scheduler, train_stats, val_stats,
                             epoch, metric, args, best_metric=None, export_source='student'):
    if export_source == 'ema':
        if ema_model is None:
            raise ValueError('EMA export requested but ema_model is None')
        primary_state = ema_model.state_dict()
    else:
        primary_state = model.state_dict()

    payload = {
        'epoch': epoch,
        'model_state_dict': primary_state,
        'model_state_dict_student': model.state_dict(),
        'model_state_dict_ema': (ema_model.state_dict() if ema_model is not None else None),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'metric': metric,
        'export_source': export_source,
    }
    if best_metric is not None:
        payload['best_metric'] = best_metric
    if args is not None:
        payload['args'] = vars(args)
    return payload


def train_one_epoch(model, ema_model, train_loader, criterion_sup, criterion_pseudo, optimizer,
                    device, epoch, args, pseudo_loss_weight):
    model.train()
    if ema_model is not None:
        ema_model.eval()

    total_loss = 0.0
    total_sup_loss = 0.0
    total_pseudo_loss = 0.0
    total_consistency_loss = 0.0
    total_samples = 0

    labeled_count = 0
    pseudo_count = 0

    is_tty = sys.stdout.isatty()
    pbar = tqdm(
        train_loader,
        desc=f'Epoch {epoch}/{args.epochs}',
        ascii=True,
        dynamic_ncols=is_tty,
        ncols=(None if is_tty else 120),
        mininterval=0.5,
        file=sys.stdout,
        leave=False,
    )

    for batch_idx, batch in enumerate(pbar):
        source_is_pseudo = None

        if len(batch) == 3:
            images, R_gt, confidences = batch
        elif len(batch) == 5:
            images, R_gt, _, confidences, _ = batch
        elif len(batch) == 6:
            images, R_gt, _, confidences, _, source_is_pseudo = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        images = images.to(device, non_blocking=True)
        R_gt = R_gt.to(device)
        confidences = confidences.to(device).reshape(-1)
        if source_is_pseudo is not None:
            source_is_pseudo = source_is_pseudo.to(device)

        # 分离 labeled / pseudo
        if source_is_pseudo is not None:
            is_pseudo = source_is_pseudo & (confidences >= args.confidence_threshold)
            is_labeled = ~source_is_pseudo
        else:
            is_labeled = torch.isclose(confidences, torch.ones_like(confidences), atol=1e-6)
            is_pseudo = (confidences < 1.0) & (confidences >= args.confidence_threshold)

        is_pseudo_active = is_pseudo if pseudo_loss_weight > 0 else torch.zeros_like(is_pseudo)

        n_labeled = int(is_labeled.sum().item())
        n_pseudo = int(is_pseudo_active.sum().item())

        # 统计计数：只统计真正参与训练的伪标签
        labeled_count += n_labeled
        pseudo_count += n_pseudo

        # 如果纯伪标签 batch 不允许，则直接跳过
        if n_labeled == 0 and n_pseudo > 0 and args.skip_pseudo_only_batch:
            continue
        if n_labeled == 0 and n_pseudo == 0:
            continue

        active_mask = is_labeled | is_pseudo_active
        images_active = images[active_mask]
        R_gt_active = R_gt[active_mask]
        confidences_active = confidences[active_mask]
        is_labeled_active = is_labeled[active_mask]
        is_pseudo_active = is_pseudo_active[active_mask]

        R_pred = model(images_active)

        sup_loss = torch.tensor(0.0, device=device)
        pseudo_loss = torch.tensor(0.0, device=device)
        consistency_loss = torch.tensor(0.0, device=device)

        if n_labeled > 0:
            sup_loss = criterion_sup(R_pred[is_labeled_active], R_gt_active[is_labeled_active])

        if n_pseudo > 0 and pseudo_loss_weight > 0:
            pseudo_confidences = confidences_active[is_pseudo_active].to(dtype=R_pred.dtype)
            pseudo_base_loss = criterion_pseudo(
                R_pred[is_pseudo_active],
                R_gt_active[is_pseudo_active]
            ).reshape(-1)
            weight_sum = torch.clamp(pseudo_confidences.sum(), min=1e-6)
            pseudo_loss = (pseudo_base_loss * pseudo_confidences).sum() / weight_sum
            pseudo_loss = pseudo_loss * pseudo_loss_weight

        if n_pseudo > 0 and pseudo_loss_weight > 0 and args.classroom_consistency_weight > 0:
            pseudo_images_weak = images_active[is_pseudo_active]
            pseudo_images_strong = apply_strong_classroom_aug_tensor(
                pseudo_images_weak,
                prob=args.classroom_aug_prob
            )

            if ema_model is not None:
                with torch.no_grad():
                    teacher_targets = ema_model(pseudo_images_weak)
            else:
                teacher_targets = R_pred[is_pseudo_active].detach()

            strong_pred = model(pseudo_images_strong)
            pseudo_confidences = confidences_active[is_pseudo_active].to(dtype=strong_pred.dtype)
            consistency_base_loss = criterion_pseudo(strong_pred, teacher_targets).reshape(-1)
            weight_sum = torch.clamp(pseudo_confidences.sum(), min=1e-6)
            consistency_loss = (consistency_base_loss * pseudo_confidences).sum() / weight_sum
            consistency_loss = consistency_loss * pseudo_loss_weight * args.classroom_consistency_weight

        if n_labeled > 0 and n_pseudo > 0:
            loss = sup_loss + pseudo_loss + consistency_loss
        elif n_labeled > 0:
            loss = sup_loss + consistency_loss
        else:
            loss = pseudo_loss + consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, args.teacher_ema_decay)

        batch_size = images_active.size(0)
        total_loss += float(loss.item()) * batch_size
        if n_labeled > 0:
            total_sup_loss += float(sup_loss.item()) * n_labeled
        if n_pseudo > 0:
            total_pseudo_loss += float(pseudo_loss.item()) * n_pseudo
        total_samples += batch_size
        total_consistency_loss += float(consistency_loss.item()) * batch_size

        if (batch_idx + 1) % args.print_freq == 0 or batch_idx == 0:
            avg_loss = total_loss / max(total_samples, 1)
            avg_sup = total_sup_loss / max(labeled_count, 1)
            avg_pseudo = total_pseudo_loss / max(pseudo_count, 1)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sup': f'{sup_loss.item():.4f}',
                'pseudo': f'{pseudo_loss.item():.4f}',
                'cons': f'{consistency_loss.item():.4f}',
                'L/P': f'{n_labeled}/{n_pseudo}'
            })

            print(
                f"\nEpoch [{epoch}/{args.epochs}], Iter [{batch_idx+1}/{len(train_loader)}], "
                f"Loss: {avg_loss:.4f} (Sup: {avg_sup:.4f}, Pseudo: {avg_pseudo:.4f}, Consistency: {total_consistency_loss / max(total_samples, 1):.4f}, "
                f"Labeled: {labeled_count}, Pseudo: {pseudo_count})"
            )

    stats = {
        'loss': total_loss / max(total_samples, 1),
        'sup_loss': total_sup_loss / max(labeled_count, 1),
        'pseudo_loss': total_pseudo_loss / max(pseudo_count, 1),
        'consistency_loss': total_consistency_loss / max(total_samples, 1),
        'labeled_count': labeled_count,
        'pseudo_count': pseudo_count
    }
    return stats



def apply_experiment_preset(args):
    """Apply paper experiment presets while allowing explicit CLI overrides to win."""
    if args.exp_setting == 'sup_only':
        args.supervised_only = True
        args.pseudo_loss_weight = 0.0
    elif args.exp_setting == 'pseudo':
        args.supervised_only = False
        if args.pseudo_loss_weight <= 0:
            args.pseudo_loss_weight = 1.0
    elif args.exp_setting == 'full_semi':
        args.supervised_only = False
        if args.pseudo_loss_weight <= 0:
            args.pseudo_loss_weight = 1.0
    return args


def create_model_from_args(args, force_torchvision_pretrained=None):
    tv_pretrained = (not args.disable_torchvision_pretrained)
    if force_torchvision_pretrained is not None:
        tv_pretrained = bool(force_torchvision_pretrained)

    mv = str(args.model_variant).lower()
    if mv == 'sixdrepnet_o':
        return SixDRepNet_o(
            backbone_name='RepVGG-B1g2',
            backbone_file=args.backbone_file,
            deploy=False,
            pretrained=True
        )
    if mv == 'strong_head':
        return SixDRepNet_StrongHead(
            backbone_name='RepVGG-B1g2',
            backbone_file=args.backbone_file,
            deploy=False,
            pretrained=True
        )
    if mv == 'effnetv2':
        return SixDRepNet_EffNetV2(
            backbone_name=args.effnet_backbone,
            pretrained=tv_pretrained,
            use_se=args.use_se,
            head_dim=512,
            dropout=0.2,
            # geo_dim=args.effnet_geo_dim,
            # use_geometry_refine=(not args.disable_effnet_geometry_refine),
            # head_style=args.effnet_head_style,
        )
    if mv == 'convnextv2':
        return SixDRepNet_ConvNeXtV2(
            backbone_name=args.convnext_backbone,
            pretrained=tv_pretrained,
            use_cbam=True,
            use_coordconv=True,
            head_dim=512,
            dropout=0.2,
        )
    if mv == 'effnetv2_advanced':
        return SixDRepNet_EffNetV2_Advanced(
            backbone_name='efficientnet_v2_s',
            pretrained=tv_pretrained,
            use_se=True,
            head_dim=256,
            geo_dim=128,
            dropout=0.2
        )
    # return SixDRepNet(
    #     backbone_name='RepVGG-B1g2',
    #     backbone_file=args.backbone_file,
    #     deploy=False,
    #     pretrained=True,
    #     use_se=args.use_se,
    return SixDRepNet_EffNetV2_Advanced(
            backbone_name='efficientnet_v2_s',
            pretrained=tv_pretrained,
            use_se=True,
            head_dim=256,
            geo_dim=128,
            dropout=0.2
        )


def main():
    args = apply_experiment_preset(parse_args())
    set_seed(args.seed)

    if (
        args.equivariance_weight != 0.0
        or args.equivariance_max_deg != 20.0
        or args.equivariance_min_deg != 0.0
        or args.equivariance_sign != 1
        or args.equivariance_detach_base
    ):
        print(
            "[WARN] In-plane rotation equivariance is disabled in this script because rotating the full image "
            "does not induce a physically consistent head-pose transform. The provided equivariance_* args will be ignored."
        )
    args.equivariance_weight = 0.0
    args.equivariance_max_deg = 0.0
    args.equivariance_min_deg = 0.0
    args.equivariance_sign = 1
    args.equivariance_detach_base = False

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.timestamp_subdir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(args.output_dir, f'semi_supervised_{timestamp}')
    else:
        save_dir = args.output_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    print("\n" + "=" * 60)
    print("Creating dataloaders...")
    print("=" * 60)
    print(f"Pseudo label files: {args.pseudo_label_file}")
    print(f"Pseudo data roots: {args.pseudo_data_root}")
    train_loader, dataset_stats = build_train_loader_with_curriculum(args, epoch=1)

    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print(f"Total batches: {len(train_loader)}")
    print(f"Labeled samples: {dataset_stats['n_labeled']}")
    print(f"Pseudo samples: {dataset_stats['n_pseudo']}")
    print(f"Total samples: {dataset_stats['total']}")
    pseudo_ratio = dataset_stats.get('pseudo_ratio', 0.0)
    if pseudo_ratio == 0.0 and dataset_stats.get('total', 0) > 0:
        pseudo_ratio = dataset_stats['n_pseudo'] / dataset_stats['total']
    print(f"Pseudo ratio: {pseudo_ratio:.1%}")
    if 'sampled_pseudo_ratio_target' in dataset_stats:
        print(f"Target sampled pseudo fraction: {dataset_stats['sampled_pseudo_ratio_target']:.1%}")
    print("=" * 60 + "\n")

    if dataset_stats['n_pseudo'] == 0 and not args.supervised_only:
        print("WARNING: No pseudo-labeled samples found!")
        print(f"  pseudo_label_file: {args.pseudo_label_file}")
        print(f"  confidence_threshold: {args.confidence_threshold}")
        print(f"  pseudo_data_root: {args.pseudo_data_root}")
        print("  -> Continue with supervised-only batches for now; curriculum/filters may admit pseudo samples in later epochs.")

    val_loader = None
    if args.val_data_dir and args.val_filename_list:
        print("Building validation loader...")
        val_loader = build_val_loader(
            args.val_data_dir,
            args.val_filename_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    print("Creating model...")
    try:
        model = create_model_from_args(args)
    except RuntimeError as e:
        err_text = str(e).lower()
        maybe_torchvision_download_issue = (
            ('invalid hash value' in err_text) or
            ('url' in err_text and 'download' in err_text) or
            ('load_state_dict_from_url' in err_text)
        )
        if maybe_torchvision_download_issue and (not args.disable_torchvision_pretrained):
            print(f"[WARN] Backbone pretrained weights download failed: {e}")
            print("[WARN] Retrying model creation with torchvision pretrained disabled...")
            model = create_model_from_args(args, force_torchvision_pretrained=False)
        else:
            raise
    # model = SixDRepNet_o(
    #     backbone_name='RepVGG-B1g2',
    #     backbone_file=args.backbone_file,
    #     deploy=False,
    #     pretrained=True
    # )

    if args.pretrained_model:
        print(f"Loading pretrained model: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        try:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded pretrained weights with strict=True")
        except RuntimeError as e:
            print(f"Strict load failed: {e}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights with strict=False "
                  f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")

    model.to(device)
    ema_model = None

    if args.loss_type == 'geodesic_plus_axis':
        criterion_sup = GeodesicPlusAxisLoss(reduction="mean", lambda_axis=args.axis_loss_lambda)
        criterion_pseudo = GeodesicPlusAxisLoss(reduction="none", lambda_axis=args.axis_loss_lambda)
    else:
        axis_weights = parse_float_list(args.axis_weights, default=[1.2, 1.2, 1.0])
        if len(axis_weights) != 3:
            raise ValueError(f"--axis_weights must contain 3 values [yaw,pitch,roll], got: {args.axis_weights}")

        bin_edges = parse_float_list(args.angle_bin_edges, default=[30.0, 60.0])
        bin_weights = parse_float_list(args.angle_bin_weights, default=[1.0, 1.3, 1.7])
        if len(bin_weights) != len(bin_edges) + 1:
            raise ValueError(
                f"--angle_bin_weights length must be len(angle_bin_edges)+1, got edges={len(bin_edges)} weights={len(bin_weights)}"
            )

        criterion_sup = RobustEulerAxisLoss(
            axis_weights=axis_weights,
            base_loss=args.robust_base_loss,
            huber_delta_deg=args.huber_delta_deg,
            adaptive_gamma=args.adaptive_gamma,
            bin_edges_deg=bin_edges,
            bin_weights=bin_weights,
            enable_angle_bin_weight=(not args.disable_angle_bin_weight),
            reduction="mean",
        )
        criterion_pseudo = RobustEulerAxisLoss(
            axis_weights=axis_weights,
            base_loss=args.robust_base_loss,
            huber_delta_deg=args.huber_delta_deg,
            adaptive_gamma=args.adaptive_gamma,
            bin_edges_deg=bin_edges,
            bin_weights=bin_weights,
            enable_angle_bin_weight=(not args.disable_angle_bin_weight),
            reduction="none",
        )
    # 保证 loss 模块内部 buffer 与模型/输入位于同一设备
    criterion_sup = criterion_sup.to(device)
    criterion_pseudo = criterion_pseudo.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler_type == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=parse_int_list(args.scheduler_milestones),
            gamma=args.scheduler_gamma
        )
    else:
        scheduler = None

    config = vars(args).copy()
    config['dataset_stats'] = dataset_stats
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    use_ema_for_eval = (not args.disable_teacher_ema) and (not args.disable_eval_with_ema)

    print("\n" + "=" * 60)
    print("Starting Semi-Supervised Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Pseudo ratio: {args.pseudo_label_ratio}")
    print(f"Pseudo loss max weight: {args.pseudo_loss_weight}")
    print(f"Classroom consistency weight: {args.classroom_consistency_weight}")
    print(f"Classroom strong aug prob: {args.classroom_aug_prob}")
    print(f"Teacher EMA enabled: {not args.disable_teacher_ema}")
    print(f"Teacher EMA decay: {args.teacher_ema_decay}")
    print(f"Pseudo curriculum enabled: {args.enable_pseudo_curriculum}")
    if args.enable_pseudo_curriculum:
        print(
            f"Curriculum face size: {args.curriculum_face_size_start} -> {args.curriculum_face_size_end}, "
            f"blur: {args.curriculum_blur_start} -> {args.curriculum_blur_end}, "
            f"model_consistency: {args.curriculum_model_consistency_start} -> {args.curriculum_model_consistency_end}"
        )
    print(f"Validate/export with EMA: {use_ema_for_eval}")
    print(f"Loss type: {args.loss_type}")
    if args.loss_type == 'geodesic_plus_axis':
        print(f"Axis loss lambda: {args.axis_loss_lambda}")
    else:
        print(f"Axis weights [yaw,pitch,roll]: {args.axis_weights}")
        print(f"Robust base loss: {args.robust_base_loss}")
        print(f"Huber delta (deg): {args.huber_delta_deg}")
        print(f"Adaptive gamma: {args.adaptive_gamma}")
        print(f"Angle bin edges: {args.angle_bin_edges}")
        print(f"Angle bin weights: {args.angle_bin_weights}")
        print(f"Angle bin weight enabled: {not args.disable_angle_bin_weight}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"Supervised only: {args.supervised_only}")
    print(f"Weighted sampler: {not args.disable_weighted_sampler}")
    print(f"Scheduler: {args.scheduler_type}")
    print("=" * 60 + "\n")

    def compute_best_metric_from_history(hist, use_val):
        if not hist:
            return float('inf')
        best = float('inf')
        for rec in hist:
            if use_val:
                val = rec.get('val_avg_mae')
            else:
                val = rec.get('train_loss')
            if val is None:
                continue
            try:
                val = float(val)
            except Exception:
                continue
            if val < best:
                best = val
        return best

    history = []
    start_epoch = 1
    best_metric = float('inf')

    if args.resume_history_path and os.path.isfile(args.resume_history_path):
        try:
            with open(args.resume_history_path, 'r', encoding='utf-8') as f:
                loaded_history = json.load(f)
            if isinstance(loaded_history, list):
                history = loaded_history
                if history:
                    start_epoch = int(history[-1].get('epoch', 0)) + 1
                print(f"Loaded resume history: {args.resume_history_path} (records={len(history)})")
        except Exception as e:
            print(f"[WARN] Failed to load resume history {args.resume_history_path}: {e}")

    if args.resume_checkpoint:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume_checkpoint}")
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        resume_ckpt = torch.load(args.resume_checkpoint, map_location=device)

        if isinstance(resume_ckpt, dict):
            model_state = resume_ckpt.get('model_state_dict_student')
            if model_state is None:
                model_state = resume_ckpt.get('model_state_dict', resume_ckpt)
        else:
            model_state = resume_ckpt

        if isinstance(model_state, dict):
            model_state = {k.replace('module.', ''): v for k, v in model_state.items()}

        try:
            model.load_state_dict(model_state, strict=True)
            print("Loaded resume model weights with strict=True")
        except RuntimeError as e:
            print(f"Resume strict load failed: {e}")
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            print(f"Loaded resume model with strict=False (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")

        if isinstance(resume_ckpt, dict):
            if 'optimizer_state_dict' in resume_ckpt:
                try:
                    optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
                    print("Loaded optimizer state from resume checkpoint")
                except Exception as e:
                    print(f"[WARN] Failed to load optimizer state: {e}")
            if scheduler is not None and 'scheduler_state_dict' in resume_ckpt:
                try:
                    scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
                    print("Loaded scheduler state from resume checkpoint")
                except Exception as e:
                    print(f"[WARN] Failed to load scheduler state: {e}")

            ckpt_epoch = int(resume_ckpt.get('epoch', 0) or 0)
            if history:
                start_epoch = max(start_epoch, ckpt_epoch + 1)
            else:
                start_epoch = ckpt_epoch + 1

    if not args.disable_teacher_ema:
        ema_model = copy.deepcopy(model)
        ema_model.to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)

        if args.resume_checkpoint and 'resume_ckpt' in locals():
            ema_state = resume_ckpt.get('model_state_dict_ema') if isinstance(resume_ckpt, dict) else None
            if ema_state:
                ema_state = {k.replace('module.', ''): v for k, v in ema_state.items()}
                try:
                    ema_model.load_state_dict(ema_state, strict=True)
                    print("Loaded EMA teacher weights from resume checkpoint")
                except RuntimeError as e:
                    print(f"[WARN] EMA strict load failed: {e}")
                    missing_keys, unexpected_keys = ema_model.load_state_dict(ema_state, strict=False)
                    print(f"Loaded EMA teacher with strict=False (missing={len(missing_keys)}, unexpected={len(unexpected_keys)})")

    use_ema_for_eval = (ema_model is not None) and (not args.disable_eval_with_ema)
    best_metric = compute_best_metric_from_history(history, use_val=(val_loader is not None))

    train_stats = {
        'loss': None,
        'sup_loss': None,
        'pseudo_loss': None,
        'consistency_loss': None,
        'labeled_count': 0,
        'pseudo_count': 0,
    }
    val_stats = None
    current_metric = best_metric

    if start_epoch > args.epochs:
        print(f"No training needed: already at epoch {start_epoch - 1}, target epochs={args.epochs}")

    is_tty = sys.stdout.isatty()
    epoch_bar = tqdm(
        range(start_epoch, args.epochs + 1),
        desc='Epoch Progress',
        ascii=True,
        dynamic_ncols=is_tty,
        ncols=(None if is_tty else 120),
        mininterval=0.5,
        file=sys.stdout,
        leave=True,
    )

    for epoch in epoch_bar:
        epoch_bar.set_postfix_str(f"epoch={epoch}/{args.epochs}")
        pseudo_weight = compute_pseudo_loss_weight(
            epoch, args.warmup_epochs, args.pseudo_loss_weight
        )

        train_loader, dataset_stats = build_train_loader_with_curriculum(args, epoch=epoch)
        print(
            f"\nEpoch {epoch}/{args.epochs} dataset: "
            f"labeled={dataset_stats['n_labeled']}, pseudo={dataset_stats['n_pseudo']}, "
            f"curr_face={dataset_stats.get('curriculum_min_face_size')}, "
            f"curr_blur={dataset_stats.get('curriculum_min_blur_score')}, "
            f"curr_model_cons={dataset_stats.get('curriculum_max_model_consistency_deg')}"
        )

        print(f"\nEpoch {epoch}/{args.epochs} - Pseudo loss weight: {pseudo_weight:.3f}")

        train_stats = train_one_epoch(
            model, ema_model, train_loader, criterion_sup, criterion_pseudo, optimizer, device,
            epoch, args, pseudo_weight
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Total Loss: {train_stats['loss']:.4f}")
        print(f"  Supervised Loss: {train_stats['sup_loss']:.4f}")
        print(f"  Pseudo Loss: {train_stats['pseudo_loss']:.4f}")
        print(f"  Classroom Consistency Loss: {train_stats['consistency_loss']:.4f}")
        print(f"  Labeled samples: {train_stats['labeled_count']}")
        print(f"  Pseudo samples: {train_stats['pseudo_count']}")
        print(f"  Learning rate: {current_lr:.6f}")

        current_metric = train_stats['loss']

        if val_loader is not None:
            val_model = ema_model if use_ema_for_eval else model
            val_source = 'ema' if use_ema_for_eval else 'student'
            val_stats = validate_one_epoch(val_model, val_loader, device)
            current_metric = val_stats['avg_mae']
            print(f"  Validation MAE ({val_source}): "
                  f"yaw={val_stats['yaw_mae']:.4f}, "
                  f"pitch={val_stats['pitch_mae']:.4f}, "
                  f"roll={val_stats['roll_mae']:.4f}, "
                  f"avg={val_stats['avg_mae']:.4f}")

        # best model 按 val_avg_mae 保存；如果没有 val，则退化为 train loss
        is_best = current_metric < best_metric
        if args.save_best_model and is_best:
            best_metric = current_metric
            torch.save(
                build_checkpoint_payload(
                    model, ema_model, optimizer, scheduler, train_stats, val_stats,
                    epoch, current_metric, args, best_metric=best_metric, export_source='student'
                ),
                os.path.join(save_dir, 'best_model.pth')
            )
            if use_ema_for_eval:
                torch.save(
                    build_checkpoint_payload(
                        model, ema_model, optimizer, scheduler, train_stats, val_stats,
                        epoch, current_metric, args, best_metric=best_metric, export_source='ema'
                    ),
                    os.path.join(save_dir, 'best_model_ema.pth')
                )
            print(f"  ✓ Saved best model ({args.selection_metric}: {current_metric:.4f})")

        if epoch % args.save_freq == 0:
            torch.save(
                build_checkpoint_payload(
                    model, ema_model, optimizer, scheduler, train_stats, val_stats,
                    epoch, current_metric, args, export_source='student'
                ),
                os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            if use_ema_for_eval:
                torch.save(
                    build_checkpoint_payload(
                        model, ema_model, optimizer, scheduler, train_stats, val_stats,
                        epoch, current_metric, args, export_source='ema'
                    ),
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}_ema.pth')
                )
            print(f"  ✓ Saved checkpoint (epoch {epoch})")

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'train_sup_loss': train_stats['sup_loss'],
            'train_pseudo_loss': train_stats['pseudo_loss'],
            'train_consistency_loss': train_stats['consistency_loss'],
            'labeled_count': train_stats['labeled_count'],
            'pseudo_count': train_stats['pseudo_count'],
            'dataset_n_labeled': dataset_stats['n_labeled'],
            'dataset_n_pseudo': dataset_stats['n_pseudo'],
            'curriculum_min_face_size': dataset_stats.get('curriculum_min_face_size'),
            'curriculum_min_blur_score': dataset_stats.get('curriculum_min_blur_score'),
            'curriculum_max_model_consistency_deg': dataset_stats.get('curriculum_max_model_consistency_deg'),
            'lr': current_lr,
            'pseudo_weight': pseudo_weight,
        }
        if val_stats is not None:
            epoch_record.update({
                'val_yaw_mae': val_stats['yaw_mae'],
                'val_pitch_mae': val_stats['pitch_mae'],
                'val_roll_mae': val_stats['roll_mae'],
                'val_avg_mae': val_stats['avg_mae'],
                'val_source': ('ema' if use_ema_for_eval else 'student'),
            })
        history.append(epoch_record)

        with open(os.path.join(save_dir, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    torch.save(
        build_checkpoint_payload(
            model, ema_model, optimizer, scheduler, train_stats, val_stats,
            args.epochs, current_metric, args, export_source='student'
        ),
        os.path.join(save_dir, 'final_model.pth')
    )
    if use_ema_for_eval:
        torch.save(
            build_checkpoint_payload(
                model, ema_model, optimizer, scheduler, train_stats, val_stats,
                args.epochs, current_metric, args, export_source='ema'
            ),
            os.path.join(save_dir, 'final_model_ema.pth')
        )

    print(f"\n{'=' * 60}")
    print("Training completed!")
    if val_loader is not None:
        print(f"Best {args.selection_metric}: {best_metric:.4f}")
    else:
        print(f"Best fallback metric(train_loss): {best_metric:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
