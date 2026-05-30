import os
import argparse
import json
import importlib.util
import random
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_QES6D_DIR = os.path.join(_PROJECT_DIR, 'QES6D')

if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if _QES6D_DIR not in sys.path:
    sys.path.insert(0, _QES6D_DIR)


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODEL_MODULE = _load_module_from_path(
    'QES6D_local_model',
    os.path.join(_PROJECT_DIR, 'QES6D', 'model.py')
)
_UTILS_MODULE = _load_module_from_path(
    'QES6D_local_utils',
    os.path.join(_PROJECT_DIR, 'QES6D', 'utils.py')
)

QES6D = _MODEL_MODULE.QES6D
QES6D_o = _MODEL_MODULE.QES6D_o
QES6D_StrongHead = _MODEL_MODULE.QES6D_StrongHead
QES6D_EffNetV2 = _MODEL_MODULE.QES6D_EffNetV2
QES6D_EffNetV2_Advanced = getattr(_MODEL_MODULE, 'QES6D_EffNetV2_Advanced', None)
QES6D_ConvNeXtV2 = getattr(_MODEL_MODULE, 'QES6D_ConvNeXtV2', None)
utils = _UTILS_MODULE


try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from gfpgan import GFPGANer
except Exception:
    GFPGANer = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_local_yolo_weights(yolo_weights, yolo_version='yolov8', yolo_size='x', strict_version=False):
    base_dir = os.path.join(_PROJECT_DIR, 'facedat')
    user_path = str(yolo_weights or '').strip()
    if user_path:
        user_path = os.path.expanduser(user_path)
        if os.path.isabs(user_path):
            return user_path

        preferred_candidates = [
            os.path.join(base_dir, user_path),
            os.path.join(base_dir, os.path.basename(user_path)),
            user_path,
        ]
        for candidate in preferred_candidates:
            if os.path.exists(candidate):
                if os.path.abspath(candidate) != os.path.abspath(user_path):
                    print(f"[YOLO] remap relative weights '{user_path}' -> '{candidate}'")
                return candidate

        if strict_version:
            raise FileNotFoundError(
                f"[YOLO] strict mode: user-provided weights not found: {user_path}. "
                f"tried={preferred_candidates}"
            )
        return user_path

    version = str(yolo_version or 'yolov8').lower().strip()
    if version in ('8', 'v8', 'yolo8'):
        version = 'yolov8'
    elif version in ('11', 'v11', 'yolo11'):
        version = 'yolov11'

    size = str(yolo_size or 'x').lower().strip()
    if size not in ('n', 'x'):
        size = 'x'

    local_map = {
        ('yolov8', 'n'): os.path.join(base_dir, 'yolov8n-face.pt'),
        ('yolov8', 'x'): os.path.join(base_dir, 'yolov8x-face.pt'),
        ('yolov11', 'n'): os.path.join(base_dir, 'yolo11n.pt'),
        ('yolov11', 'x'): os.path.join(base_dir, 'yolo11x.pt'),
    }

    primary = local_map[(version, size)]
    if os.path.exists(primary):
        return primary

    if strict_version:
        raise FileNotFoundError(
            f"[YOLO] strict mode: local weights not found for version={version}, size={size}. expected={primary}"
        )

    fallback_order = [
        local_map[(version, 'n' if size == 'x' else 'x')],
        local_map[('yolov8', size)],
        local_map[('yolov8', 'n' if size == 'x' else 'x')],
        local_map[('yolov11', size)],
        local_map[('yolov11', 'n' if size == 'x' else 'x')],
    ]
    for candidate in fallback_order:
        if os.path.exists(candidate):
            print(f"[YOLO] local fallback: requested={primary}, using={candidate}")
            return candidate

    raise FileNotFoundError(
        f"[YOLO] no local weights found under {base_dir}. "
        f"Tried primary={primary} and fallbacks={fallback_order}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="No-GT evaluation for classroom head pose model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--model_variant", type=str, default="auto",
                        choices=['auto', 'QES6D', 'QES6D_o', 'strong_head', 'effnetv2', 'convnextv2', 'effnetv2_advanced'],
                        help="Model variant for loading checkpoint; auto reads from checkpoint args when possible")
    parser.add_argument("--model_use_se", type=str, default="auto", choices=["auto", "on", "off"],
                        help="SE flag for QES6D loading; auto reads checkpoint args.use_se when available")
    parser.add_argument("--effnet_head_style", type=str, default="auto", choices=["auto", "baseline", "bn_relu6"],
                        help="Head style for effnetv2; auto reads checkpoint args.effnet_head_style when available")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of classroom frames")
    parser.add_argument("--output_json", type=str, default="", help="Path to save report json")
    parser.add_argument("--file_pattern", type=str, default=".jpg,.jpeg,.png,.bmp", help="Allowed extensions")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max images to evaluate (-1 for all)")
    parser.add_argument("--input_list", type=str, default="", help="Optional image list for no-GT augmentation consistency evaluation. ")

    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible eval")

    parser.add_argument("--use_yolo", action="store_true")
    parser.add_argument("--yolo_version", type=str, default="yolov8", choices=["yolov8", "yolov11"],
                        help="YOLO version for auto face detector weights selection")
    parser.add_argument("--yolo_size", type=str, default="x", choices=["n", "x"],
                        help="Preferred YOLO model size when auto-selecting weights")
    parser.add_argument("--strict_yolo_version", action="store_true",
                        help="Strict YOLO version/size: no fallback to other local weights")
    parser.add_argument("--yolo_weights", type=str, default="",
                        help="YOLO weights path (overrides --yolo_version auto selection)")
    parser.add_argument("--face_conf", type=float, default=0.25)
    parser.add_argument("--expand", type=float, default=1.25)

    parser.add_argument("--use_gfpgan", action="store_true")
    parser.add_argument("--gfpgan_model", type=str, default="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth")

    parser.add_argument("--reference_model", type=str, default="", help="Optional reference model for disagreement metric")

    parser.add_argument("--eval_with_labels", action="store_true", help="Force evaluate with GT labels using local datasets.py")
    parser.add_argument("--labeled_use_face_pipeline", action="store_true",
                        help="Apply YOLO/GFPGAN preprocessing in labeled eval (off by default to keep benchmark protocol)")
    parser.add_argument("--dataset", type=str, default="auto", help="Dataset name for labeled eval, e.g. AFLW2000/BIWI/auto")
    parser.add_argument("--filename_list", type=str, default="", help="Filename list for labeled eval (txt/npz depending on dataset)")
    parser.add_argument("--acc_threshold", type=float, default=10.0, help="Accuracy threshold in degree for labeled eval")
    parser.add_argument("--labeled_batch_size", type=int, default=40, help="Batch size for labeled evaluation")
    parser.add_argument("--labeled_num_workers", type=int, default=2, help="Workers for labeled evaluation")
    parser.add_argument("--labeled_topk", type=int, default=10, help="How many worst labeled samples to report")
    parser.add_argument("--labeled_thresholds", type=str, default="5,10,15,20", help="Comma-separated degree thresholds for labeled accuracy summary")
    parser.add_argument("--save_all_labeled_errors", action="store_true", help="Save per-sample labeled errors for significance tests")
    parser.add_argument(
        "--perturb_type",
        type=str,
        default="none",
        choices=["none", "blur", "noise", "jpeg", "occlusion", "lowlight"],
        help="Apply one perturbation type before prediction"
    )
    parser.add_argument("--perturb_severity", type=int, default=0, help="Perturbation severity level (0 disables, suggested 1-4)")

    parser.add_argument("--eval_all_faces", action="store_true", help="When YOLO is enabled, evaluate all detected faces per frame")
    parser.add_argument("--track_npz", type=str, default="", help="Track npz produced by pseudo_label_generator_plus (*_tracks.npz)")

    return parser.parse_args()


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clamp_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_perturb_bgr(img_bgr, perturb_type, severity):
    mode = str(perturb_type).lower()
    sev = int(max(0, severity))
    if mode == 'none' or sev <= 0:
        return img_bgr

    img = img_bgr
    if mode == 'blur':
        kernel_map = {1: 3, 2: 5, 3: 7, 4: 9}
        k = kernel_map.get(sev, 11)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)

    if mode == 'noise':
        sigma_map = {1: 4.0, 2: 8.0, 3: 12.0, 4: 16.0}
        sigma = sigma_map.get(sev, 20.0)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        return _clamp_uint8(img.astype(np.float32) + noise)

    if mode == 'jpeg':
        q_map = {1: 70, 2: 50, 3: 30, 4: 15}
        q = int(q_map.get(sev, 10))
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return img
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec if dec is not None else img

    if mode == 'occlusion':
        ratio_map = {1: 0.12, 2: 0.2, 3: 0.28, 4: 0.36}
        ratio = float(ratio_map.get(sev, 0.42))
        h, w = img.shape[:2]
        side = int(max(1, min(h, w) * ratio))
        if side <= 0:
            return img
        x0 = np.random.randint(0, max(1, w - side + 1))
        y0 = np.random.randint(0, max(1, h - side + 1))
        out = img.copy()
        out[y0:y0 + side, x0:x0 + side] = 0
        return out

    if mode == 'lowlight':
        alpha_map = {1: 0.85, 2: 0.7, 3: 0.55, 4: 0.4}
        alpha = float(alpha_map.get(sev, 0.3))
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    return img


def perturb_normalized_images(images, perturb_type, severity):
    mode = str(perturb_type).lower()
    sev = int(max(0, severity))
    if mode == 'none' or sev <= 0:
        return images

    if images.ndim != 4 or images.shape[1] != 3:
        return images

    mean = torch.as_tensor(IMAGENET_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)

    denorm = (images * std + mean).clamp(0, 1)
    out_tensors = []
    for i in range(denorm.shape[0]):
        rgb = (denorm[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_aug = apply_perturb_bgr(bgr, mode, sev)
        rgb_aug = cv2.cvtColor(bgr_aug, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb_aug.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = (t - torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.as_tensor(IMAGENET_STD).view(3, 1, 1)
        out_tensors.append(t)

    stacked = torch.stack(out_tensors, dim=0).to(device=images.device, dtype=images.dtype)
    return stacked


def _is_deploy_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    for key in state_dict.keys():
        if 'rbr_reparam' in key:
            return True
    return False


def _use_test_plus2_model(snapshot_path):
    target = 'QES6D_300W_LP_AFLW2000.pth'
    return os.path.basename(os.path.abspath(snapshot_path)) == target


def _infer_model_variant_from_checkpoint(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict):
            mv = cfg.get('model_variant', None)
            if isinstance(mv, str) and mv.strip():
                return mv.strip().lower()
    if _use_test_plus2_model(snapshot_path):
        return 'QES6D_O'
    return 'QES6D'


def _infer_use_se_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict):
            return bool(cfg.get('use_se', False))
    return False



def _load_sidecar_config(snapshot_path):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(snapshot_path)), 'config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _infer_model_variant_from_sources(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict):
            mv = cfg.get('model_variant', None)
            if isinstance(mv, str) and mv.strip():
                return mv.strip().lower()
    cfg = _load_sidecar_config(snapshot_path)
    if isinstance(cfg, dict):
        mv = cfg.get('model_variant', None)
        if isinstance(mv, str) and mv.strip():
            return mv.strip().lower()
    if _use_test_plus2_model(snapshot_path):
        return 'QES6D_O'
    return 'QES6D'


def _infer_use_se_from_sources(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict):
            return bool(cfg.get('use_se', False))
    cfg = _load_sidecar_config(snapshot_path)
    if isinstance(cfg, dict):
        return bool(cfg.get('use_se', False))
    return False


def _infer_effnet_backbone_from_sources(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict) and cfg.get('effnet_backbone'):
            return cfg['effnet_backbone']
    cfg = _load_sidecar_config(snapshot_path)
    if isinstance(cfg, dict) and cfg.get('effnet_backbone'):
        return cfg['effnet_backbone']
    return 'efficientnet_v2_l'


def _infer_effnet_head_style_from_sources(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict) and cfg.get('effnet_head_style'):
            return str(cfg['effnet_head_style']).lower()
    cfg = _load_sidecar_config(snapshot_path)
    if isinstance(cfg, dict) and cfg.get('effnet_head_style'):
        return str(cfg['effnet_head_style']).lower()
    return 'baseline'


def _infer_convnext_backbone_from_sources(checkpoint, snapshot_path):
    if isinstance(checkpoint, dict):
        cfg = checkpoint.get('args', None)
        if isinstance(cfg, dict) and cfg.get('convnext_backbone'):
            return cfg['convnext_backbone']
    cfg = _load_sidecar_config(snapshot_path)
    if isinstance(cfg, dict) and cfg.get('convnext_backbone'):
        return cfg['convnext_backbone']
    return 'convnextv2_tiny'


def load_model(snapshot_path, device, model_variant='auto', model_use_se='auto', effnet_head_style='auto'):
    checkpoint = torch.load(snapshot_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    if model_variant == 'auto':
        model_variant = _infer_model_variant_from_sources(checkpoint, snapshot_path)
    model_variant = str(model_variant).strip().upper()

    if str(model_use_se).lower() == 'auto':
        use_se = _infer_use_se_from_sources(checkpoint, snapshot_path)
    else:
        use_se = (str(model_use_se).lower() == 'on')

    if str(effnet_head_style).lower() == 'auto':
        resolved_effnet_head_style = _infer_effnet_head_style_from_sources(checkpoint, snapshot_path)
    else:
        resolved_effnet_head_style = str(effnet_head_style).lower()

    if model_variant == 'QES6D_O':
        deploy_mode = _is_deploy_state_dict(state_dict)
        model = QES6D_o(
            backbone_name='RepVGG-B1g2',
            backbone_file='',
            deploy=deploy_mode,
            pretrained=False
        ).to(device)
    elif model_variant == 'STRONG_HEAD':
        model = QES6D_StrongHead(
            backbone_name='RepVGG-B1g2',
            backbone_file='/root/autodl-tmp/QES6D/QES6D/RepVGG-B1g2-train.pth',
            deploy=False,
            pretrained=True
        ).to(device)
    elif model_variant == 'EFFNETV2':
        model = QES6D_EffNetV2(
            backbone_name=_infer_effnet_backbone_from_sources(checkpoint, snapshot_path),
            pretrained=True,
            use_se=use_se,
            head_dim=512,
            dropout=0.2,
            head_style=resolved_effnet_head_style,
        ).to(device)
    elif model_variant == 'CONVNEXTV2':
        if QES6D_ConvNeXtV2 is None:
            raise ImportError('QES6D_ConvNeXtV2 is not available in local QES6D/model.py')
        model = QES6D_ConvNeXtV2(
            backbone_name=_infer_convnext_backbone_from_sources(checkpoint, snapshot_path),
            pretrained=True,
            use_cbam=True,
            use_coordconv=True,
            head_dim=512,
            dropout=0.2,
        ).to(device)
    elif model_variant == 'EFFNETV2_ADVANCED':
        if QES6D_EffNetV2_Advanced is None:
            raise ImportError('QES6D_EffNetV2_Advanced is not available in local QES6D/model.py')
        model = QES6D_EffNetV2_Advanced(
            pretrained=True,
            use_coordconv=True,
            use_transformer=True,
            head_dim=512,
            dropout=0.2,
        ).to(device)
    elif model_variant == 'QES6D':
        model = QES6D(
            backbone_name='RepVGG-B1g2',
            backbone_file='/root/autodl-tmp/QES6D/QES6D/RepVGG-B1g2-train.pth',
            deploy=False,
            pretrained=True,
            use_se=use_se,
        ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    
    # 关键修改 2：显式迁移到 device 并返回
    model.to(device) 
    model.eval()
    return model


def list_images(data_dir, exts, max_samples):
    files = []
    exts = {e.strip().lower() for e in exts.split(',')}
    for root, _, names in os.walk(data_dir):
        for name in names:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                files.append(os.path.join(root, name))

    files.sort()
    if max_samples is not None and max_samples > 0:
        files = files[:max_samples]
    return files

def list_images_from_list(data_dir, list_path, max_samples):
    with open(list_path, "r", encoding="utf-8") as f:
        items = [x.strip() for x in f if x.strip()]
    out = []
    for item in items:
        if os.path.isabs(item):
            p = item
        else:
            p = os.path.join(data_dir, item)
        p = os.path.normpath(p)
        # 兼容 AFLW2000 这种 list 里只给 stem 的情况
        if not os.path.exists(p):
            stem = item
            if os.path.isabs(item):
                stem = os.path.splitext(item)[0]
            else:
                stem = os.path.splitext(os.path.join(data_dir, item))[0]
            candidates = [
                stem + ".jpg",
                stem + ".png",
                stem + ".jpeg",
                stem + ".JPG",
                stem + ".PNG",
                stem + ".JPEG",
            ]
            p = next((c for c in candidates if os.path.exists(c)), "")
        if p and os.path.exists(p):
            out.append(p)
    # 保持 list 顺序，同时去重
    uniq = []
    used = set()
    for p in out:
        if p in used:
            continue
        used.add(p)
        uniq.append(p)
    if max_samples is not None and max_samples > 0:
        uniq = uniq[:max_samples]
    return uniq

# def yolo_crop_face(yolo_model, img_bgr, conf=0.25, expand=1.2):
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     results = yolo_model(img_rgb, conf=conf, verbose=False)

#     best_box = None
#     max_area = 0.0
#     if results and results[0].boxes is not None:
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
#             area = float((x2 - x1) * (y2 - y1))
#             if area > max_area:
#                 max_area = area
#                 best_box = (x1, y1, x2, y2)

#     if best_box is None:
#         return None

#     x1, y1, x2, y2 = best_box
#     cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
#     side = max(x2 - x1, y2 - y1) * float(expand)

#     h, w = img_bgr.shape[:2]
#     x_min = int(max(0, cx - side / 2.0))
#     y_min = int(max(0, cy - side / 2.0))
#     x_max = int(min(w, cx + side / 2.0))
#     y_max = int(min(h, cy + side / 2.0))

#     crop = img_bgr[y_min:y_max, x_min:x_max]
#     if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
#         return None
#     return crop


def yolo_crop_face(yolo_model, img_bgr, conf=0.25, expand=1.25):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, conf=conf, verbose=False)

    best_box = None
    max_area = 0.0
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            area = float((x2 - x1) * (y2 - y1))
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # 核心对齐：使用 test_plus2 的正方形边长计算方式
    side = max(x2 - x1, y2 - y1) * float(expand)

    h, w = img_bgr.shape[:2]
    x_min = int(max(0, cx - side / 2.0))
    y_min = int(max(0, cy - side / 2.0))
    x_max = int(min(w, cx + side / 2.0))
    y_max = int(min(h, cy + side / 2.0))

    crop = img_bgr[y_min:y_max, x_min:x_max]
    if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
        return None
    return crop

def yolo_crop_faces(yolo_model, img_bgr, conf=0.25, expand=1.2):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb, conf=conf, verbose=False)

    crops = []
    h, w = img_bgr.shape[:2]
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            side = max(x2 - x1, y2 - y1) * float(expand)

            x_min = int(max(0, cx - side / 2.0))
            y_min = int(max(0, cy - side / 2.0))
            x_max = int(min(w, cx + side / 2.0))
            y_max = int(min(h, cy + side / 2.0))
            crop = img_bgr[y_min:y_max, x_min:x_max]
            if crop is not None and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                crops.append(crop)

    return crops


def gfpgan_enhance(face_enhancer, img_bgr):
    try:
        _, _, restored = face_enhancer.enhance(img_bgr, has_aligned=False)
        if restored is not None:
            return restored
    except Exception:
        pass
    return img_bgr


def preprocess_bgr(img_bgr, tfm, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return tfm(img_pil).unsqueeze(0).to(device)


def compute_confidence(R_pred):
    RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)
    I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype).unsqueeze(0)
    ortho_error = ((RtR - I) ** 2).sum(dim=(1, 2))
    conf = torch.exp(-ortho_error * 10)
    return conf


def wrap_abs_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def predict_deg(model, x):
    with torch.no_grad():
        R = model(x)
        euler = utils.compute_euler_angles_from_rotation_matrices(R) * 180.0 / np.pi
    pitch = float(euler[0, 0].item())
    yaw = float(euler[0, 1].item())
    roll = float(euler[0, 2].item())
    conf = float(compute_confidence(R)[0].item())
    return np.array([yaw, pitch, roll], dtype=np.float32), conf


def load_local_datasets_module():
    candidates = [
        os.path.join(_PROJECT_DIR, 'QES6D', 'datasets.py'),
        os.path.join(_PROJECT_DIR, 'datasets.py'),
    ]
    datasets_py = next((p for p in candidates if os.path.isfile(p)), None)
    if datasets_py is None:
        raise FileNotFoundError(f"datasets.py not found. Tried: {candidates}")

    spec = importlib.util.spec_from_file_location('QES6D_local_datasets', datasets_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {datasets_py}")
    datasets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets_module)
    return datasets_module


def infer_dataset_name(dataset_arg, data_dir):
    if dataset_arg and dataset_arg.lower() != 'auto':
        return dataset_arg

    base = os.path.basename(os.path.normpath(data_dir)).lower()
    if base == 'aflw2000':
        return 'AFLW2000'
    if base == 'biwi':
        return 'BIWI'
    if base == 'aflw':
        return 'AFLW'
    if base == 'afw':
        return 'AFW'
    return 'CUSTOM'


def resolve_filename_list(dataset_name, data_dir, filename_list_arg):
    if filename_list_arg:
        return filename_list_arg

    dataset_name = str(dataset_name).upper()
    data_dir = os.path.abspath(data_dir)
    parent_dir = os.path.dirname(data_dir)

    if dataset_name == 'AFLW2000':
        candidates = [
            os.path.join(data_dir, 'files.txt'),
            os.path.join(parent_dir, 'AFLW2000', 'files.txt'),
        ]
    elif dataset_name == 'BIWI':
        candidates = [
            os.path.join(data_dir, 'BIWI_noTrack.npz'),
            os.path.join(parent_dir, 'BIWI_noTrack.npz'),
            os.path.join(parent_dir, 'BIWI_paths_pose.npz'),
        ]
    else:
        candidates = []

    for path in candidates:
        if os.path.exists(path):
            return path
    return ''


def parse_thresholds(text, default=None):
    default = [5.0, 10.0, 15.0, 20.0] if default is None else default
    if not text:
        return default

    values = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values if values else default


def summarize_errors(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            'mean': None,
            'median': None,
            'p90': None,
            'p95': None,
            'max': None,
        }
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
        'max': float(arr.max()),
    }


def evaluate_with_labels(args, model, device, tfm, yolo_model=None, face_enhancer=None):
    if not args.filename_list:
        raise ValueError("--filename_list is required when --eval_with_labels is set")

    datasets_module = load_local_datasets_module()
    pose_dataset = datasets_module.getDataset(
        args.dataset,
        args.data_dir,
        args.filename_list,
        tfm,
        train_mode=False
    )

    loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=args.labeled_batch_size,
        num_workers=args.labeled_num_workers,
        pin_memory=True
    )

    total = 0
    yaw_err = 0.0
    pitch_err = 0.0
    roll_err = 0.0
    yaw_ok = 0
    pitch_ok = 0
    roll_ok = 0
    thresholds = parse_thresholds(args.labeled_thresholds)
    threshold_hits = {thr: {'yaw': 0, 'pitch': 0, 'roll': 0, 'avg': 0, 'geo': 0} for thr in thresholds}

    yaw_all = []
    pitch_all = []
    roll_all = []
    avg_all = []
    geo_all = []
    sample_records = []

    with torch.no_grad():
        for images, R_gt, cont_labels, names in tqdm(loader, desc='Labeled eval'):
            images = images.to(device, non_blocking=True)

            use_face_pipeline = bool(args.labeled_use_face_pipeline) and ((yolo_model is not None) or (face_enhancer is not None))
            if use_face_pipeline:
                processed = []
                for i in range(images.shape[0]):
                    denorm = (images[i].detach().cpu().permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN)
                    rgb = np.clip(denorm * 255.0, 0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    if yolo_model is not None:
                        cropped = yolo_crop_face(yolo_model, bgr, conf=args.face_conf, expand=args.expand)
                        if cropped is not None:
                            bgr = cropped

                    if face_enhancer is not None:
                        bgr = gfpgan_enhance(face_enhancer, bgr)

                    if args.perturb_type != 'none' and int(args.perturb_severity) > 0:
                        bgr = apply_perturb_bgr(bgr, args.perturb_type, args.perturb_severity)

                    processed.append(preprocess_bgr(bgr, tfm, device).squeeze(0))

                images = torch.stack(processed, dim=0)
            elif args.perturb_type != 'none' and int(args.perturb_severity) > 0:
                images = perturb_normalized_images(images, args.perturb_type, args.perturb_severity)

            R_gt = R_gt.to(device)

            y_gt = cont_labels[:, 0].float().to(device) * 180.0 / np.pi
            p_gt = cont_labels[:, 1].float().to(device) * 180.0 / np.pi
            r_gt = cont_labels[:, 2].float().to(device) * 180.0 / np.pi

            R_pred = model(images)
            euler_deg = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180.0 / np.pi
            p_pred = euler_deg[:, 0]
            y_pred = euler_deg[:, 1]
            r_pred = euler_deg[:, 2]

            dy = torch.abs((y_pred - y_gt + 180.0) % 360.0 - 180.0)
            dp = torch.abs((p_pred - p_gt + 180.0) % 360.0 - 180.0)
            dr = torch.abs((r_pred - r_gt + 180.0) % 360.0 - 180.0)
            d_avg = (dy + dp + dr) / 3.0

            R_rel = torch.bmm(R_pred, R_gt.transpose(1, 2))
            tr = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
            geo = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)) * 180.0 / np.pi

            yaw_err += float(dy.sum().item())
            pitch_err += float(dp.sum().item())
            roll_err += float(dr.sum().item())

            yaw_ok += int((dy <= args.acc_threshold).sum().item())
            pitch_ok += int((dp <= args.acc_threshold).sum().item())
            roll_ok += int((dr <= args.acc_threshold).sum().item())

            dy_np = dy.detach().cpu().numpy()
            dp_np = dp.detach().cpu().numpy()
            dr_np = dr.detach().cpu().numpy()
            davg_np = d_avg.detach().cpu().numpy()
            geo_np = geo.detach().cpu().numpy()

            yaw_all.extend(dy_np.tolist())
            pitch_all.extend(dp_np.tolist())
            roll_all.extend(dr_np.tolist())
            avg_all.extend(davg_np.tolist())
            geo_all.extend(geo_np.tolist())

            for thr in thresholds:
                thr_val = float(thr)
                threshold_hits[thr]['yaw'] += int((dy <= thr_val).sum().item())
                threshold_hits[thr]['pitch'] += int((dp <= thr_val).sum().item())
                threshold_hits[thr]['roll'] += int((dr <= thr_val).sum().item())
                threshold_hits[thr]['avg'] += int((d_avg <= thr_val).sum().item())
                threshold_hits[thr]['geo'] += int((geo <= thr_val).sum().item())

            names_list = list(names) if isinstance(names, (list, tuple)) else [str(names)] * len(dy_np)
            for i, name in enumerate(names_list):
                sample_records.append({
                    'name': str(name),
                    'yaw_err_deg': float(dy_np[i]),
                    'pitch_err_deg': float(dp_np[i]),
                    'roll_err_deg': float(dr_np[i]),
                    'avg_mae_deg': float(davg_np[i]),
                    'geodesic_deg': float(geo_np[i]),
                })

            total += int(images.shape[0])

    if len(yaw_all) != total:
        raise RuntimeError(
            f"Labeled eval count mismatch: total={total}, yaw_all={len(yaw_all)}"
        )

    if total == 0:
        raise RuntimeError('No valid labeled samples found')

    mae_yaw = yaw_err / total
    mae_pitch = pitch_err / total
    mae_roll = roll_err / total

    acc_yaw = yaw_ok / total
    acc_pitch = pitch_ok / total
    acc_roll = roll_ok / total

    sample_records.sort(key=lambda x: (x['avg_mae_deg'], x['geodesic_deg']), reverse=True)
    topk = max(0, int(args.labeled_topk))

    accuracy_multi = {}
    for thr in thresholds:
        key = f'within_{thr:g}deg'
        accuracy_multi[key] = {
            'yaw': threshold_hits[thr]['yaw'] / total,
            'pitch': threshold_hits[thr]['pitch'] / total,
            'roll': threshold_hits[thr]['roll'] / total,
            'avg': threshold_hits[thr]['avg'] / total,
            'geodesic': threshold_hits[thr]['geo'] / total,
        }

    result = {
        'n_samples': total,
        'mae_deg': {
            'yaw': mae_yaw,
            'pitch': mae_pitch,
            'roll': mae_roll,
            'avg': (mae_yaw + mae_pitch + mae_roll) / 3.0,
            'geodesic': float(np.mean(geo_all)),
        },
        'median_deg': {
            'yaw': float(np.median(yaw_all)),
            'pitch': float(np.median(pitch_all)),
            'roll': float(np.median(roll_all)),
            'avg': float(np.median(avg_all)),
            'geodesic': float(np.median(geo_all)),
        },
        'p90_deg': {
            'yaw': float(np.percentile(yaw_all, 90)),
            'pitch': float(np.percentile(pitch_all, 90)),
            'roll': float(np.percentile(roll_all, 90)),
            'avg': float(np.percentile(avg_all, 90)),
            'geodesic': float(np.percentile(geo_all, 90)),
        },
        'p95_deg': {
            'yaw': float(np.percentile(yaw_all, 95)),
            'pitch': float(np.percentile(pitch_all, 95)),
            'roll': float(np.percentile(roll_all, 95)),
            'avg': float(np.percentile(avg_all, 95)),
            'geodesic': float(np.percentile(geo_all, 95)),
        },
        'max_deg': {
            'yaw': float(np.max(yaw_all)),
            'pitch': float(np.max(pitch_all)),
            'roll': float(np.max(roll_all)),
            'avg': float(np.max(avg_all)),
            'geodesic': float(np.max(geo_all)),
        },
        'accuracy': {
            f'within_{int(args.acc_threshold)}deg_yaw': acc_yaw,
            f'within_{int(args.acc_threshold)}deg_pitch': acc_pitch,
            f'within_{int(args.acc_threshold)}deg_roll': acc_roll,
            f'within_{int(args.acc_threshold)}deg_avg': (acc_yaw + acc_pitch + acc_roll) / 3.0,
        },
        'accuracy_multi_threshold': accuracy_multi,
        'error_summary_deg': {
            'yaw': summarize_errors(yaw_all),
            'pitch': summarize_errors(pitch_all),
            'roll': summarize_errors(roll_all),
            'avg': summarize_errors(avg_all),
            'geodesic': summarize_errors(geo_all),
        },
        'worst_samples': sample_records[:topk],
    }

    if args.save_all_labeled_errors:
        result['all_sample_errors'] = sample_records

    return result


def compute_track_temporal_from_npz(track_npz_path):
    if not os.path.exists(track_npz_path):
        raise FileNotFoundError(f"track npz not found: {track_npz_path}")

    data = np.load(track_npz_path, allow_pickle=True)
    tracks = {}

    for key in data.files:
        item = data[key].item()
        if not isinstance(item, dict):
            continue
        faces = item.get('faces', [])
        for face in faces:
            if not isinstance(face, dict):
                continue
            tid = int(face.get('track_id', -1))
            if tid < 0:
                continue
            R = face.get('rotation_matrix', None)
            if R is None:
                continue
            frame_idx = int(face.get('frame_idx', 0))
            R_t = torch.as_tensor(R, dtype=torch.float32).unsqueeze(0)
            euler_deg = utils.compute_euler_angles_from_rotation_matrices(R_t) * 180.0 / np.pi
            pitch = float(euler_deg[0, 0].item())
            yaw = float(euler_deg[0, 1].item())
            roll = float(euler_deg[0, 2].item())

            tracks.setdefault(tid, []).append((frame_idx, np.array([yaw, pitch, roll], dtype=np.float32)))

    dy_all, dp_all, dr_all = [], [], []
    used_tracks = 0
    for tid, seq in tracks.items():
        seq.sort(key=lambda x: x[0])
        if len(seq) < 2:
            continue
        used_tracks += 1
        for i in range(1, len(seq)):
            prev = seq[i - 1][1]
            cur = seq[i][1]
            d = wrap_abs_diff_deg(cur, prev)
            dy_all.append(float(d[0]))
            dp_all.append(float(d[1]))
            dr_all.append(float(d[2]))

    if len(dy_all) == 0:
        return {
            'n_tracks_total': int(len(tracks)),
            'n_tracks_used': 0,
            'n_transitions': 0,
            'yaw_mean': None,
            'pitch_mean': None,
            'roll_mean': None,
            'avg_mean': None,
            'avg_p95': None
        }

    dy = np.array(dy_all, dtype=np.float32)
    dp = np.array(dp_all, dtype=np.float32)
    dr = np.array(dr_all, dtype=np.float32)
    dmean = (dy + dp + dr) / 3.0

    return {
        'n_tracks_total': int(len(tracks)),
        'n_tracks_used': int(used_tracks),
        'n_transitions': int(len(dy_all)),
        'yaw_mean': float(dy.mean()),
        'pitch_mean': float(dp.mean()),
        'roll_mean': float(dr.mean()),
        'avg_mean': float(dmean.mean()),
        'avg_p95': float(np.percentile(dmean, 95))
    }


def make_augments(img_bgr):
    aug_list = []

    aug_list.append(cv2.convertScaleAbs(img_bgr, alpha=0.85, beta=0))
    aug_list.append(cv2.convertScaleAbs(img_bgr, alpha=1.15, beta=0))
    aug_list.append(cv2.GaussianBlur(img_bgr, (5, 5), 1.0))

    noise = np.random.normal(0, 4.0, img_bgr.shape).astype(np.float32)
    noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    aug_list.append(noisy)

    return aug_list


def main():
    args = parse_args()
    args.yolo_weights = resolve_local_yolo_weights(
        args.yolo_weights,
        args.yolo_version,
        yolo_size=args.yolo_size,
        strict_version=args.strict_yolo_version
    )
    if args.use_yolo:
        print(f"[YOLO] version={args.yolo_version}, size={args.yolo_size}, strict={args.strict_yolo_version}, resolved_weights={args.yolo_weights}")
    
    set_global_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.model_path,
        device,
        model_variant=args.model_variant,
        model_use_se=args.model_use_se,
        effnet_head_style=args.effnet_head_style,
    )
    ref_model = load_model(
        args.reference_model,
        device,
        model_variant=args.model_variant,
        model_use_se=args.model_use_se,
        effnet_head_style=args.effnet_head_style,
    ) if args.reference_model else None

    yolo_model = None
    if args.use_yolo:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed but --use_yolo is set")
        yolo_model = YOLO(args.yolo_weights)

    face_enhancer = None
    if args.use_gfpgan:
        if GFPGANer is None:
            raise RuntimeError("gfpgan is not installed but --use_gfpgan is set")
        face_enhancer = GFPGANer(
            model_path=args.gfpgan_model,
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=str(device)
        )

    # tfm = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    tfm = transforms.Compose([
        transforms.Resize(256),       # 确保是 256
        transforms.CenterCrop(224),   # 确保是 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name = infer_dataset_name(args.dataset, args.data_dir)
    resolved_filename_list = resolve_filename_list(dataset_name, args.data_dir, args.filename_list)

    public_with_labels = dataset_name in {'AFLW2000', 'BIWI'}
    do_labeled_eval = bool(args.eval_with_labels or (public_with_labels and resolved_filename_list))

    labeled_metrics = None
    if do_labeled_eval:
        if not resolved_filename_list:
            raise ValueError(
                f"No filename_list found for dataset={dataset_name}. "
                f"Please pass --filename_list explicitly."
            )
        args.dataset = dataset_name
        args.filename_list = resolved_filename_list
        labeled_metrics = evaluate_with_labels(
            args,
            model,
            device,
            tfm,
            yolo_model=yolo_model,
            face_enhancer=face_enhancer,
        )

    image_paths = list_images(args.data_dir, args.file_pattern, args.max_samples)
    no_gt_enabled = len(image_paths) > 0
    if not no_gt_enabled and labeled_metrics is None:
        raise RuntimeError(f"No images found in {args.data_dir}")

    yaw_series, pitch_series, roll_series = [], [], []
    conf_series = []
    aug_consistency = []
    ref_disagreement = []

    valid_count = 0
    skipped_no_face = 0

    if no_gt_enabled:
        for path in tqdm(image_paths, desc="No-GT eval"):
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue

            eval_images = [img_bgr]
            if yolo_model is not None:
                if args.eval_all_faces:
                    eval_images = yolo_crop_faces(yolo_model, img_bgr, conf=args.face_conf, expand=args.expand)
                else:
                    cropped = yolo_crop_face(yolo_model, img_bgr, conf=args.face_conf, expand=args.expand)
                    eval_images = [cropped] if cropped is not None else []

                if len(eval_images) == 0:
                    skipped_no_face += 1
                    continue

            for one_img in eval_images:
                if face_enhancer is not None:
                    one_img = gfpgan_enhance(face_enhancer, one_img)

                if args.perturb_type != 'none' and int(args.perturb_severity) > 0:
                    one_img = apply_perturb_bgr(one_img, args.perturb_type, args.perturb_severity)

                x = preprocess_bgr(one_img, tfm, device)
                ypr, conf = predict_deg(model, x)

                yaw_series.append(float(ypr[0]))
                pitch_series.append(float(ypr[1]))
                roll_series.append(float(ypr[2]))
                conf_series.append(conf)

                base = ypr
                aug_err = []
                for aug in make_augments(one_img):
                    xa = preprocess_bgr(aug, tfm, device)
                    ypr_a, _ = predict_deg(model, xa)
                    diff = wrap_abs_diff_deg(ypr_a, base)
                    aug_err.append(float(diff.mean()))
                aug_consistency.append(float(np.mean(aug_err)))

                if ref_model is not None:
                    ypr_ref, _ = predict_deg(ref_model, x)
                    diff_ref = wrap_abs_diff_deg(ypr_ref, ypr)
                    ref_disagreement.append(float(diff_ref.mean()))

                valid_count += 1

        if valid_count < 2:
            raise RuntimeError("Too few valid samples for no-GT metrics")

        yaw_series = np.array(yaw_series, dtype=np.float32)
        pitch_series = np.array(pitch_series, dtype=np.float32)
        roll_series = np.array(roll_series, dtype=np.float32)
        conf_series = np.array(conf_series, dtype=np.float32)
        aug_consistency = np.array(aug_consistency, dtype=np.float32)

        if args.eval_all_faces:
            dy = dp = dr = dmean = None
        else:
            dy = wrap_abs_diff_deg(yaw_series[1:], yaw_series[:-1])
            dp = wrap_abs_diff_deg(pitch_series[1:], pitch_series[:-1])
            dr = wrap_abs_diff_deg(roll_series[1:], roll_series[:-1])
            dmean = (dy + dp + dr) / 3.0
    else:
        conf_series = np.array([], dtype=np.float32)
        aug_consistency = np.array([], dtype=np.float32)
        dy = dp = dr = dmean = None

    report = {
        "meta": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "dataset": dataset_name,
            "filename_list": resolved_filename_list,
            "n_images_input": len(image_paths),
            "n_valid": int(valid_count),
            "n_skipped_no_face": int(skipped_no_face),
            "no_gt_enabled": bool(no_gt_enabled),
            "use_yolo": bool(args.use_yolo),
            "use_gfpgan": bool(args.use_gfpgan),
            "seed": int(args.seed),
            "perturb_type": args.perturb_type,
            "perturb_severity": int(args.perturb_severity),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        "confidence": {
            "mean": float(conf_series.mean()) if conf_series.size > 0 else None,
            "median": float(np.median(conf_series)) if conf_series.size > 0 else None,
            "p25": float(np.percentile(conf_series, 25)) if conf_series.size > 0 else None,
            "p75": float(np.percentile(conf_series, 75)) if conf_series.size > 0 else None,
            "ratio_ge_0_8": float((conf_series >= 0.8).mean()) if conf_series.size > 0 else None,
            "ratio_ge_0_9": float((conf_series >= 0.9).mean()) if conf_series.size > 0 else None
        },
        "temporal_smoothness_deg": {
            "yaw_mean": float(dy.mean()) if dy is not None else None,
            "pitch_mean": float(dp.mean()) if dp is not None else None,
            "roll_mean": float(dr.mean()) if dr is not None else None,
            "avg_mean": float(dmean.mean()) if dmean is not None else None,
            "avg_p95": float(np.percentile(dmean, 95)) if dmean is not None else None,
            "note": "Disabled when --eval_all_faces is on (no temporal identity)" if args.eval_all_faces else ("No image-based no-GT eval" if not no_gt_enabled else "")
        },
        "augmentation_consistency_deg": {
            "mean": float(aug_consistency.mean()) if aug_consistency.size > 0 else None,
            "median": float(np.median(aug_consistency)) if aug_consistency.size > 0 else None,
            "p95": float(np.percentile(aug_consistency, 95)) if aug_consistency.size > 0 else None
        }
    }

    if len(ref_disagreement) > 0:
        ref_disagreement = np.array(ref_disagreement, dtype=np.float32)
        report["reference_disagreement_deg"] = {
            "mean": float(ref_disagreement.mean()),
            "median": float(np.median(ref_disagreement)),
            "p95": float(np.percentile(ref_disagreement, 95))
        }

    if args.track_npz:
        report["track_temporal_deg"] = compute_track_temporal_from_npz(args.track_npz)

    if labeled_metrics is not None:
        report["labeled_metrics"] = labeled_metrics

    print("\n=== No-GT Evaluation Summary ===")
    print(f"valid: {report['meta']['n_valid']} / {report['meta']['n_images_input']}")
    if report['confidence']['mean'] is not None:
        print(f"conf mean/median: {report['confidence']['mean']:.4f} / {report['confidence']['median']:.4f}")
    else:
        print("conf: skipped (no image-based no-GT eval)")
    if report['temporal_smoothness_deg']['avg_mean'] is not None:
        print(f"temporal avg mean/p95 (deg): {report['temporal_smoothness_deg']['avg_mean']:.3f} / {report['temporal_smoothness_deg']['avg_p95']:.3f}")
    else:
        print("temporal metric: skipped (eval_all_faces=True)")
    if report['augmentation_consistency_deg']['mean'] is not None:
        print(f"aug consistency mean/p95 (deg): {report['augmentation_consistency_deg']['mean']:.3f} / {report['augmentation_consistency_deg']['p95']:.3f}")
    else:
        print("aug consistency: skipped (no image-based no-GT eval)")
    if 'reference_disagreement_deg' in report:
        print(f"ref disagreement mean/p95 (deg): {report['reference_disagreement_deg']['mean']:.3f} / {report['reference_disagreement_deg']['p95']:.3f}")
    if 'track_temporal_deg' in report:
        t = report['track_temporal_deg']
        if t['avg_mean'] is not None:
            print(f"track temporal avg mean/p95 (deg): {t['avg_mean']:.3f} / {t['avg_p95']:.3f} (tracks used: {t['n_tracks_used']})")
        else:
            print("track temporal: no valid multi-frame tracks")
    if 'labeled_metrics' in report:
        lm = report['labeled_metrics']
        print(f"labeled MAE yaw/pitch/roll/avg: {lm['mae_deg']['yaw']:.3f} / {lm['mae_deg']['pitch']:.3f} / {lm['mae_deg']['roll']:.3f} / {lm['mae_deg']['avg']:.3f}")
        print(f"labeled geodesic mean/p95 (deg): {lm['mae_deg']['geodesic']:.3f} / {lm['p95_deg']['geodesic']:.3f}")
        print(f"labeled avg median/p95/max (deg): {lm['median_deg']['avg']:.3f} / {lm['p95_deg']['avg']:.3f} / {lm['max_deg']['avg']:.3f}")

    output_json = args.output_json
    if not output_json:
        output_json = os.path.join(
            '/root/autodl-tmp/output',
            f"no_gt_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_json}")


if __name__ == "__main__":
    main()
