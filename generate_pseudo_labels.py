"""
Generate pseudo labels for unlabeled classroom data.

Supports:
1. Single-model pseudo labels
2. Multi-model consensus pseudo labels
3. Optional face-crop export for downstream semi-supervised training
"""

import argparse
import glob
import os

import torch
from torchvision import transforms

try:
    from sixdrepnet.model import (
        SixDRepNet,
        SixDRepNet_o,
        SixDRepNet_StrongHead,
        SixDRepNet_EffNetV2,
        SixDRepNet_EffNetV2_Advanced,
    )
    from sixdrepnet.pseudo_label_generator import PseudoLabelGenerator
    from sixdrepnet import utils as sixd_utils
except Exception:
    from model import (
        SixDRepNet,
        SixDRepNet_o,
        SixDRepNet_StrongHead,
        SixDRepNet_EffNetV2,
        SixDRepNet_EffNetV2_Advanced,
    )
    from pseudo_label_generator import PseudoLabelGenerator
    import utils as sixd_utils


def parse_multi_path_args(values):
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
    parser = argparse.ArgumentParser(description='Generate pseudo labels for unlabeled data')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to primary model checkpoint')
    parser.add_argument('--model_variant', type=str, default='sixdrepnet_o',
                        choices=['sixdrepnet', 'sixdrepnet_o', 'strong_head', 'effnetv2', 'effnetv2_advanced'],
                        help='Primary model variant')
    parser.add_argument('--aux_model_path', nargs='+', action='append', default=None,
                        help='Optional auxiliary model checkpoints for consensus. Can be repeated.')
    parser.add_argument('--aux_model_variant', nargs='+', action='append', default=None,
                        help='Optional auxiliary model variants. One value can be broadcast to all aux models.')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing unlabeled images')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path for pseudo labels (npz format)')
    parser.add_argument('--backbone_file', type=str,
                        default='/root/6DRepNet/sixdrepnet/RepVGG-B1g2-train.pth',
                        help='Backbone pretrained weights for RepVGG-based variants')
    parser.add_argument('--effnet_backbone', type=str, default='efficientnet_v2_l',
                        choices=['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l'],
                        help='Backbone used when variant=effnetv2')
    parser.add_argument('--disable_torchvision_pretrained', action='store_true',
                        help='Disable torchvision ImageNet pretrained init when creating EfficientNet variants')
    parser.add_argument('--use_se', action='store_true',
                        help='Enable SE blocks on supported variants')

    parser.add_argument('--yolo_version', type=str, default='yolov8', choices=['yolov8', 'yolov11'],
                        help='YOLO version for auto face detector weights selection')
    parser.add_argument('--yolo_model', type=str, default='',
                        help='YOLO face detection model path (overrides auto selection)')
    parser.add_argument('--use_gfpgan', action='store_true',
                        help='Use GFPGAN for face enhancement')
    parser.add_argument('--gfpgan_model', type=str, default=None,
                        help='GFPGAN model path (auto-detect if omitted)')
    parser.add_argument('--face_det_threshold', type=float, default=0.2,
                        help='Face detection confidence threshold')
    parser.add_argument('--min_face_size', type=int, default=15,
                        help='Minimum face size in pixels')
    parser.add_argument('--expand_margin', type=float, default=0.2,
                        help='Margin ratio added around detected faces')

    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Confidence threshold for pseudo labels')
    parser.add_argument('--use_tta', action='store_true',
                        help='Use test time augmentation')
    parser.add_argument('--disable_consistency_filter', action='store_true',
                        help='Disable consistency filtering')
    parser.add_argument('--consistency_threshold_deg', type=float, default=8.0,
                        help='Max intra-model TTA consistency score in degree')
    parser.add_argument('--cross_model_threshold_deg', type=float, default=6.0,
                        help='Max inter-model consensus disagreement in degree')
    parser.add_argument('--save_face_crops', action='store_true',
                        help='Save per-face crops for downstream semi-supervised training')
    parser.add_argument('--face_crops_dir', type=str, default='',
                        help='Directory for saved face crops; defaults to output_file stem + _faces')
    parser.add_argument('--face_crop_ext', type=str, default='jpg',
                        choices=['jpg', 'jpeg', 'png'],
                        help='Image format for saved face crops')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Reserved for compatibility; current generator is face-wise')
    parser.add_argument('--single_face_mode', action='store_true',
                        help='Process as single-face images (disable multi-face detection)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--file_pattern', type=str, default='**/*.jpg',
                        help='File pattern for searching images')

    return parser.parse_args()


def build_model(variant, args):
    variant = str(variant).lower()
    if variant == 'sixdrepnet_o':
        return SixDRepNet_o(
            backbone_name='RepVGG-B1g2',
            backbone_file=args.backbone_file,
            deploy=False,
            pretrained=False,
        )
    if variant == 'strong_head':
        return SixDRepNet_StrongHead(
            backbone_name='RepVGG-B1g2',
            backbone_file=args.backbone_file,
            deploy=False,
            pretrained=False,
        )
    if variant == 'sixdrepnet':
        return SixDRepNet(
            backbone_name='RepVGG-B1g2',
            backbone_file=args.backbone_file,
            deploy=False,
            pretrained=False,
            use_se=args.use_se,
        )
    if variant == 'effnetv2':
        return SixDRepNet_EffNetV2(
            backbone_name=args.effnet_backbone,
            pretrained=(not args.disable_torchvision_pretrained),
            use_se=args.use_se,
            head_dim=512,
            dropout=0.2,
        )
    if variant == 'effnetv2_advanced':
        return SixDRepNet_EffNetV2_Advanced(
            backbone_name='efficientnet_v2_s',
            pretrained=(not args.disable_torchvision_pretrained),
            use_se=True,
            head_dim=256,
            geo_dim=128,
            dropout=0.2,
        )
    raise ValueError(f'Unsupported model_variant: {variant}')


def load_checkpoint_into_model(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get('model_state_dict_ema')
            or checkpoint.get('model_state_dict_student')
            or checkpoint.get('model_state_dict')
            or checkpoint.get('state_dict')
            or checkpoint.get('model')
            or checkpoint
        )
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
        strict_mode = True
    except RuntimeError as exc:
        print(f"[WARN] strict load failed for {ckpt_path}: {exc}")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(
            f"[WARN] loaded with strict=False for {ckpt_path} "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )
        strict_mode = False

    model.to(device)
    model.eval()
    return strict_mode


def auto_detect_gfpgan_path():
    for path in [
        '/root/miniconda3/envs/hopenet/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth',
        '/root/autodl-tmp/models/GFPGANv1.3.pth',
        '/root/6DRepNet/gfpgan/weights/GFPGANv1.4.pth',
        '/root/6DRepNet/gfpgan/weights/GFPGANv1.3.pth',
    ]:
        if os.path.exists(path):
            return path
    return None


def validate_aux_variants(aux_paths, aux_variants):
    if not aux_paths:
        return []

    if not aux_variants:
        aux_variants = ['sixdrepnet_o']

    if len(aux_variants) == 1 and len(aux_paths) > 1:
        aux_variants = aux_variants * len(aux_paths)
    elif len(aux_variants) != len(aux_paths):
        raise ValueError(
            f'aux_model_variant count ({len(aux_variants)}) does not match '
            f'aux_model_path count ({len(aux_paths)})'
        )
    return aux_variants


def main():
    args = parse_args()
    args.yolo_model = sixd_utils.resolve_yolo_weights(args.yolo_model, args.yolo_version, prefer_size='x')
    args.aux_model_path = parse_multi_path_args(args.aux_model_path)
    args.aux_model_variant = parse_multi_path_args(args.aux_model_variant)
    args.aux_model_variant = validate_aux_variants(args.aux_model_path, args.aux_model_variant)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading primary model...")
    primary_model = build_model(args.model_variant, args)
    load_checkpoint_into_model(primary_model, args.model_path, device)
    models = [primary_model]
    model_names = [f"primary:{args.model_variant}"]
    print(f"Primary model loaded from {args.model_path}")

    for aux_path, aux_variant in zip(args.aux_model_path, args.aux_model_variant):
        print(f"\nLoading auxiliary model: variant={aux_variant}, ckpt={aux_path}")
        aux_model = build_model(aux_variant, args)
        load_checkpoint_into_model(aux_model, aux_path, device)
        models.append(aux_model)
        model_names.append(f"aux:{aux_variant}")

    gfpgan_path = args.gfpgan_model
    if args.use_gfpgan and not gfpgan_path:
        gfpgan_path = auto_detect_gfpgan_path()
        if gfpgan_path:
            print(f"Auto-detected GFPGAN model: {gfpgan_path}")
        else:
            print("Warning: GFPGAN enabled but no model found")

    yolo_path = args.yolo_model if not args.single_face_mode and os.path.exists(args.yolo_model) else None
    face_crops_dir = args.face_crops_dir
    if args.save_face_crops and not face_crops_dir:
        output_stem = os.path.splitext(args.output_file)[0]
        face_crops_dir = output_stem + '_face_crops'

    generator = PseudoLabelGenerator(
        model=primary_model,
        models=models,
        model_names=model_names,
        device=device,
        confidence_threshold=args.confidence_threshold,
        use_tta=args.use_tta,
        yolo_model_path=yolo_path,
        use_gfpgan=args.use_gfpgan,
        gfpgan_model_path=gfpgan_path,
        face_det_threshold=args.face_det_threshold,
        min_face_size=args.min_face_size,
        expand_margin=args.expand_margin,
        use_consistency_filter=(not args.disable_consistency_filter),
        consistency_threshold_deg=args.consistency_threshold_deg,
        cross_model_threshold_deg=args.cross_model_threshold_deg,
        save_face_crops=args.save_face_crops,
        face_crops_dir=face_crops_dir,
        face_crop_ext=args.face_crop_ext,
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print(f"\nSearching for images in {args.data_dir}...")
    image_paths = glob.glob(os.path.join(args.data_dir, args.file_pattern), recursive=True)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in valid_extensions]
    print(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("No images found! Please check your data_dir and file_pattern.")
        return

    print("\nGenerating pseudo labels...")
    pseudo_labels = generator.generate_for_dataset(
        image_paths=image_paths,
        transform=transform,
        output_file=args.output_file,
        batch_size=args.batch_size,
        multi_face_mode=(not args.single_face_mode),
    )

    print(f"\nDone! Pseudo labels saved to {args.output_file}")
    print(f"Consensus ensemble size: {len(models)}")
    print(f"Consensus model names: {model_names}")

    total_faces = 0
    usable_faces = 0
    for img_data in pseudo_labels.values():
        if 'faces' in img_data:
            for face in img_data['faces']:
                total_faces += 1
                if face.get('use_for_training', False):
                    usable_faces += 1
        else:
            total_faces += 1
            if img_data.get('use_for_training', False):
                usable_faces += 1

    print("\nStatistics:")
    print(f"  Total frames: {len(pseudo_labels)}")
    print(f"  Total faces: {total_faces}")
    print(f"  Usable faces: {usable_faces}")
    if total_faces > 0:
        print(f"  Usable ratio: {100.0 * usable_faces / total_faces:.1f}%")


if __name__ == '__main__':
    main()
