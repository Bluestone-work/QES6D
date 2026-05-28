"""
半监督数据集
混合有标签和伪标签数据
"""

import os
import glob
import random
import importlib.util

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter

try:
    import sixdrepnet.utils as pose_utils
except ImportError:
    import utils as pose_utils


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def normalize_path_list(values):
    """将字符串/列表/嵌套列表统一转换为路径列表。"""
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


def pair_pseudo_sources(pseudo_label_file, pseudo_data_root):
    pseudo_files = normalize_path_list(pseudo_label_file)
    pseudo_roots = normalize_path_list(pseudo_data_root)

    if not pseudo_files:
        return []
    if not pseudo_roots:
        raise ValueError('pseudo_data_root is required when pseudo_label_file is provided')

    if len(pseudo_roots) == 1 and len(pseudo_files) > 1:
        pseudo_roots = pseudo_roots * len(pseudo_files)
    elif len(pseudo_files) == 1 and len(pseudo_roots) > 1:
        pseudo_files = pseudo_files * len(pseudo_roots)
    elif len(pseudo_files) != len(pseudo_roots):
        raise ValueError(
            f'Mismatch between pseudo_label_file ({len(pseudo_files)}) '
            f'and pseudo_data_root ({len(pseudo_roots)})'
        )

    return list(zip(pseudo_files, pseudo_roots))


class FlatNPZPoseDataset(Dataset):
    def __init__(self, npz_path, filename_list=None, transform=None, train_mode=True):
        self.npz_path = npz_path
        self.transform = transform
        self.train_mode = train_mode

        pack = np.load(npz_path, allow_pickle=True)
        if 'image' not in pack or 'pose' not in pack:
            raise ValueError(f"{npz_path} must contain keys: image, pose")

        self.images = pack['image']
        self.poses = pack['pose']   # [yaw, pitch, roll] in degree

        if len(self.images) != len(self.poses):
            raise ValueError("image and pose length mismatch")

        def _variants(text):
            s = os.path.normpath(str(text).strip()).replace('\\', '/').lstrip('./')
            b = os.path.basename(s)
            stem = os.path.splitext(b)[0]
            return s.lower(), b.lower(), stem.lower()

        candidate_keys = [
            'paths', 'path', 'names', 'name', 'filenames', 'filename', 'files', 'file_list'
        ]
        sample_ids = None
        for k in candidate_keys:
            if k in pack and len(pack[k]) == len(self.images):
                sample_ids = [str(x) for x in pack[k]]
                break

        selected_indices = None
        selected_names = None

        if filename_list and os.path.exists(filename_list):
            with open(filename_list, 'r', encoding='utf-8') as f:
                requested = [x.strip() for x in f if x.strip()]

            if sample_ids is None:
                if len(requested) != len(self.images):
                    raise RuntimeError(
                        f"Strict split filtering failed for {npz_path}: filename_list has {len(requested)} items, "
                        f"but NPZ has {len(self.images)} samples and no path/name keys for safe subset matching. "
                        f"Please provide a pre-split NPZ or NPZ with keys like paths/names."
                    )
                selected_names = requested
            else:
                by_norm = {}
                by_base = {}
                by_stem = {}
                for i, sid in enumerate(sample_ids):
                    n, b, st = _variants(sid)
                    if n not in by_norm:
                        by_norm[n] = i
                    if b not in by_base:
                        by_base[b] = i
                    if st not in by_stem:
                        by_stem[st] = i

                selected_indices = []
                selected_names = []
                used = set()
                missing = []

                for item in requested:
                    n, b, st = _variants(item)
                    idx = by_norm.get(n, None)
                    if idx is None:
                        idx = by_base.get(b, None)
                    if idx is None:
                        idx = by_stem.get(st, None)

                    if idx is None:
                        missing.append(item)
                        continue

                    if idx in used:
                        continue
                    used.add(idx)
                    selected_indices.append(idx)
                    selected_names.append(item)

                if missing:
                    preview = ', '.join(missing[:5])
                    raise RuntimeError(
                        f"Strict split filtering failed for {npz_path}: {len(missing)} filename_list entries cannot be matched. "
                        f"Examples: {preview}"
                    )

                self.images = self.images[selected_indices]
                self.poses = self.poses[selected_indices]

        if selected_names is not None:
            self.names = selected_names
        elif sample_ids is not None:
            self.names = [os.path.basename(str(x)) for x in sample_ids]
        else:
            self.names = [f"sample_{i:06d}" for i in range(len(self.images))]

    def __len__(self):
        return len(self.images)

    def _build_rotation_matrix(self, yaw_deg, pitch_deg, roll_deg):
        yaw = float(np.deg2rad(yaw_deg))
        pitch = float(np.deg2rad(pitch_deg))
        roll = float(np.deg2rad(roll_deg))
        return np.asarray(pose_utils.get_R(pitch, yaw, roll), dtype=np.float32)

    def __getitem__(self, idx):
        img_np = self.images[idx]

        # 原 BIWI_noTrack.npz 来自 cv2.imread，通常是 BGR，转成 RGB
        img_np = img_np[:, :, ::-1].copy()

        yaw_deg, pitch_deg, roll_deg = self.poses[idx].astype(np.float32).tolist()

        img = Image.fromarray(img_np.astype(np.uint8))

        if self.train_mode:
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw_deg = -yaw_deg
                roll_deg = -roll_deg
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        if self.transform is not None:
            img = self.transform(img)

        # cont_labels 与原训练保持一致：弧度、顺序 [yaw, pitch, roll]
        cont_labels = torch.tensor([
            np.deg2rad(yaw_deg),
            np.deg2rad(pitch_deg),
            np.deg2rad(roll_deg)
        ], dtype=torch.float32)

        # R_gt 用项目统一方式生成，避免手写顺序不一致
        R = torch.from_numpy(self._build_rotation_matrix(yaw_deg, pitch_deg, roll_deg))

        name = self.names[idx]
        return img, R, cont_labels, name


class PseudoLabelDataset(Dataset):
    """
    伪标签数据集

    优先支持新的 face-level *_faces.npz 格式：
      key: face crop 文件名
      value: {
        rotation_matrix, confidence, consistency_score_deg, det_confidence,
        track_id, bbox, yaw, pitch, roll, use_for_training, ...
      }

    同时兼容旧格式：
    1) 单样本扁平格式
    2) frame-level 多人脸 faces 列表格式
    """
    def __init__(
        self,
        pseudo_data_root,
        pseudo_label_file,
        transform=None,
        confidence_threshold=0.8,
        multi_face_mode=True,
        use_yolo_refine=False,
        yolo_model_path=None,
        face_det_threshold=0.5,
        min_face_size=20,
        expand_bbox_ratio=0.2,
        use_gfpgan=False,
        gfpgan_model_path=None,
        gfpgan_low_conf_only=False,
        gfpgan_low_conf_threshold=0.85,
        consistency_threshold_deg=None,
        max_model_consistency_deg=None,
        min_det_confidence=0.0,
        min_track_length=0,
        require_use_for_training=True,
        min_blur_score=None,
        curriculum_min_face_size=None,
        curriculum_min_blur_score=None,
        curriculum_max_model_consistency_deg=None,
    ):
        self.pseudo_data_root = pseudo_data_root
        self.transform = transform
        self.confidence_threshold = confidence_threshold
        self.multi_face_mode = multi_face_mode
        self.use_yolo_refine = use_yolo_refine
        self.face_det_threshold = face_det_threshold
        self.min_face_size = min_face_size
        self.expand_bbox_ratio = expand_bbox_ratio
        self.gfpgan_low_conf_only = gfpgan_low_conf_only
        self.gfpgan_low_conf_threshold = gfpgan_low_conf_threshold
        self.consistency_threshold_deg = consistency_threshold_deg
        self.max_model_consistency_deg = max_model_consistency_deg
        self.min_det_confidence = min_det_confidence
        self.min_track_length = min_track_length
        self.require_use_for_training = require_use_for_training
        self.min_blur_score = min_blur_score
        self.curriculum_min_face_size = curriculum_min_face_size
        self.curriculum_min_blur_score = curriculum_min_blur_score
        self.curriculum_max_model_consistency_deg = curriculum_max_model_consistency_deg

        self.yolo_detector = None
        if use_yolo_refine and yolo_model_path and os.path.exists(yolo_model_path):
            try:
                from ultralytics import YOLO
                self.yolo_detector = YOLO(yolo_model_path)
                print(f"✓ Loaded YOLO refine detector from {yolo_model_path}")
            except Exception as e:
                print(f"Warning: failed to init YOLO refine detector: {e}")

        self.gfpgan_enhancer = None
        if use_gfpgan:
            model_path = gfpgan_model_path
            if model_path is None:
                for candidate in [
                    '/root/6DRepNet/gfpgan/weights/GFPGANv1.4.pth',
                    '/root/6DRepNet/gfpgan/weights/GFPGANv1.3.pth',
                    '/root/miniconda3/envs/hopenet/lib/python3.10/site-packages/gfpgan/weights/GFPGANv1.3.pth'
                ]:
                    if os.path.exists(candidate):
                        model_path = candidate
                        break

            if model_path and os.path.exists(model_path):
                try:
                    from gfpgan import GFPGANer
                    self.gfpgan_enhancer = GFPGANer(
                        model_path=model_path,
                        upscale=1,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    print(f"✓ Loaded GFPGAN enhancer from {model_path}")
                except Exception as e:
                    print(f"Warning: failed to init GFPGAN enhancer: {e}")

        print(f"Loading pseudo labels from {pseudo_label_file}")
        pseudo_data = np.load(pseudo_label_file, allow_pickle=True)

        self.samples = []

        for key in pseudo_data.files:
            raw = pseudo_data[key]
            try:
                item = raw.item()
            except Exception:
                continue

            if not isinstance(item, dict):
                continue

            # A) 新格式：face-level
            if 'rotation_matrix' in item and (
                'face_path' in item or
                'consistency_score_deg' in item or
                'det_confidence' in item or
                key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ):
                img_rel = item.get('face_path', key)
                img_path = self._resolve_image_path(img_rel)
                if img_path is None:
                    continue

                conf = float(item.get('confidence', 0.0))
                if conf < confidence_threshold:
                    continue

                if self.require_use_for_training and not bool(item.get('use_for_training', True)):
                    continue

                cons = float(item.get('consistency_score_deg', 0.0))
                if self.consistency_threshold_deg is not None and cons > self.consistency_threshold_deg:
                    continue

                det_conf = float(item.get('det_confidence', 1.0))
                if det_conf < self.min_det_confidence:
                    continue

                face_w = int(item.get('face_w', 9999))
                face_h = int(item.get('face_h', 9999))
                if min(face_w, face_h) < self.min_face_size:
                    continue
                if self.curriculum_min_face_size is not None and min(face_w, face_h) < self.curriculum_min_face_size:
                    continue

                blur_score = float(item.get('blur_score', 1e9))
                if self.min_blur_score is not None and blur_score < self.min_blur_score:
                    continue
                if self.curriculum_min_blur_score is not None and blur_score < self.curriculum_min_blur_score:
                    continue

                model_consistency = float(item.get('model_consistency_deg', 0.0))
                if self.max_model_consistency_deg is not None and model_consistency > self.max_model_consistency_deg:
                    continue
                if self.curriculum_max_model_consistency_deg is not None and model_consistency > self.curriculum_max_model_consistency_deg:
                    continue

                track_len = int(item.get('track_length', item.get('track_len', 9999)))
                if track_len < self.min_track_length:
                    continue

                self.samples.append({
                    'image_path': img_path,
                    'rotation_matrix': item['rotation_matrix'],
                    'confidence': conf,
                    'bbox': None,
                    'is_face_crop': True,
                    'consistency_score_deg': cons,
                    'det_confidence': det_conf,
                    'track_id': int(item.get('track_id', -1)),
                    'face_w': face_w,
                    'face_h': face_h,
                    'blur_score': blur_score,
                    'model_consistency_deg': model_consistency,
                    'yaw': float(item.get('yaw', 0.0)),
                    'pitch': float(item.get('pitch', 0.0)),
                    'roll': float(item.get('roll', 0.0)),
                })
                continue

            # B) 旧格式：单样本扁平
            if 'confidence' in item and 'rotation_matrix' in item:
                img_path = self._resolve_image_path(key)
                if img_path is None:
                    continue

                conf = float(item.get('confidence', 0.0))
                if conf < confidence_threshold:
                    continue

                self.samples.append({
                    'image_path': img_path,
                    'rotation_matrix': item['rotation_matrix'],
                    'confidence': conf,
                    'bbox': item.get('bbox', None),
                    'is_face_crop': False,
                    'consistency_score_deg': float(item.get('consistency_score_deg', 0.0)),
                    'det_confidence': float(item.get('det_confidence', 1.0)),
                    'track_id': int(item.get('track_id', -1)),
                    'yaw': float(item.get('yaw', 0.0)),
                    'pitch': float(item.get('pitch', 0.0)),
                    'roll': float(item.get('roll', 0.0)),
                })
                continue

            # C) 旧格式：frame-level faces 列表
            if 'faces' in item and isinstance(item['faces'], list):
                img_path = self._resolve_image_path(key)
                if img_path is None:
                    continue

                valid_faces = []
                for face in item['faces']:
                    if not isinstance(face, dict):
                        continue
                    if 'rotation_matrix' not in face:
                        continue

                    conf = float(face.get('confidence', 0.0))
                    if conf < confidence_threshold:
                        continue

                    if self.require_use_for_training and not bool(face.get('use_for_training', True)):
                        continue

                    cons = float(face.get('consistency_score_deg', 0.0))
                    if self.consistency_threshold_deg is not None and cons > self.consistency_threshold_deg:
                        continue

                    det_conf = float(face.get('det_confidence', face.get('face_detection_conf', 1.0)))
                    if det_conf < self.min_det_confidence:
                        continue

                    face_w = int(face.get('face_w', 9999))
                    face_h = int(face.get('face_h', 9999))
                    if min(face_w, face_h) < self.min_face_size:
                        continue
                    if self.curriculum_min_face_size is not None and min(face_w, face_h) < self.curriculum_min_face_size:
                        continue

                    blur_score = float(face.get('blur_score', 1e9))
                    if self.min_blur_score is not None and blur_score < self.min_blur_score:
                        continue
                    if self.curriculum_min_blur_score is not None and blur_score < self.curriculum_min_blur_score:
                        continue

                    model_consistency = float(face.get('model_consistency_deg', 0.0))
                    if self.max_model_consistency_deg is not None and model_consistency > self.max_model_consistency_deg:
                        continue
                    if self.curriculum_max_model_consistency_deg is not None and model_consistency > self.curriculum_max_model_consistency_deg:
                        continue

                    track_len = int(face.get('track_length', face.get('track_len', 9999)))
                    if track_len < self.min_track_length:
                        continue

                    valid_faces.append({
                        'image_path': img_path,
                        'rotation_matrix': face['rotation_matrix'],
                        'confidence': conf,
                        'bbox': face.get('bbox', None),
                        'is_face_crop': False,
                        'consistency_score_deg': cons,
                        'det_confidence': det_conf,
                        'track_id': int(face.get('track_id', -1)),
                        'face_w': face_w,
                        'face_h': face_h,
                        'blur_score': blur_score,
                        'model_consistency_deg': model_consistency,
                        'yaw': float(face.get('yaw', 0.0)),
                        'pitch': float(face.get('pitch', 0.0)),
                        'roll': float(face.get('roll', 0.0)),
                    })

                if valid_faces:
                    if self.multi_face_mode:
                        self.samples.extend(valid_faces)
                    else:
                        best_face = max(valid_faces, key=lambda x: x['confidence'])
                        self.samples.append(best_face)

        print(f"Loaded {len(self.samples)} pseudo-labeled samples")
        print(f"  confidence_threshold={confidence_threshold}")
        if self.consistency_threshold_deg is not None:
            print(f"  consistency_threshold_deg={self.consistency_threshold_deg}")
        if self.max_model_consistency_deg is not None:
            print(f"  max_model_consistency_deg={self.max_model_consistency_deg}")
        print(f"  min_det_confidence={self.min_det_confidence}")
        print(f"  min_face_size={self.min_face_size}")
        if self.min_blur_score is not None:
            print(f"  min_blur_score={self.min_blur_score}")
        if self.curriculum_min_face_size is not None:
            print(f"  curriculum_min_face_size={self.curriculum_min_face_size}")
        if self.curriculum_min_blur_score is not None:
            print(f"  curriculum_min_blur_score={self.curriculum_min_blur_score}")
        if self.curriculum_max_model_consistency_deg is not None:
            print(f"  curriculum_max_model_consistency_deg={self.curriculum_max_model_consistency_deg}")

    def _resolve_image_path(self, key):
        key_norm = os.path.normpath(str(key))
        candidates = []
        stripped = key_norm
        while stripped.startswith('../'):
            stripped = stripped[3:]
        stripped = stripped.lstrip('./')

        if os.path.isabs(key_norm):
            candidates.append(key_norm)
        else:
            candidates.append(os.path.normpath(os.path.join(self.pseudo_data_root, key_norm)))
            candidates.append(os.path.normpath(os.path.join('/root', key_norm.lstrip('./'))))
            if stripped:
                candidates.append(os.path.normpath(os.path.join('/root', stripped)))
                candidates.append(os.path.normpath(os.path.join(self.pseudo_data_root, stripped)))

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _expand_bbox(self, bbox, w, h):
        if bbox is None or len(bbox) < 4:
            return [0, 0, w, h]

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        mx = int(bw * self.expand_bbox_ratio)
        my = int(bh * self.expand_bbox_ratio)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)

        return [x1, y1, x2, y2]

    def _bbox_iou(self, b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1 = max(x11, x21)
        iy1 = max(y11, y21)
        ix2 = min(x12, x22)
        iy2 = min(y12, y22)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        a1 = max(1, (x12 - x11) * (y12 - y11))
        a2 = max(1, (x22 - x21) * (y22 - y21))
        return inter / float(a1 + a2 - inter + 1e-6)

    def _refine_bbox_with_yolo(self, image_bgr, fallback_bbox):
        if self.yolo_detector is None:
            return fallback_bbox

        try:
            results = self.yolo_detector(image_bgr, verbose=False)
        except Exception:
            return fallback_bbox

        candidates = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.face_det_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                    continue
                candidates.append(([x1, y1, x2, y2], conf))

        if not candidates:
            return fallback_bbox

        if fallback_bbox is None:
            return max(candidates, key=lambda x: x[1])[0]

        best_box, _ = max(
            candidates,
            key=lambda x: (self._bbox_iou(x[0], fallback_bbox), x[1])
        )
        return best_box

    def _enhance_with_gfpgan(self, face_rgb):
        if self.gfpgan_enhancer is None:
            return face_rgb

        try:
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            _, _, output = self.gfpgan_enhancer.enhance(
                face_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )
            if output is None:
                return face_rgb
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except Exception:
            return face_rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_pil = Image.open(sample['image_path']).convert('RGB')
        img_np = np.array(img_pil)

        if sample.get('is_face_crop', False):
            face_np = img_np
        else:
            h, w = img_np.shape[:2]
            bbox = sample.get('bbox', None)
            bbox = self._expand_bbox(bbox, w, h)

            if self.use_yolo_refine:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                bbox = self._refine_bbox_with_yolo(img_bgr, bbox)

            x1, y1, x2, y2 = bbox
            face_np = img_np[y1:y2, x1:x2]
            if face_np.size == 0:
                face_np = img_np

        face_h, face_w = face_np.shape[:2]
        should_enhance = min(face_h, face_w) < max(32, self.min_face_size * 2)
        if self.gfpgan_low_conf_only:
            should_enhance = should_enhance and (
                float(sample.get('confidence', 1.0)) < self.gfpgan_low_conf_threshold
            )

        if should_enhance:
            face_np = self._enhance_with_gfpgan(face_np)

        img = Image.fromarray(face_np)
        if self.transform:
            img = self.transform(img)

        R = torch.as_tensor(sample['rotation_matrix'], dtype=torch.float32).reshape(3, 3)

        angles = torch.tensor([
            float(sample.get('yaw', 0.0)),
            float(sample.get('pitch', 0.0)),
            float(sample.get('roll', 0.0)),
        ], dtype=torch.float32)

        confidence = float(sample['confidence'])
        name = os.path.basename(sample['image_path'])

        return img, R, angles, torch.tensor([confidence], dtype=torch.float32), name


class SemiSupervisedDataset(Dataset):
    """
    老的半监督结构，保留兼容，不作为当前主训练入口
    """
    def __init__(self,
                 labeled_dataset,
                 unlabeled_image_paths,
                 pseudo_labels_file=None,
                 transform_weak=None,
                 transform_strong=None,
                 confidence_threshold=0.8,
                 unlabeled_ratio=2.0):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_image_paths = unlabeled_image_paths
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.confidence_threshold = confidence_threshold
        self.unlabeled_ratio = unlabeled_ratio

        self.pseudo_labels = None
        if pseudo_labels_file is not None and os.path.exists(pseudo_labels_file):
            print(f"Loading pseudo labels from {pseudo_labels_file}")
            data = np.load(pseudo_labels_file, allow_pickle=True)
            self.pseudo_labels = {k: v.item() for k, v in data.items()}

            self.filtered_unlabeled_paths = [
                path for path in unlabeled_image_paths
                if os.path.relpath(path) in self.pseudo_labels and
                self.pseudo_labels[os.path.relpath(path)]['confidence'] >= confidence_threshold
            ]
            print(f"Filtered unlabeled samples: {len(self.filtered_unlabeled_paths)}/{len(unlabeled_image_paths)}")
        else:
            self.filtered_unlabeled_paths = unlabeled_image_paths
            print("No pseudo labels provided, will use consistency regularization only")

        self.labeled_length = len(labeled_dataset)
        self.unlabeled_length = len(self.filtered_unlabeled_paths)

    def __len__(self):
        return self.labeled_length

    def __getitem__(self, index):
        labeled_img, labeled_R, labeled_angles, labeled_name = self.labeled_dataset[index]

        sample_data = {
            'labeled_img': labeled_img,
            'labeled_R': labeled_R,
            'labeled_angles': labeled_angles,
            'labeled_name': labeled_name,
        }

        if self.unlabeled_length > 0:
            num_unlabeled = max(1, int(self.unlabeled_ratio))

            for i in range(num_unlabeled):
                unlabeled_idx = random.randint(0, self.unlabeled_length - 1)
                unlabeled_path = self.filtered_unlabeled_paths[unlabeled_idx]

                try:
                    unlabeled_img = Image.open(unlabeled_path).convert('RGB')
                except Exception:
                    unlabeled_idx = random.randint(0, self.unlabeled_length - 1)
                    unlabeled_path = self.filtered_unlabeled_paths[unlabeled_idx]
                    unlabeled_img = Image.open(unlabeled_path).convert('RGB')

                unlabeled_img_weak = self.transform_weak(unlabeled_img) if self.transform_weak is not None else unlabeled_img
                unlabeled_img_strong = self.transform_strong(unlabeled_img) if self.transform_strong is not None else unlabeled_img

                if self.pseudo_labels is not None:
                    rel_path = os.path.relpath(unlabeled_path)
                    if rel_path in self.pseudo_labels:
                        pseudo_R = torch.FloatTensor(self.pseudo_labels[rel_path]['rotation_matrix'])
                        confidence = self.pseudo_labels[rel_path]['confidence']
                    else:
                        pseudo_R = None
                        confidence = 0.0
                else:
                    pseudo_R = None
                    confidence = 0.0

                sample_data[f'unlabeled_weak_{i}'] = unlabeled_img_weak
                sample_data[f'unlabeled_strong_{i}'] = unlabeled_img_strong
                sample_data[f'pseudo_R_{i}'] = pseudo_R
                sample_data[f'confidence_{i}'] = confidence

        return sample_data


class ClassroomVideoDataset(Dataset):
    """
    课堂图片/视频帧数据集（无标签）
    """
    def __init__(self, image_dir, transform=None, file_pattern='*.jpg'):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, file_pattern)))
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path


class MixedBatchSampler:
    """
    混合批次采样器（旧版，当前主流程不使用）
    """
    def __init__(self,
                 labeled_dataset_size,
                 unlabeled_dataset_size,
                 batch_size,
                 labeled_batch_size=None):
        self.labeled_dataset_size = labeled_dataset_size
        self.unlabeled_dataset_size = unlabeled_dataset_size
        self.batch_size = batch_size

        if labeled_batch_size is None:
            self.labeled_batch_size = batch_size // 2
        else:
            self.labeled_batch_size = labeled_batch_size

        self.unlabeled_batch_size = batch_size - self.labeled_batch_size

    def __iter__(self):
        labeled_indices = np.random.permutation(self.labeled_dataset_size).tolist()
        unlabeled_indices = np.random.permutation(self.unlabeled_dataset_size).tolist()

        num_batches = max(
            len(labeled_indices) // self.labeled_batch_size,
            len(unlabeled_indices) // self.unlabeled_batch_size
        )

        while len(labeled_indices) < num_batches * self.labeled_batch_size:
            labeled_indices.extend(np.random.permutation(self.labeled_dataset_size).tolist())

        while len(unlabeled_indices) < num_batches * self.unlabeled_batch_size:
            unlabeled_indices.extend(np.random.permutation(self.unlabeled_dataset_size).tolist())

        for i in range(num_batches):
            batch_labeled = labeled_indices[i * self.labeled_batch_size:(i + 1) * self.labeled_batch_size]
            batch_unlabeled = unlabeled_indices[i * self.unlabeled_batch_size:(i + 1) * self.unlabeled_batch_size]
            batch = batch_labeled + [idx + self.labeled_dataset_size for idx in batch_unlabeled]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return max(
            self.labeled_dataset_size // self.labeled_batch_size,
            self.unlabeled_dataset_size // self.unlabeled_batch_size
        )


def collate_fn_semi_supervised(batch):
    """
    老结构，保留兼容
    """
    labeled_imgs = []
    labeled_Rs = []
    labeled_angles = []
    labeled_names = []

    unlabeled_weaks = []
    unlabeled_strongs = []
    pseudo_Rs = []
    confidences = []

    for sample in batch:
        labeled_imgs.append(sample['labeled_img'])
        labeled_Rs.append(sample['labeled_R'])
        labeled_angles.append(sample['labeled_angles'])
        labeled_names.append(sample['labeled_name'])

        i = 0
        while f'unlabeled_weak_{i}' in sample:
            unlabeled_weaks.append(sample[f'unlabeled_weak_{i}'])
            unlabeled_strongs.append(sample[f'unlabeled_strong_{i}'])

            if sample[f'pseudo_R_{i}'] is not None:
                pseudo_Rs.append(sample[f'pseudo_R_{i}'])

            confidences.append(sample[f'confidence_{i}'])
            i += 1

    batch_data = {
        'labeled_imgs': torch.stack(labeled_imgs) if labeled_imgs else None,
        'labeled_Rs': torch.stack(labeled_Rs) if labeled_Rs else None,
        'labeled_angles': torch.stack(labeled_angles) if labeled_angles else None,
        'labeled_names': labeled_names,
        'unlabeled_weaks': torch.stack(unlabeled_weaks) if unlabeled_weaks else None,
        'unlabeled_strongs': torch.stack(unlabeled_strongs) if unlabeled_strongs else None,
        'pseudo_Rs': torch.stack(pseudo_Rs) if pseudo_Rs else None,
        'confidences': torch.FloatTensor(confidences) if confidences else None,
    }

    return batch_data


def _load_local_datasets_module():
    datasets_py = os.path.join(os.path.dirname(__file__), 'datasets.py')
    spec = importlib.util.spec_from_file_location('sixdrepnet_local_datasets', datasets_py)
    datasets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(datasets_module)
    return datasets_module


def create_semi_supervised_dataloaders(
    labeled_data_dir,
    labeled_filename_list,
    pseudo_label_file,
    pseudo_data_root,
    batch_size=64,
    num_workers=4,
    pseudo_label_ratio=0.5,
    confidence_threshold=0.8,
    multi_face_mode=True,
    use_yolo_refine=False,
    yolo_model_path=None,
    face_det_threshold=0.5,
    min_face_size=20,
    expand_bbox_ratio=0.2,
    use_gfpgan=False,
    gfpgan_model_path=None,
    gfpgan_low_conf_only=False,
    gfpgan_low_conf_threshold=0.85,
    consistency_threshold_deg=None,
    max_model_consistency_deg=None,
    min_det_confidence=0.0,
    min_track_length=0,
    require_use_for_training=True,
    include_pseudo=True,
    use_weighted_sampler=True,
    seed=None,
    min_blur_score=None,
    curriculum_min_face_size=None,
    curriculum_min_blur_score=None,
    curriculum_max_model_consistency_deg=None,
):
    """
    创建半监督数据加载器
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

    transformations = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading labeled dataset from {labeled_data_dir}")

    # 1) 优先识别你这种扁平 npz 格式
    use_flat_npz = False
    if os.path.isfile(labeled_data_dir) and labeled_data_dir.endswith('.npz'):
        try:
            pack = np.load(labeled_data_dir, allow_pickle=True)
            if 'image' in pack and 'pose' in pack:
                use_flat_npz = True
        except Exception:
            use_flat_npz = False

    if use_flat_npz:
        print("Using FlatNPZPoseDataset for labeled data")
        labeled_dataset = FlatNPZPoseDataset(
            labeled_data_dir,
            labeled_filename_list,
            transformations,
            train_mode=True
        )
    else:
        datasets_module = _load_local_datasets_module()

        if hasattr(datasets_module, 'SixDRepNet_Dataset'):
            dataset_cls = datasets_module.SixDRepNet_Dataset
        elif os.path.isdir(labeled_data_dir) and hasattr(datasets_module, 'Pose_300W_LP'):
            dataset_cls = datasets_module.Pose_300W_LP
        elif hasattr(datasets_module, 'BIWI'):
            dataset_cls = datasets_module.BIWI
        else:
            raise ImportError('No compatible labeled dataset class found in local datasets.py')

        try:
            labeled_dataset = dataset_cls(
                labeled_data_dir,
                labeled_filename_list,
                transformations,
                train_mode=True
            )
        except TypeError:
            labeled_dataset = dataset_cls(
                labeled_data_dir,
                labeled_filename_list,
                transformations
            )

    pseudo_dataset = None
    pseudo_source_pairs = pair_pseudo_sources(pseudo_label_file, pseudo_data_root) if include_pseudo else []
    if include_pseudo:
        pseudo_datasets = []
        pseudo_source_stats = []

        for source_idx, (current_pseudo_file, current_pseudo_root) in enumerate(pseudo_source_pairs, start=1):
            print(
                f"Loading pseudo source {source_idx}/{len(pseudo_source_pairs)}: "
                f"file={current_pseudo_file}, root={current_pseudo_root}"
            )
            def _build_pseudo_dataset(
                *,
                cur_min_face_size,
                cur_min_blur_score,
                cur_max_model_consistency_deg,
            ):
                return PseudoLabelDataset(
                    pseudo_data_root=current_pseudo_root,
                    pseudo_label_file=current_pseudo_file,
                    transform=transformations,
                    confidence_threshold=confidence_threshold,
                    multi_face_mode=multi_face_mode,
                    use_yolo_refine=use_yolo_refine,
                    yolo_model_path=yolo_model_path,
                    face_det_threshold=face_det_threshold,
                    min_face_size=min_face_size,
                    expand_bbox_ratio=expand_bbox_ratio,
                    use_gfpgan=use_gfpgan,
                    gfpgan_model_path=gfpgan_model_path,
                    gfpgan_low_conf_only=gfpgan_low_conf_only,
                    gfpgan_low_conf_threshold=gfpgan_low_conf_threshold,
                    consistency_threshold_deg=consistency_threshold_deg,
                    max_model_consistency_deg=max_model_consistency_deg,
                    min_det_confidence=min_det_confidence,
                    min_track_length=min_track_length,
                    require_use_for_training=require_use_for_training,
                    min_blur_score=min_blur_score,
                    curriculum_min_face_size=cur_min_face_size,
                    curriculum_min_blur_score=cur_min_blur_score,
                    curriculum_max_model_consistency_deg=cur_max_model_consistency_deg,
                )

            current_dataset = _build_pseudo_dataset(
                cur_min_face_size=curriculum_min_face_size,
                cur_min_blur_score=curriculum_min_blur_score,
                cur_max_model_consistency_deg=curriculum_max_model_consistency_deg,
            )

            if include_pseudo and len(current_dataset) == 0:
                relaxed = False

                # Curriculum thresholds can become too strict for a given pseudo set
                # (e.g., face crops are all < curriculum_min_face_size). In that case,
                # fall back to a relaxed curriculum so training doesn't degenerate.
                cur_face = curriculum_min_face_size
                cur_blur = curriculum_min_blur_score
                cur_cons = curriculum_max_model_consistency_deg

                if cur_face is not None:
                    print(
                        "[WARN] Pseudo dataset is empty after curriculum filtering; "
                        "retry without curriculum_min_face_size (face-size curriculum disabled for this source)."
                    )
                    cur_face = None
                    current_dataset = _build_pseudo_dataset(
                        cur_min_face_size=cur_face,
                        cur_min_blur_score=cur_blur,
                        cur_max_model_consistency_deg=cur_cons,
                    )
                    relaxed = True

                if len(current_dataset) == 0 and cur_blur is not None:
                    print(
                        "[WARN] Pseudo dataset still empty; retry without curriculum_min_blur_score."
                    )
                    cur_blur = None
                    current_dataset = _build_pseudo_dataset(
                        cur_min_face_size=cur_face,
                        cur_min_blur_score=cur_blur,
                        cur_max_model_consistency_deg=cur_cons,
                    )
                    relaxed = True

                if len(current_dataset) == 0 and cur_cons is not None:
                    print(
                        "[WARN] Pseudo dataset still empty; retry without curriculum_max_model_consistency_deg."
                    )
                    cur_cons = None
                    current_dataset = _build_pseudo_dataset(
                        cur_min_face_size=cur_face,
                        cur_min_blur_score=cur_blur,
                        cur_max_model_consistency_deg=cur_cons,
                    )
                    relaxed = True

                if relaxed and len(current_dataset) == 0:
                    print(
                        "[WARN] Pseudo dataset remains empty even after relaxing curriculum thresholds; "
                        "continue with supervised-only batches for this epoch/source."
                    )

            pseudo_datasets.append(current_dataset)
            pseudo_source_stats.append({
                'pseudo_label_file': current_pseudo_file,
                'pseudo_data_root': current_pseudo_root,
                'n_samples': len(current_dataset),
            })

        if len(pseudo_datasets) == 1:
            pseudo_dataset = pseudo_datasets[0]
        elif len(pseudo_datasets) > 1:
            pseudo_dataset = ConcatDataset(pseudo_datasets)
        else:
            pseudo_dataset = None
    else:
        print("Supervised-only mode: pseudo dataset is disabled")

    n_labeled = len(labeled_dataset)
    n_pseudo = len(pseudo_dataset) if pseudo_dataset is not None else 0

    print("\nDataset Statistics:")
    print(f"  Labeled samples: {n_labeled}")
    print(f"  Pseudo samples: {n_pseudo}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Pseudo/Labeled ratio: {pseudo_label_ratio}")

    if pseudo_dataset is not None and n_pseudo > 0:
        combined_dataset = ConcatDataset([labeled_dataset, pseudo_dataset])
    else:
        combined_dataset = labeled_dataset

    sampler = None
    shuffle = True
    sampled_pseudo_ratio = (n_pseudo / (n_labeled + n_pseudo)) if (n_labeled + n_pseudo) > 0 else 0.0

    # pseudo_label_ratio 定义为 P/L 比例，例如 0.5 表示伪标签:有标签=1:2
    if use_weighted_sampler and n_pseudo > 0 and pseudo_label_ratio > 0:
        pseudo_weight = (pseudo_label_ratio * n_labeled) / max(n_pseudo, 1)
        sample_weights = torch.ones(len(combined_dataset), dtype=torch.double)
        sample_weights[n_labeled:] = pseudo_weight
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(combined_dataset),
            replacement=True
        )
        shuffle = False
        sampled_pseudo_ratio = pseudo_label_ratio / (1.0 + pseudo_label_ratio)
        print(f"  Weighted sampling enabled (target pseudo fraction per epoch: {sampled_pseudo_ratio:.1%})")

    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = seed_worker

    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mixed_dataset,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    stats = {
        'n_labeled': n_labeled,
        'n_pseudo': n_pseudo,
        'total': n_labeled + n_pseudo,
        'pseudo_ratio': (n_pseudo / (n_labeled + n_pseudo)) if (n_labeled + n_pseudo) > 0 else 0.0,
        'sampled_pseudo_ratio_target': sampled_pseudo_ratio,
        'confidence_threshold': confidence_threshold,
        'max_model_consistency_deg': max_model_consistency_deg,
        'min_blur_score': min_blur_score,
        'curriculum_min_face_size': curriculum_min_face_size,
        'curriculum_min_blur_score': curriculum_min_blur_score,
        'curriculum_max_model_consistency_deg': curriculum_max_model_consistency_deg,
        'pseudo_sources': pseudo_source_stats if include_pseudo else []
    }

    return train_loader, stats


def collate_mixed_dataset(batch):
    """
    自定义 collate 函数，处理混合的有标签和伪标签数据

    有标签数据格式: (img, R, angles, name)
    伪标签数据格式: (img, R, angles, confidence, name)

    统一输出为:
      (images, R_matrices, angles, confidences, names, source_is_pseudo)
    """
    images = []
    R_matrices = []
    angles_list = []
    confidences = []
    names = []
    source_is_pseudo = []

    def _to_angle3(angle):
        angle_t = torch.as_tensor(angle, dtype=torch.float32).reshape(-1)
        if angle_t.numel() == 0:
            return torch.zeros(3, dtype=torch.float32)
        if angle_t.numel() >= 3:
            return angle_t[:3]
        pad = torch.zeros(3 - angle_t.numel(), dtype=torch.float32)
        return torch.cat([angle_t, pad], dim=0)

    def _to_rot3x3(R):
        R_t = torch.as_tensor(R, dtype=torch.float32).reshape(-1)
        if R_t.numel() >= 9:
            return R_t[:9].reshape(3, 3)
        return None

    for item in batch:
        if len(item) == 4:
            img, R, angle, name = item
            R_t = _to_rot3x3(R)
            if R_t is None:
                continue
            images.append(img)
            R_matrices.append(R_t)
            angles_list.append(_to_angle3(angle))
            confidences.append(1.0)
            names.append(name)
            source_is_pseudo.append(False)

        elif len(item) == 5:
            img, R, angle, confidence, name = item
            R_t = _to_rot3x3(R)
            if R_t is None:
                continue
            images.append(img)
            R_matrices.append(R_t)
            angles_list.append(_to_angle3(angle))

            if torch.is_tensor(confidence):
                confidence_val = float(confidence.reshape(-1)[0].item())
            else:
                confidence_val = float(confidence)

            confidences.append(confidence_val)
            names.append(name)
            source_is_pseudo.append(True)
        else:
            raise ValueError(f"Unexpected batch item format with {len(item)} elements")

    if len(images) == 0:
        raise RuntimeError('All samples in batch are invalid after collation')

    images_tensor = torch.stack(images)
    R_tensor = torch.stack(R_matrices)
    angles_tensor = torch.stack(angles_list)
    confidences_tensor = torch.tensor(confidences, dtype=torch.float32)
    source_is_pseudo_tensor = torch.tensor(source_is_pseudo, dtype=torch.bool)

    return images_tensor, R_tensor, angles_tensor, confidences_tensor, names, source_is_pseudo_tensor
