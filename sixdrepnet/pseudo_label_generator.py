# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import cv2
# from tqdm import tqdm
# import json


# class IoUTracker:
#     """轻量级 IoU 跟踪器（跨帧分配 track_id）。"""
#     def __init__(self, iou_threshold=0.3, max_missed=8):
#         self.iou_threshold = iou_threshold
#         self.max_missed = max_missed
#         self.next_id = 0
#         self.tracks = {}  # tid -> {'bbox': [x1,y1,x2,y2], 'last_frame': int}

#     @staticmethod
#     def _iou(b1, b2):
#         x11, y11, x12, y12 = b1
#         x21, y21, x22, y22 = b2
#         ix1, iy1 = max(x11, x21), max(y11, y21)
#         ix2, iy2 = min(x12, x22), min(y12, y22)
#         iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
#         inter = iw * ih
#         a1 = max(1, (x12 - x11) * (y12 - y11))
#         a2 = max(1, (x22 - x21) * (y22 - y21))
#         return inter / float(a1 + a2 - inter + 1e-6)

#     def _prune(self, frame_idx):
#         dead = []
#         for tid, t in self.tracks.items():
#             if frame_idx - t['last_frame'] > self.max_missed:
#                 dead.append(tid)
#         for tid in dead:
#             self.tracks.pop(tid, None)

#     def update(self, detections, frame_idx):
#         self._prune(frame_idx)
#         track_ids = [-1] * len(detections)
#         if len(detections) == 0:
#             return track_ids

#         candidates = []
#         for det_idx, det in enumerate(detections):
#             for tid, tr in self.tracks.items():
#                 iou = self._iou(det, tr['bbox'])
#                 if iou >= self.iou_threshold:
#                     candidates.append((iou, det_idx, tid))

#         candidates.sort(reverse=True, key=lambda x: x[0])
#         used_det = set()
#         used_tid = set()

#         for _, det_idx, tid in candidates:
#             if det_idx in used_det or tid in used_tid:
#                 continue
#             track_ids[det_idx] = tid
#             used_det.add(det_idx)
#             used_tid.add(tid)
#             self.tracks[tid]['bbox'] = detections[det_idx]
#             self.tracks[tid]['last_frame'] = frame_idx

#         for det_idx, det in enumerate(detections):
#             if det_idx in used_det:
#                 continue
#             tid = self.next_id
#             self.next_id += 1
#             self.tracks[tid] = {'bbox': det, 'last_frame': frame_idx}
#             track_ids[det_idx] = tid

#         return track_ids


# class PseudoLabelGenerator:
#     """
#     使用预训练的6DRepNet模型为无标签数据生成伪标签
#     """
#     def __init__(self, 
#                  model,
#                  device='cuda',
#                  confidence_threshold=0.8,
#                  use_tta=True,
#                  tta_rotations=[0, 5, -5],
#                  save_visualizations=False,
#                  yolo_model_path=None,
#                  use_gfpgan=False,
#                  gfpgan_model_path=None,
#                  face_det_threshold=0.5,
#                  min_face_size=20):
#         """
#         Args:
#             model: 预训练的6DRepNet模型
#             device: 运行设备
#             confidence_threshold: 置信度阈值
#             use_tta: 是否使用测试时增强
#             tta_rotations: TTA旋转角度列表
#             save_visualizations: 是否保存可视化结果
#             yolo_model_path: YOLO人脸检测模型路径（None则不使用）
#             use_gfpgan: 是否使用GFPGAN增强
#             gfpgan_model_path: GFPGAN模型路径
#             face_det_threshold: 人脸检测置信度阈值
#             min_face_size: 最小人脸尺寸
#         """
#         self.model = model
#         self.device = device
#         self.confidence_threshold = confidence_threshold
#         self.use_tta = use_tta
#         self.tta_rotations = tta_rotations
#         self.save_visualizations = save_visualizations
#         self.face_det_threshold = face_det_threshold
#         self.min_face_size = min_face_size
        
#         self.model.eval()
#         self.model.to(device)
        
#         # 加载YOLO人脸检测器
#         self.yolo_detector = None
#         if yolo_model_path:
#             if not os.path.exists(yolo_model_path):
#                 print(f"Warning: YOLO model not found at {yolo_model_path}")
#             else:
#                 try:
#                     from ultralytics import YOLO
#                     self.yolo_detector = YOLO(yolo_model_path)
#                     print(f"✓ Loaded YOLO face detector from {yolo_model_path}")
#                 except ImportError:
#                     print("Warning: Failed to import ultralytics. Please install: pip install ultralytics")
#                 except Exception as e:
#                     print(f"Warning: Failed to load YOLO detector: {e}")
        
#         # 加载GFPGAN
#         self.gfpgan_enhancer = None
#         if use_gfpgan:
#             if not gfpgan_model_path:
#                 print("Warning: GFPGAN enabled but model path is None")
#             elif not os.path.exists(gfpgan_model_path):
#                 print(f"Warning: GFPGAN model not found at {gfpgan_model_path}")
#             else:
#                 try:
#                     from gfpgan import GFPGANer
#                     self.gfpgan_enhancer = GFPGANer(
#                         model_path=gfpgan_model_path,
#                         upscale=1,
#                         arch='clean',
#                         channel_multiplier=2,
#                         bg_upsampler=None,
#                         device=device
#                     )
#                     print(f"✓ Loaded GFPGAN enhancer from {gfpgan_model_path}")
#                 except ImportError:
#                     print("Warning: Failed to import GFPGAN. Please install: pip install gfpgan")
#                 except Exception as e:
#                     print(f"Warning: Failed to load GFPGAN: {e}")
    
#     def compute_confidence(self, R_pred):
#         """
#         计算旋转矩阵预测的置信度
#         基于正交性约束：R^T * R 应该接近 I
#         """
#         RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)
#         I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype).unsqueeze(0)
        
#         # 计算偏差
#         ortho_error = ((RtR - I) ** 2).sum(dim=(1, 2))
        
#         # 转换为置信度分数
#         confidence = torch.exp(-ortho_error * 10)
        
#         return confidence.cpu().numpy()
    
#     def rotate_image(self, image, angle):
#         """旋转图像用于TTA"""
#         if angle == 0:
#             return image
        
#         # 如果是PIL Image
#         if isinstance(image, Image.Image):
#             return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
#         # 如果是numpy array
#         else:
#             h, w = image.shape[:2]
#             center = (w // 2, h // 2)
#             M = cv2.getRotationMatrix2D(center, angle, 1.0)
#             return cv2.warpAffine(image, M, (w, h))
    
#     @torch.no_grad()
#     def generate_pseudo_label(self, image_tensor):
#         """
#         为单张图像生成伪标签
        
#         Args:
#             image_tensor: 预处理后的图像张量 (1, C, H, W)
        
#         Returns:
#             R_pred: 预测的旋转矩阵 (3, 3)
#             confidence: 置信度分数
#         """
#         if self.use_tta:
#             # 使用多个增强版本进行预测，然后平均
#             predictions = []
            
#             for angle in self.tta_rotations:
#                 # 注意：这里假设image_tensor已经是tensor
#                 # 实际应用中可能需要在图像级别做旋转
#                 R_pred = self.model(image_tensor.to(self.device))
#                 predictions.append(R_pred.cpu())
            
#             # 平均多个预测（在旋转矩阵空间中平均需要特殊处理）
#             # 简单方法：直接平均然后重新正交化
#             R_pred_avg = torch.stack(predictions).mean(dim=0)
            
#             # SVD正交化
#             U, S, V = torch.svd(R_pred_avg)
#             R_pred_final = torch.mm(U, V.t()).unsqueeze(0)
            
#         else:
#             R_pred_final = self.model(image_tensor.to(self.device)).cpu()
        
#         # 计算置信度
#         confidence = self.compute_confidence(R_pred_final)[0]
        
#         return R_pred_final[0], confidence
    
#     def process_frame_with_multiple_faces(
#         self,
#         image_path: str,
#         transform
#     ) -> list[dict]:
#         """
#         处理单帧图像中的多个人脸
        
#         Args:
#             image_path: 图像路径
#             transform: torchvision变换
            
#         Returns:
#             人脸结果列表，每个包含姿态、置信度、边界框等信息
#         """
#         # 读取图像
#         img_bgr = cv2.imread(image_path)
#         if img_bgr is None:
#             return []
        
#         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
#         # 检测所有人脸
#         faces = self.detect_faces(img_bgr)
        
#         results = []
#         for face_idx, face_info in enumerate(faces):
#             bbox = face_info['bbox']
#             x1, y1, x2, y2 = bbox
            
#             # 裁剪人脸
#             face_img_bgr = img_bgr[y1:y2, x1:x2]
            
#             if face_img_bgr.size == 0:
#                 continue
            
#             # 可选的GFPGAN增强
#             if self.gfpgan is not None:
#                 face_img_bgr = self.enhance_face(face_img_bgr)
            
#             # 转换为RGB并处理
#             face_img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
#             face_pil = Image.fromarray(face_img_rgb)
            
#             # 应用变换
#             face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
            
#             # 预测姿态
#             rotation_matrix, confidence = self.predict_with_confidence(face_tensor)
            
#             # 计算欧拉角
#             pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
            
#             results.append({
#                 'face_id': face_idx,
#                 'bbox': bbox,
#                 'face_detection_conf': face_info['confidence'],
#                 'rotation_matrix': rotation_matrix,
#                 'pitch': pitch,
#                 'yaw': yaw,
#                 'roll': roll,
#                 'pose_confidence': confidence,
#                 'use_for_training': confidence >= self.confidence_threshold
#             })
        
#         return results
    
#     def detect_and_crop_face(self, img_bgr):
#         """使用YOLO检测并裁剪人脸"""
#         if self.yolo_detector is None:
#             # 如果没有YOLO，返回整张图
#             return [{'bbox': [0, 0, img_bgr.shape[1], img_bgr.shape[0]], 'conf': 1.0}]
        
#         results = self.yolo_detector(img_bgr, verbose=False)
#         faces = []
        
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 conf = float(box.conf[0])
#                 if conf < self.face_det_threshold:
#                     continue
                
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 w, h = x2 - x1, y2 - y1
                
#                 if w < self.min_face_size or h < self.min_face_size:
#                     continue
                
#                 # 扩展边界框
#                 margin = 0.2
#                 x1 = max(0, x1 - int(w * margin))
#                 y1 = max(0, y1 - int(h * margin))
#                 x2 = min(img_bgr.shape[1], x2 + int(w * margin))
#                 y2 = min(img_bgr.shape[0], y2 + int(h * margin))
                
#                 faces.append({'bbox': [x1, y1, x2, y2], 'conf': conf})
        
#         return faces if faces else [{'bbox': [0, 0, img_bgr.shape[1], img_bgr.shape[0]], 'conf': 1.0}]
    
#     def enhance_face_gfpgan(self, face_img):
#         """使用GFPGAN增强人脸图像"""
#         if self.gfpgan_enhancer is None:
#             return face_img
        
#         try:
#             _, _, output = self.gfpgan_enhancer.enhance(
#                 face_img, has_aligned=False, only_center_face=False, paste_back=True
#             )
#             return output
#         except Exception as e:
#             print(f"GFPGAN enhancement failed: {e}")
#             return face_img
    
#     def generate_for_dataset(self, 
#                             image_paths, 
#                             transform,
#                             output_file,
#                             batch_size=32,
#                             multi_face_mode=True):
#         """
#         为整个数据集生成伪标签
        
#         Args:
#             image_paths: 图像路径列表
#             transform: 图像预处理transform
#             output_file: 输出文件路径（npz格式）
#             batch_size: 批处理大小
#             multi_face_mode: 是否处理多人脸（课堂场景推荐True）
#         """
#         # 为时序跟踪稳定，先按文件名排序
#         image_paths = sorted(image_paths)

#         pseudo_labels = {}
#         track_labels = {}
#         high_confidence_count = 0
#         total_faces = 0

#         tracker = IoUTracker(iou_threshold=0.3, max_missed=8) if (multi_face_mode and self.yolo_detector) else None
#         frame_to_index = {p: i for i, p in enumerate(image_paths)}
        
#         print(f"Generating pseudo labels for {len(image_paths)} images...")
#         print(f"Multi-face mode: {multi_face_mode}, YOLO: {self.yolo_detector is not None}, GFPGAN: {self.gfpgan_enhancer is not None}")
        
#         for img_path in tqdm(image_paths):
#             try:
#                 # 读取图像
#                 img_bgr = cv2.imread(img_path)
#                 if img_bgr is None:
#                     continue
                
#                 rel_path = os.path.relpath(img_path)
                
#                 if multi_face_mode and self.yolo_detector:
#                     # 多人脸模式
#                     faces = self.detect_and_crop_face(img_bgr)
#                     face_results = []
#                     track_face_results = []

#                     frame_idx = frame_to_index.get(img_path, 0)
#                     det_bboxes = [f['bbox'] for f in faces if f['bbox'] is not None]
#                     det_track_ids = tracker.update(det_bboxes, frame_idx) if tracker is not None and len(det_bboxes) > 0 else []
#                     det_cursor = 0
                    
#                     for face_info in faces:
#                         x1, y1, x2, y2 = face_info['bbox']
#                         face_img = img_bgr[y1:y2, x1:x2]

#                         if face_info['bbox'] is not None and det_cursor < len(det_track_ids):
#                             track_id = int(det_track_ids[det_cursor])
#                             det_cursor += 1
#                         else:
#                             track_id = -1
                        
#                         if face_img.size == 0:
#                             continue
                        
#                         # GFPGAN增强
#                         if self.gfpgan_enhancer:
#                             face_img = self.enhance_face_gfpgan(face_img)
                        
#                         # 转换为PIL并预处理
#                         face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#                         face_pil = Image.fromarray(face_rgb)
#                         face_tensor = transform(face_pil).unsqueeze(0).to(self.device)
                        
#                         # 预测
#                         with torch.no_grad():
#                             R_pred = self.model(face_tensor).cpu()[0]
                        
#                         conf = self.compute_confidence(R_pred.unsqueeze(0))[0]
                        
#                         face_results.append({
#                             'rotation_matrix': R_pred.numpy(),
#                             'confidence': float(conf),
#                             'bbox': face_info['bbox'],
#                             'use_for_training': bool(conf > self.confidence_threshold)
#                         })

#                         track_face_results.append({
#                             'frame_idx': frame_idx,
#                             'track_id': track_id,
#                             'rotation_matrix': R_pred.numpy(),
#                             'confidence': float(conf),
#                             'bbox': face_info['bbox'],
#                             'use_for_training': bool(conf > self.confidence_threshold)
#                         })
                        
#                         total_faces += 1
#                         if conf > self.confidence_threshold:
#                             high_confidence_count += 1
                    
#                     pseudo_labels[rel_path] = {'faces': face_results}
#                     track_labels[rel_path] = {'faces': track_face_results}
                    
#                 else:
#                     # 单人脸模式
#                     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
#                     # GFPGAN增强（整图）
#                     if self.gfpgan_enhancer:
#                         img_rgb = cv2.cvtColor(self.enhance_face_gfpgan(img_bgr), cv2.COLOR_BGR2RGB)
                    
#                     img_pil = Image.fromarray(img_rgb)
#                     img_tensor = transform(img_pil).unsqueeze(0).to(self.device)
                    
#                     with torch.no_grad():
#                         R_pred = self.model(img_tensor).cpu()[0]
                    
#                     conf = self.compute_confidence(R_pred.unsqueeze(0))[0]
                    
#                     pseudo_labels[rel_path] = {
#                         'rotation_matrix': R_pred.numpy(),
#                         'confidence': float(conf),
#                         'use_for_training': bool(conf > self.confidence_threshold)
#                     }

#                     track_labels[rel_path] = {
#                         'faces': [{
#                             'frame_idx': frame_to_index.get(img_path, 0),
#                             'track_id': -1,
#                             'rotation_matrix': R_pred.numpy(),
#                             'confidence': float(conf),
#                             'bbox': None,
#                             'use_for_training': bool(conf > self.confidence_threshold)
#                         }]
#                     }
                    
#                     total_faces += 1
#                     if conf > self.confidence_threshold:
#                         high_confidence_count += 1
                        
#             except Exception as e:
#                 print(f"Error processing {img_path}: {e}")
#                 continue
        
#         # 保存到文件
#         print(f"\nSaving pseudo labels to {output_file}")
#         if multi_face_mode:
#             print(f"Total frames: {len(pseudo_labels)}")
#             print(f"Total faces: {total_faces}")
#             print(f"High confidence faces: {high_confidence_count}/{total_faces} ({100*high_confidence_count/total_faces:.1f}%)" if total_faces > 0 else "No faces detected")
#         else:
#             print(f"High confidence samples: {high_confidence_count}/{len(pseudo_labels)} "
#                   f"({100*high_confidence_count/len(pseudo_labels):.1f}%)")
        
#         np.savez_compressed(output_file, **pseudo_labels)

#         # 额外保存 track 版本（不修改原格式）
#         track_output_file = output_file.replace('.npz', '_tracks.npz')
#         np.savez_compressed(track_output_file, **track_labels)
        
#         # 同时保存JSON格式的统计信息
#         stats = {
#             'total_frames': len(pseudo_labels),
#             'total_faces': total_faces,
#             'high_confidence_samples': high_confidence_count,
#             'confidence_threshold': self.confidence_threshold,
#             'high_confidence_ratio': high_confidence_count / len(pseudo_labels),
#             'track_file': track_output_file
#         }
        
#         stats_file = output_file.replace('.npz', '_stats.json')
#         with open(stats_file, 'w') as f:
#             json.dump(stats, f, indent=2)
        
#         return pseudo_labels
    
#     def generate_for_video(self, 
#                           video_path, 
#                           transform,
#                           output_file,
#                           face_detector=None,
#                           sample_rate=1,
#                           temporal_smooth=True):
#         """
#         为视频生成伪标签
        
#         Args:
#             video_path: 视频文件路径
#             transform: 图像预处理transform
#             output_file: 输出文件路径
#             face_detector: 人脸检测器（可选）
#             sample_rate: 采样率（每N帧处理一次）
#             temporal_smooth: 是否进行时序平滑
#         """
#         cap = cv2.VideoCapture(video_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         pseudo_labels = {}
#         frame_idx = 0
        
#         print(f"Processing video: {video_path}")
#         print(f"Total frames: {total_frames}, FPS: {fps}, Sample rate: {sample_rate}")
        
#         pbar = tqdm(total=total_frames // sample_rate)
        
#         previous_R = None
#         temporal_window = []
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             if frame_idx % sample_rate != 0:
#                 frame_idx += 1
#                 continue
            
#             # 如果有人脸检测器，先检测人脸
#             if face_detector is not None:
#                 faces = face_detector.detect(frame)
#                 if len(faces) == 0:
#                     frame_idx += 1
#                     continue
#                 # 使用最大的人脸
#                 face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
#                 x, y, w, h = face['box']
#                 face_img = frame[y:y+h, x:x+w]
#             else:
#                 face_img = frame
            
#             # 转换为PIL Image并预处理
#             face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#             face_pil = Image.fromarray(face_img)
#             img_tensor = transform(face_pil).unsqueeze(0)
            
#             # 预测
#             R_pred, confidence = self.generate_pseudo_label(img_tensor)
            
#             # 时序平滑
#             if temporal_smooth and previous_R is not None:
#                 # 使用滑动窗口平均
#                 temporal_window.append(R_pred)
#                 if len(temporal_window) > 5:
#                     temporal_window.pop(0)
                
#                 # 平均并重新正交化
#                 R_avg = torch.stack(temporal_window).mean(dim=0)
#                 U, S, V = torch.svd(R_avg)
#                 R_pred = torch.mm(U, V.t())
            
#             previous_R = R_pred
            
#             pseudo_labels[f"frame_{frame_idx:06d}"] = {
#                 'rotation_matrix': R_pred.numpy(),
#                 'confidence': float(confidence),
#                 'use_for_training': bool(confidence > self.confidence_threshold),
#                 'timestamp': frame_idx / fps
#             }
            
#             frame_idx += 1
#             pbar.update(1)
        
#         cap.release()
#         pbar.close()
        
#         # 保存结果
#         print(f"\nSaving pseudo labels to {output_file}")
#         np.savez_compressed(output_file, **pseudo_labels)
        
#         high_confidence_count = sum(1 for v in pseudo_labels.values() 
#                                    if v['use_for_training'])
#         print(f"High confidence frames: {high_confidence_count}/{len(pseudo_labels)}")
        
#         return pseudo_labels

import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


class IoUTracker:
    """轻量级 IoU 跟踪器（跨帧分配 track_id）。"""
    def __init__(self, iou_threshold=0.3, max_missed=8):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}  # tid -> {'bbox': [x1,y1,x2,y2], 'last_frame': int}

    @staticmethod
    def _iou(b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1, iy1 = max(x11, x21), max(y11, y21)
        ix2, iy2 = min(x12, x22), min(y12, y22)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        a1 = max(1, (x12 - x11) * (y12 - y11))
        a2 = max(1, (x22 - x21) * (y22 - y21))
        return inter / float(a1 + a2 - inter + 1e-6)

    def _prune(self, frame_idx):
        dead = []
        for tid, t in self.tracks.items():
            if frame_idx - t['last_frame'] > self.max_missed:
                dead.append(tid)
        for tid in dead:
            self.tracks.pop(tid, None)

    def update(self, detections, frame_idx):
        self._prune(frame_idx)
        track_ids = [-1] * len(detections)
        if len(detections) == 0:
            return track_ids

        candidates = []
        for det_idx, det in enumerate(detections):
            for tid, tr in self.tracks.items():
                iou = self._iou(det, tr['bbox'])
                if iou >= self.iou_threshold:
                    candidates.append((iou, det_idx, tid))

        candidates.sort(reverse=True, key=lambda x: x[0])
        used_det = set()
        used_tid = set()

        for _, det_idx, tid in candidates:
            if det_idx in used_det or tid in used_tid:
                continue
            track_ids[det_idx] = tid
            used_det.add(det_idx)
            used_tid.add(tid)
            self.tracks[tid]['bbox'] = detections[det_idx]
            self.tracks[tid]['last_frame'] = frame_idx

        for det_idx, det in enumerate(detections):
            if det_idx in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {'bbox': det, 'last_frame': frame_idx}
            track_ids[det_idx] = tid

        return track_ids


class PseudoLabelGenerator:
    def __init__(
        self,
        model,
        models=None,
        model_names=None,
        device='cuda',
        confidence_threshold=0.8,
        use_tta=True,
        tta_rotations=(0, 5, -5),
        save_visualizations=False,
        yolo_model_path=None,
        use_gfpgan=False,
        gfpgan_model_path=None,
        face_det_threshold=0.5,
        min_face_size=20,
        expand_margin=0.2,
        use_consistency_filter=True,
        consistency_threshold_deg=8.0,
        cross_model_threshold_deg=6.0,
        save_face_crops=False,
        face_crops_dir=None,
        face_crop_ext='jpg',
        imgsz=1536,
        yolo_iou=0.45,
        max_det=1000,
        use_tile=False,
        tile_size=960,
        tile_overlap=0.25,
        merge_iou=0.35
    ):
        self.model = model
        self.models = list(models) if models is not None else [model]
        self.model_names = list(model_names) if model_names is not None else [
            f"model_{idx}" for idx in range(len(self.models))
        ]
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.use_tta = use_tta
        self.tta_rotations = list(tta_rotations)
        self.save_visualizations = save_visualizations
        self.face_det_threshold = face_det_threshold
        self.min_face_size = min_face_size
        self.expand_margin = expand_margin
        self.use_consistency_filter = use_consistency_filter
        self.consistency_threshold_deg = consistency_threshold_deg
        self.cross_model_threshold_deg = cross_model_threshold_deg
        self.save_face_crops = save_face_crops
        self.face_crops_dir = face_crops_dir
        self.face_crop_ext = face_crop_ext.lower()

        # High-recall YOLO settings for classroom-scale images.
        self.imgsz = int(imgsz) if imgsz is not None else 1536
        self.yolo_iou = float(yolo_iou)
        self.max_det = int(max_det)
        self.use_tile = bool(use_tile)
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)
        self.merge_iou = float(merge_iou)

        for current_model in self.models:
            current_model.eval()
            current_model.to(device)

        if self.save_face_crops and self.face_crops_dir:
            os.makedirs(self.face_crops_dir, exist_ok=True)

        self.yolo_detector = None
        if yolo_model_path:
            if not os.path.exists(yolo_model_path):
                print(f"Warning: YOLO model not found at {yolo_model_path}")
            else:
                try:
                    from ultralytics import YOLO
                    self.yolo_detector = YOLO(yolo_model_path)
                    print(f"✓ Loaded YOLO face detector from {yolo_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load YOLO detector: {e}")

        self.gfpgan_enhancer = None
        if use_gfpgan:
            if not gfpgan_model_path:
                print("Warning: GFPGAN enabled but model path is None")
            elif not os.path.exists(gfpgan_model_path):
                print(f"Warning: GFPGAN model not found at {gfpgan_model_path}")
            else:
                try:
                    from gfpgan import GFPGANer
                    self.gfpgan_enhancer = GFPGANer(
                        model_path=gfpgan_model_path,
                        upscale=1,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=None,
                        device=device
                    )
                    print(f"✓ Loaded GFPGAN enhancer from {gfpgan_model_path}")
                except Exception as e:
                    print(f"Warning: Failed to load GFPGAN: {e}")

    @staticmethod
    def _wrap_abs_diff_deg(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return np.abs(d)

    def compute_confidence(self, R_pred):
        """
        基于正交性的辅助分数。
        注意：单独使用时容易饱和，因此后续会与一致性分数融合。
        """
        RtR = torch.bmm(R_pred.transpose(1, 2), R_pred)
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype).unsqueeze(0)
        ortho_error = ((RtR - I) ** 2).sum(dim=(1, 2))
        confidence = torch.exp(-ortho_error * 10)
        return confidence.detach().cpu().numpy()

    def compute_quality_confidence(self, ortho_conf, consistency_score_deg, model_consistency_score_deg=0.0, ensemble_size=1):
        """
        更可靠的伪标签质量分数。

        核心思想：
        - `ortho_conf` 只保证输出像旋转矩阵，不保证姿态正确；
        - `consistency_score_deg` 越小，说明在亮度/模糊增强下预测更稳定；
        - 训练时使用两者融合后的 `confidence`，避免几乎所有样本都接近 1.0。
        """
        ortho_conf = float(np.clip(ortho_conf, 0.0, 1.0))
        consistency_score_deg = max(0.0, float(consistency_score_deg))

        if self.use_consistency_filter and self.consistency_threshold_deg is not None:
            scale = max(6.0, float(self.consistency_threshold_deg) * 4.0)
        else:
            scale = 24.0

        consistency_conf = float(np.exp(-consistency_score_deg / scale))

        if int(ensemble_size) > 1:
            model_scale = max(4.0, float(self.cross_model_threshold_deg) * 2.5)
            model_consistency_conf = float(np.exp(-max(0.0, float(model_consistency_score_deg)) / model_scale))
            quality_conf = 0.15 * ortho_conf + 0.35 * consistency_conf + 0.50 * model_consistency_conf
        else:
            quality_conf = 0.2 * ortho_conf + 0.8 * consistency_conf
        return float(np.clip(quality_conf, 0.0, 1.0))

    @staticmethod
    def _orthogonalize_rotation(R):
        U, _, V = torch.svd(R)
        return torch.mm(U, V.t())

    @staticmethod
    def _compute_blur_score(face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _mean_angle_diff_to_anchor(self, yprs, anchor):
        diffs = []
        for current in yprs:
            d = self._wrap_abs_diff_deg(current, anchor)
            diffs.append(float(d.mean()))
        return float(np.mean(diffs)) if diffs else 0.0

    def rotate_image(self, image, angle):
        if angle == 0:
            return image
        if isinstance(image, Image.Image):
            return image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def enhance_face_gfpgan(self, face_img):
        if self.gfpgan_enhancer is None:
            return face_img
        try:
            _, _, output = self.gfpgan_enhancer.enhance(
                face_img, has_aligned=False, only_center_face=False, paste_back=True
            )
            return output if output is not None else face_img
        except Exception as e:
            print(f"GFPGAN enhancement failed: {e}")
            return face_img


    @staticmethod
    def _clip_bbox_to_image(bbox, img_shape):
        """Clip bbox to valid image coordinates."""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return [x1, y1, x2, y2]

    def _expand_and_clip_bbox(self, bbox, img_shape):
        clipped = self._clip_bbox_to_image(bbox, img_shape)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped
        bw = max(2, x2 - x1)
        bh = max(2, y2 - y1)
        margin = float(self.expand_margin)
        expanded = [
            x1 - int(bw * margin),
            y1 - int(bh * margin),
            x2 + int(bw * margin),
            y2 + int(bh * margin),
        ]
        return self._clip_bbox_to_image(expanded, img_shape)

    @staticmethod
    def _bbox_iou(b1, b2):
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2
        ix1, iy1 = max(x11, x21), max(y11, y21)
        ix2, iy2 = min(x12, x22), min(y12, y22)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        a1 = max(1, (x12 - x11) * (y12 - y11))
        a2 = max(1, (x22 - x21) * (y22 - y21))
        return inter / float(a1 + a2 - inter + 1e-6)

    def _nms_faces(self, faces, iou_thr=None):
        """Remove duplicate boxes from full-image and tiled YOLO passes."""
        if not faces:
            return []
        thr = self.merge_iou if iou_thr is None else float(iou_thr)
        faces = sorted(faces, key=lambda f: float(f.get('det_conf', 0.0)), reverse=True)
        kept = []
        for face in faces:
            if all(self._bbox_iou(face['bbox'], k['bbox']) < thr for k in kept):
                kept.append(face)
        kept.sort(key=lambda f: (f['bbox'][1], f['bbox'][0]))
        return kept

    def _face_record_from_bbox(self, bbox, img_shape, det_conf=1.0, source='yolo'):
        bbox2 = self._expand_and_clip_bbox(bbox, img_shape)
        if bbox2 is None:
            return None
        x1, y1, x2, y2 = bbox2
        if min(x2 - x1, y2 - y1) < int(self.min_face_size):
            return None
        return {
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'det_conf': float(det_conf),
            'face_w': int(x2 - x1),
            'face_h': int(y2 - y1),
            'det_source': source,
        }

    def _detect_faces_yolo_single(self, img_bgr, offset_x=0, offset_y=0, full_shape=None, source='yolo'):
        """Run YOLO on a full image or tile, then map boxes back to full-image coordinates."""
        faces = []
        if self.yolo_detector is None:
            return faces
        target_shape = full_shape if full_shape is not None else img_bgr.shape
        try:
            results = self.yolo_detector(
                img_bgr,
                verbose=False,
                conf=float(self.face_det_threshold),
                iou=float(self.yolo_iou),
                imgsz=int(self.imgsz),
                max_det=int(self.max_det),
            )
        except TypeError:
            results = self.yolo_detector(img_bgr, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.face_det_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [
                    x1 + int(offset_x),
                    y1 + int(offset_y),
                    x2 + int(offset_x),
                    y2 + int(offset_y),
                ]
                rec = self._face_record_from_bbox(bbox, target_shape, det_conf=conf, source=source)
                if rec is not None:
                    faces.append(rec)
        return faces


    def detect_faces(self, img_bgr):
        """
        High-recall face detection for classroom images.

        Returns:
        [
          {'bbox': [x1,y1,x2,y2], 'det_conf': 0.93, 'face_w': ..., 'face_h': ..., 'det_source': ...},
          ...
        ]
        """
        h, w = img_bgr.shape[:2]

        if self.yolo_detector is None:
            return [{
                'bbox': [0, 0, w, h],
                'det_conf': 1.0,
                'face_w': w,
                'face_h': h,
                'det_source': 'whole_image',
            }]

        faces = self._detect_faces_yolo_single(
            img_bgr,
            offset_x=0,
            offset_y=0,
            full_shape=img_bgr.shape,
            source='yolo_full',
        )

        if self.use_tile:
            tile_size = max(64, int(self.tile_size))
            overlap = min(max(float(self.tile_overlap), 0.0), 0.85)
            step = max(16, int(tile_size * (1.0 - overlap)))

            y_starts = list(range(0, h, step))
            x_starts = list(range(0, w, step))
            if y_starts and y_starts[-1] + tile_size < h:
                y_starts.append(max(0, h - tile_size))
            if x_starts and x_starts[-1] + tile_size < w:
                x_starts.append(max(0, w - tile_size))

            for y0 in y_starts:
                for x0 in x_starts:
                    x1 = int(x0)
                    y1 = int(y0)
                    x2 = min(w, x1 + tile_size)
                    y2 = min(h, y1 + tile_size)
                    if x2 - x1 < 32 or y2 - y1 < 32:
                        continue
                    tile = img_bgr[y1:y2, x1:x2]
                    faces.extend(
                        self._detect_faces_yolo_single(
                            tile,
                            offset_x=x1,
                            offset_y=y1,
                            full_shape=img_bgr.shape,
                            source='yolo_tile',
                        )
                    )

        return self._nms_faces(faces, iou_thr=self.merge_iou)

    @torch.no_grad()
    def _predict_rotation(self, model, image_tensor):
        return model(image_tensor.to(self.device)).cpu()[0]

    def _rotation_matrix_to_euler(self, R):
        """
        返回 pitch, yaw, roll（度）
        """
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float()
        if R.ndim == 2:
            R = R.unsqueeze(0)

        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        singular = sy < 1e-6

        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        x = torch.where(singular, torch.atan2(-R[:, 1, 2], R[:, 1, 1]), x)
        z = torch.where(singular, torch.zeros_like(z), z)

        pitch = x * 180.0 / np.pi
        yaw = y * 180.0 / np.pi
        roll = z * 180.0 / np.pi
        return float(pitch[0]), float(yaw[0]), float(roll[0])

    def _predict_face_with_augments(self, face_bgr, transform):
        """
        返回：
        rotation_matrix(np.ndarray shape=(3,3)),
        quality_conf(float),
        ortho_conf(float),
        consistency_score_deg(float),
        tta_consistency_deg(float),
        model_consistency_deg(float),
        ensemble_size(int),
        yaw(float), pitch(float), roll(float)
        """
        variants = [face_bgr]

        if self.use_tta:
            variants.append(cv2.convertScaleAbs(face_bgr, alpha=0.9, beta=0))
            variants.append(cv2.convertScaleAbs(face_bgr, alpha=1.1, beta=0))
            variants.append(cv2.GaussianBlur(face_bgr, (5, 5), 1.0))

        model_rotations = []
        model_yprs = []
        tta_consistency_scores = []

        for model in self.models:
            preds = []
            yprs = []

            for img in variants:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                x = transform(img_pil).unsqueeze(0).to(self.device)
                R_pred = self._predict_rotation(model, x)
                preds.append(R_pred)
                pitch, yaw, roll = self._rotation_matrix_to_euler(R_pred)
                yprs.append(np.array([yaw, pitch, roll], dtype=np.float32))

            preds = torch.stack(preds, dim=0)
            R_avg = preds.mean(dim=0)
            R_model = self._orthogonalize_rotation(R_avg)

            yprs = np.stack(yprs, axis=0)
            base = yprs[0]
            diffs = []
            for i in range(1, len(yprs)):
                d = self._wrap_abs_diff_deg(yprs[i], base)
                diffs.append(float(d.mean()))
            tta_consistency_scores.append(float(np.mean(diffs)) if len(diffs) > 0 else 0.0)

            pitch, yaw, roll = self._rotation_matrix_to_euler(R_model)
            model_rotations.append(R_model)
            model_yprs.append(np.array([yaw, pitch, roll], dtype=np.float32))

        model_rotations = torch.stack(model_rotations, dim=0)
        R_final = self._orthogonalize_rotation(model_rotations.mean(dim=0))
        ortho_conf = float(self.compute_confidence(R_final.unsqueeze(0))[0])

        model_yprs = np.stack(model_yprs, axis=0)
        pitch, yaw, roll = self._rotation_matrix_to_euler(R_final)
        ensemble_ypr = np.array([yaw, pitch, roll], dtype=np.float32)

        tta_consistency_score = float(np.mean(tta_consistency_scores)) if tta_consistency_scores else 0.0
        model_consistency_score = self._mean_angle_diff_to_anchor(model_yprs, ensemble_ypr) if len(model_yprs) > 1 else 0.0

        if len(model_yprs) > 1:
            consistency_score = 0.4 * tta_consistency_score + 0.6 * model_consistency_score
        else:
            consistency_score = tta_consistency_score

        quality_conf = self.compute_quality_confidence(
            ortho_conf,
            consistency_score,
            model_consistency_score_deg=model_consistency_score,
            ensemble_size=len(self.models)
        )

        return (
            R_final.numpy(),
            quality_conf,
            ortho_conf,
            consistency_score,
            tta_consistency_score,
            model_consistency_score,
            len(self.models),
            yaw,
            pitch,
            roll,
        )

    def _save_face_crop(self, face_img, img_path, frame_idx, local_face_idx):
        if not (self.save_face_crops and self.face_crops_dir):
            return None

        stem = os.path.splitext(os.path.basename(img_path))[0]
        name = f"{stem}_f{frame_idx:06d}_face{local_face_idx:02d}.{self.face_crop_ext}"
        save_path = os.path.join(self.face_crops_dir, name)

        if self.face_crop_ext in ("jpg", "jpeg"):
            cv2.imwrite(save_path, face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            cv2.imwrite(save_path, face_img)

        return name

    def generate_for_dataset(
        self,
        image_paths,
        transform,
        output_file,
        batch_size=32,
        multi_face_mode=True
    ):
        image_paths = sorted(image_paths)

        pseudo_labels = {}       # frame-level
        track_labels = {}        # frame-level + track
        face_level_labels = {}   # face-level，后续训练推荐使用这个

        total_faces = 0
        high_conf_faces = 0
        high_quality_faces = 0
        consistency_pass_faces = 0
        cross_model_pass_faces = 0

        tracker = IoUTracker(iou_threshold=0.3, max_missed=8) if (multi_face_mode and self.yolo_detector) else None
        frame_to_index = {p: i for i, p in enumerate(image_paths)}

        print(f"Generating pseudo labels for {len(image_paths)} images...")
        print(f"Multi-face mode: {multi_face_mode}, YOLO: {self.yolo_detector is not None}, GFPGAN: {self.gfpgan_enhancer is not None}")
        print(f"Save face crops: {self.save_face_crops}, dir: {self.face_crops_dir}")
        print(f"Use consistency filter: {self.use_consistency_filter}, thr={self.consistency_threshold_deg:.2f} deg")

        for img_path in tqdm(image_paths):
            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                rel_path = os.path.relpath(img_path)
                frame_idx = frame_to_index.get(img_path, 0)

                if multi_face_mode and self.yolo_detector:
                    faces = self.detect_faces(img_bgr)
                    face_results = []
                    track_face_results = []

                    det_bboxes = [f['bbox'] for f in faces]
                    det_track_ids = tracker.update(det_bboxes, frame_idx) if tracker is not None and len(det_bboxes) > 0 else []

                    for local_face_idx, face_info in enumerate(faces):
                        x1, y1, x2, y2 = face_info['bbox']
                        face_img = img_bgr[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue

                        track_id = int(det_track_ids[local_face_idx]) if local_face_idx < len(det_track_ids) else -1

                        if self.gfpgan_enhancer is not None:
                            face_img = self.enhance_face_gfpgan(face_img)

                        face_path = self._save_face_crop(face_img, img_path, frame_idx, local_face_idx)
                        blur_score = self._compute_blur_score(face_img)

                        (
                            R_pred,
                            quality_conf,
                            ortho_conf,
                            consistency_score,
                            tta_consistency_score,
                            model_consistency_score,
                            ensemble_size,
                            yaw,
                            pitch,
                            roll,
                        ) = self._predict_face_with_augments(
                            face_img, transform
                        )

                        consistency_ok = (consistency_score <= self.consistency_threshold_deg) if self.use_consistency_filter else True
                        cross_model_ok = (
                            True if int(ensemble_size) <= 1
                            else float(model_consistency_score) <= float(self.cross_model_threshold_deg)
                        )
                        conf_ok = quality_conf >= self.confidence_threshold
                        use_for_training = bool(conf_ok and consistency_ok and cross_model_ok)

                        if conf_ok:
                            high_conf_faces += 1
                        if consistency_ok:
                            consistency_pass_faces += 1
                        if cross_model_ok:
                            cross_model_pass_faces += 1
                        if use_for_training:
                            high_quality_faces += 1

                        face_record = {
                            'face_id': int(local_face_idx),
                            'rotation_matrix': R_pred,
                            'confidence': float(quality_conf),
                            'ortho_confidence': float(ortho_conf),
                            'consistency_score_deg': float(consistency_score),
                            'tta_consistency_deg': float(tta_consistency_score),
                            'model_consistency_deg': float(model_consistency_score),
                            'ensemble_size': int(ensemble_size),
                            'bbox': face_info['bbox'],
                            'det_confidence': float(face_info['det_conf']),
                            'det_source': str(face_info.get('det_source', 'unknown')),
                            'face_w': int(face_info['face_w']),
                            'face_h': int(face_info['face_h']),
                            'blur_score': float(blur_score),
                            'track_id': int(track_id),
                            'yaw': float(yaw),
                            'pitch': float(pitch),
                            'roll': float(roll),
                            'face_path': face_path,
                            'cross_model_ok': bool(cross_model_ok),
                            'use_for_training': use_for_training
                        }

                        face_results.append(face_record)

                        track_face_results.append({
                            'frame_idx': int(frame_idx),
                            **face_record
                        })

                        if face_path is not None:
                            face_level_labels[face_path] = {
                                'rotation_matrix': R_pred,
                                'confidence': float(quality_conf),
                                'ortho_confidence': float(ortho_conf),
                                'consistency_score_deg': float(consistency_score),
                                'tta_consistency_deg': float(tta_consistency_score),
                                'model_consistency_deg': float(model_consistency_score),
                                'ensemble_size': int(ensemble_size),
                                'det_confidence': float(face_info['det_conf']),
                                'det_source': str(face_info.get('det_source', 'unknown')),
                                'track_id': int(track_id),
                                'bbox': face_info['bbox'],
                                'face_w': int(face_info['face_w']),
                                'face_h': int(face_info['face_h']),
                                'blur_score': float(blur_score),
                                'yaw': float(yaw),
                                'pitch': float(pitch),
                                'roll': float(roll),
                                'cross_model_ok': bool(cross_model_ok),
                                'use_for_training': use_for_training
                            }

                        total_faces += 1

                    pseudo_labels[rel_path] = {'faces': face_results}
                    track_labels[rel_path] = {'faces': track_face_results}

                else:
                    if self.gfpgan_enhancer is not None:
                        img_bgr = self.enhance_face_gfpgan(img_bgr)

                    face_path = self._save_face_crop(img_bgr, img_path, frame_idx, 0)
                    blur_score = self._compute_blur_score(img_bgr)
                    (
                        R_pred,
                        quality_conf,
                        ortho_conf,
                        consistency_score,
                        tta_consistency_score,
                        model_consistency_score,
                        ensemble_size,
                        yaw,
                        pitch,
                        roll,
                    ) = self._predict_face_with_augments(
                        img_bgr, transform
                    )
                    consistency_ok = (consistency_score <= self.consistency_threshold_deg) if self.use_consistency_filter else True
                    cross_model_ok = (
                        True if int(ensemble_size) <= 1
                        else float(model_consistency_score) <= float(self.cross_model_threshold_deg)
                    )
                    conf_ok = quality_conf >= self.confidence_threshold
                    use_for_training = bool(conf_ok and consistency_ok and cross_model_ok)

                    if conf_ok:
                        high_conf_faces += 1
                    if consistency_ok:
                        consistency_pass_faces += 1
                    if cross_model_ok:
                        cross_model_pass_faces += 1
                    if use_for_training:
                        high_quality_faces += 1

                    rec = {
                        'rotation_matrix': R_pred,
                        'confidence': float(quality_conf),
                        'ortho_confidence': float(ortho_conf),
                        'consistency_score_deg': float(consistency_score),
                        'tta_consistency_deg': float(tta_consistency_score),
                        'model_consistency_deg': float(model_consistency_score),
                        'ensemble_size': int(ensemble_size),
                        'blur_score': float(blur_score),
                        'yaw': float(yaw),
                        'pitch': float(pitch),
                        'roll': float(roll),
                        'face_path': face_path,
                        'cross_model_ok': bool(cross_model_ok),
                        'use_for_training': use_for_training
                    }

                    pseudo_labels[rel_path] = rec
                    track_labels[rel_path] = {
                        'faces': [{
                            'frame_idx': int(frame_idx),
                            'track_id': -1,
                            **rec,
                            'bbox': None,
                            'det_confidence': 1.0,
                            'face_w': int(img_bgr.shape[1]),
                            'face_h': int(img_bgr.shape[0]),
                        }]
                    }

                    if face_path is not None:
                        face_level_labels[face_path] = {
                            'rotation_matrix': R_pred,
                            'confidence': float(quality_conf),
                            'ortho_confidence': float(ortho_conf),
                            'consistency_score_deg': float(consistency_score),
                            'tta_consistency_deg': float(tta_consistency_score),
                            'model_consistency_deg': float(model_consistency_score),
                            'ensemble_size': int(ensemble_size),
                            'det_confidence': 1.0,
                            'track_id': -1,
                            'bbox': None,
                            'face_w': int(img_bgr.shape[1]),
                            'face_h': int(img_bgr.shape[0]),
                            'blur_score': float(blur_score),
                            'yaw': float(yaw),
                            'pitch': float(pitch),
                            'roll': float(roll),
                            'cross_model_ok': bool(cross_model_ok),
                            'use_for_training': use_for_training
                        }

                    total_faces += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        track_length_map = {}
        for rel_path, data in track_labels.items():
            faces = data.get('faces', [])
            for face in faces:
                track_id = int(face.get('track_id', -1))
                if track_id < 0:
                    continue
                track_length_map[track_id] = track_length_map.get(track_id, 0) + 1

        for rel_path, data in pseudo_labels.items():
            faces = data.get('faces')
            if isinstance(faces, list):
                for face in faces:
                    track_id = int(face.get('track_id', -1))
                    face['track_length'] = int(track_length_map.get(track_id, 1 if track_id >= 0 else 0))

        for rel_path, data in track_labels.items():
            faces = data.get('faces')
            if isinstance(faces, list):
                for face in faces:
                    track_id = int(face.get('track_id', -1))
                    face['track_length'] = int(track_length_map.get(track_id, 1 if track_id >= 0 else 0))

        for face_path, face in face_level_labels.items():
            track_id = int(face.get('track_id', -1))
            face['track_length'] = int(track_length_map.get(track_id, 1 if track_id >= 0 else 0))

        print(f"\nSaving pseudo labels to {output_file}")
        np.savez_compressed(output_file, **pseudo_labels)

        track_output_file = output_file.replace('.npz', '_tracks.npz')
        np.savez_compressed(track_output_file, **track_labels)

        face_output_file = output_file.replace('.npz', '_faces.npz')
        np.savez_compressed(face_output_file, **face_level_labels)

        stats = {
            'total_frames': int(len(pseudo_labels)),
            'total_faces': int(total_faces),
            'high_confidence_faces': int(high_conf_faces),
            'consistency_pass_faces': int(consistency_pass_faces),
            'cross_model_pass_faces': int(cross_model_pass_faces),
            'high_quality_faces': int(high_quality_faces),
            'confidence_threshold': float(self.confidence_threshold),
            'consistency_threshold_deg': float(self.consistency_threshold_deg),
            'cross_model_threshold_deg': float(self.cross_model_threshold_deg),
            'ensemble_size': int(len(self.models)),
            'model_names': list(self.model_names),
            'high_confidence_ratio': float(high_conf_faces / total_faces) if total_faces > 0 else 0.0,
            'consistency_pass_ratio': float(consistency_pass_faces / total_faces) if total_faces > 0 else 0.0,
            'cross_model_pass_ratio': float(cross_model_pass_faces / total_faces) if total_faces > 0 else 0.0,
            'high_quality_ratio': float(high_quality_faces / total_faces) if total_faces > 0 else 0.0,
            'track_file': track_output_file,
            'face_file': face_output_file,
            'face_crops_dir': self.face_crops_dir,
        }

        stats_file = output_file.replace('.npz', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Total frames: {len(pseudo_labels)}")
        print(f"Total faces: {total_faces}")
        print(f"High confidence faces: {high_conf_faces}/{total_faces}")
        print(f"Consistency pass faces: {consistency_pass_faces}/{total_faces}")
        print(f"Cross-model pass faces: {cross_model_pass_faces}/{total_faces}")
        print(f"High quality faces (usable): {high_quality_faces}/{total_faces}")
        print(f"Frame-level file: {output_file}")
        print(f"Track file: {track_output_file}")
        print(f"Face-level file: {face_output_file}")

        return pseudo_labels
