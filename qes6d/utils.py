import math

import cv2
import numpy as np
import scipy.io as sio
import torch


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    x1 = size * (math.cos(yaw) * math.cos(roll)) + tdx
    y1 = size * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw)) + tdy
    x2 = size * (-math.cos(yaw) * math.sin(roll)) + tdx
    y2 = size * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll)) + tdy
    x3 = size * math.sin(yaw) + tdx
    y3 = size * (-math.cos(yaw) * math.sin(pitch)) + tdy
    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 4)
    return img


def get_ypr_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    return mat["Pose_Para"][0][:3]


def get_pt2d_from_mat(mat_path):
    mat = sio.loadmat(mat_path)
    return mat["pt2d"]


def normalize_vector(vector):
    batch = vector.shape[0]
    vector_magnitude = torch.sqrt(vector.pow(2).sum(1))
    eps = torch.full((1,), 1e-8, device=vector.device, dtype=vector.dtype)
    vector_magnitude = torch.max(vector_magnitude, eps)
    vector_magnitude = vector_magnitude.view(batch, 1).expand(batch, vector.shape[1])
    return vector / vector_magnitude


def cross_product(first, second):
    batch = first.shape[0]
    i = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
    j = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
    k = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = normalize_vector(x_raw)
    z = normalize_vector(cross_product(x, y_raw))
    y = cross_product(z, x)
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    return torch.cat((x, y, z), 2)


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    batch = rotation_matrices.shape[0]
    matrix = rotation_matrices
    sy = torch.sqrt(matrix[:, 0, 0] * matrix[:, 0, 0] + matrix[:, 1, 0] * matrix[:, 1, 0])
    singular = (sy < 1e-6).float()
    x = torch.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
    y = torch.atan2(-matrix[:, 2, 0], sy)
    z = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])
    xs = torch.atan2(-matrix[:, 1, 2], matrix[:, 1, 1])
    ys = torch.atan2(-matrix[:, 2, 0], sy)
    zs = matrix[:, 1, 0] * 0
    out = torch.zeros(batch, 3, device=rotation_matrices.device, dtype=rotation_matrices.dtype)
    out[:, 0] = x * (1 - singular) + xs * singular
    out[:, 1] = y * (1 - singular) + ys * singular
    out[:, 2] = z * (1 - singular) + zs * singular
    return out


def get_R(x, y, z):
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz.dot(ry.dot(rx))
