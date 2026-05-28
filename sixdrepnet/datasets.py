import os

import numpy as np
import cv2
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as Rsc
import utils
import glob


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

    
class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
     
        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])


        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter

import utils  # 你项目里原有的 utils.get_R

# class BIWI(Dataset):
#     """
#     NPZ keys:
#       - paths: (N,) object
#       - R    : (N,3,3) float32  (GT rotation matrix, already in repo convention)
#       - pose : (N,3)   float32  [yaw,pitch,roll] deg (optional, derived)
#       - meta : dict(root, abs_paths,...)
#     """
#     def __init__(self, data_dir, filename_path, transform, image_mode="RGB", train_mode=True):
#         self.data_dir = data_dir
#         self.transform = transform
#         self.image_mode = image_mode
#         self.train_mode = train_mode

#         d = np.load(filename_path, allow_pickle=True)
#         for k in ["paths", "R"]:
#             if k not in d:
#                 raise RuntimeError(f"NPZ missing key '{k}', got keys={list(d.keys())}")

#         self.paths = d["paths"]
#         self.R_gt = d["R"].astype(np.float32)  # (N,3,3)

#         self.meta = d["meta"].item() if "meta" in d else {}
#         self.root = os.path.abspath(self.meta.get("root", data_dir))

#         self.length = len(self.paths)
#         if self.R_gt.shape[0] != self.length:
#             raise RuntimeError(f"len(paths)={self.length} but R.shape[0]={self.R_gt.shape[0]}")

#     def _resolve_path(self, p: str) -> str:
#         if os.path.isabs(p):
#             return p
#         return os.path.join(self.root, p)

#     def __getitem__(self, index):
#         img_path = self._resolve_path(str(self.paths[index]))
#         img = Image.open(img_path).convert(self.image_mode)

#         R = torch.from_numpy(self.R_gt[index])  # (3,3)

#         # 从 R 统一算 euler（rad），注意 utils 输出通常是 [pitch,yaw,roll]
#         euler = utils.compute_euler_angles_from_rotation_matrices(R.unsqueeze(0))[0]
#         pitch, yaw, roll = float(euler[0]), float(euler[1]), float(euler[2])

#         if self.train_mode:
#             # Flip: 你原来逻辑是 yaw/roll 取反
#             if np.random.random_sample() < 0.5:
#                 yaw = -yaw
#                 roll = -roll
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)

#             if np.random.random_sample() < 0.05:
#                 img = img.filter(ImageFilter.BLUR)

#             # 重要：flip 后 R 也应该跟着变（否则图像翻了但 GT 没翻，训练会崩）
#             # 最简单：flip 后重新用 get_R 生成 R
#             R = torch.from_numpy(np.array(utils.get_R(pitch, yaw, roll), dtype=np.float32))

#         if self.transform is not None:
#             img = self.transform(img)

#         cont_labels = torch.FloatTensor([yaw, pitch, roll])  # 与你现有代码保持一致：[yaw,pitch,roll]
#         return img, R.float(), cont_labels, f"biwi_{index:05d}"

#     def __len__(self):
#         return self.length

class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path, allow_pickle=True)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi

        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        # Return a simple index-based name instead of the image data
        return img, torch.FloatTensor(R), cont_labels, f"biwi_{index:05d}"

    def __len__(self):
        # 15,667
        return self.length

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] # * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2] # * 180 / np.pi

        # Gray images

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        #mu, sigma = 0, 0.01 
        #noise = np.random.normal(mu, sigma, [3,3])
        #print(noise) 

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)#+ noise
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        #labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img = self.transform(img)

        return img,  torch.FloatTensor(R),cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

def getDataset(dataset, data_dir, filename_list, transformations, train_mode = True):
    if dataset == 'Pose_300W_LP':
            pose_dataset = Pose_300W_LP(
                data_dir, filename_list, transformations)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000(
            data_dir, filename_list, transformations)
    # elif dataset == 'BIWI':
    #     pose_dataset = BIWI(
    #         data_dir, filename_list, transformations, train_mode= train_mode)
    elif dataset == 'BIWI':
        # filename_list:
        # - "" / None：扫描 data_dir/faces_0 下全部 *_rgb.png
        # - txt：每行一个 rgb 路径（可绝对/相对 data_dir/相对 faces_0）
        pose_dataset = BIWI(
            data_dir, filename_list, transformations, train_mode= train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(
            data_dir, filename_list, transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(
            data_dir, filename_list, transformations)
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset
