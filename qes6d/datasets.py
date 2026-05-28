import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import utils


def get_list_from_filenames(file_path):
    with open(file_path) as file:
        return file.read().splitlines()


class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext=".jpg", annot_ext=".mat", image_mode="RGB"):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.names = get_list_from_filenames(filename_path)
        self.image_mode = image_mode

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(os.path.join(self.data_dir, name + self.img_ext)).convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, name + self.annot_ext)
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])
        crop_ratio = 0.20
        x_min -= 2 * crop_ratio * abs(x_max - x_min)
        y_min -= 2 * crop_ratio * abs(y_max - y_min)
        x_max += 2 * crop_ratio * abs(x_max - x_min)
        y_max += 0.6 * crop_ratio * abs(y_max - y_min)
        image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        pitch, yaw, roll = utils.get_ypr_from_mat(mat_path)
        rotation = utils.get_R(pitch, yaw, roll)
        labels = torch.FloatTensor([yaw, pitch, roll])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(rotation), labels, name

    def __len__(self):
        return len(self.names)


def get_dataset(dataset, data_dir, filename_list, transformations):
    if dataset != "AFLW2000":
        raise ValueError("The minimal QES6D package currently supports AFLW2000 testing only.")
    return AFLW2000(data_dir, filename_list, transformations)
