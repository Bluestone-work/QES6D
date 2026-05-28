import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from qes6d.datasets import get_dataset
from qes6d.model import DEFAULT_WEIGHTS, QES6DNet
from qes6d.utils import compute_euler_angles_from_rotation_matrices


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QES6D on AFLW2000")
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--filename_list", required=True, type=str)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    return parser.parse_args()


def angular_error(pred_deg, gt_deg):
    diff = (pred_deg - gt_deg + 180.0) % 360.0 - 180.0
    return torch.abs(diff)


def main():
    args = parse_args()
    cudnn.enabled = True
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = QES6DNet(deploy=True).to(device)
    state = torch.load(args.weights, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = get_dataset("AFLW2000", args.data_dir, args.filename_list, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    total = 0
    yaw_error = 0.0
    pitch_error = 0.0
    roll_error = 0.0
    with torch.no_grad():
        for images, _rotation_labels, cont_labels, _names in loader:
            images = images.to(device, non_blocking=True)
            cont_labels = cont_labels.to(device, non_blocking=True)
            batch_size = cont_labels.size(0)
            total += batch_size
            gt_yaw = cont_labels[:, 0].float() * 180.0 / np.pi
            gt_pitch = cont_labels[:, 1].float() * 180.0 / np.pi
            gt_roll = cont_labels[:, 2].float() * 180.0 / np.pi
            prediction = model(images)
            euler_deg = compute_euler_angles_from_rotation_matrices(prediction) * 180.0 / np.pi
            pred_pitch = euler_deg[:, 0]
            pred_yaw = euler_deg[:, 1]
            pred_roll = euler_deg[:, 2]
            pitch_error += angular_error(pred_pitch, gt_pitch).sum().item()
            yaw_error += angular_error(pred_yaw, gt_yaw).sum().item()
            roll_error += angular_error(pred_roll, gt_roll).sum().item()
    if total == 0:
        raise RuntimeError("No AFLW2000 samples were evaluated.")
    print("Yaw: %.4f, Pitch: %.4f, Roll: %.4f, MAE: %.4f" % (yaw_error / total, pitch_error / total, roll_error / total, (yaw_error + pitch_error + roll_error) / (3 * total)))


if __name__ == "__main__":
    main()
