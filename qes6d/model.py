from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from .backbone.repvgg import get_RepVGG_func_by_name
from .utils import compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d, draw_axis


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "QES6D_300W_LP_AFLW2000.pth"


class QES6DNet(nn.Module):
    def __init__(self, backbone_name="RepVGG-B1g2", deploy=True):
        super().__init__()
        backbone = get_RepVGG_func_by_name(backbone_name)(deploy=deploy)
        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear_reg = nn.Linear(2048, 6)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear_reg(x)
        return compute_rotation_matrix_from_ortho6d(x)


class QES6D:
    def __init__(self, weights=DEFAULT_WEIGHTS, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = QES6DNet(deploy=True).to(self.device)
        state = torch.load(str(weights), map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image[:, :, ::-1])
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            rotation = self.model(tensor)
            euler = compute_euler_angles_from_rotation_matrices(rotation) * 180.0 / torch.pi
        pitch = float(euler[0, 0].detach().cpu())
        yaw = float(euler[0, 1].detach().cpu())
        roll = float(euler[0, 2].detach().cpu())
        return pitch, yaw, roll

    def draw_axis(self, image, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        return draw_axis(image, yaw, pitch, roll, tdx=tdx, tdy=tdy, size=size)
