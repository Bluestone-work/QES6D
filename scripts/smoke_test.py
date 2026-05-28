import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from qes6d import QES6DNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QES6DNet(deploy=True).to(device).eval()
    image = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(image)
    assert output.shape == (1, 3, 3), output.shape
    print("QES6D smoke test passed", tuple(output.shape))


if __name__ == "__main__":
    main()
