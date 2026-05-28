# QES6D

QES6D is a minimal runnable package extracted from `/root/autodl-tmp/6DRepNet`. It keeps the RepVGG-B1g2 head-pose model, local pretrained weights, single-image prediction, and AFLW2000 evaluation.

## Install

```bash
cd /root/autodl-tmp/QES6D
pip install -r requirements.txt
```

## Smoke Test

```bash
cd /root/autodl-tmp/QES6D
python scripts/smoke_test.py
```

## Predict One Image

```bash
cd /root/autodl-tmp/QES6D
python scripts/predict.py --image /path/to/image.jpg --output output/pred.jpg
```

## Test AFLW2000

```bash
cd /root/autodl-tmp/QES6D
python scripts/test_aflw2000.py --data_dir /path/to/AFLW2000 --filename_list /path/to/files.txt
```

The bundled default weights are stored at `weights/QES6D_300W_LP_AFLW2000.pth`.
