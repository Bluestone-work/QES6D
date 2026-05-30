# QES6D

This repository contains the paper-version QES6D training and evaluation framework extracted from `/root/autodl-tmp/QES6D`.

QES6D is not only an inference wrapper. The main experiment path is the two-stage semi-supervised pipeline implemented by `run_qes_experiments.sh`:

1. supervised Stage-1 warm-up on labeled 300W-LP data;
2. classroom pseudo-label refresh using the Stage-1 checkpoint;
3. Stage-2 semi-supervised refinement with pseudo-label reliability weighting, pseudo-label curriculum, and weak/strong classroom consistency;
4. fair benchmark evaluation on AFLW2000 and BIWI, with augmentation-consistency diagnostics.

## Important files

- `run_qes_experiments.sh`: one-click QES6D training/evaluation pipeline.
- `train_qes_paper.py`: paper training implementation.
- `generate_pseudo_labels.py`: pseudo-label generation and refresh entry.
- `eval_qes_paper.py`: unified benchmark and diagnostic evaluation.
- `QES6D/model.py`: QES6D model variants, including EfficientNetV2-based 6D rotation regression.
- `QES6D/loss.py`: QES6D losses, including `GeodesicPlusAxisLoss` and `RobustEulerAxisLoss`.
- `QES6D/semi_supervised_datasets.py`: labeled/pseudo semi-supervised dataset composition, filtering, reliability, and curriculum support.
- `QES6D/pseudo_label_generator.py`: TTA, face-crop export, confidence, and consensus pseudo-label logic.
- `docs/QES6D_Manuscript_KBS_repaired_no_classroom_labels.tex`: manuscript copy with clarified metric definitions.

## Environment

Install the core Python dependencies:

```bash
pip install -r requirements.txt
```

Optional modules used by the full pipeline include `ultralytics` and `gfpgan`. They are required when pseudo-label refresh uses YOLO face detection or GFPGAN enhancement.

## Run the paper pipeline

The default paths follow the AutoDL layout used in the experiments. Override environment variables if your data paths differ.

```bash
cd /root/autodl-tmp/QES6D
ROOT_DIR=/root/autodl-tmp bash run_qes_experiments.sh
```

Typical external data expected by the script include:

- `300W_LP` and `300W_LP/files.txt` for labeled source training;
- classroom pseudo-label source images/crops;
- `BIWI_split/BIWI_val.npz` and `BIWI_split/BIWI_val_stems.txt` for validation;
- AFLW2000/BIWI benchmark files for fair evaluation.

## Metrics

The manuscript and evaluation code report:

- `Yaw MAE`, `Pitch MAE`, `Roll MAE`: wrapped per-axis angular errors averaged over the benchmark.
- `Avg MAE`: per-sample mean of yaw/pitch/roll wrapped errors, averaged over the benchmark.
- `Acc@10`: percentage of samples whose per-sample average angular error is no larger than 10 degrees.
- `Aug Mean`: mean prediction drift under the shared augmentation-consistency protocol.
- `Aug P95`: 95th percentile of the same sample-level augmentation drift scores.

`Aug Mean` and `Aug P95` are reference-free diagnostics and do not use pose ground truth.
