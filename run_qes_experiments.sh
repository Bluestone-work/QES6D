#!/usr/bin/env bash
set -euo pipefail

pick_first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PROJECT_DIR}/.." && pwd)}"
TRAIN_PY="${TRAIN_PY:-${PROJECT_DIR}/train_qes_paper.py}"
PSEUDO_GEN_PY="${PSEUDO_GEN_PY:-${PROJECT_DIR}/generate_pseudo_labels.py}"
FAIR_EVAL_SH="${FAIR_EVAL_SH:-${PROJECT_DIR}/scripts/run_fair_five_eval.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

LABELED_DATA="${LABELED_DATA:-${ROOT_DIR}/300W_LP}"
LABELED_LIST="${LABELED_LIST:-${ROOT_DIR}/300W_LP/files.txt}"
PSEUDO_FILE="${PSEUDO_FILE:-${ROOT_DIR}/data_pseudo_train_faces.npz}"
PSEUDO_ROOT="${PSEUDO_ROOT:-${ROOT_DIR}/output/qes_paper_runs/effnetv2_full_semi_seed/pseudo_refresh/face_crops_test}"
VAL_DATA="${VAL_DATA:-${ROOT_DIR}/BIWI_split/BIWI_val.npz}"
VAL_LIST="${VAL_LIST:-${ROOT_DIR}/BIWI_split/BIWI_val_stems.txt}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/output/qes_paper_runs}"
EXP_NAME="${EXP_NAME:-effnetv2_full_semi_seed}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUT_ROOT}/${EXP_NAME}}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"

RUN_MODE="${RUN_MODE:-train_eval}"
TRAIN_STRATEGY="${TRAIN_STRATEGY:-two_stage_refresh}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-22}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
PSEUDO_LABEL_RATIO="${PSEUDO_LABEL_RATIO:-0.25}"
# CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.9}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.93}"
PSEUDO_LOSS_WEIGHT="${PSEUDO_LOSS_WEIGHT:-0.1}"
CLASSROOM_CONSISTENCY_WEIGHT="${CLASSROOM_CONSISTENCY_WEIGHT:-0.2}"
CLASSROOM_AUG_PROB="${CLASSROOM_AUG_PROB:-0.4}"
TEACHER_EMA_DECAY="${TEACHER_EMA_DECAY:-0.999}"
# CONSISTENCY_THRESHOLD_DEG="${CONSISTENCY_THRESHOLD_DEG:-6.0}"
CONSISTENCY_THRESHOLD_DEG="${CONSISTENCY_THRESHOLD_DEG:-3.0}"
MIN_DET_CONFIDENCE="${MIN_DET_CONFIDENCE:-0.7}"
# MIN_FACE_SIZE="${MIN_FACE_SIZE:-40}"
MIN_FACE_SIZE="${MIN_FACE_SIZE:-48}"
MODEL_VARIANT="${MODEL_VARIANT:-effnetv2}"
EFFNET_BACKBONE="${EFFNET_BACKBONE:-efficientnet_v2_l}"
EFFNET_HEAD_STYLE="${EFFNET_HEAD_STYLE:-baseline}"
EXP_SETTING="${EXP_SETTING:-full_semi}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-multistep}"
SCHEDULER_MILESTONES="${SCHEDULER_MILESTONES:-10,20}"
SCHEDULER_GAMMA="${SCHEDULER_GAMMA:-0.5}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-}"
ENABLE_PSEUDO_CURRICULUM="${ENABLE_PSEUDO_CURRICULUM:-1}"
CURRICULUM_RAMP_EPOCHS="${CURRICULUM_RAMP_EPOCHS:-20}"
CURRICULUM_FACE_SIZE_START="${CURRICULUM_FACE_SIZE_START:-72}"
CURRICULUM_FACE_SIZE_END="${CURRICULUM_FACE_SIZE_END:-40}"
CURRICULUM_BLUR_START="${CURRICULUM_BLUR_START:-120}"
CURRICULUM_BLUR_END="${CURRICULUM_BLUR_END:-40}"
CURRICULUM_MODEL_CONS_START="${CURRICULUM_MODEL_CONS_START:-3.5}"
CURRICULUM_MODEL_CONS_END="${CURRICULUM_MODEL_CONS_END:-6.0}"

BACKBONE_FILE="${BACKBONE_FILE:-}"
if [[ -z "${BACKBONE_FILE}" ]]; then
  BACKBONE_FILE="$(pick_first_existing \
    /root/6DRepNet/sixdrepnet/RepVGG-B1g2-train.pth \
    ${PROJECT_DIR}/weights/RepVGG-B1g2-train.pth \
    ${ROOT_DIR}/6DRepNet/sixdrepnet/RepVGG-B1g2-train.pth || true)"
fi
YOLO_WEIGHTS="${YOLO_WEIGHTS:-}"
if [[ -z "${YOLO_WEIGHTS}" ]]; then
  YOLO_WEIGHTS="$(pick_first_existing \
    ${PROJECT_DIR}/facedat/yolov8x-face.pt \
    ${ROOT_DIR}/6DRepNet/facedat/yolov8x-face.pt \
    ${ROOT_DIR}/6DRepNet/facedat/yolov8n-face.pt \
    /root/6DRepNet/facedat/yolov8x.pt \
    ${ROOT_DIR}/6DRepNet/facedat/yolo11x.pt || true)"
fi
GFPGAN_MODEL="${GFPGAN_MODEL:-${ROOT_DIR}/models/GFPGANv1.3.pth}"
UNLABELED_DATA_DIR="${UNLABELED_DATA_DIR:-${PSEUDO_ROOT}}"
PSEUDO_REFRESH_SINGLE_FACE_MODE="${PSEUDO_REFRESH_SINGLE_FACE_MODE:-0}"
PSEUDO_FACE_DET_THRESHOLD="${PSEUDO_FACE_DET_THRESHOLD:-0.2}"
CONSENSUS_AUX_MODEL_PATH="${CONSENSUS_AUX_MODEL_PATH:-${PROJECT_DIR}/weights/QES6D_300W_LP_AFLW2000.pth}"
CONSENSUS_AUX_MODEL_VARIANT="${CONSENSUS_AUX_MODEL_VARIANT:-sixdrepnet_o}"

STAGE1_DIR="${STAGE1_DIR:-${OUTPUT_DIR}/stage1_sup}"
STAGE1_LOG="${STAGE1_DIR}/train.log"
PSEUDO_REFRESH_DIR="${PSEUDO_REFRESH_DIR:-${OUTPUT_DIR}/pseudo_refresh}"
PSEUDO_REFRESH_LOG="${PSEUDO_REFRESH_DIR}/refresh.log"
REFRESH_BASE_FILE="${PSEUDO_REFRESH_DIR}/refreshed_pseudo_labels.npz"
REFRESH_FACE_FILE="${PSEUDO_REFRESH_DIR}/refreshed_pseudo_labels_faces.npz"
REFRESH_FACE_DIR="${PSEUDO_REFRESH_DIR}/face_crops"
STAGE2_LOG="${OUTPUT_DIR}/train.log"

EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${ROOT_DIR}/fair_eval_runs}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${EVAL_OUT_ROOT}/${EXP_NAME}_fair_eval}"
MODEL_FOR_EVAL="${MODEL_FOR_EVAL:-}"
DAD3D_REPO="${DAD3D_REPO:-${ROOT_DIR}/DAD-3DHeads}"
FAIR_ONLY_CUSTOM_SIXD="${FAIR_ONLY_CUSTOM_SIXD:-0}"
FAIR_CONTINUE_ON_ERROR="${FAIR_CONTINUE_ON_ERROR:-0}"

mkdir -p "${OUT_ROOT}" "${OUTPUT_DIR}"

pick_best_model() {
  local search_dir="$1"
  local candidate
  for candidate in \
    "${search_dir}/best_model_ema.pth" \
    "${search_dir}/best_model.pth" \
    "${search_dir}/final_model_ema.pth" \
    "${search_dir}/final_model.pth"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

run_train_stage() {
  local stage_output_dir="$1"
  local stage_log="$2"
  local stage_exp_setting="$3"
  local stage_pretrained="$4"
  local stage_pseudo_file="$5"
  local stage_pseudo_root="$6"
  shift 6

  mkdir -p "${stage_output_dir}"

  local -a train_args=(
    --labeled_data_dir "${LABELED_DATA}"
    --labeled_filename_list "${LABELED_LIST}"
    --pseudo_label_file "${stage_pseudo_file}"
    --pseudo_data_root "${stage_pseudo_root}"
    --val_data_dir "${VAL_DATA}"
    --val_filename_list "${VAL_LIST}"
    --model_variant "${MODEL_VARIANT}"
    --effnet_backbone "${EFFNET_BACKBONE}"
    --effnet_head_style "${EFFNET_HEAD_STYLE}"
    --exp_setting "${stage_exp_setting}"
    --pseudo_label_ratio "${PSEUDO_LABEL_RATIO}"
    --confidence_threshold "${CONFIDENCE_THRESHOLD}"
    --pseudo_loss_weight "${PSEUDO_LOSS_WEIGHT}"
    --classroom_consistency_weight "${CLASSROOM_CONSISTENCY_WEIGHT}"
    --classroom_aug_prob "${CLASSROOM_AUG_PROB}"
    --teacher_ema_decay "${TEACHER_EMA_DECAY}"
    --consistency_threshold_deg "${CONSISTENCY_THRESHOLD_DEG}"
    --min_det_confidence "${MIN_DET_CONFIDENCE}"
    --min_face_size "${MIN_FACE_SIZE}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --num_workers "${NUM_WORKERS}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --scheduler_type "${SCHEDULER_TYPE}"
    --scheduler_milestones "${SCHEDULER_MILESTONES}"
    --scheduler_gamma "${SCHEDULER_GAMMA}"
    --gpu "${GPU_ID}"
    --seed "${SEED}"
    --output_dir "${stage_output_dir}"
    --save_best_model
    --skip_pseudo_only_batch
    --save_freq 25
    --print_freq 75
  )

  if [[ -n "${stage_pretrained}" ]]; then
    train_args+=(--pretrained_model "${stage_pretrained}")
  else
    train_args+=(--pretrained_model "")
  fi
  if [[ -n "${BACKBONE_FILE}" ]]; then
    train_args+=(--backbone_file "${BACKBONE_FILE}")
  fi

  if [[ "${ENABLE_PSEUDO_CURRICULUM}" == "1" && "${stage_exp_setting}" != "sup_only" ]]; then
    train_args+=(
      --enable_pseudo_curriculum
      --curriculum_ramp_epochs "${CURRICULUM_RAMP_EPOCHS}"
      --curriculum_face_size_start "${CURRICULUM_FACE_SIZE_START}"
      --curriculum_face_size_end "${CURRICULUM_FACE_SIZE_END}"
      --curriculum_blur_start "${CURRICULUM_BLUR_START}"
      --curriculum_blur_end "${CURRICULUM_BLUR_END}"
      --curriculum_model_consistency_start "${CURRICULUM_MODEL_CONS_START}"
      --curriculum_model_consistency_end "${CURRICULUM_MODEL_CONS_END}"
    )
  fi

  if [[ "$#" -gt 0 ]]; then
    train_args+=("$@")
  fi

  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${TRAIN_PY}" "${train_args[@]}" 2>&1 | tee "${stage_log}"
}

echo "=========================================="
echo "QES6D One-Click Pipeline"
echo "=========================================="
echo "RUN_MODE=${RUN_MODE}"
echo "TRAIN_STRATEGY=${TRAIN_STRATEGY}"
echo "TRAIN_PY=${TRAIN_PY}"
echo "PSEUDO_GEN_PY=${PSEUDO_GEN_PY}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "EVAL_RUN_DIR=${EVAL_RUN_DIR}"
echo ""

for required_file in "${TRAIN_PY}" "${PSEUDO_GEN_PY}" "${FAIR_EVAL_SH}" "${LABELED_LIST}" "${PSEUDO_FILE}" "${VAL_DATA}" "${VAL_LIST}" "${YOLO_WEIGHTS}" "${GFPGAN_MODEL}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[ERROR] Missing file: ${required_file}" >&2
    exit 1
  fi
done

for required_dir in "${LABELED_DATA}" "${PSEUDO_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "[ERROR] Missing directory: ${required_dir}" >&2
    exit 1
  fi
done

if [[ -n "${BACKBONE_FILE}" && ! -f "${BACKBONE_FILE}" ]]; then
  echo "[ERROR] Missing backbone file: ${BACKBONE_FILE}" >&2
  exit 1
fi

case "${RUN_MODE}" in
  train_eval|train_only|eval_only)
    ;;
  *)
    echo "[ERROR] RUN_MODE must be one of: train_eval, train_only, eval_only" >&2
    exit 1
    ;;
esac

case "${TRAIN_STRATEGY}" in
  single_stage|two_stage_refresh)
    ;;
  *)
    echo "[ERROR] TRAIN_STRATEGY must be one of: single_stage, two_stage_refresh" >&2
    exit 1
    ;;
esac

if [[ "${RUN_MODE}" != "eval_only" ]]; then
  if [[ "${TRAIN_STRATEGY}" == "single_stage" ]]; then
    if [[ -z "$(pick_best_model "${OUTPUT_DIR}" || true)" ]]; then
      echo "[TRAIN] launching single-stage QES6D training"
      run_train_stage "${OUTPUT_DIR}" "${STAGE2_LOG}" "${EXP_SETTING}" "${PRETRAINED_MODEL}" "${PSEUDO_FILE}" "${PSEUDO_ROOT}"
    else
      echo "[SKIP] single-stage training already completed at ${OUTPUT_DIR}"
    fi
  else
    STAGE1_MODEL="$(pick_best_model "${STAGE1_DIR}" || true)"
    if [[ -z "${STAGE1_MODEL}" ]]; then
      echo "[TRAIN] stage 1 supervised warm-up"
      run_train_stage "${STAGE1_DIR}" "${STAGE1_LOG}" "sup_only" "${PRETRAINED_MODEL}" "${PSEUDO_FILE}" "${PSEUDO_ROOT}"
      STAGE1_MODEL="$(pick_best_model "${STAGE1_DIR}" || true)"
    else
      echo "[SKIP] stage 1 already completed: ${STAGE1_MODEL}"
    fi

    if [[ -z "${STAGE1_MODEL}" ]]; then
      echo "[ERROR] Stage 1 model not found after training" >&2
      exit 1
    fi

    mkdir -p "${PSEUDO_REFRESH_DIR}"
    if [[ ! -f "${REFRESH_FACE_FILE}" ]]; then
      echo "[PSEUDO] refreshing consensus pseudo labels"
      PSEUDO_GEN_ARGS=(
        --model_path "${STAGE1_MODEL}"
        --model_variant "${MODEL_VARIANT}"
        --data_dir "${UNLABELED_DATA_DIR}"
        --output_file "${REFRESH_BASE_FILE}"
        --backbone_file "${BACKBONE_FILE}"
        --effnet_backbone "${EFFNET_BACKBONE}"
        --confidence_threshold "${CONFIDENCE_THRESHOLD}"
        --use_tta
        --consistency_threshold_deg "${CONSISTENCY_THRESHOLD_DEG}"
        --cross_model_threshold_deg "${CURRICULUM_MODEL_CONS_END}"
        --save_face_crops
        --face_crops_dir "${REFRESH_FACE_DIR}"
        --yolo_model "${YOLO_WEIGHTS}"
        --yolo_version "yolov8"
        --gfpgan_model "${GFPGAN_MODEL}"
        --face_det_threshold "${PSEUDO_FACE_DET_THRESHOLD}"
        --min_face_size "${MIN_FACE_SIZE}"
        --gpu "${GPU_ID}"
      )
      if [[ "${PSEUDO_REFRESH_SINGLE_FACE_MODE}" == "1" ]]; then
        PSEUDO_GEN_ARGS+=(--single_face_mode)
      fi
      if [[ -n "${CONSENSUS_AUX_MODEL_PATH}" && -f "${CONSENSUS_AUX_MODEL_PATH}" ]]; then
        PSEUDO_GEN_ARGS+=(--aux_model_path "${CONSENSUS_AUX_MODEL_PATH}" --aux_model_variant "${CONSENSUS_AUX_MODEL_VARIANT}")
      fi
      if [[ -f "${GFPGAN_MODEL}" ]]; then
        PSEUDO_GEN_ARGS+=(--use_gfpgan)
      fi
      PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${PSEUDO_GEN_PY}" "${PSEUDO_GEN_ARGS[@]}" 2>&1 | tee "${PSEUDO_REFRESH_LOG}"
    else
      echo "[SKIP] refreshed pseudo labels already exist: ${REFRESH_FACE_FILE}"
    fi

    if [[ ! -f "${REFRESH_FACE_FILE}" ]]; then
      echo "[ERROR] Refreshed face-level pseudo labels not found: ${REFRESH_FACE_FILE}" >&2
      exit 1
    fi

    if [[ -z "$(pick_best_model "${OUTPUT_DIR}" || true)" ]]; then
      echo "[TRAIN] stage 2 semi-supervised refinement"
      run_train_stage "${OUTPUT_DIR}" "${STAGE2_LOG}" "full_semi" "${STAGE1_MODEL}" "${REFRESH_FACE_FILE}" "${REFRESH_FACE_DIR}"
    else
      echo "[SKIP] stage 2 already completed at ${OUTPUT_DIR}"
    fi
  fi
fi

if [[ -z "${MODEL_FOR_EVAL}" ]]; then
  MODEL_FOR_EVAL="$(pick_best_model "${OUTPUT_DIR}" || true)"
  if [[ -z "${MODEL_FOR_EVAL}" ]]; then
    echo "[ERROR] No model found for evaluation in ${OUTPUT_DIR}" >&2
    exit 1
  fi
fi

if [[ "${RUN_MODE}" != "train_only" ]]; then
  echo "[EVAL] running fair evaluation suite"
  RESUME_RUN_DIR="${EVAL_RUN_DIR}" \
  GPU_ID="${GPU_ID}" \
  SEED="${SEED}" \
  CUSTOM_SIXD_ENABLE=1 \
  CUSTOM_SIXD_CKPT="${MODEL_FOR_EVAL}" \
  CUSTOM_SIXD_MODEL_VARIANT="${MODEL_VARIANT}" \
  CUSTOM_SIXD_YOLO_WEIGHTS="${YOLO_WEIGHTS}" \
  CUSTOM_SIXD_GFPGAN_MODEL="${GFPGAN_MODEL}" \
  ONLY_CUSTOM_SIXD="${FAIR_ONLY_CUSTOM_SIXD}" \
  CONTINUE_ON_ERROR="${FAIR_CONTINUE_ON_ERROR}" \
  DAD3D_REPO="${DAD3D_REPO}" \
  bash "${FAIR_EVAL_SH}"
fi

echo ""
echo "Pipeline finished."
echo "QES6D checkpoint: ${MODEL_FOR_EVAL}"
echo "Fair eval dir: ${EVAL_RUN_DIR}"
echo "Paper CSV: ${EVAL_RUN_DIR}/paper_metrics.csv"
echo "Paper Markdown: ${EVAL_RUN_DIR}/paper_metrics.md"
echo "Montage dir: ${EVAL_RUN_DIR}/viz_compare"
