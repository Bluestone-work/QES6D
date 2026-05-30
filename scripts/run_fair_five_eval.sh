#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
RUN_MODE="${RUN_MODE:-sequential}"
SEED="${SEED:-42}"
FORCE_RERUN="${FORCE_RERUN:-0}"
VIZ_EVERY="${VIZ_EVERY:-200}"
VIZ_MAX="${VIZ_MAX:-20}"
AXIS_SIZE="${AXIS_SIZE:-70}"
ACC_THRESHOLD="${ACC_THRESHOLD:-10}"
MAX_SAMPLES_BIWI="${MAX_SAMPLES_BIWI:--1}"
MAX_SAMPLES_AFLW="${MAX_SAMPLES_AFLW:--1}"
ONLY_CUSTOM_QES6D="${ONLY_CUSTOM_QES6D:-0}"   # 1: run only custom QES6D (for ablation)
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}" # 1: do not abort whole suite on single-job failure
SAVE_ALL_LABELED_ERRORS="${SAVE_ALL_LABELED_ERRORS:-1}"
BUILD_STATS_REPORT="${BUILD_STATS_REPORT:-1}"
STATS_BOOTSTRAP_ITERS="${STATS_BOOTSTRAP_ITERS:-10000}"
STATS_PERMUTATION_ITERS="${STATS_PERMUTATION_ITERS:-10000}"
CLASSROOM_NOGT_ENABLE="${CLASSROOM_NOGT_ENABLE:-0}"
CLASSROOM_NOGT_DATA_DIR="${CLASSROOM_NOGT_DATA_DIR:-/root/autodl-tmp/data_test}"
CLASSROOM_NOGT_MAX_SAMPLES="${CLASSROOM_NOGT_MAX_SAMPLES:--1}"
CLASSROOM_NOGT_YOLO_WEIGHTS="${CLASSROOM_NOGT_YOLO_WEIGHTS:-/root/autodl-tmp/QES6D/facedat/yolo11x.pt}"
# CLASSROOM_NOGT_PERSON_WEIGHTS="${CLASS  ROOM_NOGT_PERSON_WEIGHTS:-}"

OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/fair_eval_runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RUN_DIR="${RESUME_RUN_DIR:-${OUT_ROOT}/${STAMP}}"
LOG_DIR="${RUN_DIR}/logs"
JSON_DIR="${RUN_DIR}/json"
VIZ_DIR="${RUN_DIR}/viz"
TMP_DIR="${RUN_DIR}/tmp"
SUMMARY_JSON="${RUN_DIR}/summary_metrics.json"
SUMMARY_MD="${RUN_DIR}/summary_metrics.md"
SUMMARY_CSV="${RUN_DIR}/summary_metrics.csv"
PAPER_JSON="${RUN_DIR}/paper_metrics.json"
PAPER_MD="${RUN_DIR}/paper_metrics.md"
PAPER_CSV="${RUN_DIR}/paper_metrics.csv"
VIZ_COMPARE_DIR="${RUN_DIR}/viz_compare"
VIZ_COMPARE_MANIFEST="${RUN_DIR}/viz_compare_manifest.csv"
STATS_JSON="${RUN_DIR}/stats_report.json"
STATS_MD="${RUN_DIR}/stats_report.md"
STATS_CI_CSV="${RUN_DIR}/stats_ci.csv"
STATS_PVALUE_CSV="${RUN_DIR}/stats_pvalues.csv"
CLASSROOM_NOGT_DIR="${RUN_DIR}/classroom_nogt"
CLASSROOM_NOGT_MODELS_JSON="${CLASSROOM_NOGT_DIR}/models.json"
CLASSROOM_NOGT_JSON="${CLASSROOM_NOGT_DIR}/classroom_nogt_report.json"
CLASSROOM_NOGT_CSV="${CLASSROOM_NOGT_DIR}/classroom_nogt_report.csv"
CLASSROOM_NOGT_MD="${CLASSROOM_NOGT_DIR}/classroom_nogt_report.md"

mkdir -p "${LOG_DIR}" "${JSON_DIR}" "${VIZ_DIR}" "${TMP_DIR}"
touch "${TMP_DIR}/empty_inputs.txt"
mkdir -p "${TMP_DIR}/empty_biwi_dir"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

QES6D_REPO="${QES6D_REPO:-/root/autodl-tmp/QES6D/QES6D}"
QES6D_CKPT="${QES6D_CKPT:-/root/autodl-tmp/QES6D/weights/QES6D_300W_LP_AFLW2000.pth}"
CUSTOM_QES6D_ENABLE="${CUSTOM_QES6D_ENABLE:-1}"
CUSTOM_QES6D_CKPT="${CUSTOM_QES6D_CKPT:-/root/autodl-tmp/output/qes_paper_runs/stage2_from_pseudo_refresh_moderate/best_model.pth}"
CUSTOM_QES6D_MODEL_VARIANT="${CUSTOM_QES6D_MODEL_VARIANT:-effnetv2}"
CUSTOM_QES6D_YOLO_WEIGHTS="${CUSTOM_QES6D_YOLO_WEIGHTS:-/root/autodl-tmp/QES6D/facedat/yolo11x.pt}"
CUSTOM_QES6D_GFPGAN_MODEL="${CUSTOM_QES6D_GFPGAN_MODEL:-/root/autodl-tmp/models/GFPGANv1.3.pth}"
CUSTOM_QES6D_USE_YOLO="${CUSTOM_QES6D_USE_YOLO:-0}"
CUSTOM_QES6D_USE_GFPGAN="${CUSTOM_QES6D_USE_GFPGAN:-0}"

FSA_REPO="${FSA_REPO:-/root/autodl-tmp/FSA-Net}"
FSA_TRAIN_DB="${FSA_TRAIN_DB:-300W_LP}"
CONDA_BASE="${CONDA_BASE:-/root/miniconda3}"
CONDA_SH="${CONDA_SH:-${CONDA_BASE}/etc/profile.d/conda.sh}"
FSA_CONDA_ENV="${FSA_CONDA_ENV:-fsanet_tf1}"

WHENET_REPO="${WHENET_REPO:-/root/autodl-tmp/HeadPoseEstimation-WHENet}"
WHENET_WEIGHTS="${WHENET_WEIGHTS:-/root/autodl-tmp/HeadPoseEstimation-WHENet/WHENet.h5}"

SEMI_REPO="${SEMI_REPO:-/root/autodl-tmp/SemiUHPE}"
SEMI_OUTPUT_ROOT="${SEMI_OUTPUT_ROOT:-/root/autodl-tmp}"
SEMI_EXP_NAME="${SEMI_EXP_NAME:-300WLP_AFLW2000}"
SEMI_CONFIG_PATH="${SEMI_CONFIG_PATH:-/root/autodl-tmp/SemiUHPE/settings/300WLP_AFLW2000_effinetv2.yml}"
SEMI_TEST_CKPT="${SEMI_TEST_CKPT:-SSL1.0_r0.05_ce_effinetv2_t-5.3_b32_ema/Apr18_174048/best.pth}"

HOPENET_REPO="${HOPENET_REPO:-/root/autodl-tmp/deep-head-pose}"
HOPENET_SNAPSHOT="${HOPENET_SNAPSHOT:-/root/autodl-tmp/hopenet/hopenet_alpha1.pkl}"

DAD3D_REPO="${DAD3D_REPO:-/root/autodl-tmp/DAD-3DHeads}"

BIWI_NPZ="${BIWI_NPZ:-/root/autodl-tmp/BIWI_noTrack.npz}"
AFLW2000_DIR="${AFLW2000_DIR:-/root/autodl-tmp/AFLW2000}"
AFLW2000_LIST="${AFLW2000_LIST:-/root/autodl-tmp/AFLW2000/files.txt}"
CUSTOM_QES6D_BIWI_DATA_DIR="${CUSTOM_QES6D_BIWI_DATA_DIR:-/root/autodl-tmp/biwi}"
CUSTOM_QES6D_AFLW_DATA_DIR="${CUSTOM_QES6D_AFLW_DATA_DIR:-${AFLW2000_DIR}}"
NO_GT_DATA_DIR="${NO_GT_DATA_DIR:-}"
NO_GT_INPUT_LIST="${NO_GT_INPUT_LIST:-}"

cat > "${RUN_DIR}/protocol.txt" <<EOF
fairness_protocol:
  datasets: BIWI + AFLW2000
  upstream_modules: YOLO=off, GFPGAN=off
  seed: ${SEED}
  augment_protocol: classroom_paper_v1
  augment_set: raw, small_face, blur, uneven_light
  common_input_protocol:
    - BIWI: shared BIWI_noTrack.npz raw frames
    - AFLW2000: shared landmark-square crop before model-specific resize
  note:
    - QES6D baseline is split into BIWI and AFLW2000 runs because its script evaluates one labeled dataset per run.
    - Use labeled metrics and classroom_augmentation_consistency_deg for the paper table.
    - Custom QES6D checkpoint uses manual-style eval flow by default (no YOLO/GFPGAN, BIWI data_dir points to real BIWI frames).
    - DAD-3DNet is evaluated with the same BIWI/AFLW2000 protocol and unified heavy-viz style.
EOF

declare -a JOB_NAMES=()
declare -a JOB_PIDS=()

QES6D_BIWI_DATA_DIR="${TMP_DIR}/empty_biwi_dir"
QES6D_AFLW_DATA_DIR="${AFLW2000_DIR}"
declare -a QES6D_NO_GT_INPUT_ARGS=(--input_list "${TMP_DIR}/empty_inputs.txt")

if [[ -n "${NO_GT_DATA_DIR}" ]]; then
  QES6D_BIWI_DATA_DIR="${NO_GT_DATA_DIR}"
  QES6D_AFLW_DATA_DIR="${NO_GT_DATA_DIR}"
  if [[ -n "${NO_GT_INPUT_LIST}" ]]; then
    QES6D_NO_GT_INPUT_ARGS=(--input_list "${NO_GT_INPUT_LIST}")
  else
    QES6D_NO_GT_INPUT_ARGS=()
  fi
fi

declare -a ALL_LABELED_ERRORS_ARGS=()
if [[ "${SAVE_ALL_LABELED_ERRORS}" == "1" ]]; then
  ALL_LABELED_ERRORS_ARGS+=(--save_all_labeled_errors)
fi

timestamp_now() {
  date '+%H:%M:%S'
}

fail_or_continue() {
  local msg="$1"
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    echo "[$(timestamp_now)] [warn] ${msg}" >&2
    return 0
  fi
  echo "[$(timestamp_now)] [failed] ${msg}" >&2
  exit 1
}

should_skip_job() {
  local output_json="$1"
  if [[ "${FORCE_RERUN}" == "1" ]]; then
    return 1
  fi
  [[ -f "${output_json}" ]]
}

write_summary() {
  python /root/autodl-tmp/summarize_fair_eval.py \
    --json-dir "${JSON_DIR}" \
    --output-json "${SUMMARY_JSON}" \
    --output-md "${SUMMARY_MD}" \
    --output-csv "${SUMMARY_CSV}"
  python /root/autodl-tmp/paper_results_from_fair_eval.py \
    --json-dir "${JSON_DIR}" \
    --output-json "${PAPER_JSON}" \
    --output-md "${PAPER_MD}" \
    --output-csv "${PAPER_CSV}"
  python /root/autodl-tmp/build_viz_comparison_montage.py \
    --viz-root "${VIZ_DIR}" \
    --output-dir "${VIZ_COMPARE_DIR}" \
    --manifest-csv "${VIZ_COMPARE_MANIFEST}" \
    --tile-height 240 \
    --methods-per-row 4 \
    --contact-sheet-count 12 \
    --contact-sheet-cols 4 \
    --all-sheet-cols 1
  if [[ "${BUILD_STATS_REPORT}" == "1" ]]; then
    python /root/autodl-tmp/build_qes_stats_report.py \
      --json-dir "${JSON_DIR}" \
      --output-json "${STATS_JSON}" \
      --output-md "${STATS_MD}" \
      --output-ci-csv "${STATS_CI_CSV}" \
      --output-pvalue-csv "${STATS_PVALUE_CSV}" \
      --anchor-method QES6D \
      --baseline-method QES6D \
      --bootstrap-iters "${STATS_BOOTSTRAP_ITERS}" \
      --permutation-iters "${STATS_PERMUTATION_ITERS}"
  fi
  echo "[$(timestamp_now)] [summary] ${SUMMARY_JSON}"
  echo "[$(timestamp_now)] [summary] ${SUMMARY_MD}"
  echo "[$(timestamp_now)] [summary] ${SUMMARY_CSV}"
  echo "[$(timestamp_now)] [paper] ${PAPER_JSON}"
  echo "[$(timestamp_now)] [paper] ${PAPER_MD}"
  echo "[$(timestamp_now)] [paper] ${PAPER_CSV}"
  echo "[$(timestamp_now)] [viz-compare] ${VIZ_COMPARE_DIR}"
  echo "[$(timestamp_now)] [viz-compare] ${VIZ_COMPARE_MANIFEST}"
  if [[ "${BUILD_STATS_REPORT}" == "1" ]]; then
    echo "[$(timestamp_now)] [stats] ${STATS_JSON}"
    echo "[$(timestamp_now)] [stats] ${STATS_MD}"
    echo "[$(timestamp_now)] [stats] ${STATS_CI_CSV}"
    echo "[$(timestamp_now)] [stats] ${STATS_PVALUE_CSV}"
  fi
}

run_classroom_nogt_eval() {
  if [[ "${CLASSROOM_NOGT_ENABLE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${CLASSROOM_NOGT_DATA_DIR}" ]]; then
    fail_or_continue "missing classroom no-GT data dir: ${CLASSROOM_NOGT_DATA_DIR}"
    return 0
  fi

  mkdir -p "${CLASSROOM_NOGT_DIR}"

  python - <<PY
import json
models = []

only_custom = "${ONLY_CUSTOM_QES6D}" == "1"
custom_enable = "${CUSTOM_QES6D_ENABLE}" != "0"

if (not only_custom) and "${QES6D_CKPT}":
    models.append({
        "name": "QES6D",
        "family": "QES6D",
        "ckpt": "${QES6D_CKPT}",
        "variant": "auto",
        "enabled": True,
    })

if custom_enable and "${CUSTOM_QES6D_CKPT}":
    models.append({
        "name": "QES6D",
        "family": "QES6D",
        "ckpt": "${CUSTOM_QES6D_CKPT}",
        "variant": "${CUSTOM_QES6D_MODEL_VARIANT}",
        "enabled": True,
    })

if not only_custom:
    models.extend([
        {
            "name": "SemiUHPE",
            "family": "semiuhpe",
            "repo_root": "${SEMI_REPO}",
            "exp_name": "${SEMI_EXP_NAME}",
            "test_ckpt": "${SEMI_TEST_CKPT}",
            "enabled": True,
        },
        {
            "name": "Hopenet",
            "family": "hopenet",
            "snapshot": "${HOPENET_SNAPSHOT}",
            "enabled": True,
        },
        {
            "name": "WHENet",
            "family": "whenet",
            "repo_root": "${WHENET_REPO}",
            "snapshot": "${WHENET_WEIGHTS}",
            "enabled": True,
        },
        {
            "name": "FSA-Net",
            "family": "fsanet",
            "repo_root": "${FSA_REPO}",
            "fsanet_train_db": "${FSA_TRAIN_DB}",
            "fsanet_use_pretrained": True,
            "enabled": True,
        },
        {
            "name": "DAD-3DNet",
            "family": "dad3dnet",
            "repo_root": "${FSA_REPO}",
            "dad3d_repo": "${DAD3D_REPO}",
            "enabled": True,
        },
    ])

payload = {
    "meta": {
        "reference_model": "QES6D" if custom_enable else (models[0]["name"] if models else "")
    },
    "models": models,
}
with open("${CLASSROOM_NOGT_MODELS_JSON}", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
PY

  local log_path="${LOG_DIR}/classroom_nogt_compare.log"
  echo "[$(timestamp_now)] [start] classroom_nogt_compare"
  echo "[$(timestamp_now)] [log] ${log_path}"
  if (
    cd /root/autodl-tmp/QES6D/tools
    python eval_classroom_nogt_tracking_compare.py \
      --models_json "${CLASSROOM_NOGT_MODELS_JSON}" \
      --data_dir "${CLASSROOM_NOGT_DATA_DIR}" \
      --output_json "${CLASSROOM_NOGT_JSON}" \
      --output_csv "${CLASSROOM_NOGT_CSV}" \
      --output_md "${CLASSROOM_NOGT_MD}" \
      --gpu 0 \
      --seed "${SEED}" \
      --max_samples "${CLASSROOM_NOGT_MAX_SAMPLES}" \
      --yolo_weights "${CLASSROOM_NOGT_YOLO_WEIGHTS}" \
      --person_weights "${CLASSROOM_NOGT_PERSON_WEIGHTS}" \
      > "${log_path}" 2>&1
  ); then
    echo "[$(timestamp_now)] [done] classroom_nogt_compare"
    echo "[$(timestamp_now)] [classroom-nogt] ${CLASSROOM_NOGT_JSON}"
    echo "[$(timestamp_now)] [classroom-nogt] ${CLASSROOM_NOGT_CSV}"
    echo "[$(timestamp_now)] [classroom-nogt] ${CLASSROOM_NOGT_MD}"
  else
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[$(timestamp_now)] [failed] classroom_nogt_compare" >&2
      echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
      echo "[$(timestamp_now)] [continue] CONTINUE_ON_ERROR=1" >&2
    else
      echo "[$(timestamp_now)] [failed] classroom_nogt_compare" >&2
      echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
      exit 1
    fi
  fi
}

run_job() {
  local name="$1"
  local workdir="$2"
  shift 2
  local log_path="${LOG_DIR}/${name}.log"
  JOB_NAMES+=("${name}")

  if [[ "${RUN_MODE}" == "parallel" ]]; then
    (
      cd "${workdir}"
      "$@" > "${log_path}" 2>&1
    ) &
    JOB_PIDS+=("$!")
    echo "[$(timestamp_now)] [launch][parallel] ${name}"
    echo "[$(timestamp_now)] [log] ${log_path}"
  else
    echo "[$(timestamp_now)] [start ${#JOB_NAMES[@]}] ${name}"
    echo "[$(timestamp_now)] [log] ${log_path}"
    if (
      cd "${workdir}"
      "$@" > "${log_path}" 2>&1
    ); then
      echo "[$(timestamp_now)] [done] ${name}"
    else
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[$(timestamp_now)] [failed] ${name}" >&2
        echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
        echo "[$(timestamp_now)] [continue] CONTINUE_ON_ERROR=1" >&2
      else
        echo "[$(timestamp_now)] [failed] ${name}" >&2
        echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
        exit 1
      fi
    fi
  fi
}

run_job_in_conda() {
  local name="$1"
  local workdir="$2"
  local conda_env="$3"
  shift 3
  local log_path="${LOG_DIR}/${name}.log"
  JOB_NAMES+=("${name}")

  if [[ ! -f "${CONDA_SH}" ]]; then
    fail_or_continue "missing conda init script: ${CONDA_SH}"
    return 0
  fi

  if [[ "${RUN_MODE}" == "parallel" ]]; then
    (
      source "${CONDA_SH}"
      conda activate "${conda_env}"
      cd "${workdir}"
      "$@" > "${log_path}" 2>&1
    ) &
    JOB_PIDS+=("$!")
    echo "[$(timestamp_now)] [launch][parallel] ${name} (conda:${conda_env})"
    echo "[$(timestamp_now)] [log] ${log_path}"
  else
    echo "[$(timestamp_now)] [start ${#JOB_NAMES[@]}] ${name} (conda:${conda_env})"
    echo "[$(timestamp_now)] [log] ${log_path}"
    if (
      source "${CONDA_SH}"
      conda activate "${conda_env}"
      cd "${workdir}"
      "$@" > "${log_path}" 2>&1
    ); then
      echo "[$(timestamp_now)] [done] ${name}"
    else
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[$(timestamp_now)] [failed] ${name}" >&2
        echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
        echo "[$(timestamp_now)] [continue] CONTINUE_ON_ERROR=1" >&2
      else
        echo "[$(timestamp_now)] [failed] ${name}" >&2
        echo "[$(timestamp_now)] [check-log] ${log_path}" >&2
        exit 1
      fi
    fi
  fi
}

if [[ "${ONLY_CUSTOM_QES6D}" == "1" ]]; then
  echo "[$(timestamp_now)] [mode] ONLY_CUSTOM_QES6D=1 (skip baseline families)"
else
  if should_skip_job "${JSON_DIR}/QES6D_biwi.json"; then
    echo "[$(timestamp_now)] [skip] QES6D_biwi"
  else
    run_job "QES6D_biwi" "${QES6D_REPO}" \
      python eval_no_gt_labeled_heavyviz_v4_norot.py \
      --model_path "${QES6D_CKPT}" \
      --data_dir "${QES6D_BIWI_DATA_DIR}" \
      "${QES6D_NO_GT_INPUT_ARGS[@]}" \
      --eval_with_labels \
      --dataset BIWI \
      --filename_list "${BIWI_NPZ}" \
      --gpu 0 \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --labeled_batch_size 32 \
      --labeled_num_workers 2 \
      --save_viz \
      --viz_dir "${VIZ_DIR}/QES6D_biwi" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/QES6D_biwi.json"
  fi

  if should_skip_job "${JSON_DIR}/QES6D_aflw2000.json"; then
    echo "[$(timestamp_now)] [skip] QES6D_aflw2000"
  else
    run_job "QES6D_aflw2000" "${QES6D_REPO}" \
      python eval_no_gt_labeled_heavyviz_v4_norot.py \
      --model_path "${QES6D_CKPT}" \
      --data_dir "${QES6D_AFLW_DATA_DIR}" \
      "${QES6D_NO_GT_INPUT_ARGS[@]}" \
      --eval_with_labels \
      --dataset AFLW2000 \
      --filename_list "${AFLW2000_LIST}" \
      --gpu 0 \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --labeled_batch_size 32 \
      --labeled_num_workers 2 \
      --save_viz \
      --viz_dir "${VIZ_DIR}/QES6D_aflw2000" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/QES6D_aflw2000.json"
  fi
fi

if [[ "${CUSTOM_QES6D_ENABLE}" != "0" ]]; then
  if [[ ! -f "${CUSTOM_QES6D_CKPT}" ]]; then
    echo "[$(timestamp_now)] [failed] missing custom QES6D checkpoint: ${CUSTOM_QES6D_CKPT}" >&2
    exit 1
  fi
  if [[ ! -d "${CUSTOM_QES6D_BIWI_DATA_DIR}" ]]; then
    echo "[$(timestamp_now)] [failed] missing BIWI dir for custom QES6D eval: ${CUSTOM_QES6D_BIWI_DATA_DIR}" >&2
    exit 1
  fi
  if [[ ! -d "${CUSTOM_QES6D_AFLW_DATA_DIR}" ]]; then
    echo "[$(timestamp_now)] [failed] missing AFLW dir for custom QES6D eval: ${CUSTOM_QES6D_AFLW_DATA_DIR}" >&2
    exit 1
  fi
  if [[ "${CUSTOM_QES6D_USE_YOLO}" == "1" && ! -f "${CUSTOM_QES6D_YOLO_WEIGHTS}" ]]; then
    echo "[$(timestamp_now)] [failed] missing YOLO weights for custom QES6D eval: ${CUSTOM_QES6D_YOLO_WEIGHTS}" >&2
    exit 1
  fi
  if [[ "${CUSTOM_QES6D_USE_GFPGAN}" == "1" && ! -f "${CUSTOM_QES6D_GFPGAN_MODEL}" ]]; then
    echo "[$(timestamp_now)] [failed] missing GFPGAN weights for custom QES6D eval: ${CUSTOM_QES6D_GFPGAN_MODEL}" >&2
    exit 1
  fi
fi

declare -a CUSTOM_QES6D_PREPROC_ARGS=()
if [[ "${CUSTOM_QES6D_USE_YOLO}" == "1" ]]; then
  CUSTOM_QES6D_PREPROC_ARGS+=(
    --use_yolo
    --yolo_version yolov11
    --yolo_weights "${CUSTOM_QES6D_YOLO_WEIGHTS}"
  )
fi
if [[ "${CUSTOM_QES6D_USE_GFPGAN}" == "1" ]]; then
  CUSTOM_QES6D_PREPROC_ARGS+=(
    --use_gfpgan
    --gfpgan_model "${CUSTOM_QES6D_GFPGAN_MODEL}"
  )
fi

if [[ "${CUSTOM_QES6D_ENABLE}" == "0" ]]; then
  echo "[$(timestamp_now)] [skip] QES6D_effnetv2_biwi (disabled)"
elif should_skip_job "${JSON_DIR}/QES6D_effnetv2_biwi.json"; then
  echo "[$(timestamp_now)] [skip] QES6D_effnetv2_biwi"
else
  run_job "QES6D_effnetv2_biwi" "${QES6D_REPO}" \
    python eval_no_gt_labeled_heavyviz_v4_norot.py \
    --model_path "${CUSTOM_QES6D_CKPT}" \
    --model_variant "${CUSTOM_QES6D_MODEL_VARIANT}" \
    "${CUSTOM_QES6D_PREPROC_ARGS[@]}" \
    --data_dir "${CUSTOM_QES6D_BIWI_DATA_DIR}" \
    "${QES6D_NO_GT_INPUT_ARGS[@]}" \
    --eval_with_labels \
    --dataset BIWI \
    --filename_list "${BIWI_NPZ}" \
    --gpu 0 \
    --seed "${SEED}" \
    --acc_threshold "${ACC_THRESHOLD}" \
    --labeled_batch_size 32 \
    --labeled_num_workers 2 \
    --save_viz \
    --viz_dir "${VIZ_DIR}/QES6D_effnetv2_biwi" \
    --viz_every "${VIZ_EVERY}" \
    --viz_max "${VIZ_MAX}" \
    --axis_size "${AXIS_SIZE}" \
    "${ALL_LABELED_ERRORS_ARGS[@]}" \
    --output_json "${JSON_DIR}/QES6D_effnetv2_biwi.json"
fi

if [[ "${CUSTOM_QES6D_ENABLE}" == "0" ]]; then
  echo "[$(timestamp_now)] [skip] QES6D_effnetv2_aflw2000 (disabled)"
elif should_skip_job "${JSON_DIR}/QES6D_effnetv2_aflw2000.json"; then
  echo "[$(timestamp_now)] [skip] QES6D_effnetv2_aflw2000"
else
  run_job "QES6D_effnetv2_aflw2000" "${QES6D_REPO}" \
    python eval_no_gt_labeled_heavyviz_v4_norot.py \
    --model_path "${CUSTOM_QES6D_CKPT}" \
    --model_variant "${CUSTOM_QES6D_MODEL_VARIANT}" \
    "${CUSTOM_QES6D_PREPROC_ARGS[@]}" \
    --data_dir "${CUSTOM_QES6D_AFLW_DATA_DIR}" \
    "${QES6D_NO_GT_INPUT_ARGS[@]}" \
    --eval_with_labels \
    --dataset AFLW2000 \
    --filename_list "${AFLW2000_LIST}" \
    --gpu 0 \
    --seed "${SEED}" \
    --acc_threshold "${ACC_THRESHOLD}" \
    --labeled_batch_size 32 \
    --labeled_num_workers 2 \
    --save_viz \
    --viz_dir "${VIZ_DIR}/QES6D_effnetv2_aflw2000" \
    --viz_every "${VIZ_EVERY}" \
    --viz_max "${VIZ_MAX}" \
    --axis_size "${AXIS_SIZE}" \
    "${ALL_LABELED_ERRORS_ARGS[@]}" \
    --output_json "${JSON_DIR}/QES6D_effnetv2_aflw2000.json"
fi

if [[ "${ONLY_CUSTOM_QES6D}" != "1" ]]; then
  if should_skip_job "${JSON_DIR}/fsanet.json"; then
    echo "[$(timestamp_now)] [skip] fsanet"
  else
    run_job_in_conda "fsanet" "${FSA_REPO}" "${FSA_CONDA_ENV}" \
      python eval_fsanet_dad3dnet_extended_gpu_fix_heavyviz_v4_norot.py \
      --model_type fsanet \
      --datasets BIWI,AFLW2000 \
      --biwi_npz "${BIWI_NPZ}" \
      --aflw2000_dir "${AFLW2000_DIR}" \
      --gpu 0 \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --batch_size 64 \
      --max_samples_biwi "${MAX_SAMPLES_BIWI}" \
      --max_samples_aflw "${MAX_SAMPLES_AFLW}" \
      --fsanet_repo "${FSA_REPO}" \
      --fsanet_train_db "${FSA_TRAIN_DB}" \
      --fsanet_use_pretrained \
      --save_viz \
      --viz_dir "${VIZ_DIR}/fsanet" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/fsanet.json"
  fi

  if should_skip_job "${JSON_DIR}/whenet.json"; then
    echo "[$(timestamp_now)] [skip] whenet"
  else
    run_job "whenet" "${WHENET_REPO}" \
      env TF_CPP_MIN_LOG_LEVEL=3 ABSL_LOGGING_MIN_SEVERITY=3 \
      python eval_whenet_labeled_heavyviz_v4_norot.py \
      --datasets BIWI,AFLW2000 \
      --biwi_npz "${BIWI_NPZ}" \
      --aflw2000_dir "${AFLW2000_DIR}" \
      --whenet_weights "${WHENET_WEIGHTS}" \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --batch_size 32 \
      --max_samples_biwi "${MAX_SAMPLES_BIWI}" \
      --max_samples_aflw "${MAX_SAMPLES_AFLW}" \
      --save_viz \
      --viz_dir "${VIZ_DIR}/whenet" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/whenet.json"
  fi

  if should_skip_job "${JSON_DIR}/semiuhpe.json"; then
    echo "[$(timestamp_now)] [skip] semiuhpe"
  else
    run_job "semiuhpe" "${SEMI_REPO}" \
      python eval_yolo_gfpgan_labeled_heavyviz_v4_norot_fixed.py \
      --config "${SEMI_CONFIG_PATH}" \
      --output_root "${SEMI_OUTPUT_ROOT}" \
      --exp_name "${SEMI_EXP_NAME}" \
      --datasets BIWI,AFLW2000 \
      --output_json "${JSON_DIR}/semiuhpe.json" \
      --save_viz \
      --viz_dir "${VIZ_DIR}/semiuhpe" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --no_yolo \
      --disable_augment_consistency \
      --enable_raw_augment_consistency \
      --gpu_ids 0 \
      "${SEMI_TEST_CKPT}"
  fi

  if [[ ! -d "${DAD3D_REPO}" ]]; then
    fail_or_continue "missing DAD-3DHeads repo: ${DAD3D_REPO}"
  elif [[ ! -f "${DAD3D_REPO}/predictor.py" || ! -f "${DAD3D_REPO}/demo_utils.py" ]]; then
    fail_or_continue "invalid DAD-3DHeads repo (predictor.py/demo_utils.py missing): ${DAD3D_REPO}"
  elif should_skip_job "${JSON_DIR}/dad3dnet.json"; then
    echo "[$(timestamp_now)] [skip] dad3dnet"
  else
    run_job "dad3dnet" "${FSA_REPO}" \
      python eval_fsanet_dad3dnet_extended_gpu_fix_heavyviz_v4_norot.py \
      --model_type dad3dnet \
      --datasets BIWI,AFLW2000 \
      --biwi_npz "${BIWI_NPZ}" \
      --aflw2000_dir "${AFLW2000_DIR}" \
      --gpu 0 \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --batch_size 32 \
      --max_samples_biwi "${MAX_SAMPLES_BIWI}" \
      --max_samples_aflw "${MAX_SAMPLES_AFLW}" \
      --dad3d_repo "${DAD3D_REPO}" \
      --dad3d_device auto \
      --save_viz \
      --viz_dir "${VIZ_DIR}/dad3dnet" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/dad3dnet.json"
  fi

  if should_skip_job "${JSON_DIR}/hopenet.json"; then
    echo "[$(timestamp_now)] [skip] hopenet"
  else
    run_job "hopenet" "${HOPENET_REPO}" \
      python eval_hopenet_labeled_extended_heavyviz_v3_norot.py \
      --snapshot "${HOPENET_SNAPSHOT}" \
      --datasets BIWI,AFLW2000 \
      --biwi_npz "${BIWI_NPZ}" \
      --aflw2000_dir "${AFLW2000_DIR}" \
      --gpu 0 \
      --seed "${SEED}" \
      --acc_threshold "${ACC_THRESHOLD}" \
      --batch_size 64 \
      --max_samples_biwi "${MAX_SAMPLES_BIWI}" \
      --max_samples_aflw "${MAX_SAMPLES_AFLW}" \
      --save_viz \
      --viz_dir "${VIZ_DIR}/hopenet" \
      --viz_every "${VIZ_EVERY}" \
      --viz_max "${VIZ_MAX}" \
      --axis_size "${AXIS_SIZE}" \
      "${ALL_LABELED_ERRORS_ARGS[@]}" \
      --output_json "${JSON_DIR}/hopenet.json"
  fi
fi

if [[ "${RUN_MODE}" == "parallel" ]]; then
  status=0
  for idx in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[$idx]}"
    name="${JOB_NAMES[$idx]}"
    if wait "${pid}"; then
      echo "[$(timestamp_now)] [done] ${name}"
    else
      echo "[$(timestamp_now)] [failed] ${name}" >&2
      status=1
    fi
  done
  if [[ "${status}" == "0" || "${CONTINUE_ON_ERROR}" == "1" ]]; then
    run_classroom_nogt_eval
    write_summary
    if [[ "${status}" != "0" ]]; then
      echo "[$(timestamp_now)] [warn] some jobs failed, but CONTINUE_ON_ERROR=1"
    fi
    echo "[$(timestamp_now)] [done] all jobs completed"
    echo "run_dir=${RUN_DIR}"
    exit 0
  fi
  exit "${status}"
fi

run_classroom_nogt_eval
write_summary
echo "[$(timestamp_now)] [done] all jobs completed"
echo "run_dir=${RUN_DIR}"
