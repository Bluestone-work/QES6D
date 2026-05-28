#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple


def load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_cases(ablation_out_root: str, ablation_eval_root: str) -> List[str]:
    cases = set()
    if os.path.isdir(ablation_out_root):
        for name in os.listdir(ablation_out_root):
            p = os.path.join(ablation_out_root, name)
            if os.path.isdir(p) and name.startswith("ablation_"):
                cases.add(name)
    if os.path.isdir(ablation_eval_root):
        for name in os.listdir(ablation_eval_root):
            p = os.path.join(ablation_eval_root, name)
            if not os.path.isdir(p):
                continue
            if name.startswith("ablation_") and name.endswith("_fair_eval"):
                cases.add(name.replace("_fair_eval", ""))
    return sorted(cases)


def summarize_history(path: str) -> Dict[str, Optional[float]]:
    data = load_json(path)
    out = {
        "epochs": None,
        "best_val_avg_mae": None,
        "final_val_avg_mae": None,
        "final_train_loss": None,
    }
    if not isinstance(data, list) or not data:
        return out

    out["epochs"] = len(data)
    val_list = []
    for rec in data:
        v = rec.get("val_avg_mae")
        if isinstance(v, (int, float)):
            val_list.append(float(v))
    if val_list:
        out["best_val_avg_mae"] = min(val_list)
        out["final_val_avg_mae"] = val_list[-1]
    last = data[-1]
    tl = last.get("train_loss")
    if isinstance(tl, (int, float)):
        out["final_train_loss"] = float(tl)
    return out


def extract_eval_avg_mae(json_path: str) -> Optional[float]:
    d = load_json(json_path)
    if not isinstance(d, dict):
        return None
    labeled_eval = d.get("labeled_eval", {})
    if isinstance(labeled_eval, dict):
        v = labeled_eval.get("avg_mae_deg")
        if isinstance(v, (int, float)):
            return float(v)
    labeled_metrics = d.get("labeled_metrics", {})
    if isinstance(labeled_metrics, dict):
        mae_deg = labeled_metrics.get("mae_deg", {})
        if isinstance(mae_deg, dict):
            v = mae_deg.get("avg")
            if isinstance(v, (int, float)):
                return float(v)
    return None


def fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def to_bool_01(v) -> str:
    return "1" if bool(v) else "0"


def get_config_field(cfg: Optional[dict], key: str, default=None):
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def plot_curves(cases: List[dict], output_dir: str) -> Optional[str]:
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        used = 0
        for case in cases:
            case_name = case["case"]
            hpath = case.get("_stage2_history_path")
            data = load_json(hpath) if hpath else None
            if not isinstance(data, list) or not data:
                continue
            epochs = [rec.get("epoch", i + 1) for i, rec in enumerate(data)]
            train_loss = [rec.get("train_loss") for rec in data]
            val_mae = [rec.get("val_avg_mae") for rec in data]

            if any(isinstance(x, (int, float)) for x in train_loss):
                axes[0].plot(
                    epochs,
                    [x if isinstance(x, (int, float)) else None for x in train_loss],
                    label=case_name,
                    linewidth=1.5,
                )
            if any(isinstance(x, (int, float)) for x in val_mae):
                axes[1].plot(
                    epochs,
                    [x if isinstance(x, (int, float)) else None for x in val_mae],
                    label=case_name,
                    linewidth=1.5,
                )
            used += 1

        if used == 0:
            plt.close(fig)
            return None

        axes[0].set_title("Stage-2 Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Stage-2 Val Avg MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (deg)")
        axes[1].grid(alpha=0.3)

        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        out_path = os.path.join(output_dir, "ablation_curves.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        return out_path
    except Exception:
        # Degrade gracefully when matplotlib/numpy backend is broken.
        try:
            plt.close("all")
        except Exception:
            pass
        return None


def infer_case_defaults(case: str) -> dict:
    out = {}
    if "sup_only" in case:
        out.update({
            "train_strategy": "single_stage",
            "exp_setting": "sup_only",
            "pseudo_loss_weight": 0.0,
            "classroom_consistency_weight": 0.0,
        })
    elif "single_stage_full_semi" in case:
        out.update({
            "train_strategy": "single_stage",
            "exp_setting": "full_semi",
        })
    elif "two_stage" in case:
        out.update({
            "train_strategy": "two_stage_refresh",
            "exp_setting": "full_semi",
        })

    if "no_curriculum" in case:
        out["enable_pseudo_curriculum"] = False
    if "no_consistency" in case:
        out["classroom_consistency_weight"] = 0.0
    if "pseudo_weight_005" in case:
        out["pseudo_loss_weight"] = 0.05
    if "pseudo_weight_020" in case:
        out["pseudo_loss_weight"] = 0.2
    if "conf_095" in case:
        out["confidence_threshold"] = 0.95
    if "conf_085" in case:
        out["confidence_threshold"] = 0.85
    if "min_face_56" in case:
        out["min_face_size"] = 56
    if "min_det_conf_05" in case:
        out["min_det_confidence"] = 0.5
    return out


def build_case_record(case: str, out_root: str, eval_root: str, canonical_full_run_dir: str) -> dict:
    case_dir = os.path.join(out_root, case)
    eval_dir = os.path.join(eval_root, f"{case}_fair_eval")
    eval_json_dir = os.path.join(eval_dir, "json")
    canonical_stage1_dir = os.path.join(canonical_full_run_dir, "stage1_sup")

    stage2_cfg = load_json(os.path.join(case_dir, "config.json"))
    stage1_cfg = load_json(os.path.join(case_dir, "stage1_sup", "config.json"))
    canonical_stage2_cfg = load_json(os.path.join(canonical_full_run_dir, "config.json"))
    canonical_stage1_cfg = load_json(os.path.join(canonical_stage1_dir, "config.json"))
    cfg = stage2_cfg if isinstance(stage2_cfg, dict) else stage1_cfg
    if not isinstance(cfg, dict):
        cfg = canonical_stage2_cfg if isinstance(canonical_stage2_cfg, dict) else canonical_stage1_cfg
    case_defaults = infer_case_defaults(case)
    if not isinstance(cfg, dict):
        cfg = case_defaults

    stage2_hist_path = os.path.join(case_dir, "history.json")
    stage1_hist_path = os.path.join(case_dir, "stage1_sup", "history.json")
    if not os.path.isfile(stage2_hist_path) and (
        "single_stage_full_semi" in case or "two_stage_full" in case
    ):
        stage2_hist_path = os.path.join(canonical_full_run_dir, "history.json")
    if not os.path.isfile(stage1_hist_path) and "sup_only" in case:
        stage1_hist_path = os.path.join(canonical_stage1_dir, "history.json")
    stage2_hist = summarize_history(stage2_hist_path)
    stage1_hist = summarize_history(stage1_hist_path)

    pseudo_stats = (
        load_json(os.path.join(case_dir, "pseudo_refresh", "refreshed_pseudo_labels_stats.json"))
        or load_json(os.path.join(case_dir, "pseudo_refresh", "refreshed_pseudo_labels_test_stats.json"))
    )

    qes_biwi = extract_eval_avg_mae(os.path.join(eval_json_dir, "6drepnet_qes_effnetv2_biwi.json"))
    qes_aflw = extract_eval_avg_mae(os.path.join(eval_json_dir, "6drepnet_qes_effnetv2_aflw2000.json"))

    record = {
        "case": case,
        "train_strategy": get_config_field(cfg, "train_strategy", case_defaults.get("train_strategy", "-")),
        "exp_setting": get_config_field(cfg, "exp_setting", case_defaults.get("exp_setting", "-")),
        "pseudo_loss_weight": get_config_field(cfg, "pseudo_loss_weight", case_defaults.get("pseudo_loss_weight")),
        "classroom_consistency_weight": get_config_field(
            cfg, "classroom_consistency_weight", case_defaults.get("classroom_consistency_weight")
        ),
        "confidence_threshold": get_config_field(cfg, "confidence_threshold", case_defaults.get("confidence_threshold")),
        "enable_pseudo_curriculum": to_bool_01(
            get_config_field(cfg, "enable_pseudo_curriculum", case_defaults.get("enable_pseudo_curriculum", False))
        ),
        "min_face_size": get_config_field(cfg, "min_face_size", case_defaults.get("min_face_size")),
        "min_det_confidence": get_config_field(cfg, "min_det_confidence", case_defaults.get("min_det_confidence")),
        "stage1_epochs": stage1_hist.get("epochs"),
        "stage1_best_val_mae": stage1_hist.get("best_val_avg_mae"),
        "stage2_epochs": stage2_hist.get("epochs"),
        "stage2_best_val_mae": stage2_hist.get("best_val_avg_mae"),
        "stage2_final_val_mae": stage2_hist.get("final_val_avg_mae"),
        "pseudo_total_faces": (
            pseudo_stats.get("total_faces")
            if isinstance(pseudo_stats, dict)
            else None
        ),
        "pseudo_high_quality_ratio": (
            pseudo_stats.get("high_quality_ratio")
            if isinstance(pseudo_stats, dict)
            else None
        ),
        "qes_biwi_avg_mae": qes_biwi,
        "qes_aflw_avg_mae": qes_aflw,
        "_stage2_history_path": stage2_hist_path if os.path.isfile(stage2_hist_path) else None,
        "_case_dir": case_dir,
        "_eval_dir": eval_dir,
    }
    return record


def write_csv(records: List[dict], out_csv: str) -> None:
    fields = [
        "case",
        "train_strategy",
        "exp_setting",
        "pseudo_loss_weight",
        "classroom_consistency_weight",
        "confidence_threshold",
        "enable_pseudo_curriculum",
        "min_face_size",
        "min_det_confidence",
        "stage1_epochs",
        "stage1_best_val_mae",
        "stage2_epochs",
        "stage2_best_val_mae",
        "stage2_final_val_mae",
        "pseudo_total_faces",
        "pseudo_high_quality_ratio",
        "qes_biwi_avg_mae",
        "qes_aflw_avg_mae",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fields})


def write_md(records: List[dict], out_md: str, curve_path: Optional[str]) -> None:
    sorted_records = sorted(
        records,
        key=lambda r: (
            float("inf") if r.get("qes_aflw_avg_mae") is None else float(r["qes_aflw_avg_mae"]),
            float("inf") if r.get("qes_biwi_avg_mae") is None else float(r["qes_biwi_avg_mae"]),
        ),
    )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# QES6D Ablation Report\n\n")
        if curve_path:
            f.write(f"- Curves: `{curve_path}`\n")
        f.write("\n")
        f.write("## Metrics + Params\n\n")
        f.write("| Case | Strategy | Exp | PW | ConsW | Conf | Curriculum | BIWI MAE | AFLW MAE | Stage2 Best Val |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted_records:
            f.write(
                "| {case} | {ts} | {exp} | {pw} | {cw} | {conf} | {cur} | {biwi} | {aflw} | {val} |\n".format(
                    case=r["case"],
                    ts=r.get("train_strategy", "-"),
                    exp=r.get("exp_setting", "-"),
                    pw=fmt(r.get("pseudo_loss_weight"), 3),
                    cw=fmt(r.get("classroom_consistency_weight"), 3),
                    conf=fmt(r.get("confidence_threshold"), 3),
                    cur=r.get("enable_pseudo_curriculum", "0"),
                    biwi=fmt(r.get("qes_biwi_avg_mae"), 4),
                    aflw=fmt(r.get("qes_aflw_avg_mae"), 4),
                    val=fmt(r.get("stage2_best_val_mae"), 4),
                )
            )

        f.write("\n## Stage Artifacts Index\n\n")
        for r in sorted_records:
            case_dir = r["_case_dir"]
            eval_dir = r["_eval_dir"]
            f.write(f"### {r['case']}\n")
            stage1_hist = os.path.join(case_dir, "stage1_sup", "history.json")
            stage2_hist = os.path.join(case_dir, "history.json")
            pseudo_stats = os.path.join(case_dir, "pseudo_refresh", "refreshed_pseudo_labels_stats.json")
            pseudo_npz = os.path.join(case_dir, "pseudo_refresh", "refreshed_pseudo_labels_faces.npz")
            paper_csv = os.path.join(eval_dir, "paper_metrics.csv")
            viz_root = os.path.join(eval_dir, "viz")

            f.write(f"- stage1 history: `{stage1_hist}` {'(ok)' if os.path.isfile(stage1_hist) else '(missing)'}\n")
            f.write(f"- pseudo stats: `{pseudo_stats}` {'(ok)' if os.path.isfile(pseudo_stats) else '(missing)'}\n")
            f.write(f"- pseudo faces npz: `{pseudo_npz}` {'(ok)' if os.path.isfile(pseudo_npz) else '(missing)'}\n")
            f.write(f"- stage2 history: `{stage2_hist}` {'(ok)' if os.path.isfile(stage2_hist) else '(missing)'}\n")
            f.write(f"- fair paper csv: `{paper_csv}` {'(ok)' if os.path.isfile(paper_csv) else '(missing)'}\n")
            f.write(f"- fair viz root: `{viz_root}` {'(ok)' if os.path.isdir(viz_root) else '(missing)'}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Build QES6D ablation report with curves + tables + stage index.")
    ap.add_argument("--ablation_out_root", required=True)
    ap.add_argument("--ablation_eval_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--canonical_full_run_dir",
        default="/root/autodl-tmp/output/qes_paper_runs/effnetv2_full_semi_seed",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cases = find_cases(args.ablation_out_root, args.ablation_eval_root)
    records = [
        build_case_record(c, args.ablation_out_root, args.ablation_eval_root, args.canonical_full_run_dir)
        for c in cases
    ]

    out_csv = os.path.join(args.output_dir, "ablation_report.csv")
    out_md = os.path.join(args.output_dir, "ablation_report.md")
    curve_path = plot_curves(records, args.output_dir)
    write_csv(records, out_csv)
    write_md(records, out_md, curve_path)

    print("[DONE] Ablation report generated.")
    print(f"cases={len(records)}")
    print(f"csv={out_csv}")
    print(f"md={out_md}")
    if curve_path:
        print(f"curves={curve_path}")
    else:
        print("curves=skipped (matplotlib unavailable or no history)")


if __name__ == "__main__":
    main()
