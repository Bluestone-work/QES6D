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


def extract_avg_mae(eval_json: str) -> Optional[float]:
    d = load_json(eval_json)
    if not isinstance(d, dict):
        return None
    v = d.get("labeled_eval", {}).get("avg_mae_deg")
    if isinstance(v, (int, float)):
        return float(v)
    v = d.get("labeled_metrics", {}).get("mae_deg", {}).get("avg")
    if isinstance(v, (int, float)):
        return float(v)
    return None


def summarize_history(path: str) -> Tuple[Optional[int], Optional[float], Optional[int]]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        return None, None, None
    best = None
    best_ep = None
    for i, rec in enumerate(data):
        v = rec.get("val_avg_mae")
        if isinstance(v, (int, float)):
            fv = float(v)
            if best is None or fv < best:
                best = fv
                best_ep = int(rec.get("epoch", i + 1))
    return len(data), best, best_ep


def fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def collect_exp1(exp1_eval_root: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isdir(exp1_eval_root):
        return rows
    for name in sorted(os.listdir(exp1_eval_root)):
        if not name.startswith("exp1_label_ratio_") or not name.endswith("_fair_eval"):
            continue
        case = name.replace("_fair_eval", "")
        ratio_token = case.split("_r")[-1]
        ratio = None
        if ratio_token.isdigit():
            ratio = float(ratio_token) / 100.0
        jdir = os.path.join(exp1_eval_root, name, "json")
        rows.append(
            {
                "case": case,
                "pseudo_label_ratio": ratio,
                "biwi_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_biwi.json")),
                "aflw_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_aflw2000.json")),
            }
        )
    rows.sort(key=lambda x: (1e9 if x["pseudo_label_ratio"] is None else x["pseudo_label_ratio"]))
    return rows


def collect_exp2(ablation_out_root: str, ablation_eval_root: str) -> List[dict]:
    cases = [
        "ablation_01_sup_only_single_stage",
        "ablation_02_single_stage_full_semi",
        "ablation_03_two_stage_full",
        "ablation_04_two_stage_no_curriculum",
        "ablation_05_two_stage_no_consistency",
    ]
    rows: List[dict] = []
    for c in cases:
        jdir = os.path.join(ablation_eval_root, f"{c}_fair_eval", "json")
        cfg = load_json(os.path.join(ablation_out_root, c, "config.json")) or {}
        rows.append(
            {
                "case": c,
                "pseudo_loss_weight": cfg.get("pseudo_loss_weight"),
                "classroom_consistency_weight": cfg.get("classroom_consistency_weight"),
                "enable_pseudo_curriculum": cfg.get("enable_pseudo_curriculum"),
                "biwi_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_biwi.json")),
                "aflw_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_aflw2000.json")),
            }
        )
    return rows


def collect_exp4(exp4_out_root: str, exp4_eval_root: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.isdir(exp4_eval_root):
        return rows
    for name in sorted(os.listdir(exp4_eval_root)):
        if not name.startswith("exp4_seed_") or not name.endswith("_fair_eval"):
            continue
        case = name.replace("_fair_eval", "")
        try:
            seed = int(case.split("_")[-1])
        except Exception:
            seed = None
        jdir = os.path.join(exp4_eval_root, name, "json")
        hist = os.path.join(exp4_out_root, case, "history.json")
        n_epochs, best_val, best_epoch = summarize_history(hist)
        rows.append(
            {
                "case": case,
                "seed": seed,
                "epochs": n_epochs,
                "best_val_mae": best_val,
                "best_val_epoch": best_epoch,
                "biwi_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_biwi.json")),
                "aflw_avg_mae": extract_avg_mae(os.path.join(jdir, "6drepnet_qes_effnetv2_aflw2000.json")),
            }
        )
    rows.sort(key=lambda x: (1e9 if x["seed"] is None else x["seed"]))
    return rows


def collect_exp5_hard_subset(ablation_eval_root: str) -> List[dict]:
    mapping = {
        "ablation_01_sup_only_single_stage": "sup_only",
        "ablation_03_two_stage_full": "full",
        "ablation_04_two_stage_no_curriculum": "no_curriculum",
        "ablation_05_two_stage_no_consistency": "no_consistency",
    }
    rows: List[dict] = []
    for case, tag in mapping.items():
        j = os.path.join(
            ablation_eval_root,
            f"{case}_fair_eval",
            "json",
            "6drepnet_qes_effnetv2_biwi.json",
        )
        d = load_json(j) or {}
        lm = d.get("labeled_metrics", {}) if isinstance(d, dict) else {}
        p90 = (lm.get("p90_deg") or {}).get("avg")
        p95 = (lm.get("p95_deg") or {}).get("avg")
        maxv = (lm.get("max_deg") or {}).get("avg")
        worst = lm.get("worst_samples") if isinstance(lm, dict) else None
        worst_name = None
        worst_mae = None
        if isinstance(worst, list) and worst:
            worst_name = worst[0].get("name")
            worst_mae = worst[0].get("avg_mae_deg")
        rows.append(
            {
                "case": case,
                "tag": tag,
                "p90_avg_mae": p90 if isinstance(p90, (int, float)) else None,
                "p95_avg_mae": p95 if isinstance(p95, (int, float)) else None,
                "max_avg_mae": maxv if isinstance(maxv, (int, float)) else None,
                "worst_sample": worst_name,
                "worst_sample_avg_mae": worst_mae if isinstance(worst_mae, (int, float)) else None,
            }
        )
    return rows


def write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_markdown(path: str, exp1: List[dict], exp2: List[dict], exp4: List[dict], exp5: List[dict]) -> None:
    lines: List[str] = []
    lines.append("# QES Experiments 1+2+4+5 Report")
    lines.append("")

    lines.append("## Exp1 Label-Ratio Sweep")
    lines.append("")
    lines.append("| Case | Pseudo Ratio | BIWI MAE | AFLW MAE |")
    lines.append("|---|---:|---:|---:|")
    for r in exp1:
        pr = "-" if r["pseudo_label_ratio"] is None else f"{r['pseudo_label_ratio']:.2f}"
        lines.append(f"| {r['case']} | {pr} | {fmt(r['biwi_avg_mae'])} | {fmt(r['aflw_avg_mae'])} |")
    if not exp1:
        lines.append("| - | - | - | - |")

    lines.append("")
    lines.append("## Exp2 Component Ablation")
    lines.append("")
    lines.append("| Case | PseudoW | ConsW | Curriculum | BIWI MAE | AFLW MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in exp2:
        cur = "-"
        if isinstance(r.get("enable_pseudo_curriculum"), bool):
            cur = "1" if r["enable_pseudo_curriculum"] else "0"
        lines.append(
            f"| {r['case']} | {fmt(r.get('pseudo_loss_weight'),3)} | {fmt(r.get('classroom_consistency_weight'),3)} | {cur} | {fmt(r['biwi_avg_mae'])} | {fmt(r['aflw_avg_mae'])} |"
        )

    lines.append("")
    lines.append("## Exp4 Seed Stability")
    lines.append("")
    lines.append("| Case | Seed | Epochs | Best Val MAE | Best Epoch | BIWI MAE | AFLW MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in exp4:
        lines.append(
            f"| {r['case']} | {r.get('seed') if r.get('seed') is not None else '-'} | {r.get('epochs') if r.get('epochs') is not None else '-'} | {fmt(r.get('best_val_mae'))} | {r.get('best_val_epoch') if r.get('best_val_epoch') is not None else '-'} | {fmt(r.get('biwi_avg_mae'))} | {fmt(r.get('aflw_avg_mae'))} |"
        )
    if not exp4:
        lines.append("| - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Exp5 Hard-Subset (BIWI tail)")
    lines.append("")
    lines.append("| Tag | Case | P90 Avg MAE | P95 Avg MAE | Max Avg MAE | Worst Sample | Worst MAE |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    for r in exp5:
        lines.append(
            f"| {r['tag']} | {r['case']} | {fmt(r.get('p90_avg_mae'))} | {fmt(r.get('p95_avg_mae'))} | {fmt(r.get('max_avg_mae'))} | {r.get('worst_sample') or '-'} | {fmt(r.get('worst_sample_avg_mae'))} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build combined report for QES experiments 1+2+4+5")
    ap.add_argument("--exp1_out_root", default="/root/autodl-tmp/output/qes_exp1_label_ratio")
    ap.add_argument("--exp1_eval_root", default="/root/autodl-tmp/fair_eval_runs/qes_exp1_label_ratio")
    ap.add_argument("--ablation_out_root", default="/root/autodl-tmp/output/qes_ablation_runs")
    ap.add_argument("--ablation_eval_root", default="/root/autodl-tmp/fair_eval_runs/qes_ablation_runs")
    ap.add_argument("--exp4_out_root", default="/root/autodl-tmp/output/qes_exp4_seed_stability")
    ap.add_argument("--exp4_eval_root", default="/root/autodl-tmp/fair_eval_runs/qes_exp4_seed_stability")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    exp1 = collect_exp1(args.exp1_eval_root)
    exp2 = collect_exp2(args.ablation_out_root, args.ablation_eval_root)
    exp4 = collect_exp4(args.exp4_out_root, args.exp4_eval_root)
    exp5 = collect_exp5_hard_subset(args.ablation_eval_root)

    write_csv(
        os.path.join(args.output_dir, "exp1_label_ratio.csv"),
        exp1,
        ["case", "pseudo_label_ratio", "biwi_avg_mae", "aflw_avg_mae"],
    )
    write_csv(
        os.path.join(args.output_dir, "exp2_component_ablation.csv"),
        exp2,
        [
            "case",
            "pseudo_loss_weight",
            "classroom_consistency_weight",
            "enable_pseudo_curriculum",
            "biwi_avg_mae",
            "aflw_avg_mae",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "exp4_seed_stability.csv"),
        exp4,
        ["case", "seed", "epochs", "best_val_mae", "best_val_epoch", "biwi_avg_mae", "aflw_avg_mae"],
    )
    write_csv(
        os.path.join(args.output_dir, "exp5_hard_subset.csv"),
        exp5,
        ["tag", "case", "p90_avg_mae", "p95_avg_mae", "max_avg_mae", "worst_sample", "worst_sample_avg_mae"],
    )
    write_markdown(os.path.join(args.output_dir, "report_1245.md"), exp1, exp2, exp4, exp5)

    print("[DONE] 1+2+4+5 report generated")
    print(f"output_dir={args.output_dir}")
    print(f"exp1_rows={len(exp1)}")
    print(f"exp2_rows={len(exp2)}")
    print(f"exp4_rows={len(exp4)}")
    print(f"exp5_rows={len(exp5)}")


if __name__ == "__main__":
    main()
