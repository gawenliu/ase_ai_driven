#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _read_input(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of summary dicts")
        return data

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        quality = row.get("quality", {}) if isinstance(row.get("quality"), dict) else {}
        security = row.get("security", {}) if isinstance(row.get("security"), dict) else {}
        effective = row.get("candidate_effective", {}) if isinstance(row.get("candidate_effective"), dict) else {}

        nrow = {
            "candidate_id": row.get("candidate_id"),
            "repeat_index": _to_int(row.get("repeat_index")),
            "batch_id": row.get("batch_id"),
            "status": row.get("status"),
            "overall_score": _to_float(row.get("overall_score")),
            "quality_score": _to_float(row.get("quality_score") if "quality_score" in row else quality.get("code_quality_score")),
            "security_score": _to_float(row.get("security_score") if "security_score" in row else security.get("code_security_score")),
            "stability": _to_float(row.get("stability")),
            "retrieval_top_k": _to_int(row.get("retrieval_top_k") if "retrieval_top_k" in row else effective.get("retrieval_top_k")),
            "include_readme": row.get("include_readme") if "include_readme" in row else effective.get("include_readme"),
            "prompt_template": row.get("prompt_template") if "prompt_template" in row else effective.get("prompt_template"),
            "failure_reason": row.get("failure_reason"),
            "run_log": row.get("run_log"),
        }
        normalized.append(nrow)
    return normalized


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _plot_scatter(rows: List[Dict[str, Any]], out_path: str) -> None:
    valid = [r for r in rows if r["quality_score"] is not None and r["security_score"] is not None]
    fig, ax = plt.subplots(figsize=(10, 6))
    if not valid:
        ax.text(0.5, 0.5, "No valid quality/security points", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = [r["quality_score"] for r in valid]
        y = [r["security_score"] for r in valid]
        ax.scatter(x, y, alpha=0.8)
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Security Score")
        ax.set_title("Security vs Quality")

        top = sorted(
            [r for r in valid if r["overall_score"] is not None],
            key=lambda r: r["overall_score"],
            reverse=True,
        )[:5]
        for row in top:
            label = f"{row.get('candidate_id')}#r{row.get('repeat_index') or 1}"
            ax.annotate(label, (row["quality_score"], row["security_score"]), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_bar_topk(rows: List[Dict[str, Any]], out_path: str) -> None:
    groups: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        k = row.get("retrieval_top_k")
        score = row.get("overall_score")
        if k is None or score is None:
            continue
        groups[int(k)].append(float(score))

    fig, ax = plt.subplots(figsize=(8, 5))
    if not groups:
        ax.text(0.5, 0.5, "No valid top_k groups", ha="center", va="center")
        ax.set_axis_off()
    else:
        ks = sorted(groups.keys())
        vals = [mean(groups[k]) for k in ks]
        ax.bar([str(k) for k in ks], vals)
        ax.set_xlabel("retrieval.top_k")
        ax.set_ylabel("Average Overall Score")
        ax.set_title("Average Overall Score by top_k")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_report(rows: List[Dict[str, Any]], out_path: str, scatter_path: str, bar_path: str) -> None:
    total = len(rows)
    success_rows = [r for r in rows if r.get("status") == "success"]
    fail_rows = [r for r in rows if r.get("status") != "success"]

    top5 = sorted(
        [r for r in success_rows if r.get("overall_score") is not None],
        key=lambda r: r["overall_score"],
        reverse=True,
    )[:5]

    fail_counter = Counter((r.get("failure_reason") or "unknown") for r in fail_rows)

    topk_groups: Dict[int, List[float]] = defaultdict(list)
    for r in success_rows:
        if r.get("retrieval_top_k") is not None and r.get("overall_score") is not None:
            topk_groups[int(r["retrieval_top_k"])].append(float(r["overall_score"]))

    readme_groups: Dict[str, List[float]] = defaultdict(list)
    for r in success_rows:
        include_readme = r.get("include_readme")
        score = r.get("overall_score")
        if score is None:
            continue
        readme_groups[str(include_readme)].append(float(score))

    quality_vals = []
    security_vals = []
    for r in success_rows:
        q = r.get("quality_score")
        s = r.get("security_score")
        if q is None or s is None:
            continue
        quality_vals.append(float(q))
        security_vals.append(float(s))
    corr = _pearson(quality_vals, security_vals)

    lines: List[str] = []
    lines.append("# Evaluation Report Snippet")
    lines.append("")
    lines.append(f"- Candidates/runs analyzed: **{total}**")
    lines.append(f"- Success: **{len(success_rows)}**")
    lines.append(f"- Failed: **{len(fail_rows)}**")
    lines.append("")

    lines.append("## Top-5 by overall_score")
    if not top5:
        lines.append("- No successful runs with valid overall_score.")
    else:
        for row in top5:
            lines.append(
                f"- {row.get('candidate_id')} (repeat={row.get('repeat_index') or 1}, "
                f"batch={row.get('batch_id')}): overall={row.get('overall_score')}, "
                f"quality={row.get('quality_score')}, security={row.get('security_score')}, "
                f"top_k={row.get('retrieval_top_k')}, template={row.get('prompt_template')}"
            )
    lines.append("")

    lines.append("## Observations")
    if topk_groups:
        group_text = ", ".join(
            f"k={k}: avg={mean(v):.2f} (n={len(v)})" for k, v in sorted(topk_groups.items())
        )
        lines.append(f"- top_k effect: {group_text}")
    else:
        lines.append("- top_k effect: insufficient successful data.")

    if readme_groups:
        group_text = ", ".join(
            f"include_readme={k}: avg={mean(v):.2f} (n={len(v)})" for k, v in sorted(readme_groups.items())
        )
        lines.append(f"- include_readme effect: {group_text}")
    else:
        lines.append("- include_readme effect: insufficient successful data.")

    if corr is None:
        lines.append("- quality/security correlation: insufficient data.")
    else:
        lines.append(f"- quality/security Pearson correlation: {corr:.3f}")
    lines.append("")

    lines.append("## Common Failure Reasons")
    if not fail_counter:
        lines.append("- No failed runs.")
    else:
        for reason, cnt in fail_counter.most_common():
            lines.append(f"- {reason}: {cnt}")
    lines.append("")

    lines.append("## Generated Figures")
    lines.append(f"- Security vs Quality scatter: `{os.path.basename(scatter_path)}`")
    lines.append(f"- Average Overall by top_k bar: `{os.path.basename(bar_path)}`")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze consolidated run results and generate report snippet")
    parser.add_argument("--input", required=True, help="Path to summary_all.csv or summary_all.json")
    parser.add_argument("--outdir", default="./outputs_course")
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    rows = _normalize_rows(_read_input(os.path.abspath(args.input)))
    scatter_path = os.path.join(outdir, "security_vs_quality.png")
    bar_path = os.path.join(outdir, "overall_by_topk.png")
    report_path = os.path.join(outdir, "REPORT_SNIPPET.md")

    _plot_scatter(rows, scatter_path)
    _plot_bar_topk(rows, bar_path)
    _build_report(rows, report_path, scatter_path, bar_path)

    print(
        json.dumps(
            {
                "status": "ok",
                "input_rows": len(rows),
                "report": report_path,
                "scatter_png": scatter_path,
                "bar_png": bar_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
