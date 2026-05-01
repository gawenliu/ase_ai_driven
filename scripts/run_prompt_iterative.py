#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _safe_rate(x: int, n: int) -> float:
    return (x / n) if n > 0 else 0.0


def _score_from_summary(summary_path: str) -> float:
    s = _read_json(summary_path)
    overall = s.get("overall_score")
    try:
        overall_val = float(overall)
    except Exception:
        overall_val = 0.0

    paths = s.get("paths", {}) if isinstance(s, dict) else {}
    processed = paths.get("processed_instances_json")
    scan = paths.get("scan_results_json")

    total = patch_ok = 0
    if processed and os.path.exists(processed):
        pobj = _read_json(processed)
        if isinstance(pobj, dict):
            for it in pobj.values():
                if isinstance(it, dict):
                    total += 1
                    if bool(it.get("success")):
                        patch_ok += 1

    scan_total = run_ok = test_ok = poc_ok = joint_ok = 0
    if scan and os.path.exists(scan):
        arr = _read_json(scan)
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                scan_total += 1
                rr = bool(it.get("image_status_check"))
                tt = bool(it.get("test_case_check"))
                pp = bool(it.get("poc_check"))
                run_ok += 1 if rr else 0
                test_ok += 1 if tt else 0
                poc_ok += 1 if pp else 0
                joint_ok += 1 if (tt and pp) else 0

    patch_rate = _safe_rate(patch_ok, total) if total > 0 else 0.0
    run_rate = _safe_rate(run_ok, scan_total) if scan_total > 0 else 0.0
    test_rate = _safe_rate(test_ok, scan_total) if scan_total > 0 else 0.0
    poc_rate = _safe_rate(poc_ok, scan_total) if scan_total > 0 else 0.0
    joint_rate = _safe_rate(joint_ok, scan_total) if scan_total > 0 else 0.0

    refined = (
        0.15 * patch_rate
        + 0.15 * run_rate
        + 0.25 * test_rate
        + 0.30 * poc_rate
        + 0.15 * joint_rate
    ) * 100.0
    return (overall_val + refined) / 2.0


def _make_candidate(candidate_id: str, template: str, security_hint: bool, max_tokens: int, temperature: float) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "retrieval": {"method": "bm25", "top_k": 4, "include_readme": True},
        "context_packing": {"ordering": "by_score", "max_chars": None},
        "prompt": {"template": template, "security_hint": security_hint},
        "generation": {"temperature": temperature, "max_tokens": max_tokens},
    }


def _run_many_once(
    candidates_dir: str,
    dataset: str,
    output_root: str,
    parallel_candidates: int,
    num_cycles: int,
    config: str,
) -> int:
    cmd = [
        "python3",
        os.path.join("scripts", "run_many.py"),
        "--candidates_dir",
        candidates_dir,
        "--dataset",
        dataset,
        "--output_root",
        output_root,
        "--repeat",
        "1",
        "--max_workers",
        "1",
        "--parallel_candidates",
        str(parallel_candidates),
        "--config",
        config,
        "--num_cycles",
        str(num_cycles),
        "--force",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.returncode


def _collect_round_scores(output_root: str, round_tag: str) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for p in glob.glob(os.path.join(output_root, f"{round_tag}_*", "summary.json")):
        try:
            s = _read_json(p)
        except Exception:
            continue
        if s.get("status") != "success":
            continue
        cid = str(s.get("candidate_id") or "")
        if not cid.startswith(round_tag + "_"):
            continue
        try:
            score = _score_from_summary(p)
        except Exception:
            score = 0.0
        scores.append((cid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _cfg_from_id(cid: str) -> Optional[Tuple[str, bool, int]]:
    # pattern: ai_rXX_<tpl_short>_h{0/1}_m{tokens}_iYY
    m = re.search(r"_(default|secmin|check)_h([01])_m(\d+)_", cid)
    if not m:
        return None
    tpl_map = {
        "default": "default_unified_diff",
        "secmin": "security_minimal_edit",
        "check": "security_checklist",
    }
    return (tpl_map[m.group(1)], bool(int(m.group(2))), int(m.group(3)))


def _short_tpl(template: str) -> str:
    return {
        "default_unified_diff": "default",
        "security_minimal_edit": "secmin",
        "security_checklist": "check",
    }[template]


def _mk_id(round_idx: int, template: str, security_hint: bool, max_tokens: int, idx: int) -> str:
    return f"ai_r{round_idx:02d}_{_short_tpl(template)}_h{1 if security_hint else 0}_m{max_tokens}_i{idx:02d}"


def _uniq_configs(configs: List[Tuple[str, bool, int]]) -> List[Tuple[str, bool, int]]:
    out: List[Tuple[str, bool, int]] = []
    seen = set()
    for c in configs:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-only iterative optimizer with exploit/mutate policy.")
    parser.add_argument("--dataset", default="./data/data_v2_n5.json")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--candidates_root", required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--per_round", type=int, default=4)
    parser.add_argument("--parallel_candidates", type=int, default=6)
    parser.add_argument("--num_cycles", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--config", default="./config.ini")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(args.candidates_root, exist_ok=True)

    templates = ["default_unified_diff", "security_minimal_edit", "security_checklist"]
    token_levels = [2048, 4096, 8192]
    mutation_order = [
        ("security_minimal_edit", True, 4096),
        ("default_unified_diff", True, 4096),
        ("security_checklist", True, 4096),
        ("security_minimal_edit", True, 2048),
        ("default_unified_diff", True, 8192),
        ("security_checklist", True, 2048),
        ("security_minimal_edit", True, 8192),
    ]

    history_best: List[Tuple[str, float]] = []
    iter_log_path = os.path.join(args.output_root, "prompt_iter_iterations.jsonl")

    for r in range(1, max(1, args.rounds) + 1):
        round_tag = f"ai_r{r:02d}"
        round_dir = os.path.join(args.candidates_root, f"iter_{r:02d}")
        os.makedirs(round_dir, exist_ok=True)

        candidates_cfg: List[Tuple[str, bool, int]] = []

        if r == 1:
            candidates_cfg = [
                ("default_unified_diff", True, 4096),
                ("security_minimal_edit", True, 4096),
                ("security_checklist", True, 4096),
                ("default_unified_diff", True, 2048),
            ]
        else:
            # Exploit top-2 from previous round, then mutate around elites.
            prev_scores = _collect_round_scores(args.output_root, f"ai_r{r-1:02d}")
            elites_cfg: List[Tuple[str, bool, int]] = []
            for cid, _ in prev_scores:
                c = _cfg_from_id(cid)
                if c is not None:
                    elites_cfg.append(c)
                if len(elites_cfg) >= 2:
                    break

            if not elites_cfg:
                elites_cfg = [("security_minimal_edit", True, 4096), ("default_unified_diff", True, 4096)]

            # keep elites
            candidates_cfg.extend(elites_cfg)

            # mutate elite-1 on token dimension
            t0, h0, m0 = elites_cfg[0]
            for mt in token_levels:
                if mt != m0:
                    candidates_cfg.append((t0, h0, mt))

            # mutate template with safe hint
            for tp in templates:
                if tp != t0:
                    candidates_cfg.append((tp, True, 4096))

            # fallback prioritized high-value prompts
            candidates_cfg.extend(mutation_order)

        candidates_cfg = _uniq_configs(candidates_cfg)[: max(1, args.per_round)]
        if len(candidates_cfg) < args.per_round:
            for c in mutation_order:
                if c not in candidates_cfg:
                    candidates_cfg.append(c)
                if len(candidates_cfg) >= args.per_round:
                    break

        written: List[str] = []
        for i, (tpl, hint, mt) in enumerate(candidates_cfg[: args.per_round], start=1):
            cid = _mk_id(r, tpl, hint, mt, i)
            c = _make_candidate(cid, tpl, hint, mt, args.temperature)
            p = os.path.join(round_dir, f"{cid}.json")
            _write_json(p, c)
            written.append(os.path.basename(p))

        rc = _run_many_once(
            candidates_dir=round_dir,
            dataset=os.path.abspath(args.dataset),
            output_root=os.path.abspath(args.output_root),
            parallel_candidates=max(1, args.parallel_candidates),
            num_cycles=max(1, args.num_cycles),
            config=args.config,
        )

        scored = _collect_round_scores(args.output_root, round_tag)
        history_best.extend(scored[:1])

        log_obj = {
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "round": r,
            "selected_candidates": written,
            "run_many_return_code": rc,
            "top_scores": [{"candidate_id": cid, "loop_score": score} for cid, score in scored[:3]],
        }
        with open(iter_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "rounds": args.rounds,
                "per_round": args.per_round,
                "output_root": os.path.abspath(args.output_root),
                "iterations_log": iter_log_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
