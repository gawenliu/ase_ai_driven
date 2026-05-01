#!/usr/bin/env python3
import argparse
import configparser
import csv
import datetime as dt
import glob
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

ALLOWED_TOP_K = [2, 4, 8, 16, 24]
ALLOWED_INCLUDE_README = [True, False]
ALLOWED_TEMPLATES = [
    "default_unified_diff",
    "security_minimal_edit",
    "security_checklist",
]
ALLOWED_TEMPERATURES = [0.0, 0.2, 0.4]
DEFAULT_MAX_TOKENS = 4096
ALLOWED_SECURITY_HINT = [True, False]
ALLOWED_MAX_TOKENS = [2048, 4096, 8192]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_config_ini(path: str) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    parser = configparser.ConfigParser()
    values: Dict[str, str] = {}
    try:
        parser.read(path, encoding="utf-8")
        for key, value in parser.defaults().items():
            values[key.lower()] = value
        for section in parser.sections():
            for key, value in parser.items(section):
                values[key.lower()] = value
        return values
    except configparser.MissingSectionHeaderError:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or raw.startswith(";") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                values[key.strip().lower()] = value.strip()
        return values


def _pick_value(
    cli_value: Optional[str],
    config_values: Dict[str, str],
    config_keys: List[str],
    env_keys: List[str],
) -> Optional[str]:
    if cli_value:
        return cli_value
    for key in config_keys:
        v = config_values.get(key)
        if v:
            return v
    for key in env_keys:
        v = os.getenv(key)
        if v:
            return v
    return None


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return None


def _scan_summary_files(output_root: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in glob.glob(os.path.join(output_root, "*", "summary.json")):
        try:
            s = _read_json(path)
            s["_summary_path"] = path
            results.append(s)
        except Exception:
            continue
    return results


def _score(summary: Dict[str, Any]) -> float:
    try:
        return float(summary.get("overall_score"))
    except Exception:
        return -1.0


def _candidate_fingerprint(
    candidate: Dict[str, Any],
    prompt_optimize_only: bool = False,
) -> Tuple[Any, ...]:
    retrieval = candidate.get("retrieval", {}) if isinstance(candidate.get("retrieval"), dict) else {}
    prompt = candidate.get("prompt", {}) if isinstance(candidate.get("prompt"), dict) else {}
    generation = candidate.get("generation", {}) if isinstance(candidate.get("generation"), dict) else {}
    top_k = int(retrieval.get("top_k", 4))
    include_readme = bool(retrieval.get("include_readme", True))
    template = str(prompt.get("template", "default_unified_diff"))
    temperature = float(generation.get("temperature", 0.2))
    security_hint = bool(prompt.get("security_hint", True))
    max_tokens = int(generation.get("max_tokens", DEFAULT_MAX_TOKENS))
    if prompt_optimize_only:
        return (template, security_hint, max_tokens)
    return (top_k, include_readme, template, temperature)


def _summary_to_candidate(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eff = summary.get("candidate_effective", {})
    top_k = eff.get("retrieval_top_k")
    include_readme = eff.get("include_readme")
    template = eff.get("prompt_template")
    temperature = eff.get("temperature")
    max_token = eff.get("max_gen_token") or DEFAULT_MAX_TOKENS
    if top_k is None or include_readme is None or template is None or temperature is None:
        return None
    return {
        "candidate_id": summary.get("candidate_id"),
        "retrieval": {"method": "bm25", "top_k": int(top_k), "include_readme": bool(include_readme)},
        "context_packing": {"ordering": "by_score", "max_chars": None},
        "prompt": {"template": str(template), "security_hint": True},
        "generation": {"temperature": float(temperature), "max_tokens": int(max_token)},
    }


def _normalize_candidate(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    retrieval = raw.get("retrieval", {}) if isinstance(raw.get("retrieval"), dict) else {}
    prompt = raw.get("prompt", {}) if isinstance(raw.get("prompt"), dict) else {}
    generation = raw.get("generation", {}) if isinstance(raw.get("generation"), dict) else {}

    try:
        top_k = int(retrieval.get("top_k"))
    except Exception:
        return None
    if top_k not in ALLOWED_TOP_K:
        return None

    include_readme = retrieval.get("include_readme")
    if isinstance(include_readme, str):
        include_readme = include_readme.strip().lower() in {"1", "true", "yes", "y", "on"}
    include_readme = bool(include_readme)
    if include_readme not in ALLOWED_INCLUDE_README:
        return None

    template = str(prompt.get("template", "default_unified_diff"))
    if template not in ALLOWED_TEMPLATES:
        return None

    try:
        temperature = float(generation.get("temperature"))
    except Exception:
        return None
    if temperature not in ALLOWED_TEMPERATURES:
        return None

    try:
        max_tokens = int(generation.get("max_tokens", DEFAULT_MAX_TOKENS))
    except Exception:
        max_tokens = DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS

    security_hint = prompt.get("security_hint", True)
    if isinstance(security_hint, str):
        security_hint = security_hint.strip().lower() in {"1", "true", "yes", "y", "on"}
    security_hint = bool(security_hint)
    if security_hint not in ALLOWED_SECURITY_HINT:
        return None
    if max_tokens not in ALLOWED_MAX_TOKENS:
        max_tokens = DEFAULT_MAX_TOKENS

    return {
        "candidate_id": str(raw.get("candidate_id") or ""),
        "retrieval": {"method": "bm25", "top_k": top_k, "include_readme": include_readme},
        "context_packing": {"ordering": "by_score", "max_chars": None},
        "prompt": {"template": template, "security_hint": security_hint},
        "generation": {"temperature": temperature, "max_tokens": max_tokens},
    }


def _format_candidate_id(candidate: Dict[str, Any], round_idx: int, idx: int, used_ids: Set[str]) -> str:
    fp = _candidate_fingerprint(candidate)
    tpl_map = {
        "default_unified_diff": "default",
        "security_minimal_edit": "secmin",
        "security_checklist": "check",
    }
    cid = (
        f"ai_r{round_idx:02d}_k{fp[0]}_"
        f"readme{1 if fp[1] else 0}_t{str(fp[3]).replace('.', '')}_{tpl_map.get(fp[2], 'tpl')}_{idx:02d}"
    )
    if cid not in used_ids:
        return cid
    i = 2
    while f"{cid}_{i}" in used_ids:
        i += 1
    return f"{cid}_{i}"


def _build_prompt(
    per_round: int,
    top_rows: List[Dict[str, Any]],
    bottom_rows: List[Dict[str, Any]],
    used_fingerprints: Set[Tuple[int, bool, str, float]],
) -> str:
    top_text = []
    for x in top_rows:
        c = _summary_to_candidate(x)
        if not c:
            continue
        top_text.append(
            {
                "candidate_id": x.get("candidate_id"),
                "overall_score": x.get("overall_score"),
                "candidate_effective": c,
            }
        )
    bottom_text = []
    for x in bottom_rows:
        c = _summary_to_candidate(x)
        if not c:
            continue
        bottom_text.append(
            {
                "candidate_id": x.get("candidate_id"),
                "overall_score": x.get("overall_score"),
                "candidate_effective": c,
            }
        )

    used_list = []
    for t in sorted(used_fingerprints):
        if len(t) >= 4:
            used_list.append(
                {
                    "top_k": t[0],
                    "include_readme": t[1],
                    "template": t[2],
                    "temperature": t[3],
                }
            )
        elif len(t) == 3:
            used_list.append(
                {
                    "template": t[0],
                    "security_hint": t[1],
                    "max_tokens": t[2],
                }
            )

    prompt = {
        "task": "Propose new candidate configurations for secure patch generation evaluation.",
        "target_new_candidates": per_round,
        "search_goal": "maximize overall_score while maintaining diversity",
        "rules": {
            "top_k_allowed": ALLOWED_TOP_K,
            "include_readme_allowed": ALLOWED_INCLUDE_README,
            "template_allowed": ALLOWED_TEMPLATES,
            "temperature_allowed": ALLOWED_TEMPERATURES,
            "max_tokens_default": DEFAULT_MAX_TOKENS,
            "security_hint_allowed": ALLOWED_SECURITY_HINT,
            "max_tokens_allowed": ALLOWED_MAX_TOKENS,
            "must_be_unique_within_batch": True,
            "must_avoid_used_fingerprints": True,
            "output_format": "JSON array only, each item with retrieval/context_packing/prompt/generation",
        },
        "observations_top": top_text,
        "observations_bottom": bottom_text,
        "already_used_fingerprints": used_list,
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _propose_candidates(
    client: OpenAI,
    model_name: str,
    per_round: int,
    top_rows: List[Dict[str, Any]],
    bottom_rows: List[Dict[str, Any]],
    used_fingerprints: Set[Tuple[int, bool, str, float]],
) -> List[Dict[str, Any]]:
    system = (
        "You are a rigorous ML experiment designer. "
        "Return only valid JSON array. No prose. "
        "Each element must include keys: retrieval, context_packing, prompt, generation."
    )
    user = _build_prompt(per_round, top_rows, bottom_rows, used_fingerprints)
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        max_tokens=1800,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    parsed = _extract_json(content or "")
    if not isinstance(parsed, list):
        return []
    out: List[Dict[str, Any]] = []
    for x in parsed:
        norm = _normalize_candidate(x)
        if norm:
            out.append(norm)
    return out


def _write_candidates(candidates: List[Dict[str, Any]], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for c in candidates:
        cid = c["candidate_id"]
        path = os.path.join(out_dir, f"{cid}.json")
        _write_json(path, c)
        paths.append(path)
    return paths


def _run_many(
    candidates_dir: str,
    dataset: str,
    output_root: str,
    repeat: int,
    max_workers: int,
    parallel_candidates: int,
    config: str,
    num_cycles: Optional[int],
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
        str(repeat),
        "--max_workers",
        str(max_workers),
        "--parallel_candidates",
        str(parallel_candidates),
        "--config",
        config,
    ]
    if num_cycles is not None:
        cmd.extend(["--num_cycles", str(num_cycles)])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return proc.returncode


def _write_consolidated(output_root: str, out_prefix: str = "ai_search") -> Tuple[str, str]:
    summaries = _scan_summary_files(output_root)
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        eff = s.get("candidate_effective", {})
        quality = s.get("quality", {})
        security = s.get("security", {})
        rows.append(
            {
                "candidate_id": s.get("candidate_id"),
                "repeat_index": s.get("repeat_index"),
                "batch_id": s.get("batch_id"),
                "status": s.get("status"),
                "dataset_path": s.get("dataset_path"),
                "runtime_seconds": s.get("runtime_seconds"),
                "overall_score": s.get("overall_score"),
                "quality_score": quality.get("code_quality_score"),
                "security_score": security.get("code_security_score"),
                "stability": s.get("stability"),
                "retrieval_top_k": eff.get("retrieval_top_k"),
                "include_readme": eff.get("include_readme"),
                "prompt_template": eff.get("prompt_template"),
                "temperature": eff.get("temperature"),
                "summary_json": s.get("_summary_path"),
            }
        )
    rows.sort(key=lambda x: (str(x["candidate_id"]), str(x["repeat_index"]), str(x["batch_id"])))

    out_json = os.path.join(output_root, f"{out_prefix}_summary_all.json")
    out_csv = os.path.join(output_root, f"{out_prefix}_summary_all.csv")
    _write_json(out_json, rows)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        headers = list(rows[0].keys()) if rows else [
            "candidate_id",
            "repeat_index",
            "batch_id",
            "status",
            "dataset_path",
            "runtime_seconds",
            "overall_score",
            "quality_score",
            "security_score",
            "stability",
            "retrieval_top_k",
            "include_readme",
            "prompt_template",
            "temperature",
            "summary_json",
        ]
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv, out_json


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-driven candidate search loop for AICGSecEval")
    parser.add_argument("--dataset", default="./data/data_v2_tiny.json")
    parser.add_argument("--output_root", default="./outputs_course")
    parser.add_argument("--seed_candidates_dir", default="./configs/candidates")
    parser.add_argument("--ai_candidates_root", default="./configs/candidates_ai")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per_round", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--parallel_candidates", type=int, default=1)
    parser.add_argument("--config", default="./config.ini")
    parser.add_argument("--num_cycles", type=int, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--search_model_name", default=None)
    parser.add_argument("--prompt_optimize_only", action="store_true")
    parser.add_argument("--fixed_top_k", type=int, default=4)
    parser.add_argument("--fixed_include_readme", default="true")
    parser.add_argument("--fixed_temperature", type=float, default=0.2)
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    output_root = os.path.abspath(args.output_root)
    seed_candidates_dir = os.path.abspath(args.seed_candidates_dir)
    ai_candidates_root = os.path.abspath(args.ai_candidates_root)
    config_values = _load_config_ini(args.config)

    api_key = _pick_value(args.api_key, config_values, ["api_key"], ["API_KEY"])
    base_url = _pick_value(args.base_url, config_values, ["base_url"], ["OPENAI_BASE_URL", "BASE_URL"])
    eval_model_name = _pick_value(args.model_name, config_values, ["model_name"], ["MODEL_NAME"]) or "deepseek-chat"
    search_model_name = args.search_model_name or eval_model_name
    if not api_key or not base_url:
        raise ValueError("Missing api_key/base_url from CLI, config.ini, or environment")

    client = OpenAI(api_key=api_key, base_url=base_url)
    os.makedirs(ai_candidates_root, exist_ok=True)

    # Start from existing candidate fingerprints/ids.
    used_ids: Set[str] = set()
    used_fps: Set[Tuple[int, bool, str, float]] = set()
    for p in glob.glob(os.path.join(seed_candidates_dir, "*.json")) + glob.glob(
        os.path.join(ai_candidates_root, "**", "*.json"), recursive=True
    ):
        try:
            c = _normalize_candidate(_read_json(p))
            if not c:
                continue
            cid = str(_read_json(p).get("candidate_id") or os.path.splitext(os.path.basename(p))[0])
            used_ids.add(cid)
            used_fps.add(_candidate_fingerprint(c, prompt_optimize_only=args.prompt_optimize_only))
        except Exception:
            continue

    iteration_log_path = os.path.join(output_root, "ai_search_iterations.jsonl")
    os.makedirs(output_root, exist_ok=True)

    for round_idx in range(1, max(1, args.rounds) + 1):
        summaries = [s for s in _scan_summary_files(output_root) if s.get("status") == "success"]
        summaries.sort(key=_score, reverse=True)
        top_rows = summaries[:8]
        bottom_rows = sorted(summaries, key=_score)[:8]

        proposals = _propose_candidates(
            client=client,
            model_name=search_model_name,
            per_round=args.per_round,
            top_rows=top_rows,
            bottom_rows=bottom_rows,
            used_fingerprints=used_fps,
        )

        selected: List[Dict[str, Any]] = []
        seen_round: Set[Tuple[int, bool, str, float]] = set()
        for p in proposals:
            fp = _candidate_fingerprint(p, prompt_optimize_only=args.prompt_optimize_only)
            if fp in used_fps or fp in seen_round:
                continue
            seen_round.add(fp)
            selected.append(p)
            if len(selected) >= args.per_round:
                break

        # Fill remaining slots from deterministic fallback sweep.
        if len(selected) < args.per_round:
            for top_k in ALLOWED_TOP_K:
                for include_readme in ALLOWED_INCLUDE_README:
                    for template in ALLOWED_TEMPLATES:
                        for temperature in ALLOWED_TEMPERATURES:
                            for security_hint in ALLOWED_SECURITY_HINT:
                                for max_tokens in ALLOWED_MAX_TOKENS:
                                    c = {
                                        "candidate_id": "",
                                        "retrieval": {"method": "bm25", "top_k": top_k, "include_readme": include_readme},
                                        "context_packing": {"ordering": "by_score", "max_chars": None},
                                        "prompt": {"template": template, "security_hint": security_hint},
                                        "generation": {"temperature": temperature, "max_tokens": max_tokens},
                                    }
                                    fp = _candidate_fingerprint(c, prompt_optimize_only=args.prompt_optimize_only)
                                    if fp in used_fps or fp in seen_round:
                                        continue
                                    seen_round.add(fp)
                                    selected.append(c)
                                    if len(selected) >= args.per_round:
                                        break
                                if len(selected) >= args.per_round:
                                    break
                            if len(selected) >= args.per_round:
                                break
                        if len(selected) >= args.per_round:
                            break
                    if len(selected) >= args.per_round:
                        break
                if len(selected) >= args.per_round:
                    break

        for i, c in enumerate(selected, start=1):
            if args.prompt_optimize_only:
                include_readme_fixed = str(args.fixed_include_readme).strip().lower() in {"1", "true", "yes", "y", "on"}
                c["retrieval"]["top_k"] = int(args.fixed_top_k)
                c["retrieval"]["include_readme"] = bool(include_readme_fixed)
                c["generation"]["temperature"] = float(args.fixed_temperature)
            cid = _format_candidate_id(c, round_idx=round_idx, idx=i, used_ids=used_ids)
            c["candidate_id"] = cid
            used_ids.add(cid)
            used_fps.add(_candidate_fingerprint(c, prompt_optimize_only=args.prompt_optimize_only))

        iter_dir = os.path.join(ai_candidates_root, f"iter_{round_idx:02d}")
        candidate_paths = _write_candidates(selected, iter_dir)
        rc = _run_many(
            candidates_dir=iter_dir,
            dataset=dataset,
            output_root=output_root,
            repeat=max(1, args.repeat),
            max_workers=args.max_workers,
            parallel_candidates=max(1, args.parallel_candidates),
            config=args.config,
            num_cycles=args.num_cycles,
        )

        iter_log = {
            "time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "round": round_idx,
            "search_model_name": search_model_name,
            "eval_model_name": eval_model_name,
            "selected_candidates": [os.path.basename(p) for p in candidate_paths],
            "run_many_return_code": rc,
        }
        with open(iteration_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(iter_log, ensure_ascii=False) + "\n")

    out_csv, out_json = _write_consolidated(output_root, out_prefix="ai_search")
    print(
        json.dumps(
            {
                "status": "ok",
                "rounds": args.rounds,
                "per_round": args.per_round,
                "ai_candidates_root": ai_candidates_root,
                "iterations_log": iteration_log_path,
                "summary_csv": out_csv,
                "summary_json": out_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
