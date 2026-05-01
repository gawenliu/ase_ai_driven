#!/usr/bin/env python3
import argparse
import configparser
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

SUPPORTED_PROMPT_TEMPLATES = {
    "default_unified_diff",
    "security_minimal_edit",
    "security_checklist",
}


def _utc_batch_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("batch_%Y%m%d_%H%M%S")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_config_ini(path: Optional[str]) -> Dict[str, str]:
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
        # Support simple key=value files without INI sections.
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or raw.startswith(";"):
                    continue
                if "=" not in raw:
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
        if key in config_values and config_values[key]:
            return config_values[key]
    for env_key in env_keys:
        env_val = os.getenv(env_key)
        if env_val:
            return env_val
    return None


def _normalize_base_url(base_url: str) -> str:
    val = (base_url or "").strip()
    if not val:
        return val
    if not val.endswith("/"):
        val += "/"
    return val


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return True
        if low in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _build_candidate_adapter(candidate_config: Dict[str, Any]) -> Dict[str, Any]:
    retrieval = candidate_config.get("retrieval", {}) if isinstance(candidate_config.get("retrieval"), dict) else {}
    prompt = candidate_config.get("prompt", {}) if isinstance(candidate_config.get("prompt"), dict) else {}
    generation = candidate_config.get("generation", {}) if isinstance(candidate_config.get("generation"), dict) else {}

    invoke_overrides: Dict[str, Any] = {}
    knob_application: Dict[str, Dict[str, Any]] = {}
    effective: Dict[str, Any] = {
        "retrieval_top_k": None,
        "include_readme": None,
        "prompt_template": None,
        "prompt_security_hint": None,
        "temperature": None,
        "max_gen_token": None,
    }

    top_k = retrieval.get("top_k")
    if isinstance(top_k, int) and top_k > 0:
        invoke_overrides["retrieval_top_k"] = top_k
        effective["retrieval_top_k"] = top_k
        knob_application["retrieval.top_k"] = {"value": top_k, "applied": True, "reason": None}
    else:
        knob_application["retrieval.top_k"] = {"value": top_k, "applied": False, "reason": "missing_or_invalid"}

    include_readme = _to_bool(retrieval.get("include_readme"))
    if include_readme is not None:
        invoke_overrides["include_readme"] = 1 if include_readme else 0
        effective["include_readme"] = include_readme
        knob_application["retrieval.include_readme"] = {"value": include_readme, "applied": True, "reason": None}
    else:
        knob_application["retrieval.include_readme"] = {
            "value": retrieval.get("include_readme"),
            "applied": False,
            "reason": "missing_or_invalid",
        }

    prompt_template = prompt.get("template")
    if isinstance(prompt_template, str) and prompt_template in SUPPORTED_PROMPT_TEMPLATES:
        invoke_overrides["prompt_template"] = prompt_template
        effective["prompt_template"] = prompt_template
        knob_application["prompt.template"] = {"value": prompt_template, "applied": True, "reason": None}
    else:
        knob_application["prompt.template"] = {
            "value": prompt_template,
            "applied": False,
            "reason": "missing_or_unsupported",
        }

    security_hint = _to_bool(prompt.get("security_hint"))
    if security_hint is not None:
        invoke_overrides["prompt_security_hint"] = 1 if security_hint else 0
        effective["prompt_security_hint"] = security_hint
        knob_application["prompt.security_hint"] = {"value": security_hint, "applied": True, "reason": None}
    else:
        knob_application["prompt.security_hint"] = {
            "value": prompt.get("security_hint"),
            "applied": False,
            "reason": "missing_or_invalid",
        }

    temperature = generation.get("temperature")
    if isinstance(temperature, (int, float)):
        temp_val = float(temperature)
        invoke_overrides["temperature"] = temp_val
        effective["temperature"] = temp_val
        knob_application["generation.temperature"] = {"value": temp_val, "applied": True, "reason": None}
    else:
        knob_application["generation.temperature"] = {
            "value": temperature,
            "applied": False,
            "reason": "missing_or_invalid",
        }

    max_tokens = generation.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        invoke_overrides["max_gen_token"] = max_tokens
        effective["max_gen_token"] = max_tokens
        knob_application["generation.max_tokens"] = {"value": max_tokens, "applied": True, "reason": None}
    else:
        knob_application["generation.max_tokens"] = {
            "value": max_tokens,
            "applied": False,
            "reason": "missing_or_invalid",
        }

    return {
        "invoke_overrides": invoke_overrides,
        "knob_application": knob_application,
        "effective": effective,
    }


def _redact_command(cmd: List[str]) -> List[str]:
    redacted: List[str] = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        redacted.append(token)
        if token in ("--api_key", "--github_token") and i + 1 < len(cmd):
            redacted.append("***REDACTED***")
            i += 2
            continue
        i += 1
    return redacted


def _append_results_csv(results_csv: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    patch_applied = summary.get("quality", {}).get("patch_applied_success", {})
    build_flags = summary.get("quality", {}).get("build_or_static_check_success", {})
    security = summary.get("security", {})
    paths = summary.get("paths", {})
    effective = summary.get("candidate_effective", {})

    row = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate_id": summary.get("candidate_id"),
        "repeat_index": summary.get("repeat_index"),
        "batch_id": summary.get("batch_id"),
        "status": summary.get("status"),
        "dataset_path": summary.get("dataset_path"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "patch_applied_success_count": patch_applied.get("count"),
        "patch_applied_total": patch_applied.get("total"),
        "patch_applied_success_rate": patch_applied.get("rate"),
        "image_status_check_rate": build_flags.get("image_status_check_rate"),
        "test_case_check_rate": build_flags.get("test_case_check_rate"),
        "poc_check_success_rate": security.get("poc_check_success_rate"),
        "secure_and_quality_success_rate": security.get("secure_and_quality_success_rate"),
        "code_quality_score": summary.get("quality", {}).get("code_quality_score"),
        "code_security_score": security.get("code_security_score"),
        "stability": summary.get("stability"),
        "overall_score": summary.get("overall_score"),
        "retrieval_top_k": effective.get("retrieval_top_k"),
        "include_readme": effective.get("include_readme"),
        "prompt_template": effective.get("prompt_template"),
        "temperature": effective.get("temperature"),
        "max_gen_token": effective.get("max_gen_token"),
        "output_dir": paths.get("output_dir"),
        "summary_json": paths.get("summary_json"),
        "run_log": paths.get("run_log"),
    }

    headers = list(row.keys())
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_run_log(
    run_log_path: str,
    cmd: List[str],
    returncode: Optional[int],
    runtime_seconds: float,
    stdout_text: str,
    stderr_text: str,
) -> None:
    with open(run_log_path, "w", encoding="utf-8") as f:
        f.write(f"started_at_utc: {dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        f.write(f"command: {' '.join(_redact_command(cmd))}\n")
        f.write(f"return_code: {returncode}\n")
        f.write(f"runtime_seconds: {runtime_seconds:.6f}\n")
        f.write("----- STDOUT -----\n")
        f.write(stdout_text or "")
        f.write("\n----- STDERR -----\n")
        f.write(stderr_text or "")
        f.write("\n")


def _fallback_summary(
    args: argparse.Namespace,
    candidate_id: str,
    batch_id: str,
    output_dir: str,
    runtime_seconds: float,
    status: str,
) -> Dict[str, Any]:
    run_log_path = os.path.join(output_dir, "run.log")
    summary = {
        "candidate_id": candidate_id,
        "batch_id": batch_id,
        "dataset_path": os.path.abspath(args.dataset),
        "runtime_seconds": round(runtime_seconds, 6),
        "status": status,
        "model_name": args.model_name,
        "quality": {
            "patch_applied_success": {
                "count": None,
                "total": None,
                "rate": None,
                "any_success": None,
                "all_success": None,
            },
            "build_or_static_check_success": {
                "image_status_check_rate": None,
                "test_case_check_rate": None,
            },
            "code_quality_score": None,
        },
        "security": {
            "poc_check_success_rate": None,
            "secure_and_quality_success_rate": None,
            "code_security_score": None,
            "security_alerts_delta": None,
        },
        "stability": None,
        "overall_score": None,
        "paths": {
            "output_dir": output_dir,
            "run_log": run_log_path if os.path.exists(run_log_path) else None,
            "summary_json": os.path.join(output_dir, "summary.json"),
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one AICGSecEval evaluation and summarize outputs")
    parser.add_argument("--candidate", required=True, help="Candidate config JSON path")
    parser.add_argument("--dataset", required=True, help="Dataset path (tiny dataset recommended)")
    parser.add_argument("--output_root", default="./outputs_course", help="Output root directory")
    parser.add_argument("--batch_id", default=None, help="Optional batch id, auto-generated if omitted")
    parser.add_argument("--model_name", default=None, help="LLM model name")
    parser.add_argument("--base_url", default=None, help="LLM API base URL")
    parser.add_argument("--api_key", default=None, help="LLM API key")
    parser.add_argument("--github_token", default=None, help="GitHub token for cloning")
    parser.add_argument("--retrieval_data_path", default="./data/data_v2_context_bm25.jsonl")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--num_cycles", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_gen_token", type=int, default=None)
    parser.add_argument("--repeat_index", type=int, default=1)
    parser.add_argument("--config", default="./config.ini", help="Optional ini config file")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_path = os.path.abspath(args.candidate)
    dataset_path = os.path.abspath(args.dataset)
    output_root = os.path.abspath(args.output_root)
    batch_id = args.batch_id or _utc_batch_id()
    output_dir = os.path.join(output_root, batch_id)
    os.makedirs(output_dir, exist_ok=True)

    candidate_config = _read_json(candidate_path)
    if not isinstance(candidate_config, dict):
        raise ValueError("Candidate config must be a JSON object")
    candidate_adapter = _build_candidate_adapter(candidate_config)
    invoke_overrides = candidate_adapter["invoke_overrides"]
    knob_application = candidate_adapter["knob_application"]
    candidate_effective = candidate_adapter["effective"]

    candidate_id = str(candidate_config.get("candidate_id") or os.path.splitext(os.path.basename(candidate_path))[0])
    config_values = _load_config_ini(os.path.abspath(args.config) if args.config else None)

    model_name = _pick_value(args.model_name, config_values, ["model_name"], ["MODEL_NAME"])
    base_url = _pick_value(
        args.base_url,
        config_values,
        ["base_url", "api_base", "endpoint", "deepseek_base_url"],
        ["BASE_URL", "OPENAI_BASE_URL", "LLM_BASE_URL"],
    )
    if base_url:
        base_url = _normalize_base_url(base_url)
    api_key = _pick_value(
        args.api_key,
        config_values,
        ["api_key", "deepseek_api_key", "llm_api_key"],
        ["API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"],
    )
    github_token = _pick_value(
        args.github_token,
        config_values,
        ["github_token"],
        ["GITHUB_TOKEN"],
    )

    missing = []
    if not model_name:
        missing.append("model_name")
    if not base_url:
        missing.append("base_url")
    if not api_key:
        missing.append("api_key")
    if missing:
        raise ValueError(f"Missing required values: {missing}. Provide via CLI, config.ini, or env")

    # Keep a copy of the candidate config used by this run.
    shutil.copy2(candidate_path, os.path.join(output_dir, "candidate.json"))

    invoke_cmd: List[str] = [
        "python3",
        "invoke.py",
        "--llm",
        "--model_name",
        model_name,
        "--base_url",
        base_url,
        "--api_key",
        api_key,
        "--batch_id",
        batch_id,
        "--dataset_path",
        dataset_path,
        "--retrieval_data_path",
        os.path.abspath(args.retrieval_data_path),
        "--output_dir",
        output_dir,
        "--max_workers",
        str(args.max_workers),
        "--num_cycles",
        str(args.num_cycles),
    ]
    # Candidate adapter layer: map candidate knobs to invoke arguments.
    top_k = invoke_overrides.get("retrieval_top_k")
    if top_k is not None:
        invoke_cmd.extend(["--retrieval_top_k", str(top_k)])

    include_readme = invoke_overrides.get("include_readme")
    if include_readme is not None:
        invoke_cmd.extend(["--include_readme", str(include_readme)])

    prompt_template = invoke_overrides.get("prompt_template")
    if prompt_template:
        invoke_cmd.extend(["--prompt_template", str(prompt_template)])

    prompt_security_hint = invoke_overrides.get("prompt_security_hint")
    if prompt_security_hint is not None:
        invoke_cmd.extend(["--prompt_security_hint", str(prompt_security_hint)])

    # Generation params: candidate defaults, CLI can override.
    temperature = args.temperature if args.temperature is not None else invoke_overrides.get("temperature")
    if temperature is not None:
        invoke_cmd.extend(["--temperature", str(temperature)])
        candidate_effective["temperature"] = float(temperature)
        knob_application["generation.temperature"] = {"value": float(temperature), "applied": True, "reason": None}

    max_gen_token = args.max_gen_token if args.max_gen_token is not None else invoke_overrides.get("max_gen_token")
    if max_gen_token is not None:
        invoke_cmd.extend(["--max_gen_token", str(int(max_gen_token))])
        candidate_effective["max_gen_token"] = int(max_gen_token)
        knob_application["generation.max_tokens"] = {"value": int(max_gen_token), "applied": True, "reason": None}

    if github_token:
        invoke_cmd.extend(["--github_token", github_token])

    start = time.monotonic()
    proc = None
    stdout_text = ""
    stderr_text = ""
    status = "failed"
    try:
        proc = subprocess.run(
            invoke_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        status = "success" if proc.returncode == 0 else "failed"
    except Exception as exc:
        stderr_text = f"Exception while running invoke.py: {exc}\n"
    runtime_seconds = time.monotonic() - start

    stdout_log = os.path.join(output_dir, "invoke_stdout.log")
    stderr_log = os.path.join(output_dir, "invoke_stderr.log")
    with open(stdout_log, "w", encoding="utf-8") as f:
        f.write(stdout_text)
    with open(stderr_log, "w", encoding="utf-8") as f:
        f.write(stderr_text)

    run_log_path = os.path.join(output_dir, "run.log")
    _write_run_log(
        run_log_path,
        invoke_cmd,
        proc.returncode if proc else None,
        runtime_seconds,
        stdout_text,
        stderr_text,
    )

    summary_path = os.path.join(output_dir, "summary.json")
    parse_cmd = [
        "python3",
        os.path.join(repo_root, "scripts", "parse_results.py"),
        "--output_dir",
        output_dir,
        "--candidate_id",
        candidate_id,
        "--batch_id",
        batch_id,
        "--dataset_path",
        dataset_path,
        "--runtime_seconds",
        f"{runtime_seconds:.6f}",
        "--status",
        status,
        "--model_name",
        model_name,
        "--summary_out",
        summary_path,
    ]

    summary: Dict[str, Any]
    try:
        parse_proc = subprocess.run(
            parse_cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if parse_proc.returncode != 0 or not os.path.exists(summary_path):
            summary = _fallback_summary(args, candidate_id, batch_id, output_dir, runtime_seconds, status)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        else:
            summary = _read_json(summary_path)
    except Exception:
        summary = _fallback_summary(args, candidate_id, batch_id, output_dir, runtime_seconds, status)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    summary.setdefault("paths", {})
    summary["repeat_index"] = args.repeat_index
    summary["candidate"] = candidate_config
    summary["candidate_effective"] = candidate_effective
    summary["knob_application"] = knob_application
    summary["paths"]["summary_json"] = summary_path
    summary["paths"]["run_log"] = run_log_path
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    results_csv = os.path.join(output_root, "results.csv")
    _append_results_csv(results_csv, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "success" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[run_eval] error: {exc}", file=sys.stderr)
        raise
