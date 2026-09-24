#!/usr/bin/env python3
"""Run 1 AgentDojo condition/suite/attack combination and record the result.

Runs AgentDojo with an undefended agent or the ACF hooks selected by the
condition. For ACF conditions, builds a sidecar with a fresh socket and key.

Requires `pip install agentdojo` (see README.md). --model can be a real
ModelsEnum value against a cloud provider (needs that provider's API key,
e.g. OPENAI_API_KEY), or 'local'/'vllm_parsed' against an OpenAI-compatible
server on localhost (paired with --model-id, no API key needed) -- see
README.md's "Live-tested against a local model" for a real, working example
against Ollama.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
from importlib.metadata import version
import json
import os
import platform
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = REPO_ROOT / "sdk" / "python"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from conditions import CONDITIONS  # noqa: E402
from acf_pipeline import build_pipeline  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _import_agentdojo():
    try:
        import agentdojo  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "agentdojo is not installed. pip install agentdojo (see "
            "benchmarks/agentdojo/README.md)."
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS.keys()))
    parser.add_argument(
        "--suite", required=True, choices=["banking", "slack", "travel", "workspace"]
    )
    parser.add_argument("--attack", required=True, help="AgentDojo attack name, e.g. important_instructions")
    parser.add_argument(
        "--model",
        required=True,
        help="AgentDojo ModelsEnum value, e.g. gpt-4o-mini-2024-07-18 (needs the matching "
        "provider's API key), or 'local'/'vllm_parsed' to talk to an OpenAI-compatible "
        "server on localhost (e.g. Ollama) -- pair with --model-id and, for 'local', the "
        "LOCAL_LLM_PORT env var if not on port 8000. No API key needed for either.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model name the local server serves, e.g. llama3.2:latest. Only used when "
        "--model is 'local' or 'vllm_parsed'.",
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--user-tasks", nargs="*", default=None, help="Subset of user task IDs; default all")
    parser.add_argument("--injection-tasks", nargs="*", default=None, help="Subset of injection task IDs; default all")
    parser.add_argument("--acf-config", type=Path, default=None, help="Sidecar config to use for ACF conditions")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path; default auto-named under results/")
    return parser.parse_args(argv)


def policy_dir_for(config_path: Path) -> Path:
    import yaml

    if not config_path.is_file():
        raise ValueError(f"ACF config missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy_dir = Path(config["policy_dir"])
    return policy_dir if policy_dir.is_absolute() else (config_path.parent / policy_dir).resolve()


def check_tool_allowlists(tool_names: set[str], config_path: Path) -> None:
    """Stop if a demo allowlist would block ordinary benchmark tools."""
    import yaml

    if not config_path.is_file():
        raise ValueError(f"ACF config missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy_file = policy_dir_for(config_path) / "data" / "policy_config.yaml"
    policy = yaml.safe_load(policy_file.read_text(encoding="utf-8"))
    for label, allowed in (
        ("sidecar", config.get("tool_allowlist")),
        ("OPA", policy.get("tool_allowlist")),
    ):
        if label == "sidecar" and not allowed:
            continue
        if label == "OPA" and allowed is None:
            continue
        missing = tool_names - set(allowed or ())
        if missing:
            source = config_path if label == "sidecar" else policy_file
            raise ValueError(
                f"{label} tool allowlist in {source} excludes AgentDojo tools: "
                f"{', '.join(sorted(missing))}"
            )


def file_hashes(config_path: Path) -> dict[str, str]:
    policy_dir = policy_dir_for(config_path)
    files = [config_path, *sorted(
        path for path in policy_dir.rglob("*")
        if path.is_file() and path.suffix in {".rego", ".yaml", ".json"}
    )]
    if len(files) == 1:
        raise ValueError(f"No policies found in {policy_dir}")
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def source_revision() -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", "sidecar", "sdk/python", "benchmarks/agentdojo"],
        cwd=REPO_ROOT,
    )
    return {"commit": commit, "local_diff_sha256": hashlib.sha256(diff).hexdigest()}


def model_runtime(pipeline: Any, model: str) -> dict[str, Any]:
    llm = pipeline.elements[-2] if hasattr(pipeline, "elements") else pipeline
    return {
        "class": type(llm).__name__,
        "checkpoint": getattr(llm, "model", None),
        "temperature": getattr(llm, "temperature", None),
        "top_p": getattr(llm, "top_p", None),
        "seed": "random per request in AgentDojo 0.1.35" if model == "local" else getattr(llm, "seed", None),
    }


@contextmanager
def managed_sidecar(config_path: Path) -> Iterator[tuple[Any, dict[str, Any]]]:
    from acf import Firewall

    if os.name == "nt":
        raise RuntimeError("Managed sidecar runs require a Unix socket; Windows is not supported")
    before = file_hashes(config_path)
    with tempfile.TemporaryDirectory(prefix="acf-agentdojo-") as temp_name:
        temp_dir = Path(temp_name)
        binary = temp_dir / "sidecar"
        socket_path = temp_dir / "acf.sock"
        log_path = temp_dir / "sidecar.log"
        subprocess.run(
            ["go", "build", "-o", str(binary), "./cmd/sidecar"],
            cwd=REPO_ROOT / "sidecar", check=True, capture_output=True, text=True,
        )
        key = secrets.token_bytes(32)
        environment = os.environ.copy()
        environment.update({
            "ACF_CONFIG": str(config_path),
            "ACF_SOCKET_PATH": str(socket_path),
            "ACF_HMAC_KEY": key.hex(),
        })
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(binary)], cwd=REPO_ROOT, env=environment, stdout=log, stderr=log,
            )
            try:
                deadline = time.monotonic() + 10
                while not socket_path.exists():
                    if process.poll() is not None:
                        raise RuntimeError(f"ACF sidecar exited with code {process.returncode}; check the config")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"ACF sidecar did not open {socket_path}")
                    time.sleep(0.05)
                firewall = Firewall(socket_path=str(socket_path), hmac_key=key)
                scanner = firewall._semantic_scanner
                metadata: dict[str, Any] = {
                    "sidecar": "built and started from this checkout",
                    "config_path": str(config_path),
                    "config_and_policy_sha256": before,
                    "semantic_scan": {
                        "enabled": scanner is not None,
                        "backend": type(scanner._backend).__name__ if scanner else None,
                        "model_name": scanner._config.model_name if scanner else None,
                        "signal_threshold": firewall._semantic_signal_threshold,
                    },
                    "tool_output_hook": "on_context proxy with rag provenance",
                }
                yield firewall, metadata
                if file_hashes(config_path) != before:
                    raise RuntimeError("ACF config or policy files changed during the run")
                if process.poll() is not None:
                    raise RuntimeError(f"ACF sidecar exited during the run with code {process.returncode}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


def run(args: argparse.Namespace) -> dict[str, Any]:
    for field in ("user_tasks", "injection_tasks"):
        if getattr(args, field) == []:
            raise ValueError(f"{field} cannot be empty")
    _import_agentdojo()
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import aggregate_results, benchmark_suite_with_injections, benchmark_suite_without_injections
    from agentdojo.task_suite.load_suites import get_suite

    condition = CONDITIONS[args.condition]
    suite = get_suite(args.benchmark_version, args.suite)
    config_path = (
        getattr(args, "acf_config", None) or Path(os.environ.get("ACF_CONFIG", REPO_ROOT / "config" / "sidecar.yaml"))
    ).resolve()
    if "on_tool_call" in condition.acf_hooks:
        check_tool_allowlists({tool.name for tool in suite.tools}, config_path)

    record: dict[str, Any] = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": "benchmarks/agentdojo",
        "condition": {
            "name": condition.name,
            "description": condition.description,
            "acf_hooks": list(condition.acf_hooks),
        },
        "suite": args.suite,
        "attack": args.attack,
        "model": args.model,
        "model_id": args.model_id,
        "user_tasks": args.user_tasks,
        "injection_tasks": args.injection_tasks,
        "benchmark_version": args.benchmark_version,
        "agentdojo_version": version("agentdojo"),
        "source_revision": source_revision(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    # benchmark_suite_with_injections doesn't open a logger context itself --
    # it calls run_task_with(out)_injection_tasks directly, which reads
    # Logger.get() to build a TraceLogger. With no context entered,
    # Logger.get() falls back to a bare, never-__enter__-ed NullLogger()
    # whose .logdir was never set, and that AttributeErrors. Confirmed live
    # against a local Ollama model (see README.md) -- NullLogger's own
    # __enter__ doesn't push onto LOGGER_STACK either, so `with NullLogger():`
    # doesn't fix it; OutputLogger does, via the base Logger.__enter__/__exit__
    # it inherits.
    from agentdojo.logging import OutputLogger

    context = managed_sidecar(config_path) if condition.acf_hooks else nullcontext((None, None))
    with context as (firewall, metadata):
        pipeline = build_pipeline(condition, args.model, model_id=args.model_id, firewall=firewall)
        record["model_runtime"] = model_runtime(pipeline, args.model)
        attack = load_attack(args.attack, suite, pipeline)
        if attack.is_dos_attack:
            if args.injection_tasks is not None:
                raise ValueError("AgentDojo ignores --injection-tasks for DoS attacks; omit the selector")
            record["effective_injection_tasks"] = [next(iter(suite.injection_tasks))]
        else:
            record["effective_injection_tasks"] = (
                args.injection_tasks if args.injection_tasks is not None else list(suite.injection_tasks)
            )
        with OutputLogger(logdir=None):
            clean = benchmark_suite_without_injections(
                agent_pipeline=pipeline,
                suite=suite,
                logdir=None,
                force_rerun=False,
                user_tasks=args.user_tasks,
                benchmark_version=args.benchmark_version,
            )
            results = benchmark_suite_with_injections(
                agent_pipeline=pipeline,
                suite=suite,
                attack=attack,
                logdir=None,
                force_rerun=False,
                user_tasks=args.user_tasks,
                injection_tasks=args.injection_tasks,
                benchmark_version=args.benchmark_version,
            )
        if metadata is not None:
            record["acf_runtime"] = metadata

    record["status"] = "completed"
    record["benign_utility_results"] = {u: v for (u, _), v in clean["utility_results"].items()}
    record["benign_utility_rate"] = aggregate_results([clean["utility_results"]])
    record["utility_results"] = {
        f"{u}|{i}": v for (u, i), v in results["utility_results"].items()
    }
    record["security_results"] = {
        f"{u}|{i}": v for (u, i), v in results["security_results"].items()
    }
    record["injection_tasks_utility_results"] = dict(results["injection_tasks_utility_results"])
    record["utility_under_attack_rate"] = aggregate_results([results["utility_results"]])
    # AgentDojo's security oracle returns True when the attacker succeeds.
    record["attack_success_rate"] = aggregate_results([results["security_results"]])
    return record


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = run(args)

    out = args.out
    if out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        run_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
        out = RESULTS_DIR / f"{args.condition}_{args.suite}_{args.attack}_{run_id}.json"
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
