from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
SAMPLE_DIR = PROJECT_ROOT / "sample"
DEFAULT_MODEL = "nebius/moonshotai/Kimi-K2.6"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    agent_dir: Path
    eval_dir: Path
    metrics_path: Path
    manifest_path: Path


def slugify(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("-")
    return "".join(safe).strip("-") or "run"


def build_run_config(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    now = datetime.now(timezone.utc)
    model = str(params.get("model") or DEFAULT_MODEL)
    split = str(params.get("split") or "test")
    subset = str(params.get("subset") or "verified")
    workers = int(params.get("workers") or 1)
    task_slice = str(params.get("task_slice") or "0:1")
    cost_limit = str(params.get("cost_limit") if params.get("cost_limit") is not None else "0")
    dry_run = bool(params.get("dry_run", False))
    run_id = str(params.get("run_id") or "")
    if not run_id:
        run_id = slugify(f"{now.strftime('%Y%m%dT%H%M%SZ')}-{subset}-{split}-{task_slice}-{model}")

    return {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "split": split,
        "subset": subset,
        "workers": workers,
        "model": model,
        "task_slice": task_slice,
        "cost_limit": cost_limit,
        "dry_run": dry_run,
        "dataset_name": str(params.get("dataset_name") or "princeton-nlp/SWE-bench_Verified"),
        "agent_config": str(
            params.get("agent_config")
            or "mini-swe-agent/src/minisweagent/config/benchmarks/swebench.yaml"
        ),
        "mlflow_tracking_uri": str(params.get("mlflow_tracking_uri") or os.getenv("MLFLOW_TRACKING_URI", "")),
        "mlflow_experiment": str(params.get("mlflow_experiment") or "coding-agent-evals"),
        "artifact_uri": str(params.get("artifact_uri") or ""),
    }


def paths_for(run_id: str) -> RunPaths:
    run_dir = RUNS_DIR / slugify(run_id)
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        agent_dir=run_dir / "run-agent",
        eval_dir=run_dir / "run-eval",
        metrics_path=run_dir / "metrics.json",
        manifest_path=run_dir / "manifest.json",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_run_dir(run_config: dict[str, Any]) -> str:
    paths = paths_for(run_config["run_id"])
    paths.agent_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(paths.config_path, run_config)
    return str(paths.run_dir)


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"\nexit_code={proc.returncode} elapsed_seconds={time.monotonic() - started:.3f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")


def _copytree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def run_agent_batch(run_config: dict[str, Any], run_dir: str | Path | None = None) -> str:
    paths = paths_for(run_config["run_id"]) if run_dir is None else paths_for(Path(run_dir).name)
    paths.agent_dir.mkdir(parents=True, exist_ok=True)
    preds_path = paths.agent_dir / "preds.json"

    if run_config.get("dry_run"):
        _copytree_contents(SAMPLE_DIR / "trajectories", paths.agent_dir / "trajectories")
        shutil.copy2(SAMPLE_DIR / "trajectories" / "preds.json", preds_path)
        (paths.agent_dir / "run-agent.log").write_text(
            "Dry run: copied bundled sample mini-swe-agent trajectories and preds.json.\n",
            encoding="utf-8",
        )
        return str(preds_path)

    command = [
        "uv",
        "run",
        "mini-extra",
        "swebench",
        "--subset",
        run_config["subset"],
        "--split",
        run_config["split"],
        "--model",
        run_config["model"],
        "--slice",
        run_config["task_slice"],
        "--config",
        run_config["agent_config"],
        "--workers",
        str(run_config["workers"]),
        "-o",
        str(paths.agent_dir / "trajectories"),
    ]
    if run_config.get("cost_limit") not in {None, ""}:
        command.extend(["--cost-limit", str(run_config["cost_limit"])])

    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
    _run_command(command, cwd=PROJECT_ROOT, env=env, log_path=paths.agent_dir / "run-agent.log")

    generated_preds = paths.agent_dir / "trajectories" / "preds.json"
    if not generated_preds.exists():
        raise FileNotFoundError(f"mini-swe-agent did not produce {generated_preds}")
    shutil.copy2(generated_preds, preds_path)
    return str(preds_path)


def run_swebench_eval(
    run_config: dict[str, Any],
    preds_path: str | Path,
    run_dir: str | Path | None = None,
) -> str:
    paths = paths_for(run_config["run_id"]) if run_dir is None else paths_for(Path(run_dir).name)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = paths.eval_dir / "evaluation_report.json"

    if run_config.get("dry_run"):
        shutil.copy2(SAMPLE_DIR / "nebius__moonshotai__Kimi-K2.6.test.json", report_path)
        (paths.eval_dir / "run-eval.log").write_text(
            "Dry run: copied bundled sample SWE-bench evaluation report.\n",
            encoding="utf-8",
        )
        return str(paths.eval_dir)

    command = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        run_config["dataset_name"],
        "--predictions_path",
        str(preds_path),
        "--max_workers",
        str(run_config["workers"]),
        "--run_id",
        run_config["run_id"],
    ]
    _run_command(command, cwd=PROJECT_ROOT, env=os.environ.copy(), log_path=paths.eval_dir / "run-eval.log")
    return str(paths.eval_dir)


def _read_first_json(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def collect_metrics(eval_dir: str | Path) -> dict[str, Any]:
    eval_dir = Path(eval_dir)
    reports = sorted(eval_dir.glob("**/*.json"))
    payload = _read_first_json(reports)

    submitted = int(payload.get("submitted_instances") or payload.get("total_instances") or 0)
    resolved = int(payload.get("resolved_instances") or 0)
    completed = int(payload.get("completed_instances") or 0)
    unresolved = int(payload.get("unresolved_instances") or 0)
    error_instances = int(payload.get("error_instances") or 0)
    empty_patch = int(payload.get("empty_patch_instances") or 0)

    return {
        "submitted_instances": submitted,
        "completed_instances": completed,
        "resolved_instances": resolved,
        "unresolved_instances": unresolved,
        "error_instances": error_instances,
        "empty_patch_instances": empty_patch,
        "resolution_rate": resolved / submitted if submitted else 0.0,
        "completion_rate": completed / submitted if submitted else 0.0,
        "source_report": str(reports[0]) if reports else "",
    }


def write_manifest(run_config: dict[str, Any], metrics: dict[str, Any]) -> str:
    paths = paths_for(run_config["run_id"])
    manifest = {
        "run_id": run_config["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(paths.config_path),
        "agent": {
            "directory": str(paths.agent_dir),
            "predictions": str(paths.agent_dir / "preds.json"),
            "trajectories": str(paths.agent_dir / "trajectories"),
            "log": str(paths.agent_dir / "run-agent.log"),
        },
        "evaluation": {
            "directory": str(paths.eval_dir),
            "log": str(paths.eval_dir / "run-eval.log"),
        },
        "metrics": str(paths.metrics_path),
        "artifact_uri": run_config.get("artifact_uri") or str(paths.run_dir),
        "summary": metrics,
    }
    write_json(paths.manifest_path, manifest)
    return str(paths.manifest_path)


def log_mlflow_run(run_config: dict[str, Any], metrics: dict[str, Any], artifact_uri: str) -> dict[str, Any]:
    status = {"enabled": False, "tracking_uri": run_config.get("mlflow_tracking_uri", ""), "run_id": None}
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"mlflow unavailable: {type(exc).__name__}: {exc}"
        return status

    if run_config.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(run_config["mlflow_tracking_uri"])
    mlflow.set_experiment(run_config.get("mlflow_experiment") or "coding-agent-evals")
    with mlflow.start_run(run_name=run_config["run_id"]) as run:
        mlflow.log_params(
            {
                key: value
                for key, value in run_config.items()
                if key not in {"mlflow_tracking_uri"} and isinstance(value, (str, int, float, bool))
            }
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
        mlflow.set_tag("artifact_uri", artifact_uri)
        mlflow.set_tag("run_id", run_config["run_id"])
        status.update({"enabled": True, "run_id": run.info.run_id})
    return status


def summarize_and_log(run_config: dict[str, Any], eval_dir: str | Path) -> str:
    paths = paths_for(run_config["run_id"])
    metrics = collect_metrics(eval_dir)
    write_json(paths.metrics_path, metrics)
    manifest_path = write_manifest(run_config, metrics)
    mlflow_status = log_mlflow_run(run_config, metrics, run_config.get("artifact_uri") or str(paths.run_dir))
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    manifest["mlflow"] = mlflow_status
    write_json(paths.manifest_path, manifest)
    return manifest_path


def run_pipeline(params: dict[str, Any] | None = None) -> str:
    config = build_run_config(params)
    run_dir = prepare_run_dir(config)
    preds_path = run_agent_batch(config, run_dir)
    eval_dir = run_swebench_eval(config, preds_path, run_dir)
    return summarize_and_log(config, eval_dir)
