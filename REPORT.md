# Evaluation Pipeline Report

## Summary

This repository turns the original ad-hoc mini-swe-agent and SWE-bench scripts
into a configurable Airflow pipeline for coding-agent evaluation experiments.
The implemented pipeline follows:

```text
prepare_run -> run_agent -> run_eval -> summarize_and_log
```

Each run writes a durable `runs/<run-id>/` folder with configuration,
predictions, trajectories, evaluation output, metrics, and a manifest that
points to the important artifacts.

## Architecture

- `dags/evaluate_agent.py` defines the Airflow DAG `evaluate-agent`.
- `pipeline/evaluate_agent.py` contains reusable helper functions for local
  testing and Airflow tasks.
- `scripts/mini-swe-bench-batch.sh` and `scripts/swe-bench-eval.sh` remain as
  reference ad-hoc commands.
- `runs/sample-dry-run/` is a small committed sample run based on bundled sample
  outputs, useful for reviewing the artifact structure without credentials.

The DAG exposes these Airflow parameters:

- `split`
- `subset`
- `workers`
- `model`
- `task_slice`
- `run_id`
- `cost_limit`
- `dry_run`
- `mlflow_tracking_uri`
- `mlflow_experiment`
- `artifact_uri`

## Artifact Layout

Every run is structured like this:

```text
runs/<run-id>/
  config.json
  run-agent/
    preds.json
    run-agent.log
    trajectories/
  run-eval/
    evaluation_report.json
    run-eval.log
  metrics.json
  manifest.json
```

`manifest.json` is the entry point for reconstructing a run. It records the
config path, prediction path, trajectory directory, evaluation directory,
metrics path, MLflow status, and artifact URI.

## Completed Sample Run

The committed sample run is:

```text
runs/sample-dry-run/
```

It uses the bundled sample mini-swe-agent trajectories and evaluation report.
This validates the orchestration, artifact layout, metric parsing, and manifest
generation without requiring Nebius credentials.

Sample metrics:

```json
{
  "submitted_instances": 3,
  "completed_instances": 3,
  "resolved_instances": 1,
  "unresolved_instances": 2,
  "error_instances": 0,
  "empty_patch_instances": 0,
  "resolution_rate": 0.3333333333333333,
  "completion_rate": 1.0
}
```

## How To Run

Install dependencies:

```bash
uv sync
source .venv/bin/activate
cp .env.example .env
```

Start Airflow:

```bash
bash run-airflow-standalone.sh
```

Open Airflow at:

```text
http://localhost:8080
```

Trigger DAG:

```text
evaluate-agent
```

For a safe local validation run, set:

```text
dry_run = true
run_id = sample-dry-run-local
task_slice = 0:3
workers = 2
```

For a real run, set:

```text
dry_run = false
NEBIUS_API_KEY=<configured in .env>
```

## MLflow

The pipeline supports MLflow logging when MLflow is installed and
`mlflow_tracking_uri` is set.

Example:

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

Then trigger the DAG with:

```text
mlflow_tracking_uri = http://localhost:5000
mlflow_experiment = coding-agent-evals
```

The DAG logs parameters, numeric metrics, `run_id`, and the artifact URI.
If MLflow is not available, the pipeline still writes `metrics.json` and
`manifest.json`, and records the MLflow status in the manifest.

## Evidence

Expected screenshot paths:

```text
screenshots/airflow_dag.png
screenshots/mlflow_runs.png
```

These screenshots should be captured from the Nebius VM after running Airflow
and MLflow with port forwarding:

```bash
ssh -L 8080:localhost:8080 -L 5000:localhost:5000 <vm-host>
```

## Validation

The helper test passes locally:

```bash
python3 -m pytest tests/test_pipeline_helpers.py -p no:cacheprovider
```

Result:

```text
1 passed
```

## Limitations

- The committed run is a dry-run sample, not a fresh live SWE-bench execution.
- Object Storage upload is represented by the `artifact_uri` field and manifest
  structure, but no S3 upload task is implemented in this first iteration.
- Production isolation can be improved by replacing subprocess calls with
  DockerOperator tasks using the provided `Dockerfile`.
