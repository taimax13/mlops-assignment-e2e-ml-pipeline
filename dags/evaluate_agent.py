from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task
from airflow.models.param import Param

from pipeline.evaluate_agent import (
    build_run_config,
    prepare_run_dir,
    run_agent_batch,
    run_swebench_eval,
    summarize_and_log,
)


@dag(
    dag_id="evaluate-agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string"),
        "workers": Param(1, type="integer", minimum=1),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:1", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param("0", type="string"),
        "dry_run": Param(False, type="boolean"),
        "mlflow_tracking_uri": Param("", type="string"),
        "mlflow_experiment": Param("coding-agent-evals", type="string"),
        "artifact_uri": Param("", type="string"),
    },
    tags=["swe-bench", "mini-swe-agent", "mlops"],
)
def evaluate_agent_dag():
    @task
    def prepare_run(**context) -> dict:
        conf = dict(context["params"])
        run_config = build_run_config(conf)
        prepare_run_dir(run_config)
        return run_config

    @task
    def run_agent(run_config: dict) -> str:
        return run_agent_batch(run_config)

    @task
    def run_eval(run_config: dict, preds_path: str) -> str:
        return run_swebench_eval(run_config, preds_path)

    @task
    def summarize(run_config: dict, eval_dir: str) -> str:
        return summarize_and_log(run_config, eval_dir)

    config = prepare_run()
    preds = run_agent(config)
    eval_dir = run_eval(config, preds)
    summarize(config, eval_dir)


evaluate_agent_dag()
