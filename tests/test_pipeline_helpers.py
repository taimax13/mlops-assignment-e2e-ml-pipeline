from pathlib import Path

from pipeline.evaluate_agent import (
    build_run_config,
    collect_metrics,
    prepare_run_dir,
    run_agent_batch,
    run_swebench_eval,
    summarize_and_log,
)


def test_dry_run_pipeline_creates_reproducible_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.evaluate_agent.RUNS_DIR", tmp_path)

    config = build_run_config(
        {
            "run_id": "unit-dry-run",
            "split": "test",
            "subset": "verified",
            "workers": 2,
            "task_slice": "0:3",
            "dry_run": True,
        }
    )

    run_dir = Path(prepare_run_dir(config))
    preds = Path(run_agent_batch(config, run_dir))
    eval_dir = Path(run_swebench_eval(config, preds, run_dir))
    manifest = Path(summarize_and_log(config, eval_dir))

    assert (run_dir / "config.json").exists()
    assert preds.exists()
    assert (run_dir / "metrics.json").exists()
    assert manifest.exists()
    assert (run_dir / "run-agent" / "trajectories").exists()
    assert collect_metrics(eval_dir)["resolution_rate"] == 1 / 3
