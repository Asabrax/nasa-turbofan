from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import predictive_maintenance as pipeline
from src import result_cache as cache
from src import tcn_tuning_experiment as tuning


@pytest.fixture
def samples():
    rng = np.random.default_rng(7)
    return rng.normal(size=(12, 30, 2)), np.linspace(0, 125, 12)


@pytest.mark.parametrize("model_type", [pipeline.GRURulModel, pipeline.TCNRulModel])
def test_seed_precedes_initialization_and_repeated_training_matches(samples, model_type):
    x, y = samples
    factory = lambda: model_type(input_size=2, hidden_size=4)
    first, mean, std, _ = pipeline.fit_torch_sequence_model(factory, x, y, max_epochs=2)
    torch.manual_seed(987)
    torch.randn(100)
    second, _, _, _ = pipeline.fit_torch_sequence_model(factory, x, y, max_epochs=2)
    for name, weight in first.state_dict().items():
        torch.testing.assert_close(weight, second.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(
        pipeline.predict_torch_sequence_model(first, x, mean, std),
        pipeline.predict_torch_sequence_model(second, x, mean, std),
    )


def test_checkpoint_uses_validation_and_restores_best_epoch(samples, monkeypatch):
    x, y = samples
    states = []
    original_predict = pipeline.predict_torch_sequence_model

    def capture(model, *args):
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        return original_predict(model, *args)

    scores = iter([8, 3, 5, 6])

    def score(actual, predicted):
        assert actual.max() == 200  # Evaluation targets stay uncapped.
        return next(scores)

    monkeypatch.setattr(pipeline, "predict_torch_sequence_model", capture)
    monkeypatch.setattr(pipeline, "maintenance_validation_score", score)
    model, _, _, epoch = pipeline.fit_torch_sequence_model(
        lambda: pipeline.GRURulModel(2, hidden_size=4), x, y,
        validation_x=x[:2], validation_y=np.array([10, 200]),
        max_epochs=10, patience=2,
    )
    assert epoch == 2
    assert len(states) == 4
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)


def test_full_refit_runs_exact_selected_epoch_budget(samples):
    x, y = samples

    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.calls = 0

        def forward(self, values):
            self.calls += 1
            return self.head(values[:, -1]).squeeze(-1)

    model, _, _, epochs = pipeline.fit_torch_sequence_model(
        CountingModel, x, y, max_epochs=4, patience=1, learning_rate=0,
    )
    assert epochs == model.calls == 4  # Flat training loss must not stop refitting.


@pytest.mark.parametrize("maker", [pipeline.make_sequence_training_set, tuning.make_sequence_snapshot_set])
def test_training_cap_and_raw_validation_targets(maker):
    data = pd.DataFrame({"unit_number": 1, "time_in_cycles": range(1, 201), "sensor_1": 1.0})
    _, capped = maker(data, ["sensor_1"])
    _, raw = maker(data, ["sensor_1"], cap_targets=False)
    assert capped.max() == 125
    assert raw.max() == 170
    assert raw[-1] == 0
    np.testing.assert_array_equal(capped, np.minimum(raw, 125))


def test_engine_split_keeps_trajectories_disjoint():
    data = pd.DataFrame({"unit_number": np.repeat(np.arange(1, 11), 20)})
    train, validation = pipeline.split_engine_units(data)
    assert train and validation
    assert train.isdisjoint(validation)
    assert train | validation == set(range(1, 11))


def test_tuned_candidate_uses_shared_validation_training(samples):
    x, y = samples
    config = {**tuning.TUNING_CONFIGS[1], "hidden_size": 4}
    model, mean, std, epoch = tuning.fit_tcn_candidate(
        config, x, y, max_epochs=2, patience=1,
        validation_x=x[:3], validation_y=np.array([15, 70, 190]),
    )
    assert 1 <= epoch <= 2
    assert np.isfinite(tuning.predict_tcn(model, x, mean, std)).all()


def cache_frame():
    return pd.DataFrame({"subset": pipeline.SUBSETS, "mae": [1., 2., 3., 4.]})


def test_legacy_changed_and_corrupt_caches_are_rejected(tmp_path):
    path = tmp_path / "results.csv"
    frame = cache_frame()
    frame.to_csv(path, index=False)
    assert cache.load_cached_results(path, "v1", {"mae"}) is None
    cache.save_cached_results(frame, path, "v1")
    pd.testing.assert_frame_equal(cache.load_cached_results(path, "v1", {"mae"}), frame)
    assert cache.load_cached_results(path, "v2", {"mae"}) is None
    assert cache.load_cached_results(path, "v1", {"missing"}) is None
    path.write_text(path.read_text() + "FD001,99\n")
    assert cache.load_cached_results(path, "v1", {"mae"}) is None
    cache.save_cached_results(frame.iloc[:3], path, "v1")
    assert cache.load_cached_results(path, "v1", {"mae"}) is None
    path.with_suffix(".meta.json").write_text("invalid json")
    assert cache.load_cached_results(path, "v1", {"mae"}) is None


def test_fingerprint_changes_with_source_data_and_dependency_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src"
    source.mkdir()
    source_file = source / "result_cache.py"
    source_file.write_text("original")
    monkeypatch.setattr(cache, "__file__", str(source_file))
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    for subset in pipeline.SUBSETS:
        for kind in ("train", "test", "RUL"):
            (raw / f"{kind}_{subset}.txt").write_text("data")
    before = cache.experiment_fingerprint()
    assert before == cache.experiment_fingerprint()
    source_file.write_text("changed")
    after_source = cache.experiment_fingerprint()
    assert before != after_source
    (raw / "train_FD001.txt").write_text("changed")
    after_data = cache.experiment_fingerprint()
    assert after_source != after_data
    monkeypatch.setattr(cache, "version", lambda name: "other-version")
    assert after_data != cache.experiment_fingerprint()


def test_comparison_preserves_only_verified_tuned_rows(tmp_path):
    baseline = pd.DataFrame({"model": ["Tuned XGBoost"], "subset": ["FD001"]})
    tuning_path = tmp_path / "tcn_tuning_results.csv"
    cache.save_cached_results(cache_frame(), tuning_path, "v1")
    tuned = pd.read_csv(Path(__file__).parents[1] / "results/tcn_tuned_test_metrics.csv")
    path = tmp_path / "tcn_tuned_test_metrics.csv"
    key = cache.tuned_test_fingerprint("v1", tuning_path)
    cache.save_cached_results(tuned, path, key)
    merged = cache.merge_tuned_comparison(baseline, tmp_path, "v1")
    assert len(merged) == 5
    assert len(cache.merge_tuned_comparison(merged, tmp_path, "v1")) == 5
    pd.testing.assert_frame_equal(cache.merge_tuned_comparison(baseline, tmp_path, "v2"), baseline)
    tuning_path.write_text(tuning_path.read_text() + "\n")
    pd.testing.assert_frame_equal(cache.merge_tuned_comparison(baseline, tmp_path, "v1"), baseline)


def test_tuning_cli_cache_force_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(tuning, "experiment_fingerprint", lambda: "v1")
    calls = {"tune": 0, "test": 0}
    saved_test = pd.read_csv(Path(__file__).parents[1] / "results/tcn_tuned_test_metrics.csv")

    def fake_tune(subset):
        calls["tune"] += 1
        row = {**tuning.TUNING_CONFIGS[0], "subset": subset, "dilations": "1-2-4-8",
               "selected_epochs": 2, "validation_score": 1., "validation_mae": 1.,
               "validation_critical_recall": 1., "validation_critical_misses": 0}
        return pd.DataFrame([row]), row

    def fake_evaluate(subset, config):
        calls["test"] += 1
        assert config["selected_epochs"] == 2
        row = saved_test[saved_test["subset"] == subset].iloc[0].to_dict()
        row["selected_epochs"] = 2
        row["selected_config"] = config["config"]
        return row

    monkeypatch.setattr(tuning, "tune_subset", fake_tune)
    monkeypatch.setattr(tuning, "evaluate_best_on_test", fake_evaluate)
    tuning.main([])
    assert calls == {"tune": 4, "test": 4}
    tuning.main([])
    assert calls == {"tune": 4, "test": 4}
    tuning.main(["--force"])
    assert calls == {"tune": 8, "test": 8}
    monkeypatch.setattr(tuning, "experiment_fingerprint", lambda: "v2")
    tuning.main([])
    assert calls == {"tune": 12, "test": 12}


def test_sequence_workflows_select_on_engines_then_refit(monkeypatch):
    train = pd.DataFrame([
        {"unit_number": unit, "time_in_cycles": cycle, "sensor_1": unit + cycle / 100}
        for unit in range(1, 7) for cycle in range(1, 181)
    ])
    test = train[(train["unit_number"] <= 2) & (train["time_in_cycles"] <= 40)].copy()
    for module in (pipeline, tuning):
        monkeypatch.setattr(module, "load_train_data", lambda subset: train.copy())
        monkeypatch.setattr(module, "load_test_data", lambda subset: test.copy())
        monkeypatch.setattr(module, "load_test_rul", lambda subset: pd.DataFrame({"true_rul": [20, 160]}))

    seen_units = []
    original_windows = pipeline.make_sequence_training_set

    def track_windows(data, *args, **kwargs):
        seen_units.append(set(data["unit_number"]))
        return original_windows(data, *args, **kwargs)

    monkeypatch.setattr(pipeline, "make_sequence_training_set", track_windows)
    original_fit = pipeline.fit_torch_sequence_model
    fit_calls = []

    def short_fit(*args, **kwargs):
        kwargs["max_epochs"] = min(kwargs.get("max_epochs", 100), 2)
        fit_calls.append(kwargs.copy())
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(pipeline, "fit_torch_sequence_model", short_fit)
    report, metrics = pipeline.train_gru_sequence_subset("FD001")
    assert seen_units[0].isdisjoint(seen_units[1])
    assert seen_units[0] | seen_units[1] == seen_units[2] == set(range(1, 7))
    assert fit_calls[0]["validation_y"].max() > 125
    assert "validation_x" not in fit_calls[1]
    assert fit_calls[1]["max_epochs"] == metrics["selected_epochs"]
    assert len(report) == 2
    assert set(report["actual_rul"]) == {20, 160}

    monkeypatch.setattr(tuning, "fit_torch_sequence_model", short_fit)
    monkeypatch.setattr(tuning, "TUNING_CONFIGS", ({**tuning.TUNING_CONFIGS[0], "hidden_size": 4},))
    results, best = tuning.tune_subset("FD001")
    test_metrics = tuning.evaluate_best_on_test("FD001", best)
    assert len(results) == 1
    assert fit_calls[2]["validation_y"].max() > 125
    assert fit_calls[3]["validation_x"] is None
    assert fit_calls[3]["max_epochs"] == best["selected_epochs"]
    assert test_metrics["test_engines"] == 2
    assert np.isfinite(test_metrics["mae"])


def test_main_pipeline_keeps_tuned_model_in_csv_and_dashboard(tmp_path, monkeypatch):
    source_results = Path(__file__).parents[1] / "results"
    reports = pd.read_csv(source_results / "maintenance_report.csv")
    comparisons = pd.read_csv(source_results / "model_comparison.csv")
    metrics = pd.read_csv(source_results / "subset_metrics.csv")
    monkeypatch.setattr(pipeline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "experiment_fingerprint", lambda: "v1")
    tuning_path = tmp_path / "tcn_tuning_results.csv"
    cache.save_cached_results(cache_frame(), tuning_path, "v1")
    tuned = comparisons[comparisons["model"] == "Tuned TCN Sequence Model"]
    cache.save_cached_results(tuned, tmp_path / "tcn_tuned_test_metrics.csv",
                              cache.tuned_test_fingerprint("v1", tuning_path))

    def fake_subset(subset):
        return (reports[reports["subset"] == subset],
                metrics[metrics["subset"] == subset].iloc[0].to_dict(),
                pd.Series({"sensor_1": 1.0}, name=subset),
                pd.DataFrame({"rul": [1, 2], "subset": subset}), cache_frame())

    def fake_sequence(subset, name):
        row = comparisons[(comparisons["subset"] == subset) & (comparisons["model"] == name)]
        return reports[reports["subset"] == subset], row.iloc[0].to_dict()

    monkeypatch.setattr(pipeline, "train_subset", fake_subset)
    monkeypatch.setattr(pipeline, "train_gru_sequence_subset", lambda s: fake_sequence(s, "GRU Sequence Model"))
    monkeypatch.setattr(pipeline, "train_tcn_sequence_subset", lambda s: fake_sequence(s, "TCN Sequence Model"))
    for name in ("save_metrics", "plot_rul_distribution", "plot_predicted_vs_actual",
                 "plot_feature_importance", "plot_risk_summary", "plot_prediction_error",
                 "plot_model_comparison", "save_engine_timeseries"):
        monkeypatch.setattr(pipeline, name, lambda *args: None)
    dashboard_models = []
    monkeypatch.setattr(pipeline, "build_dashboard", lambda r, m, c: dashboard_models.extend(c["model"]))
    pipeline.train_predictive_maintenance_model()
    saved = pd.read_csv(tmp_path / "model_comparison.csv")
    assert len(saved) == 16
    assert dashboard_models.count("Tuned TCN Sequence Model") == 4
    assert set(saved["model"]) == set(pipeline.MODEL_ORDER)
