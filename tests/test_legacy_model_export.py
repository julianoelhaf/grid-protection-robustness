from types import SimpleNamespace
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sparse_fault_detect.models import run_model


@pytest.fixture
def trained_export(tmp_path, monkeypatch):
    config = SimpleNamespace(
        cluster=SimpleNamespace(port=8080),
        training=SimpleNamespace(n_splits=3, fault_target="fault"),
        model=SimpleNamespace(model="decision_tree_classifier", model_directory="models"),
    )
    x = np.array([[100.], [200.]] * 6)
    labels = pd.DataFrame({"sample_id": np.repeat(np.arange(6), 2), "fault": [0, 1] * 6})
    monkeypatch.setattr(run_model, "load_windows_and_labels", lambda _: (x, labels, True))
    monkeypatch.setattr(run_model.data_sparsity, "apply_sparsity_transform", lambda *_: (x, (12, 1, 1)))
    monkeypatch.setattr(run_model, "configure_mlflow_tags", lambda *_: None)
    monkeypatch.setattr(run_model, "log_dataset_params", lambda *_: None)
    monkeypatch.setattr(run_model.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    tracker = MagicMock()
    monkeypatch.setattr(run_model, "mlflow", tracker)
    models, predictions = [], []
    factory = MagicMock(side_effect=lambda _: DecisionTreeClassifier(random_state=42))
    monkeypatch.setattr(run_model, "create_model_from_name", factory)

    def score(model, test, truth):
        models.append(model)
        predictions.append(model.predict(x))
        return (1., 1., [1., .5, .25][len(models) - 1])

    monkeypatch.setattr(run_model, "evaluate_model", score)
    run_model.main.__wrapped__(config)
    exported = joblib.load(next((tmp_path / "models").glob("*.joblib")))
    return x, models, predictions, exported, tracker, factory


def test_best_fold_is_not_overwritten_by_later_fits(trained_export):
    _, models, _, _, tracker, factory = trained_export
    assert factory.call_count == 3
    assert len({id(model) for model in models}) == 3
    assert len({id(model.steps[-1][1]) for model in models}) == 3
    assert tracker.sklearn.log_model.call_args.args[0] is models[0]


def test_joblib_and_mlflow_exports_include_selected_scaler(trained_export):
    x, models, predictions, exported, tracker, _ = trained_export
    logged = tracker.sklearn.log_model.call_args.args[0]
    for model in (exported, logged):
        assert isinstance(model, Pipeline)
        np.testing.assert_array_equal(model.predict(x), predictions[0])
        np.testing.assert_array_equal(model.steps[0][1].mean_, models[0].steps[0][1].mean_)
        # Raw windows would produce different predictions without the saved scaler.
        assert not np.array_equal(model.steps[-1][1].predict(x), predictions[0])
