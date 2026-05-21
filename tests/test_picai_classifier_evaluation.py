import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/classification/evaluate_picai_classifier.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_picai_classifier", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_picai_classifier_imports() -> None:
    module = load_module()

    assert module.TARGET_COLUMN == "case_cspca_binary"


def test_feature_columns_from_schema_rejects_unsafe_columns() -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    df = pd.DataFrame(
        {
            "patient_age": [60],
            "case_isup_int": [2],
            "t2w_wholegland_original_firstorder_Mean": [1.0],
        }
    )
    schema = {
        "feature_columns": [
            "patient_age",
            "case_isup_int",
            "t2w_wholegland_original_firstorder_Mean",
        ]
    }

    with pytest.raises(ValueError, match="Unsafe predictor"):
        module.feature_columns_from_schema(schema, df)


def test_resolve_threshold_prefers_selected_model_metric() -> None:
    module = load_module()

    threshold = module.resolve_threshold(
        None,
        {
            "selected_model": "logistic_regression",
            "models": {
                "logistic_regression": {
                    "threshold": 0.37,
                }
            },
        },
    )

    assert threshold == 0.37


def test_make_model_card_contains_decision_support_warning() -> None:
    module = load_module()

    card = module.make_model_card(
        {
            "features": "/tmp/features.csv",
            "model_dir": "/tmp/model",
            "input_shape": [10, 8],
            "feature_count": 5,
            "target_counts": {"0": 6, "1": 4},
            "split": {
                "method": "stratified_case_split",
                "train_cases": 8,
                "validation_cases": 2,
            },
            "metrics": {
                "roc_auc": 0.7,
                "pr_auc": 0.5,
                "sensitivity": 0.6,
                "specificity": 0.8,
                "balanced_accuracy": 0.7,
                "f1": 0.55,
                "confusion_matrix": {
                    "matrix": [[1, 0], [1, 0]],
                },
            },
        }
    )

    assert "Not clinically validated" in card
    assert "not a standalone diagnosis" in card


def test_evaluate_model_smoke(tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    pytest.importorskip("matplotlib")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    train_module_path = Path("scripts/classification/train_picai_baseline_classifier.py")
    spec = importlib.util.spec_from_file_location(
        "train_picai_baseline_classifier", train_module_path
    )
    assert spec is not None
    assert spec.loader is not None
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    eval_module = load_module()

    rows = []
    for idx in range(40):
        label = idx % 2
        rows.append(
            {
                "case_key": f"case_{idx}",
                "patient_id": f"patient_{idx}",
                "study_id": f"study_{idx}",
                "patient_age": 55 + idx % 15,
                "psa": 3.0 + label + idx * 0.01,
                "psad": 0.08 + label * 0.02,
                "prostate_volume": 35 + idx % 10,
                "center": "A" if idx % 3 else "B",
                "case_cspca_binary": label,
                "case_isup_int": 0 if label == 0 else 2,
                "feature_error": "",
                "t2w_wholegland_original_firstorder_Mean": float(label)
                + np.random.default_rng(idx).normal(0, 0.01),
                "t2w_wholegland_original_shape_VoxelVolume": 100.0 + idx,
            }
        )

    features = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(features, index=False)
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "eval"

    train_module.train_baseline_classifier(
        features,
        model_dir,
        validation_size=0.25,
        random_seed=7,
        n_estimators=5,
        overwrite=False,
    )
    payload = eval_module.evaluate_model(
        features,
        model_dir,
        output_dir,
        overwrite=False,
    )

    assert payload["split"]["validation_cases"] == 10
    assert (output_dir / "evaluation_metrics.json").is_file()
    assert (output_dir / "roc_curve.png").is_file()
    assert (output_dir / "precision_recall_curve.png").is_file()
    assert (output_dir / "confusion_matrix.png").is_file()
    assert (output_dir / "model_card_draft.md").is_file()
