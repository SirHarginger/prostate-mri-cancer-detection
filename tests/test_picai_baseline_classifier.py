import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/classification/train_picai_baseline_classifier.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "train_picai_baseline_classifier", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_picai_baseline_classifier_imports() -> None:
    module = load_module()

    assert module.TARGET_COLUMN == "case_cspca_binary"
    assert module.RADIOMICS_PREFIX == "t2w_wholegland_"


def test_select_feature_columns_uses_allowlist_only() -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    df = pd.DataFrame(
        {
            "case_key": ["a", "b"],
            "patient_id": ["p1", "p2"],
            "study_id": ["s1", "s2"],
            "patient_age": [60, 61],
            "psa": [4.0, 5.0],
            "psad": [0.1, 0.2],
            "prostate_volume": [40, 50],
            "center": ["A", "B"],
            "case_cspca_binary": [0, 1],
            "case_isup_int": [0, 2],
            "feature_error": ["", ""],
            "lesion_mask_voxel_count": [0, 10],
            "t2w_wholegland_original_firstorder_Mean": [1.0, 2.0],
        }
    )

    columns = module.select_feature_columns(df)

    assert columns == [
        "patient_age",
        "psa",
        "psad",
        "prostate_volume",
        "center",
        "t2w_wholegland_original_firstorder_Mean",
    ]
    assert "case_cspca_binary" not in columns
    assert "case_isup_int" not in columns
    assert "lesion_mask_voxel_count" not in columns


def test_assert_no_feature_errors_rejects_failed_rows() -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    df = pd.DataFrame({"feature_error": ["", "mask missing"]})

    with pytest.raises(ValueError, match="feature_error"):
        module.assert_no_feature_errors(df)


def test_validate_target_requires_binary_labels() -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    df = pd.DataFrame({"case_cspca_binary": [0, 1, 2]})

    with pytest.raises(ValueError, match="exactly 0 and 1"):
        module.validate_target(df)


def test_train_baseline_classifier_smoke(tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    module = load_module()

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
    output_dir = tmp_path / "classifier"

    metrics = module.train_baseline_classifier(
        features,
        output_dir,
        validation_size=0.25,
        random_seed=7,
        n_estimators=5,
        overwrite=False,
    )

    assert metrics["selected_model"] in {"logistic_regression", "random_forest"}
    assert (output_dir / "model.joblib").is_file()
    assert (output_dir / "preprocessing_pipeline.joblib").is_file()
    assert (output_dir / "feature_schema.json").is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "training_summary.md").is_file()
