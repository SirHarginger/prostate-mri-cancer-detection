import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path("scripts/classification/audit_deep_learning_readiness.py")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_deep_learning_readiness", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_deep_learning_readiness_imports() -> None:
    module = load_module()

    assert module.CORE_MODALITIES == ["t2w", "adc", "hbv"]


def test_read_core_manifest_filters_and_counts(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    manifest = tmp_path / "manifest.csv"
    df = pd.DataFrame(
        [
            {
                "case_key": "case_1",
                "has_core_bpMRI": True,
                "case_cspca_binary": 0,
                "t2w_image_path": "/tmp/case_1_t2w.mha",
                "adc_image_path": "/tmp/case_1_adc.mha",
                "hbv_image_path": "/tmp/case_1_hbv.mha",
            },
            {
                "case_key": "case_2",
                "has_core_bpMRI": "true",
                "case_cspca_binary": 1,
                "t2w_image_path": "/tmp/case_2_t2w.mha",
                "adc_image_path": "/tmp/case_2_adc.mha",
                "hbv_image_path": "/tmp/case_2_hbv.mha",
            },
            {
                "case_key": "case_3",
                "has_core_bpMRI": False,
                "case_cspca_binary": 0,
                "t2w_image_path": "",
                "adc_image_path": "",
                "hbv_image_path": "",
            },
        ]
    )
    df.to_csv(manifest, index=False)

    core = module.read_core_manifest(manifest)

    assert core["case_key"].tolist() == ["case_1", "case_2"]
    assert module.label_counts(core) == {"0": 1, "1": 1}
    assert module.modality_availability(core)["t2w"]["path_present"] == 2


def test_optional_module_status_handles_missing_module() -> None:
    module = load_module()

    status = module.optional_module_status("module_that_should_not_exist_123")

    assert status["installed"] is False
    assert status["version"] is None
    assert status["error"]


def test_build_readiness_summary_is_json_safe(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    module = load_module()

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_key": "case_1",
                "has_core_bpMRI": True,
                "case_cspca_binary": 0,
                "t2w_image_path": "/tmp/case_1_t2w.mha",
                "adc_image_path": "/tmp/case_1_adc.mha",
                "hbv_image_path": "/tmp/case_1_hbv.mha",
            },
            {
                "case_key": "case_2",
                "has_core_bpMRI": True,
                "case_cspca_binary": 1,
                "t2w_image_path": "/tmp/case_2_t2w.mha",
                "adc_image_path": "/tmp/case_2_adc.mha",
                "hbv_image_path": "/tmp/case_2_hbv.mha",
            },
        ]
    ).to_csv(manifest, index=False)

    summary = module.build_readiness_summary(manifest, limit=0)
    output = tmp_path / "summary.json"
    module.write_summary(summary, output)

    assert output.is_file()
    assert summary["total_core_bpmri_cases"] == 2
    assert summary["readiness_verdict"]["ready_for_serious_training"] is False
    assert "torch" in summary["dependencies"]
