from pathlib import Path


def test_outputs_metrics_directory_exists() -> None:
    assert Path("outputs/metrics").is_dir()

