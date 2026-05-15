from pathlib import Path


def test_manifest_directory_exists() -> None:
    assert Path("data/manifests").is_dir()

