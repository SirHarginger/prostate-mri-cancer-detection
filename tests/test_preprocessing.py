from pathlib import Path


def test_preprocessing_directories_exist() -> None:
    assert Path("data/interim").is_dir()
    assert Path("data/processed").is_dir()

