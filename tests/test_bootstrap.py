from pathlib import Path

import talking_photo


def test_package_version() -> None:
    assert talking_photo.__version__ == "0.1.0"


def test_required_project_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    required = [
        "AGENTS.md",
        "README.md",
        "docs/SPEC.md",
        "docs/TASKS.md",
        "pyproject.toml",
        "requirements.txt",
        "app.py",
    ]
    for relative in required:
        assert (root / relative).exists(), relative
