import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "AI_Talking_Photo_MuseTalk_Colab.ipynb"


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _text(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_musetalk_notebook_is_clean_valid_nbformat_and_gpu_targeted() -> None:
    notebook = _load()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_musetalk_notebook_run_all_bootstraps_isolated_environments() -> None:
    text = _text(_load())
    assert "similaitw/ai-talking-photo.git" in text
    assert "origin/main" in text
    assert "uv" in text and "3.11" in text and ".colab-venv" in text
    assert "torch==2.5.1" in text
    assert "scripts/setup_musetalk_colab.sh" in text
    assert ".venv-musetalk" in text


def test_musetalk_notebook_surfaces_setup_log_on_failure() -> None:
    text = _text(_load())
    assert "musetalk_setup.log" in text
    assert "CalledProcessError" in text
    assert "最後 120 行" in text
    assert "RuntimeError" in text


def test_musetalk_notebook_launches_share_ui_with_musetalk_default() -> None:
    text = _text(_load())
    assert "TALKING_PHOTO_DEFAULT_BACKEND" in text
    assert "musetalk" in text
    assert "build_app().launch(share=True)" in text
    assert "GFPGAN" in text and "關閉" in text


def test_musetalk_notebook_warns_about_gpu_and_public_share_privacy() -> None:
    text = _text(_load())
    assert "nvidia-smi" in text
    assert "至少 4GB" in text
    assert "GTX 1050 2GB" in text
    assert "gradio.live" in text
    assert "敏感" in text


def test_musetalk_notebook_contains_no_embedded_secrets() -> None:
    text = _text(_load()).lower()
    forbidden = ("api_key=", "api-token", "github_token", "hf_token", "password=")
    assert not any(item in text for item in forbidden)
