import json
from pathlib import Path

NOTEBOOK = Path("notebooks/AI_Talking_Photo_Colab.ipynb")


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _all_source(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        chunks.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(chunks)


def test_colab_notebook_is_valid_and_clean() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_colab_notebook_is_run_all_bootstrap() -> None:
    source = _all_source(_notebook())
    required = [
        "https://github.com/similaitw/ai-talking-photo.git",
        'subprocess.run([UV, "python", "install", "3.11"]',
        "torch==2.5.1",
        "https://download.pytorch.org/whl/cu118",
        "requirements-wav2lip-lock.txt",
        "https://github.com/Rudrabha/Wav2Lip.git",
        "bac9a81e63ecc153202353372e5724b83d9e6322",
        "15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk",
        "s3fd-619a316812.pth",
        "prepare_wav2lip.py",
        "doctor.py",
        "share=True",
    ]
    for item in required:
        assert item in source


def test_colab_notebook_requires_gpu_and_does_not_embed_secrets() -> None:
    source = _all_source(_notebook())
    assert "nvidia-smi" in source
    assert "未偵測到 NVIDIA GPU" in source
    forbidden = [
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "api_key=",
        "password=",
    ]
    for item in forbidden:
        assert item not in source
