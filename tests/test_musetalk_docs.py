from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "MUSETALK.md"
NOTEBOOK = ROOT / "notebooks" / "AI_Talking_Photo_MuseTalk_Colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/similaitw/ai-talking-photo/"
    "blob/main/notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb"
)


def test_readme_has_musetalk_high_quality_entrypoint() -> None:
    text = README.read_text(encoding="utf-8")
    assert "MuseTalk 1.5 High Quality Colab" in text
    assert COLAB_URL in text
    assert "docs/MUSETALK.md" in text
    assert "至少 4GB" in text
    assert "GTX 1050 2GB" in text


def test_musetalk_guide_documents_isolation_and_no_cpu_fallback() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        ".venv-musetalk",
        "vendor/MuseTalk",
        "0a89dec45a0192b824e3cf4daf96c239440c5ed8",
        "不會因 CUDA 不可用或 OOM 而靜默改成 CPU",
        "use_float16=True",
        "fps=25",
        "batch_size=4",
        "GFPGAN",
    ):
        assert phrase in text


def test_musetalk_guide_is_honest_about_real_gpu_validation_and_license() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert "不代表所有 4GB GPU 都保證成功" in text
    assert "MIT License" in text
    assert "包括商業用途" in text
    assert "其他開源模型仍各自受自己的授權條款約束" in text
    assert "gradio.live" in text


def test_musetalk_notebook_is_tracked() -> None:
    assert NOTEBOOK.is_file()
