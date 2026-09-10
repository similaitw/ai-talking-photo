from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COLAB_DOC = ROOT / "docs" / "COLAB.md"
NOTEBOOK = "notebooks/AI_Talking_Photo_Colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/similaitw/ai-talking-photo/"
    f"blob/main/{NOTEBOOK}"
)


def test_readme_has_colab_entrypoint() -> None:
    text = README.read_text(encoding="utf-8")
    assert "Open In Colab" in text
    assert COLAB_URL in text
    assert "docs/COLAB.md" in text
    assert "全部執行" in text


def test_colab_doc_covers_operation_and_limits() -> None:
    text = COLAB_DOC.read_text(encoding="utf-8")
    for required in (
        "變更執行階段類型",
        "全部執行",
        "gradio.live",
        "Doctor",
        "CUDA OOM",
        "資源不是保證資源",
        "公開",
        "非商業用途",
    ):
        assert required in text


def test_colab_links_target_tracked_notebook() -> None:
    assert (ROOT / NOTEBOOK).is_file()
    readme = README.read_text(encoding="utf-8")
    guide = COLAB_DOC.read_text(encoding="utf-8")
    assert readme.count(COLAB_URL) >= 1
    assert guide.count(COLAB_URL) >= 1
