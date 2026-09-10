from app import PIPELINE_NOT_READY, build_app, pipeline_not_ready


def test_pipeline_placeholder_is_explicit() -> None:
    audio, video, status = pipeline_not_ready(None, "測試", "台灣女聲", 0.95)
    assert audio is None
    assert video is None
    assert status == PIPELINE_NOT_READY


def test_gradio_shell_contains_required_components() -> None:
    demo = build_app()
    config = demo.get_config_file()
    components = config["components"]

    component_types = {component["type"] for component in components}
    labels = {
        component.get("props", {}).get("label")
        for component in components
        if component.get("props", {}).get("label")
    }

    assert {"image", "textbox", "dropdown", "slider", "audio", "video"} <= component_types
    assert {"人物照片", "講稿", "聲音", "語速", "語音預覽", "影片預覽"} <= labels
