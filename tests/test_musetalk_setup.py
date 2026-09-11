from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup_musetalk_colab.sh"
DOWNLOADER = ROOT / "scripts" / "download_musetalk_models.py"


def test_setup_pins_official_musetalk_and_python310() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "https://github.com/TMElyralab/MuseTalk.git" in text
    assert "0a89dec45a0192b824e3cf4daf96c239440c5ed8" in text
    assert 'uv" python install 3.10' in text or '"$UV" python install 3.10' in text
    assert '"$UV" venv --python 3.10 --seed' in text
    assert ".venv-musetalk" in text


def test_setup_uses_openmim_compatible_packaging_tools() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert '"pip==24.0"' in text
    assert '"setuptools==69.5.1"' in text
    assert "openmim" in text
    assert "musetalk_setup.log" in text
    assert "最後 120 行" in text


def test_setup_verifies_packaging_versions_without_importing_setuptools() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "from importlib.metadata import version" in text
    assert "version('pip')" in text
    assert "version('setuptools')" in text
    assert "import pip\nimport setuptools" not in text


def test_setup_uses_official_recommended_torch_and_mmlab_versions() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "torch==2.0.1" in text
    assert "torchvision==0.15.2" in text
    assert "torchaudio==2.0.2" in text
    assert "https://download.pytorch.org/whl/cu118" in text
    assert '"mmcv==2.0.1"' in text
    assert '"mmdet==3.1.0"' in text
    assert '"mmpose==1.1.0"' in text


def test_setup_requires_cuda_and_never_installs_driver() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "nvidia-smi" in text
    assert "至少 4GB" in text
    lowered = text.lower()
    assert "nvidia-driver" not in lowered
    assert "apt install nvidia" not in lowered


def test_model_downloader_uses_expected_official_sources_only() -> None:
    text = DOWNLOADER.read_text(encoding="utf-8")
    for source in (
        "TMElyralab/MuseTalk",
        "stabilityai/sd-vae-ft-mse",
        "openai/whisper-tiny",
        "yzd-v/DWPose",
        "ManyOtherFunctions/face-parse-bisent",
    ):
        assert source in text
    assert "musetalkV15/unet.pth" in text
    assert "musetalk/pytorch_model.bin" not in text
    assert "latentsync_syncnet.pt" not in text


def test_setup_calls_project_model_downloader() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "scripts/download_musetalk_models.py" in text
    assert '--target "$MUSETALK/models"' in text


def test_setup_bash_syntax() -> None:
    bash = shutil.which("bash")
    if not bash:
        return
    result = subprocess.run([bash, "-n", str(SETUP)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
