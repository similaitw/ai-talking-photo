from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import doctor
from talking_photo import device


@pytest.mark.parametrize("memory,low", [(2, True), (2.5, True), (12, False)])
def test_cuda_device(monkeypatch, memory, low):
    torch = SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test-cuda"),
        cuda=SimpleNamespace(
            is_available=lambda: True, current_device=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name=f"GPU {index}", total_memory=memory * 1024 ** 3,
            ),
        ),
    )
    monkeypatch.setattr(device, "import_module", lambda name: torch)
    info = device.get_device_info()
    assert info.device == "cuda:1"
    assert info.gpu_name == "GPU 1"
    assert info.vram_gb == memory
    assert info.low_vram is low
    assert ("Colab" in doctor.recommended_mode(info)) is low


def test_cpu_device(monkeypatch):
    torch = SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(device, "import_module", lambda name: torch)
    info = device.get_device_info()
    assert info.device == "cpu"
    assert info.error is None
    assert info.gpu_name is None
    assert "CPU" in doctor.recommended_mode(info)


@pytest.mark.parametrize("error", [ImportError(), OSError(), RuntimeError()])
def test_torch_load_failure(monkeypatch, error):
    def fail(name):
        raise error
    monkeypatch.setattr(device, "import_module", fail)
    info = device.get_device_info()
    assert info.error
    assert info.torch_version is None
    assert "Colab" in doctor.recommended_mode(info)


def test_cuda_failure(monkeypatch):
    def fail():
        raise RuntimeError("driver failure")
    torch = SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(is_available=fail),
    )
    monkeypatch.setattr(device, "import_module", lambda name: torch)
    assert "CUDA 偵測失敗" in device.get_device_info().error


@pytest.fixture
def ready_environment(tmp_path, monkeypatch):
    model = tmp_path / "模型.pth"
    model.write_bytes(b"test fixture, not model weights")
    monkeypatch.setattr(doctor, "get_device_info", lambda: device.DeviceInfo(torch_version="test"))
    monkeypatch.setattr(doctor, "import_module", lambda name: SimpleNamespace(Communicate=lambda: None))
    monkeypatch.setattr(doctor.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="ffmpeg version test\n"))
    return model


def test_ready_report_and_cli(ready_environment, capsys):
    assert doctor.main(["--model-path", str(ready_environment)]) == 0
    report = capsys.readouterr().out
    for label in ["Python", "PyTorch", "CUDA", "GPU", "顯示記憶體", "FFmpeg", "edge-tts", "模型", "建議模式"]:
        assert label in report
    assert "未驗證權重" in report
    assert "未測試網路" in report


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(1, "ffmpeg"), subprocess.TimeoutExpired("ffmpeg", 10)])
def test_ffmpeg_failure_still_reports_other_checks(ready_environment, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(doctor.subprocess, "run", fail)
    lines, ready = doctor.diagnose(ready_environment)
    assert not ready
    assert any("FFmpeg：無法執行" in line for line in lines)
    assert any("建議模式" in line for line in lines)


@pytest.mark.parametrize("state", ["missing", "empty", "directory"])
def test_invalid_model(ready_environment, state):
    ready_environment.unlink()
    if state == "empty":
        ready_environment.touch()
    elif state == "directory":
        ready_environment.mkdir()
    assert not doctor.diagnose(ready_environment)[1]


def test_missing_dependencies(ready_environment, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "get_device_info", lambda: device.DeviceInfo(error="PyTorch 未安裝"))
    def fail(name):
        raise ImportError()
    monkeypatch.setattr(doctor, "import_module", fail)
    assert doctor.main(["--model-path", str(ready_environment)]) == 1
    report = capsys.readouterr().out
    assert "edge-tts：未安裝" in report
    assert "Colab" in report


def test_script_runs_outside_repository(tmp_path):
    script = Path(doctor.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--model-path", str(tmp_path / "missing.pth")],
        cwd=tmp_path, capture_output=True, timeout=60,
    )
    assert result.returncode == 1
    assert b"Traceback" not in result.stderr
    assert b"FFmpeg" in result.stdout
    assert "環境診斷" in result.stdout.decode("utf-8")
