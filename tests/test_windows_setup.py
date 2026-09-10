from pathlib import Path
import shutil
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_windows.ps1"


def test_windows_setup_script_exists_and_is_path_independent() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "$PSScriptRoot" in text
    assert 'Join-Path $PSScriptRoot ".."' in text
    assert "H:\\AI_Project" not in text


def test_windows_setup_checks_required_tools_without_installing_driver() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'Get-CommandPath "git' in text
    assert "Resolve-Python311" in text
    assert 'Get-CommandPath "ffmpeg' in text
    assert 'Get-CommandPath "nvidia-smi' in text
    lowered = text.lower()
    assert "nvidia driver" not in lowered
    assert "winget install nvidia" not in lowered
    assert "choco install nvidia" not in lowered
    assert "本腳本不會安裝或更新 NVIDIA 驅動程式" in text


def test_windows_setup_uses_verified_runtime_and_upstream_revision() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "torch==2.5.1" in text
    assert "https://download.pytorch.org/whl/cu118" in text
    assert "requirements-wav2lip-lock.txt" in text
    assert "https://github.com/Rudrabha/Wav2Lip.git" in text
    assert "bac9a81e63ecc153202353372e5724b83d9e6322" in text
    assert "scripts\\prepare_wav2lip.py" in text
    assert "scripts\\doctor.py" in text


def test_windows_setup_is_safe_to_rerun() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "if (Test-Path $VenvPython)" in text
    assert "if (-not (Test-Path $Wav2LipDir))" in text
    assert "if (-not (Test-Path $Checkpoint) -and -not (Test-Path $OfficialGanCopy))" in text
    assert "if (-not (Test-Path $Detector))" in text
    assert '@("-C", $Wav2LipDir, "checkout", "--force", $Wav2LipRevision)' in text


def test_windows_setup_shows_next_steps_in_traditional_chinese() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for phrase in (
        "Windows 環境設定完成",
        "下一步",
        "啟動後預設網址：http://127.0.0.1:7860",
        "GTX 1050 2GB 建議優先使用 Colab",
    ):
        assert phrase in text


def test_windows_setup_powershell_syntax_when_pwsh_available() -> None:
    pwsh = shutil.which("pwsh")
    if not pwsh:
        return
    command = (
        "$e=$null;$t=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT.as_posix()}',[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count -gt 0){$e|ForEach-Object{$_.Message};exit 1}"
    )
    result = subprocess.run(
        [pwsh, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
