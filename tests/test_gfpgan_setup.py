from pathlib import Path
import shutil
import subprocess

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_gfpgan.ps1"
LOCK = Path(__file__).resolve().parents[1] / "requirements-gfpgan-lock.txt"


def test_gfpgan_setup_is_path_independent_and_isolated() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "$PSScriptRoot" in text
    assert 'Join-Path $PSScriptRoot ".."' in text
    assert '.venv-gfpgan' in text
    assert '.venv\\Scripts' not in text
    assert "H:\\AI_Project" not in text


def test_gfpgan_setup_uses_only_official_checkout_and_pinned_revision() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "https://github.com/TencentARC/GFPGAN.git" in text
    assert "7552a7791caad982045a7bbe5634bbf1cd5c8679" in text
    assert "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth" in text
    assert "GFPGANv1.3.pth" in text


def test_gfpgan_setup_pins_compatible_torchvision_pair() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "torch==2.1.2" in text
    assert "torchvision==0.16.2" in text
    assert "https://download.pytorch.org/whl/cu118" in text
    assert "requirements-gfpgan-lock.txt" in text
    lock = LOCK.read_text(encoding="utf-8")
    assert "basicsr==1.4.2" in lock
    assert "facexlib==0.3.0" in lock


def test_gfpgan_setup_is_safe_to_rerun() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "if (Test-Path $VenvPython)" in text
    assert "if (-not (Test-Path $GFPGANDir))" in text
    assert "if (-not (Test-Path $ModelPath))" in text
    assert '@("-C", $GFPGANDir, "checkout", "--force", $GFPGANRevision)' in text


def test_gfpgan_setup_tolerates_benign_native_stderr_on_windows_powershell_51() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "[System.IO.Path]::GetTempFileName()" in text
    assert "2> $stderrPath" in text
    assert "$exitCode = $LASTEXITCODE" in text
    assert "Write-Host $stderrText -ForegroundColor DarkYellow" in text
    assert '$previousErrorActionPreference = $ErrorActionPreference' in text
    assert '$ErrorActionPreference = "Continue"' in text
    assert '$ErrorActionPreference = $previousErrorActionPreference' in text


def test_gfpgan_setup_explains_low_vram_profile_in_traditional_chinese() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for phrase in (
        "GFPGAN 高清修復設定完成",
        "畫質後處理",
        "GTX 1050 2GB",
        "不啟用 Real-ESRGAN 背景放大",
    ):
        assert phrase in text


def test_gfpgan_setup_powershell_syntax_when_pwsh_available() -> None:
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
