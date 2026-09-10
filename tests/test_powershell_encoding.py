from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_setup_scripts_have_utf8_bom_for_powershell_51() -> None:
    for relative in ("scripts/setup_windows.ps1", "scripts/setup_gfpgan.ps1"):
        script = ROOT / relative
        assert script.read_bytes().startswith(b"\xef\xbb\xbf"), relative
