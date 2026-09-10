from pathlib import Path


def test_gfpgan_setup_has_utf8_bom_for_windows_powershell_51() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_gfpgan.ps1"
    assert script.read_bytes().startswith(b"\xef\xbb\xbf")
