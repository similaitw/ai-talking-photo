"""Lazy, allocation-free PyTorch device inspection."""

from dataclasses import dataclass
from importlib import import_module

from talking_photo.config import LOW_VRAM_THRESHOLD_GB


@dataclass(frozen=True)
class DeviceInfo:
    device: str = "cpu"
    gpu_name: str | None = None
    vram_gb: float = 0.0
    low_vram: bool = False
    torch_version: str | None = None
    cuda_version: str | None = None
    error: str | None = None


def get_device_info() -> DeviceInfo:
    """Inspect the current CUDA device without loading models or GPU tensors."""
    try:
        torch = import_module("torch")
    except ImportError:
        return DeviceInfo(error="PyTorch 未安裝或無法載入，請安裝適合執行環境的 PyTorch。")
    except Exception:
        return DeviceInfo(error="PyTorch 載入失敗，請檢查安裝環境與系統相依套件。")
    version = str(torch.__version__)
    cuda_version = torch.version.cuda
    try:
        if not torch.cuda.is_available():
            return DeviceInfo(torch_version=version, cuda_version=cuda_version)
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        vram = properties.total_memory / (1024 ** 3)
        return DeviceInfo(
            device=f"cuda:{index}", gpu_name=properties.name, vram_gb=vram,
            low_vram=vram <= LOW_VRAM_THRESHOLD_GB,
            torch_version=version, cuda_version=cuda_version,
        )
    except Exception:
        return DeviceInfo(
            torch_version=version, cuda_version=cuda_version,
            error="CUDA 偵測失敗，請檢查 NVIDIA 驅動程式與 PyTorch 相容性。",
        )
