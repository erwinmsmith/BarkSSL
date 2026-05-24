"""
BarkSSL Device Module
GPU/CUDA adaptive device selection.
"""

import os
import torch
from typing import Optional


def get_device(cuda_if_available: bool = True) -> torch.device:
    """
    Get the appropriate device for training/inference.

    Args:
        cuda_if_available: If True, prefer CUDA when available.
                         If False, always return CPU.

    Returns:
        torch.device: 'cuda' or 'cpu'
    """
    if not cuda_if_available:
        return torch.device('cpu')

    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'current_device': None,
        'device_name': 'cpu',
    }

    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)

        # Get memory info
        if torch.cuda.device_count() > 0:
            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            info['memory_allocated_gb'] = round(mem_allocated, 2)
            info['memory_reserved_gb'] = round(mem_reserved, 2)

    return info


def set_device(device: Optional[str] = None) -> torch.device:
    """
    Set the default device.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        The selected device
    """
    if device is None:
        return get_device()

    selected = torch.device(device)
    if selected.type == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Using CPU instead.")
        return torch.device('cpu')

    return selected


def to_device(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Move tensor to device.

    Args:
        data: Input tensor
        device: Target device

    Returns:
        Tensor on target device
    """
    return data.to(device)


class DeviceManager:
    """
    Singleton device manager for consistent device handling.
    """
    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def init(self, cuda_if_available: bool = True) -> 'DeviceManager':
        """Initialize device manager."""
        self._device = get_device(cuda_if_available)
        return self

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            self._device = get_device()
        return self._device

    def to(self, data: torch.Tensor) -> torch.Tensor:
        """Move data to device."""
        return data.to(self.device)

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.device})"


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def get_optimal_device() -> torch.device:
    """
    Get the optimal device considering CUDA and MPS.

    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif is_mps_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')