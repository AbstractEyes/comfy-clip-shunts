# rose_tensor_bank.py
# ---------------------------------------------------------------
# ROSE Tensor Bank
# ---------------------------------------------------------------
# Author: AbstractPhil
# Date: 07/31/2025
# license: Apache-2.0
# ---------------------------------------------------------------
# Regulated, thread-safe tensor container for ROSE architecture
# Singleton-managed, with per-tensor metadata encapsulated
# ---------------------------------------------------------------

import os
import time

import torch
import tempfile
import psutil
from typing import Dict, Optional, Tuple, Any, List
from threading import RLock

from .rose_config import RoseTensorBankConfig


class _RoseTensorRecord:
    def __init__(self, name: str, tensor: torch.Tensor, origin: str, source: Any):
        self.name = name
        self.tensor = tensor
        self.origin = origin
        self.device = str(tensor.device)
        self.shape = tuple(tensor.shape)
        self.dtype = str(tensor.dtype)
        self.source_type = type(source).__name__
        self.in_use = False
        self.offloaded = False
        self.locked = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "origin": self.origin,
            "device": self.device,
            "shape": self.shape,
            "dtype": self.dtype,
            "type": self.source_type,
            "in_use": self.in_use,
            "offloaded": self.offloaded,
            "locked": self.locked
        }


class RoseTensorBank:
    _instance: Optional['RoseTensorBank'] = None
    _lock = RLock()

    @classmethod
    def get(cls, initial: Optional[Dict[str, torch.Tensor]] = None, config: Optional[RoseTensorBankConfig] = None) -> 'RoseTensorBank':
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(initial or {}, config=config)
            elif initial:
                for key, tensor in initial.items():
                    cls._instance.add_tensor(key, tensor)
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.cleanup()
                cls._instance = None

    def __init__(self, tensors: Dict[str, torch.Tensor], config: Optional[RoseTensorBankConfig] = None):
        self._lock = RLock()
        self.config = config or RoseTensorBankConfig()

        self.tensors: Dict[str, torch.Tensor] = {}
        self._records: Dict[str, _RoseTensorRecord] = {}
        self.allocation_state: Dict[str, str] = {}
        self.allocation_deltas: List[Dict[str, Any]] = []

        self.vram_allocation: int = 0
        self.ram_allocation: int = 0
        self.drive_allocation: int = 0
        self.cache_directory: str = self.config.cache_directory or tempfile.gettempdir()
        os.makedirs(self.cache_directory, exist_ok=True)

        self.offload_order = self.config.offload_order
        for key, tensor in tensors.items():
            self.add_tensor(key, tensor)

    def add_tensor(self, name: str, tensor: torch.Tensor):
        with self._lock:
            device = self._allocate_to_best_fit(tensor)
            tensor = tensor.to(device) if isinstance(device, torch.device) else tensor.cpu()
            self.tensors[name] = tensor
            self.allocation_state[name] = str(device)

            origin = self._extract_prefix(name)
            self._records[name] = _RoseTensorRecord(name, tensor, origin, source=tensor)

            self._track_allocation(name, tensor, str(device))
            if device == "disk":
                self._cache_to_disk(tensor, name)

            if self.config.track_allocation_deltas:
                self._log_allocation_delta(name, event="add")

            if self.config.verbose:
                print(f"[BANK:{self.config.tag}] Tensor '{name}' added to {device}")

    def get_tensor(self, name: str) -> torch.Tensor:
        with self._lock:
            state = self.allocation_state.get(name)
            if state == "disk":
                tensor = torch.load(self._disk_path(name), map_location="cpu")
                self.add_tensor(name, tensor)
                os.remove(self._disk_path(name))
                return tensor.to(self.primary_device())
            if self.config.track_allocation_deltas:
                self._log_allocation_delta(name, event="read")
            return self.tensors[name]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        return self._records[name].to_dict() if name in self._records else {}

    def _log_allocation_delta(self, name: str, event: str):
        self.allocation_deltas.append({
            "name": name,
            "event": event,
            "timestamp": time.time(),
            "vram_MB": self.vram_allocation / 1024**2,
            "ram_MB": self.ram_allocation / 1024**2,
            "disk_MB": self.drive_allocation / 1024**2,
        })

    def _allocate_to_best_fit(self, tensor: torch.Tensor) -> str:
        for target in self.offload_order:
            if target == "cuda" and torch.cuda.is_available() and not self.config.force_cpu:
                if self._has_room_on_cuda(tensor):
                    return torch.device("cuda")
            elif target == "cpu":
                if self._has_room_on_cpu(tensor):
                    return torch.device("cpu")
            elif target == "disk" and self.config.allow_disk_offload:
                return "disk"
        return torch.device("cpu")

    def _track_allocation(self, name: str, tensor: torch.Tensor, location: str):
        size_bytes = tensor.element_size() * tensor.nelement()
        if location == "cuda":
            self.vram_allocation += size_bytes
        elif location == "cpu":
            self.ram_allocation += size_bytes
        elif location == "disk":
            self.drive_allocation += size_bytes

    def _cache_to_disk(self, tensor: torch.Tensor, name: str):
        path = self._disk_path(name)
        torch.save(
            tensor.cpu(),
            path,
            _use_new_zipfile_serialization=not self.config.compress_disk_tensors
        )

    def _disk_path(self, name: str) -> str:
        return os.path.join(self.cache_directory, f"{name}.rose_tensor.pt")

    def _has_room_on_cuda(self, tensor: torch.Tensor) -> bool:
        required = tensor.element_size() * tensor.nelement()
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        available = total - reserved
        return required < available * self.config.max_vram_utilization

    def _has_room_on_cpu(self, tensor: torch.Tensor) -> bool:
        required = tensor.element_size() * tensor.nelement()
        available = psutil.virtual_memory().available
        return required < available * self.config.max_ram_utilization

    def _extract_prefix(self, name: str) -> str:
        if "__" in name:
            return name.split("__", 1)[0]
        return "unknown"

    def primary_device(self) -> torch.device:
        return torch.device("cpu") if self.config.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def estimate_size(self, key: Optional[str] = None) -> int:
        with self._lock:
            if not self.config.enable_size_estimation:
                return -1
            if key:
                t = self.tensors[key]
                return int(t.element_size() * t.nelement() * (1 + self.config.estimate_margin))
            else:
                total = sum(t.element_size() * t.nelement() for t in self.tensors.values())
                return int(total * (1 + self.config.estimate_margin))

    def info(self) -> Dict[str, any]:
        with self._lock:
            return {
                "keys": list(self.tensors.keys()),
                "vram_allocation_MB": self.vram_allocation / 1024**2,
                "ram_allocation_MB": self.ram_allocation / 1024**2,
                "disk_allocation_MB": self.drive_allocation / 1024**2,
                "primary_device": str(self.primary_device()),
                "offload_order": self.offload_order,
                "cache_directory": self.cache_directory
            }

    def cleanup(self):
        with self._lock:
            if not self.config.clear_cache_on_exit:
                return
            for name in self.allocation_state:
                if self.allocation_state[name] == "disk":
                    disk_path = self._disk_path(name)
                    if os.path.exists(disk_path):
                        os.remove(disk_path)
            torch.cuda.empty_cache()

    def __del__(self):
        self.cleanup()

