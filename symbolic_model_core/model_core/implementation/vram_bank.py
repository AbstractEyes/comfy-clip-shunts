import torch
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable, List, Tuple

from model_core.framework import AbstractVramBankFirmware
from model_core.framework.automodel import BaseEncoder

logger = logging.getLogger(__name__)


class VramBank(AbstractVramBankFirmware):
    def __init__(self, max_vram_mb: float = 8192.0):
        self.max_vram_bytes = max_vram_mb * 1e6
        self.registry: Dict[str, BaseEncoder] = {}
        self._lock = asyncio.Lock()
        self._request_queue: List[Tuple[int, Dict[str, Any]]] = []
        self._queue_task = asyncio.create_task(self._process_requests())

    def register_encoder(self, name: str, encoder: BaseEncoder):
        self.registry[name] = encoder
        logger.info(f"VRAMBank: Encoder '{name}' registered.")

    def can_lend(self, encoder: BaseEncoder) -> bool:
        device = encoder.device
        current_usage = self.get_total_usage(device)
        additional = encoder.memory.get_total_size()
        return (current_usage + additional) <= self.max_vram_bytes

    def get_total_usage(self, device: Union[str, torch.device]) -> float:
        device = torch.device(device)
        total = 0
        for encoder in self.registry.values():
            if encoder.device == device:
                total += encoder.memory.get_total_size()
        return total

    def report(self) -> Dict[str, Any]:
        report = {}
        for name, encoder in self.registry.items():
            dev = str(encoder.device)
            usage = encoder.memory.get_total_size()
            report[name] = {
                "device": dev,
                "usage_mb": usage / 1e6,
                "layers": len(encoder.memory.layers),
            }
        return report

    def print_report(self):
        logger.debug("\\n=== VRAM BANK REPORT ===")
        for name, info in self.report().items():
            logger.debug(f"[{name}]")
            logger.debug(f"  Device: {info['device']}")
            logger.debug(f"  Usage:  {info['usage_mb']:.2f} MB")
            logger.debug(f"  Layers: {info['layers']}")
        logger.debug("==========================\\n")

    async def request_layer(self, encoder_name: str, layer_name: str, layer_size: int,
                            device: Union[str, torch.device], callback: Callable):
        """Queue a prioritized layer load request based on size."""
        device = torch.device(device)
        priority = layer_size  # Smaller = higher priority
        request = {
            "encoder_name": encoder_name,
            "layer_name": layer_name,
            "layer_size": layer_size,
            "device": device,
            "callback": callback,
        }
        self._request_queue.append((priority, request))
        self._request_queue.sort(key=lambda x: x[0])  # Sort ascending by size

    async def _process_requests(self):
        while True:
            if not self._request_queue:
                await asyncio.sleep(0.01)
                continue

            priority, req = self._request_queue.pop(0)
            async with self._lock:
                usage = self.get_total_usage(req["device"])
                available = self.max_vram_bytes - usage

                if req["layer_size"] <= self.max_vram_bytes:
                    if req["layer_size"] <= available:
                        logger.info(f"VRAMBank: Lending layer '{req['layer_name']}' to '{req['encoder_name']}'")
                        await asyncio.sleep(0)  # allow context switch
                        req["callback"]()
                    else:
                        logger.info(f"VRAMBank: Insufficient memory for '{req['layer_name']}', requeuing")
                        self._request_queue.append((priority, req))  # Requeue
                        self._request_queue.sort(key=lambda x: x[0])
                else:
                    logger.warning(f"VRAMBank: Layer '{req['layer_name']}' exceeds total budget, denied")

            await asyncio.sleep(0.001)