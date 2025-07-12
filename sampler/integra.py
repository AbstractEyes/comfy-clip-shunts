import torch
from typing import Tuple, List, Dict
from dataclasses import dataclass
from .alucard import FieldWalker, FieldWalkerConfig
from .sliding_window import ShuntStackConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntegraConfig:
    walker_config: FieldWalkerConfig
    stack_config: ShuntStackConfig
    trace_folds: bool = False  # Optional: store each windowed fold for debug
    enforce_projection: bool = True  # Optional: auto-project symbolic fields if needed
    enable_clip_alignment: bool = True  # Optional: perform scheduler-aware comparison to CLIP


class IntegraOrchestrator:
    def __init__(self, config: IntegraConfig):
        self.config = config
        self.walker = FieldWalker(config.walker_config)

        stack = config.stack_config
        self.window_size = stack.sliding_window_size
        self.stride = stack.sliding_window_stride
        self.max_length = stack.max_length
        self.override_context = stack.override_context_window
        self.context_window_size = stack.context_window_size

    def walk_encoder_field(self,
                           a: torch.Tensor,
                           b: torch.Tensor,
                           d: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Walks a full symbolic encoder field via sliding windows, governed by Integra.
        Returns recombined tensor and orchestration report.
        """
        with torch.autocast(device_type=a.device.type, enabled=a.device.type != 'cpu'):
            B, T_full, D = a.shape
            limit = self.context_window_size if self.override_context else T_full
            limit = min(limit, self.max_length)
            T = min(limit, T_full)

            # Slice the initial context window (if override is active)
            a = a[:, :T, :]
            b = b[:, :T, :]
            d = d[:, :T, :]

            folds = []
            starts = range(0, max(1, T - self.window_size + 1), self.stride)

            for start in starts:
                end = start + self.window_size
                # Clip bounds to avoid overrun
                if end > T:
                    end = T
                    start = max(0, end - self.window_size)

                a_win = a[:, start:end, :]
                b_win = b[:, start:end, :]
                d_win = d[:, start:end, :]
                # mask the first and last token if the window to see but not utilize them
                if self.override_context:
                    a_win[:, 0, :] = -100.0  # Mask first token
                    a_win[:, -1, :] = -100.0
                    b_win[:, 0, :] = -100.0
                    b_win[:, -1, :] = -100.0
                    d_win[:, 0, :] = -100.0
                    d_win[:, -1, :] = -100.0

                logger.info(f"Window slice [{start}:{end}] a_win shape: {a_win.shape}")
                folded = self.walker.walk(a_win, b_win, d_win)
                logger.info(f"Folded shape: {folded.shape}")

                folds.append((start, end, folded))

            # Aggregate windowed output
            logger.info(f"Aggregating {len(folds)} folds with total tokens: {T_full}")
            aggregated = self.aggregate(folds, T)

            return aggregated, {
                "tokens_processed": T,
                "tokens_total": T_full,
                "folds": len(folds),
                "stride": self.stride,
                "window_size": self.window_size,
                "override_context": self.override_context
            }

    def aggregate(self, folds, total_tokens):
        B, _, D = folds[0][2].shape
        device = folds[0][2].device
        acc = torch.zeros(B, total_tokens, D, device=device)
        wsum = torch.zeros(B, total_tokens, 1, device=device)

        for start, end, chunk in folds:
            length = end - start
            tri = torch.linspace(0, 1, length, device=device).unsqueeze(0).unsqueeze(-1)
            tri = torch.minimum(tri, 1 - tri) * 2
            acc[:, start:end, :] += chunk * tri
            wsum[:, start:end, :] += tri

        return acc / wsum.clamp(min=1e-6)