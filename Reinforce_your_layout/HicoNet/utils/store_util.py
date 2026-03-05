# store_util.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import threading
import time
import torch

def _to_cpu(x: Any, dtype: Optional[torch.dtype] = None):
    if isinstance(x, torch.Tensor):
        y = x.detach().to("cpu")
        return y.to(dtype) if dtype is not None else y
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu(xx, dtype) for xx in x)
    if isinstance(x, dict):
        return {k: _to_cpu(v, dtype) for k, v in x.items()}
    return x

@dataclass
class SampleEntry:
    image: Any
    all_latents: Any
    all_encoders: Any
    all_cross_attention_kwargs: Any
    all_down_block_res_samples: Any
    all_mid_block_additional_residual: Any
    rewards: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)

class CPUStore:
    def __init__(
        self,
        *,
        keep_images: bool = True,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        self.keep_images = keep_images
        self.cast_dtype = cast_dtype

        self._lock = threading.Lock()
        self._entries: Dict[int, SampleEntry] = {}
        self._next_idx: int = 0

    def __enter__(self) -> "CPUStore":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.clear()
        return False

    def add(
        self,
        *,
        image: Any,
        all_latents: Any,
        all_encoders: Any,
        all_cross_attention_kwargs: Any,
        all_down_block_res_samples: Any,
        all_mid_block_additional_residual: Any,
        rewards: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        entry = SampleEntry(
            image=_to_cpu(image) if self.keep_images else None,
            all_latents=_to_cpu(all_latents, self.cast_dtype),
            all_encoders=_to_cpu(all_encoders, self.cast_dtype),
            all_cross_attention_kwargs=_to_cpu(all_cross_attention_kwargs),
            all_down_block_res_samples=_to_cpu(all_down_block_res_samples, self.cast_dtype),
            all_mid_block_additional_residual=_to_cpu(all_mid_block_additional_residual, self.cast_dtype),
            rewards=_to_cpu(rewards) if rewards is not None else None,
            metadata=metadata,
        )
        with self._lock:
            idx = self._next_idx
            self._entries[idx] = entry
            self._next_idx += 1
            return idx

    def get(self, data_idx) -> SampleEntry:
        try:
            import torch
            if torch.is_tensor(data_idx):
                if data_idx.numel() != 1:
                    raise TypeError(f"`data_idx` tensor must be scalar, got shape {tuple(data_idx.shape)}")
                data_idx = data_idx.item()
        except Exception:
            pass

        try:
            import numpy as np
            if isinstance(data_idx, np.ndarray):
                if data_idx.size != 1:
                    raise TypeError(f"`data_idx` numpy array must be scalar (size==1), got shape {data_idx.shape}")
                data_idx = data_idx.item()
            elif isinstance(data_idx, np.integer):
                data_idx = int(data_idx)
        except Exception:
            pass

        try:
            data_idx = int(data_idx)
        except Exception as e:
            raise TypeError(f"Unsupported index type for `data_idx`: {type(data_idx)}") from e

        with self._lock:
            try:
                return self._entries[data_idx]
            except KeyError:
                existing_keys = list(self._entries.keys())
                preview = existing_keys[:10]
                raise KeyError(
                    f"data_idx {data_idx} not found; "
                    f"existing keys (first 10): {preview}, total={len(existing_keys)}"
                )

    def pop(self, data_idx: int) -> SampleEntry:
        with self._lock:
            if data_idx not in self._entries:
                raise KeyError(f"data_idx {data_idx} not found")
            return self._entries.pop(data_idx)

    def release(self, data_idx: int) -> None:
        with self._lock:
            self._entries.pop(data_idx, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
