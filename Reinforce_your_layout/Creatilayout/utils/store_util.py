# utils/store_util.py
from typing import Any, Dict, List, Tuple, Union, Optional
import torch

Tensor = torch.Tensor

def _to_cpu_obj(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_cpu_obj(v) for v in obj]
        return type(obj)(t)
    return obj

def _to_device_obj(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device_obj(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_device_obj(v, device) for v in obj]
        return type(obj)(t)
    return obj

def _cast_fp_dtype(obj: Any, dtype: torch.dtype) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {k: _cast_fp_dtype(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_cast_fp_dtype(v, dtype) for v in obj]
        return type(obj)(t)
    return obj

def _cat_first_dim(a: Any, b: Any) -> Any:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.cat([a, b], dim=0)
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _cat_first_dim(a[k], b[k]) for k in a.keys()}
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        t = [_cat_first_dim(x, y) for x, y in zip(a, b)]
        return type(a)(t)
    raise ValueError("Layout kwargs structure mismatch when concatenating.")

class LayoutKwargsStore:
    def __init__(
        self,
        ids_device: Union[str, torch.device] = "cuda",
        backend: str = "ram",
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ):
        if backend != "ram":
            raise ValueError("Only backend='ram' is supported for now.")
        self.backend = backend
        self.ids_device = torch.device(ids_device)

        if rank is None or world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                default_rank = torch.distributed.get_rank()
                default_ws = torch.distributed.get_world_size()
            else:
                default_rank, default_ws = 0, 1
            self.rank = default_rank if rank is None else rank
            self.world_size = default_ws if world_size is None else world_size
        else:
            self.rank = rank
            self.world_size = world_size

        self._store: Dict[int, Any] = {}   # local_id -> cpu object
        self._next_local_id: int = 1
        self._dtype_id = torch.long

    @staticmethod
    def _encode(rank: int, local_id: int) -> int:
        return (int(rank) << 32) | int(local_id)

    @staticmethod
    def _decode(encoded: int) -> Tuple[int, int]:
        rank = (int(encoded) >> 32) & 0xFFFFFFFF
        local_id = int(encoded) & 0xFFFFFFFF
        return rank, local_id

    def add(self, layout_kwargs: Dict[str, Any]) -> Tensor:
        cpu_obj = _to_cpu_obj(layout_kwargs)
        local_id = self._next_local_id
        self._next_local_id += 1
        self._store[local_id] = cpu_obj
        enc = self._encode(self.rank, local_id)
        return torch.tensor([enc], device=self.ids_device, dtype=self._dtype_id)

    def add_many(self, layouts: List[Dict[str, Any]]) -> Tensor:
        ids = [self.add(x) for x in layouts]
        return torch.cat(ids, dim=0)

    def clear(self) -> None:
        self._store.clear()
        self._next_local_id = 1

    def free(self, ids: Optional[Union[Tensor, List[int], List[Tensor]]] = None) -> None:
        if ids is None:
            self.clear()
            return
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().view(-1).tolist()
        elif isinstance(ids, list):
            flat = []
            for x in ids:
                if isinstance(x, torch.Tensor):
                    flat.extend(x.detach().cpu().view(-1).tolist())
                else:
                    flat.append(int(x))
            ids = flat
        else:
            ids = [int(ids)]

        for enc in ids:
            r, lid = self._decode(enc)
            if r == self.rank and lid in self._store:
                del self._store[lid]

    def __len__(self) -> int:
        return len(self._store)

    def is_local(self, enc_id: Union[int, Tensor]) -> bool:
        if isinstance(enc_id, torch.Tensor):
            enc_id = int(enc_id.item())
        r, _ = self._decode(enc_id)
        return r == self.rank

    def filter_local(self, ids: Tensor) -> Tensor:
        ids_cpu = ids.detach().cpu().view(-1)
        mask = [(self._decode(int(v))[0] == self.rank) for v in ids_cpu.tolist()]
        out = ids_cpu[torch.tensor(mask, dtype=torch.bool)]
        return out.to(device=self.ids_device, dtype=self._dtype_id)

    def get(
        self,
        enc_id: Union[int, Tensor],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        if isinstance(enc_id, torch.Tensor):
            enc_id = int(enc_id.view(-1)[0].item())
        r, lid = self._decode(enc_id)
        if r != self.rank:
            raise RuntimeError(
                f"Layout id belongs to rank={r}, but current rank={self.rank}. "
                "Call filter_local() first and only get local ids."
            )
        if lid not in self._store:
            raise KeyError(f"Local layout id {lid} not found. It may have been freed.")
        out = _to_device_obj(self._store[lid], torch.device(device))
        if dtype is not None:
            out = _cast_fp_dtype(out, dtype)
        return out

    def get_batch(
        self,
        enc_ids: Union[Tensor, List[int]],
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        if isinstance(enc_ids, torch.Tensor):
            ids_list = enc_ids.detach().cpu().view(-1).tolist()
        else:
            ids_list = [int(v) for v in enc_ids]

        local_ids: List[int] = []
        for v in ids_list:
            r, lid = self._decode(v)
            if r != self.rank:
                raise RuntimeError(
                    f"Found non-local id rank={r}. Use store.filter_local(ids) first to keep local ids only."
                )
            local_ids.append(lid)

        objs = []
        for lid in local_ids:
            if lid not in self._store:
                raise KeyError(f"Local layout id {lid} not found. It may have been freed.")
            obj = _to_device_obj(self._store[lid], torch.device(device))
            if dtype is not None:
                obj = _cast_fp_dtype(obj, dtype)  
            objs.append(obj)

        out = objs[0]
        for k in range(1, len(objs)):
            out = _cat_first_dim(out, objs[k])
        return out

    def decode(self, enc_ids: Union[Tensor, List[int], int]) -> Tuple[Tensor, Tensor]:
        if isinstance(enc_ids, int):
            enc_ids = torch.tensor([enc_ids], dtype=self._dtype_id, device="cpu")
        elif isinstance(enc_ids, torch.Tensor):
            enc_ids = enc_ids.detach().cpu().view(-1).to(dtype=self._dtype_id)
        else:
            enc_ids = torch.tensor([int(x) for x in enc_ids], dtype=self._dtype_id, device="cpu")

        ranks = (enc_ids >> 32) & 0xFFFFFFFF
        lids = enc_ids & 0xFFFFFFFF
        return ranks.to(self.ids_device), lids.to(self.ids_device)
