"""
shm_bus.py -- POSIX /dev/shm zero-copy KV-cache tensor bus with HMAC integrity.

Implements the OS-level latent handoff transport described in the system design:
- On Linux, Python's `multiprocessing.shared_memory.SharedMemory` creates
  shm_open()-backed files in `/dev/shm`, enabling zero-copy inter-agent KV
  cache transfer across processes without serialization/deserialization copies.
- Each handoff payload is framed with a fixed binary header (magic, version,
  sender/receiver IDs, layer count, shape, dtype code, sequence nonce, payload
  length) and a 32-byte HMAC-SHA256 authentication tag computed over the header
  prefix + contiguous tensor buffer.
- Also provides `SerializedBytesBus` (`io.BytesIO` / `pickle` / `torch.save`)
  as the copy-based IPC baseline for latency and throughput benchmarking.

Security note (explicitly demonstrated in M8):
  HMAC-SHA256 detects rogue external writers and channel corruption (100%
  tamper rejection), but a compromised endpoint node that holds its own valid
  signing key can still sign an adversarially perturbed tensor. Therefore,
  cryptographic transport integrity MUST be paired with `EdgeTensorReferenceMonitor`
  to inspect the tensor geometry before downstream attention ingestion.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import pickle
import struct
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import torch


_MAGIC = b"LMASKV01"
# Header layout (big-endian):
#   8s  magic
#   H   version (1)
#   H   nlayers
#   I   batch
#   I   n_heads
#   I   seq_len
#   I   head_dim
#   B   dtype_code (0=float32, 1=float16, 2=bfloat16)
#   B   reserved
#   Q   seq_id (uint64)
#   Q   payload_bytes (uint64)
#   32s hmac_sha256
_HEADER_PREFIX_FMT = ">8sHHIIIIBBQQ"
_HEADER_PREFIX_SIZE = struct.calcsize(_HEADER_PREFIX_FMT)
_HMAC_SIZE = 32
_HEADER_SIZE = _HEADER_PREFIX_SIZE + _HMAC_SIZE

_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class BusTamperError(RuntimeError):
    """Raised when a shared-memory handoff fails magic or HMAC verification."""


@dataclass(frozen=True)
class HandoffHeader:
    nlayers: int
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    seq_id: int
    payload_bytes: int
    shm_name: str
    backing_path: str


def _tensor_list_to_u8_view(tensors: list[torch.Tensor]) -> tuple[np.ndarray, torch.dtype, tuple[int, ...]]:
    """Pack a list of same-shaped tensors into a contiguous uint8 numpy view."""
    if not tensors:
        raise ValueError("Cannot pack empty tensor list.")
    dtype = tensors[0].dtype
    shape = tuple(tensors[0].shape)
    stacked = torch.stack([t.detach().cpu().contiguous() for t in tensors], dim=0)
    # View as uint8 because numpy lacks native bfloat16 dtype.
    u8 = stacked.view(torch.uint8).numpy().reshape(-1)
    return u8, dtype, shape


def _u8_view_to_tensor_list(
    u8_buf: memoryview | np.ndarray,
    nlayers: int,
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
    zero_copy: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reconstruct per-layer K/V tensors from a uint8 shared memory slice."""
    arr = np.frombuffer(u8_buf, dtype=np.uint8)
    t_u8 = torch.from_numpy(arr if zero_copy else arr.copy())
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    full_shape = (2, nlayers, *shape)
    expected_elems = 2 * nlayers * int(np.prod(shape))
    t_typed = t_u8[: expected_elems * elem_bytes].view(dtype).reshape(full_shape)
    if str(device) != "cpu":
        t_typed = t_typed.to(device)
    K = [t_typed[0, l] for l in range(nlayers)]
    V = [t_typed[1, l] for l in range(nlayers)]
    return K, V


class SharedMemoryTensorBus:
    """Zero-copy POSIX /dev/shm KV-cache bus with HMAC-SHA256 integrity."""

    def __init__(self, secret_key: bytes | None = None, prefix: str = "lmas_kv") -> None:
        self.secret_key = secret_key or os.urandom(32)
        self.prefix = prefix
        self._active_shm: dict[str, shared_memory.SharedMemory] = {}

    @staticmethod
    def backing_location(shm_name: str) -> str:
        """Return OS backing path (/dev/shm/<name> on Linux, shm://<name> otherwise)."""
        dev_shm = f"/dev/shm/{shm_name}"
        if os.name == "posix" and os.path.exists("/dev/shm"):
            return dev_shm
        return f"shm://{shm_name}"

    def _compute_hmac(self, prefix_bytes: bytes, payload_view: memoryview) -> bytes:
        mac = hmac.new(self.secret_key, digestmod=hashlib.sha256)
        mac.update(prefix_bytes)
        mac.update(payload_view)
        return mac.digest()

    def publish(
        self,
        K: list[torch.Tensor],
        V: list[torch.Tensor],
        seq_id: int = 0,
    ) -> HandoffHeader:
        """Write K and V caches into a new shared-memory segment and sign with HMAC."""
        if len(K) != len(V) or not K:
            raise ValueError("K and V must be non-empty lists of equal length.")
        nlayers = len(K)
        dtype = K[0].dtype
        if dtype not in _DTYPE_TO_CODE:
            raise ValueError(f"Unsupported dtype for SHM bus: {dtype}")
        shape = tuple(K[0].shape)
        if len(shape) != 4:
            raise ValueError(f"Expected 4D KV tensor (B, H, S, D), got {shape}")

        k_stack = torch.stack([t.detach().cpu().contiguous() for t in K], dim=0)
        v_stack = torch.stack([t.detach().cpu().contiguous() for t in V], dim=0)
        kv_pair = torch.stack([k_stack, v_stack], dim=0)  # (2, L, B, H, S, D)
        u8 = kv_pair.view(torch.uint8).numpy().reshape(-1)
        payload_bytes = int(u8.nbytes)

        shm_name = f"{self.prefix}_{uuid.uuid4().hex[:12]}"
        total_bytes = _HEADER_SIZE + payload_bytes
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=total_bytes)
        self._active_shm[shm_name] = shm

        prefix_bytes = struct.pack(
            _HEADER_PREFIX_FMT,
            _MAGIC,
            1,
            nlayers,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            _DTYPE_TO_CODE[dtype],
            0,
            seq_id,
            payload_bytes,
        )
        shm.buf[:_HEADER_PREFIX_SIZE] = prefix_bytes
        payload_view = shm.buf[_HEADER_SIZE : _HEADER_SIZE + payload_bytes]
        payload_view[:] = u8.data
        tag = self._compute_hmac(prefix_bytes, payload_view)
        payload_view.release()
        shm.buf[_HEADER_PREFIX_SIZE:_HEADER_SIZE] = tag

        return HandoffHeader(
            nlayers=nlayers,
            shape=shape,
            dtype=dtype,
            seq_id=seq_id,
            payload_bytes=payload_bytes,
            shm_name=shm_name,
            backing_path=self.backing_location(shm_name),
        )

    def receive(
        self,
        shm_name: str,
        device: torch.device | str = "cpu",
        verify_hmac: bool = True,
        zero_copy: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], HandoffHeader]:
        """Attach to an existing shared-memory segment, verify HMAC, and return (K, V)."""
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        payload_view = None
        try:
            prefix_bytes = bytes(shm.buf[:_HEADER_PREFIX_SIZE])
            (
                magic,
                version,
                nlayers,
                b,
                h,
                s,
                d,
                dtype_code,
                _,
                seq_id,
                payload_bytes,
            ) = struct.unpack(_HEADER_PREFIX_FMT, prefix_bytes)
            if magic != _MAGIC or version != 1:
                raise BusTamperError(f"Invalid SHM header magic/version: {magic!r}, v{version}")
            if dtype_code not in _CODE_TO_DTYPE:
                raise BusTamperError(f"Unknown dtype code: {dtype_code}")
            dtype = _CODE_TO_DTYPE[dtype_code]
            tag_stored = bytes(shm.buf[_HEADER_PREFIX_SIZE:_HEADER_SIZE])
            payload_view = shm.buf[_HEADER_SIZE : _HEADER_SIZE + payload_bytes]
            if verify_hmac:
                tag_expected = self._compute_hmac(prefix_bytes, payload_view)
                if not hmac.compare_digest(tag_stored, tag_expected):
                    payload_view.release()
                    payload_view = None
                    raise BusTamperError("HMAC-SHA256 verification failed on /dev/shm handoff.")

            shape = (b, h, s, d)
            K, V = _u8_view_to_tensor_list(
                payload_view, nlayers, shape, dtype, device=device, zero_copy=zero_copy
            )
            header = HandoffHeader(
                nlayers=nlayers,
                shape=shape,
                dtype=dtype,
                seq_id=seq_id,
                payload_bytes=payload_bytes,
                shm_name=shm_name,
                backing_path=self.backing_location(shm_name),
            )
            if not zero_copy or str(device) != "cpu":
                payload_view.release()
                payload_view = None
                shm.close()
            else:
                # Keep attached handle + view alive while zero-copy CPU views reference it.
                self._active_shm[f"recv_{shm_name}_{uuid.uuid4().hex[:6]}"] = shm
            return K, V, header
        except Exception:
            if payload_view is not None:
                try:
                    payload_view.release()
                except Exception:
                    pass
            try:
                shm.close()
            except Exception:
                pass
            raise

    def corrupt_payload_for_test(self, shm_name: str, byte_offset: int = 0) -> None:
        """Simulate an unauthorized rogue writer modifying shared memory bytes."""
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        try:
            idx = _HEADER_SIZE + byte_offset
            shm.buf[idx] = (shm.buf[idx] ^ 0xFF) & 0xFF
        finally:
            shm.close()

    def release(self, shm_name: str) -> None:
        """Close and unlink a shared-memory segment."""
        # Close any receiver handles first
        recv_keys = [k for k in self._active_shm if k.startswith(f"recv_{shm_name}_")]
        for rk in recv_keys:
            try:
                self._active_shm.pop(rk).close()
            except Exception:
                pass
        shm = self._active_shm.pop(shm_name, None)
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        else:
            try:
                tmp = shared_memory.SharedMemory(name=shm_name, create=False)
                tmp.close()
                tmp.unlink()
            except Exception:
                pass

    def close_all(self) -> None:
        """Release all shared-memory segments created or attached by this bus."""
        names = [k for k in self._active_shm if not k.startswith("recv_")]
        for name in names:
            self.release(name)
        for k in list(self._active_shm.keys()):
            try:
                self._active_shm.pop(k).close()
            except Exception:
                pass


class SerializedBytesBus:
    """Copy-based serialized IPC baseline (torch.save / pickle over BytesIO)."""

    @staticmethod
    def roundtrip(
        K: list[torch.Tensor],
        V: list[torch.Tensor],
        device: torch.device | str = "cpu",
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        buf = io.BytesIO()
        payload = (
            [t.detach().cpu() for t in K],
            [t.detach().cpu() for t in V],
        )
        torch.save(payload, buf)
        raw = buf.getvalue()
        in_buf = io.BytesIO(raw)
        K_out, V_out = torch.load(in_buf, map_location=device, weights_only=True)
        return K_out, V_out, len(raw)
