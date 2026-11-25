import pytest
import torch

from lmas.bus import BusTamperError, SerializedBytesBus, SharedMemoryTensorBus


def _make_kv(nlayers=4, batch=1, n_heads=2, seq_len=16, head_dim=16, dtype=torch.bfloat16):
    K = [torch.randn((batch, n_heads, seq_len, head_dim), dtype=dtype) for _ in range(nlayers)]
    V = [torch.randn((batch, n_heads, seq_len, head_dim), dtype=dtype) for _ in range(nlayers)]
    return K, V


def test_shm_bus_roundtrip_bfloat16_and_float32():
    bus = SharedMemoryTensorBus()
    try:
        for dtype in (torch.bfloat16, torch.float32):
            K, V = _make_kv(dtype=dtype)
            hdr = bus.publish(K, V, seq_id=7)
            assert hdr.nlayers == 4
            assert hdr.shape == (1, 2, 16, 16)
            assert hdr.dtype == dtype
            rxK, rxV, rx_hdr = bus.receive(hdr.shm_name, verify_hmac=True, zero_copy=True)
            assert rx_hdr.seq_id == 7
            for k_exp, k_got, v_exp, v_got in zip(K, rxK, V, rxV):
                assert torch.equal(k_exp, k_got)
                assert torch.equal(v_exp, v_got)
            del rxK, rxV, k_got, v_got
            bus.release(hdr.shm_name)
    finally:
        bus.close_all()


def test_shm_bus_hmac_rejects_tampering():
    bus = SharedMemoryTensorBus()
    try:
        K, V = _make_kv()
        hdr = bus.publish(K, V, seq_id=1)
        bus.corrupt_payload_for_test(hdr.shm_name, byte_offset=4)
        with pytest.raises(BusTamperError):
            bus.receive(hdr.shm_name, verify_hmac=True)
        bus.release(hdr.shm_name)
    finally:
        bus.close_all()


def test_serialized_bytes_bus_roundtrip():
    K, V = _make_kv(dtype=torch.bfloat16)
    rxK, rxV, nbytes = SerializedBytesBus.roundtrip(K, V)
    assert nbytes > 0
    for k_exp, k_got in zip(K, rxK):
        assert torch.equal(k_exp, k_got)
    for v_exp, v_got in zip(V, rxV):
        assert torch.equal(v_exp, v_got)
