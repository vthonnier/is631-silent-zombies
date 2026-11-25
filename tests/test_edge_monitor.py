import pytest
import torch

from lmas.bus import EdgeTensorReferenceMonitor


def _make_clean_pair(base_k, base_v, noise=0.01):
    K = [t + noise * torch.randn_like(t) for t in base_k]
    V = [t + noise * torch.randn_like(t) for t in base_v]
    return K, V


def test_edge_monitor_detects_adversarial_shift_and_gates():
    torch.manual_seed(42)
    nlayers = 6
    layers = [2, 3, 4, 5]
    base_k = [torch.randn((1, 2, 16, 32), dtype=torch.float32) for _ in range(nlayers)]
    base_v = [torch.randn((1, 2, 16, 32), dtype=torch.float32) for _ in range(nlayers)]

    clean_calib = [_make_clean_pair(base_k, base_v, noise=0.01) for _ in range(5)]
    mon = EdgeTensorReferenceMonitor(layers=layers, cos_threshold=0.01, norm_ratio_threshold=1.10)
    mon.calibrate_thresholds_from_clean(clean_calib, cos_margin=1.5, norm_margin=1.08)

    # Clean test sample should pass
    test_clean_k, test_clean_v = _make_clean_pair(base_k, base_v, noise=0.01)
    clean_res = mon.inspect(test_clean_k, test_clean_v)
    assert not clean_res.flagged

    # Perturbed sample should be flagged and gated back to fallback
    adv_k = [t.clone() for t in test_clean_k]
    adv_v = [t.clone() for t in test_clean_v]
    for l in layers:
        adv_k[l] = adv_k[l] + 3.0 * torch.ones_like(adv_k[l])
        adv_v[l] = adv_v[l] - 2.5 * torch.ones_like(adv_v[l])

    adv_res = mon.inspect(adv_k, adv_v)
    assert adv_res.flagged
    assert adv_res.anomaly_score >= 1.0
    assert len(adv_res.reasons) >= 1

    safe_k, safe_v, res = mon.enforce(adv_k, adv_v, fallback_kv=(test_clean_k, test_clean_v), mode="gate")
    assert res.flagged
    for l in layers:
        assert torch.equal(safe_k[l], test_clean_k[l])
        assert torch.equal(safe_v[l], test_clean_v[l])

    # Gate without fallback raises PermissionError
    with pytest.raises(PermissionError):
        mon.enforce(adv_k, adv_v, fallback_kv=None, mode="gate")


def test_edge_monitor_project_sanitizer_clips_norm():
    torch.manual_seed(0)
    nlayers = 4
    layers = [1, 2]
    base_k = [torch.randn((1, 2, 8, 16), dtype=torch.float32) for _ in range(nlayers)]
    base_v = [torch.randn((1, 2, 8, 16), dtype=torch.float32) for _ in range(nlayers)]
    mon = EdgeTensorReferenceMonitor(layers=layers, norm_ratio_threshold=1.10)
    mon.fit([(base_k, base_v)])

    adv_k = [t * 4.0 if i in layers else t for i, t in enumerate(base_k)]
    adv_v = [t * 4.0 if i in layers else t for i, t in enumerate(base_v)]
    proj_k, proj_v, res = mon.enforce(adv_k, adv_v, mode="project")
    assert res.flagged
    for l in layers:
        assert proj_k[l].norm().item() <= 1.101 * base_k[l].norm().item()
        assert proj_v[l].norm().item() <= 1.101 * base_v[l].norm().item()
