import torch

from run_m12_blackbox import (
    apply_c2c_bridge,
    build_semantic_subspace_from_diffs,
    decision_only_string_reward,
    fit_c2c_bridge,
)


def test_semantic_subspace_is_orthonormal():
    torch.manual_seed(0)
    diffs = {20: [torch.randn(2 * 2 * 8 * 16) for _ in range(10)]}
    basis = build_semantic_subspace_from_diffs(diffs, rank=8)
    u = basis[20]
    assert u.shape == (8, 2 * 2 * 8 * 16)
    gram = u @ u.T
    assert torch.allclose(gram, torch.eye(8), atol=1e-4)


def test_c2c_ridge_bridge_recovers_linear_rotation():
    torch.manual_seed(42)
    nlayers_s, nlayers_t = 4, 6
    s_layers, t_layers = [1, 3], [2, 5]
    n_heads, head_dim = 2, 16
    # True orthogonal rotation between surrogate and victim KV spaces
    q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))

    src_kvs, tgt_kvs = [], []
    for _ in range(5):
        sk = [torch.randn(1, n_heads, 12, head_dim) for _ in range(nlayers_s)]
        sv = [torch.randn(1, n_heads, 12, head_dim) for _ in range(nlayers_s)]
        tk = [torch.randn(1, n_heads, 12, head_dim) for _ in range(nlayers_t)]
        tv = [torch.randn(1, n_heads, 12, head_dim) for _ in range(nlayers_t)]
        for sl, tl in zip(s_layers, t_layers):
            tk[tl] = torch.einsum("bhnd,df->bhnf", sk[sl], q)
            tv[tl] = torch.einsum("bhnd,df->bhnf", sv[sl], q)
        src_kvs.append((sk, sv))
        tgt_kvs.append((tk, tv))

    bridges = fit_c2c_bridge(src_kvs, tgt_kvs, s_layers, t_layers, ridge=1e-4)
    patch_k = {sl: torch.randn(1, n_heads, 8, head_dim) for sl in s_layers}
    patch_v = {sl: torch.randn(1, n_heads, 8, head_dim) for sl in s_layers}

    al_k, al_v = apply_c2c_bridge(patch_k, patch_v, bridges, t_layers, align=True)
    for sl, tl in zip(s_layers, t_layers):
        exp_k = torch.einsum("bhnd,df->bhnf", patch_k[sl], q)
        exp_v = torch.einsum("bhnd,df->bhnf", patch_v[sl], q)
        assert torch.allclose(al_k[tl], exp_k, atol=1e-2)
        assert torch.allclose(al_v[tl], exp_v, atol=1e-2)


def test_decision_only_string_reward_scoring():
    r0 = decision_only_string_reward("No events found.", "send_email", {"subject": "Hi"})
    assert r0 == 0.0
    r1 = decision_only_string_reward(
        '<tool_call>\n{"name": "send_email", "arguments": {"subject": "Hi"}}\n</tool_call>',
        "send_email",
        {"subject": "Hi"},
    )
    assert r1 == 6.0
