"""HBM allocation + categorisation + OOM behaviour."""
from cloud_tpu_lab.src.memory.activation_memory import (
    estimate_mlp_activations, estimate_transformer_activations, total_bytes,
)
from cloud_tpu_lab.src.memory.hbm_sim import HbmSimulator, make_hbm_for_spec
from cloud_tpu_lab.src.tpu_versions.cloud_tpu_catalog import get_spec


def test_allocate_under_capacity() -> None:
    hbm = HbmSimulator(capacity_bytes=1_000_000, bandwidth_bytes_s=1e9)
    assert hbm.allocate("a", "parameters", 500_000)
    assert hbm.used_bytes() == 500_000
    assert hbm.utilization() == 0.5
    assert hbm.oom_events == 0


def test_allocate_over_capacity_triggers_oom() -> None:
    hbm = HbmSimulator(capacity_bytes=100, bandwidth_bytes_s=1e9)
    ok1 = hbm.allocate("a", "parameters", 80)
    ok2 = hbm.allocate("b", "parameters", 50)
    assert ok1 and not ok2
    assert hbm.oom_events == 1


def test_categories_sum_to_used_bytes() -> None:
    hbm = HbmSimulator(capacity_bytes=10_000_000, bandwidth_bytes_s=1e9)
    hbm.allocate("p", "parameters", 1000)
    hbm.allocate("a", "activations", 2000)
    hbm.allocate("o", "optimizer", 3000)
    cats = hbm.by_category()
    assert cats == {"parameters": 1000, "activations": 2000, "optimizer": 3000}
    assert sum(cats.values()) == hbm.used_bytes()


def test_activation_memory_grows_with_seq_squared() -> None:
    a = estimate_transformer_activations(batch_size=1, seq_len=16,
                                         hidden_size=64, num_layers=1)
    b = estimate_transformer_activations(batch_size=1, seq_len=32,
                                         hidden_size=64, num_layers=1)
    # seq_len doubles → seq^2 term should ≈ 4× → total grows substantially.
    assert total_bytes(b) > 1.5 * total_bytes(a)


def test_activation_memory_grows_linearly_with_batch() -> None:
    a = estimate_mlp_activations(batch_size=8, hidden_size=128, num_layers=4)
    b = estimate_mlp_activations(batch_size=16, hidden_size=128, num_layers=4)
    assert total_bytes(b) == 2 * total_bytes(a)


def test_make_hbm_for_spec_matches_catalog() -> None:
    spec = get_spec("v5e")
    hbm = make_hbm_for_spec(spec)
    assert hbm.capacity_bytes == int(spec.hbm_per_chip_gb * 1e9)
    assert hbm.bandwidth_bytes_s > 0
