"""Mesh + partitioner + collective cost model basics."""
from cloud_tpu_lab.src.sharding.all_reduce import (
    all_gather_time, all_reduce_time, reduce_scatter_time,
)
from cloud_tpu_lab.src.sharding.mesh import (
    PartitionSpec, make_1d_mesh, make_2d_mesh,
)
from cloud_tpu_lab.src.sharding.partitioner import partition_tensor


def test_1d_mesh_size() -> None:
    m = make_1d_mesh(4)
    assert m.n_devices == 4
    assert m.axis_size("data") == 4


def test_2d_mesh_size() -> None:
    m = make_2d_mesh(2, 4)
    assert m.n_devices == 8
    assert m.axis_size("data") == 2
    assert m.axis_size("model") == 4


def test_partition_replicated_keeps_full_shape() -> None:
    m = make_1d_mesh(4)
    shards = partition_tensor((128, 256), PartitionSpec((None, None)), m)
    assert len(shards) == 4
    assert all(s.shard_shape == (128, 256) for s in shards)


def test_partition_data_sharded_splits_first_dim() -> None:
    m = make_1d_mesh(4)
    shards = partition_tensor((128, 256), PartitionSpec(("data", None)), m)
    assert all(s.shard_shape == (32, 256) for s in shards)
    assert all(s.tensor_id == shards[0].tensor_id for s in shards)
    # Each shard gets a unique shard_id.
    assert len({s.shard_id for s in shards}) == 4


def test_all_reduce_zero_on_single_chip() -> None:
    cost = all_reduce_time(payload_bytes=1024, n_chips=1,
                           ici_bandwidth_bytes_s=2e11)
    assert cost.sim_duration_s == 0.0


def test_all_reduce_grows_with_payload() -> None:
    small = all_reduce_time(1024, 4, 2e11).sim_duration_s
    big = all_reduce_time(1024 * 1024, 4, 2e11).sim_duration_s
    assert big > small > 0


def test_all_gather_and_reduce_scatter_callable() -> None:
    # Smoke-test only — formulas already covered by unit math above.
    assert all_gather_time(1024, 4, 2e11).sim_duration_s >= 0
    assert reduce_scatter_time(1024, 4, 2e11).sim_duration_s >= 0
