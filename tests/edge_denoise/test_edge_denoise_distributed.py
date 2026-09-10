from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from conftest import make_config, write_burst

from edge_denoise.config import Config
from edge_denoise.distributed import DistributedRuntime
from edge_denoise.train import Trainer, load_checkpoint


def mock_group(monkeypatch, *, rank: int = 0, world: int = 2):
    import edge_denoise.distributed as module

    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: world)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)
    init, broadcast, reduce, gather, destroy = (Mock() for _ in range(5))
    monkeypatch.setattr(module.dist, "init_process_group", init)
    monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    monkeypatch.setattr(module.dist, "gather_object", gather)
    monkeypatch.setattr(module.dist, "destroy_process_group", destroy)
    return init, broadcast, reduce, gather, destroy


def test_rank_zero_writes_and_broadcasts_failures(monkeypatch) -> None:
    init, broadcast, reduce, gather, destroy = mock_group(monkeypatch)
    runtime = DistributedRuntime("cpu")
    assert init.call_args.kwargs["backend"] == "gloo"
    action = Mock(return_value="written")
    assert runtime.on_primary(action) == "written"
    action.assert_called_once()
    with pytest.raises(RuntimeError, match="rank 0 operation failed: ValueError: refused"):
        runtime.on_primary(Mock(side_effect=ValueError("refused")))
    reduce.side_effect = lambda flag, **kwargs: flag.fill_(1)
    assert runtime.should_stop(False)
    factory = Mock()
    factory.state_dict.return_value = {"rng_state": "state"}
    gather.side_effect = lambda local, states, **kwargs: states.__setitem__(slice(None), [local, local])
    state = runtime.gather_state(factory, 16)
    assert state["world_size"] == 2 and state["effective_batch"] == 16
    assert state["ranks"][1]["factory"] == factory.state_dict()
    runtime.close()
    runtime.close()
    destroy.assert_called_once()


def test_other_ranks_do_not_write_and_receive_primary_errors(monkeypatch) -> None:
    _, broadcast, _, _, _ = mock_group(monkeypatch, rank=1)
    runtime = DistributedRuntime("cpu")
    action = Mock(side_effect=AssertionError("non-primary writer"))
    assert runtime.on_primary(action) is None
    action.assert_not_called()
    broadcast.side_effect = lambda message, **kwargs: message.__setitem__(0, "FileExistsError: occupied")
    with pytest.raises(RuntimeError, match="occupied"):
        runtime.on_primary(action)


def test_gpu_rank_binding_and_backend_are_selected_without_real_cuda(monkeypatch) -> None:
    init, *_ = mock_group(monkeypatch, rank=1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    set_device = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    runtime = DistributedRuntime("cuda")
    assert runtime.device == torch.device("cuda", 1)
    set_device.assert_called_once_with(torch.device("cuda", 1))
    assert init.call_args.kwargs["backend"] == "nccl"


def test_single_process_has_no_distributed_calls(monkeypatch) -> None:
    init, broadcast, reduce, gather, _ = mock_group(monkeypatch, world=1)
    runtime = DistributedRuntime("cpu")
    assert runtime.on_primary(lambda: 4) == 4
    assert not runtime.should_stop(False)
    assert runtime.gather_state(Mock(), 4) is None
    assert runtime.mean_values({"loss": 3}) == {"loss": 3}
    for operation in (init, broadcast, reduce, gather):
        operation.assert_not_called()


def test_accumulation_matches_a_larger_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = write_burst(tmp_path / "data")
    cfg = make_config(dataset, tmp_path / "large", representation="image", max_steps=1)
    large = cfg.model_dump()
    large["training"].update(batch_size=8)
    micro = cfg.model_dump()
    micro["training"].update(batch_size=2, accumulation_steps=4, run_dir=tmp_path / "micro")
    trainers = [Trainer(Config.model_validate(raw)) for raw in (large, micro)]
    # SGD exposes the averaged-gradient equivalence directly, without Adam's
    # sensitivity to tiny rounding differences in nearly zero gradients.
    for trainer in trainers:
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        trainer.run()
        assert trainer.effective_batch == 8
    for name, value in trainers[0].model.state_dict().items():
        torch.testing.assert_close(value, trainers[1].model.state_dict()[name], rtol=1e-5, atol=1e-7)


def test_rank_sampling_is_distinct_and_ddp_wraps_one_forward_for_consistency(tmp_path: Path, monkeypatch) -> None:
    class FakeDDP(torch.nn.Module):
        calls = 0

        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.module(*args, **kwargs)

        def no_sync(self):
            return nullcontext()

    dataset = write_burst(tmp_path / "data")
    cfg = make_config(dataset, tmp_path / "run", representation="image", lambda_consistency=1, max_steps=1)
    monkeypatch.setattr("edge_denoise.train.DistributedDataParallel", FakeDDP)
    mock_group(monkeypatch, rank=0)
    first = Trainer(cfg)
    mock_group(monkeypatch, rank=1)
    second = Trainer(cfg)
    _, a = first.factory.sample_batch(count=8, return_info=True)
    _, b = second.factory.sample_batch(count=8, return_info=True)
    assert a != b
    batch = second.factory.sample_batch()
    _, terms = second._pair_step(batch)
    second._combine(terms).backward()
    assert second.train_model.calls == 1
    assert all(parameter.grad is not None for parameter in second.model.parameters())


def test_changing_gpu_count_resumes_weights_with_fresh_streams(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    dataset = write_burst(tmp_path / "data")
    cfg = make_config(dataset, tmp_path / "run", representation="image", max_steps=1)
    trainer = Trainer(cfg)
    checkpoint = trainer.run()
    state = load_checkpoint(checkpoint)
    state["distributed"] = {"world_size": 4, "ranks": [], "effective_batch": 8}
    torch.save(state, checkpoint)
    with pytest.warns(UserWarning, match="GPU count changed"):
        resumed = Trainer(cfg, resume_from=checkpoint)
    assert resumed.step == trainer.step
    for name, parameter in trainer.model.state_dict().items():
        torch.testing.assert_close(parameter, resumed.model.state_dict()[name])
    assert resumed.factory.state_dict() != trainer.factory.state_dict()


def test_invalid_distributed_environment_and_cpu_bf16_are_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "2")
    with pytest.raises(ValueError, match="invalid torchrun"):
        DistributedRuntime("cpu")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    cfg = make_config(tmp_path / "data", tmp_path / "run").model_dump()
    cfg["training"]["precision"] = "bf16"
    with pytest.raises(ValueError, match="bf16 requires CUDA"):
        Trainer(Config.model_validate(cfg))
