from pathlib import Path

import torch

from pr_pilot.training.checkpointing import CheckpointManager


def test_checkpoint_manager_keeps_only_resumable_last_and_best(tmp_path: Path):
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path, "score")
    assert manager.save_epoch(model, optimizer, None, 1, {"score": 2.0}, {"seed": 1})
    assert manager.save_epoch(model, optimizer, None, 2, {"score": 3.0}, {"seed": 1}) is False
    assert manager.save_epoch(model, optimizer, None, 3, {"score": 1.0}, {"seed": 1})
    assert sorted(path.name for path in tmp_path.iterdir()) == ["best.pt", "last.pt", "metrics.jsonl"]
    assert torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)["epoch"] == 3
    assert manager.load_last()["epoch"] == 3


def test_checkpoint_manager_restores_optimizer_and_epoch(tmp_path: Path):
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path, "score")
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    manager.save_epoch(model, optimizer, None, 7, {"score": 1.0})

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    next_epoch = manager.restore_last(restored_model, restored_optimizer)
    assert next_epoch == 8
    assert optimizer.state_dict()["state"]
    assert restored_optimizer.state_dict()["state"]
