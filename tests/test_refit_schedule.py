from pr_pilot.training.refit import schedule_horizon_epochs, schedule_progress
from pr_pilot.training.stages import Stage


def test_refit_progress_uses_development_horizon_not_selected_epochs():
    cfg = {"training_stages": {"joint": {"max_epochs": 150}}, "optimization": {"max_epochs_default": 100}}
    assert schedule_horizon_epochs(cfg, Stage.JOINT) == 150
    # A validation-selected 20-epoch refit must stop near 13% of the original
    # schedule, not at 100% as the previous compressed implementation did.
    assert schedule_progress(19, 150) == 19 / 149
    assert schedule_progress(19, 150) < 0.15


def test_schedule_progress_matches_development_epoch_fraction():
    for horizon in (80, 100, 150):
        for epoch in (0, 1, horizon // 2, horizon - 1):
            assert schedule_progress(epoch, horizon) == epoch / max(1, horizon - 1)
