""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler

def create_scheduler(optimizer, num_epochs, lr_min, warmup_lr_init, warmup_t, noise_range = None):

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    # Stage1 MLP LR profile:
    # 1) Epoch 1-5: linear warmup 1e-6 -> 5e-5
    # 2) Epoch 6-60: keep 5e-5
    # 3) After epoch 60: every 20 epochs decay by factor 5
    def _stage1_mlp_lr(epoch):
        epoch = int(epoch)
        # Piecewise linear schedule:
        # epoch 1 -> 5: 1e-6 -> 5e-5
        # epoch 5 -> 10: 5e-5 -> 1e-4
        # epoch 10 -> 20: 1e-4 -> 5e-4
        # epoch 21 -> 60: hold at 5e-4
        # epoch 61: set to 5e-5, then every 20 epochs decay by factor 5
        if epoch <= 1:
            lr = 1e-6
        elif epoch <= 5:
            # linear from epoch1..5
            progress = float(epoch - 1) / max(1.0, 5 - 1)
            lr = 1e-6 + progress * (5e-5 - 1e-6)
        elif epoch <= 10:
            # linear from epoch5..10
            progress = float(epoch - 5) / max(1.0, 10 - 5)
            lr = 5e-5 + progress * (1e-4 - 5e-5)
        elif epoch <= 20:
            # linear from epoch10..20
            progress = float(epoch - 10) / max(1.0, 20 - 10)
            lr = 1e-4 + progress * (5e-4 - 1e-4)
        elif epoch <= 60:
            lr = 5e-4
        else:
            # at epoch 61 we drop to 5e-5, then every decay_every(20) epochs divide by 5
            if epoch <= 60:
                decay_steps = 0
            else:
                # count how many full 20-epoch windows after 60
                decay_steps = (epoch - 61) // 20 + 1 if epoch >= 61 else 0
            lr = 5e-5 / (5.0 ** max(0, decay_steps))
        return [lr for _ in lr_scheduler.base_values]

    lr_scheduler._get_lr = _stage1_mlp_lr

    return lr_scheduler
