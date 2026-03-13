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
        if epoch <= 5:
            progress = float(epoch - 1) / 4.0
            lr = 1e-6 + progress * (5e-5 - 1e-6)
        elif epoch <= 60:
            lr = 5e-5
        else:
            decay_steps = (epoch - 60) // 20
            lr = 5e-5 / (5.0 ** decay_steps)
        return [lr for _ in lr_scheduler.base_values]

    lr_scheduler._get_lr = _stage1_mlp_lr

    return lr_scheduler
