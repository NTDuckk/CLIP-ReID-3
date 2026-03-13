import torch


# def make_optimizer_1stage(cfg, model):
#     # Freeze all parameters then enable gradients only for parameters
#     # whose name contains "inversion_prompt_learner" (stage1 text inversion prompts).
#     for p in model.parameters():
#         p.requires_grad = False

#     params = []
#     found = []
#     for name, param in model.named_parameters():
#         if "inversion_prompt_learner" in name:
#             param.requires_grad = True
#             # cap inversion LR to a safe value to avoid instability
#             base_lr = float(getattr(cfg.SOLVER.STAGE1, 'BASE_LR', 3e-4))
#             safe_peak = 5e-4
#             if base_lr > safe_peak:
#                 lr = safe_peak
#                 print(f"Capping Stage1 inversion lr {base_lr} -> {lr} to avoid instability")
#             else:
#                 lr = base_lr
#             weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
#             params.append({"params": [param], "lr": lr, "weight_decay": weight_decay})
#             found.append(name)

#     if len(found) == 0:
#         raise RuntimeError("No parameters with 'inversion_prompt_learner' found for stage1 optimizer")

#     opt_name = cfg.SOLVER.STAGE1.OPTIMIZER_NAME
#     if opt_name == 'SGD':
#         momentum = getattr(cfg.SOLVER.STAGE1, 'MOMENTUM', 0.9)
#         optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.STAGE1.BASE_LR, momentum=momentum, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
#     elif opt_name == 'AdamW':
#         optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
#     else:
#         optimizer = getattr(torch.optim, opt_name)(params)

#     return optimizer


def make_optimizer_1stage(cfg, model):
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    params = []
    found = []

    for name, param in model.named_parameters():
        # Train the whole Stage1 prompt generator branch:
        # - prompt_text_person
        # - prompt_text_clothes / hairstyles / shoes / carrying
        # - prompt_queries
        # - cross_attn
        # - cross_modal_transformer
        # - layer norms
        # - IM2TEXT mapper
        if "inversion_prompt_learner" in name:
            param.requires_grad = True
            params.append({
                "params": [param],
                "lr": cfg.SOLVER.STAGE1.BASE_LR,
                "weight_decay": cfg.SOLVER.STAGE1.WEIGHT_DECAY
            })
            found.append(name)

    if len(found) == 0:
        raise RuntimeError("No Stage1 trainable params found under 'inversion_prompt_learner'")

    print("Stage1 trainable params:")
    for n in found:
        print("  ", n)

    opt_name = cfg.SOLVER.STAGE1.OPTIMIZER_NAME
    if opt_name == "SGD":
        momentum = getattr(cfg.SOLVER.STAGE1, "MOMENTUM", 0.9)
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            momentum=momentum,
            weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY
        )
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.STAGE1.BASE_LR,
            weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY
        )
    else:
        optimizer = getattr(torch.optim, opt_name)(params)

    return optimizer


def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        # Exclude text encoder and any prompt learner params from stage2 training
        if "text_encoder" in key or "prompt_learner" in key or "inversion_prompt_learner" in key:
            value.requires_grad_(False)
            continue

        # Enable training for remaining parameters (ensure not empty)
        value.requires_grad_(True)

        lr = cfg.SOLVER.STAGE2.BASE_LR
        weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.STAGE2.BASE_LR * cfg.SOLVER.STAGE2.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.STAGE2.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if len(params) == 0:
        raise RuntimeError("No parameters found for stage2 optimizer. Check model parameter names or previous freezing.")

    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.STAGE2.BASE_LR, momentum=cfg.SOLVER.STAGE2.MOMENTUM, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center
