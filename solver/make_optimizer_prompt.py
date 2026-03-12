import torch


def make_optimizer_1stage(cfg, model):
    # Freeze all parameters then enable gradients only for parameters
    # whose name contains "inversion_prompt_learner" (stage1 text inversion prompts).
    for p in model.parameters():
        p.requires_grad = False

    params = []
    found = []
    for name, param in model.named_parameters():
        if "inversion_prompt_learner" in name:
            param.requires_grad = True
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params.append({"params": [param], "lr": lr, "weight_decay": weight_decay})
            found.append(name)

    if len(found) == 0:
        raise RuntimeError("No parameters with 'inversion_prompt_learner' found for stage1 optimizer")

    opt_name = cfg.SOLVER.STAGE1.OPTIMIZER_NAME
    if opt_name == 'SGD':
        momentum = getattr(cfg.SOLVER.STAGE1, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.STAGE1.BASE_LR, momentum=momentum, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE1.BASE_LR, weight_decay=cfg.SOLVER.STAGE1.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, opt_name)(params)

    return optimizer



def make_optimizer_2stage(cfg, model, center_criterion):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue   
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
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
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.STAGE2.BASE_LR, weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.STAGE2.CENTER_LR)

    return optimizer, optimizer_center
