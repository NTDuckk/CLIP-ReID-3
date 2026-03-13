import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def _check_prompt_feature_shapes(global_cls, local_patches):
    if global_cls.dim() != 2:
        raise RuntimeError(f"global_cls must be [B, D], got {tuple(global_cls.shape)}")
    if local_patches.dim() != 3:
        raise RuntimeError(f"local_patches must be [B, N, D], got {tuple(local_patches.shape)}")
    if global_cls.size(0) != local_patches.size(0):
        raise RuntimeError(
            f"Batch mismatch between global_cls {tuple(global_cls.shape)} and local_patches {tuple(local_patches.shape)}"
        )
    if global_cls.size(-1) != local_patches.size(-1):
        raise RuntimeError(
            f"Feature dim mismatch between global_cls {tuple(global_cls.shape)} and local_patches {tuple(local_patches.shape)}"
        )

@torch.no_grad()
def _extract_and_cache_prompt_features(model, train_loader_stage1, device):
    labels = []
    global_cls_cache = []
    local_patch_cache = []

    model.eval()
    for _, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
        img = img.to(device)
        global_cls, local_patches = model(img, get_prompt_image_features=True)

        _check_prompt_feature_shapes(global_cls, local_patches)

        for label_i, cls_i, patches_i in zip(vid, global_cls, local_patches):
            labels.append(label_i)
            global_cls_cache.append(cls_i.half().cpu())
            local_patch_cache.append(patches_i.half().cpu())

    labels_list = torch.stack(labels, dim=0).cuda(non_blocking=True)
    global_cls_list = torch.stack(global_cls_cache, dim=0).cuda(non_blocking=True)
    local_patch_list = torch.stack(local_patch_cache, dim=0).cuda(non_blocking=True)

    _check_prompt_feature_shapes(global_cls_list, local_patch_list)

    return labels_list, global_cls_list, local_patch_list


def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    logger.info("Stage1 prompt follows FLaN-Net detail-focused generation with prompt: A photo of a S1 person with S2 clothes, S3 hairstyles, S4 shoes and carrying S5")

    # Step 1: Pre-extract frozen projected CLS + local patch tokens.
    logger.info("Caching frozen CLS + patch tokens for Stage1 prompt generation...")
    labels_list, global_cls_list, local_patch_list = _extract_and_cache_prompt_features(
        model=model,
        train_loader_stage1=train_loader_stage1,
        device=device,
    )

    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]
    i_ter = num_image // batch

    # Step 2: Train FLaN-style prompt generator with contrastive loss.
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            if b_list.numel() == 0:
                continue

            target = labels_list[b_list]
            global_cls = global_cls_list[b_list]
            local_patches = local_patch_list[b_list]
            _check_prompt_feature_shapes(global_cls, local_patches)

            with amp.autocast(enabled=True):
                text_features = model(
                    image_features_for_inversion=(global_cls, local_patches),
                    get_text_from_image=True
                )

            if text_features.dim() != 2:
                raise RuntimeError(f"text_features must be [B, D], got {tuple(text_features.shape)}")
            if text_features.size(0) != global_cls.size(0):
                raise RuntimeError(
                    f"Batch mismatch between text_features {tuple(text_features.shape)} and global_cls {tuple(global_cls.shape)}"
                )
            if text_features.size(1) != global_cls.size(1):
                raise RuntimeError(
                    f"Feature dim mismatch between text_features {tuple(text_features.shape)} and global_cls {tuple(global_cls.shape)}"
                )

            loss_i2t = xent(global_cls, text_features, target, target)
            loss_t2i = xent(text_features, global_cls, target, target)
            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), b_list.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (i + 1), i_ter + 1, loss_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch))
                    )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch))
                )

    # Step 3: Compute averaged text features per ID.
    logger.info("Computing averaged Stage1 text features per ID...")
    num_classes = labels_list.max().item() + 1
    model.eval()
    all_text_features = []
    with torch.no_grad():
        for i in range(0, num_image, batch):
            end = min(i + batch, num_image)
            global_cls = global_cls_list[i:end]
            local_patches = local_patch_list[i:end]
            _check_prompt_feature_shapes(global_cls, local_patches)

            with amp.autocast(enabled=True):
                text_feats = model(
                    image_features_for_inversion=(global_cls, local_patches),
                    get_text_from_image=True
                )
            all_text_features.append(text_feats.float().cpu())

    all_text_features = torch.cat(all_text_features, dim=0)  # [N, D]
    if all_text_features.dim() != 2:
        raise RuntimeError(f"Expected all_text_features to be [N, D], got {tuple(all_text_features.shape)}")

    avg_text_features = torch.zeros(num_classes, all_text_features.shape[-1])
    labels_cpu = labels_list.cpu()
    for c in range(num_classes):
        mask = (labels_cpu == c)
        if mask.sum() > 0:
            avg_text_features[c] = all_text_features[mask].mean(dim=0)

    avg_text_features = avg_text_features.cuda(non_blocking=True)
    logger.info("Averaged text features computed for {} classes with dim {}".format(
        num_classes, avg_text_features.shape[-1]
    ))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))

    return avg_text_features
