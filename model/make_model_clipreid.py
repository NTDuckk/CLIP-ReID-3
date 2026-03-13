import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[
            torch.arange(x.shape[0], device=x.device),
            tokenized_prompts.argmax(dim=-1).to(x.device)
        ] @ self.text_projection
        return x


class IM2TEXT(nn.Module):
    """AP-Attack style MLP used to map features into a pseudo-token embedding."""
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=3, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)


class CrossModalTransformer(nn.Module):
    """
    Lightweight transformer used after cross-attention.
    Input / output format follows the referenced cross_former implementation:
    sequence-first tensors with shape [L, B, D].
    """
    def __init__(self, width=512, layers=1, heads=8, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x)


class DetailCrossAttentionBlock(nn.Module):
    """
    Cross-attention block rewritten to follow the same calling pattern as:
        cross_attn(LN(q), LN(k), LN(v)) -> transformer -> LN_post

    Important: the input features are kept the same as the current
    Detail-focused Token Generation design:
    - q comes from the learnable prompt queries X
    - k, v come from concat([X, V_l])
    where V_l are the local patch features.
    """
    def __init__(self, dim=512, cmt_depth=1, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = max(1, dim // 64)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_modal_transformer = CrossModalTransformer(
            width=dim,
            layers=cmt_depth,
            heads=self.num_heads,
            dropout=dropout,
        )
        self.ln_pre_t = nn.LayerNorm(dim)
        self.ln_pre_i = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)

    def forward(self, prompt_queries, local_patches):
        if prompt_queries.dim() != 3:
            raise RuntimeError(f"prompt_queries must be [B, M, D], got {tuple(prompt_queries.shape)}")
        if local_patches.dim() != 3:
            raise RuntimeError(f"local_patches must be [B, N, D], got {tuple(local_patches.shape)}")
        if prompt_queries.size(0) != local_patches.size(0):
            raise RuntimeError(
                f"Batch mismatch: prompt_queries {tuple(prompt_queries.shape)} vs local_patches {tuple(local_patches.shape)}"
            )
        if prompt_queries.size(-1) != local_patches.size(-1):
            raise RuntimeError(
                f"Feature-dim mismatch: prompt_queries {tuple(prompt_queries.shape)} vs local_patches {tuple(local_patches.shape)}"
            )

        kv_input = torch.cat([prompt_queries, local_patches], dim=1)
        x = self.cross_attn(
            self.ln_pre_t(prompt_queries),
            self.ln_pre_i(kv_input),
            self.ln_pre_i(kv_input),
            need_weights=False,
        )[0]
        x = x.permute(1, 0, 2)  # [B, M, D] -> [M, B, D]
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # [M, B, D] -> [B, M, D]
        x = self.ln_post(x)
        return x


class DetailTokenNetwork(nn.Module):
    """
    One independent local-token generator.
    Each network has its own 3 randomly initialized prompt queries.
    After cross attention, the 3 queries are average pooled and passed through IM2TEXT.
    """
    def __init__(self, dim=512, num_queries=3, num_blocks=6, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.prompt_queries = nn.Parameter(torch.empty(1, num_queries, dim))
        nn.init.normal_(self.prompt_queries, std=0.02)

        self.block1 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)
        self.block2 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)
        self.block3 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)
        self.block4 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)
        self.block5 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)
        self.block6 = DetailCrossAttentionBlock(dim=dim, cmt_depth=1, dropout=dropout)

        self.mapper = IM2TEXT(
            embed_dim=dim,
            middle_dim=dim,
            output_dim=dim,
            n_layer=3,
            dropout=dropout,
        )

    def forward(self, local_patches):
        if local_patches.dim() != 3:
            raise RuntimeError(f"local_patches must be [B, N, D], got {tuple(local_patches.shape)}")
        if local_patches.size(-1) != self.dim:
            raise RuntimeError(
                f"Expected local patch dim {self.dim}, got {local_patches.size(-1)}"
            )

        bsz = local_patches.size(0)
        queries = self.prompt_queries.expand(bsz, -1, -1)
        queries = self.block1(queries, local_patches)
        queries = self.block2(queries, local_patches)
        queries = self.block3(queries, local_patches)
        queries = self.block4(queries, local_patches)
        queries = self.block5(queries, local_patches)
        queries = self.block6(queries, local_patches)
        pooled = queries.mean(dim=1)
        token = self.mapper(pooled)
        return token


class InversionPromptLearner(nn.Module):
    """
    Stage-1 prompt learner with the requested prompt design:
    "A photo of a S1 person with S2 clothes, S3 hairstyles, S4 shoes and carrying S5"

    - S1: from global CLS through an IM2TEXT MLP.
    - S2..S5: from local patch tokens through four independent DetailTokenNetwork modules.
    - No loop is used to define or call the five token networks.
    """
    def __init__(self, dataset_name, dtype, token_embedding, clip_proj_dim=512):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            raise NotImplementedError("This prompt design is implemented only for person ReID datasets.")

        template = "A photo of a X person with X clothes , X hairstyles , X shoes and carrying X ."
        self.dtype = dtype
        self.clip_proj_dim = clip_proj_dim

        tokenized_prompts = clip.tokenize(template).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.tokenized_prompts = tokenized_prompts
        self.register_buffer("template_embedding", embedding)

        x_token_id = clip.tokenize("X")[0, 1].item()
        x_positions = (tokenized_prompts[0] == x_token_id).nonzero(as_tuple=False).view(-1)
        if x_positions.numel() != 5:
            raise RuntimeError(f"Expected 5 placeholder X tokens, got {x_positions.numel()}")
        self.register_buffer("x_positions", x_positions)

        ctx_dim = embedding.shape[-1]
        if ctx_dim != clip_proj_dim:
            raise RuntimeError(
                f"Token embedding dim ({ctx_dim}) must match projected image dim ({clip_proj_dim})"
            )

        self.prompt_text_person = IM2TEXT(
            embed_dim=clip_proj_dim,
            middle_dim=clip_proj_dim,
            output_dim=ctx_dim,
            n_layer=3,
            dropout=0.1,
        )
        self.prompt_text_clothes = DetailTokenNetwork(dim=clip_proj_dim, num_queries=3, num_blocks=6, dropout=0.1)
        self.prompt_text_hairstyles = DetailTokenNetwork(dim=clip_proj_dim, num_queries=3, num_blocks=6, dropout=0.1)
        self.prompt_text_shoes = DetailTokenNetwork(dim=clip_proj_dim, num_queries=3, num_blocks=6, dropout=0.1)
        self.prompt_text_carrying = DetailTokenNetwork(dim=clip_proj_dim, num_queries=3, num_blocks=6, dropout=0.1)

    def forward(self, global_cls, local_patches):
        if global_cls.dim() != 2:
            raise RuntimeError(f"global_cls must be [B, D], got {tuple(global_cls.shape)}")
        if local_patches.dim() != 3:
            raise RuntimeError(f"local_patches must be [B, N, D], got {tuple(local_patches.shape)}")
        if global_cls.size(0) != local_patches.size(0):
            raise RuntimeError(
                f"Batch mismatch: global_cls {tuple(global_cls.shape)} vs local_patches {tuple(local_patches.shape)}"
            )
        if global_cls.size(1) != self.clip_proj_dim:
            raise RuntimeError(
                f"Expected global_cls dim {self.clip_proj_dim}, got {global_cls.size(1)}"
            )
        if local_patches.size(-1) != self.clip_proj_dim:
            raise RuntimeError(
                f"Expected local patch dim {self.clip_proj_dim}, got {local_patches.size(-1)}"
            )

        prompt_person = self.prompt_text_person(global_cls.float()).unsqueeze(dim=1)
        prompt_clothes = self.prompt_text_clothes(local_patches.float()).unsqueeze(dim=1)
        prompt_hairstyles = self.prompt_text_hairstyles(local_patches.float()).unsqueeze(dim=1)
        prompt_shoes = self.prompt_text_shoes(local_patches.float()).unsqueeze(dim=1)
        prompt_carrying = self.prompt_text_carrying(local_patches.float()).unsqueeze(dim=1)

        bsz = global_cls.size(0)
        prompts = self.template_embedding.expand(bsz, -1, -1).clone()
        prompts[:, self.x_positions[0], :] = prompt_person[:, 0, :].type(self.dtype)
        prompts[:, self.x_positions[1], :] = prompt_clothes[:, 0, :].type(self.dtype)
        prompts[:, self.x_positions[2], :] = prompt_hairstyles[:, 0, :].type(self.dtype)
        prompts[:, self.x_positions[3], :] = prompt_shoes[:, 0, :].type(self.dtype)
        prompts[:, self.x_positions[4], :] = prompt_carrying[:, 0, :].type(self.dtype)
        return prompts


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4

        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label]
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,
                cls_ctx,
                suffix,
            ],
            dim=1,
        )
        return prompts


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.inversion_prompt_learner = InversionPromptLearner(
            dataset_name, clip_model.dtype, clip_model.token_embedding,
            clip_proj_dim=self.in_planes_proj
        )
        self.text_encoder = TextEncoder(clip_model)

    def _get_cv_embed(self, cam_label=None, view_label=None):
        if self.model_name != 'ViT-B-16':
            return None
        if cam_label is not None and view_label is not None:
            return self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label is not None:
            return self.sie_coe * self.cv_embed[cam_label]
        elif view_label is not None:
            return self.sie_coe * self.cv_embed[view_label]
        return None

    def _extract_prompt_image_features(self, x, cam_label=None, view_label=None):
        if self.model_name != 'ViT-B-16':
            raise NotImplementedError("The requested local-patch prompt generation is implemented for ViT-B-16 only.")

        cv_embed = self._get_cv_embed(cam_label=cam_label, view_label=view_label)
        image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)

        if image_features_proj.dim() != 3:
            raise RuntimeError(
                f"Expected projected image tokens [B, N+1, D], got {tuple(image_features_proj.shape)}"
            )
        if image_features_proj.size(-1) != self.in_planes_proj:
            raise RuntimeError(
                f"Projected image dim mismatch: expected {self.in_planes_proj}, got {image_features_proj.size(-1)}"
            )
        if image_features_proj.size(1) < 2:
            raise RuntimeError(
                f"Need CLS plus patch tokens, but got token count {image_features_proj.size(1)}"
            )

        global_cls = image_features_proj[:, 0]
        local_patches = image_features_proj[:, 1:]
        if global_cls.dim() != 2 or local_patches.dim() != 3:
            raise RuntimeError(
                f"Unexpected projected token shapes: global_cls {tuple(global_cls.shape)}, local_patches {tuple(local_patches.shape)}"
            )
        return image_features_last, image_features, image_features_proj, global_cls, local_patches

    def forward(self, x=None, label=None, get_image=False, get_text=False,
                get_text_from_image=False, image_features_for_inversion=None,
                get_prompt_image_features=False, cam_label=None, view_label=None):
        if get_text_from_image:
            if not isinstance(image_features_for_inversion, (tuple, list)) or len(image_features_for_inversion) != 2:
                raise RuntimeError(
                    "image_features_for_inversion must be a tuple/list: (global_cls, local_patches)"
                )
            global_cls, local_patches = image_features_for_inversion
            prompts = self.inversion_prompt_learner(global_cls, local_patches)
            text_features = self.text_encoder(prompts, self.inversion_prompt_learner.tokenized_prompts)
            return text_features

        if get_text:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_prompt_image_features:
            _, _, _, global_cls, local_patches = self._extract_prompt_image_features(
                x, cam_label=cam_label, view_label=view_label
            )
            return global_cls, local_patches

        if get_image:
            if self.model_name == 'ViT-B-16':
                _, _, _, global_cls, _ = self._extract_prompt_image_features(
                    x, cam_label=cam_label, view_label=view_label
                )
                return global_cls
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            cv_embed = self._get_cv_embed(cam_label=cam_label, view_label=view_label)
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model
