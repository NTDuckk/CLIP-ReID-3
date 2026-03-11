# CLIP-ReID: Tổng hợp kiến trúc 2-Stage Training

> Tài liệu tổng hợp toàn bộ pipeline huấn luyện CLIP-ReID, từ load data đến evaluation.
> Entry point: `train_clipreid.py` | Config: `configs/person/vit_clipreid.yml` + `config/defaults_base.py`

---

## 1. Tổng quan Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIP-ReID 2-Stage Training                │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Text Prompt Learning (chỉ train PromptLearner)    │
│  Stage 2: Image Encoder Fine-tuning (freeze text, train ảnh)│
└─────────────────────────────────────────────────────────────┘
```

**Ý tưởng cốt lõi**: Tận dụng CLIP pretrained (image encoder + text encoder) cho bài toán Person Re-Identification. Stage 1 học các text prompt phù hợp cho từng identity. Stage 2 fine-tune image encoder với prompt đã học cố định, kết hợp loss ID + Triplet + Image-to-Text matching.

---

## 2. Load Dataset — Market1501

### 2.1 Dataset class (`datasets/market1501.py`)

```python
class Market1501(BaseImageDataset):
    dataset_dir = 'Market-1501-v15.09.15'
```

- **Cấu trúc**: 
  - `bounding_box_train/` → 12,936 ảnh, 751 identities
  - `query/` → 3,368 ảnh
  - `bounding_box_test/` (gallery) → 15,913 ảnh
- **Mỗi sample** = tuple `(img_path, pid, camid, viewid=0)`
- Tên file format: `{pid}_{camid}_...jpg`, parse bằng regex `r'([-\d]+)_c(\d)'`
- Train set: pid được relabel `0 → N-1`, camid bắt đầu từ 0

### 2.2 ImageDataset (`datasets/bases.py`)

```python
class ImageDataset(Dataset):
    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)  # PIL Image → RGB
        img = self.transform(img)
        return img, pid, camid, trackid, img_path.split('/')[-1]
```

### 2.3 DataLoader (`datasets/make_dataloader_clipreid.py`)

Tạo **3 DataLoader** riêng biệt:

| DataLoader | Dùng cho | Transform | Sampler | Batch size |
|---|---|---|---|---|
| `train_loader_stage1` | Stage 1 | `val_transforms` (không augment) | Shuffle | `STAGE1.IMS_PER_BATCH` = 64 |
| `train_loader_stage2` | Stage 2 | `train_transforms` (có augment) | `RandomIdentitySampler` | `STAGE2.IMS_PER_BATCH` = 64 |
| `val_loader` | Evaluation | `val_transforms` | Sequential | `TEST.IMS_PER_BATCH` = 64 |

**Train transforms** (Stage 2 — có data augmentation):
```python
T.Resize([256, 128], interpolation=3)   # Bicubic
T.RandomHorizontalFlip(p=0.5)
T.Pad(10)
T.RandomCrop([256, 128])
T.ToTensor()
T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
RandomErasing(probability=0.5, mode='pixel')
```

**Val transforms** (Stage 1 & Eval — không augmentation):
```python
T.Resize([256, 128])
T.ToTensor()
T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
```

**Sampler Stage 2**: `RandomIdentitySampler` — chọn `IMS_PER_BATCH / NUM_INSTANCE = 64/4 = 16` identities, mỗi identity chọn 4 ảnh → đảm bảo mỗi batch có positive pairs cho triplet loss.

**Return**: `train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, cam_num, view_num`

---

## 3. Build Model — CLIP Pretrained

### 3.1 Load CLIP pretrained (`model/make_model_clipreid.py` + `model/clip/`)

```python
# Tính patch grid resolution
h_resolution = (256 - 16) // 16 + 1 = 16
w_resolution = (128 - 16) // 16 + 1 = 8

clip_model = load_clip_to_cpu('ViT-B-16', h_resolution=16, w_resolution=8, vision_stride_size=16)
```

**`load_clip_to_cpu()`**: Download pretrained CLIP ViT-B/16 từ OpenAI → build model với `clip.build_model()`, truyền `h_resolution`, `w_resolution` để resize positional embedding cho input 256×128.

### 3.2 Kiến trúc `build_transformer` (model chính)

```
build_transformer
├── image_encoder: VisionTransformer (từ CLIP)      ← ViT-B/16, 768-dim
├── text_encoder: TextEncoder (từ CLIP transformer)  ← 512-dim projected
├── prompt_learner: PromptLearner                    ← Learnable prompts
├── classifier: Linear(768 → num_classes)            ← ID classification (image feat)
├── classifier_proj: Linear(512 → num_classes)       ← ID classification (projected feat)
├── bottleneck: BatchNorm1d(768)                     ← BNNeck cho image feature
├── bottleneck_proj: BatchNorm1d(512)                ← BNNeck cho projected feature
└── cv_embed (optional): SIE camera/view embedding
```

**Feature dimensions** (ViT-B-16):
- `in_planes = 768` (ViT hidden dim)
- `in_planes_proj = 512` (CLIP projection dim, cùng không gian với text)

### 3.3 VisionTransformer — Image Encoder (`model/clip/model.py`)

```python
class VisionTransformer:
    def forward(self, x, cv_emb=None):
        x = self.conv1(x)              # Patch embedding: [B,3,256,128] → [B,768,16,8]
        x = reshape + permute           # → [B, 128, 768]
        x = cat([CLS_token, x])        # → [B, 129, 768]  (128 patches + 1 CLS)
        if cv_emb: x[:,0] += cv_emb    # SIE embedding (optional)
        x = x + positional_embedding   # Add position info
        x = self.ln_pre(x)
        
        # Chia transformer thành 2 phần:
        x11 = transformer.resblocks[:11](x)   # Layer 0-10 output
        x12 = transformer.resblocks[11](x11)  # Layer 11 output (final)
        
        x12 = self.ln_post(x12)
        xproj = x12 @ self.proj               # Project 768 → 512
        
        return x11, x12, xproj
        # x11:    [B, 129, 768]  — second-to-last layer features
        # x12:    [B, 129, 768]  — last layer features
        # xproj:  [B, 129, 512]  — projected features (CLIP space)
```

**3 output features** và cách sử dụng trong `build_transformer.forward()` (ViT-B-16):
```python
image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed)
img_feature_last = image_features_last[:, 0]   # CLS token, [B, 768]
img_feature      = image_features[:, 0]         # CLS token, [B, 768]  
img_feature_proj = image_features_proj[:, 0]    # CLS token, [B, 512]
```

### 3.4 PromptLearner — Học text prompt (`model/make_model_clipreid.py`)

```python
class PromptLearner(nn.Module):
    # Template: "A photo of a [X] [X] [X] [X] person."
    # [X] = learnable class-specific vectors
```

**Cấu trúc prompt**:
```
[prefix]          [cls_ctx]        [suffix]
"A photo of a"    [V1 V2 V3 V4]   "person."
(fixed tokens)    (learnable)      (fixed tokens)
```

- `cls_ctx`: `nn.Parameter(num_classes × 4 × 512)` — mỗi class có 4 vector learnable riêng
- `token_prefix`: embedding cố định của "A photo of a" (5 tokens)
- `token_suffix`: embedding cố định của "person." + padding
- `forward(label)`: ghép prefix + cls_ctx[label] + suffix → input cho text encoder

### 3.5 TextEncoder (`model/make_model_clipreid.py`)

```python
class TextEncoder(nn.Module):
    def forward(self, prompts, tokenized_prompts):
        x = prompts + positional_embedding     # Add position
        x = transformer(x)                     # CLIP text transformer
        x = ln_final(x)
        # Lấy feature tại vị trí EOT token
        x = x[arange, tokenized_prompts.argmax(-1)] @ text_projection  # → [B, 512]
        return x
```

Output: text feature `[B, 512]` — cùng không gian với `image_features_proj`.

---

## 4. Stage 1: Text Prompt Learning

> **File**: `processor/processor_clipreid_stage1.py`
> **Mục tiêu**: Học `PromptLearner.cls_ctx` sao cho text feature khớp với image feature (contrastive learning)

### 4.1 Quá trình chi tiết

```
1. TRÍCH XUẤT IMAGE FEATURES (frozen, 1 lần duy nhất):
   ┌─────────────────────────────────────────────────────────┐
   │ with torch.no_grad():                                    │
   │   for img, vid in train_loader_stage1:                   │
   │     image_feature = model(img, vid, get_image=True)      │
   │     # → image_encoder(img) → lấy xproj[:,0] → [B, 512]  │
   │   image_features_list = stack(all_features)  # [N, 512]  │
   │   labels_list = stack(all_labels)            # [N]        │
   └─────────────────────────────────────────────────────────┘
   
2. TRAINING LOOP (120 epochs):
   ┌─────────────────────────────────────────────────────────┐
   │ for epoch in range(1, 121):                              │
   │   iter_list = random_permutation(N)                      │
   │   for each mini-batch:                                   │
   │     target = labels_list[batch_indices]                   │
   │     image_features = image_features_list[batch_indices]   │
   │                                                           │
   │     # Forward: sinh text features TỪ learnable prompts   │
   │     text_features = model(label=target, get_text=True)    │
   │     # → PromptLearner(target) → TextEncoder() → [B, 512] │
   │                                                           │
   │     # Loss: Symmetric contrastive                         │
   │     loss_i2t = SupConLoss(image_feat, text_feat, ...)     │
   │     loss_t2i = SupConLoss(text_feat, image_feat, ...)     │
   │     loss = loss_i2t + loss_t2i                            │
   │                                                           │
   │     # Backward + Update (chỉ PromptLearner)              │
   │     scaler.scale(loss).backward()                         │
   │     scaler.step(optimizer)                                │
   └─────────────────────────────────────────────────────────┘
```

### 4.2 SupConLoss (Supervised Contrastive Loss) — `loss/supcontrast.py`

```python
class SupConLoss(nn.Module):
    # temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        # mask[i,j] = 1 nếu t_label[i] == i_targets[j] (cùng identity)
        mask = (t_label.unsqueeze(1) == i_targets.unsqueeze(0)).float()
        
        logits = (text_features @ image_features.T) / temperature
        logits = logits - logits.max(dim=1, keepdim=True)  # stability
        
        log_prob = logits - log(exp(logits).sum(1, keepdim=True))
        loss = -(mask * log_prob).sum(1) / mask.sum(1)     # avg over positives
        return loss.mean()
```

**Ý nghĩa**: Kéo text feature của class i gần với image features cùng class, đẩy xa image features khác class. Tương tự InfoNCE loss có supervised signal.

### 4.3 Optimizer & Scheduler Stage 1

```python
# Chỉ optimize PromptLearner parameters:
optimizer_1stage = make_optimizer_1stage(cfg, model)
# → Lọc: chỉ params có "prompt_learner" trong key
# → Adam, lr=0.00035, weight_decay=1e-4

scheduler_1stage = CosineLRScheduler(
    optimizer, t_initial=120,  lr_min=1e-6,
    warmup_lr_init=1e-5, warmup_t=5    # 5 epochs warmup
)
```

### 4.4 Tham số nào được cập nhật?

| Component | Trainable? |
|---|---|
| `PromptLearner.cls_ctx` (num_classes × 4 × 512) | ✅ **CÓ** |
| Image Encoder (ViT-B/16) | ❌ Frozen |
| Text Encoder (Transformer) | ❌ Frozen |
| Classifiers, BNNeck | ❌ Frozen (chưa dùng) |

---

## 5. Stage 2: Image Encoder Fine-tuning

> **File**: `processor/processor_clipreid_stage2.py`
> **Mục tiêu**: Fine-tune image encoder + classifiers, sử dụng text features đã học ở Stage 1 làm cầu nối

### 5.1 Chuẩn bị Text Features (frozen, 1 lần)

```python
# Trích xuất text features cho TẤT CẢ classes (dùng prompt đã học)
with torch.no_grad():
    for i in range(0, num_classes, batch):
        text_feature = model(label=l_list, get_text=True)
    text_features = cat(all_text_features)  # [num_classes, 512]
```

### 5.2 Training Loop (60 epochs)

```
for epoch in range(1, 61):
  for img, vid, cam, view in train_loader_stage2:
    ┌─────────────────────────────────────────────────────────┐
    │ FORWARD PASS:                                            │
    │   score, feat, image_features = model(img, label, ...)   │
    │                                                           │
    │   # model.forward() trả về (training mode):              │
    │   #   score = [cls_score, cls_score_proj]                 │
    │   #     cls_score     = classifier(BN(img_feat))          │
    │   #     cls_score_proj = classifier_proj(BN(img_feat_proj))│
    │   #   feat = [img_feat_last, img_feat, img_feat_proj]     │
    │   #     img_feat_last = x11[:,0]  (768-dim, layer 10)     │
    │   #     img_feat      = x12[:,0]  (768-dim, layer 11)     │
    │   #     img_feat_proj = xproj[:,0] (512-dim, projected)   │
    │   #   image_features = img_feat_proj  (512-dim)           │
    │                                                           │
    │ IMAGE-TO-TEXT MATCHING:                                   │
    │   logits = image_features @ text_features.T               │
    │   # logits[i,j] = similarity ảnh i với text class j      │
    │   # Shape: [batch_size, num_classes]                      │
    │                                                           │
    │ LOSS COMPUTATION:                                         │
    │   loss = loss_fn(score, feat, target, target_cam, logits) │
    └─────────────────────────────────────────────────────────┘
```

### 5.3 Loss Function chi tiết (`loss/make_loss.py`)

Với config `softmax_triplet` + `triplet` + `label_smooth='on'`:

```
loss = ID_LOSS_WEIGHT × ID_LOSS 
     + TRIPLET_LOSS_WEIGHT × TRI_LOSS 
     + I2T_LOSS_WEIGHT × I2T_LOSS

Trong đó:
  ID_LOSS_WEIGHT    = 0.25
  TRIPLET_LOSS_WEIGHT = 1.0
  I2T_LOSS_WEIGHT   = 1.0
```

#### a) ID Loss (CrossEntropyLabelSmooth)

```python
# score = [cls_score, cls_score_proj]  — cả 2 branch
ID_LOSS = xent(cls_score, target) + xent(cls_score_proj, target)

# CrossEntropyLabelSmooth (epsilon=0.1):
#   targets_smooth = (1-0.1)*one_hot + 0.1/num_classes
#   loss = (-targets_smooth * log_softmax(inputs)).mean(0).sum()
```

- `cls_score = classifier(bottleneck(img_feature))` — 768-dim branch  
- `cls_score_proj = classifier_proj(bottleneck_proj(img_feature_proj))` — 512-dim branch

#### b) Triplet Loss (Hard Mining)

```python
# feat = [img_feat_last, img_feat, img_feat_proj]  — cả 3 features
TRI_LOSS = triplet(img_feat_last, target)[0] 
         + triplet(img_feat, target)[0] 
         + triplet(img_feat_proj, target)[0]

# TripletLoss(margin=0.3):
#   dist_mat = euclidean_dist(feat, feat)
#   dist_ap, dist_an = hard_example_mining(dist_mat, labels)
#   loss = relu(dist_ap - dist_an + margin).mean()
```

**Hard example mining**: Trong mỗi batch, cho mỗi anchor:
- Tìm **hardest positive** (cùng ID, khoảng cách xa nhất)
- Tìm **hardest negative** (khác ID, khoảng cách gần nhất)

#### c) I2T Loss (Image-to-Text CrossEntropy)

```python
# logits = image_features @ text_features.T  → [B, num_classes]
I2TLOSS = xent(logits, target)
# Xem logits như classification score trong CLIP space
# → Ảnh phải gần nhất với text prompt của đúng class
```

**Tổng loss Stage 2**:
```
loss = 0.25 × (CE_768 + CE_512) + 1.0 × (Tri_last + Tri_feat + Tri_proj) + 1.0 × CE_i2t
```

### 5.4 Optimizer & Scheduler Stage 2

```python
# Freeze text_encoder và prompt_learner
# Train: image_encoder, classifiers, bottlenecks, cv_embed
optimizer_2stage = Adam(params, lr=5e-6, weight_decay=1e-4)
# bias: lr *= 2

scheduler_2stage = WarmupMultiStepLR(
    optimizer, milestones=[30, 50], gamma=0.1,
    warmup_factor=0.1, warmup_iters=10, warmup_method='linear'
)
# LR schedule: warmup 10 iters → lr=5e-6 → ×0.1 tại epoch 30 → ×0.1 tại epoch 50
```

### 5.5 Tham số nào được cập nhật?

| Component | Trainable? |
|---|---|
| Image Encoder (ViT-B/16) | ✅ **CÓ** (lr=5e-6, rất nhỏ) |
| `classifier` (768→N) | ✅ **CÓ** |
| `classifier_proj` (512→N) | ✅ **CÓ** |
| `bottleneck` (BN 768) | ✅ **CÓ** |
| `bottleneck_proj` (BN 512) | ✅ **CÓ** |
| `cv_embed` (SIE, nếu bật) | ✅ **CÓ** |
| PromptLearner | ❌ **Frozen** |
| Text Encoder | ❌ **Frozen** |

---

## 6. Evaluation

> **File**: `utils/metrics.py` | Gọi trong `processor_clipreid_stage2.py`

### 6.1 Feature Extraction (Inference mode)

```python
model.eval()
for img, pid, camid, ... in val_loader:
    feat = model(img, cam_label=camids, view_label=target_view)
    # Inference mode → forward() trả về:
    #   torch.cat([feat, feat_proj], dim=1)  
    #   feat:      BN(img_feature)       → 768-dim  (hoặc raw nếu neck_feat='before')
    #   feat_proj: BN(img_feature_proj)  → 512-dim
    #   → Concatenated: 768 + 512 = 1280-dim feature vector
    evaluator.update((feat, pid, camid))
```

### 6.2 Distance & Ranking (`R1_mAP_eval`)

```python
# 1. Normalize features (L2 norm)
feats = F.normalize(feats, dim=1, p=2)

# 2. Split query / gallery
qf = feats[:num_query]      # query features
gf = feats[num_query:]      # gallery features

# 3. Compute distance matrix (Euclidean)
distmat = euclidean_distance(qf, gf)  # [num_query × num_gallery]

# 4. Evaluate với market1501 protocol
cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
```

### 6.3 Market-1501 Evaluation Protocol

- Loại bỏ gallery samples có **cùng pid VÀ cùng camid** với query (same camera, same person)
- **CMC (Cumulative Matching Characteristics)**: Rank-1, Rank-5, Rank-10 accuracy
- **mAP (mean Average Precision)**: Trung bình AP qua tất cả valid queries

---

## 7. Sơ đồ tổng hợp Flow

```
╔══════════════════════════════════════════════════════════════════════╗
║                         STAGE 1 (120 epochs)                        ║
║                                                                      ║
║  Image(train) ──→ Image Encoder ──→ image_proj_feat [N,512]         ║
║                    (FROZEN)          (trích 1 lần, cache)            ║
║                                                                      ║
║  Labels ──→ PromptLearner ──→ TextEncoder ──→ text_feat [B,512]     ║
║              (TRAINING)         (FROZEN)                             ║
║                                                                      ║
║  Loss = SupConLoss(img→txt) + SupConLoss(txt→img)                   ║
║  Update: PromptLearner.cls_ctx only                                  ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         STAGE 2 (60 epochs)                         ║
║                                                                      ║
║  All labels ──→ PromptLearner ──→ TextEncoder ──→ text_feat [C,512] ║
║                  (FROZEN)          (FROZEN)       (trích 1 lần)      ║
║                                                                      ║
║  Image(train) ──→ Image Encoder ──→ 3 features:                     ║
║                    (TRAINING)        img_last  [B,768] (layer 10)    ║
║                                      img_feat  [B,768] (layer 11)   ║
║                                      img_proj  [B,512] (projected)  ║
║                                                                      ║
║  BNNeck:  feat     = BN(img_feat)   → classifier → cls_score        ║
║           feat_proj = BN(img_proj)  → classifier_proj → cls_score_p ║
║                                                                      ║
║  I2T matching: logits = img_proj @ text_feat.T  → [B, C]            ║
║                                                                      ║
║  Loss = 0.25×(CE₁+CE₂) + 1.0×(Tri₁+Tri₂+Tri₃) + 1.0×CE_i2t      ║
║  Update: Image Encoder, Classifiers, BNNeck                         ║
╚══════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════╗
║                         EVALUATION                                   ║
║                                                                      ║
║  Image ──→ Image Encoder ──→ BNNeck ──→ cat(feat, feat_proj)        ║
║                                          → 1280-dim feature          ║
║                                                                      ║
║  Query vs Gallery: Euclidean Distance → Rank → CMC + mAP            ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 8. Hyperparameters tóm tắt (ViT-B-16, Market1501)

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Epochs | 120 | 60 |
| Batch size | 64 | 64 (16 IDs × 4 imgs) |
| Optimizer | Adam | Adam |
| Base LR | 3.5e-4 | 5e-6 |
| LR Schedule | Cosine (warmup 5 ep) | MultiStep [30,50] γ=0.1 |
| Warmup | 5 epochs, lr_init=1e-5 | 10 iters, factor=0.1 |
| Weight decay | 1e-4 | 1e-4 |
| Loss | SupConLoss (symmetric) | ID + Triplet + I2T |
| Trainable params | PromptLearner only | Image Encoder + heads |
| Mixed precision | ✅ AMP | ✅ AMP |

---

## 9. File Reference Map

```
train_clipreid.py                          ← Entry point, orchestrates 2 stages
├── config/defaults_base.py                ← Default config values
├── configs/person/vit_clipreid.yml        ← ViT-B-16 specific config
├── datasets/
│   ├── make_dataloader_clipreid.py        ← 3 DataLoaders creation
│   ├── market1501.py                      ← Market-1501 dataset parser
│   ├── bases.py                           ← ImageDataset, BaseImageDataset
│   └── sampler.py                         ← RandomIdentitySampler
├── model/
│   ├── make_model_clipreid.py             ← build_transformer, PromptLearner, TextEncoder
│   └── clip/
│       ├── clip.py                        ← CLIP download & build
│       ├── model.py                       ← VisionTransformer, ModifiedResNet, CLIP class
│       └── simple_tokenizer.py            ← BPE tokenizer
├── loss/
│   ├── make_loss.py                       ← Loss function factory (ID+Tri+I2T)
│   ├── supcontrast.py                     ← SupConLoss (Stage 1)
│   ├── softmax_loss.py                    ← CrossEntropyLabelSmooth
│   └── triplet_loss.py                    ← TripletLoss + hard mining
├── processor/
│   ├── processor_clipreid_stage1.py       ← Stage 1 training loop
│   └── processor_clipreid_stage2.py       ← Stage 2 training loop + eval
├── solver/
│   ├── make_optimizer_prompt.py           ← Optimizer cho Stage 1 & 2
│   ├── scheduler_factory.py              ← CosineLRScheduler (Stage 1)
│   ├── lr_scheduler.py                   ← WarmupMultiStepLR (Stage 2)
│   └── cosine_lr.py                      ← CosineLRScheduler implementation
└── utils/
    ├── metrics.py                         ← R1_mAP_eval, eval_func, distance functions
    └── logger.py                          ← Logging setup
```
