# Colab Execution Guide — Deep Social Sentiment Analysis

> **Dành cho AI assistant**: Đọc toàn bộ file này trước khi thực hiện bất kỳ bước nào.  
> Thực hiện **tuần tự từ Step 0 → Step 10**. Không bỏ qua bước nào trừ khi có ghi chú rõ ràng.

---

## Project Overview

| Mục | Chi tiết |
|-----|----------|
| **Tên dự án** | Deep Social Sentiment Analysis |
| **Bài toán** | Phân loại cảm xúc tiếng Việt — 7 nhãn Ekman |
| **Kiến trúc** | Late Fusion: XLM-RoBERTa-base (text) + FT-Transformer (tabular) → MLP → 7-class softmax |
| **7 nhãn** | `joy`, `sadness`, `anger`, `fear`, `disgust`, `surprise`, `neutral` |
| **GitHub** | `https://github.com/nhiney/deep-social-sentiment-analysis` |
| **Drive root** | `MyDrive/colab_sentiment/` |

---

## Cấu Trúc Dữ Liệu

```
MyDrive/colab_sentiment/
├── data/
│   ├── raw/
│   │   ├── crawled_emotions.xlsx        # ~200 labeled posts (có sẵn)
│   │   ├── unlabeled_new_posts.json     # 990 Apify posts (có sẵn)
│   │   └── UIT-VSMEC.csv               # 6,927 Facebook comments (có sẵn)
│   └── processed/                       # tự động tạo khi chạy pipeline
├── models/                              # checkpoint lưu sau training
├── reports/                             # figures, metrics, ablation results
└── hf_cache/                            # HuggingFace model cache
```

---

## Yêu Cầu Môi Trường

- **GPU**: T4 (tối thiểu) — bắt buộc, không chạy CPU
- **Runtime**: Runtime → Change runtime type → **T4 GPU**
- **Python**: 3.10+
- **Working directory**: `/content/deep-social-sentiment-analysis`

---

## Quy Tắc Xử Lý Lỗi

> Áp dụng cho tất cả các bước bên dưới.

| Lỗi | Xử lý |
|-----|-------|
| `CUDA out of memory` | Giảm `batch_size` xuống 16 → 8 trong config, chạy lại |
| `ModuleNotFoundError` | Chạy lại **Step 4** pip install |
| `FileNotFoundError` data | Kiểm tra symlink Drive ở **Step 2** |
| Drive bị ngắt kết nối | Chạy lại **Step 2** (mount + symlink), tiếp tục từ bước đang dở |
| Bất kỳ lỗi nào khác | In toàn bộ traceback, **dừng lại, báo cáo** — không tự ý bỏ qua |

---

## Báo Cáo Sau Mỗi Bước

Sau mỗi step hoàn thành, in ra:
1. ✅ / ❌ trạng thái
2. Số liệu chính (số rows, loss, F1...)
3. Cảnh báo hoặc bất thường nếu có

---

## Các Bước Thực Hiện

---

### Step 0 — Kiểm Tra GPU

```python
import torch

print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU             :', torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM            : {vram:.1f} GB')
else:
    raise RuntimeError('GPU không khả dụng. Vào Runtime → Change runtime type → T4 GPU')

!nvidia-smi | head -15
```

**Kết quả mong đợi**: CUDA = True, GPU = Tesla T4, VRAM ≥ 15 GB  
**Nếu không có GPU**: dừng ngay, không thực hiện các bước tiếp theo.

---

### Step 1 — Clone Repository

```python
import os

GITHUB_USER = 'nhiney'
REPO_NAME   = 'deep-social-sentiment-analysis'
REPO_DIR    = f'/content/{REPO_NAME}'
REPO_URL    = f'https://github.com/{GITHUB_USER}/{REPO_NAME}.git'

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
    print(f'✅ Cloned → {REPO_DIR}')
else:
    !cd {REPO_DIR} && git pull origin main
    print(f'✅ Pulled latest → {REPO_DIR}')

os.chdir(REPO_DIR)
print('Working directory:', os.getcwd())
!ls -la
```

**Kết quả mong đợi**: thấy các thư mục `app/`, `src/`, `scripts/`, `configs/`, `data/`, `models/`

---

### Step 2 — Mount Google Drive + Tạo Symlink

**Cell 2a — Mount Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')
print('✅ Google Drive mounted')
```

> Popup xuất hiện → click **Connect to Google Drive** → chọn tài khoản → **Allow**

**Cell 2b — Tạo thư mục và symlink:**

```python
import os, shutil

DRIVE_ROOT    = '/content/drive/MyDrive/colab_sentiment'
DRIVE_DATA    = f'{DRIVE_ROOT}/data'
DRIVE_MODELS  = f'{DRIVE_ROOT}/models'
DRIVE_REPORTS = f'{DRIVE_ROOT}/reports'
HF_CACHE      = f'{DRIVE_ROOT}/hf_cache'

# Tạo thư mục nếu chưa có
for d in [DRIVE_DATA, f'{DRIVE_DATA}/raw', f'{DRIVE_DATA}/processed',
          DRIVE_MODELS, DRIVE_REPORTS, HF_CACHE]:
    os.makedirs(d, exist_ok=True)
    print(f'  ✅ {d}')

# Tạo symlink: data/, models/, reports/ trong repo → Drive
def _symlink(src, dst):
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.isdir(dst) and not os.path.islink(dst):
        shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f'  linked: {dst} → {src}')

os.chdir(REPO_DIR)
_symlink(DRIVE_DATA,    'data')
_symlink(DRIVE_MODELS,  'models')
_symlink(DRIVE_REPORTS, 'reports')

# HuggingFace cache → Drive (tránh tải lại model mỗi session)
os.environ['HF_HOME']             = HF_CACHE
os.environ['TRANSFORMERS_CACHE']  = HF_CACHE

print('\n✅ Symlinks và cache đã được thiết lập')
print('Data files trong Drive:')
!ls data/raw/
```

**Kết quả mong đợi**:
```
linked: data    → /content/drive/MyDrive/colab_sentiment/data
linked: models  → /content/drive/MyDrive/colab_sentiment/models
linked: reports → /content/drive/MyDrive/colab_sentiment/reports
Data files trong Drive:
crawled_emotions.xlsx   unlabeled_new_posts.json   UIT-VSMEC.csv
```

---

### Step 3 — Kiểm Tra File Dữ Liệu

```python
required_files = [
    'data/raw/crawled_emotions.xlsx',
    'data/raw/unlabeled_new_posts.json',
    'data/raw/UIT-VSMEC.csv',
]
optional_files = [
    'data/processed/cleaned_unlabeled_posts.csv',
    'data/processed/pseudo_labeled_apify.csv',
]

print('=== Required files ===')
all_ok = True
for f in required_files:
    exists = os.path.exists(f)
    print(f'  {"✅" if exists else "❌ MISSING"} {f}')
    if not exists:
        all_ok = False

print('\n=== Optional files (sẽ tự tạo) ===')
for f in optional_files:
    print(f'  {"✅" if os.path.exists(f) else "⬜ chưa có"} {f}')

if not all_ok:
    raise FileNotFoundError('Một số file bắt buộc bị thiếu. Kiểm tra lại Drive.')
else:
    print('\n✅ Tất cả file bắt buộc đã có — tiếp tục.')
```

> Nếu có file ❌: dừng lại, kiểm tra tên file trên Drive có đúng chính xác không (phân biệt chữ hoa/thường).

---

### Step 4 — Cài Đặt Thư Viện

```python
print('📦 Installing dependencies...')
!pip install -q -r requirements.txt 2>&1 | tail -10

# Verify các import quan trọng
import torch, transformers, pandas, numpy, sklearn
print('\n✅ Versions:')
print('  torch        :', torch.__version__)
print('  transformers :', transformers.__version__)
print('  pandas       :', pandas.__version__)
print('  sklearn      :', sklearn.__version__)
print('  CUDA         :', torch.cuda.is_available())
```

**Thời gian**: ~3–5 phút lần đầu.  
**Kết quả mong đợi**: tất cả import không có lỗi.

---

### Step 5 — Pseudo-Label 990 Facebook Posts

> Dùng mDeBERTa zero-shot NLI để tự động gán nhãn cảm xúc cho unlabeled Apify posts.  
> **Bỏ qua step này** nếu `data/processed/pseudo_labeled_apify.csv` đã tồn tại trên Drive.

```python
import pandas as pd

PSEUDO_OUT = 'data/processed/pseudo_labeled_apify.csv'

if os.path.exists(PSEUDO_OUT):
    print(f'✅ File đã tồn tại — bỏ qua. Xóa file trên Drive nếu muốn chạy lại.')
    df = pd.read_csv(PSEUDO_OUT)
    print(f'   Rows: {len(df)} | Confident: {df["pseudo_confident"].sum()}')
else:
    print('🔄 Chạy pseudo-labeling (~15–20 phút)...')
    !python -m scripts.pseudo_label_apify \
        --input      data/processed/cleaned_unlabeled_posts.csv \
        --output     {PSEUDO_OUT} \
        --model      MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
        --batch-size 32 \
        --threshold  0.35 \
        --device     cuda

# Kiểm tra kết quả
if os.path.exists(PSEUDO_OUT):
    df = pd.read_csv(PSEUDO_OUT)
    print(f'\n✅ Pseudo-labeled: {len(df)} rows')
    print(f'   Confident (≥0.35): {df["pseudo_confident"].sum()}')
    print('\n   Label distribution:')
    print(df['label'].value_counts().to_string())
```

**Lần đầu**: tải model ~560 MB → ~5 phút. Lần sau load từ Drive cache → nhanh hơn.  
**Kết quả mong đợi**: 990 rows, confident ≥ 600 rows.

---

### Step 6 — Chuẩn Bị Dataset (Merge 3 Nguồn)

```python
import os

# Build command tuỳ theo file có sẵn
cmd = ('python -m scripts.prepare_data'
       ' --crawled    data/raw/crawled_emotions.xlsx'
       ' --output-dir data/processed'
       ' --seed       42')

if os.path.exists('data/raw/UIT-VSMEC.csv'):
    cmd += ' --uit-vsmec data/raw/UIT-VSMEC.csv'
    print('✅ UIT-VSMEC sẽ được merge (~6,927 rows)')

if os.path.exists('data/processed/pseudo_labeled_apify.csv'):
    cmd += (' --pseudo-labeled data/processed/pseudo_labeled_apify.csv'
            ' --confidence-threshold 0.35')
    print('✅ Pseudo-labeled Apify sẽ được merge')

print(f'\nRunning: {cmd}\n')
!{cmd}
```

**Kiểm tra kết quả:**

```python
import pandas as pd

print('=== Dataset splits ===')
for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/processed/{split}.parquet')
    print(f'  {split:5s}: {len(df):>6,} rows | {len(df.columns)} cols')
    print(f'         {df["label"].value_counts().to_dict()}')

# Verify tabular columns
train = pd.read_parquet('data/processed/train.parquet')
required_cols = ['text_length', 'n_words', 'n_exclamation', 'n_question',
                 'n_emoji_token', 'n_hashtag', 'n_latin_words',
                 'likes', 'comments', 'shares', 'has_emoji', 'is_crawled']
missing = [c for c in required_cols if c not in train.columns]
if missing:
    raise ValueError(f'Thiếu tabular columns: {missing}')
else:
    print(f'\n✅ Tất cả {len(required_cols)} tabular columns có đủ')
```

**Kết quả mong đợi**: train ≥ 5,000 rows, tổng 3 splits ≥ 7,500 rows, 20 columns.  
**Dừng nếu**: train < 1,000 rows hoặc thiếu tabular columns.

---

### Step 7 — Cấu Hình Training

```python
import yaml

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Override để lưu checkpoint vào Drive
cfg['training']['output_dir'] = f'{DRIVE_MODELS}/best_model'
cfg['training']['device']     = 'cuda'

COLAB_CFG = '/tmp/config_colab.yaml'
with open(COLAB_CFG, 'w') as f:
    yaml.dump(cfg, f)

print('=== Training Config ===')
print(f"  output_dir   : {cfg['training']['output_dir']}")
print(f"  device       : {cfg['training']['device']}")
print(f"  epochs       : {cfg['training'].get('epochs', 10)}")
print(f"  batch_size   : {cfg['training'].get('batch_size', 32)}")
print(f"  learning_rate: {cfg['training'].get('learning_rate', 2e-5)}")
print(f"\n✅ Config saved → {COLAB_CFG}")
```

---

### Step 8 — Training Model

```python
print('🚀 Bắt đầu training — ước tính ~2–3 giờ trên T4...')
print('   Checkpoint sẽ tự động lưu vào Drive sau mỗi epoch tốt nhất.\n')

!python -m src.train --config {COLAB_CFG}
```

**Trong khi chạy**:
- Không đóng tab Colab
- Nếu OOM: dừng, giảm `batch_size` trong config xuống 16 hoặc 8, chạy lại Step 7 + 8

**Sau khi xong, kiểm tra checkpoint:**

```python
import os

ckpt_dir = f'{DRIVE_MODELS}/best_model'
if os.path.exists(ckpt_dir):
    print(f'✅ Checkpoint saved → {ckpt_dir}')
    for f in sorted(os.listdir(ckpt_dir)):
        size_mb = os.path.getsize(f'{ckpt_dir}/{f}') / 1e6
        print(f'   {f:<40} {size_mb:>8.1f} MB')
else:
    raise FileNotFoundError('Checkpoint không tìm thấy — kiểm tra log training ở trên.')
```

**Kết quả mong đợi**: thấy `pytorch_model.bin` (~500 MB) hoặc `model.safetensors`.

---

### Step 9 — Evaluate Trên Test Set

```python
CHECKPOINT = f'{DRIVE_MODELS}/best_model'

print('📊 Evaluating on test set...')
!python -m src.evaluate \
    --checkpoint {CHECKPOINT} \
    --data       data/processed/test.parquet \
    --output-dir reports/

print('\n✅ Evaluation complete. Reports saved to reports/')
!ls reports/
```

In ra metrics cuối cùng:

```python
import json, os

metrics_path = 'reports/metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    print('\n=== Final Metrics ===')
    for k, v in metrics.items():
        print(f'  {k:<20}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
```

---

### Step 10 — Ablation Study

> Chạy sau khi Step 8 hoàn thành. Ước tính ~3× thời gian training = 6–9 giờ.  
> Có thể chạy trên session riêng (checkpoint đã lưu trên Drive).

```python
ABLATION_DIR = f'{DRIVE_MODELS}/ablation'
os.makedirs(ABLATION_DIR, exist_ok=True)

print('🔬 Running ablation study (3 experiments)...')
print('   Experiment 1: text-only (no tabular, no normalizer)')
print('   Experiment 2: text + normalizer (no tabular)')
print('   Experiment 3: full model (text + tabular + normalizer)\n')

!python -m scripts.run_ablation \
    --raw        data/raw/crawled_emotions.xlsx \
    --output-dir {ABLATION_DIR} \
    --epochs     4 \
    --batch-size 32 \
    --device     cuda
```

**Xem kết quả:**

```python
import pandas as pd

results_path = 'reports/ablation_results.csv'
if os.path.exists(results_path):
    df = pd.read_csv(results_path, index_col=0)
    cols = ['use_normalizer', 'use_tabular', 'f1_macro',
            'precision_macro', 'recall_macro', 'accuracy']
    print('\n=== Ablation Results ===')
    print(df[cols].to_string(float_format=lambda v: f'{v:.4f}'))

    # Copy sang Drive
    import shutil
    shutil.copy(results_path, f'{DRIVE_REPORTS}/ablation_results.csv')
    print(f'\n✅ Results copied → Drive')
```

---

## Tóm Tắt Cuối Cùng

Sau khi tất cả bước hoàn thành, in bảng tóm tắt:

```python
steps = [
    ('Step 0', 'GPU check',         'T4 GPU, 15.8 GB VRAM'),
    ('Step 1', 'Clone repo',        '/content/deep-social-sentiment-analysis'),
    ('Step 2', 'Mount Drive',       'Symlinks: data/, models/, reports/'),
    ('Step 3', 'Verify data',       '3 required files ✅'),
    ('Step 4', 'Install deps',      'requirements.txt'),
    ('Step 5', 'Pseudo-label',      '990 posts → data/processed/pseudo_labeled_apify.csv'),
    ('Step 6', 'Prepare dataset',   'train/val/test parquet, 20 columns'),
    ('Step 7', 'Config training',   'output → Drive/models/best_model'),
    ('Step 8', 'Training',          'Late Fusion XLM-R + FT-Transformer'),
    ('Step 9', 'Evaluate',          'F1-macro, per-class metrics'),
    ('Step 10','Ablation',          '3 experiments, reports/ablation_results.csv'),
]

print(f'\n{"="*65}')
print(f'{"Step":<10} {"Task":<22} {"Result":<33}')
print(f'{"="*65}')
for step, task, result in steps:
    print(f'{step:<10} {task:<22} {result:<33}')
print(f'{"="*65}')
print('\n✅ Pipeline hoàn thành.')
```

---

## Ghi Chú Quan Trọng

| Tình huống | Giải pháp |
|------------|-----------|
| Session reset sau 12h | Chạy lại Step 1 → 2 → 4, tiếp tục từ step đang dở |
| Muốn tiếp tục training từ checkpoint | Thêm `--resume {DRIVE_MODELS}/best_model` vào lệnh train |
| Muốn chạy lại pseudo-labeling | Xóa `data/processed/pseudo_labeled_apify.csv` khỏi Drive |
| Muốn chạy lại prepare_data | Xóa `data/processed/train.parquet` khỏi Drive |
| Colab Pro hết credit | Checkpoint đã trên Drive — tiếp tục trên session mới |
