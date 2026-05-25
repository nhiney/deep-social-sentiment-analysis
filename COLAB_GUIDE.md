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

## Cấu Trúc Dữ Liệu Trên Drive

```
MyDrive/colab_sentiment/
├── data/
│   ├── raw/
│   │   ├── crawled_emotions.xlsx        ← 2034 labeled posts
│   │   ├── unlabeled_new_posts.json     ← 999 Apify posts
│   │   └── UIT-VSMEC.csv               ← 6927 Facebook comments
│   └── processed/
│       └── cleaned_unlabeled_posts.csv  ← 990 posts cleaned
├── models/                              ← checkpoint lưu sau training
├── reports/                             ← figures, metrics, ablation
└── hf_cache/                            ← HuggingFace model cache
```

---

## Quy Tắc Bắt Buộc

| Quy tắc | Chi tiết |
|---------|---------|
| **Tuần tự** | Chạy từng step một, chờ xong mới sang bước tiếp |
| **Báo cáo** | Sau mỗi step: in ✅/❌ + số liệu chính + cảnh báo nếu có |
| **Dừng khi lỗi** | Không tự ý bỏ qua lỗi — in traceback đầy đủ và báo cáo |
| **Không đóng tab** | Trong khi training (Step 8) — trang sẽ bị ngắt nếu đóng |

---

## Xử Lý Lỗi

| Lỗi | Xử lý |
|-----|-------|
| `credential propagation was unsuccessful` | Chạy cell auth thủ công (xem Step 2b) |
| `CUDA out of memory` | Đổi `batch_size: 32 → 16` trong config, chạy lại Step 7+8 |
| `ModuleNotFoundError` | Chạy lại Step 4 |
| `FileNotFoundError` data | Kiểm tra symlink Step 2c, kiểm tra file đã upload Drive |
| Drive bị ngắt kết nối | Chạy lại Step 2b+2c, tiếp tục từ bước đang dở |
| Bất kỳ lỗi nào khác | In toàn bộ traceback — dừng lại — không tự ý bỏ qua |

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
    DEVICE = 'cuda'
else:
    raise RuntimeError(
        '❌ GPU không khả dụng.\n'
        'Vào Runtime → Change runtime type → T4 GPU → Save → chạy lại.'
    )

!nvidia-smi | head -15
```

**Kết quả mong đợi**: CUDA = True, GPU = Tesla T4, VRAM ≥ 15 GB

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
!ls
```

**Kết quả mong đợi**: thấy `app/`, `src/`, `scripts/`, `configs/`, `data/`, `models/`

---

### Step 2a — Mount Google Drive (thử cách thông thường trước)

```python
from google.colab import drive
drive.mount('/content/drive')
print('✅ Google Drive mounted')
```

> Popup xuất hiện → click **Connect to Google Drive** → chọn tài khoản → **Allow**

**Nếu lỗi `credential propagation was unsuccessful`** → bỏ qua cell này, chạy **Step 2b**.

---

### Step 2b — Mount Drive (fallback nếu Step 2a lỗi)

```python
# Xác thực thủ công khi credential tự động bị lỗi
from google.colab import auth
auth.authenticate_user()
print('✅ Auth done')

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print('✅ Drive mounted (force)')
```

> Sẽ có link → click → đăng nhập Google → copy code → dán vào ô → Enter

---

### Step 2c — Tạo Symlink Drive ↔ Repo

```python
import os, shutil

DRIVE_ROOT    = '/content/drive/MyDrive/colab_sentiment'
DRIVE_DATA    = f'{DRIVE_ROOT}/data'
DRIVE_MODELS  = f'{DRIVE_ROOT}/models'
DRIVE_REPORTS = f'{DRIVE_ROOT}/reports'
HF_CACHE      = f'{DRIVE_ROOT}/hf_cache'

# Tạo thư mục nếu chưa có
for d in [f'{DRIVE_DATA}/raw', f'{DRIVE_DATA}/processed',
          DRIVE_MODELS, DRIVE_REPORTS, HF_CACHE]:
    os.makedirs(d, exist_ok=True)

# Tạo symlink: data/, models/, reports/ → Drive
def _symlink(src, dst):
    if os.path.islink(dst):   os.remove(dst)
    elif os.path.isdir(dst):  shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f'  linked: {dst} → {src}')

os.chdir(REPO_DIR)
_symlink(DRIVE_DATA,    'data')
_symlink(DRIVE_MODELS,  'models')
_symlink(DRIVE_REPORTS, 'reports')

# HuggingFace cache → Drive (tránh tải lại mỗi session)
os.environ['HF_HOME']            = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE

print('\n✅ Symlinks OK. Files trong data/raw/:')
!ls data/raw/
```

**Kết quả mong đợi**:
```
crawled_emotions.xlsx   UIT-VSMEC.csv   unlabeled_new_posts.json
```

---

### Step 3 — Kiểm Tra File Dữ Liệu

```python
import os

required = [
    'data/raw/crawled_emotions.xlsx',
    'data/raw/UIT-VSMEC.csv',
    'data/raw/unlabeled_new_posts.json',
    'data/processed/cleaned_unlabeled_posts.csv',
]

print('=== Kiểm tra file ===')
all_ok = True
for f in required:
    ok = os.path.exists(f)
    print(f'  {"✅" if ok else "❌ THIẾU"} {f}')
    if not ok:
        all_ok = False

if not all_ok:
    raise FileNotFoundError(
        '\n❌ Còn file bị thiếu.\n'
        f'Upload lên Drive tại: {DRIVE_DATA}/raw/\n'
        'Sau đó chạy lại cell này.'
    )
print('\n✅ Tất cả file có đủ — tiếp tục.')
```

---

### Step 4 — Cài Đặt Thư Viện

```python
print('📦 Installing dependencies (~3-5 phút)...')
!pip install -q -r requirements.txt 2>&1 | tail -5

# Verify import
import torch, transformers, pandas, numpy, sklearn
try:
    import rtdl_revisiting_models
    print('✅ rtdl_revisiting_models OK')
except ImportError:
    !pip install -q rtdl-revisiting-models
    import rtdl_revisiting_models

print('\n✅ Versions:')
print('  torch        :', torch.__version__)
print('  transformers :', transformers.__version__)
print('  CUDA         :', torch.cuda.is_available())
```

---

### Step 5 — Pseudo-Label 990 Facebook Posts

> Bỏ qua nếu `data/processed/pseudo_labeled_apify.csv` đã tồn tại.

```python
import os, pandas as pd

PSEUDO_OUT = 'data/processed/pseudo_labeled_apify.csv'

if os.path.exists(PSEUDO_OUT):
    df = pd.read_csv(PSEUDO_OUT)
    print(f'✅ Đã có pseudo_labeled_apify.csv — {len(df)} rows, bỏ qua.')
else:
    print('🔄 Chạy pseudo-labeling (~15 phút)...')
    !python -m scripts.pseudo_label_apify \
        --input      data/processed/cleaned_unlabeled_posts.csv \
        --output     {PSEUDO_OUT} \
        --model      MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
        --batch-size 32 \
        --threshold  0.35 \
        --device     cuda

    df = pd.read_csv(PSEUDO_OUT)
    confident = df['pseudo_confident'].sum() if 'pseudo_confident' in df.columns else 'N/A'
    print(f'\n✅ Pseudo-labeled: {len(df)} rows | Confident: {confident}')
    print(df['label'].value_counts().to_string())
```

**Kết quả mong đợi**: 990 rows, confident ≥ 600

---

### Step 6 — Chuẩn Bị Dataset (Merge 3 Nguồn)

```python
import os

cmd = ('python -m scripts.prepare_data'
       ' --crawled    data/raw/crawled_emotions.xlsx'
       ' --output-dir data/processed'
       ' --seed       42')

if os.path.exists('data/raw/UIT-VSMEC.csv'):
    cmd += ' --uit-vsmec data/raw/UIT-VSMEC.csv'
    print('✅ UIT-VSMEC sẽ được merge')

if os.path.exists('data/processed/pseudo_labeled_apify.csv'):
    cmd += (' --pseudo-labeled data/processed/pseudo_labeled_apify.csv'
            ' --confidence-threshold 0.35')
    print('✅ Pseudo-labeled Apify sẽ được merge')

print(f'\n▶ {cmd}\n')
!{cmd}
```

**Kiểm tra kết quả:**

```python
import pandas as pd

for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/processed/{split}.parquet')
    print(f'{split:5s}: {len(df):>6,} rows | {len(df.columns)} cols')

train = pd.read_parquet('data/processed/train.parquet')
required_cols = ['text_length', 'n_words', 'likes', 'comments',
                 'shares', 'has_emoji', 'is_crawled']
missing = [c for c in required_cols if c not in train.columns]
if missing:
    raise ValueError(f'❌ Thiếu tabular columns: {missing}')
print(f'\n✅ Tabular columns đủ | train columns: {list(train.columns)}')
```

**Kết quả mong đợi**: train ≥ 5,000 rows, 20 columns

---

### Step 7 — Cấu Hình Training

```python
import yaml

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Override output → Drive
cfg['training']['output_dir'] = f'{DRIVE_MODELS}/best_model'
cfg['training']['device']     = 'cuda'

COLAB_CFG = '/tmp/config_colab.yaml'
with open(COLAB_CFG, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('=== Training Config ===')
print(f"  output_dir    : {cfg['training']['output_dir']}")
print(f"  epochs        : {cfg['training'].get('epochs', 10)}")
print(f"  batch_size    : {cfg['training'].get('batch_size', 32)}")
print(f"  learning_rate : {cfg['training'].get('learning_rate', 2e-5)}")
print(f"  device        : {cfg['training']['device']}")
print(f'\n✅ Config saved → {COLAB_CFG}')
```

> **Nếu bị OOM sau khi training bắt đầu**: quay lại cell này, đổi
> `cfg['training']['batch_size'] = 16` rồi chạy lại Step 7 + Step 8.

---

### Step 8 — Training Model

```python
import time
start = time.time()
print('🚀 Bắt đầu training...')
print('   Checkpoint → Drive sau mỗi epoch tốt nhất')
print('   Ước tính: ~1.5–2.5 giờ trên T4\n')

!python -m src.train --config {COLAB_CFG}

elapsed = (time.time() - start) / 60
print(f'\n⏱ Training time: {elapsed:.1f} phút')
```

**Kiểm tra checkpoint:**

```python
import os

ckpt = f'{DRIVE_MODELS}/best_model'
if os.path.exists(ckpt):
    print(f'✅ Checkpoint saved → {ckpt}')
    for f in sorted(os.listdir(ckpt)):
        size = os.path.getsize(f'{ckpt}/{f}') / 1e6
        print(f'   {f:<45} {size:>8.1f} MB')
else:
    raise FileNotFoundError('❌ Checkpoint không tìm thấy — kiểm tra log training.')
```

---

### Step 9 — Evaluate Trên Test Set

```python
CHECKPOINT = f'{DRIVE_MODELS}/best_model'

print('📊 Evaluating on test set...')
!python -m src.evaluate \
    --checkpoint {CHECKPOINT} \
    --data       data/processed/test.parquet \
    --output-dir reports/

print('\n✅ Evaluation complete')
!ls reports/
```

**In kết quả:**

```python
import json, os

for metrics_path in ['reports/metrics.json', f'{DRIVE_REPORTS}/metrics.json']:
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        print('\n=== Final Metrics ===')
        for k, v in m.items():
            print(f'  {k:<25}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
        break
```

---

### Step 10 — Ablation Study

> Chứng minh đóng góp từng thành phần — quan trọng nhất cho báo cáo.  
> Thời gian ~4–8 giờ. Có thể chạy trên **session riêng** (checkpoint đã lưu trên Drive).

```python
ABLATION_DIR = f'{DRIVE_MODELS}/ablation'
os.makedirs(ABLATION_DIR, exist_ok=True)

print('🔬 Running ablation (3 experiments)...')
print('   Exp1: XLM-R only (no Teencode, no tabular)')
print('   Exp2: XLM-R + Teencode (no tabular)')
print('   Exp3: Full model (XLM-R + Teencode + FT-Transformer)\n')

!python -m scripts.run_ablation \
    --raw        data/raw/crawled_emotions.xlsx \
    --uit-vsmec  data/raw/UIT-VSMEC.csv \
    --output-dir {ABLATION_DIR} \
    --epochs     5 \
    --batch-size 32 \
    --device     cuda
```

**Xem kết quả:**

```python
import pandas as pd, shutil

results = 'reports/ablation_results.csv'
if os.path.exists(results):
    df = pd.read_csv(results, index_col=0)
    print('\n===== ABLATION RESULTS =====')
    cols = [c for c in ['use_normalizer','use_tabular','f1_macro',
                        'precision_macro','recall_macro','accuracy'] if c in df.columns]
    print(df[cols].to_string(float_format=lambda v: f'{v:.4f}'))

    shutil.copy(results, f'{DRIVE_REPORTS}/ablation_results.csv')
    print(f'\n✅ Copied → Drive')
```

---

### Step 11 — Copy Tất Cả Kết Quả Về Drive

```python
import shutil, glob, os

print('💾 Copying results to Drive...')
for f in glob.glob('reports/**/*', recursive=True):
    if os.path.isfile(f):
        dest = f'{DRIVE_ROOT}/{f}'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)

print('✅ Reports copied. Drive contents:')
!ls {DRIVE_REPORTS}/
```

---

## Tóm Tắt Kiểm Tra Cuối

```python
import os, json

checks = {
    'Checkpoint (best_model)':   os.path.exists(f'{DRIVE_MODELS}/best_model'),
    'ablation_results.csv':      os.path.exists(f'{DRIVE_REPORTS}/ablation_results.csv'),
    'metrics.json':              os.path.exists(f'{DRIVE_REPORTS}/metrics.json'),
    'train.parquet':             os.path.exists('data/processed/train.parquet'),
    'test.parquet':              os.path.exists('data/processed/test.parquet'),
}

print(f'\n{"="*50}')
print(f'{"Artifact":<35} {"Status"}')
print(f'{"="*50}')
for name, ok in checks.items():
    print(f'  {name:<33} {"✅" if ok else "❌"}')
print(f'{"="*50}')

all_done = all(checks.values())
print(f'\n{"✅ Pipeline hoàn thành!" if all_done else "⚠️  Còn artifact chưa có — kiểm tra lại"}')
```

---

## Ghi Chú Quan Trọng

| Tình huống | Giải pháp |
|------------|-----------|
| Session reset sau 12h | Chạy lại Step 1 → 2a/2b → 2c → 4, tiếp tục từ step đang dở |
| Muốn tiếp tục training | Thêm `--resume {DRIVE_MODELS}/best_model` vào lệnh train |
| Chạy lại pseudo-labeling | Xóa `data/processed/pseudo_labeled_apify.csv` khỏi Drive |
| Chạy lại prepare_data | Xóa `data/processed/train.parquet` khỏi Drive |
| Colab Pro hết credit | Checkpoint đã trên Drive — tiếp tục trên session mới |
| OOM khi training | `batch_size: 32 → 16 → 8` trong Step 7 |
