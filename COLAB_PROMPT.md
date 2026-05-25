# PROMPT — Dán cho AI để chạy pipeline trên Colab

> Copy toàn bộ nội dung bên dưới dòng `---` và dán vào chat với AI (Claude, GPT...) khi bạn đang ở trong Google Colab.

---

Bạn là AI assistant đang chạy trong Google Colab. Nhiệm vụ của bạn là thực hiện toàn bộ pipeline training cho project **Deep Social Sentiment Analysis** — phân loại cảm xúc tiếng Việt 7 lớp (Ekman model).

## Thông tin project

- **GitHub**: `https://github.com/nhiney/deep-social-sentiment-analysis`
- **Kiến trúc**: Late Fusion — XLM-RoBERTa-base (text branch) + FT-Transformer (tabular branch) → MLP head → 7 classes
- **7 nhãn**: joy, sadness, anger, fear, disgust, surprise, neutral
- **Google Drive root**: `MyDrive/colab_sentiment/`
- **Repo dir**: `/content/deep-social-sentiment-analysis`

## Dữ liệu đã có trên Drive (MyDrive/colab_sentiment/data/raw/)

- `crawled_emotions.xlsx` — 2034 labeled posts
- `UIT-VSMEC.csv` — 6927 labeled Facebook comments
- `unlabeled_new_posts.json` — 999 Facebook posts chưa có nhãn
- `data/processed/cleaned_unlabeled_posts.csv` — 990 posts đã clean

## Quy tắc bắt buộc

1. **Chạy tuần tự từng bước** — không bỏ qua, không chạy song song
2. **Sau mỗi bước**: báo cáo ✅/❌ + số liệu chính (số rows, loss, F1...)
3. **Khi gặp lỗi**: in traceback đầy đủ, dừng lại, hỏi người dùng — không tự ý bỏ qua
4. **Không đóng tab** trong lúc training

## Xử lý lỗi thường gặp

- `credential propagation was unsuccessful` → chạy cell auth thủ công (xem Step 2b trong hướng dẫn)
- `CUDA out of memory` → đổi batch_size xuống 16 trong Step 7 rồi chạy lại
- `ModuleNotFoundError` → chạy lại pip install
- `FileNotFoundError` → kiểm tra symlink Drive, kiểm tra file đã upload chưa

---

## Bắt đầu thực hiện theo thứ tự sau:

### BƯỚC 0 — Kiểm tra GPU

Chạy cell này và báo cáo kết quả:

```python
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
    DEVICE = 'cuda'
else:
    raise RuntimeError('Không có GPU. Vào Runtime → Change runtime type → T4 GPU')
!nvidia-smi | head -12
```

Kết quả mong đợi: CUDA=True, GPU=Tesla T4, VRAM≥15GB. Nếu không có GPU thì dừng lại và báo.

---

### BƯỚC 1 — Clone repo

```python
import os
REPO_DIR = '/content/deep-social-sentiment-analysis'
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/nhiney/deep-social-sentiment-analysis.git {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull origin main
os.chdir(REPO_DIR)
print('✅ Repo ready:', os.getcwd())
!ls
```

Kết quả mong đợi: thấy các thư mục `src/`, `scripts/`, `configs/`, `data/`, `models/`.

---

### BƯỚC 2 — Mount Google Drive

**Cell 2a — thử trước:**
```python
from google.colab import drive
drive.mount('/content/drive')
print('✅ Drive mounted')
```

**Nếu lỗi `credential propagation was unsuccessful` → chạy cell 2b thay thế:**
```python
from google.colab import auth
auth.authenticate_user()
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print('✅ Drive mounted (force auth)')
```

Báo cáo: mount thành công hay thất bại, lỗi gì nếu có.

---

### BƯỚC 2c — Tạo symlink và kiểm tra data

```python
import os, shutil

DRIVE_ROOT    = '/content/drive/MyDrive/colab_sentiment'
DRIVE_DATA    = f'{DRIVE_ROOT}/data'
DRIVE_MODELS  = f'{DRIVE_ROOT}/models'
DRIVE_REPORTS = f'{DRIVE_ROOT}/reports'
HF_CACHE      = f'{DRIVE_ROOT}/hf_cache'

for d in [f'{DRIVE_DATA}/raw', f'{DRIVE_DATA}/processed',
          DRIVE_MODELS, DRIVE_REPORTS, HF_CACHE]:
    os.makedirs(d, exist_ok=True)

def _symlink(src, dst):
    if os.path.islink(dst): os.remove(dst)
    elif os.path.isdir(dst): shutil.rmtree(dst)
    os.symlink(src, dst)
    print(f'  linked: {dst} → {src}')

os.chdir('/content/deep-social-sentiment-analysis')
_symlink(DRIVE_DATA,    'data')
_symlink(DRIVE_MODELS,  'models')
_symlink(DRIVE_REPORTS, 'reports')

os.environ['HF_HOME']            = HF_CACHE
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE

print('\nFiles trong data/raw/:')
!ls data/raw/
```

Kiểm tra file:
```python
required = [
    'data/raw/crawled_emotions.xlsx',
    'data/raw/UIT-VSMEC.csv',
    'data/raw/unlabeled_new_posts.json',
    'data/processed/cleaned_unlabeled_posts.csv',
]
all_ok = True
for f in required:
    ok = os.path.exists(f)
    print(f'  {"✅" if ok else "❌ THIẾU"} {f}')
    if not ok: all_ok = False
if not all_ok:
    raise FileNotFoundError('File bị thiếu — kiểm tra Drive')
print('\n✅ Tất cả file có đủ')
```

Báo cáo: file nào có, file nào thiếu.

---

### BƯỚC 3 — Cài thư viện

```python
!pip install -q -r requirements.txt 2>&1 | tail -5

import torch, transformers, pandas, sklearn
try:
    import rtdl_revisiting_models
except ImportError:
    !pip install -q rtdl-revisiting-models
    import rtdl_revisiting_models

print('✅ torch:', torch.__version__)
print('✅ transformers:', transformers.__version__)
print('✅ CUDA:', torch.cuda.is_available())
```

Báo cáo: tất cả import OK hay có module nào lỗi.

---

### BƯỚC 4 — Pseudo-label 990 posts

```python
import os, pandas as pd

PSEUDO_OUT = 'data/processed/pseudo_labeled_apify.csv'

if os.path.exists(PSEUDO_OUT):
    df = pd.read_csv(PSEUDO_OUT)
    print(f'✅ Đã có {len(df)} rows — bỏ qua bước này')
else:
    print('🔄 Pseudo-labeling (~15 phút)...')
    !python -m scripts.pseudo_label_apify \
        --input      data/processed/cleaned_unlabeled_posts.csv \
        --output     {PSEUDO_OUT} \
        --model      MoritzLaurer/mDeBERTa-v3-base-mnli-xnli \
        --batch-size 32 \
        --threshold  0.35 \
        --device     cuda
    df = pd.read_csv(PSEUDO_OUT)
    print(f'✅ {len(df)} rows | Confident: {df["pseudo_confident"].sum()}')
    print(df['label'].value_counts().to_string())
```

Kết quả mong đợi: 990 rows, confident ≥ 600. Báo cáo label distribution.

---

### BƯỚC 5 — Merge dataset

```python
import os

cmd = ('python -m scripts.prepare_data'
       ' --crawled    data/raw/crawled_emotions.xlsx'
       ' --output-dir data/processed --seed 42')

if os.path.exists('data/raw/UIT-VSMEC.csv'):
    cmd += ' --uit-vsmec data/raw/UIT-VSMEC.csv'
if os.path.exists('data/processed/pseudo_labeled_apify.csv'):
    cmd += ' --pseudo-labeled data/processed/pseudo_labeled_apify.csv --confidence-threshold 0.35'

print(f'▶ {cmd}')
!{cmd}
```

Kiểm tra:
```python
import pandas as pd
for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data/processed/{split}.parquet')
    print(f'{split}: {len(df):,} rows | {len(df.columns)} cols')

train = pd.read_parquet('data/processed/train.parquet')
must_have = ['text_length','n_words','likes','comments','shares','has_emoji','is_crawled']
missing = [c for c in must_have if c not in train.columns]
if missing: raise ValueError(f'Thiếu cols: {missing}')
print('✅ Tabular columns đủ')
```

Kết quả mong đợi: train ≥ 5000 rows, 20 cols. Báo cáo số rows từng split.

---

### BƯỚC 6 — Cấu hình training

```python
import yaml

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

cfg['training']['output_dir'] = f'{DRIVE_MODELS}/best_model'
cfg['training']['device']     = 'cuda'

COLAB_CFG = '/tmp/config_colab.yaml'
with open(COLAB_CFG, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('Config:')
for k in ['output_dir','device','epochs','batch_size','learning_rate']:
    print(f'  {k}: {cfg["training"].get(k, "?")}')
print(f'✅ Saved → {COLAB_CFG}')
```

Nếu OOM xảy ra ở bước sau: quay lại đây, thêm `cfg['training']['batch_size'] = 16` rồi chạy lại.

---

### BƯỚC 7 — Training (quan trọng nhất)

```python
import time
t0 = time.time()
print('🚀 Training bắt đầu...')
!python -m src.train --config {COLAB_CFG}
print(f'\n⏱ Thời gian: {(time.time()-t0)/60:.1f} phút')
```

Kiểm tra checkpoint:
```python
ckpt = f'{DRIVE_MODELS}/best_model'
if os.path.exists(ckpt):
    for f in sorted(os.listdir(ckpt)):
        print(f'  {f}: {os.path.getsize(f"{ckpt}/{f}")/1e6:.1f} MB')
    print('✅ Checkpoint saved to Drive')
else:
    raise FileNotFoundError('Checkpoint không tồn tại — xem log lỗi trên')
```

Kết quả mong đợi: thấy `pytorch_model.bin` hoặc `model.safetensors` (~500MB). Báo cáo best val F1.

---

### BƯỚC 8 — Evaluate test set

```python
!python -m src.evaluate \
    --checkpoint {DRIVE_MODELS}/best_model \
    --data       data/processed/test.parquet \
    --output-dir reports/

import json
for p in ['reports/metrics.json']:
    if os.path.exists(p):
        m = json.load(open(p))
        print('\n=== Test Metrics ===')
        for k, v in m.items():
            print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
```

Báo cáo: F1-Macro, Accuracy, và per-class F1.

---

### BƯỚC 9 — Ablation study

```python
!python -m scripts.run_ablation \
    --raw        data/raw/crawled_emotions.xlsx \
    --uit-vsmec  data/raw/UIT-VSMEC.csv \
    --output-dir {DRIVE_MODELS}/ablation \
    --epochs     5 \
    --batch-size 32 \
    --device     cuda

import pandas as pd
if os.path.exists('reports/ablation_results.csv'):
    df = pd.read_csv('reports/ablation_results.csv', index_col=0)
    print(df.to_string(float_format=lambda v: f'{v:.4f}'))
```

Kết quả mong đợi: bảng 3 rows (Exp1/2/3) với F1-Macro tăng dần.

---

### BƯỚC 10 — Copy tất cả về Drive

```python
import shutil, glob

for f in glob.glob('reports/**/*', recursive=True):
    if os.path.isfile(f):
        dest = f'{DRIVE_ROOT}/{f}'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
print('✅ Reports copied to Drive')

# Kiểm tra cuối
checks = {
    'Checkpoint':            os.path.exists(f'{DRIVE_MODELS}/best_model'),
    'ablation_results.csv':  os.path.exists(f'{DRIVE_REPORTS}/ablation_results.csv'),
    'metrics.json':          os.path.exists(f'{DRIVE_REPORTS}/metrics.json'),
}
for name, ok in checks.items():
    print(f'  {"✅" if ok else "❌"} {name}')
```

---

## Kết thúc

Sau khi tất cả bước xong, báo cáo tóm tắt:
- Tổng thời gian
- Best val F1-Macro
- Test F1-Macro  
- Ablation: F1 của Exp1 → Exp2 → Exp3
- Đường dẫn Drive chứa checkpoint
