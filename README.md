# Deep Social Sentiment Analysis

Phân loại cảm xúc tiếng Việt 7 lớp trên mạng xã hội — kiến trúc **Late Fusion** kết hợp:

- **Text branch:** `XLM-RoBERTa-base` + Teencode Normalization (chuẩn hoá tiếng Việt mạng).
- **Tabular branch:** `FT-Transformer` mã hoá hành vi người dùng (likes, comments, shares, text-surface features).
- **Fusion head:** MLP phân loại 7 cảm xúc (Ekman + Neutral).
- **Explainable AI:** LIME tích hợp trong demo app — token tô màu theo mức đóng góp.

## Kết quả

### Test set (9616 mẫu, split 15%)

| Metric | Giá trị |
|---|---|
| **F1-Macro** | **0.6877** |
| Accuracy | 0.7020 |
| F1-Weighted | 0.7029 |
| Precision-Macro | 0.6976 |
| Recall-Macro | 0.6874 |

### Per-class F1

| Emotion | F1 | Support |
|---|---|---|
| joy | 0.7893 | 378 |
| sadness | 0.7431 | 205 |
| anger | 0.5413 | 108 |
| fear | 0.7205 | 90 |
| disgust | 0.5871 | 229 |
| surprise | 0.8243 | 197 |
| neutral | 0.6081 | 236 |

### Ablation study (8961 mẫu, 5 epochs/experiment)

| Experiment | Teencode | Tabular | F1-Macro | Accuracy |
|---|---|---|---|---|
| XLM-R only | ❌ | ❌ | 0.6235 | 0.6424 |
| + Teencode normalization | ✅ | ❌ | **0.6548** | **0.6647** |
| + Tabular fusion | ✅ | ✅ | 0.6454 | 0.6587 |

**Kết luận:** Teencode normalization đóng góp +3.1 F1-Macro so với raw text. Tabular branch với text-derived proxy features không bù thêm trong setting này — real engagement data (likes/comments/shares) cần nhiều mẫu có nhãn thật để thể hiện hiệu quả.

## Dataset

| Nguồn | Mẫu | Mô tả |
|---|---|---|
| `crawled_emotions.xlsx` | 2034 | Bài đăng Facebook tự crawl — 7 cảm xúc |
| `UIT-VSMEC.csv` | 6927 | Facebook comments công khai ([UIT-NLP](https://github.com/uitnlp/UIT-VSMEC)) |
| `pseudo_labeled_apify.csv` | ~655 | 990 posts Apify → zero-shot NLI (mDeBERTa) với confidence ≥ 0.35 |
| **Tổng train set** | **~6700** | Sau stratified split 70/15/15 |

## Cấu trúc dự án

```
deep-social-sentiment-analysis/
├── app/                  # Streamlit demo + FastAPI service
├── configs/              # YAML hyperparameter configs
├── data/
│   ├── raw/              # Dữ liệu gốc (không commit lên git)
│   ├── processed/        # Parquet splits (không commit)
│   └── external/         # teencode.json (commit)
├── models/               # Checkpoint (không commit — lưu Drive)
├── notebooks/
│   ├── 01_eda.ipynb      # EDA — 10 sections, word clouds, Kruskal-Wallis
│   └── 02_model_analysis.ipynb  # Learning curves, confusion matrix, ablation, LIME
├── reports/
│   ├── figures/          # 6 EDA figures
│   ├── metrics.json      # Test set metrics
│   ├── ablation_results.csv
│   └── ablation_results.md
├── scripts/
│   ├── prepare_data.py   # Multi-source merge → parquet
│   ├── pseudo_label_apify.py  # Zero-shot NLI labeling
│   ├── download_uit_vsmec.py  # Auto-download UIT-VSMEC
│   ├── run_ablation.py   # 3-experiment ablation
│   └── eda_interactions.py    # EDA + Kruskal-Wallis
├── src/
│   ├── preprocessing.py  # TeencodeNormalizer, TabularPreprocessor
│   ├── dataset.py        # SocialSentimentDataset (PyTorch)
│   ├── models.py         # TfidfBaseline, DnnBaseline, LateFusionModel
│   ├── train.py          # Training loop (AMP, early stopping, AdamW)
│   └── evaluate.py       # F1-Macro, per-class, confusion matrix
└── tests/                # 116 pytest unit tests (100% pass)
```

## Thiết lập môi trường

```bash
git clone https://github.com/nhiney/deep-social-sentiment-analysis.git
cd deep-social-sentiment-analysis

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce kết quả

```bash
# 1. Tải UIT-VSMEC và chuẩn bị data
python -m scripts.download_uit_vsmec --prepare \
    --crawled data/raw/crawled_emotions.xlsx

# 2. Pseudo-label Apify posts (cần GPU, ~15 phút)
python -m scripts.pseudo_label_apify \
    --input  data/processed/cleaned_unlabeled_posts.csv \
    --output data/processed/pseudo_labeled_apify.csv \
    --device cuda

# 3. Merge tất cả sources
python -m scripts.prepare_data \
    --crawled        data/raw/crawled_emotions.xlsx \
    --uit-vsmec      data/raw/UIT-VSMEC.csv \
    --pseudo-labeled data/processed/pseudo_labeled_apify.csv \
    --output-dir     data/processed

# 4. Train (cần GPU, ~10 phút trên T4)
python -m src.train --config configs/config.yaml

# 5. Evaluate
python -m src.evaluate \
    --checkpoint models/best_model \
    --data       data/processed/test.parquet \
    --output-dir reports/

# 6. Ablation study (~25 phút trên T4)
python -m scripts.run_ablation \
    --raw       data/raw/crawled_emotions.xlsx \
    --uit-vsmec data/raw/UIT-VSMEC.csv \
    --epochs    5 --device cuda
```

## Chạy tests

```bash
pytest tests/ -v
# 116 tests: 25 API + 64 preprocessing + 27 models — 100% pass, không cần GPU
```

## Demo & API

```bash
# Streamlit app (cần checkpoint)
streamlit run app/app.py

# FastAPI service
uvicorn app.main:app --reload
# GET  /health
# POST /predict
# POST /predict/batch
# POST /predict/explain
```

## Google Colab

Xem [`COLAB_GUIDE.md`](COLAB_GUIDE.md) để chạy toàn bộ pipeline trên Colab T4 GPU (~45 phút).
