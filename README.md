# Deep Social Sentiment Analysis

Học sâu cho phân tích cảm xúc người dùng trên mạng xã hội — kiến trúc **Late Fusion** kết hợp:

- **Text branch:** `XLM-R` + Teencode Normalization (chuẩn hoá tiếng Việt mạng).
- **Tabular branch:** `FT-Transformer` mã hoá hành vi người dùng (số bài đăng, lượt thích, thời lượng phiên, …).
- **Fusion head:** MLP phân loại cảm xúc đa lớp.
- **Explainable AI:** SHAP / Captum để giải thích đóng góp của từng nhánh.

## Cấu trúc dự án

```
deep-social-sentiment-analysis/
├── app/                  # FastAPI inference service
├── configs/              # YAML hyperparameter configs
├── data/
│   ├── raw/              # Dữ liệu gốc (KHÔNG commit)
│   ├── processed/        # Dữ liệu đã preprocess (KHÔNG commit)
│   └── external/         # Từ điển teencode, embeddings, ... (KHÔNG commit)
├── models/               # Checkpoint (KHÔNG commit)
├── notebooks/            # EDA, prototyping
├── reports/              # Figures, metrics, runs
├── scripts/              # CLI helpers (data prep, sweep, ...)
├── src/
│   ├── preprocessing.py  # TeencodeNormalizer, TabularPreprocessor
│   ├── dataset.py        # SocialSentimentDataset (PyTorch)
│   ├── models.py         # TextBranch, TabularBranch, LateFusionModel
│   ├── train.py          # Training loop
│   └── evaluate.py       # Metrics & evaluation
└── tests/                # pytest unit tests
```

## Thiết lập môi trường

```bash
git clone https://github.com/nhiney/deep-social-sentiment-analysis.git
cd deep-social-sentiment-analysis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Quy trình

```bash
# 1. Train
python -m src.train --config configs/config.yaml

# 2. Evaluate
python -m src.evaluate --checkpoint models/best_model.pt

# 3. Serve demo
uvicorn app.main:app --reload
```
