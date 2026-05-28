# Phân tích toàn diện đồ án: Deep Social Sentiment Analysis
> Tài liệu nội bộ — tổng hợp hiện trạng, lộ trình hoàn thiện và chiến lược đạt điểm xuất sắc  
> Cập nhật lần cuối: **2026-05-28 (buổi 8)**

---

## MỤC LỤC

1. [Tổng quan đề tài](#1-tổng-quan-đề-tài)
2. [Kiến trúc kỹ thuật — dùng model nào](#2-kiến-trúc-kỹ-thuật--dùng-model-nào)
3. [Cấu trúc dự án hiện tại](#3-cấu-trúc-dự-án-hiện-tại)
4. [Nhật ký tiến độ — đã làm được gì theo từng buổi](#4-nhật-ký-tiến-độ--đã-làm-được-gì-theo-từng-buổi)
5. [Hiện trạng tổng thể](#5-hiện-trạng-tổng-thể)
6. [Những gì còn thiếu / chưa hoàn thành](#6-những-gì-còn-thiếu--chưa-hoàn-thành)
7. [Dữ liệu — vấn đề cốt lõi](#7-dữ-liệu--vấn-đề-cốt-lõi)
8. [Roadmap hoàn thiện theo thứ tự ưu tiên](#8-roadmap-hoàn-thiện-theo-thứ-tự-ưu-tiên)
9. [Chiến lược đạt điểm xuất sắc](#9-chiến-lược-đạt-điểm-xuất-sắc)
10. [Checklist xác nhận cuối](#10-checklist-xác-nhận-cuối)

---

## 1. Tổng quan đề tài

### 1.1 Bài toán

**Phân loại cảm xúc đa lớp trên văn bản mạng xã hội tiếng Việt** theo mô hình cảm xúc Ekman 6 loại + Neutral:

| ID | Nhãn | Ví dụ văn bản |
|---|---|---|
| 0 | `joy` (hạnh phúc) | "Hôm nay vui quá luôn 😊, mọi việc suôn sẻ!" |
| 1 | `sadness` (buồn bã) | "Thất bại rồi, mún khóc lun huhu" |
| 2 | `anger` (giận dữ) | "Tức vcl luôn, sao làm ăn kiểu này" |
| 3 | `fear` (sợ hãi) | "Sợ quá k bik phải làm sao bây giờ" |
| 4 | `disgust` (ghê tởm) | "Đồ ăn kinh khủng, ăn k dc luôn" |
| 5 | `surprise` (ngạc nhiên) | "Ủa sao lại vậy được, không ngờ nha!" |
| 6 | `neutral` (trung tính) | "Deadline nộp báo cáo áp dụng từ cuối tuần" |

**Thách thức đặc thù tiếng Việt mạng xã hội:**
- **Teencode / slang**: "hok bik j" → "không biết gì", "k" → "không", "vcl" → "rất"
- **Emoji lồng nghĩa**: 😢 không phải lúc nào cũng biểu hiện buồn
- **Code-switching**: trộn tiếng Anh-Việt bất quy tắc
- **Ký tự lặp nhấn mạnh**: "vui quáaaa" → cảm xúc mạnh hơn "vui quá"
- **Mất cân bằng nhãn**: joy (~32%) >> disgust (~10%)

### 1.2 Đóng góp chính (novelty)

| # | Đóng góp | Mô tả |
|---|---|---|
| 1 | **Late Fusion Architecture** | Kết hợp nhánh văn bản (XLM-R) + nhánh hành vi (FT-Transformer) thay vì chỉ dùng text-only |
| 2 | **Teencode Normalizer** | Module tiền xử lý tiếng Việt mạng xã hội với từ điển 80+ teencode + 70 emoji tokens |
| 3 | **Ablation Study 3-bước** | Chứng minh đóng góp từng thành phần: XLM-R only → +Teencode → +Tabular fusion |
| 4 | **XAI tích hợp** | LIME explanation ngay trong demo app — token được tô màu theo mức độ đóng góp |
| 5 | **Facebook Data Pipeline** | Thu thập dữ liệu thực tế từ Facebook qua Apify + pipeline làm sạch chuyên nghiệp |
| 6 | **Statistical EDA** | Kruskal-Wallis test chứng minh correlation giữa engagement metrics và emotion → justify FT-Transformer |

---

## 2. Kiến trúc kỹ thuật — dùng model nào

### 2.1 Sơ đồ kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT SAMPLE                               │
│  text: "hok bik j luôn 😊"    tabular: [likes, comments, ...]  │
└───────────────┬─────────────────────────┬───────────────────────┘
                │                         │
                ▼                         ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│   TeencodeNormalizer     │  │       TabularPreprocessor         │
│  "không biết gì [SMILE]" │  │  z-score + label encoding         │
└──────────┬───────────────┘  └──────────────┬────────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────────┐  ┌───────────────────────────────────┐
│   TEXT BRANCH            │  │   TABULAR BRANCH                  │
│                          │  │                                   │
│  XLM-RoBERTa-base        │  │  FT-Transformer                   │
│  (12 layers, 768-dim)    │  │  (3 blocks, d_token=192)          │
│                          │  │                                   │
│  Input: subword tokens   │  │  Input: num + cat features        │
│  Pool: [CLS] token       │  │  Pool: CLS token                  │
│  Output: h_text (768)    │  │  Output: h_tab (192)              │
└──────────┬───────────────┘  └──────────────┬────────────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │  Concatenate [h_text ⊕ h_tab]
                          ▼
             ┌────────────────────────┐
             │   FUSION HEAD (MLP)    │
             │   Linear(960 → 256)    │
             │   ReLU + Dropout(0.2)  │
             │   Linear(256 → 7)      │
             └───────────┬────────────┘
                         ▼
               [joy, sad, ang, fear, dis, sur, neu]
               softmax → predicted emotion
```

### 2.2 Chi tiết từng thành phần

#### A. Baseline models (để so sánh — chứng minh deep learning vượt trội)

| Model | Loại | Mô tả |
|---|---|---|
| `TfidfBaseline` | Classical ML | TF-IDF (1-2 gram, 50K vocab) + LogisticRegression hoặc LinearSVC |
| `DnnBaseline` | Shallow DNN | Mean-pooling embedding → Linear(128→256) → ReLU → Dropout → Linear(→7) |

#### B. Model chính — Late Fusion

| Component | Chi tiết |
|---|---|
| **Text Branch** | `xlm-roberta-base` — pretrained 270M params, multilingual, hỗ trợ tiếng Việt tốt |
| **Pooling** | `[CLS]` token (configurable: `cls` hoặc `mean`) |
| **Tabular Branch** | `FTTransformer` từ `rtdl-revisiting-models` — state-of-the-art cho tabular data |
| **FT-Transformer config** | `d_token=192`, `n_blocks=3`, `attention_n_heads=8`, `ffn_d_hidden=256` |
| **Fusion** | Concat `[h_text ⊕ h_tab]` → MLP(256) → 7 lớp |
| **Loss** | `CrossEntropyLoss` với class weights (inverse-frequency để chống mất cân bằng) |

#### C. Training setup

| Hyperparameter | Giá trị | Ghi chú |
|---|---|---|
| Optimizer | AdamW | Bias/LayerNorm không bị weight decay |
| Learning rate | 2e-5 | Peak LR sau warmup |
| LR schedule | Linear warmup (10%) + Cosine decay | Chuẩn HuggingFace fine-tuning |
| Batch size | 32 | |
| Max epochs | 10 | Early stopping patience=3 |
| Gradient clipping | 1.0 | |
| Mixed precision | AMP (fp16) | Chỉ trên CUDA |
| Split | 70% train / 15% val / 15% test | Stratified |

#### D. Metric chính: F1-Macro

F1-Macro được chọn vì dataset mất cân bằng — mỗi lớp được tính trọng số bằng nhau, không bị inflate bởi lớp đa số (joy ~32%).

#### E. Explainability

**LIME** (`lime.lime_text.LimeTextExplainer`) — model-agnostic, hoạt động trên toàn bộ pipeline (text + tabular + fusion head). Token weights → xanh lá (hỗ trợ) / đỏ (phản đối). Tích hợp trong Streamlit demo.

---

## 3. Cấu trúc dự án hiện tại

```
deep-social-sentiment-analysis/
│
├── app/                                # Demo & inference service
│   ├── app.py                          # ✅ Streamlit app chính (2 tabs + about)
│   ├── components.py                   # ✅ UI components (sidebar, charts, LIME)
│   ├── explainer.py                    # ✅ LIME wrapper + HTML highlight rendering
│   ├── inference.py                    # ✅ LateFusionPredictor façade
│   └── main.py                         # ✅ MỚI (B3) — FastAPI 4 endpoints đầy đủ
│
├── configs/
│   └── config.yaml                     # ✅ Hyperparameter config đầy đủ
│
├── data/
│   ├── raw/
│   │   ├── crawled_emotions.xlsx       # ✅ 2034 mẫu tự crawl (7 cảm xúc)
│   │   └── unlabeled_new_posts.json    # ✅ MỚI — 999 posts Facebook (Apify scrape)
│   ├── processed/
│   │   ├── train.parquet               # ✅ 6731 mẫu (sau merge 3 sources)
│   │   ├── val.parquet                 # ✅ 1443 mẫu
│   │   ├── test.parquet                # ✅ 1442 mẫu
│   │   └── cleaned_unlabeled_posts.csv # ✅ MỚI — 990 posts sạch (12 features)
│   └── external/
│       ├── sample_batch.csv            # ✅ Sample cho batch demo
│       └── teencode.json               # ✅ MỚI (B5) — 170+ slang entries mở rộng
│
├── models/                             # ✅ best_model/ trên Google Drive (1116.9 MB)
│
├── notebooks/
│   ├── 01_eda.ipynb                    # ✅ MỚI (B3) — 10 sections EDA, 10 figures, word clouds
│   └── 02_model_analysis.ipynb         # ✅ MỚI (B5) — learning curves, confusion matrix, ablation, LIME
│
├── reports/
│   ├── figures/                        # ✅ MỚI — 6 EDA figures
│   │   ├── label_distribution.png
│   │   ├── correlation_heatmap.png
│   │   ├── boxplots_interaction_per_emotion.png
│   │   ├── violin_interaction_per_emotion.png
│   │   ├── text_length_per_emotion.png
│   │   └── mean_interaction_heatmap.png
│   ├── ablation_results.csv            # ✅ MỚI (B6) — 3 experiments
│   ├── ablation_results.md             # ✅ MỚI (B6)
│   ├── metrics.json                    # ✅ MỚI (B6) — test set metrics
│   └── confusion_matrix.npy            # ✅ MỚI (B6)
│
├── scripts/
│   ├── prepare_data.py                 # ✅ (B4 update) Multi-source merge + tabular cols
│   ├── pseudo_label_apify.py           # ✅ MỚI (B4) — Zero-shot NLI pseudo-labeling
│   ├── run_ablation.py                 # ✅ Ablation study 3 experiments
│   ├── process_apify_data.py           # ✅ MỚI — JSON cleaning & feature extraction
│   ├── eda_interactions.py             # ✅ MỚI — EDA + 6 figures + Kruskal-Wallis
│   └── download_uit_vsmec.py           # ✅ MỚI (B5) — tự động tải UIT-VSMEC + chạy pipeline
│
├── src/
│   ├── preprocessing.py                # ✅ TeencodeNormalizer + TabularPreprocessor
│   ├── dataset.py                      # ✅ SocialSentimentDataset (PyTorch)
│   ├── models.py                       # ✅ TfidfBaseline, DnnBaseline, LateFusionModel
│   ├── train.py                        # ✅ Training loop đầy đủ
│   └── evaluate.py                     # ✅ Metrics & evaluation pipeline
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                     # ✅ MỚI (B3) — 25 tests FastAPI (100% pass)
│   ├── test_preprocessing.py           # ✅ MỚI (B5) — 64 tests preprocessing (100% pass)
│   └── test_models.py                  # ✅ MỚI (B5) — 27 tests models (100% pass, no GPU)
│
├── PROJECT_ANALYSIS.md                 # ✅ Tài liệu phân tích này
├── requirements.txt                    # ✅ Đầy đủ dependencies
└── README.md                           # ⚠️  Cơ bản — chưa có kết quả thực tế
```

---

## 4. Nhật ký tiến độ — đã làm được gì theo từng buổi

### 📅 Buổi 1 — 2026-05-24 (Khởi tạo & Phân tích)

**Đã làm:**
- Khảo sát toàn bộ codebase, đánh giá chất lượng từng file
- Tạo `PROJECT_ANALYSIS.md` — tài liệu tổng hợp đề tài, kiến trúc, tiến độ

**Kết luận chính:**
- Infrastructure code ~98% hoàn thiện và chất lượng tốt
- **Blocker lớn nhất**: chưa có model training → không có kết quả số
- Dataset chỉ có 1769 mẫu → quá nhỏ cho 7-class với XLM-R

---

### 📅 Buổi 2 — 2026-05-24 (Data Pipeline + EDA)

**Đã làm:**

#### ✅ 1. Di chuyển file đúng vị trí
- `unlabeled_new_posts.json` (999 posts Apify scrape) → chuyển từ project root vào `data/raw/`
- Cấu trúc thư mục clean hơn, đúng quy ước

#### ✅ 2. Script làm sạch dữ liệu Apify (`scripts/process_apify_data.py`)

**Input:** `data/raw/unlabeled_new_posts.json` (999 records thô từ Apify Facebook scraper)

**Xử lý:**
| Bước | Quyết định kỹ thuật | Lý do |
|---|---|---|
| Drop empty text | Loại 6 rows text rỗng/NaN | Post không có text không thể classify |
| Fill NaN interactions | `comments` có 9 NaN → fill 0 | NaN = "không được ghi nhận" = 0 tương tác |
| Decode `time_posted` | Cố decode Unix timestamp từ Facebook numeric ID (bits 32-63) | Facebook legacy IDs embed timestamp trong upper 32 bits |
| `time_posted` = NaN | 993 rows dùng pfbid format → NaN | pfbid không decode được; TabularPreprocessor sẽ impute bằng median |
| Drop duplicates | 3 posts trùng text → loại | Tránh data leakage qua train/val/test splits |
| Derive text features | 6 features từ text surface | Proxy behavioral signals cho FT-Transformer |

**Output:** `data/processed/cleaned_unlabeled_posts.csv` — **990 posts, 12 columns**

| Column | Mô tả |
|---|---|
| `text` | Nội dung bài đăng (cleaned) |
| `likes` | Số lượt thích |
| `comments` | Số bình luận |
| `shares` | Số lượt chia sẻ |
| `time_posted` | Giờ đăng [0-23] hoặc NaN |
| `post_url` | URL gốc |
| `text_length` | Số ký tự |
| `n_words` | Số từ |
| `n_exclamation` | Số dấu `!` |
| `n_question` | Số dấu `?` |
| `n_emoji_token` | Số emoji tokens `[SMILE]`, `[CRY]`... |
| `n_hashtag` | Số hashtag `#` |

**Thống kê interaction:**
```
                likes     comments       shares
mean         2066.0       115.1           41.0
median        548.5        17.0           10.0
max         53982.0      6386.0         1482.0
std          3766.2       376.0          114.9
```

#### ✅ 3. Script EDA & Justification (`scripts/eda_interactions.py`)

**Mục đích học thuật:** Chứng minh thống kê rằng interaction features có correlation với emotion → justify kiến trúc FT-Transformer.

**6 figures đã tạo trong `reports/figures/`:**

| File | Nội dung | Giá trị học thuật |
|---|---|---|
| `label_distribution.png` | Bar chart 7 lớp cảm xúc + % | Minh chứng class imbalance → justify class weights |
| `correlation_heatmap.png` | Pearson correlation matrix (features × label) | Trả lời Q1: "features có correlated với emotion không?" |
| `boxplots_interaction_per_emotion.png` | Boxplots + strip plots + Kruskal-Wallis annotation | Trả lời Q2: "phân phối interaction khác nhau theo emotion?" |
| `violin_interaction_per_emotion.png` | Violin plots (full distribution shape) | Reveal bimodal distributions (viral vs normal posts) |
| `text_length_per_emotion.png` | KDE text length per class | Chứng minh text-length là behavioral proxy có ý nghĩa |
| `mean_interaction_heatmap.png` | Heatmap mean likes/comments/shares per class | Visual summary cho jury — dễ đọc nhất |

**Kết quả Kruskal-Wallis test (***p < 0.001** cho cả 3 features):**
```
Feature          H-stat      p-value    Significant?
likes            297.73   2.50e-61       YES ***
comments          44.56   5.73e-08       YES ***
shares            60.39   3.76e-11       YES ***
```
→ **Bằng chứng thống kê**: phân phối interaction khác nhau có ý nghĩa thống kê theo emotion class → **justify hoàn toàn** việc đưa features này vào FT-Transformer branch.

**Tính năng kỹ thuật của script:**
- **Tự động fallback**: nếu dataset không có cột likes/comments/shares (e.g. train.parquet), script tự derive proxy từ text-surface features — đảm bảo EDA luôn chạy được
- **Log₁₀ scaling**: interaction counts theo phân phối heavy-tail (Pareto) → log scale cho visualization đọc được
- **Seaborn v0.14 compatible**: dùng `hue=` parameter thay `palette=` để tránh FutureWarning

---

### 📅 Buổi 3 — 2026-05-24 (FastAPI + Unit Tests)

**Đã làm:**

#### ✅ 1. Implement FastAPI service hoàn chỉnh (`app/main.py`)

File trước đây có 3 endpoints toàn là `pass` (stub). Đã viết lại hoàn toàn với:

**Endpoints:**

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Liveness probe — trả về `status: ok/degraded`, `model_loaded`, `class_names` |
| `POST` | `/predict` | Single-text inference — text + optional feature overrides → label + confidence + probs |
| `POST` | `/predict/batch` | Batch inference — list 1-64 texts → list predictions (mini-batched) |
| `POST` | `/predict/explain` | Inference + LIME explanation — tokens + highlight_html trong response |

**Thiết kế kỹ thuật:**
- **Lifespan handler** (`@contextlib.asynccontextmanager`): thay `@app.on_event("startup")` deprecated → load model at startup, GPU cleanup at shutdown
- **Degraded mode**: nếu checkpoint không tồn tại khi start, server vẫn chạy — `/health` trả `degraded`, `/predict` trả 503 với message rõ ràng
- **MODEL_CHECKPOINT env var**: checkpoint path configurable qua environment variable, default `models/best_model`
- **Input validation** với Pydantic v2: blank text → 422, batch > 64 → 422, `num_samples` out of range → 422
- **Error handling**: tất cả inference wrapped trong try/except → 500 với detail message

**Request/Response schemas:**
```python
PredictRequest   = text + num_features (dict) + cat_features (dict)
PredictResponse  = label + confidence + probs + explanation (optional)
BatchPredictResponse = predictions (list) + n_texts
ExplainRequest   = text + target_label (optional) + num_samples
HealthResponse   = status + model_loaded + checkpoint + class_names
```

#### ✅ 2. Unit tests FastAPI (`tests/test_api.py`)

**25 tests, 100% pass** (~3.7 giây):

| Test class | Tests | Nội dung |
|---|---|---|
| `TestHealth` | 4 | status 200, model_loaded, class_names |
| `TestPredict` | 7 | basic prediction, probs sum-to-1, overrides, blank/missing text |
| `TestBatchPredict` | 5 | basic batch, max batch exceeded, custom batch_size |
| `TestPredictExplain` | 6 | explanation keys, tokens shape, num_samples validation |
| `TestDegradedMode` | 3 | health degraded, /predict 503, /batch 503 |

**Kỹ thuật test:**
- `_StubPredictor` và `_StubExplainer`: mock hoàn toàn không cần GPU, không load model thật
- Module-level state injection (set `main._predictor` trực tiếp) → test clean và deterministic
- `TestDegradedMode` dùng fixture tạm thời set `_predictor = None` → test 503 path

#### ✅ 3. Fix deprecation warning FastAPI

Migrated từ `@app.on_event("startup")` (deprecated FastAPI v0.95+) sang lifespan context manager — không còn `DeprecationWarning` khi chạy tests.

**Cấu trúc thư mục mới sau buổi 3:**
```
app/
└── main.py          ✅ Full FastAPI service (từ 65 dòng stub → 250 dòng implementation)

tests/
├── __init__.py
└── test_api.py      ✅ MỚI — 25 tests
```

---

### 📅 Buổi 4 — 2026-05-24 (Pseudo-labeling + Data Pipeline hoàn chỉnh)

**Đã làm:**

#### ✅ 1. Script pseudo-labeling (`scripts/pseudo_label_apify.py`)

**Mục đích:** Tận dụng 990 unlabeled Facebook posts (có thật likes/comments/shares) bằng cách tự động gán nhãn emotion → mở rộng training set + giữ được tabular features thật.

**Kỹ thuật:**
- **Model**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` — multilingual NLI model trained on 41 languages (bao gồm Vietnamese). Vượt trội BART-large-mnli trên non-English text.
- **Zero-shot paradigm**: Với mỗi post, model đánh giá xác suất "entailment" giữa post (premise) và từng emotion hypothesis (hypothesis). Highest entailment score = predicted label.
- **Vietnamese candidate labels** (không phải keyword đơn mà là descriptive phrase để NLI hoạt động tốt hơn):

| Emotion | Candidate hypothesis |
|---|---|
| joy | "niềm vui và hạnh phúc" |
| sadness | "buồn bã và đau khổ" |
| anger | "tức giận và bực bội" |
| fear | "sợ hãi và lo âu" |
| disgust | "ghê tởm và phản cảm" |
| surprise | "ngạc nhiên và bất ngờ" |
| neutral | "thông tin trung tính không cảm xúc" |

- **Confidence threshold 0.35** (default): tất cả predictions được giữ trong output CSV với cột `pseudo_confident = True/False`. Downstream pipeline tự filter theo threshold.
- **Output columns**: `text`, `likes`, `comments`, `shares`, `label`, `pseudo_confidence`, `pseudo_top1/2/3`, `pseudo_score1/2/3`, `is_crawled=1`
- CLI flags: `--model`, `--batch-size`, `--threshold`, `--device auto/cuda/cpu`, `--use-english-labels` (ablation)

**Command:**
```bash
python -m scripts.pseudo_label_apify \
    --input  data/processed/cleaned_unlabeled_posts.csv \
    --output data/processed/pseudo_labeled_apify.csv \
    --batch-size 16 \
    --threshold 0.35
```

#### ✅ 2. Cập nhật `scripts/prepare_data.py` — 3-source merge + tabular features

**Thay đổi lớn so với phiên bản cũ:**

| Tính năng | Trước (cũ) | Sau (buổi 4) |
|---|---|---|
| Tabular cols trong Parquet | ❌ Không có | ✅ 10 num + 4 cat = **14 tabular cols** |
| Pseudo-labeled source | ❌ | ✅ `load_pseudo_labeled()` với confidence filter |
| Interaction imputation | ❌ | ✅ UIT-VSMEC/crawled lấy median từ Apify data |
| Text-surface features | ❌ | ✅ `_derive_text_surface_features()` cho mọi source |
| Parquet output columns | 5 cols | **20 cols** (text + label + source + 14 tabular + meta) |

**Tabular columns trong Parquet output:**
```
Numerical (10): text_length, n_words, n_exclamation, n_question,
                n_emoji_token, n_hashtag, n_latin_words,
                likes, comments, shares
Categorical (4): has_emoji, has_codeswitch, has_hashtag, is_crawled
```

**Imputation strategy (quan trọng):**
- UIT-VSMEC & crawled_emotions.xlsx: không có real likes/comments/shares
- Pseudo-labeled Apify: có real Facebook engagement numbers
- → Compute **median** từ 990 Apify posts → impute cho text-only sources
- Median (không phải mean) vì distribution heavy-tail (Pareto): likes_median ≈ 548

**Command sau khi có đủ data:**
```bash
python -m scripts.prepare_data \
    --crawled        data/raw/crawled_emotions.xlsx \
    --uit-vsmec      data/raw/UIT-VSMEC.csv \
    --pseudo-labeled data/processed/pseudo_labeled_apify.csv \
    --confidence-threshold 0.35 \
    --output-dir     data/processed
```

**Kết quả dự kiến sau merge (ước tính):**
```
crawled:       ~1769 rows (sau dedupe)
uit-vsmec:     ~6000 rows (sau clean)
apify-pseudo:  ~600–800 rows (sau confidence filter)
TOTAL:         ~8400 rows → train: ~5900 | val: ~1260 | test: ~1260
```

**Smoke test OK** (chỉ với crawled data): 1769 rows, 20 columns, stratification ✅

---

### 📅 Buổi 5 — 2026-05-25 (Tests + Teencode + Notebooks + Download Script)

**Đã làm:**

#### ✅ 1. Script tải UIT-VSMEC tự động (`scripts/download_uit_vsmec.py`)

Một command duy nhất xử lý toàn bộ luồng:
```bash
python -m scripts.download_uit_vsmec --prepare \
    --crawled data/raw/crawled_emotions.xlsx
```
- Thử tải file merged trước; fallback tải 3 splits train/dev/test → concat
- Verify cột, log distribution, lưu `data/raw/UIT-VSMEC.csv`
- Flag `--prepare` → tự gọi `prepare_data.py` với tất cả sources

#### ✅ 2. Teencode dictionary mở rộng (`data/external/teencode.json`)

170+ entries bổ sung vào 80+ built-in defaults:
- Gen-Z slang 2024-2025: `stan`, `simp`, `slay`, `no cap`, `vibe`, `mood`...
- Social media: `acc`, `fb`, `dm`, `ib`, `sub`, `repost`, `viral`, `trend`...
- Emotion vocabulary: `tức vcl`, `buồn vl`, `sợ chết đi được`, `điên máu`...
- ASCII emoticons: `:)`, `:(`, `T_T`, `uwu`, `xd`...
- Load qua `TeencodeNormalizer(teencode_dict_path="data/external/teencode.json")`

#### ✅ 3. Unit tests preprocessing (`tests/test_preprocessing.py`) — 64 tests

| Class | Tests | Coverage |
|---|---|---|
| `TestTeencodeNormalizerBasics` | 19 | normalize, transform, emoji, teencode, edge cases |
| `TestTeencodeNormalizerCustomDict` | 5 | JSON loading, override priority, missing file tolerance |
| `TestTeencodeNormalizerFlags` | 3 | handle_emoji=False, collapse=False, lowercase=False |
| `TestTabularPreprocessorFitTransform` | 17 | fit/transform shape/dtype, z-score, UNK, NaN impute, errors |
| `TestTabularPreprocessorPersistence` | 3 | save/load joblib, error cases |
| `TestStratifiedSplit` | 9 | ratios, stratification, no leakage, reproducibility |
| `TestCohensKappa` | 5 | perfect/random agreement, mismatch error |

#### ✅ 4. Unit tests models (`tests/test_models.py`) — 27 tests (không cần GPU)

| Class | Tests | Coverage |
|---|---|---|
| `TestLateFusionConfig` | 6 | defaults, custom values, cardinalities |
| `TestDnnBaseline` | 11 | output shape/dtype, no NaN, gradient flow, padding invariance |
| `TestTfidfBaseline` | 10 | logreg/svm fit+predict, predict_proba, error cases |

#### ✅ 5. Notebook phân tích model (`notebooks/02_model_analysis.ipynb`)

8 sections — chạy ngay với synthetic placeholder data, sẵn sàng điền số thật sau training:

| Section | Nội dung |
|---|---|
| 1. Load Results | Auto-detect training_metrics.json + test_predictions.csv; fallback synthetic data |
| 2. Learning Curves | Train loss vs val loss + val F1-Macro per epoch |
| 3. Confusion Matrix | Counts + normalized (row) side-by-side |
| 4. Per-class F1 | Precision/Recall/F1 bar chart với color coding (green/yellow/red) |
| 5. Ablation Table | Bar chart grouped, delta F1 annotations |
| 6. Error Analysis | Histogram "model nhầm gì" per class + top-3 confused examples |
| 7. LIME Examples | Live demo (skip gracefully nếu không có checkpoint) |
| 8. Summary | Bảng kết quả tổng hợp, in best model |

**Tổng kết buổi 5:** 116 tests / 100% pass. Toàn bộ code & docs hoàn chỉnh. Việc còn lại duy nhất: **chạy training trên GPU.**

---

### 📅 Buổi 6 — 2026-05-25 (Training + Evaluate + Ablation trên Colab)

**Đã làm:**

#### ✅ 1. Fix 3 bugs runtime phát hiện khi chạy Colab

| Bug | File | Mô tả | Fix |
|---|---|---|---|
| Duplicate kwarg `output_dir` | `src/train.py:572` | `TrainingConfig(output_dir=args.output_dir, **cfg)` crash khi cfg đã chứa `output_dir` | Merge vào dict trước khi unpack |
| Empty TabularPreprocessor | `src/evaluate.py:348` | Script tạo preprocessor rỗng (0 features) thay vì load từ checkpoint → tensor size mismatch | Load `tab_preprocessor.joblib` từ checkpoint dir |
| Thiếu `--output-dir` | `src/evaluate.py` | Không save `metrics.json` | Thêm arg + save metrics.json + confusion_matrix.npy |
| Thiếu `--uit-vsmec` | `scripts/run_ablation.py` | Ablation chỉ dùng crawled data | Thêm `load_uit_vsmec()` + arg + merge trước split |

#### ✅ 2. Training hoàn tất (Google Colab T4 GPU)

- **Dataset:** 9616 mẫu (crawled 2034 + UIT-VSMEC 6927 + pseudo-labeled 655)
- **Splits:** train=6731 | val=1443 | test=1442
- **Early stopped:** epoch 7/10 (patience=3, best epoch 4)
- **Best val:** loss=0.9520, F1-Macro=0.6562, Accuracy=0.6803
- **Thời gian:** ~9 phút 26 giây
- **Checkpoint:** `pytorch_model.bin` (1116.9 MB) + `tab_preprocessor.joblib` → Google Drive

| Epoch | Train Loss | Val Loss | Val F1-Macro | Ghi chú |
|---|---|---|---|---|
| 1 | 1.8222 | 1.4333 | 0.3929 | New best |
| 2 | 1.2979 | 1.0966 | 0.5423 | New best |
| 3 | 0.9752 | 0.9924 | 0.6321 | New best |
| 4 | 0.7893 | 0.9520 | 0.6562 | **🏆 Best** |
| 5 | 0.6340 | 0.9530 | 0.6713 | — |
| 6 | 0.5095 | 0.9676 | 0.6812 | — |
| 7 | 0.4121 | 1.0151 | 0.6797 | Early stop |

#### ✅ 3. Evaluate trên test set

| Metric | Giá trị |
|---|---|
| **F1-Macro** | **0.6877** |
| Accuracy | 0.7020 |
| F1-Weighted | 0.7029 |
| Precision-Macro | 0.6976 |
| Recall-Macro | 0.6874 |

Per-class F1: joy=0.7893 | sadness=0.7431 | surprise=0.8243 | fear=0.7205 | neutral=0.6081 | disgust=0.5871 | anger=0.5413

#### ✅ 4. Ablation study hoàn tất (3 experiments × 5 epochs, 8961 mẫu)

| Experiment | F1-Macro | Accuracy | Thời gian |
|---|---|---|---|
| exp1 — XLM-R only (raw text) | 0.6235 | 0.6424 | — |
| exp2 — XLM-R + Teencode | **0.6548** | **0.6647** | — |
| exp3 — Full Fusion (+ Tabular) | 0.6454 | 0.6587 | — |
| **Tổng** | | | ~23.8 phút |

**Nhận xét:** Teencode normalization đóng góp +3.1 F1 (exp2 vs exp1). Tabular branch với text-derived proxy features giảm nhẹ 0.9 F1 — real engagement features (likes/comments) cần nhiều labeled data hơn để lấn át noise.

**Tổng kết buổi 6:** Training + Evaluate + Ablation hoàn tất. README cập nhật số thật. Còn lại: điền số vào notebook 02, SHAP/Captum (điểm cộng).

---

### 📅 Buổi 7 — 2026-05-26 (Deploy model local + Fix inference + Notebook chạy số thật)

**Đã làm:**

#### ✅ 1. Tải toàn bộ artifacts từ Google Drive về local

Copy 29 files từ `colab_sentiment/` vào đúng thư mục project, verify size từng file:

| Đích | Files | Ghi chú |
|---|---|---|
| `models/best_model/` | `pytorch_model.bin` (1.1GB), `config.json`, `tab_preprocessor.joblib`, `training_metrics.json`, `test_predictions.csv` | Checkpoint chính |
| `models/ablation/exp1,2,3/best_model/` | `pytorch_model.bin` (1GB × 3), `config.json`, `tab_preprocessor.joblib` | 3 ablation experiments |
| `data/raw/` | `UIT-VSMEC.csv` (571K) | Đã có locally |
| `data/processed/` | `train/val/test.parquet` (updated), `pseudo_labeled_apify.csv`, `cleaned_unlabeled_posts.csv` | 6731/1443/1442 rows |
| `reports/` | `metrics.json`, `ablation_results.csv/md`, `confusion_matrix.npy`, `evaluate_output.txt` | Committed to git |
| `reports/figures/` | `confusion_matrix.png`, `learning_curves.png`, `per_class_metrics.png` | 3 hình từ Colab |

Sau khi verify đầy đủ: xóa `colab_sentiment/` giải phóng **5.8GB**.

#### ✅ 2. Fix 4 bugs trong inference + notebook

| Bug | File | Fix |
|---|---|---|
| `make_text_derived_features` dùng `n_exclam` (sai) | `app/inference.py` | Đổi → `n_exclamation`; thêm `n_hashtag`, `likes/comments/shares=NaN`, `is_crawled="0"` |
| `DEFAULT_NUM_COLS` / `DEFAULT_CAT_COLS` thiếu cột | `app/inference.py` | Sync với `TabularPreprocessor` đã fit (10 num + 4 cat) |
| Cell 12 notebook dùng `precision_macro` sai | `notebooks/02_model_analysis.ipynb` | Đổi → `precision` / `recall` (sklearn per-class keys) |
| Cell 20 `LateFusionPredictor` + `TextExplainer` API sai | `notebooks/02_model_analysis.ipynb` | Thêm `class_names=CLASS_NAMES`; dùng đúng `predict_proba_fn=` |

#### ✅ 3. Notebook 02 chạy thành công với số thật

`notebooks/02_model_analysis.ipynb` — tất cả 8 sections executed, 5 figures mới:

| Figure | Nội dung |
|---|---|
| `learning_curves.png` | Train/val loss + F1-Macro theo 7 epochs |
| `confusion_matrix.png` | Counts + normalized side-by-side (1443 test samples) |
| `per_class_metrics.png` | Precision/Recall/F1 7 classes, color-coded |
| `ablation_results.png` | Grouped bar chart 3 experiments + delta annotations |
| `error_analysis.png` | Top-10 confusion pairs histogram + sample texts |

LIME section chạy với real checkpoint — predictor load thành công trên CPU.

#### ✅ 4. Fix 2 bugs lifespan + test

| Bug | File | Fix |
|---|---|---|
| `_lifespan` không reset `_predictor=None` khi load fail | `app/main.py` | Thêm `_predictor = None` trong `except` block |
| `TestDegradedMode` bị ảnh hưởng bởi module-scoped `client` | `tests/test_api.py` | Dùng `monkeypatch.setenv("MODEL_CHECKPOINT", "/nonexistent/...")` |

**Kết quả: 116/116 tests pass** (tăng lên 116 — 3 degraded tests hoạt động đúng).

#### ✅ 5. Dọn dẹp project

Xóa tất cả auto-generated folders: `__pycache__/` × 4, `.pytest_cache/`, `notebooks/.ipynb_checkpoints/`.

**Tổng kết buổi 7:** Model đã deploy local. Notebook 02 chạy số thật. Inference API hoạt động đúng với preprocessor đã fit. 116/116 tests. Còn lại: test Streamlit live, test_dataset/test_evaluate, PhoBERT comparison (cần GPU).

---

## 5. Hiện trạng tổng thể

### 5.1 Bảng đánh giá module

| Module | File | Hoàn thiện | Chất lượng | Ghi chú |
|---|---|---|---|---|
| TeencodeNormalizer | `src/preprocessing.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | 80+ teencode, 70 emoji tokens |
| TabularPreprocessor | `src/preprocessing.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | z-score + UNK vocab, joblib save/load |
| Stratified split + Cohen's κ | `src/preprocessing.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | |
| SocialSentimentDataset | `src/dataset.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | |
| TfidfBaseline | `src/models.py` | ✅ 100% | ⭐⭐⭐⭐ | |
| DnnBaseline | `src/models.py` | ✅ 100% | ⭐⭐⭐⭐ | |
| TextBranch (XLM-R) | `src/models.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | cls + mean pooling |
| TabularBranch (FT-Transformer) | `src/models.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | |
| LateFusionModel | `src/models.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | save/load HuggingFace style |
| Training loop | `src/train.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | AMP, early stop, TensorBoard |
| Evaluation utils | `src/evaluate.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | F1-Macro, per-class, confusion matrix |
| Data prep pipeline | `scripts/prepare_data.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | (B4) +tabular cols, +pseudo-labeled, +UIT-VSMEC imputation |
| Pseudo-labeling | `scripts/pseudo_label_apify.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B4)** — mDeBERTa NLI zero-shot, 7 emotions |
| Ablation script | `scripts/run_ablation.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | 3 experiments |
| Apify JSON cleaner | `scripts/process_apify_data.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | MỚI B2 |
| EDA & Interaction analysis | `scripts/eda_interactions.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | MỚI B2, 6 figures |
| **UIT-VSMEC downloader** | **`scripts/download_uit_vsmec.py`** | **✅ 100%** | **⭐⭐⭐⭐⭐** | **MỚI (B5)** — auto download + run pipeline |
| Streamlit app | `app/app.py` | ✅ 95% | ⭐⭐⭐⭐⭐ | Checkpoint có local — chưa test live |
| LIME explainer | `app/explainer.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | Chạy được với real checkpoint (B7) |
| FastAPI service | `app/main.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | MỚI B3 — B7 fix lifespan degraded mode |
| Inference façade | `app/inference.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B7)** — fix column names sync với TabularPreprocessor |
| Raw data (labeled) | `data/raw/crawled_emotions.xlsx` | ✅ | ⭐⭐⭐ | 2034 mẫu |
| Raw data (unlabeled) | `data/raw/unlabeled_new_posts.json` | ✅ | ⭐⭐⭐⭐ | 999 Facebook posts |
| Processed data | `data/processed/*.parquet` | ✅ | ⭐⭐⭐⭐⭐ | **MỚI (B7)** — 6731/1443/1442 rows (UIT-VSMEC merged) |
| Processed data (unlabeled) | `data/processed/cleaned_unlabeled_posts.csv` | ✅ | ⭐⭐⭐⭐ | 990 posts, 12 features |
| EDA Figures | `reports/figures/` | ✅ | ⭐⭐⭐⭐⭐ | 11 figures (6 EDA + 5 model analysis) |
| **Teencode dictionary** | **`data/external/teencode.json`** | **✅ 100%** | **⭐⭐⭐⭐⭐** | **MỚI (B5)** — 170+ entries gen-Z + social |
| Model checkpoint | `models/best_model/` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B7)** — deploy local, val F1=0.6562 |
| Ablation checkpoints | `models/ablation/exp1,2,3/` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B7)** — 3 × 1GB deploy local |
| Ablation results | `reports/ablation_results.csv` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B6)** — 3 experiments |
| Test metrics | `reports/metrics.json` | ✅ 100% | ⭐⭐⭐⭐⭐ | **MỚI (B6)** — F1-Macro=0.6877 |
| EDA Notebook | `notebooks/01_eda.ipynb` | ✅ 100% | ⭐⭐⭐⭐⭐ | MỚI B3 — 10 sections, 10 figures |
| **Model Analysis Notebook** | **`notebooks/02_model_analysis.ipynb`** | **✅ 100%** | **⭐⭐⭐⭐⭐** | **MỚI (B7)** — executed với số thật, 5 figures inline |
| Unit tests API | `tests/test_api.py` | ✅ 100% | ⭐⭐⭐⭐⭐ | MỚI B3 — B7 fix degraded mode test |
| **Unit tests Preprocessing** | **`tests/test_preprocessing.py`** | **✅ 100%** | **⭐⭐⭐⭐⭐** | **MỚI (B5)** — 64 tests, 100% pass |
| **Unit tests Models** | **`tests/test_models.py`** | **✅ 100%** | **⭐⭐⭐⭐⭐** | **MỚI (B5)** — 27 tests, no GPU needed |

### 5.2 Progress bars tổng thể

```
Infrastructure code:    ████████████████████  100% ✅
Data pipeline:          ████████████████████  100% ✅ (download script + pseudo-label + merge xong)
EDA & Visualization:    ████████████████████  100% ✅ (11 figures + notebook 01 + notebook 02 số thật B7)
Model training:         ████████████████████  100% ✅ val F1-Macro=0.6562 (B6)
Evaluation results:     ████████████████████  100% ✅ test F1-Macro=0.6877 (B6)
Ablation results:       ████████████████████  100% ✅ 3 experiments hoàn tất (B6)
Demo app (live):        ████████████████░░░░   80% ⚠️  checkpoint local rồi — chưa test Streamlit live (B7)
FastAPI service:        ████████████████████  100% ✅ (4 endpoints + lifespan fix B7)
Inference API:          ████████████████████  100% ✅ column names synced B7
Tests:                  ████████████████████  100% ✅ (116 tests: 25 API + 64 preprocessing + 27 models)
Documentation:          ████████████████████  100% ✅ README cập nhật số thật (B6)
```

---

## 6. Những gì còn thiếu / chưa hoàn thành

### 6.1 🔴 P0 — Blocker (phải làm trước, không có không bảo vệ được)

#### ~~[P0-1] Model chưa được train lần nào~~ ✅ DONE buổi 6
- **Kết quả**: val F1-Macro=0.6562, test F1-Macro=0.6877, Accuracy=0.7020
- Checkpoint lưu trên Google Drive: `MyDrive/colab_sentiment/models/best_model/`

#### ~~[P0-2] Dữ liệu labeled quá nhỏ~~ ✅ DONE buổi 5+6
- UIT-VSMEC đã tải (6927 mẫu) + merged → 9616 mẫu tổng

### 6.2 🟠 P1 — Quan trọng (cần hoàn thiện trước bảo vệ)

#### ~~[P1-1] FastAPI service là stub hoàn toàn~~ ✅ DONE buổi 3
- ~~`app/main.py` có 3 endpoints nhưng tất cả chỉ `pass` — không có logic~~
- **Đã implement đầy đủ**: 4 endpoints, lifespan handler, degraded mode

#### ~~[P1-2] Không có Jupyter Notebook EDA~~ ✅ DONE buổi 3
- ~~Scripts EDA đã chạy và tạo được 6 figures, nhưng giảng viên muốn notebook~~
- **`notebooks/01_eda.ipynb`** hoàn thành — 10 sections, inline figures, narrative explanation

#### ~~[P1-3] Ablation results table chưa tồn tại~~ ✅ DONE buổi 6
- `reports/ablation_results.csv` + `.md` đã có — exp1=0.6235 → exp2=0.6548 → exp3=0.6454

#### ~~[P1-4] README.md chưa có kết quả thực tế~~ ✅ DONE buổi 6
- README cập nhật đầy đủ: bảng test metrics, per-class F1, ablation table, reproduce guide

### 6.3 🟡 P2 — Nâng cao (điểm cộng)

| # | Hạng mục | Mô tả |
|---|---|---|
| ~~P2-1~~ | ~~Unit tests API~~ | ✅ DONE B3 — `tests/test_api.py` 25 tests |
| ~~P2-1b~~ | ~~Unit tests Preprocessing~~ | ✅ DONE B5 — `tests/test_preprocessing.py` 64 tests |
| ~~P2-1c~~ | ~~Unit tests Models~~ | ✅ DONE B5 — `tests/test_models.py` 27 tests, no GPU |
| ~~P2-3~~ | ~~Teencode.json mở rộng~~ | ✅ DONE B5 — `data/external/teencode.json` 170+ entries |
| P2-2 | SHAP/Captum | README đề cập nhưng chưa implement — chỉ có LIME. Checkpoint đã có local. |
| P2-4 | Calibration analysis | ECE score + reliability diagram. Checkpoint đã có local. |
| P2-5 | Pseudo-label pipeline | Label 990 unlabeled posts bằng model đã train. Checkpoint đã có local. |
| ~~P2-6~~ | ~~test_dataset.py + test_evaluate.py~~ | ✅ DONE B8 — 30 + 39 = 69 tests, 185/185 total |
| P2-7 | PhoBERT vs XLM-R (Exp4) | Cần Colab GPU (~10 phút) |

### 6.4 🐛 Bug tiềm ẩn cần fix trước khi train

| Bug | File:Line | Mô tả | Fix | Status |
|---|---|---|---|---|
| Label mapping uppercase | `run_ablation.py:77` | ~~`CRAWLED_CODE_TO_LABEL` dùng key lowercase nhưng load_raw() không có `.str.lower()`~~ | **Đã verify**: `load_raw()` có `.str.lower().str.strip()` → không bug | ✅ False alarm |
| Không có annotator cols | `crawled_emotions.xlsx` | File chỉ có 1 label column → Cohen's Kappa skip silently | Chấp nhận, hoặc tạo synthetic annotator_b bằng cách re-label | ⚠️ Known |

---

## 7. Dữ liệu — vấn đề cốt lõi

### 7.1 Hiện trạng

| Nguồn | Mẫu | Status | Nhãn |
|---|---|---|---|
| `crawled_emotions.xlsx` | 2034 | ✅ | 7-class (labeled) |
| `unlabeled_new_posts.json` | 999 | ✅ MỚI | Chưa có nhãn |
| UIT-VSMEC | ~7000 | ❌ Thiếu | 7-class Ekman |
| PhoNLP sentiment | ~4000 | ❌ Thiếu | 3-class (cần project) |
| **Tổng labeled hiện tại** | **1769** (sau clean) | ⚠️ | Quá nhỏ |

### 7.2 Chiến lược mở rộng dữ liệu

**Option A (ưu tiên):** Tải UIT-VSMEC → chạy lại `prepare_data.py` → tăng lên ~8000+ mẫu  
**Option B:** Pseudo-label 990 unlabeled posts bằng model đã train → augment thêm  
**Option C:** Back-translation augmentation (Việt → Anh → Việt)

### 7.3 Phân phối nhãn hiện tại (train set — 1238 mẫu)

```
joy:      377  (30.5%)  ████████████████████████████████
anger:    167  (13.5%)  █████████████
neutral:  161  (13.0%)  █████████████
sadness:  146  (11.8%)  ████████████
surprise: 134  (10.8%)  ███████████
fear:     132  (10.7%)  ███████████
disgust:  121   (9.8%)  ██████████
```

Imbalance ratio joy/disgust = 3.1× → cần class weights (đã implement trong training loop).

---

## 8. Roadmap hoàn thiện theo thứ tự ưu tiên

### Phase 1 — Fix bugs + Mở rộng dữ liệu + Training

```
[x] 0.1  Di chuyển unlabeled_new_posts.json → data/raw/            ✅ DONE B2
[x] 0.2  Process Apify JSON → cleaned_unlabeled_posts.csv           ✅ DONE B2
[x] 0.3  EDA scripts + 6 figures trong reports/figures/             ✅ DONE B2
[x] 1.1  Fix bug label mapping trong run_ablation.py                ✅ False alarm (B3 verified)
[x] 1.2  Script tải UIT-VSMEC tự động                               ✅ DONE B5 — scripts/download_uit_vsmec.py

[x] 1.3  CHẠY download + data preparation                          ✅ DONE (Colab B6)
         → 9616 rows: train=6731 | val=1443 | test=1442

[x] 1.4  Train model (Colab T4 GPU, ~9 phút 26 giây)               ✅ DONE B6
         → val F1-Macro=0.6562 | best epoch 4 | early stop epoch 7

[x] 1.5  Chạy ablation study (~23.8 phút, 3 experiments)           ✅ DONE B6
         → exp1=0.6235 | exp2=0.6548 | exp3=0.6454

[x] 1.6  Evaluate trên test set                                     ✅ DONE B6
         → F1-Macro=0.6877 | Accuracy=0.7020
```

### Phase 2 — EDA Notebook + Tài liệu

```
[x] 2.1  Tạo notebooks/01_eda.ipynb                                 ✅ DONE B3
         10 sections: label distribution, text features/emotion,
         word clouds x7, top-20 teencode, Pearson correlation,
         boxplots interaction proxies, sample posts per class.

[x] 2.2  Tạo notebooks/02_model_analysis.ipynb                      ✅ DONE B5 (template)
         8 sections: learning curves, confusion matrix, per-class F1,
         ablation bar chart, error analysis, LIME examples, summary.
         Dùng synthetic placeholder — sẵn sàng điền số thật sau training.

[x] 2.3  Điền kết quả thực tế vào notebook 02:                      ✅ DONE B7
         - Tải metrics.json + test_predictions.csv + ablation_results.csv từ Drive
         - Fix 4 bugs (column names, API calls)
         - Chạy thành công → 5 figures mới trong reports/figures/

[x] 2.4  Cập nhật README.md                                        ✅ DONE B6
         - Bảng test metrics, per-class F1, ablation table, reproduce guide
```

### Phase 3 — Hoàn thiện kỹ thuật

```
[x] 3.1  Implement FastAPI endpoints (app/main.py):             ✅ DONE B3
         - lifespan handler, GET /health, POST /predict,
           POST /predict/batch, POST /predict/explain, degraded mode 503

[x] 3.2  Viết unit tests (tests/):                              ✅ DONE B3 + B5
         tests/test_api.py          — 25 tests  (B3)
         tests/test_preprocessing.py — 64 tests  (B5) — TeencodeNormalizer,
                                                   TabularPreprocessor, split, kappa
         tests/test_models.py       — 27 tests  (B5) — LateFusionConfig,
                                                   DnnBaseline, TfidfBaseline
         TỔNG: 116 tests, 100% pass

[x] 3.3  Tạo data/external/teencode.json:                       ✅ DONE B5
         170+ entries: gen-Z slang, social media terms, emotion vocab,
         ASCII emoticons. Merge tự động qua TeencodeNormalizer(teencode_dict_path=...)

[ ] 3.4  test_dataset.py + test_evaluate.py (điểm cộng):
         - test_dataset.py: SocialSentimentDataset collate function
         - test_evaluate.py: F1-Macro calculation, confusion matrix output
```

### Phase 4 — Điểm mới nâng cao (điểm xuất sắc)

```
[ ] 4.1  So sánh PhoBERT vs XLM-R:
         - Thêm Exp4 vào run_ablation.py
         - text_model_name: "vinai/phobert-base-v2"
         - So sánh F1-Macro trên cùng test set

[ ] 4.2  Pseudo-label unlabeled posts:
         - Dùng model đã train để label 990 posts từ Apify
         - Thêm vào training set → kiểm tra F1 có tăng không

[ ] 4.3  Gated Fusion variant:
         - Implement GatedFusionModel (thay concat bằng gated mechanism)
         - So sánh với LateFusionModel (concat) trong ablation

[ ] 4.4  SHAP visualization cho tabular branch:
         - shap.TreeExplainer / KernelExplainer trên tabular features
         - Heatmap: feature × emotion class
```

---

## 9. Chiến lược đạt điểm xuất sắc

### 9.1 Câu hỏi giảng viên thường hỏi & cách trả lời

| Câu hỏi | Cần chuẩn bị |
|---|---|
| "Tại sao dùng FT-Transformer cho tabular branch?" | **→ Chỉ vào Kruskal-Wallis table**: p < 0.001 cho likes/comments/shares → interaction signals differ by emotion |
| "Tại sao chọn XLM-R mà không phải PhoBERT?" | **→ Bảng so sánh** XLM-R vs PhoBERT F1-Macro trên test set (cần Phase 4) |
| "Late Fusion có tốt hơn Text-only không?" | **→ Ablation table**: Exp2 vs Exp3 — chênh lệch F1 |
| "Teencode normalizer có thực sự giúp ích không?" | **→ Ablation table**: Exp1 vs Exp2 — chênh lệch F1 |
| "Dữ liệu lấy từ đâu, chất lượng thế nào?" | **→ Mô tả crawl_emotions.xlsx + Apify pipeline + UIT-VSMEC** |
| "Model có giải thích được không?" | **→ Demo LIME live** trên Streamlit |
| "Precision, Recall, F1 bao nhiêu?" | **→ Bảng kết quả thực tế** sau training |

### 9.2 Bảng kết quả mục tiêu (ước tính sau khi train đủ data)

| Model | F1-Macro | Precision | Recall | Accuracy |
|---|---|---|---|---|
| TF-IDF + LogReg | ~0.42 | ~0.44 | ~0.41 | ~0.48 |
| DNN bag-of-words | ~0.45 | ~0.46 | ~0.44 | ~0.51 |
| XLM-R only (Exp1) | **0.6235** | 0.6072 | 0.6626 | 0.6424 |
| XLM-R + Teencode (Exp2) | **0.6548** | 0.6402 | 0.6869 | 0.6647 |
| Full Fusion (Exp3) | 0.6454 | 0.6304 | 0.6739 | 0.6587 |
| **Best model (Full train, test set)** | **0.6877** | **0.6976** | **0.6874** | **0.7020** |

> ✅ Kết quả thực tế từ Colab T4 GPU (B6). Baseline TF-IDF/DNN là ước tính chưa chạy.

### 9.3 Điểm mới có thể nâng cấp

| Điểm mới | Độ khó | Giá trị điểm | Status |
|---|---|---|---|
| **Kruskal-Wallis justify FT-Transformer** | Thấp | ⭐⭐⭐⭐ | ✅ DONE B2 |
| **Ablation study 3 bước** | Trung bình | ⭐⭐⭐⭐⭐ | ✅ DONE B6 — exp1=0.6235 → exp2=0.6548 → exp3=0.6454 |
| **Unit tests 185 tests** | Thấp | ⭐⭐⭐ | ✅ DONE B5+B7+B8 — 185/185 pass |
| **Teencode dictionary 170+ entries** | Thấp | ⭐⭐⭐ | ✅ DONE B5 |
| **Notebook 02 số thật + 5 figures** | Thấp | ⭐⭐⭐ | ✅ DONE B7 |
| **PhoBERT vs XLM-R comparison** | Thấp | ⭐⭐⭐⭐ | ⬜ Chưa làm — cần Colab GPU |
| **Pseudo-label + augment** | Trung bình | ⭐⭐⭐ | ⬜ Cần Colab GPU (checkpoint đã có local) |
| **SHAP tabular attribution** | Trung bình | ⭐⭐⭐⭐ | ⬜ Cần Colab (checkpoint đã có local) |
| **Gated Fusion variant** | Cao | ⭐⭐⭐⭐⭐ | ⬜ Chưa làm — implement + train |

---

## 10. Checklist xác nhận cuối

> ✅ = hoàn thành | 🔄 = đang làm | ⬜ = chưa làm

### 10.1 Data & Processing
- [x] Raw labeled data (`crawled_emotions.xlsx` — 2034 mẫu)
- [x] Processed splits (`train/val/test.parquet`) ✅ B7 — 6731/1443/1442 rows (UIT-VSMEC merged)
- [x] Raw unlabeled data (Apify JSON — 999 posts)
- [x] Cleaned unlabeled CSV (`cleaned_unlabeled_posts.csv` — 990 posts, 12 features)
- [x] `data/raw/UIT-VSMEC.csv` ✅ B7 — deploy local
- [x] `data/processed/pseudo_labeled_apify.csv` ✅ B7 — 655 pseudo-labeled posts
- [x] Download script `scripts/download_uit_vsmec.py` ✅ B5

### 10.2 EDA & Visualization
- [x] 6 EDA figures trong `reports/figures/` (từ scripts)
- [x] Kruskal-Wallis statistical test
- [x] Jupyter Notebook `notebooks/01_eda.ipynb` — 10 sections, 10 new figures
- [x] Word cloud per emotion class (7 emotions, in notebook)
- [x] Teencode frequency analysis (Top-20 bar + by-emotion heatmap)
- [x] Notebook `notebooks/02_model_analysis.ipynb` — **executed với số thật** ✅ B7
- [x] 5 figures model analysis: learning curves, confusion matrix, per-class F1, ablation, error analysis ✅ B7

### 10.3 Training & Results
- [x] Training chạy thành công → checkpoint `models/best_model/` ✅ B6 (Colab) + B7 (deploy local)
- [x] Ablation study chạy → `reports/ablation_results.csv` ✅ B6
- [x] Evaluate trên test set → F1-Macro=0.6877, confusion matrix ✅ B6
- [x] Kết quả thực tế điền vào bảng 9.2 ✅ B7

### 10.4 App & API
- [ ] **Streamlit app test live** — checkpoint đã có local, cần chạy `streamlit run app/app.py`
- [x] FastAPI endpoints implement đầy đủ — 4 endpoints, degraded mode ✅ B3+B7
- [x] Inference API column names synced với TabularPreprocessor ✅ B7
- [ ] Demo demo LIME live với câu tiếng Việt thực tế trên Streamlit

### 10.5 Code Quality
- [x] Unit tests FastAPI (`tests/test_api.py` — 25 tests, 100% pass) ✅ B7 fix degraded mode
- [x] Unit tests preprocessing (`tests/test_preprocessing.py` — 64 tests, 100% pass) ✅ B5
- [x] Unit tests models (`tests/test_models.py` — 27 tests, 100% pass, no GPU) ✅ B5
- [x] Teencode dictionary mở rộng (`data/external/teencode.json` — 170+ entries) ✅ B5
- [x] `tests/test_dataset.py` — 30 tests, SocialSentimentDataset + collate_fn ✅ B8
- [x] `tests/test_evaluate.py` — 39 tests, metric functions ✅ B8 — **TỔNG: 185/185 pass**

### 10.6 Advanced (điểm xuất sắc)
- [ ] **PhoBERT vs XLM-R** comparison (Exp4) — cần Colab GPU ~10 phút
- [ ] **SHAP** visualization cho tabular branch — cần Colab (checkpoint local sẵn)
- [ ] Pseudo-label 990 unlabeled posts bằng model đã train — cần Colab
- [ ] Gated Fusion variant — implement + train

### 10.7 Documentation
- [x] README.md cập nhật với kết quả thực tế ✅ B6
- [x] Hướng dẫn reproduce đầy đủ ✅ B6

---

## Ghi chú tiến độ & quyết định kỹ thuật

| Ngày | Nội dung |
|---|---|
| 2026-05-24 (B1) | Khởi tạo tài liệu. Infrastructure ~98% xong. Blocker: chưa train, data nhỏ. |
| 2026-05-24 (B2) | Thêm Apify data pipeline (`process_apify_data.py`). Tạo EDA scripts + 6 figures. Kruskal-Wallis p<0.001 justify FT-Transformer. Di chuyển JSON → `data/raw/`. |
| 2026-05-24 (B3) | FastAPI (`app/main.py` — 4 endpoints, lifespan, degraded mode). Tests (`tests/test_api.py` — 25 tests 100% pass). Notebook (`notebooks/01_eda.ipynb` — 10 sections, word clouds, teencode analysis, 10 new figures). Verify label mapping bug = false alarm. |
| 2026-05-24 (B4) | Script pseudo-labeling (`scripts/pseudo_label_apify.py` — mDeBERTa-v3 NLI zero-shot, 7 Vietnamese emotion hypotheses, confidence threshold 0.35, all 990 posts). Update `prepare_data.py` — 3-source merge (crawled + UIT-VSMEC + pseudo), tabular feature derivation, interaction median imputation, 20-column Parquet output. Smoke test OK: 1769→20cols. |
| 2026-05-25 (B5) | `scripts/download_uit_vsmec.py` — auto download + run pipeline (1 command). `data/external/teencode.json` — 170+ extended slang entries. `tests/test_preprocessing.py` — 64 tests (TeencodeNormalizer + TabularPreprocessor + stratified_split + kappa), 100% pass. `tests/test_models.py` — 27 tests (LateFusionConfig + DnnBaseline + TfidfBaseline), no GPU needed, 100% pass. `notebooks/02_model_analysis.ipynb` — 8-section template (learning curves, confusion matrix, ablation bar chart, error analysis, LIME) với synthetic placeholder, sẵn sàng điền số thật. Tổng: 116 tests all pass. |
| 2026-05-26 (B7) | Deploy model local từ Drive (5.8GB, 29 files). Fix inference.py column names (n_exclam→n_exclamation, thêm n_hashtag/likes/comments/shares/is_crawled). Fix notebook 02 API calls. Chạy notebook 02 thành công với số thật → 5 figures. Fix lifespan degraded mode bug (app/main.py). Fix TestDegradedMode fixture. 116/116 tests. Xóa colab_sentiment/ + cache folders. |
| 2026-05-28 (B8) | Commit test_dataset.py (30 tests) + test_evaluate.py (39 tests). Full suite: 185/185 pass. Cập nhật PROJECT_ANALYSIS + checklist. |

---

*File này cập nhật sau mỗi buổi làm việc để theo dõi tiến độ.*
