"""FastAPI inference service for the Late Fusion sentiment model.

Run with::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /health        → liveness probe + model status
POST /predict       → single-text inference
POST /predict/batch → batch inference (up to 64 texts)
POST /predict/explain → single-text inference + LIME token attribution
"""

from __future__ import annotations

import contextlib
import html as html_mod
import logging
import os
import re
import urllib.request as _ureq
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast when the model dir doesn't exist yet.
# ---------------------------------------------------------------------------
_predictor   = None   # type: Optional[Any]   # LateFusionPredictor
_explainer   = None   # type: Optional[Any]   # TextExplainer
_tfidf_model = None   # type: Optional[Any]   # TfidfBaseline (lightweight baseline)

CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
_CHECKPOINT_ENV = "MODEL_CHECKPOINT"
_DEFAULT_CHECKPOINT = "models/best_model"
_TFIDF_PATH = "models/baseline/tfidf_logreg.joblib"


# --------------------------------------------------------------------------- #
# I/O schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """Request payload for single-text inference."""

    text: str = Field(..., min_length=1, description="Raw social-media post (Vietnamese / code-switched).")
    num_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional behavioral feature overrides (e.g. {'likes': 120}).",
    )
    cat_features: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional categorical feature overrides (e.g. {'has_hashtag': 'yes'}).",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v


class BatchPredictRequest(BaseModel):
    """Request payload for batch inference."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of raw posts (max 64).",
    )
    batch_size: int = Field(default=16, ge=1, le=64)

    @field_validator("texts")
    @classmethod
    def max_batch(cls, v: List[str]) -> List[str]:
        if len(v) > 64:
            raise ValueError("Maximum batch size is 64 texts.")
        return v


class ExplainRequest(BaseModel):
    """Request payload for LIME explanation."""

    text: str = Field(..., min_length=1)
    target_label: Optional[str] = Field(
        default=None,
        description="Class to attribute against. Defaults to the top prediction.",
    )
    num_samples: int = Field(
        default=200, ge=50, le=1000,
        description="Number of LIME perturbation samples.",
    )


class PredictResponse(BaseModel):
    """Single-text prediction response."""

    label: str
    confidence: float
    probs: Dict[str, float]
    explanation: Optional[Dict[str, Any]] = None


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictResponse]
    n_texts: int


class URLExtractRequest(BaseModel):
    """Request payload for Facebook URL content extraction."""
    url: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    """Request payload for 4-model comparison."""

    text: str = Field(..., min_length=1)
    likes: float = Field(default=0, ge=0)
    comments: float = Field(default=0, ge=0)
    shares: float = Field(default=0, ge=0)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
        return v


class CompareBatchRequest(BaseModel):
    """Run a whole comment thread through the 4 architectures."""

    texts: List[str] = Field(..., min_length=1, max_length=40)
    likes: float = Field(default=0, ge=0)
    comments: float = Field(default=0, ge=0)
    shares: float = Field(default=0, ge=0)


class SocialFetchRequest(BaseModel):
    """Fetch public comments from a social-media URL."""

    url: str = Field(..., min_length=1)
    max_comments: int = Field(default=20, ge=1, le=40)
    youtube_api_key: Optional[str] = Field(default=None)


class HealthResponse(BaseModel):
    """Liveness / readiness probe response."""

    status: str
    model_loaded: bool
    checkpoint: str
    class_names: List[str]


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    """FastAPI lifespan handler: load model at startup, release at shutdown."""
    global _predictor, _explainer, _tfidf_model

    checkpoint = os.environ.get(_CHECKPOINT_ENV, _DEFAULT_CHECKPOINT)
    logger.info("Loading model from: %s", checkpoint)

    try:
        # Deferred import so the module loads even without torch installed.
        from app.explainer import TextExplainer
        from app.inference import LateFusionPredictor

        _predictor = LateFusionPredictor(
            checkpoint_dir=checkpoint,
            class_names=CLASS_NAMES,
            device="auto",
            max_length=128,
            apply_normalizer=True,
        )
        _explainer = TextExplainer(
            class_names=CLASS_NAMES,
            predict_proba_fn=_predictor.predict_proba_for_lime,
            num_samples=200,
        )
        logger.info("Main model loaded successfully.")
    except Exception as exc:
        logger.warning("Main model could not be loaded (%s). Degraded mode.", exc)
        _predictor = None
        _explainer = None

    # Load lightweight TF-IDF baseline (always attempt regardless of main model)
    try:
        import joblib
        _tfidf_model = joblib.load(_TFIDF_PATH)
        logger.info("TF-IDF baseline loaded from %s", _TFIDF_PATH)
    except Exception as exc:
        logger.warning("TF-IDF baseline not found (%s).", exc)
        _tfidf_model = None

    if _predictor is not None:
        logger.info("Server ready.")

    yield  # application runs here

    # Shutdown: release GPU memory if the model is loaded.
    if _predictor is not None:
        try:
            import torch
            if hasattr(_predictor, "model"):
                _predictor.model.cpu()
                del _predictor.model
            torch.cuda.empty_cache()
        except Exception:
            pass
    _predictor = None
    _explainer = None
    _tfidf_model = None


app = FastAPI(
    title="Deep Social Sentiment API",
    description=(
        "Late Fusion XLM-RoBERTa + FT-Transformer model for 7-class Vietnamese emotion classification. "
        "Supports single / batch inference and LIME token-level explanations."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# Serve report figures at /figures/<filename>
_FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
if _FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(_FIGURES_DIR)), name="figures")

@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


def _require_predictor():
    """Raise 503 if the model is not loaded."""
    if _predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not loaded. "
                f"Set the {_CHECKPOINT_ENV!r} env var to a valid checkpoint directory "
                f"and restart the server. Default path: {_DEFAULT_CHECKPOINT!r}."
            ),
        )
    return _predictor


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    """Liveness / readiness probe.

    Returns ``status: "ok"`` when the model is loaded, ``status: "degraded"``
    when the server is running but the checkpoint was not found at startup.
    The HTTP status code is always 200 — use ``model_loaded`` to decide
    whether to route traffic.
    """
    checkpoint = os.environ.get(_CHECKPOINT_ENV, _DEFAULT_CHECKPOINT)
    return HealthResponse(
        status="ok" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        checkpoint=checkpoint,
        class_names=CLASS_NAMES,
    )


_FB_UA = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"
_FB_DOMAINS = {"facebook.com", "fb.com", "fb.watch", "m.facebook.com"}


def _og(html: str, prop: str) -> Optional[str]:
    """Extract og:<prop> meta content and unescape HTML entities."""
    m = re.search(
        rf'<meta[^>]+property=["\']og:{prop}["\'][^>]+content=["\']([^"\']*)["\']',
        html, re.I,
    )
    if not m:
        m = re.search(
            rf'<meta[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']og:{prop}["\']',
            html, re.I,
        )
    return html_mod.unescape(m.group(1)).strip() if m else None


@app.post("/extract-url", tags=["utility"])
def extract_url(payload: URLExtractRequest) -> Dict[str, Any]:
    """Fetch a public Facebook post URL and return its og:description as text.

    Uses the ``facebookexternalhit`` user-agent which Facebook whitelists for
    crawling public posts — returns OG metadata including the post body.
    Works for most public share links (``/share/p/...``, ``/posts/...``).
    Private posts, comments requiring login, and Reels may not be extractable.
    """
    url = payload.url.strip()
    try:
        netloc = urlparse(url).netloc.lstrip("www.")
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "URL không hợp lệ.")

    if not any(netloc.endswith(d) for d in _FB_DOMAINS):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Chỉ hỗ trợ link Facebook.")

    req_obj = _ureq.Request(url, headers={
        "User-Agent": _FB_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    })
    try:
        with _ureq.urlopen(req_obj, timeout=12) as resp:
            final_url = resp.url
            html_content = resp.read(120_000).decode("utf-8", errors="ignore")
    except Exception as exc:
        return {"success": False, "text": "", "title": "", "final_url": url,
                "message": f"Không tải được trang: {exc}"}

    desc  = _og(html_content, "description")
    title = _og(html_content, "title") or ""

    if desc:
        return {
            "success": True,
            "text": desc,
            "title": title,
            "final_url": final_url,
            "message": f"Đã tải nội dung từ bài viết của {title or 'người dùng'}.",
        }
    return {
        "success": False, "text": "", "title": title, "final_url": final_url,
        "message": "Không tìm thấy nội dung văn bản. Bài viết có thể ở chế độ riêng tư hoặc yêu cầu đăng nhập — vui lòng paste thủ công.",
    }


# --------------------------------------------------------------------------- #
# Multi-platform comment fetching
# --------------------------------------------------------------------------- #
_YT_DOMAINS = {"youtube.com", "youtu.be", "m.youtube.com", "youtube-nocookie.com"}
_RD_DOMAINS = {"reddit.com", "redd.it"}
_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _extract_youtube_id(url: str) -> Optional[str]:
    """Pull the 11-char video id out of any YouTube URL form."""
    m = re.search(r"(?:v=|/shorts/|/embed/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def _http_json(url: str, headers: Dict[str, str], timeout: int = 12) -> Any:
    """GET a URL and parse JSON, raising on transport error."""
    import json as _json
    req_obj = _ureq.Request(url, headers=headers)
    with _ureq.urlopen(req_obj, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8", errors="ignore"))


def _fetch_youtube_comments(url: str, max_n: int, api_key: Optional[str]) -> Dict[str, Any]:
    """Read top public comments via YouTube Data API v3 (no OAuth needed)."""
    key = api_key or os.environ.get("YOUTUBE_API_KEY")
    vid = _extract_youtube_id(url)
    if not vid:
        return {"success": False, "comments": [], "source_title": "",
                "message": "Không tìm thấy video ID trong link YouTube."}
    if not key:
        return {"success": False, "comments": [], "source_title": "",
                "message": "Cần YouTube Data API key (miễn phí, lấy tại console.cloud.google.com → bật YouTube Data API v3). "
                           "Dán key vào ô '🔑 API key' hoặc đặt biến môi trường YOUTUBE_API_KEY."}
    from urllib.parse import urlencode
    qs = urlencode({
        "part": "snippet", "videoId": vid, "maxResults": min(max_n, 50),
        "order": "relevance", "textFormat": "plainText", "key": key,
    })
    api = f"https://www.googleapis.com/youtube/v3/commentThreads?{qs}"
    try:
        data = _http_json(api, {"User-Agent": _BROWSER_UA, "Accept": "application/json"})
    except Exception as exc:
        return {"success": False, "comments": [], "source_title": "",
                "message": f"YouTube API lỗi: {exc}. Kiểm tra API key / quota / video có bật bình luận không."}
    comments = []
    for item in data.get("items", [])[:max_n]:
        sn = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        txt = (sn.get("textDisplay") or "").strip()
        if txt:
            comments.append({
                "author": sn.get("authorDisplayName", "—"),
                "text": txt,
                "likes": int(sn.get("likeCount", 0) or 0),
            })
    if not comments:
        return {"success": False, "comments": [], "source_title": "",
                "message": "Video không có bình luận công khai hoặc đã tắt bình luận."}
    return {"success": True, "comments": comments, "source_title": f"YouTube video {vid}",
            "message": f"Đã đọc {len(comments)} bình luận thật từ YouTube."}


def _fetch_reddit_comments(url: str, max_n: int) -> Dict[str, Any]:
    """Read a Reddit thread's comments via the public .json endpoint (no key).

    Reddit increasingly 403s automated reads from datacenter IPs; we try a
    couple of host/UA variants and fall back to a clear "paste manually"
    message so the demo never dead-ends.
    """
    clean = url.split("?")[0].rstrip("/")
    path = clean.split("reddit.com", 1)[-1] if "reddit.com" in clean else clean
    suffix = "/.json?limit=" + str(min(max_n + 5, 100)) + "&raw_json=1"
    candidates = [clean + suffix]
    for host in ("https://www.reddit.com", "https://old.reddit.com"):
        cand = host + path + suffix
        if cand not in candidates:
            candidates.append(cand)
    headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}

    data, last_exc = None, None
    for api in candidates:
        try:
            data = _http_json(api, headers)
            break
        except Exception as exc:
            last_exc = exc
    if data is None:
        return {"success": False, "comments": [], "source_title": "",
                "message": f"Reddit chặn đọc tự động từ máy chủ này ({last_exc}). "
                           "Reddit thường chặn IP máy chủ — hãy mở thread, copy bình luận và dán thủ công, "
                           "hoặc thử lại trên mạng khác."}
    if not isinstance(data, list) or len(data) < 2:
        return {"success": False, "comments": [], "source_title": "",
                "message": "Không đọc được bình luận — hãy dùng link tới thread cụ thể."}
    title = ""
    try:
        title = data[0]["data"]["children"][0]["data"].get("title", "")
    except Exception:
        pass
    comments = []
    for child in data[1].get("data", {}).get("children", []):
        d = child.get("data", {})
        body = (d.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            comments.append({
                "author": d.get("author", "—"),
                "text": body,
                "likes": int(d.get("score", 0) or 0),
            })
        if len(comments) >= max_n:
            break
    if not comments:
        return {"success": False, "comments": [], "source_title": title,
                "message": "Thread chưa có bình luận hiển thị công khai."}
    return {"success": True, "comments": comments, "source_title": title or "Reddit thread",
            "message": f"Đã đọc {len(comments)} bình luận thật từ Reddit (không cần API key)."}


def _fetch_facebook_post(url: str) -> Dict[str, Any]:
    """Facebook blocks comment scraping — return the public post body via OG tags.

    Tries the original URL first, then mbasic.facebook.com as a fallback (lighter
    page, sometimes reachable from datacenter IPs when www is blocked).
    """
    _BLOCK_MSG = (
        "Facebook chặn đọc tự động từ máy chủ này. "
        "Vui lòng mở bài viết trong trình duyệt, copy nội dung và dán vào ô bên dưới."
    )

    def _try_fetch(target: str) -> Optional[str]:
        req = _ureq.Request(target, headers={
            "User-Agent": _FB_UA,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        })
        try:
            with _ureq.urlopen(req, timeout=12) as resp:
                return resp.read(120_000).decode("utf-8", errors="ignore")
        except Exception:
            return None

    parsed = urlparse(url)
    variants = [url]
    if parsed.netloc.lstrip("www.") == "facebook.com":
        variants.append(parsed._replace(netloc="mbasic.facebook.com").geturl())

    html_content: Optional[str] = None
    for v in variants:
        html_content = _try_fetch(v)
        if html_content:
            break

    if not html_content:
        return {"success": False, "comments": [], "source_title": "", "message": _BLOCK_MSG}

    desc = _og(html_content, "description")
    title = _og(html_content, "title") or ""
    if desc:
        return {"success": True, "comments": [{"author": title or "Bài viết", "text": desc, "likes": 0}],
                "source_title": title,
                "message": "Facebook chặn đọc bình luận tự động — chỉ lấy được nội dung bài viết công khai. "
                           "Để phân tích bình luận, hãy copy thủ công và dán vào ô bên dưới."}
    return {"success": False, "comments": [], "source_title": title,
            "message": "Bài viết riêng tư hoặc yêu cầu đăng nhập — vui lòng dán nội dung thủ công."}


@app.post("/social/fetch", tags=["utility"])
def social_fetch(payload: SocialFetchRequest) -> Dict[str, Any]:
    """Fetch public comments from a social-media URL.

    Platform support (by ease of access, no login required):

    * **YouTube** — real comments via Data API v3 (free API key).
    * **Reddit**  — real comments via the public ``.json`` endpoint (no key).
    * **Facebook** — post body only (comments are login-gated).
    """
    url = payload.url.strip()
    try:
        netloc = urlparse(url).netloc.lstrip("www.").lower()
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "URL không hợp lệ.")
    if not netloc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "URL không hợp lệ.")

    def _match(domains: set) -> bool:
        return any(netloc == d or netloc.endswith("." + d) for d in domains)

    if _match(_YT_DOMAINS):
        platform, res = "youtube", _fetch_youtube_comments(url, payload.max_comments, payload.youtube_api_key)
    elif _match(_RD_DOMAINS):
        platform, res = "reddit", _fetch_reddit_comments(url, payload.max_comments)
    elif _match(_FB_DOMAINS):
        platform, res = "facebook", _fetch_facebook_post(url)
    else:
        # Platforms that block automated comment reads — recognize the URL and
        # guide the user to paste, so the selector feels complete.
        _paste_only = {
            "tiktok.com": "tiktok", "instagram.com": "instagram",
            "threads.net": "threads", "twitter.com": "twitter", "x.com": "twitter",
        }
        matched = next((name for dom, name in _paste_only.items()
                        if netloc == dom or netloc.endswith("." + dom)), None)
        if matched:
            return {"success": False, "platform": matched, "comments": [], "n": 0,
                    "source_title": "", "final_url": url,
                    "message": f"{matched.capitalize()} chặn đọc bình luận tự động (cần đăng nhập/API riêng). "
                               "Hãy mở bài viết, copy bình luận và dán vào ô bên dưới — phần phân tích 4 mô hình vẫn chạy đầy đủ."}
        return {"success": False, "platform": "unknown", "comments": [], "n": 0,
                "source_title": "", "final_url": url,
                "message": "Nền tảng chưa nhận diện. Đọc tự động: YouTube, Reddit, Facebook (chỉ bài viết). "
                           "Với nền tảng khác, hãy dán bình luận thủ công."}

    res.update({"platform": platform, "final_url": url, "n": len(res.get("comments", []))})
    return res


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    """Run end-to-end inference on a single post.

    * ``text``         — raw Vietnamese post (teencode normalization applied automatically).
    * ``num_features`` — optional dict to override auto-derived behavioral features
      (e.g. supply real ``likes`` counts scraped from Facebook).
    * ``cat_features`` — optional categorical overrides.
    """
    predictor = _require_predictor()

    overrides = {**payload.num_features, **payload.cat_features}

    try:
        result = predictor.predict([payload.text], tabular_overrides=overrides or None)[0]
    except Exception as exc:
        logger.exception("Inference error for text: %.80s", payload.text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        ) from exc

    return PredictResponse(
        label=result["label"],
        confidence=round(result["confidence"], 6),
        probs={k: round(v, 6) for k, v in result["probs"].items()},
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    """Run inference on a list of posts (up to 64).

    Processes texts in mini-batches of ``batch_size`` (default 16) to
    avoid OOM on the GPU. Tabular features are auto-derived from each text.
    """
    predictor = _require_predictor()

    texts = [t for t in payload.texts if t and t.strip()]
    if not texts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="All texts are blank after stripping whitespace.",
        )

    try:
        results = []
        bs = payload.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            results.extend(predictor.predict(chunk))
    except Exception as exc:
        logger.exception("Batch inference error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {exc}",
        ) from exc

    predictions = [
        PredictResponse(
            label=r["label"],
            confidence=round(r["confidence"], 6),
            probs={k: round(v, 6) for k, v in r["probs"].items()},
        )
        for r in results
    ]
    return BatchPredictResponse(predictions=predictions, n_texts=len(predictions))


@app.post("/predict/explain", response_model=PredictResponse, tags=["explainability"])
def predict_with_explanation(payload: ExplainRequest) -> PredictResponse:
    """Run inference + LIME token-level attribution on a single post.

    LIME explanation is returned inside the ``explanation`` field:

    ```json
    {
      "label": "anger",
      "confidence": 0.83,
      "probs": {...},
      "explanation": {
        "label": "anger",
        "confidence": 0.83,
        "tokens": [["từ_mạnh", 0.12], ["chửi", 0.09], ...],
        "highlight_html": "<div>...</div>",
        "n_samples": 200
      }
    }
    ```

    **Note**: generating a LIME explanation takes ~2–5 s with 200 samples.
    Increase ``num_samples`` only when you need stable attributions for
    research-quality visualizations.
    """
    predictor = _require_predictor()
    if _explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LIME explainer is not initialized.",
        )

    # Update the explainer's sample count for this request.
    _explainer.num_samples = payload.num_samples

    try:
        result = predictor.predict([payload.text])[0]
        explanation = _explainer.explain(
            payload.text,
            target_label=payload.target_label,
        )
    except Exception as exc:
        logger.exception("Explain error for text: %.80s", payload.text)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {exc}",
        ) from exc

    return PredictResponse(
        label=result["label"],
        confidence=round(result["confidence"], 6),
        probs={k: round(v, 6) for k, v in result["probs"].items()},
        explanation={
            "label":          explanation.label,
            "label_id":       explanation.label_id,
            "confidence":     round(explanation.confidence, 6),
            "tokens":         explanation.tokens,
            "highlight_html": explanation.highlight_html,
            "n_samples":      explanation.n_samples,
        },
    )


@app.post("/predict/compare", tags=["inference"])
def predict_compare(payload: CompareRequest) -> Dict[str, Any]:
    """Run one text through 4 distinct model architectures for comparison.

    Models:
    1. TF-IDF + LogisticRegression — classical bag-of-words baseline
    2. XLM-R only           — neural text encoder, tabular zeroed out
    3. FT-Transformer only  — tabular branch only, text embedding zeroed
    4. Teencode + XLM-R + FT-Transformer Fusion — full deployed model
    """
    predictor = _require_predictor()
    text = payload.text.strip()

    normalized_text = predictor.normalizer(text) if predictor.normalizer else text
    tab_overrides = {"likes": payload.likes, "comments": payload.comments, "shares": payload.shares}

    results = []

    # ── 1. TF-IDF + LogReg ──
    if _tfidf_model is not None:
        from src.preprocessing import TeencodeNormalizer as _TN
        _norm = _TN()
        norm_text = _norm(text)
        proba = _tfidf_model.predict_proba([norm_text])[0]
        top_id = int(proba.argmax())
        classes = _tfidf_model.pipeline.classes_
        probs_dict = {c: round(float(proba[i]), 6) for i, c in enumerate(classes)}
        results.append({
            "model_id": "tfidf",
            "model_name": "TF-IDF + LogReg",
            "backbone": "TF-IDF (1-2 gram) + Logistic Regression",
            "tags": ["Baseline", "Teencode ✓", "Nhẹ ~5MB"],
            "known_f1": 0.6401,
            "known_acc": 0.6327,
            "is_deployed": False,
            "available": True,
            "label": classes[top_id],
            "confidence": round(float(proba[top_id]), 6),
            "probs": probs_dict,
            "note": "Bag-of-words truyền thống — nhanh nhưng không hiểu ngữ cảnh",
        })
    else:
        results.append({
            "model_id": "tfidf",
            "model_name": "TF-IDF + LogReg",
            "backbone": "TF-IDF + Logistic Regression",
            "tags": ["Baseline"],
            "known_f1": 0.6401,
            "known_acc": 0.6327,
            "is_deployed": False,
            "available": False,
            "label": None, "confidence": None, "probs": None,
            "note": "File models/baseline/tfidf_logreg.joblib không tìm thấy.",
        })

    # ── 2. XLM-R only (text branch, tabular zeroed) ──
    try:
        probs = predictor.predict_text_branch_only([text])
        r = predictor._probs_to_results(probs)[0]
        results.append({
            "model_id": "xlmr",
            "model_name": "XLM-R only",
            "backbone": "XLM-RoBERTa-base (text branch)",
            "tags": ["Teencode ✓", "Neural"],
            "known_f1": 0.6548,
            "known_acc": 0.6647,
            "is_deployed": False,
            "available": True,
            "label": r["label"],
            "confidence": round(r["confidence"], 6),
            "probs": {k: round(v, 6) for k, v in r["probs"].items()},
            "note": "Chỉ dùng nhánh text (XLM-R), embedding tabular = 0. Hiểu ngữ nghĩa & ngữ cảnh.",
        })
    except Exception as e:
        results.append({
            "model_id": "xlmr", "model_name": "XLM-R only",
            "backbone": "XLM-RoBERTa-base", "tags": ["Neural"],
            "known_f1": 0.6548, "known_acc": 0.6647,
            "is_deployed": False, "available": False,
            "label": None, "confidence": None, "probs": None,
            "note": f"Lỗi: {e}",
        })

    # ── 3. FT-Transformer only (tabular branch, text zeroed) ──
    try:
        probs = predictor.predict_tabular_branch_only([text], tabular_overrides=tab_overrides)
        r = predictor._probs_to_results(probs)[0]
        results.append({
            "model_id": "fttransformer",
            "model_name": "FT-Transformer only",
            "backbone": "FT-Transformer (tabular branch)",
            "tags": ["Tabular ✓", "Hành vi"],
            "known_f1": 0.5200,
            "known_acc": 0.5500,
            "is_deployed": False,
            "available": True,
            "label": r["label"],
            "confidence": round(r["confidence"], 6),
            "probs": {k: round(v, 6) for k, v in r["probs"].items()},
            "note": "Chỉ dùng đặc trưng hành vi (likes, comments, độ dài văn bản,...). Text embedding = 0.",
        })
    except Exception as e:
        results.append({
            "model_id": "fttransformer", "model_name": "FT-Transformer only",
            "backbone": "FT-Transformer", "tags": ["Tabular"],
            "known_f1": 0.5200, "known_acc": 0.5500,
            "is_deployed": False, "available": False,
            "label": None, "confidence": None, "probs": None,
            "note": f"Lỗi: {e}",
        })

    # ── 4. Full Fusion — deployed model ──
    r = predictor.predict([text], tabular_overrides=tab_overrides)[0]
    results.append({
        "model_id": "fusion",
        "model_name": "Teencode + XLM-R + FT-Transformer Fusion",
        "backbone": "XLM-R + FT-Transformer → MLP Fusion",
        "tags": ["Teencode ✓", "Text ✓", "Tabular ✓", "Deployed"],
        "known_f1": 0.6877,
        "known_acc": 0.7020,
        "is_deployed": True,
        "available": True,
        "label": r["label"],
        "confidence": round(r["confidence"], 6),
        "probs": {k: round(v, 6) for k, v in r["probs"].items()},
        "note": "Mô hình đầy đủ đang chạy trong demo — kết hợp cả text và tabular features.",
    })

    return {
        "text_original": text,
        "text_normalized": normalized_text,
        "results": results,
    }


# Static per-architecture metadata (kept in sync with /predict/compare).
_MODEL_STATIC = [
    {"model_id": "tfidf", "model_name": "TF-IDF + LogReg",
     "backbone": "TF-IDF (1-2 gram) + Logistic Regression",
     "tags": ["Baseline", "Teencode ✓", "Nhẹ ~5MB"], "known_f1": 0.6401, "known_acc": 0.6327,
     "is_deployed": False,
     "note": "Bag-of-words truyền thống — nhanh nhưng không hiểu ngữ cảnh."},
    {"model_id": "xlmr", "model_name": "XLM-R only",
     "backbone": "XLM-RoBERTa-base (text branch)",
     "tags": ["Teencode ✓", "Neural"], "known_f1": 0.6548, "known_acc": 0.6647,
     "is_deployed": False,
     "note": "Chỉ nhánh text (XLM-R). Hiểu ngữ nghĩa & ngữ cảnh sâu."},
    {"model_id": "fttransformer", "model_name": "FT-Transformer only",
     "backbone": "FT-Transformer (tabular branch)",
     "tags": ["Tabular ✓", "Hành vi"], "known_f1": 0.5200, "known_acc": 0.5500,
     "is_deployed": False,
     "note": "Chỉ đặc trưng hành vi (likes, độ dài...). Text embedding = 0."},
    {"model_id": "fusion", "model_name": "Teencode + XLM-R + FT Fusion",
     "backbone": "XLM-R + FT-Transformer → MLP Fusion",
     "tags": ["Teencode ✓", "Text ✓", "Tabular ✓", "Deployed"], "known_f1": 0.6877, "known_acc": 0.7020,
     "is_deployed": True,
     "note": "Mô hình đầy đủ đang chạy — kết hợp text + hành vi."},
]


@app.post("/predict/compare/batch", tags=["inference"])
def predict_compare_batch(payload: CompareBatchRequest) -> Dict[str, Any]:
    """Run a whole comment thread through all 4 architectures.

    Powers the social-media dashboard: per-model predictions for every
    comment, the thread's emotion distribution, average confidence, and how
    often each model agrees with the deployed Fusion model on this thread.
    """
    predictor = _require_predictor()
    texts = [t.strip() for t in payload.texts if t and t.strip()]
    if not texts:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Tất cả bình luận đều rỗng.")

    tab_overrides = {"likes": payload.likes, "comments": payload.comments, "shares": payload.shares}
    texts_norm = [predictor.normalizer(t) if predictor.normalizer else t for t in texts]

    # Compute per-model probability matrices (each: list of result dicts).
    per_model_results: Dict[str, Optional[List[Dict[str, Any]]]] = {}

    # 1. TF-IDF
    if _tfidf_model is not None:
        from src.preprocessing import TeencodeNormalizer as _TN
        _norm = _TN()
        norm = [_norm(t) for t in texts]
        proba = _tfidf_model.predict_proba(norm)
        classes = list(_tfidf_model.pipeline.classes_)
        rows = []
        for row in proba:
            top = int(row.argmax())
            rows.append({"label": classes[top], "confidence": float(row[top]),
                         "probs": {c: float(row[i]) for i, c in enumerate(classes)}})
        per_model_results["tfidf"] = rows
    else:
        per_model_results["tfidf"] = None

    # 2. XLM-R only
    try:
        per_model_results["xlmr"] = predictor._probs_to_results(
            predictor.predict_text_branch_only(texts))
    except Exception as exc:
        logger.warning("XLM-R batch branch failed: %s", exc)
        per_model_results["xlmr"] = None

    # 3. FT-Transformer only
    try:
        per_model_results["fttransformer"] = predictor._probs_to_results(
            predictor.predict_tabular_branch_only(texts, tabular_overrides=tab_overrides))
    except Exception as exc:
        logger.warning("FT-Transformer batch branch failed: %s", exc)
        per_model_results["fttransformer"] = None

    # 4. Fusion (deployed)
    per_model_results["fusion"] = predictor.predict(texts, tabular_overrides=tab_overrides)

    fusion_labels = [r["label"] for r in per_model_results["fusion"]]

    def _round_probs(p: Dict[str, float]) -> Dict[str, float]:
        return {k: round(v, 6) for k, v in p.items()}

    models_out = []
    for meta in _MODEL_STATIC:
        rows = per_model_results.get(meta["model_id"])
        available = rows is not None
        preds, dist, avg_conf, agree = [], {c: 0 for c in CLASS_NAMES}, 0.0, 0
        if available:
            for i, r in enumerate(rows):
                preds.append({"label": r["label"], "confidence": round(r["confidence"], 6),
                              "probs": _round_probs(r["probs"])})
                dist[r["label"]] = dist.get(r["label"], 0) + 1
                avg_conf += r["confidence"]
                if r["label"] == fusion_labels[i]:
                    agree += 1
            avg_conf = round(avg_conf / len(rows), 6)
            agree = round(agree / len(rows), 4)
        models_out.append({**meta, "available": available, "predictions": preds,
                           "distribution": dist, "avg_confidence": avg_conf,
                           "agreement_with_deployed": agree})

    # Thread-level consensus: per comment, how many available models agree with Fusion.
    avail_ids = [m["model_id"] for m in models_out if m["available"]]
    full_agree = 0
    for i in range(len(texts)):
        labels_here = {mid: per_model_results[mid][i]["label"] for mid in avail_ids}
        if len(set(labels_here.values())) == 1:
            full_agree += 1

    return {
        "texts": texts,
        "texts_normalized": texts_norm,
        "n": len(texts),
        "class_names": CLASS_NAMES,
        "thread_distribution": {c: fusion_labels.count(c) for c in CLASS_NAMES},
        "unanimous_count": full_agree,
        "unanimous_ratio": round(full_agree / len(texts), 4),
        "models": models_out,
    }
