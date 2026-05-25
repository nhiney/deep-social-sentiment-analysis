"""Evaluation utilities: metrics, confusion matrix, classification report.

We deliberately AVOID using accuracy as the headline metric — emotion
classification on social text is class-imbalanced (e.g. ``joy`` ≈ 30% of the
crawled corpus), so a model that always predicts ``joy`` would clock 30%
accuracy while completely failing the minority classes. **F1-Macro** treats
every class equally and is therefore the primary metric of record for this
project. Precision and Recall are reported alongside for diagnosability.

Run from the project root::

    python -m src.evaluate --checkpoint models/best_model \
                           --data data/processed/test.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from torch import Tensor
from torch.utils.data import DataLoader

from src.models import LateFusionModel

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


# =========================================================================== #
# Metrics
# =========================================================================== #
def compute_classification_metrics(
    y_true: Union[Tensor, np.ndarray, Sequence[int]],
    y_pred: Union[Tensor, np.ndarray, Sequence[int]],
    labels: Optional[Sequence[int]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute Accuracy + F1 / Precision / Recall under ``average``.

    The headline metric of this project is **F1-Macro** — it weights every
    class equally regardless of support, so it cannot be inflated by a
    classifier that ignores rare emotions.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (N,)
        Integer class labels — accepts ``torch.Tensor``, ``numpy.ndarray``
        or any sequence of ints. Tensors are detached and moved to CPU.
    labels : sequence of int, optional
        Restrict the computation to this subset of class ids
        (passed straight to ``sklearn.metrics``).
    average : {"macro", "micro", "weighted"}, default="macro"
        Averaging strategy for the multi-class scores.

    Returns
    -------
    dict[str, float]
        ``{"accuracy", "f1_<avg>", "precision_<avg>", "recall_<avg>"}``.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    # ``zero_division=0`` keeps the per-class score at 0 when sklearn would
    # otherwise emit a warning + NaN for classes with no predicted samples.
    p, r, f, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr,
        labels=labels,
        average=average,
        zero_division=0,
    )
    return {
        "accuracy":          float(accuracy_score(y_true_arr, y_pred_arr)),
        f"f1_{average}":         float(f),
        f"precision_{average}":  float(p),
        f"recall_{average}":     float(r),
    }


def per_class_metrics(
    y_true: Union[Tensor, np.ndarray, Sequence[int]],
    y_pred: Union[Tensor, np.ndarray, Sequence[int]],
    class_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Per-class precision / recall / F1 / support, keyed by class name.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (N,)
    class_names : sequence of str
        Human-readable label names (index = class id).

    Returns
    -------
    dict
        ``{class_name: {"precision", "recall", "f1", "support"}}``.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)

    label_ids = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true_arr, y_pred_arr,
        labels=label_ids,
        average=None,                 # per-class breakdown
        zero_division=0,
    )
    # Build the keyed dict in deterministic class-order.
    return {
        class_names[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(s[i]),
        }
        for i in label_ids
    }


def confusion_matrix_array(
    y_true: Union[Tensor, np.ndarray, Sequence[int]],
    y_pred: Union[Tensor, np.ndarray, Sequence[int]],
    n_classes: int,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Confusion matrix as a NumPy array of shape ``(n_classes, n_classes)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (N,)
    n_classes : int
        Total number of classes (defines the matrix shape).
    normalize : {"true", "pred", "all"}, optional
        Normalization mode (see ``sklearn.metrics.confusion_matrix``).
    """
    return confusion_matrix(
        _to_numpy(y_true),
        _to_numpy(y_pred),
        labels=list(range(n_classes)),
        normalize=normalize,
    )


def classification_report_str(
    y_true: Union[Tensor, np.ndarray, Sequence[int]],
    y_pred: Union[Tensor, np.ndarray, Sequence[int]],
    class_names: Sequence[str],
    digits: int = 4,
) -> str:
    """Wrap ``sklearn.metrics.classification_report`` for direct logging."""
    return classification_report(
        _to_numpy(y_true),
        _to_numpy(y_pred),
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=digits,
        zero_division=0,
    )


# =========================================================================== #
# Inference helpers
# =========================================================================== #
@torch.no_grad()
def predict(
    model: LateFusionModel,
    loader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run inference over ``loader`` and collect ``(y_true, y_pred[, y_prob])``.

    Parameters
    ----------
    model : LateFusionModel
        Model to evaluate. Will be put in ``eval`` mode and moved to ``device``.
    loader : DataLoader
        Yields the dictionary produced by
        :func:`~src.dataset.sentiment_collate_fn` with the ``"label"`` key.
    device : torch.device
        Target device.
    return_probs : bool, default=False
        If ``True``, the third element of the tuple is the softmax
        probabilities; otherwise it is ``None``.
    """
    model.eval()
    model.to(device)

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    for batch in loader:
        # Move every tensor in the batch to the target device. We pop ``label``
        # so it isn't forwarded into the model (the model accepts it but we
        # want explicit control here for clarity).
        labels = batch.pop("label").to(device)
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch_on_device)              # (B, n_classes)
        preds = logits.argmax(dim=-1)                  # (B,)
        all_true.append(labels.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

        if return_probs:
            # Softmax along class dim — used by calibration / ensembling code.
            probs = torch.softmax(logits, dim=-1)      # (B, n_classes)
            all_prob.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([], dtype=np.int64)
    y_prob = np.concatenate(all_prob, axis=0) if return_probs and all_prob else None
    return y_true, y_pred, y_prob


def evaluate(
    model: LateFusionModel,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run :func:`predict`, then aggregate every reporting metric.

    Returns
    -------
    dict
        ::

            {
                "overall":          {accuracy, f1_macro, precision_macro,
                                     recall_macro, f1_weighted, ...},
                "per_class":        {<class>: {precision, recall, f1, support}, ...},
                "confusion_matrix": np.ndarray[(C, C)],
                "report":           str,   # classification_report text
            }
    """
    y_true, y_pred, _ = predict(model, loader, device, return_probs=False)

    # Headline metrics — Macro is the primary metric for this imbalanced task.
    macro = compute_classification_metrics(y_true, y_pred, average="macro")
    weighted = compute_classification_metrics(y_true, y_pred, average="weighted")

    # Merge macro + weighted into a single flat dict for easy table-rendering.
    overall = {**macro,
               "f1_weighted":        weighted["f1_weighted"],
               "precision_weighted": weighted["precision_weighted"],
               "recall_weighted":    weighted["recall_weighted"]}

    n_classes = (
        len(class_names) if class_names is not None
        else int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    )
    cm = confusion_matrix_array(y_true, y_pred, n_classes=n_classes)

    out: Dict[str, Any] = {"overall": overall, "confusion_matrix": cm}
    if class_names is not None:
        out["per_class"] = per_class_metrics(y_true, y_pred, class_names)
        out["report"] = classification_report_str(y_true, y_pred, class_names)
    return out


# =========================================================================== #
# Internal helpers
# =========================================================================== #
def _to_numpy(x: Union[Tensor, np.ndarray, Sequence[Any]]) -> np.ndarray:
    """Coerce a tensor / array / list into a 1-D NumPy int array."""
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args() -> argparse.Namespace:
    """Parse args for ``python -m src.evaluate``."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved LateFusionModel checkpoint on a "
                    "processed parquet split."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Directory created by LateFusionModel.save_pretrained.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to a processed parquet (e.g. test.parquet).")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--class-names", type=str, nargs="+",
        default=["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"],
        help="Class names in id order.",
    )
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto' | 'cpu' | 'cuda' | 'cuda:N'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write metrics.json and confusion_matrix.npy. "
                             "If omitted, results are only printed.")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint + a processed dataset and print a metrics report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Lazy imports so the metrics functions can be used without these.
    import pandas as pd
    from transformers import AutoTokenizer

    from src.dataset import SocialSentimentDataset, sentiment_collate_fn
    from src.preprocessing import TabularPreprocessor

    # ---- Resolve device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Evaluating on device: %s", device)

    # ---- Load model ----
    logger.info("Loading checkpoint from %s", args.checkpoint)
    model = LateFusionModel.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_model_name)

    # ---- Load tabular preprocessor ----
    import joblib
    tab_pp_path = Path(args.checkpoint) / "tab_preprocessor.joblib"
    if tab_pp_path.exists():
        tab_pp = joblib.load(tab_pp_path)
        logger.info("Loaded tab_preprocessor from %s", tab_pp_path)
    else:
        logger.warning("tab_preprocessor.joblib not found in checkpoint — using empty preprocessor (tabular features will be zeros)")
        df_tmp = pd.read_parquet(args.data)
        tab_pp = TabularPreprocessor(numerical_cols=[], categorical_cols=[]).fit(df_tmp)

    # ---- Build dataset ----
    df = pd.read_parquet(args.data)
    label_to_id = {name: i for i, name in enumerate(args.class_names)}
    dataset = SocialSentimentDataset(
        dataframe=df,
        text_column=args.text_column,
        label_column=args.label_column,
        tokenizer=tokenizer,
        tabular_preprocessor=tab_pp,
        teencode_normalizer=None,
        label_to_id=label_to_id,
        max_length=args.max_length,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=sentiment_collate_fn,
    )

    # ---- Evaluate & print ----
    report = evaluate(model, loader, device, class_names=args.class_names)
    logger.info("Overall metrics:")
    for k, v in report["overall"].items():
        logger.info("  %-20s %.4f", k, v)
    logger.info("\nClassification report:\n%s", report["report"])
    logger.info("\nConfusion matrix:\n%s", report["confusion_matrix"])

    # ---- Save outputs ----
    if args.output_dir:
        import json as _json
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {**report["overall"]}
        if "per_class" in report:
            metrics_payload["per_class"] = report["per_class"]
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            _json.dump(metrics_payload, fh, ensure_ascii=False, indent=2)
        logger.info("Saved metrics.json → %s", out_dir / "metrics.json")

        np.save(str(out_dir / "confusion_matrix.npy"), report["confusion_matrix"])
        logger.info("Saved confusion_matrix.npy → %s", out_dir / "confusion_matrix.npy")


if __name__ == "__main__":
    main()
