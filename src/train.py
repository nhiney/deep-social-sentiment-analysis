"""Training loop for the :class:`~src.models.LateFusionModel`.

Run from the project root::

    python -m src.train --config configs/config.yaml

Highlights
----------
* Multi-input dictionary batches (``input_ids``, ``attention_mask``,
  ``num_features``, ``cat_features``) — handled uniformly by ``model(**batch)``.
* AdamW with parameter-group weight-decay split (bias / LayerNorm exempt).
* Linear warmup + cosine decay LR schedule.
* Optional ``torch.cuda.amp`` mixed-precision (no-op on CPU).
* Gradient clipping.
* :class:`EarlyStopping` on validation **loss** with best-checkpoint dump.
* ``logging`` + TensorBoard writer (no ``print`` calls in the training path).
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import SocialSentimentDataset, sentiment_collate_fn
from src.evaluate import compute_classification_metrics, evaluate
from src.models import LateFusionConfig, LateFusionModel

logger = logging.getLogger(__name__)


# =========================================================================== #
# Config
# =========================================================================== #
@dataclass
class TrainingConfig:
    """Bundle of hyperparameters consumed by :func:`train_model`.

    Attributes
    ----------
    output_dir : str
        Where checkpoints, logs and metrics are written. ``best_model/`` is
        the canonical sub-directory used by :meth:`LateFusionModel.save_pretrained`.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size used for both train & val.
    learning_rate : float
        Peak learning rate after warmup.
    weight_decay : float
        AdamW weight decay coefficient (excluded from bias/LayerNorm).
    warmup_ratio : float
        Fraction of total steps used for linear warmup before cosine decay.
    grad_clip_norm : float
        Max L2 norm of gradients before clipping; ``0`` disables.
    mixed_precision : bool
        Enable ``torch.cuda.amp`` mixed-precision training (CUDA only).
    early_stopping_patience : int
        Stop after this many epochs without val-loss improvement.
    early_stopping_min_delta : float
        Required improvement (in val-loss units) to reset the patience counter.
    seed : int
        Reproducibility seed.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"`` | explicit ``"cuda:0"``.
    log_interval_steps : int
        Step-level TensorBoard logging interval.
    num_workers : int
        ``DataLoader`` worker count. Default ``0`` for safer CPU dev runs.
    """

    output_dir: str = "models"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    seed: int = 42
    device: str = "auto"
    log_interval_steps: int = 50
    num_workers: int = 0


# =========================================================================== #
# Training primitives
# =========================================================================== #
def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seed all CUDA devices — the loop below is a no-op on CPU machines.
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """Convert ``"auto"`` / ``"cuda"`` / ``"cpu"`` strings to ``torch.device``."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    """Build AdamW with parameter groups (no decay on bias / LayerNorm).

    HuggingFace convention: weight decay should NOT be applied to bias terms
    or to LayerNorm weights — doing so harms transfer-learning fine-tuning.
    """
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    grouped = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=learning_rate)


def build_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    warmup_ratio: float,
) -> LambdaLR:
    """Linear-warmup-then-cosine-decay LR schedule.

    Uses a ``LambdaLR`` so we don't depend on the optional
    ``transformers.optimization`` import path.
    """
    num_warmup = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        # Linear warmup phase.
        if current_step < num_warmup:
            return float(current_step) / float(num_warmup)
        # Cosine-decay phase.
        progress = (current_step - num_warmup) / max(1, num_training_steps - num_warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =========================================================================== #
# Early stopping
# =========================================================================== #
class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int, default=3
        Number of epochs with no improvement after which training stops.
    mode : {"min", "max"}, default="min"
        ``"min"`` for losses, ``"max"`` for F1 etc.
    min_delta : float, default=0.0
        Minimum change in the monitored quantity to qualify as an improvement.

    Attributes
    ----------
    best : float
        Best monitored value seen so far.
    counter : int
        Epochs since last improvement.
    should_stop : bool
        Set to True once ``counter`` ≥ ``patience``.
    """

    def __init__(
        self,
        patience: int = 3,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Register a new value; return ``True`` if it improved on the best."""
        # Comparator branches by mode — keeps the call site free of branches.
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# =========================================================================== #
# Per-step / per-epoch helpers
# =========================================================================== #
def _move_batch(
    batch: Mapping[str, Tensor], device: torch.device,
) -> Dict[str, Tensor]:
    """Move every tensor in a batch dict onto ``device``."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_one_epoch(
    model: LateFusionModel,
    loader: DataLoader,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    mixed_precision: bool = True,
    epoch: int = 0,
    writer: Optional[SummaryWriter] = None,
    log_interval_steps: int = 50,
) -> Dict[str, float]:
    """Train ``model`` for one full pass over ``loader``.

    Loss = cross-entropy over the model's logits and the integer labels in
    the batch (key ``"label"``). Mixed precision is enabled only on CUDA.
    """
    model.train()

    # ``GradScaler`` avoids underflow when training with float16 activations.
    # On CPU we use a no-op scaler (``enabled=False``) so the same code path
    # works regardless of device.
    use_amp = bool(mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    running_loss = 0.0
    n_samples = 0
    n_steps = len(loader)

    for step, batch in enumerate(loader):
        batch = _move_batch(batch, device)
        labels = batch.pop("label")            # (B,) — kept out of model kwargs

        optimizer.zero_grad(set_to_none=True)

        # Forward + loss under autocast (no-op on CPU / when AMP disabled).
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(**batch)            # (B, n_classes)
            # CrossEntropyLoss expects raw logits + integer targets — do NOT
            # softmax here; the loss applies log-softmax internally.
            loss = loss_fn(logits, labels)

        # Scaled backward + clipping + step.
        scaler.scale(loss).backward()
        if grad_clip_norm and grad_clip_norm > 0:
            # Unscale before clipping so the threshold operates on real grad
            # magnitudes (not the AMP-scaled ones).
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Track running average — weight by batch size for the partial last
        # batch case where labels.size(0) < batch_size.
        bsz = labels.size(0)
        running_loss += loss.item() * bsz
        n_samples += bsz

        # Step-level TensorBoard logging (lr + loss).
        if writer is not None and (step % log_interval_steps == 0):
            global_step = epoch * n_steps + step
            writer.add_scalar("train/step_loss", loss.item(), global_step)
            writer.add_scalar(
                "train/lr", scheduler.get_last_lr()[0], global_step,
            )

    avg_loss = running_loss / max(1, n_samples)
    return {"loss": avg_loss}


@torch.no_grad()
def validate(
    model: LateFusionModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Compute val loss + F1-Macro / Precision / Recall / Accuracy."""
    model.eval()

    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = _move_batch(batch, device)
        labels = batch.pop("label")

        logits = model(**batch)                 # (B, n_classes)
        loss = loss_fn(logits, labels)

        bsz = labels.size(0)
        running_loss += loss.item() * bsz
        n_samples += bsz

        # Argmax inference — kept outside the autograd graph by the decorator.
        all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    val_loss = running_loss / max(1, n_samples)

    # Aggregate predictions across the full val set and compute headline
    # metrics (Macro is the project's primary metric).
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro = compute_classification_metrics(y_true, y_pred, average="macro")

    return {"val_loss": val_loss, **{f"val_{k}": v for k, v in macro.items()}}


# =========================================================================== #
# Top-level entry point
# =========================================================================== #
def train_model(
    train_dataset: SocialSentimentDataset,
    val_dataset: SocialSentimentDataset,
    model_config: LateFusionConfig,
    training_config: TrainingConfig,
    class_weights: Optional[Tensor] = None,
    tab_preprocessor: Optional[Any] = None,
) -> Tuple[LateFusionModel, Dict[str, Any]]:
    """Run the full training pipeline.

    Parameters
    ----------
    train_dataset, val_dataset : SocialSentimentDataset
        Datasets sharing the same fitted preprocessors.
    model_config : LateFusionConfig
        Architecture hyperparameters.
    training_config : TrainingConfig
        Optimization hyperparameters.
    class_weights : Tensor, optional
        Per-class weight passed to ``CrossEntropyLoss`` to combat class
        imbalance. Shape ``(n_classes,)``. Pass ``None`` to disable.

    Returns
    -------
    tuple
        ``(best_model, history)`` where ``history`` contains:
            ``"best_metrics"``  — dict of val metrics at best epoch.
            ``"best_epoch"``    — int index of the best epoch.
            ``"history"``       — list of per-epoch dicts.
            ``"checkpoint"``    — path to the saved best checkpoint.
    """
    cfg = training_config

    # -------- Reproducibility & device --------
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info("Training on device: %s", device)

    # -------- Loaders --------
    # ``pin_memory`` only helps when transferring to CUDA — on CPU it's a no-op.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sentiment_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=sentiment_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=pin,
    )

    # -------- Model + loss --------
    model = LateFusionModel(model_config).to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    # CrossEntropyLoss = log_softmax + NLL — accepts raw logits + integer
    # targets, so the model's forward returns logits (not probabilities).
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # -------- Optimizer + LR schedule --------
    total_steps = len(train_loader) * cfg.epochs
    optimizer = build_optimizer(model, cfg.learning_rate, cfg.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    # -------- Logging --------
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))
    best_ckpt_dir = output_dir / "best_model"

    # -------- Early stopping --------
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        mode="min",                            # monitor val_loss
        min_delta=cfg.early_stopping_min_delta,
    )
    best_metrics: Dict[str, float] = {}
    best_epoch = -1
    history: list = []

    # -------- Epoch loop --------
    for epoch in range(cfg.epochs):
        t0 = time.time()
        logger.info("==== Epoch %d/%d ====", epoch + 1, cfg.epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            grad_clip_norm=cfg.grad_clip_norm,
            mixed_precision=cfg.mixed_precision,
            epoch=epoch, writer=writer,
            log_interval_steps=cfg.log_interval_steps,
        )
        val_metrics = validate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        logger.info(
            "  train_loss=%.4f | val_loss=%.4f | val_f1_macro=%.4f | "
            "val_precision=%.4f | val_recall=%.4f | %.1fs",
            train_metrics["loss"],
            val_metrics["val_loss"],
            val_metrics["val_f1_macro"],
            val_metrics["val_precision_macro"],
            val_metrics["val_recall_macro"],
            elapsed,
        )

        # TensorBoard epoch-level scalars.
        writer.add_scalar("train/epoch_loss", train_metrics["loss"], epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            **val_metrics,
            "elapsed_sec": elapsed,
        }
        history.append(epoch_record)

        # Best-checkpoint logic — improvement on validation loss.
        improved = early_stopper.step(val_metrics["val_loss"])
        if improved:
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            model.save_pretrained(str(best_ckpt_dir))
            # Persist the fitted TabularPreprocessor next to the model so the
            # demo app can reproduce inference-time feature encoding exactly.
            if tab_preprocessor is not None:
                try:
                    tab_preprocessor.save(best_ckpt_dir / "tab_preprocessor.joblib")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to save tab_preprocessor: %s", e)
            logger.info("  New best (val_loss=%.4f) — saved to %s",
                        val_metrics["val_loss"], best_ckpt_dir)

        if early_stopper.should_stop:
            logger.info("Early stopping triggered (patience=%d).",
                        cfg.early_stopping_patience)
            break

    writer.close()

    # -------- Reload best model so the returned object IS the best --------
    if best_ckpt_dir.exists():
        model = LateFusionModel.from_pretrained(str(best_ckpt_dir)).to(device)

    return model, {
        "best_metrics": best_metrics,
        "best_epoch":   best_epoch,
        "history":      history,
        "checkpoint":   str(best_ckpt_dir),
    }


# =========================================================================== #
# CLI
# =========================================================================== #
def _load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML config into a Python dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for ``python -m src.train``."""
    p = argparse.ArgumentParser(description="Train the LateFusion model.")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--train-data", type=str, default="data/processed/train.parquet")
    p.add_argument("--val-data",   type=str, default="data/processed/val.parquet")
    p.add_argument("--output-dir", type=str, default="models")
    return p.parse_args()


def main() -> None:
    """Load YAML config, build datasets, call :func:`train_model`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    # Lazy imports — keep `from src.train import train_model` cheap.
    import pandas as pd
    from transformers import AutoTokenizer

    from src.preprocessing import TabularPreprocessor

    cfg_yaml = _load_yaml(args.config)
    class_names = cfg_yaml["data"]["class_names"]
    label_to_id = {n: i for i, n in enumerate(class_names)}

    # ---- Build configs from YAML ----
    model_config = LateFusionConfig(
        text_model_name=cfg_yaml["text"]["model_name"],
        text_pooling=cfg_yaml["text"]["pooling"],
        freeze_text_encoder=cfg_yaml["text"]["freeze_encoder"],
        n_num_features=len(cfg_yaml["data"].get("numerical_cols", [])),
        cat_cardinalities=[],   # filled in after fitting the preprocessor
        ft_d_token=cfg_yaml["tabular"]["d_token"],
        ft_n_blocks=cfg_yaml["tabular"]["n_blocks"],
        ft_attention_n_heads=cfg_yaml["tabular"]["attention_n_heads"],
        ft_ffn_d_hidden=cfg_yaml["tabular"]["ffn_d_hidden"],
        ft_dropout=cfg_yaml["tabular"]["dropout"],
        fusion_hidden_dim=cfg_yaml["fusion"]["hidden_dim"],
        fusion_dropout=cfg_yaml["fusion"]["dropout"],
        n_classes=cfg_yaml["fusion"]["n_classes"],
    )
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        **cfg_yaml["training"],
    )

    # ---- Datasets ----
    train_df = pd.read_parquet(args.train_data)
    val_df   = pd.read_parquet(args.val_data)
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)

    num_cols = cfg_yaml["data"].get("numerical_cols", [])
    cat_cols = cfg_yaml["data"].get("categorical_cols", [])
    # Filter to columns that actually exist (the crawled-only dataset has none).
    num_cols = [c for c in num_cols if c in train_df.columns]
    cat_cols = [c for c in cat_cols if c in train_df.columns]
    tab_pp = TabularPreprocessor(
        numerical_cols=num_cols, categorical_cols=cat_cols,
    ).fit(train_df)
    model_config.n_num_features = tab_pp.n_num_features
    model_config.cat_cardinalities = tab_pp.cat_cardinalities

    train_ds = SocialSentimentDataset(
        train_df, "text", "label", tokenizer, tab_pp,
        teencode_normalizer=None, label_to_id=label_to_id,
        max_length=cfg_yaml["text"]["max_length"],
    )
    val_ds = SocialSentimentDataset(
        val_df, "text", "label", tokenizer, tab_pp,
        teencode_normalizer=None, label_to_id=label_to_id,
        max_length=cfg_yaml["text"]["max_length"],
    )

    model, hist = train_model(
        train_ds, val_ds, model_config, training_config,
        tab_preprocessor=tab_pp,
    )
    logger.info("Best metrics: %s", hist["best_metrics"])
    logger.info("Best checkpoint: %s", hist["checkpoint"])


if __name__ == "__main__":
    main()
