"""Deep Social Sentiment Analysis — source package.

Late Fusion architecture combining:
    * Text branch:    XLM-R (with Teencode normalization)
    * Tabular branch: FT-Transformer (user behavior features)

Sub-modules
-----------
preprocessing : Text/tabular preprocessing utilities.
dataset       : PyTorch Dataset wrappers.
models        : Model definitions (TextBranch, TabularBranch, LateFusionModel).
train         : Training loop & checkpointing.
evaluate      : Metrics computation & evaluation utilities.
"""

__version__ = "0.1.0"
