"""TOMLTransformers: transistor-level energy modeling of transformer inference.

Extends the TOML (Transistor Operations for Machine Learning) framework from
CNNs/RNNs/GBDTs to transformer architectures: encoder-only, decoder-only, and
encoder-decoder, with standard vs FlashAttention accounting, KV-cache memory
modeling, precision awareness, and Mixture-of-Experts.

All prior TOML work models inference energy only; this package does the same.
Training energy is the scope of a separate planned work (TOMLtraining).
"""

__version__ = "0.1.0"
