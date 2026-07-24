"""SAP — Stack-and-Scale Asynchronous Pretraining.

Structural merging of independently pretrained decoder-only transformers:
attention heads and FFN neurons from N parent models are concatenated into
one wider model whose every sublayer computes the exact (optionally
alpha-weighted) combination of the parents' sublayer functions.
"""

__version__ = "0.1.0"
