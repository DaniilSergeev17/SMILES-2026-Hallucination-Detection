"""
aggregation.py - Token aggregation strategy and feature extraction.

Converts per-token, per-layer hidden states from the extraction loop in
solution.py into a flat feature vector for the probe classifier.

The chosen recipe is simple: return the hidden state at the **last real
token** of **layer 14** of Qwen2.5-0.5B.  No geometric features, no
multi-layer concatenation, no mean-pooling.  That's the whole feature vector,
896 numbers.

Why layer 14 and not the final layer:
- The probing literature consistently finds the cleanest truthfulness signal
  in middle layers; the very last layer is over-specialised for next-token
  prediction.
- A short sweep over layers (on a paragraph-grouped split, so the numbers are
  honest) put layers 14 and 20 on top; 14 won by a hair.
"""

from __future__ import annotations

import torch

# Index into the 25-element hidden_states tuple returned by Qwen2.5-0.5B.
# 0 is the embedding output; 1..24 are the 24 transformer layers.
_LAYER = 14


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the hidden state at the last real token of layer 14.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(hidden_dim,)``.
    """
    attention_mask = attention_mask.to(hidden_states.device)
    last_pos = int(attention_mask.nonzero(as_tuple=False)[-1].item())
    return hidden_states[_LAYER, last_pos, :].float()


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Geometric feature hook - left empty.

    A variety of hand-crafted geometric features were tried during development
    (per-layer activation norms, inter-layer cosine drift, an EigenScore-style
    covariance spectrum, an intrinsic-dimension proxy).  None of them improved
    the cross-validated accuracy on the honest grouped split, so the shipped
    version returns nothing.  'USE_GEOMETRIC' in 'solution.py' is also
    hardcoded to 'False', so this function is not called in the submission
    pipeline.
    """
    return torch.zeros(0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Main entry point called from ``solution.py`` for each sample."""
    features = aggregate(hidden_states, attention_mask)
    if use_geometric:
        geo = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([features, geo.to(features.device)], dim=0)
    return features
