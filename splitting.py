"""
splitting.py - Train / validation / test split utilities.

``split_data`` returns a list of ``(idx_train, idx_val, idx_test)`` integer
index tuples (one per fold). The shipped strategy is 5-fold
``StratifiedGroupKFold`` keyed on a hash of the **context paragraph** parsed
out of each prompt.

Why grouped k-fold:
- The dataset is SQuAD-style - a context paragraph, a question about it, the
  model's answer. Multiple questions reuse the same paragraph.
- A plain stratified random split therefore puts the same paragraph in train
  and test, and the probe learns paragraph-specific shortcuts. That inflates
  the cross-validated number and the gain does not transfer to a held-out
  set built from new paragraphs.
- Grouping by paragraph removes that leak. As a bonus, with k-fold every
  row ends up in some training fold, so ``solution.py``'s "fit the final
  probe on the union of train and validation indices across folds" step uses
  all 689 rows.

Inside each fold a 15% slice of the training groups is held out as a
validation set for threshold tuning. If that grouped slice would land all
of one class (rare, but possible on degenerate folds), the code falls back
to a plain stratified slice of the training indices.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` (it never is here, but the contract allows it).
* All indices within a fold are non-overlapping; together they cover every
  sample.
* The function returns a list with one tuple per fold.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

_N_SPLITS = 5
_VAL_FRACTION = 0.15
_SEED = 42


def _paragraph_key(prompt: str) -> str:
    """Hash the user-message body (without the question) of a ChatML prompt.

    The data uses a fixed ChatML template; the user turn contains boilerplate,
    the context paragraph and finally the question after "Here is the
    question:". Cutting at that marker gives a string that is the same for
    every question that shares a paragraph.
    """
    s = str(prompt)
    if "<|im_start|>user" in s:
        s = s.split("<|im_start|>user", 1)[1]
    for marker in ("Here is the question", "<|im_end|>"):
        if marker in s:
            s = s.split(marker, 1)[0]
            break
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return a list of ``(idx_train, idx_val, idx_test)`` tuples - one per fold.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           The full DataFrame, used to derive the paragraph group
                      key from the ``prompt`` column.
        test_size:    Kept for signature compatibility with the reference
                      implementation; not used by the grouped k-fold path.
        val_size:     Same.
        random_state: Same.

    Returns:
        A list of length 5 (``_N_SPLITS``) with one tuple per fold.
    """
    y = np.asarray(y)

    # Build the paragraph-level group labels
    if df is None or "prompt" not in df.columns:
        idx = np.arange(len(y))
        idx_tv, idx_te = train_test_split(
            idx, test_size=test_size, random_state=_SEED, stratify=y,
        )
        rel = val_size / (1.0 - test_size)
        idx_tr, idx_va = train_test_split(
            idx_tv, test_size=rel, random_state=_SEED, stratify=y[idx_tv],
        )
        return [(idx_tr, idx_va, idx_te)]

    groups = df["prompt"].astype(str).map(_paragraph_key).to_numpy()
    sgkf = StratifiedGroupKFold(
        n_splits=_N_SPLITS, shuffle=True, random_state=_SEED,
    )

    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for tr, te in sgkf.split(np.zeros(len(y)), y, groups):
        # Carve a group-disjoint approximately 15% validation slice out of the training
        # groups so that paragraphs don't leak from train into val either.
        tr_groups = np.array(sorted(set(groups[tr])))
        rng = np.random.default_rng(_SEED + len(folds))
        rng.shuffle(tr_groups)
        n_val_groups = max(1, int(round(_VAL_FRACTION * len(tr_groups))))
        val_groups = set(tr_groups[:n_val_groups].tolist())
        is_val = np.array([g in val_groups for g in groups[tr]])
        idx_val = tr[is_val]
        idx_train = tr[~is_val]

        # Fall back to a plain stratified slice if the grouped slice ends up all one-class
        if (
            idx_val.size == 0
            or np.unique(y[idx_val]).size < 2
            or np.unique(y[idx_train]).size < 2
        ):
            idx_train, idx_val = train_test_split(
                tr, test_size=_VAL_FRACTION, random_state=_SEED, stratify=y[tr],
            )

        folds.append((idx_train, idx_val, te))
    return folds
