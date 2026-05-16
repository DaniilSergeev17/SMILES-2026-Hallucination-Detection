"""
probe.py - Hallucination probe classifier.

'HallucinationProbe' is an ensemble of eleven small MLPs that read the
896-dim hidden-state vector produced by 'aggregation.py'. Each MLP has a
single hidden layer of 256 units with ReLU; the eleven nets are trained from
different random seeds and their sigmoid outputs are averaged at prediction
time, which cuts the per-fold variance.

Preprocessing is 'RobustScaler' (centre on the median, scale by the IQR)
followed by clipping every scaled value to the range +/-5. The clipping is
the key step - intermediate-layer hidden states have a handful of "massive
activation" dimensions that dominate the variance even after standardisation
and prevent the MLP from learning. Clipping at +/-5 tames them without
losing information from the normal-range features.

'fit_hyperparameters' sweeps the decision threshold to maximise accuracy
(the competition metric). 'evaluate.py' calls it on the validation fold
during cross-validation; 'fit' calls it once at the end on the training
data so that the *submitted* probe - for which 'solution.py' never invokes
'fit_hyperparameters' separately - does not stay at the default 0.5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler


# Hyperparameters of the final recipe:
_HIDDEN = 256 # MLP hidden size
_EPOCHS = 200
_LR = 1e-3
_N_NETS = 11 # MLPs averaged in the ensemble
_CLIP = 5.0 # clip RobustScaler-standardized features to +/- this
_SEED = 42


def _build_net(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, _HIDDEN),
        nn.ReLU(),
        nn.Linear(_HIDDEN, 1),
    )


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Public API expected by ``evaluate.py`` / ``solution.py``:
        ``fit(X, y)``, ``fit_hyperparameters(X_val, y_val)``,
        ``predict(X)``, ``predict_proba(X)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = RobustScaler()
        self._nets: nn.ModuleList = nn.ModuleList()
        self._threshold: float = 0.5

    def _transform(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        """Apply RobustScaler + symmetric clip."""
        Xs = self._scaler.fit_transform(X) if fit else self._scaler.transform(X)
        Xs = np.clip(Xs, -_CLIP, _CLIP)
        return np.ascontiguousarray(Xs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._nets:
            raise RuntimeError("Call fit() before forward()")
        return self._nets[0](x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the eleven-MLP ensemble on labelled feature vectors.

        Args:
            X: Feature matrix of shape (n_samples, feature_dim).
            y: Integer label vector of shape (n_samples,);
               0 = truthful, 1 = hallucinated.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        Xt = self._transform(X, fit=True)
        X_t = torch.from_numpy(Xt).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        # Re-weight the positive class to counter class imbalance (70/30)
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._nets = nn.ModuleList()
        for i in range(_N_NETS):
            torch.manual_seed(_SEED + i)
            net = _build_net(X_t.shape[1])
            optimiser = torch.optim.Adam(net.parameters(), lr=_LR)
            net.train()
            for _ in range(_EPOCHS):
                optimiser.zero_grad()
                criterion(net(X_t).squeeze(-1), y_t).backward()
                optimiser.step()
            net.eval()
            self._nets.append(net)

        if np.unique(y).size > 1:
            self.fit_hyperparameters(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 2).

        Column 1 is the probability of the hallucinated class (label 1) -
        used by evaluate.py to compute AUROC.
        """
        X = np.asarray(X, dtype=np.float32)
        xt = torch.from_numpy(self._transform(X, fit=False)).float()
        with torch.no_grad():
            probs = np.mean(
                [torch.sigmoid(net(xt).squeeze(-1)).numpy() for net in self._nets],
                axis=0,
            )
        prob_pos = np.asarray(probs, dtype=np.float64).reshape(-1)
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 0/1 labels using the tuned decision threshold."""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold to maximise accuracy on a validation set.

        Sweeps every unique predicted probability plus a coarse 101-point grid
        on [0, 1] and keeps the threshold with the highest accuracy.
        """
        y_val = np.asarray(y_val).astype(int)
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 101)])
        )

        best_threshold, best_score = 0.5, -1.0
        for t in candidates:
            score = accuracy_score(y_val, (probs >= t).astype(int))
            if score > best_score:
                best_score, best_threshold = score, float(t)
        self._threshold = best_threshold
        return self
