from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TERNARY_LABELS = [-1, 0, 1]


@dataclass
class MajorityClassPredictor:
    """
    Majority-class classifier.

    This model learns only one thing from the training labels:
        the most common class.

    It ignores all features.

    Tie-breaking rule:
        if two or more classes are tied, choose the first according to
        TERNARY_LABELS = [-1, 0, 1].
    """
    majority_class_: int | None = None
    class_counts_: dict[int, int] | None = None
    class_proportions_: dict[int, float] | None = None

    def fit(self, y_train: Iterable[int]) -> "MajorityClassPredictor":
        y = pd.Series(y_train)

        if y.empty:
            raise ValueError("Cannot fit majority baseline on empty y_train.")

        if y.isna().any():
            raise ValueError("y_train contains missing labels.")

        observed = set(y.unique())
        if not observed.issubset(set(TERNARY_LABELS)):
            raise ValueError(f"Unexpected labels in y_train: {observed}")
        
        counts = y.value_counts().reindex(TERNARY_LABELS, fill_value=0)

        n = int(counts.sum())

        self.majority_class_ = int(counts.idxmax())
        self.class_counts_ = {int(k): int(v) for k, v in counts.items()}
        self.class_proportions_ = {int(k): float(v / n) for k, v in counts.items()}

        return self
    
    def predict(self, n: int) -> np.ndarray:
        if self.majority_class_ is None:
            raise ValueError("MajorityClassPredictor must be fitted before prediction.")
        
        if n < 0:
            raise ValueError("n must be non-negative.")
        
        return np.repeat(self.majority_class_, n).astype(int)