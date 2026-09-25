"""Combinatorial Purged Cross-Validation (AFML Bölüm 12).

Walk-forward tek bir tarihsel yol üretir; bu yolun Sharpe'ı tek bir
gerçekleşmedir ve kolayca aşırı uydurulabilir. CPCV, olayları N gruba böler
ve her seferinde k grubu test olarak ayırarak C(N, k) eğitim/test bölmesi
oluşturur. Her grup C(N-1, k-1) bölmede test edilir; bu tahminler birleştirilerek
φ = C(N-1, k-1) adet TAM backtest yolu elde edilir. Böylece tek bir Sharpe
yerine bir Sharpe DAĞILIMI raporlanır.

Her bölmede purging (test gruplarının etiket aralıklarıyla örtüşen eğitim
olaylarının atılması) ve her test grubunun sonrasına embargo uygulanır.
Not: CPCV, walk-forward'ın aksine test grubundan SONRAKİ verilerle de eğitim
yapar (purge + embargo ile); bu bir "gerçek zamanlı" simülasyon değil,
yolların dağılımını tahmin etmeye yönelik bir tekniktir.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


class CombinatorialPurgedKFold:
    def __init__(self, t1: pd.Series, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01):
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("1 <= n_test_groups < n_groups olmalı")
        if not t1.index.is_monotonic_increasing:
            raise ValueError("t1 indeksi kronolojik olmalı")
        self.t1 = t1
        self.n_groups = n_groups
        self.k = n_test_groups
        self.embargo_pct = embargo_pct
        self.groups = np.array_split(np.arange(len(t1)), n_groups)

    @property
    def n_paths(self) -> int:
        return comb(self.n_groups - 1, self.k - 1)

    def split(self) -> Iterator[tuple[tuple[int, ...], np.ndarray, np.ndarray]]:
        t0 = self.t1.index
        t_end = pd.DatetimeIndex(self.t1.to_numpy())
        embargo = (t0[-1] - t0[0]) * self.embargo_pct
        n = len(self.t1)
        for combo in combinations(range(self.n_groups), self.k):
            train = np.ones(n, dtype=bool)
            for g in combo:
                idx = self.groups[g]
                start, end = t0[idx[0]], t_end[idx].max()
                overlap = np.asarray((t0 <= end) & (t_end >= start))
                embargoed = np.asarray((t0 > end) & (t0 <= end + embargo))
                train &= ~(overlap | embargoed)
            test = np.concatenate([self.groups[g] for g in combo])
            yield combo, np.flatnonzero(train), test

    def paths(self, combos: list[tuple[int, ...]]) -> list[dict[int, int]]:
        """Her yol için {grup: bölme_sırası}; grup g'nin p. test edildiği bölme p. yola atanır."""
        by_group: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for s, combo in enumerate(combos):
            for g in combo:
                by_group[g].append(s)
        return [{g: by_group[g][p] for g in range(self.n_groups)} for p in range(self.n_paths)]
