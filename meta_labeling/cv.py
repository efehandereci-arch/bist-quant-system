"""Purged & Embargoed zaman serisi çapraz doğrulaması (AFML Bölüm 7).

Standart K-Fold neden finansta başarısız olur?
--------------------------------------------
1. Gözlemler IID değildir. Triple Barrier etiketi y_i, [t0_i, t1_i] aralığındaki
   fiyat yoluna bağlıdır. Test setindeki bir etiketle zaman aralığı örtüşen bir
   eğitim etiketi aynı getirileri "görür": model test sonucunu dolaylı olarak
   ezberler (bilgi sızıntısı / leakage).
2. Öznitelikler seri korelasyonludur. Test setinin hemen ardından gelen eğitim
   gözlemlerinin öznitelikleri (ör. 50 barlık rolling pencereler) test
   dönemindeki fiyatları içerir.

Çözüm
-----
* PURGING: Etiket aralığı [t0_i, t1_i] test setinin kapsadığı
  [min t0_test, max t1_test] aralığıyla kesişen her eğitim gözlemi atılır.
* EMBARGO: Test setinin bitişini takip eden h süre (toplam sürenin
  ``embargo_pct`` kadarı) içinde başlayan eğitim gözlemleri de atılır.
  Purging test *öncesi* ve test *içi* örtüşmeyi temizler; embargo test
  *sonrası* seri korelasyon kanalını kapatır.

Bu modül iki kullanım sunar:
* ``walk_forward=False``: Prado'nun PurgedKFold'u. Eğitim test öncesi ve
  sonrasındaki (purge + embargo uygulanmış) verileri kullanır. Model teşhisi,
  öznitelik önemi ve hiperparametre seçimi için.
* ``walk_forward=True``: Yalnızca test setinden ÖNCEKİ (purge edilmiş) veriyle
  eğitim. Her tahmin gerçek zamanlı olarak üretilebilecek bir tahmindir;
  backtest için kullanılan OOS olasılıkları buradan gelir.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import positive_class_proba


class PurgedKFold:
    """Purging ve embargo uygulayan zaman sıralı K-Fold (scikit-learn uyumlu ``split`` arayüzü).

    Test fold'ları kronolojik ve bitişiktir (karıştırma YOK).

    Args:
        t1: İndeksi olay başlangıcı (t0), değeri etiket bitişi (t1, ilk bariyer teması)
            olan seri. Kronolojik sıralı olmalı ve X ile aynı indekse sahip olmalı.
        n_splits: Fold sayısı.
        embargo_pct: Embargo süresi, olayların toplam zaman aralığının bu oranı kadardır.
        walk_forward: True ise eğitim seti yalnızca test öncesi gözlemlerden oluşur.
    """

    def __init__(
        self,
        t1: pd.Series,
        n_splits: int = 6,
        embargo_pct: float = 0.01,
        walk_forward: bool = False,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits >= 2 olmalı")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError("embargo_pct [0, 1) aralığında olmalı")
        if not t1.index.is_monotonic_increasing or not t1.index.is_unique:
            raise ValueError("t1 indeksi (t0) tekil ve kronolojik sıralı olmalı")
        if (pd.DatetimeIndex(t1.to_numpy()) < t1.index).any():
            raise ValueError("Her olay için t1 >= t0 olmalı")
        if len(t1) < n_splits:
            raise ValueError("Olay sayısı n_splits'ten küçük olamaz")
        self.t1 = t1
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.walk_forward = walk_forward

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X=None, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if X is not None:
            if len(X) != len(self.t1):
                raise ValueError("X ve t1 aynı uzunlukta olmalı")
            if isinstance(X, (pd.DataFrame, pd.Series)) and not X.index.equals(self.t1.index):
                raise ValueError("X indeksi t1 indeksiyle (t0) aynı olmalı")

        t0 = self.t1.index
        t_end = pd.DatetimeIndex(self.t1.to_numpy())
        embargo = (t0[-1] - t0[0]) * self.embargo_pct
        indices = np.arange(len(self.t1))

        for test_idx in np.array_split(indices, self.n_splits):
            test_start = t0[test_idx[0]]
            test_end = t_end[test_idx].max()
            # PURGE: etiket aralığı test aralığıyla kesişen eğitim gözlemleri
            overlaps = np.asarray((t0 <= test_end) & (t_end >= test_start))
            # EMBARGO: test bitişinden sonraki h süre içinde başlayan gözlemler
            embargoed = np.asarray((t0 > test_end) & (t0 <= test_end + embargo))
            train_mask = ~(overlaps | embargoed)
            if self.walk_forward:
                train_mask &= indices < test_idx[0]
            yield indices[train_mask], test_idx


def _fold_metrics(
    y_true: np.ndarray, proba: np.ndarray, weight: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (proba > threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "auc": roc_auc_score(y_true, proba, sample_weight=weight) if two_classes else np.nan,
        "log_loss": log_loss(y_true, proba, sample_weight=weight, labels=[0, 1]),
        "accuracy": accuracy_score(y_true, pred, sample_weight=weight),
        "precision": precision_score(y_true, pred, sample_weight=weight, zero_division=0),
        "recall": recall_score(y_true, pred, sample_weight=weight, zero_division=0),
        "f1": f1_score(y_true, pred, sample_weight=weight, zero_division=0),
        "base_rate": float(np.average(y_true, weights=weight)),
    }


def cross_validate_purged(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedKFold,
    sample_weight: pd.Series | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Purged K-Fold ile fold bazında skorlar (AFML Snippet 7.4 ``cvScore`` karşılığı).

    Prado'nun uyardığı iki scikit-learn tuzağından kaçınılır:
    (1) örnek ağırlıkları hem ``fit``'e hem de skorlamaya aktarılır;
    (2) log-loss, ``classes_`` sıralamasından bağımsız olarak pozitif sınıf
    olasılığıyla hesaplanır.
    """
    w = np.ones(len(y)) if sample_weight is None else sample_weight.to_numpy(dtype=float)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        y_train = y.iloc[train_idx]
        if len(train_idx) == 0 or y_train.nunique() < 2:
            continue
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y_train, sample_weight=w[train_idx])
        proba = positive_class_proba(model, X.iloc[test_idx])
        metrics = _fold_metrics(y.iloc[test_idx].to_numpy(), proba, w[test_idx], threshold)
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_start": X.index[test_idx[0]].date(),
                "test_end": X.index[test_idx[-1]].date(),
                **metrics,
            }
        )
    if not rows:
        raise ValueError("Hiçbir fold'da iki sınıfı da içeren eğitim seti oluşmadı")
    return pd.DataFrame(rows).set_index("fold")


def walk_forward_predict(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedKFold,
    sample_weight: pd.Series | None = None,
    min_train_events: int = 100,
) -> tuple[pd.Series, list]:
    """Purged walk-forward ile örneklem dışı (OOS) P(Y=1) tahminleri.

    Her test fold'u için model yalnızca o fold'dan önce *tamamen kapanmış*
    etiketlerle eğitilir. Yeterli eğitim verisi olmayan ilk fold(lar) için
    tahmin üretilmez (NaN).

    Returns:
        (oos_proba, fitted_models)
    """
    if not cv.walk_forward:
        raise ValueError("walk_forward_predict, walk_forward=True olan bir PurgedKFold ister")
    w = np.ones(len(y)) if sample_weight is None else sample_weight.to_numpy(dtype=float)
    oos = pd.Series(np.nan, index=X.index, name="proba")
    models = []
    for train_idx, test_idx in cv.split(X):
        y_train = y.iloc[train_idx]
        if len(train_idx) < min_train_events or y_train.nunique() < 2:
            continue
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y_train, sample_weight=w[train_idx])
        oos.iloc[test_idx] = positive_class_proba(model, X.iloc[test_idx])
        models.append(model)
    return oos, models
