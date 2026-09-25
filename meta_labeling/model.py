"""İkincil (meta) model fabrikası.

Aşırı öğrenmeye karşı tasarım tercihleri:

* Sınıf dengesizliği: ``class_weight="balanced"`` (LightGBM) /
  ``"balanced_subsample"`` (RF). Aksi halde model çoğunluk sınıfını tahmin
  ederek yüksek doğruluk elde edebilir. Not: Dengeleme, olasılıkları sınıf
  oranları eşitmiş gibi ölçekler; bu yüzden P(Y=1) = 0.5 "birincil model kadar
  iyi" anlamına gelmez, eşik (ör. 0.55) dengelenmiş ölçekte yorumlanmalıdır.
* Benzersizlik: Örtüşen etiketlerde her ağacın alt örneklem oranı ortalama
  benzersizliğe eşitlenir (AFML 4.5 / 6.4). Böylece her ağaç birbirinin
  kopyası olan gözlemlerle aşırı uyum sağlamaz.
* Düşük karmaşıklık: Sığ ağaçlar, yaprak başına minimum örnek, L2
  düzenlileştirme, öznitelik alt örneklemesi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import MetaModelConfig


def build_meta_model(cfg: MetaModelConfig, avg_uniqueness: float = 1.0):
    """Konfigürasyona göre eğitilmemiş bir sınıflandırıcı döndürür."""
    subsample = float(np.clip(avg_uniqueness, 0.1, 1.0))
    if cfg.kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - ortam bağımlı
            raise ImportError("lightgbm kurulu değil: pip install lightgbm (ya da kind='rf')") from exc
        return LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            max_depth=cfg.max_depth,
            min_child_samples=cfg.min_child_samples,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            class_weight="balanced",
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
            verbose=-1,
        )
    if cfg.kind == "rf":
        # Prado (AFML 6.4): max_samples = ortalama benzersizlik, entropy kriteri,
        # min_weight_fraction_leaf ile erken durdurma ve balanced_subsample.
        # Öznitelik önemi analizinde maskeleme etkisini azaltmak için
        # max_features=1 de tercih edilebilir (AFML 8.3.1).
        return RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators,
            criterion="entropy",
            max_features="sqrt",
            min_weight_fraction_leaf=cfg.rf_min_weight_fraction_leaf,
            class_weight="balanced_subsample",
            bootstrap=True,
            max_samples=subsample,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
        )
    raise ValueError(f"Bilinmeyen meta-model türü: {cfg.kind}")


def positive_class_proba(model, X: pd.DataFrame) -> np.ndarray:
    """P(Y=1). ``classes_`` sırasına güvenmek yerine pozitif sınıfın sütunu aranır."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def feature_importance(models: list, columns: pd.Index) -> pd.Series:
    """Modeller üzerinden ortalama, normalize edilmiş MDI önemi.

    Uyarı (AFML 8.3): MDI örneklem içi bir ölçüdür ve ikame etkilerinden
    etkilenir; araştırma amaçlı yorumlanmalı, backtest sonucu gibi
    kullanılmamalıdır. Daha güvenilir alternatif, purged CV ile MDA'dır.
    """
    if not models:
        return pd.Series(dtype=float)
    imps = []
    for m in models:
        imp = np.asarray(m.feature_importances_, dtype=float)
        imps.append(imp / imp.sum() if imp.sum() > 0 else imp)
    return pd.Series(np.mean(imps, axis=0), index=columns, name="importance").sort_values(
        ascending=False
    )
