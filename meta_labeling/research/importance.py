"""OOS-güvenli öznitelik önemi.

* MDI  : Walk-forward modellerinin kendi eğitim verisindeki (in-sample) ağaç
  önemi. Örneklem içi ve ikame etkilerine duyarlıdır; yalnızca referans.
* Permutation (MDA): Her walk-forward modeli YALNIZCA kendi (geçmişte
  eğitildiği) OOS test fold'unda değerlendirilir; bir öznitelik karıştırılınca
  AUC'deki düşüş ölçülür. Test verisi eğitimde hiç görülmemiştir.
* SHAP : LightGBM'in yerleşik TreeSHAP'ı (``pred_contrib=True``) ile OOS test
  fold'larında ortalama |SHAP| değeri; ek bir ``shap`` bağımlılığı gerekmez.
  RandomForest seçiliyse ``shap`` paketi kuruluysa o kullanılır, yoksa atlanır.

UYARI: Bu sonuçlara bakarak öznitelik seçip modeli yeniden eğitmek, OOS
verisini model seçimine sızdırır. Bu modül yalnızca raporlama içindir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..model import positive_class_proba


def mdi_importance(models: list, columns: pd.Index) -> pd.Series:
    imps = []
    for _, m in models:
        imp = np.asarray(m.feature_importances_, dtype=float)
        imps.append(imp / imp.sum() if imp.sum() > 0 else imp)
    return pd.Series(np.mean(imps, axis=0), index=columns, name="MDI") if imps else pd.Series(dtype=float)


def permutation_importance_oos(
    models: list, splits: list, X: pd.DataFrame, y: pd.Series, repeats: int, rng: np.random.Generator
) -> pd.Series:
    """Fold bazında OOS AUC düşüşü (model yalnızca geçmişte eğitildi)."""
    test_by_fold = {k: te for k, _, te in splits}
    drops: dict[str, list[float]] = {c: [] for c in X.columns}
    for k, model in models:
        te = test_by_fold[k]
        Xt, yt = X.iloc[te], y.iloc[te]
        if yt.nunique() < 2:
            continue
        base = roc_auc_score(yt, positive_class_proba(model, Xt))
        for col in X.columns:
            for _ in range(repeats):
                Xp = Xt.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                drops[col].append(base - roc_auc_score(yt, positive_class_proba(model, Xp)))
    return pd.Series({c: float(np.mean(v)) if v else np.nan for c, v in drops.items()}, name="Permutation (ΔAUC)")


def shap_importance_oos(models: list, splits: list, X: pd.DataFrame) -> tuple[pd.Series, str]:
    test_by_fold = {k: te for k, _, te in splits}
    vals = []
    method = "none"
    for k, model in models:
        Xt = X.iloc[test_by_fold[k]]
        if hasattr(model, "booster_"):
            contrib = model.booster_.predict(Xt, pred_contrib=True)[:, :-1]
            method = "LightGBM TreeSHAP"
        else:
            try:
                import shap
            except ImportError:
                return pd.Series(np.nan, index=X.columns, name="SHAP"), "shap paketi yok (RF için gerekli)"
            sv = shap.TreeExplainer(model).shap_values(Xt)
            contrib = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
            method = "shap.TreeExplainer"
        vals.append(np.abs(contrib).mean(axis=0))
    if not vals:
        return pd.Series(np.nan, index=X.columns, name="SHAP"), method
    return pd.Series(np.mean(vals, axis=0), index=X.columns, name="SHAP |mean|"), method
