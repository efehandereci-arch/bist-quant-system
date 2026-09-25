"""Triple Barrier Method ve meta-etiketler (AFML Bölüm 3).

Sabit ufuklu etiketleme (ör. "5 gün sonraki getiri > 0") iki temel kusur taşır:
(1) volatiliteden bağımsız sabit eşik kullanır, (2) yol bağımlılığını yok sayar:
pozisyon vadeden önce stop-loss'a takılıp kapanmış olabilir. Triple Barrier
yöntemi işlemin gerçekte nasıl yönetileceğini taklit eder:

    Üst bariyer (pt) : kâr al   -> fiyat giriş * exp(+pt x trgt) seviyesine ulaşırsa
    Alt bariyer (sl) : zarar kes -> fiyat giriş * exp(-sl x trgt) seviyesine düşerse
    Dikey bariyer    : zaman aşımı -> max_holding_bars bar sonra pozisyon kapanır

Burada ``trgt`` olay anındaki dinamik volatilitedir. İlk dokunulan bariyer
işlemin sonucunu ve bitiş zamanını (``t1``) belirler. Yön (side) bilindiği
için bariyerler pozisyona göre yorumlanır: short pozisyonda fiyatın düşmesi
üst (kâr) bariyerine dokunmak demektir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_vertical_barriers(
    t_events: pd.DatetimeIndex, index: pd.DatetimeIndex, num_bars: int
) -> pd.Series:
    """Her olay için ``num_bars`` bar sonrasındaki zaman damgasını (dikey bariyer) döndürür.

    Dikey bariyeri veri sonunun ötesine düşen olaylar atılır: etiketleri henüz
    gözlenmemiştir ve kesik bir yol üzerinden etiketlemek sonucu yanlı kılar.
    """
    if num_bars < 1:
        raise ValueError("num_bars >= 1 olmalı")
    pos = index.get_indexer(t_events)
    if (pos < 0).any():
        raise KeyError("Bazı olay zamanları fiyat indeksinde yok")
    end = pos + num_bars
    valid = end < len(index)
    return pd.Series(index[end[valid]], index=t_events[valid], name="vertical")


def apply_triple_barrier(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    target: pd.Series,
    side: pd.Series,
    vertical_barriers: pd.Series,
    pt_mult: float,
    sl_mult: float,
    min_target: float = 0.0,
) -> pd.DataFrame:
    """Olaylara üç bariyeri uygular ve ilk temas bilgisini döndürür (AFML Snippet 3.2/3.6).

    Args:
        close: Kapanış fiyatları (bariyer temasları kapanış yolu üzerinden kontrol edilir).
        t_events: Olay başlangıç zamanları (t0), işleme t0 kapanışında girilir.
        target: Bar bazında volatilite (bariyer genişliğinin birimi).
        side: Birincil model yönü (+1/-1); 0 olan olaylar atılır.
        vertical_barriers: :func:`get_vertical_barriers` çıktısı.
        pt_mult, sl_mult: Kâr al / zarar kes katsayıları (0 -> ilgili bariyer devre dışı).
        min_target: Bu değerin altındaki volatiliteye sahip olaylar atılır
            (maliyetleri karşılayamayacak kadar sakin anlar).

    Returns:
        t0 indeksli DataFrame:
            t1        : ilk temas zamanı (işlemin kapandığı bar)
            vertical  : dikey bariyer zamanı
            trgt      : olay anındaki volatilite
            side      : pozisyon yönü
            barrier   : "pt", "sl" veya "vertical"
            ret       : yöne göre düzeltilmiş log getiri, side * ln(P_t1 / P_t0)
            gross_ret : yöne göre düzeltilmiş basit getiri, side * (P_t1 / P_t0 - 1)
    """
    events = pd.DataFrame(
        {
            "vertical": vertical_barriers,
            "trgt": target.reindex(t_events),
            "side": side.reindex(t_events),
        }
    ).reindex(t_events)
    events = events.dropna()
    events = events[(events["trgt"] > min_target) & (events["side"] != 0)]
    if events.empty:
        raise ValueError("Bariyer uygulanacak geçerli olay kalmadı")

    index = close.index
    log_px = np.log(close.to_numpy(dtype=float))
    t0_pos = index.get_indexer(events.index)
    t1_pos = index.get_indexer(pd.DatetimeIndex(events["vertical"]))
    trgt = events["trgt"].to_numpy(dtype=float)
    sides = events["side"].to_numpy(dtype=float)

    touch_pos = np.empty(len(events), dtype=np.int64)
    barrier = np.empty(len(events), dtype=object)
    for k in range(len(events)):
        i0, i1 = t0_pos[k], t1_pos[k]
        # Girişten sonraki kapanışlar boyunca yöne göre kümülatif log getiri
        path = (log_px[i0 + 1 : i1 + 1] - log_px[i0]) * sides[k]
        upper = pt_mult * trgt[k] if pt_mult > 0 else np.inf
        lower = -sl_mult * trgt[k] if sl_mult > 0 else -np.inf
        hit_pt = np.flatnonzero(path >= upper)
        hit_sl = np.flatnonzero(path <= lower)
        first_pt = hit_pt[0] if hit_pt.size else np.inf
        first_sl = hit_sl[0] if hit_sl.size else np.inf
        if first_pt == np.inf and first_sl == np.inf:
            offset, barrier[k] = len(path) - 1, "vertical"
        elif first_pt <= first_sl:
            offset, barrier[k] = int(first_pt), "pt"
        else:
            offset, barrier[k] = int(first_sl), "sl"
        touch_pos[k] = i0 + 1 + offset

    price_ratio = np.exp(log_px[touch_pos] - log_px[t0_pos])
    out = events.assign(
        t1=index[touch_pos],
        barrier=barrier,
        ret=sides * np.log(price_ratio),
        gross_ret=sides * (price_ratio - 1.0),
    )
    out["side"] = out["side"].astype(int)
    out.index.name = "t0"
    return out[["t1", "vertical", "trgt", "side", "barrier", "ret", "gross_ret"]]


def get_meta_labels(events: pd.DataFrame, cost_per_side: float = 0.0) -> pd.DataFrame:
    """Birincil yön ile Triple Barrier sonucunu karşılaştırarak ikili meta-etiket üretir.

    Meta-etiketleme (AFML 3.6, Snippet 3.7): Yön birincil modelden geldiği için
    etiket artık "fiyat yukarı mı aşağı mı?" değil, "birincil modelin önerdiği
    işlem kârlı mı?" sorusunun cevabıdır:

        Y = 1  <=>  side * (P_t1 / P_t0 - 1) - 2 * maliyet > 0   (işlem kâr etti)
        Y = 0  <=>  aksi halde (zarar ya da maliyeti karşılamayan getiri)

    Dikey bariyere takılan işlemler de getirilerinin işaretine göre etiketlenir.
    """
    out = events.copy()
    out["net_ret"] = out["gross_ret"] - 2.0 * cost_per_side
    out["bin"] = (out["net_ret"] > 0).astype(int)
    return out
