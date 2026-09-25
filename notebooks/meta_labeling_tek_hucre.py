# ============================================================
# META-LABELING + TRIPLE BARRIER ML PIPELINE (AFML / López de Prado)
# Tek hücre: Kopyala -> Jupyter hücresine yapıştır -> Çalıştır
# Gereksinimler: %pip install numpy pandas scikit-learn lightgbm
#
# Bu dosya scripts/build_notebook.py tarafından meta_labeling/ paketinden
# otomatik üretilmiştir; elle düzenlemeyin.
# ============================================================


from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 1. KONFIGÜRASYON
# ============================================================
# Pipeline konfigürasyonu.
#
# Tüm hiperparametreler tek bir yerde, değiştirilemez (frozen) dataclass'lar
# olarak tutulur.
#
# Prado'nun "backtest overfitting" uyarısı (AFML Bölüm 11-12): Aynı veri üzerinde
# denenen her ek konfigürasyon (farklı eşik, farklı bariyer katsayısı, farklı
# model), bulunan "en iyi" stratejinin Sharpe oranının şans eseri yüksek çıkma
# olasılığını artırır. Bu nedenle parametreler burada *ex-ante* sabitlenir ve
# özellikle karar eşiği (``MetaModelConfig.threshold``) test verisine bakılarak
# optimize EDİLMEZ.

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SimulationConfig:
    """Sentetik OHLCV üreticisinin parametreleri.

    Üretilen seri iki gizli rejim arasında geçiş yapan (Markov zinciri) bir
    süreçtir: *trend* rejiminde getiriler kalıcı bir drift ve pozitif
    otokorelasyon taşır, *ortalamaya dönüş* (mean-reversion) rejiminde fiyat bir
    çapa seviyesine geri çekilir. Volatilite GARCH(1,1), şoklar Student-t
    dağılımlıdır (kalın kuyruk). Böylece trend takipçisi bir birincil model bazı
    rejimlerde kazanır, bazılarında kaybeder; meta-modelin öğrenebileceği
    gerçekçi bir yapı oluşur.
    """

    n_bars: int = 4000
    start: str = "2010-01-04"
    s0: float = 100.0
    seed: int = 42
    regime_persistence: float = 0.99   # P(rejim_{t+1} = rejim_t) -> ort. süre ~100 bar
    trend_drift: float = 0.0015        # trend rejiminde günlük log drift (yönü rastgele)
    trend_ar: float = 0.10             # trend rejiminde getiri AR(1) katsayısı
    mr_kappa: float = 0.08             # ortalamaya dönüş hızı (yarı ömür ~ ln2/kappa bar)
    long_run_vol: float = 0.015        # GARCH uzun dönem günlük volatilitesi
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    trend_vol_mult: float = 0.90       # rejime bağlı volatilite çarpanları
    mr_vol_mult: float = 1.10
    t_dof: float = 5.0                 # Student-t serbestlik derecesi


@dataclass(frozen=True)
class PrimaryConfig:
    """Birincil (yön) modeli ve olay örnekleme ayarları.

    ``event_mode``:
      * ``"cusum"``  : Olaylar simetrik CUSUM filtresiyle örneklenir (AFML 2.5.2.1),
        yön birincil modelden okunur. Yüksek recall sağlar: kayda değer her fiyat
        hareketinde bir pozisyon önerisi üretilir.
      * ``"signal"`` : Olay yalnızca birincil sinyalin yeni başladığı / yön
        değiştirdiği barlarda oluşur (klasik "giriş sinyali").
    """

    kind: Literal["ema", "bollinger"] = "ema"
    ema_fast: int = 10
    ema_slow: int = 40
    ema_neutral_band: float = 0.0      # |fast-slow|/slow bu değerin altındaysa 0 (nötr)
    bb_window: int = 20
    bb_num_std: float = 1.5            # düşük katsayı -> daha çok sinyal -> daha yüksek recall
    event_mode: Literal["cusum", "signal"] = "cusum"
    cusum_vol_mult: float = 1.0        # CUSUM eşiği = mult x dinamik volatilite


@dataclass(frozen=True)
class BarrierConfig:
    """Triple Barrier Method parametreleri (AFML Bölüm 3)."""

    vol_method: Literal["ewma", "rolling"] = "ewma"
    vol_span: int = 50                 # EWMA span'i ya da rolling pencere uzunluğu
    pt_mult: float = 2.0               # üst bariyer = pt_mult x volatilite (0 -> devre dışı)
    sl_mult: float = 2.0               # alt bariyer = sl_mult x volatilite (0 -> devre dışı)
    max_holding_bars: int = 10         # dikey bariyer (bar cinsinden)
    min_target: float = 0.002          # bu volatilitenin altındaki olaylar atılır


@dataclass(frozen=True)
class FeatureConfig:
    """Meta-model için rejim özniteliklerinin pencere uzunlukları."""

    rsi_window: int = 14
    atr_window: int = 14
    bb_window: int = 20
    bb_num_std: float = 2.0
    vol_fast_span: int = 10
    vol_slow_span: int = 50
    autocorr_window: int = 50
    efficiency_window: int = 20
    momentum_window: int = 20
    skew_window: int = 50
    volume_window: int = 20


@dataclass(frozen=True)
class CVConfig:
    """Purged & embargoed çapraz doğrulama ayarları (AFML Bölüm 7)."""

    n_splits: int = 6
    embargo_pct: float = 0.01          # toplam zaman aralığının %1'i kadar embargo
    min_train_events: int = 100        # walk-forward'da bundan az eğitim olayı varsa fold atlanır


@dataclass(frozen=True)
class MetaModelConfig:
    """İkincil (meta) model, bet sizing ve maliyet ayarları."""

    kind: Literal["lgbm", "rf"] = "lgbm"
    threshold: float = 0.55            # P(Y=1) > threshold ise işlem alınır
    step_size: float = 0.0             # bet size ayrıklaştırma adımı (0 -> sürekli)
    use_side_feature: bool = True      # birincil yönü de öznitelik olarak ver
    weighting: Literal["uniqueness", "return", "none"] = "uniqueness"
    cost_per_side: float = 0.0005      # tek yön işlem maliyeti (5 bps)
    # LightGBM (sığ ağaçlar + güçlü düzenlileştirme = düşük varyans)
    n_estimators: int = 300
    learning_rate: float = 0.02
    num_leaves: int = 8
    max_depth: int = 3
    min_child_samples: int = 40
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    # RandomForest (Prado'nun önerdiği bagging ayarları, AFML 6.4)
    rf_n_estimators: int = 500
    rf_min_weight_fraction_leaf: float = 0.05
    seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: MetaModelConfig = field(default_factory=MetaModelConfig)
    periods_per_year: int = 252


# ============================================================
# 2. VERI: SENTETIK OHLCV SIMÜLASYONU VE CSV YÜKLEYICI
# ============================================================
# Veri katmanı: sentetik OHLCV simülasyonu ve CSV yükleyici.

from pathlib import Path

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

TREND, MEAN_REVERT = 0, 1


def simulate_ohlcv(
    cfg: SimulationConfig | None = None,
    return_regimes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Rejim değiştiren, GARCH volatiliteli sentetik günlük OHLCV serisi üretir.

    Log getiri dinamiği:

    * Trend rejimi      : r_t = s * mu + phi * r_{t-1} + sigma_t * z_t
      (s = rejime girişte rastgele seçilen yön, phi > 0 momentum)
    * Mean-revert rejimi: r_t = -kappa * (x_{t-1} - anchor) + sigma_t * z_t
      (anchor = rejime girişteki log fiyat)

    sigma_t GARCH(1,1) ile güncellenir, z_t birim varyanslı Student-t'dir.
    Gizli rejim serisi yalnızca teşhis amaçlıdır; öznitelik olarak KULLANILMAMALIDIR
    (gerçek piyasada gözlenemez).

    Args:
        cfg: Simülasyon parametreleri.
        return_regimes: True ise (ohlcv, rejim_serisi) döndürülür.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_bars

    # 1) Gizli Markov rejim zinciri
    regimes = np.empty(n, dtype=np.int8)
    regimes[0] = TREND
    switch = rng.random(n) > cfg.regime_persistence
    for t in range(1, n):
        regimes[t] = 1 - regimes[t - 1] if switch[t] else regimes[t - 1]

    # 2) Student-t şokları (birim varyansa ölçeklenmiş)
    dof = cfg.t_dof
    z = rng.standard_t(dof, size=n) / np.sqrt(dof / (dof - 2.0))

    # 3) GARCH(1,1) + rejime bağlı ortalama denklemi
    var_lr = cfg.long_run_vol**2
    omega = var_lr * (1.0 - cfg.garch_alpha - cfg.garch_beta)
    vol_mult = {TREND: cfg.trend_vol_mult, MEAN_REVERT: cfg.mr_vol_mult}

    log_close = np.empty(n)
    rets = np.empty(n)
    sigmas = np.empty(n)
    x_prev, r_prev, eps_prev, sig2 = np.log(cfg.s0), 0.0, 0.0, var_lr
    trend_sign, anchor = 1.0, x_prev
    for t in range(n):
        if t > 0:
            sig2 = omega + cfg.garch_alpha * eps_prev**2 + cfg.garch_beta * sig2
        if t == 0 or regimes[t] != regimes[t - 1]:
            trend_sign = rng.choice([-1.0, 1.0])
            anchor = x_prev
        sigma = np.sqrt(sig2) * vol_mult[int(regimes[t])]
        if regimes[t] == TREND:
            mu = trend_sign * cfg.trend_drift + cfg.trend_ar * r_prev
        else:
            mu = -cfg.mr_kappa * (x_prev - anchor)
        eps = sigma * z[t]
        r = mu + eps
        x_prev = x_prev + r
        log_close[t], rets[t], sigmas[t] = x_prev, r, sigma
        r_prev, eps_prev = r, eps / vol_mult[int(regimes[t])]

    # 4) Kapanıştan tutarlı OHLC ve hacim üret
    close = np.exp(log_close)
    prev_close = np.concatenate([[cfg.s0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.2 * sigmas))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0.0, 0.5 * sigmas)))
    low = body_lo * np.exp(-np.abs(rng.normal(0.0, 0.5 * sigmas)))
    volume = 1e6 * np.exp(rng.normal(0.0, 0.3, n)) * (1.0 + np.abs(rets) / sigmas)

    index = pd.bdate_range(cfg.start, periods=n, name="Date")
    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.round()},
        index=index,
    )
    if return_regimes:
        return ohlcv, pd.Series(regimes, index=index, name="regime")
    return ohlcv


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'ini doğrular ve normalize eder (sıralı, tekil, tz'siz indeks)."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik OHLCV kolonları: {missing}")
    out = df[OHLCV_COLUMNS].astype(float).dropna()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if (out["Close"] <= 0).any():
        raise ValueError("Kapanış fiyatları pozitif olmalı.")
    return out


def load_ohlcv_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """CSV'den OHLCV yükler (ör. BIST hisse verisi). Kolon adları büyük/küçük harf duyarsızdır."""
    raw = pd.read_csv(path)
    rename = {c: c.strip().capitalize() for c in raw.columns}
    raw = raw.rename(columns=rename)
    date_key = date_col.strip().capitalize()
    if date_key not in raw.columns:
        raise ValueError(f"Tarih kolonu bulunamadı: {date_col}")
    raw = raw.set_index(pd.to_datetime(raw.pop(date_key)))
    return validate_ohlcv(raw)


# ============================================================
# 3. DINAMIK VOLATILITE
# ============================================================
# Dinamik volatilite tahmini (AFML Snippet 3.1).
#
# Prado, bariyerleri sabit yüzdelerle (ör. %2 kâr al / %2 zarar kes) belirlemenin
# hatalı olduğunu vurgular: Volatilitenin yüksek olduğu dönemlerde sabit bariyerler
# gürültüyle tetiklenir, düşük olduğu dönemlerde ise hiç tetiklenmez. Bu yüzden
# bariyer genişliği ("trgt") olay anındaki *tahmini* volatiliteye göre ölçeklenir.
# Tahmin yalnızca t anına kadarki getirileri kullanır (look-ahead yok).

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def get_volatility(close: pd.Series, span: int = 50, method: str = "ewma") -> pd.Series:
    """Bar başına log getiri volatilitesi.

    Args:
        close: Kapanış fiyatları.
        span: EWMA span'i (``method="ewma"``) veya pencere uzunluğu (``"rolling"``).
        method: ``"ewma"`` (Prado'nun tercihi, yakın geçmişe daha fazla ağırlık)
            veya ``"rolling"`` (eşit ağırlıklı standart sapma).
    """
    ret = log_returns(close)
    min_periods = max(10, span // 2)
    if method == "ewma":
        vol = ret.ewm(span=span, min_periods=min_periods).std()
    elif method == "rolling":
        vol = ret.rolling(span, min_periods=min_periods).std()
    else:
        raise ValueError(f"Bilinmeyen volatilite yöntemi: {method}")
    return vol.rename("vol")


# ============================================================
# 4. BIRINCIL MODEL (YÖN SINYALI)
# ============================================================
# Birincil (primary) model: yalnızca işlemin YÖNÜNÜ belirler.
#
# Meta-labeling (AFML 3.6) iş bölümüne dayanır:
#
# * Birincil model -> "Hangi yönde?" (side ∈ {-1, 0, +1}). Yüksek *recall*
#   hedeflenir: fırsatların büyük kısmını yakalamalı, yanlış pozitifler kabul
#   edilebilir.
# * İkincil (meta) model -> "Bu sinyale uymalı mıyım, ne kadar büyüklükte?"
#   Yanlış pozitifleri eleyerek *precision*'ı yükseltir.
#
# Bu ayrım, ML modelinin yön tahmini gibi zor bir problemi çözmesi yerine daha
# kolay bir ikili sınıflandırma problemi (sinyal doğru mu?) çözmesini sağlar ve
# aşırı öğrenme riskini azaltır. Birincil model bir kural, bir ekonometrik model
# ya da bir insan (discretionary trader) olabilir.

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class PrimaryModel(ABC):
    """Birincil model arayüzü. ``side`` her bar kapanışında bilinen yönü döndürür."""

    name: str = "primary"

    @abstractmethod
    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Her bar için {-1, 0, +1} yön serisi (yalnızca geçmiş veriye dayanır)."""


@dataclass
class EMACrossover(PrimaryModel):
    """Trend takipçisi EMA kesişimi: hızlı EMA > yavaş EMA ise Long, tersi Short.

    ``neutral_band`` > 0 ise iki EMA arasındaki göreli fark bu bandın altındayken
    nötr (0) kalınır; bu, kesişim anlarındaki whipsaw'ları azaltır.
    """

    fast: int = 10
    slow: int = 40
    neutral_band: float = 0.0

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast < slow olmalı")
        self.name = f"EMA({self.fast}/{self.slow})"

    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["Close"]
        fast = close.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        slow = close.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        spread = (fast - slow) / slow
        side = np.sign(spread).where(spread.abs() > self.neutral_band, 0.0)
        return side.fillna(0.0).astype(int).rename("side")


@dataclass
class BollingerBreakout(PrimaryModel):
    """Bollinger bant kırılımı (trend takipçisi).

    Kapanış üst bandın üzerindeyse +1, alt bandın altındaysa -1, bant içindeyse 0.
    Dar bant (düşük ``num_std``) daha fazla sinyal ve daha yüksek recall demektir.
    """

    window: int = 20
    num_std: float = 1.5

    def __post_init__(self) -> None:
        self.name = f"Bollinger({self.window},{self.num_std})"

    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["Close"]
        mid = close.rolling(self.window).mean()
        sd = close.rolling(self.window).std(ddof=0)
        upper, lower = mid + self.num_std * sd, mid - self.num_std * sd
        side = pd.Series(0, index=close.index, dtype=int)
        side[close > upper] = 1
        side[close < lower] = -1
        return side.rename("side")


def build_primary_model(cfg: PrimaryConfig) -> PrimaryModel:
    if cfg.kind == "ema":
        return EMACrossover(cfg.ema_fast, cfg.ema_slow, cfg.ema_neutral_band)
    if cfg.kind == "bollinger":
        return BollingerBreakout(cfg.bb_window, cfg.bb_num_std)
    raise ValueError(f"Bilinmeyen birincil model: {cfg.kind}")


# ============================================================
# 5. OLAY ÖRNEKLEME (CUSUM FILTRESI)
# ============================================================
# Olay (event) örnekleme: hangi barlarda bir işlem kararı değerlendirilecek?
#
# Prado (AFML 2.5) her barı bir gözlem olarak kullanmak yerine *bilgi içeren*
# anların örneklenmesini önerir. Her bar etiketlenirse gözlemler aşırı derecede
# örtüşür ve IID varsayımı daha da bozulur. Simetrik CUSUM filtresi, fiyatın
# son sıfırlamadan bu yana eşik kadar yukarı ya da aşağı sürüklendiği anları
# yakalar; yani yalnızca "anlamlı" hareketlerde bir olay üretir.

import numpy as np
import pandas as pd


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Simetrik CUSUM filtresi (AFML Snippet 2.4).

    S+_t = max(0, S+_{t-1} + r_t),  S-_t = min(0, S-_{t-1} + r_t)
    S+_t > h_t veya S-_t < -h_t olduğunda bir olay kaydedilir ve ilgili toplam sıfırlanır.

    Args:
        close: Kapanış fiyatları.
        threshold: Sabit eşik ya da bar bazında dinamik eşik serisi (ör. k x volatilite).
            Dinamik eşik, t anına kadar bilinen bilgiyle hesaplanmalıdır.
    """
    ret = np.log(close).diff().to_numpy()
    if isinstance(threshold, pd.Series):
        thr = threshold.reindex(close.index).to_numpy(dtype=float)
    else:
        thr = np.full(len(close), float(threshold))

    events: list[int] = []
    s_pos = s_neg = 0.0
    for i in range(1, len(ret)):
        r, h = ret[i], thr[i]
        if np.isnan(r) or np.isnan(h):
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_neg < -h:
            s_neg = 0.0
            events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            events.append(i)
    return close.index[events]


def sample_events(
    side: pd.Series,
    close: pd.Series,
    mode: str = "cusum",
    cusum_threshold: float | pd.Series | None = None,
) -> pd.DatetimeIndex:
    """Birincil modelin yön serisinden işlem olaylarını (t0) seçer.

    * ``"cusum"``  : CUSUM olayları ∩ {side != 0}. Yön olay anındaki birincil sinyaldir.
    * ``"signal"`` : side'ın sıfırdan farklı yeni bir değere geçtiği barlar
      (yeni giriş veya yön değişimi).
    """
    side = side.reindex(close.index).fillna(0)
    if mode == "cusum":
        if cusum_threshold is None:
            raise ValueError("cusum modu için cusum_threshold gerekli")
        t_events = cusum_filter(close, cusum_threshold)
        return t_events[side.loc[t_events].to_numpy() != 0]
    if mode == "signal":
        changed = (side != side.shift(1)) & (side != 0)
        return side.index[changed.to_numpy()]
    raise ValueError(f"Bilinmeyen olay modu: {mode}")


# ============================================================
# 6. TRIPLE BARRIER METHOD VE META-ETIKETLER
# ============================================================
# Triple Barrier Method ve meta-etiketler (AFML Bölüm 3).
#
# Sabit ufuklu etiketleme (ör. "5 gün sonraki getiri > 0") iki temel kusur taşır:
# (1) volatiliteden bağımsız sabit eşik kullanır, (2) yol bağımlılığını yok sayar:
# pozisyon vadeden önce stop-loss'a takılıp kapanmış olabilir. Triple Barrier
# yöntemi işlemin gerçekte nasıl yönetileceğini taklit eder:
#
#     Üst bariyer (pt) : kâr al   -> fiyat giriş * exp(+pt x trgt) seviyesine ulaşırsa
#     Alt bariyer (sl) : zarar kes -> fiyat giriş * exp(-sl x trgt) seviyesine düşerse
#     Dikey bariyer    : zaman aşımı -> max_holding_bars bar sonra pozisyon kapanır
#
# Burada ``trgt`` olay anındaki dinamik volatilitedir. İlk dokunulan bariyer
# işlemin sonucunu ve bitiş zamanını (``t1``) belirler. Yön (side) bilindiği
# için bariyerler pozisyona göre yorumlanır: short pozisyonda fiyatın düşmesi
# üst (kâr) bariyerine dokunmak demektir.

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


# ============================================================
# 7. ÖRTÜŞEN ETIKETLER IÇIN ÖRNEK AĞIRLIKLARI
# ============================================================
# Örtüşen etiketler için örnek ağırlıkları (AFML Bölüm 4).
#
# Triple Barrier etiketleri zaman içinde örtüşür: t0=10'da açılıp t1=18'de
# kapanan bir işlem ile t0=12'de açılan işlem aynı getirilere bağlıdır. Bu,
# gözlemlerin IID olmadığı anlamına gelir. Örtüşen gözlemlere eşit ağırlık
# vermek modeli aynı bilginin kopyalarını ezberlemeye (overfitting) iter.
#
# * Eşzamanlılık (concurrency) c_t : t barında aktif olan etiket sayısı.
# * Benzersizlik (uniqueness) u_i  : etiketin ömrü boyunca 1/c_t'nin ortalaması.
#   Tamamen bağımsız bir etiket için 1, çok kalabalık dönemdeki etiket için ~0.
# * Ortalama benzersizlik ayrıca bagging'deki ``max_samples`` için doğal bir
#   üst sınırdır: bootstrap örnekleri gereğinden fazla örtüşen gözlem içermez.

import numpy as np
import pandas as pd


def _event_positions(index: pd.DatetimeIndex, t1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    start = index.get_indexer(t1.index)
    end = index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    if (start < 0).any() or (end < 0).any():
        raise KeyError("Olay başlangıç/bitiş zamanları fiyat indeksinde olmalı")
    return start, end


def num_concurrent_events(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Her bar için aktif (ömrü [t0, t1] aralığında o barı kapsayan) etiket sayısı (Snippet 4.1)."""
    start, end = _event_positions(index, t1)
    diff = np.zeros(len(index) + 1)
    np.add.at(diff, start, 1.0)
    np.add.at(diff, end + 1, -1.0)
    return pd.Series(np.cumsum(diff[:-1]), index=index, name="num_co_events")


def average_uniqueness(
    index: pd.DatetimeIndex, t1: pd.Series, num_co_events: pd.Series | None = None
) -> pd.Series:
    """Her etiketin ortalama benzersizliği: mean_{t ∈ [t0, t1]} 1 / c_t (Snippet 4.2)."""
    if num_co_events is None:
        num_co_events = num_concurrent_events(index, t1)
    start, end = _event_positions(index, t1)
    c = num_co_events.to_numpy(dtype=float)
    inv = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    avg_u = (cum[end + 1] - cum[start]) / (end - start + 1)
    return pd.Series(avg_u, index=t1.index, name="uniqueness")


def return_attribution_weights(
    close: pd.Series, t1: pd.Series, num_co_events: pd.Series | None = None
) -> pd.Series:
    """Getiri atfı ile ağırlıklandırma (Snippet 4.10).

    w_i ∝ | Σ_{t ∈ (t0, t1]} r_t / c_t |  -- büyük ve benzersiz mutlak getiriye
    sahip etiketler daha fazla ağırlık alır. Ağırlıklar toplamı gözlem sayısına
    eşit olacak şekilde normalize edilir.
    """
    index = close.index
    if num_co_events is None:
        num_co_events = num_concurrent_events(index, t1)
    start, end = _event_positions(index, t1)
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    c = num_co_events.to_numpy(dtype=float)
    attributed = np.divide(ret, c, out=np.zeros_like(ret), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(attributed)])
    w = np.abs(cum[end + 1] - cum[start + 1])
    w = w * len(w) / w.sum() if w.sum() > 0 else np.ones_like(w)
    return pd.Series(w, index=t1.index, name="return_weight")


# ============================================================
# 8. META-MODEL ÖZNITELIKLERI (REJIM GÖSTERGELERI)
# ============================================================
# Meta-model öznitelikleri: yönden bağımsız piyasa rejimi göstergeleri.
#
# Meta-model "birincil sinyal bu koşullarda güvenilir mi?" sorusunu yanıtlar;
# dolayısıyla ihtiyaç duyduğu bilgi yön değil, *ortam*dır: trend mi yatay mı,
# volatilite genişliyor mu daralıyor mu, getiriler momentum mu yoksa ortalamaya
# dönüş mü sergiliyor? Aşağıdaki özniteliklerin tamamı t barının kapanışına
# kadar bilinen veriyle hesaplanır (geriye bakan pencereler, ``shift(-k)`` yok).
# Bu özellik ``tests/test_features_weights_sizing.py`` içinde kesilmiş veri üzerinde
# doğrulanır.

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI (0-100). Aşırı alım/satım rejimini ölçer."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0).where(gain.notna())


def average_true_range(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR: gap'leri de içeren bar içi volatilite ölçüsü."""
    high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman verimlilik oranı: |net hareket| / toplam yol. 1'e yakın = trend, 0'a yakın = gürültü."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net / path.replace(0.0, np.nan)


def build_feature_matrix(ohlcv: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Bar bazında rejim öznitelik matrisi.

    Öznitelikler:
        rsi            : RSI (aşırı alım/satım)
        atr_pct        : ATR / kapanış (normalize bar içi volatilite)
        atr_vol_ratio  : ATR% / kapanıştan-kapanışa volatilite (bar içi gürültü oranı)
        vol_ratio      : kısa / uzun EWMA volatilite (volatilite rejimi genişliyor mu?)
        bb_width       : (üst bant - alt bant) / orta bant (sıkışma / genişleme)
        autocorr       : getirilerin 1 gecikmeli otokorelasyonu (+ momentum, - ortalamaya dönüş)
        efficiency     : Kaufman verimlilik oranı (trend kalitesi)
        abs_momentum   : |k-bar log getiri| / (vol * sqrt(k)) (yönsüz trend gücü, z-skoru)
        ret_skew       : getiri çarpıklığı
        volume_z       : log hacim z-skoru
    """
    cfg = cfg or FeatureConfig()
    close = ohlcv["Close"]
    log_ret = np.log(close).diff()

    vol_fast = log_ret.ewm(span=cfg.vol_fast_span, min_periods=cfg.vol_fast_span).std()
    vol_slow = log_ret.ewm(span=cfg.vol_slow_span, min_periods=cfg.vol_slow_span).std()
    atr_pct = average_true_range(ohlcv, cfg.atr_window) / close

    mid = close.rolling(cfg.bb_window).mean()
    sd = close.rolling(cfg.bb_window).std(ddof=0)

    mom = np.log(close / close.shift(cfg.momentum_window))
    log_vol = np.log(ohlcv["Volume"].replace(0.0, np.nan))
    vol_mean = log_vol.rolling(cfg.volume_window).mean()
    vol_std = log_vol.rolling(cfg.volume_window).std()

    features = pd.DataFrame(
        {
            "rsi": rsi(close, cfg.rsi_window),
            "atr_pct": atr_pct,
            "atr_vol_ratio": atr_pct / vol_slow,
            "vol_ratio": vol_fast / vol_slow,
            "bb_width": 2.0 * cfg.bb_num_std * sd / mid,
            "autocorr": log_ret.rolling(cfg.autocorr_window).corr(log_ret.shift(1)),
            "efficiency": efficiency_ratio(close, cfg.efficiency_window),
            "abs_momentum": mom.abs() / (vol_slow * np.sqrt(cfg.momentum_window)),
            "ret_skew": log_ret.rolling(cfg.skew_window).skew(),
            "volume_z": (log_vol - vol_mean) / vol_std.replace(0.0, np.nan),
        },
        index=ohlcv.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


# ============================================================
# 9. META-MODEL (LIGHTGBM / RANDOM FOREST)
# ============================================================
# İkincil (meta) model fabrikası.
#
# Aşırı öğrenmeye karşı tasarım tercihleri:
#
# * Sınıf dengesizliği: ``class_weight="balanced"`` (LightGBM) /
#   ``"balanced_subsample"`` (RF). Aksi halde model çoğunluk sınıfını tahmin
#   ederek yüksek doğruluk elde edebilir. Not: Dengeleme, olasılıkları sınıf
#   oranları eşitmiş gibi ölçekler; bu yüzden P(Y=1) = 0.5 "birincil model kadar
#   iyi" anlamına gelmez, eşik (ör. 0.55) dengelenmiş ölçekte yorumlanmalıdır.
# * Benzersizlik: Örtüşen etiketlerde her ağacın alt örneklem oranı ortalama
#   benzersizliğe eşitlenir (AFML 4.5 / 6.4). Böylece her ağaç birbirinin
#   kopyası olan gözlemlerle aşırı uyum sağlamaz.
# * Düşük karmaşıklık: Sığ ağaçlar, yaprak başına minimum örnek, L2
#   düzenlileştirme, öznitelik alt örneklemesi.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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


# ============================================================
# 10. PURGED & EMBARGOED ÇAPRAZ DOĞRULAMA
# ============================================================
# Purged & Embargoed zaman serisi çapraz doğrulaması (AFML Bölüm 7).
#
# Standart K-Fold neden finansta başarısız olur?
# --------------------------------------------
# 1. Gözlemler IID değildir. Triple Barrier etiketi y_i, [t0_i, t1_i] aralığındaki
#    fiyat yoluna bağlıdır. Test setindeki bir etiketle zaman aralığı örtüşen bir
#    eğitim etiketi aynı getirileri "görür": model test sonucunu dolaylı olarak
#    ezberler (bilgi sızıntısı / leakage).
# 2. Öznitelikler seri korelasyonludur. Test setinin hemen ardından gelen eğitim
#    gözlemlerinin öznitelikleri (ör. 50 barlık rolling pencereler) test
#    dönemindeki fiyatları içerir.
#
# Çözüm
# -----
# * PURGING: Etiket aralığı [t0_i, t1_i] test setinin kapsadığı
#   [min t0_test, max t1_test] aralığıyla kesişen her eğitim gözlemi atılır.
# * EMBARGO: Test setinin bitişini takip eden h süre (toplam sürenin
#   ``embargo_pct`` kadarı) içinde başlayan eğitim gözlemleri de atılır.
#   Purging test *öncesi* ve test *içi* örtüşmeyi temizler; embargo test
#   *sonrası* seri korelasyon kanalını kapatır.
#
# Bu modül iki kullanım sunar:
# * ``walk_forward=False``: Prado'nun PurgedKFold'u. Eğitim test öncesi ve
#   sonrasındaki (purge + embargo uygulanmış) verileri kullanır. Model teşhisi,
#   öznitelik önemi ve hiperparametre seçimi için.
# * ``walk_forward=True``: Yalnızca test setinden ÖNCEKİ (purge edilmiş) veriyle
#   eğitim. Her tahmin gerçek zamanlı olarak üretilebilecek bir tahmindir;
#   backtest için kullanılan OOS olasılıkları buradan gelir.

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


# ============================================================
# 11. SINYAL FILTRELEME VE BET SIZING
# ============================================================
# Olasılıktan pozisyon büyüklüğüne (AFML Bölüm 10).
#
# Meta-modelin P(Y=1) çıktısı iki şekilde kullanılır:
#
# 1. Filtre: P(Y=1) > eşik ise birincil sinyale uy, değilse pas geç.
# 2. Bet sizing: Güven arttıkça pozisyon büyür. Prado, olasılığı bir test
#    istatistiğine dönüştürür (H0: p = 1/2):
#
#        z = (p - 1/2) / sqrt(p (1 - p)),     m = 2 Φ(z) - 1  ∈ [0, 1)
#
#    p = 0.5 -> m = 0 (bahis yok), p -> 1 -> m -> 1 (tam pozisyon).
#    Nihai sinyal = side x m. İsteğe bağlı ayrıklaştırma (step_size), küçük
#    olasılık oynamalarının gereksiz işlem (overtrading) üretmesini engeller.

import numpy as np
import pandas as pd
from scipy.special import ndtr  # scikit-learn'ün zorunlu bağımlılığı olan scipy'den


def bet_size_from_proba(proba: pd.Series, threshold: float = 0.5, step_size: float = 0.0) -> pd.Series:
    """P(Y=1) -> [0, 1] aralığında pozisyon büyüklüğü; eşik altı 0."""
    p = proba.clip(1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    size = pd.Series(2.0 * ndtr(z.to_numpy()) - 1.0, index=proba.index).clip(lower=0.0)
    if step_size > 0:
        size = (size / step_size).round() * step_size
    size[proba <= threshold] = 0.0
    return size.clip(0.0, 1.0).rename("size")


def meta_signals(
    side: pd.Series, proba: pd.Series, threshold: float, step_size: float = 0.0
) -> dict[str, pd.Series]:
    """Karşılaştırılacak üç sinyal seti (hepsi aynı OOS olay kümesi üzerinde).

    * primary : birincil modelin tüm sinyalleri, sabit büyüklük
    * filtered: yalnızca P(Y=1) > eşik olan sinyaller, sabit büyüklük
    * sized   : P(Y=1) > eşik olan sinyaller, büyüklük = m(p)
    """
    taken = (proba > threshold).astype(float)
    return {
        "primary": side.astype(float),
        "filtered": side * taken,
        "sized": side * bet_size_from_proba(proba, threshold, step_size),
    }


# ============================================================
# 12. BACKTEST VE PERFORMANS METRIKLERI
# ============================================================
# Olay tabanlı basit backtest ve performans metrikleri.
#
# İki seviyede ölçüm yapılır:
#
# * İşlem seviyesi: Her olay için t0 kapanışında giriş, t1 (ilk bariyer) kapanışında
#   çıkış. Win rate, ortalama getiri, işlem Sharpe'ı.
# * Portföy seviyesi: Aynı anda açık olan işlemlerin sinyalleri ortalanır
#   (AFML Snippet 10.2 ``avgActiveSignals``) ve bar bazında pozisyon oluşturulur.
#   Sharpe oranı bu günlük getiri serisi üzerinden yıllıklandırılır. Örtüşen
#   işlemlerde işlem Sharpe'ı şişebileceği için asıl referans budur.
#
# Ek olarak Probabilistic Sharpe Ratio (PSR; Bailey & López de Prado, 2012)
# raporlanır: Gözlenen Sharpe'ın, getirilerin çarpıklık ve basıklığı hesaba
# katıldığında 0'dan büyük olma olasılığı.

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class StrategyStats:
    n_trades: int
    win_rate: float
    avg_trade_ret: float
    trade_sharpe: float
    ann_return: float
    ann_vol: float
    sharpe: float
    psr: float
    max_drawdown: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def trade_returns(
    close: pd.Series, t1: pd.Series, signal: pd.Series, cost_per_side: float = 0.0
) -> pd.Series:
    """Sinyali sıfırdan farklı olayların maliyet sonrası işlem getirileri."""
    signal = signal[signal != 0]
    p0 = close.reindex(signal.index).to_numpy()
    p1 = close.reindex(pd.DatetimeIndex(t1.reindex(signal.index))).to_numpy()
    gross = signal.to_numpy() * (p1 / p0 - 1.0)
    net = gross - 2.0 * cost_per_side * signal.abs().to_numpy()
    return pd.Series(net, index=signal.index, name="trade_ret")


def average_active_positions(index: pd.DatetimeIndex, t1: pd.Series, signal: pd.Series) -> pd.Series:
    """Bar bazında pozisyon = o bar boyunca açık olan işlemlerin sinyal ortalaması.

    t0'da kapanışta açılan işlem, (t0, t1] aralığındaki bar getirilerine maruz kalır.
    """
    signal = signal[signal != 0]
    start = index.get_indexer(signal.index) + 1
    end = index.get_indexer(pd.DatetimeIndex(t1.reindex(signal.index)))
    n = len(index)
    total, count = np.zeros(n + 1), np.zeros(n + 1)
    np.add.at(total, start, signal.to_numpy())
    np.add.at(total, end + 1, -signal.to_numpy())
    np.add.at(count, start, 1.0)
    np.add.at(count, end + 1, -1.0)
    total, count = np.cumsum(total[:-1]), np.cumsum(count[:-1])
    pos = np.divide(total, count, out=np.zeros(n), where=count > 0.5)
    return pd.Series(pos, index=index, name="position")


def portfolio_returns(
    close: pd.Series, t1: pd.Series, signal: pd.Series, cost_per_side: float = 0.0
) -> pd.Series:
    """Bar bazında strateji getirisi: pozisyon x bar getirisi - maliyet x |Δpozisyon|."""
    pos = average_active_positions(close.index, t1, signal)
    bar_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return (pos * bar_ret - cost_per_side * turnover).rename("strategy_ret")


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR = Φ( (SR - SR*) sqrt(n-1) / sqrt(1 - γ3 SR + (γ4 - 1)/4 SR²) ), periyot bazında SR."""
    r = returns.dropna()
    n, sd = len(r), r.std()
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew, kurt = r.skew(), r.kurt() + 3.0  # pandas kurt() fazlalık basıklık döndürür
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(ndtr((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def evaluate_strategy(
    close: pd.Series,
    t1: pd.Series,
    signal: pd.Series,
    cost_per_side: float = 0.0,
    periods_per_year: int = 252,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> StrategyStats:
    """Tek bir sinyal seti için işlem ve portföy seviyesi metrikler.

    ``window`` verilirse portföy getirileri bu aralıkla sınırlanır; farklı
    stratejiler aynı takvim üzerinde (boşta geçen günler dahil) karşılaştırılır.
    """
    trades = trade_returns(close, t1, signal, cost_per_side)
    port = portfolio_returns(close, t1, signal, cost_per_side)
    if window is not None:
        port = port.loc[window[0] : window[1]]
    years = max(len(port) / periods_per_year, 1e-9)

    if len(trades) > 1 and trades.std() > 0:
        trade_sharpe = trades.mean() / trades.std() * np.sqrt(len(trades) / years)
    else:
        trade_sharpe = float("nan")
    ann_ret = port.mean() * periods_per_year
    ann_vol = port.std() * np.sqrt(periods_per_year)
    return StrategyStats(
        n_trades=len(trades),
        win_rate=float((trades > 0).mean()) if len(trades) else float("nan"),
        avg_trade_ret=float(trades.mean()) if len(trades) else float("nan"),
        trade_sharpe=float(trade_sharpe),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        psr=probabilistic_sharpe_ratio(port),
        max_drawdown=max_drawdown(port),
    )


def compare_strategies(
    close: pd.Series,
    t1: pd.Series,
    signals: dict[str, pd.Series],
    cost_per_side: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Birden çok sinyal setini aynı OOS penceresinde karşılaştırır."""
    window = (t1.index.min(), t1.max())
    rows = {
        name: evaluate_strategy(close, t1, sig, cost_per_side, periods_per_year, window).as_dict()
        for name, sig in signals.items()
    }
    return pd.DataFrame(rows).T


# ============================================================
# 13. PIPELINE ORKESTRASYONU VE ÖZET RAPOR
# ============================================================
# Uçtan uca Meta-Labeling pipeline'ı.
#
# Akış
# ----
# 1. Veri          : OHLCV (simülasyon ya da CSV)
# 2. Birincil model: side ∈ {-1, 0, +1}; CUSUM ile olay örnekleme (yüksek recall)
# 3. Triple Barrier: dinamik volatiliteye ölçekli pt/sl + dikey bariyer
# 4. Meta-etiket   : Y = 1 (işlem kârlı) / 0 (zarar)
# 5. Öznitelikler  : yönden bağımsız rejim göstergeleri (+ opsiyonel side)
# 6. Ağırlıklar    : ortalama benzersizlik (örtüşen etiketler)
# 7. Doğrulama     : Purged K-Fold (teşhis) + Purged walk-forward (OOS olasılıklar)
# 8. Karar         : P(Y=1) > eşik filtresi ve olasılıktan bet sizing
# 9. Değerlendirme : filtre öncesi / sonrası Win Rate ve Sharpe karşılaştırması

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score


STRATEGY_LABELS = {
    "primary": "Birincil (filtresiz)",
    "filtered": "Meta filtre",
    "sized": "Meta filtre + bet sizing",
}


@dataclass
class PipelineResult:
    """Pipeline çıktıları. ``summary()`` insan tarafından okunabilir rapor üretir."""

    config: PipelineConfig
    primary_name: str
    ohlcv: pd.DataFrame
    events: pd.DataFrame           # t0 indeksli: t1, trgt, side, barrier, ret, bin, weight, proba, ...
    X: pd.DataFrame                # meta-model öznitelikleri (events ile aynı indeks)
    cv_scores: pd.DataFrame        # Purged K-Fold fold skorları
    comparison: pd.DataFrame       # strateji karşılaştırması (OOS)
    oos_metrics: dict[str, float]
    importance: pd.Series
    final_model: object            # tüm etiketli olaylarla eğitilmiş, canlı kullanım modeli
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        return format_summary(self)


class MetaLabelingPipeline:
    """Meta-Labeling + Triple Barrier araştırma/üretim pipeline'ı.

    Örnek:
        >>> from meta_labeling import MetaLabelingPipeline
        >>> result = MetaLabelingPipeline().run()     # sentetik veri
        >>> print(result.summary())
        >>> live = MetaLabelingPipeline().score_events(ohlcv, result.final_model)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ adımlar
    def build_primary(self) -> PrimaryModel:
        return build_primary_model(self.config.primary)

    def volatility(self, close: pd.Series) -> pd.Series:
        b = self.config.barrier
        return get_volatility(close, span=b.vol_span, method=b.vol_method)

    def candidate_events(self, ohlcv: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
        """Birincil sinyal + olay örnekleme. Etiketi henüz bilinmeyen güncel olayları da içerir.

        Volatilitesi ``min_target``'ın altında kalan olaylar burada elenir; böylece
        eğitim (``label``) ve canlı skorlama (``score_events``) aynı olay kümesini görür.
        """
        close = ohlcv["Close"]
        vol = self.volatility(close)
        side = self.build_primary().side(ohlcv)
        p = self.config.primary
        t_events = sample_events(side, close, p.event_mode, cusum_threshold=vol * p.cusum_vol_mult)
        t_events = t_events[(vol.reindex(t_events) > self.config.barrier.min_target).to_numpy()]
        return t_events, side, vol

    def label(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Adım 2-4: olaylar, Triple Barrier ve meta-etiketler."""
        close = ohlcv["Close"]
        b = self.config.barrier
        t_events, side, vol = self.candidate_events(ohlcv)
        vertical = get_vertical_barriers(t_events, close.index, b.max_holding_bars)
        events = apply_triple_barrier(
            close, t_events, vol, side, vertical, b.pt_mult, b.sl_mult, b.min_target
        )
        return get_meta_labels(events, self.config.model.cost_per_side)

    def features(self, ohlcv: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Adım 5: olay anındaki (t0 kapanışı) öznitelikler."""
        X = build_feature_matrix(ohlcv, self.config.features).reindex(events.index)
        if self.config.model.use_side_feature:
            X["side"] = events["side"].astype(float)
        return X

    def sample_weights(self, close: pd.Series, events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Adım 6: (eğitim ağırlıkları, ortalama benzersizlik)."""
        co = num_concurrent_events(close.index, events["t1"])
        uniq = average_uniqueness(close.index, events["t1"], co)
        scheme = self.config.model.weighting
        if scheme == "uniqueness":
            w = uniq / uniq.mean()
        elif scheme == "return":
            w = return_attribution_weights(close, events["t1"], co)
        elif scheme == "none":
            w = pd.Series(1.0, index=events.index)
        else:
            raise ValueError(f"Bilinmeyen ağırlıklandırma: {scheme}")
        return w.rename("weight"), uniq

    # ------------------------------------------------------------------ çalıştır
    def run(self, ohlcv: pd.DataFrame | None = None) -> PipelineResult:
        cfg = self.config
        ohlcv = validate_ohlcv(ohlcv) if ohlcv is not None else simulate_ohlcv(cfg.sim)
        close = ohlcv["Close"]

        # 1-4) Etiketleme
        events = self.label(ohlcv)

        # 5) Öznitelikler; ısınma dönemindeki eksik gözlemler atılır
        X = self.features(ohlcv, events)
        valid = X.notna().all(axis=1)
        events, X = events.loc[valid].copy(), X.loc[valid]
        y = events["bin"]
        if len(events) < 2 * cfg.cv.n_splits or y.nunique() < 2:
            raise ValueError("Meta-model eğitimi için yeterli/çeşitli etiket yok")

        # 6) Örnek ağırlıkları
        w, uniq = self.sample_weights(close, events)
        events["weight"], events["uniqueness"] = w, uniq
        estimator = build_meta_model(cfg.model, avg_uniqueness=float(uniq.mean()))

        # 7a) Purged K-Fold teşhisleri
        cv_kfold = PurgedKFold(events["t1"], cfg.cv.n_splits, cfg.cv.embargo_pct)
        cv_scores = cross_validate_purged(estimator, X, y, cv_kfold, w, cfg.model.threshold)

        # 7b) Purged walk-forward OOS olasılıkları (backtest girdisi)
        cv_wf = PurgedKFold(events["t1"], cfg.cv.n_splits, cfg.cv.embargo_pct, walk_forward=True)
        proba, wf_models = walk_forward_predict(
            estimator, X, y, cv_wf, w, min_train_events=cfg.cv.min_train_events
        )
        events["proba"] = proba
        oos = events.loc[proba.notna()]
        if oos.empty:
            raise ValueError("OOS tahmin üretilemedi; daha fazla veri ya da daha düşük min_train_events gerekli")

        # 8) Sinyal filtreleme & bet sizing
        signals = meta_signals(oos["side"], oos["proba"], cfg.model.threshold, cfg.model.step_size)
        events["size"] = bet_size_from_proba(oos["proba"], cfg.model.threshold, cfg.model.step_size)

        # 9) Karşılaştırma (aynı OOS olayları, aynı takvim penceresi)
        comparison = compare_strategies(
            close, oos["t1"], signals, cfg.model.cost_per_side, cfg.periods_per_year
        )
        comparison.index = [
            f"{STRATEGY_LABELS[k]} (p>{cfg.model.threshold:.2f})" if k != "primary" else STRATEGY_LABELS[k]
            for k in comparison.index
        ]

        taken = oos["proba"] > cfg.model.threshold
        y_oos = oos["bin"]
        oos_metrics = {
            "n_oos": float(len(oos)),
            "auc": float(roc_auc_score(y_oos, oos["proba"])) if y_oos.nunique() == 2 else float("nan"),
            "base_precision": float(y_oos.mean()),
            "meta_precision": float(y_oos[taken].mean()) if taken.any() else float("nan"),
            "meta_recall": float(y_oos[taken].sum() / max(y_oos.sum(), 1)),
            "coverage": float(taken.mean()),
        }

        # Canlı kullanım için tüm etiketli veriyle eğitilmiş nihai model
        final_model = clone(estimator).fit(X, y, sample_weight=w.to_numpy())

        return PipelineResult(
            config=cfg,
            primary_name=self.build_primary().name,
            ohlcv=ohlcv,
            events=events,
            X=X,
            cv_scores=cv_scores,
            comparison=comparison,
            oos_metrics=oos_metrics,
            importance=feature_importance(wf_models, X.columns),
            final_model=final_model,
        )

    # ------------------------------------------------------------------ canlı
    def score_events(self, ohlcv: pd.DataFrame, model, last_n: int | None = None) -> pd.DataFrame:
        """Güncel veride olayları üretir ve eğitilmiş meta-modelle puanlar.

        Etiketi henüz oluşmamış (dikey bariyeri gelecekte olan) son olaylar da
        dahildir; canlı işlem kararı bunların sonuncusundan okunur.
        """
        ohlcv = validate_ohlcv(ohlcv)
        t_events, side, _ = self.candidate_events(ohlcv)
        frame = pd.DataFrame({"side": side.reindex(t_events)}, index=t_events)
        X = self.features(ohlcv, frame).dropna()
        if last_n is not None:
            X = X.iloc[-last_n:]
        proba = pd.Series(positive_class_proba(model, X), index=X.index, name="proba")
        m = self.config.model
        size = bet_size_from_proba(proba, m.threshold, m.step_size)
        return pd.DataFrame(
            {"side": frame["side"].reindex(X.index), "proba": proba, "size": size,
             "signal": frame["side"].reindex(X.index) * size}
        )


# ---------------------------------------------------------------------- rapor
def _pct(x: float) -> str:
    return "  n/a" if pd.isna(x) else f"{100 * x:5.1f}%"


def format_summary(res: PipelineResult) -> str:
    cfg, ev, cmp_ = res.config, res.events, res.comparison
    b, m, cvc = cfg.barrier, cfg.model, cfg.cv
    lines: list[str] = []
    bar = "=" * 78
    lines += [bar, " META-LABELING + TRIPLE BARRIER PIPELINE — ÖZET", bar]
    idx = res.ohlcv.index
    lines.append(
        f"Veri          : {len(idx)} bar ({idx[0].date()} → {idx[-1].date()})"
    )
    lines.append(
        f"Birincil model: {res.primary_name}, olay örnekleme = {cfg.primary.event_mode}"
    )
    lines.append(
        f"Bariyerler    : pt={b.pt_mult}σ, sl={b.sl_mult}σ, dikey={b.max_holding_bars} bar, "
        f"σ = {b.vol_method.upper()}({b.vol_span}) log-getiri volatilitesi"
    )
    dist = ev["barrier"].value_counts(normalize=True)
    lines.append(
        f"Olaylar       : {len(ev)} | Y=1 oranı {_pct(ev['bin'].mean())} | "
        f"ort. benzersizlik {ev['uniqueness'].mean():.2f} | "
        f"temas: pt {_pct(dist.get('pt', 0))}, sl {_pct(dist.get('sl', 0))}, "
        f"dikey {_pct(dist.get('vertical', 0))}"
    )

    lines += ["", f"[1] Purged K-Fold CV (k={cvc.n_splits}, embargo={cvc.embargo_pct:.0%}) — meta-model teşhisi"]
    cols = ["n_train", "n_test", "auc", "log_loss", "precision", "recall", "base_rate"]
    cv_tbl = res.cv_scores[cols].copy()
    lines.append(cv_tbl.to_string(float_format=lambda v: f"{v:.3f}"))
    lines.append(
        f"  ortalama AUC = {cv_tbl['auc'].mean():.3f} ± {cv_tbl['auc'].std():.3f} | "
        f"precision {cv_tbl['precision'].mean():.3f} vs baz oran {cv_tbl['base_rate'].mean():.3f}"
    )

    om = res.oos_metrics
    lines += [
        "",
        "[2] Purged walk-forward OOS (her fold yalnızca geçmişte kapanmış etiketlerle eğitilir)",
        f"  OOS olay: {int(om['n_oos'])} | AUC: {om['auc']:.3f} | "
        f"birincil precision (baz): {_pct(om['base_precision'])} → meta precision: "
        f"{_pct(om['meta_precision'])} | recall: {_pct(om['meta_recall'])} | "
        f"kapsama: {_pct(om['coverage'])}",
    ]

    lines += ["", f"[3] Strateji karşılaştırması (OOS, maliyet {m.cost_per_side * 1e4:.0f} bps/yön)"]
    view = pd.DataFrame(
        {
            "İşlem": cmp_["n_trades"].astype(int),
            "WinRate": cmp_["win_rate"].map(_pct),
            "Ort.İşlem": cmp_["avg_trade_ret"].map(lambda v: f"{100 * v:+.2f}%"),
            "İşlemSR": cmp_["trade_sharpe"].map(lambda v: f"{v:5.2f}"),
            "Sharpe": cmp_["sharpe"].map(lambda v: f"{v:5.2f}"),
            "PSR": cmp_["psr"].map(_pct),
            "Yıl.Getiri": cmp_["ann_return"].map(_pct),
            "MaxDD": cmp_["max_drawdown"].map(_pct),
        }
    )
    lines.append(view.to_string())
    base, filt = cmp_.iloc[0], cmp_.iloc[1]
    lines.append(
        f"  Δ Win Rate (filtre - birincil): {100 * (filt['win_rate'] - base['win_rate']):+.1f} puan | "
        f"Δ Sharpe: {filt['sharpe'] - base['sharpe']:+.2f} | "
        f"işlem sayısı {int(base['n_trades'])} → {int(filt['n_trades'])}"
    )

    if not res.importance.empty:
        top = ", ".join(f"{k} {v:.2f}" for k, v in res.importance.head(5).items())
        lines += ["", f"[4] Öznitelik önemi (MDI, walk-forward ort., yalnızca araştırma amaçlı): {top}"]
    lines += [
        "",
        "Not: Eşik ve hiperparametreler ex-ante sabittir; OOS sonuçlara bakarak ayarlamak",
        "     backtest overfitting'e yol açar (AFML Bölüm 11). Sharpe günlük portföy getirileri",
        "     üzerinden, PSR ise P(SR > 0) olarak raporlanmıştır.",
        bar,
    ]
    return "\n".join(lines)


# ============================================================
# ÇALIŞTIR
# ============================================================
from dataclasses import replace

cfg = PipelineConfig()
# cfg = replace(cfg, primary=replace(cfg.primary, kind="bollinger"))
# cfg = replace(cfg, model=replace(cfg.model, kind="rf", threshold=0.60))

pipeline = MetaLabelingPipeline(cfg)
result = pipeline.run()  # gerçek veri: pipeline.run(load_ohlcv_csv("SASA.csv"))
print(result.summary())
print()
print("Canlı karar (son 5 olay):")
print(pipeline.score_events(result.ohlcv, result.final_model, last_n=5))
