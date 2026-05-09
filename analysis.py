# ============================================================
# SASA / BIST DAILY SYSTEM
# AGGRESSIVE + CONTROLLED + INSTITUTION FLOW + ZSCORE + REGIME FLOW IMPACT
# Tek hücre: Kopyala -> Çalıştır
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:
    raise ImportError("yfinance yok. Jupyter'da: %pip install -U yfinance")

try:
    import pandas_ta as ta
except Exception:
    raise ImportError("pandas_ta yok. Jupyter'da: %pip install -U pandas_ta")

from sklearn.ensemble import RandomForestClassifier

_HAS_PLOTLY = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


# ----------------------------
# CONFIG
# ----------------------------
CFG = {
    "ticker": "SASA.IS",
    "period": "5y",
    "interval": "1d",

    # flow csv
    "use_flow_data": True,
    "flow_csv_path": "sasa_kurum_alimlari.csv",
    "flow_generate_sample_if_missing": True,

    # target
    "H": 5,
    "atr_len": 14,
    "k_atr": 0.8,
    "min_atr_pct": 0.0025,

    # regime
    "adx_trend": 20,
    "use_regime": True,
    "allow_range": True,

    # ML
    "seed": 42,
    "walk_train_min": 450,
    "walk_retrain_step": 10,
    "rf_n_estimators": 800,
    "rf_max_depth": 8,
    "rf_min_samples_leaf": 30,
    "rf_max_features": "sqrt",

    # decision
    "use_edge": True,
    "edge_thr": 0.015,
    "none_max": 0.70,

    # trading
    "cooldown": 0,
    "time_stop": 2,
    "atr_stop_mult": 1.6,
    "atr_tp_mult": 3.0,

    # flip
    "allow_flip": True,
    "flip_edge_boost": 0.003,

    # risk & costs
    "initial_cash": 100000.0,
    "risk_per_trade": 0.01,
    "fee_bps": 6,
    "slippage_bps": 3,

    # controlled filters
    "bear_long_min_p": 0.52,
    "bear_short_min_p": 0.42,
    "bear_min_edge": 0.10,
    "max_size_mult": 1.60,

    # calibration
    "target_trades_lo": 400,
    "target_trades_hi": 600,
    "do_calibrate": True,

    # zscore windows
    "z_win_short": 20,
    "z_win_long": 60,
    "cum_flow_win": 5,

    # plot
    "plot": True,
}

BOFA_ALIASES = [
    "bofa",
    "bank of america",
    "bank of america corporation",
    "bank of america, national association",
    "merrill lynch",
    "merrill lynch international",
    "mliu",
]

TRACKED_INST_MAP = {
    "Citibank": ["citibank", "citi"],
    "Deutsche Bank": ["deutsche bank"],
    "Goldman Sachs": ["goldman sachs", "goldman"],
    "HSBC": ["hsbc"],
    "JP Morgan": ["jp morgan", "jpmorgan", "j.p. morgan", "jpm"],
    "UBS": ["ubs"],
}


# ----------------------------
# helpers
# ----------------------------
def rolling_zscore(s, win=20):
    m = s.rolling(win, min_periods=max(5, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 3)).std(ddof=0)
    z = (s - m) / sd.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def safe_dfcol(df, prefix):
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    return cols[0] if cols else None

def mfi_custom(high, low, close, volume, length=14):
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0)
    neg = rmf.where(d < 0, 0.0).abs()
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum().replace(0.0, np.nan)
    mfr = pos_sum / neg_sum
    return 100.0 - (100.0 / (1.0 + mfr))


# ----------------------------
# data loader
# ----------------------------
def load_yfinance(ticker, period="5y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        raise ValueError(f"yfinance veri döndürmedi: {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    df = df[needed].dropna()
    df = df[~df.index.duplicated(keep="last")]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ----------------------------
# flow loader
# ----------------------------
def maybe_generate_sample_flow_csv(csv_path, price_index):
    rng = np.random.default_rng(42)
    sample_dates = pd.DatetimeIndex(price_index).sort_values()
    sample_dates = sample_dates[-200:] if len(sample_dates) > 200 else sample_dates

    institutions = ["Bank of America", "HSBC", "JP Morgan", "Goldman Sachs", "Deutsche Bank", "Citibank", "UBS"]
    rows = []
    for d in sample_dates:
        for inst in institutions:
            rows.append({
                "Date": d.strftime("%Y-%m-%d"),
                "Institution": inst,
                "NetLot": int(rng.normal(0, 250000))
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")

def normalize_inst(x):
    x0 = str(x).strip()
    x1 = x0.lower().replace(".", "").replace(",", "").replace("  ", " ")
    if any(alias in x1 for alias in BOFA_ALIASES):
        return "BofA"
    for canon, aliases in TRACKED_INST_MAP.items():
        if any(a in x1 for a in aliases):
            return canon
    return x0

def load_flow_features(price_df, cfg):
    price_index = pd.DatetimeIndex(price_df.index).tz_localize(None)
    out = pd.DataFrame(index=price_index)

    base_cols = [
        "Total_NetLot", "Total_Abs_Flow", "Buyer_Count", "Seller_Count",
        "NetLot_to_Volume", "AbsFlow_to_Volume",
        "Total_NetLot_Lag1", "Total_NetLot_Lag2", "Total_NetLot_Lag3",
        "BofA_NetLot", "BofA_Lag1", "BofA_Lag2", "BofA_Lag3", "BofA_to_Volume",
        "Citibank_NetLot", "Deutsche_Bank_NetLot", "Goldman_Sachs_NetLot",
        "HSBC_NetLot", "JP_Morgan_NetLot", "UBS_NetLot",
        "Foreign_Selected_Total"
    ]
    for c in base_cols:
        out[c] = 0.0

    if not cfg.get("use_flow_data", True):
        return out

    csv_path = cfg.get("flow_csv_path", "sasa_kurum_alimlari.csv")

    if not os.path.exists(csv_path):
        if cfg.get("flow_generate_sample_if_missing", False):
            print(f"[INFO] Flow CSV bulunamadı, örnek dosya oluşturuluyor: {csv_path}")
            maybe_generate_sample_flow_csv(csv_path, price_index)
        else:
            print(f"[WARN] Flow CSV yok: {csv_path}")
            return out

    flows = pd.read_csv(csv_path)
    required_cols = {"Date", "Institution", "NetLot"}
    if not required_cols.issubset(flows.columns):
        print(f"[WARN] Flow CSV kolon eksik. Beklenen: {required_cols}")
        return out

    flows["Date"] = pd.to_datetime(flows["Date"], errors="coerce").dt.tz_localize(None)
    flows["Institution"] = flows["Institution"].astype(str).str.strip()
    flows["NetLot"] = pd.to_numeric(flows["NetLot"], errors="coerce")
    flows = flows.dropna(subset=["Date", "Institution", "NetLot"]).copy()

    if flows.empty:
        print("[WARN] Flow CSV boş/geçersiz.")
        return out

    flows["Institution_Clean"] = flows["Institution"].apply(normalize_inst)

    daily_total = flows.groupby("Date")["NetLot"].sum().rename("Total_NetLot")
    daily_abs = flows.groupby("Date")["NetLot"].apply(lambda x: np.abs(x).sum()).rename("Total_Abs_Flow")
    daily_stats = flows.groupby("Date").agg(
        Buyer_Count=("NetLot", lambda x: float((x > 0).sum())),
        Seller_Count=("NetLot", lambda x: float((x < 0).sum()))
    )

    inst_pivot = flows.pivot_table(
        index="Date",
        columns="Institution_Clean",
        values="NetLot",
        aggfunc="sum",
        fill_value=0
    )

    def inst_series(inst_name):
        if inst_name in inst_pivot.columns:
            return inst_pivot[inst_name].reindex(price_index).fillna(0.0)
        return pd.Series(0.0, index=price_index)

    merged = pd.DataFrame(index=price_index)
    merged = merged.join(daily_total, how="left")
    merged = merged.join(daily_abs, how="left")
    merged = merged.join(daily_stats, how="left")

    merged["BofA_NetLot"] = inst_series("BofA")
    merged["Citibank_NetLot"] = inst_series("Citibank")
    merged["Deutsche_Bank_NetLot"] = inst_series("Deutsche Bank")
    merged["Goldman_Sachs_NetLot"] = inst_series("Goldman Sachs")
    merged["HSBC_NetLot"] = inst_series("HSBC")
    merged["JP_Morgan_NetLot"] = inst_series("JP Morgan")
    merged["UBS_NetLot"] = inst_series("UBS")

    merged["Foreign_Selected_Total"] = (
        merged["BofA_NetLot"] + merged["Citibank_NetLot"] + merged["Deutsche_Bank_NetLot"] +
        merged["Goldman_Sachs_NetLot"] + merged["HSBC_NetLot"] + merged["JP_Morgan_NetLot"] +
        merged["UBS_NetLot"]
    )

    for c in ["Total_NetLot", "Total_Abs_Flow", "Buyer_Count", "Seller_Count"]:
        merged[c] = merged[c].fillna(0.0)

    vol = price_df["Volume"].replace(0, np.nan).reindex(price_index)
    merged["NetLot_to_Volume"] = (merged["Total_NetLot"] / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["AbsFlow_to_Volume"] = (merged["Total_Abs_Flow"] / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["BofA_to_Volume"] = (merged["BofA_NetLot"] / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged["Total_NetLot_Lag1"] = merged["Total_NetLot"].shift(1).fillna(0.0)
    merged["Total_NetLot_Lag2"] = merged["Total_NetLot"].shift(2).fillna(0.0)
    merged["Total_NetLot_Lag3"] = merged["Total_NetLot"].shift(3).fillna(0.0)
    merged["BofA_Lag1"] = merged["BofA_NetLot"].shift(1).fillna(0.0)
    merged["BofA_Lag2"] = merged["BofA_NetLot"].shift(2).fillna(0.0)
    merged["BofA_Lag3"] = merged["BofA_NetLot"].shift(3).fillna(0.0)

    return merged.reindex(price_index).fillna(0.0)


# ----------------------------
# indicators
# ----------------------------
def add_indicators(df, cfg):
    out = df.copy()
    out["EMA20"] = ta.ema(out["Close"], length=20)
    out["EMA50"] = ta.ema(out["Close"], length=50)
    out["EMA200"] = ta.ema(out["Close"], length=200)
    out["RSI14"] = ta.rsi(out["Close"], length=14)

    macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
    if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 3:
        out["MACD"] = macd.iloc[:, 0]
        out["MACDh"] = macd.iloc[:, 1]
        out["MACDs"] = macd.iloc[:, 2]
    else:
        out["MACD"] = out["MACDh"] = out["MACDs"] = np.nan

    bb = ta.bbands(out["Close"], length=20, std=2.0)
    if isinstance(bb, pd.DataFrame):
        bbp = safe_dfcol(bb, "BBP_")
        out["BBP"] = bb[bbp] if bbp else np.nan
    else:
        out["BBP"] = np.nan

    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=int(cfg["atr_len"]))
    out["ATR_PCT"] = out["ATR"] / out["Close"]

    adx = ta.adx(out["High"], out["Low"], out["Close"], length=14)
    if isinstance(adx, pd.DataFrame):
        c_adx = safe_dfcol(adx, "ADX_")
        out["ADX14"] = adx[c_adx] if c_adx else np.nan
    else:
        out["ADX14"] = np.nan

    out["ROC10"] = ta.roc(out["Close"], length=10)
    out["CCI20"] = ta.cci(out["High"], out["Low"], out["Close"], length=20)
    out["WILLR14"] = ta.willr(out["High"], out["Low"], out["Close"], length=14)

    st = ta.stoch(out["High"], out["Low"], out["Close"], k=14, d=3, smooth_k=3)
    if isinstance(st, pd.DataFrame) and st.shape[1] >= 2:
        out["STOCHk"] = st.iloc[:, 0]
        out["STOCHd"] = st.iloc[:, 1]
    else:
        out["STOCHk"] = out["STOCHd"] = np.nan

    out["OBV"] = ta.obv(out["Close"], out["Volume"])
    out["CMF20"] = ta.cmf(out["High"], out["Low"], out["Close"], out["Volume"], length=20)
    out["MFI14"] = mfi_custom(out["High"], out["Low"], out["Close"], out["Volume"], length=14)

    out["RET1"] = out["Close"].pct_change(1)
    out["RET5"] = out["Close"].pct_change(5)

    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def add_regime(df, cfg):
    out = df.copy()
    close = out["Close"]
    adx = out["ADX14"]
    ema = out["EMA200"].copy()
    if ema.notna().mean() < 0.5:
        ema = out["EMA50"]

    trend = adx >= float(cfg["adx_trend"])
    bull = trend & (close > ema)
    bear = trend & (close < ema)

    out["Regime"] = np.select([bull, bear, trend], ["TREND_BULL", "TREND_BEAR", "TREND_MIX"], default="RANGE")
    out["IS_BULL"] = (out["Regime"] == "TREND_BULL").astype(float)
    out["IS_BEAR"] = (out["Regime"] == "TREND_BEAR").astype(float)
    out["IS_RANGE"] = (out["Regime"] == "RANGE").astype(float)
    return out


# ----------------------------
# flow + zscore + regime features
# ----------------------------
def add_flow_to_main_df(df, cfg):
    flow_df = load_flow_features(df, cfg)
    out = df.join(flow_df, how="left")
    for c in flow_df.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    z20 = int(cfg.get("z_win_short", 20))
    z60 = int(cfg.get("z_win_long", 60))
    cwin = int(cfg.get("cum_flow_win", 5))

    # raw cumulative flows
    out["Total_NetLot_Cum5"] = out["Total_NetLot"].rolling(cwin, min_periods=1).sum()
    out["BofA_Cum5"] = out["BofA_NetLot"].rolling(cwin, min_periods=1).sum()
    out["Foreign_Selected_Cum5"] = out["Foreign_Selected_Total"].rolling(cwin, min_periods=1).sum()

    # zscore features
    out["Total_NetLot_Z20"] = rolling_zscore(out["Total_NetLot"], z20)
    out["BofA_Z20"] = rolling_zscore(out["BofA_NetLot"], z20)
    out["Foreign_Selected_Z20"] = rolling_zscore(out["Foreign_Selected_Total"], z20)

    out["Total_NetLot_Cum5_Z20"] = rolling_zscore(out["Total_NetLot_Cum5"], z20)
    out["BofA_Cum5_Z20"] = rolling_zscore(out["BofA_Cum5"], z20)
    out["Foreign_Selected_Cum5_Z20"] = rolling_zscore(out["Foreign_Selected_Cum5"], z20)

    out["Total_NetLot_Z60"] = rolling_zscore(out["Total_NetLot"], z60)
    out["BofA_Z60"] = rolling_zscore(out["BofA_NetLot"], z60)
    out["Foreign_Selected_Z60"] = rolling_zscore(out["Foreign_Selected_Total"], z60)

    # regime interaction features
    if "IS_BULL" not in out.columns:
        out["IS_BULL"] = 0.0
    if "IS_BEAR" not in out.columns:
        out["IS_BEAR"] = 0.0
    if "IS_RANGE" not in out.columns:
        out["IS_RANGE"] = 0.0

    out["Flow_in_BULL"] = out["Total_NetLot_Z20"] * out["IS_BULL"]
    out["Flow_in_BEAR"] = out["Total_NetLot_Z20"] * out["IS_BEAR"]
    out["Flow_in_RANGE"] = out["Total_NetLot_Z20"] * out["IS_RANGE"]

    out["BofA_in_BULL"] = out["BofA_Z20"] * out["IS_BULL"]
    out["BofA_in_BEAR"] = out["BofA_Z20"] * out["IS_BEAR"]
    out["Foreign_in_BEAR"] = out["Foreign_Selected_Z20"] * out["IS_BEAR"]

    # divergence-like features
    out["BuyPressure_vs_Return1"] = out["Total_NetLot_Z20"] - rolling_zscore(out["RET1"].fillna(0.0), z20)
    out["BofA_vs_Return1"] = out["BofA_Z20"] - rolling_zscore(out["RET1"].fillna(0.0), z20)

    for c in out.columns:
        if c not in ["Regime"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ----------------------------
# target
# ----------------------------
def add_target(df, cfg):
    out = df.copy()
    H = int(cfg["H"])
    k = float(cfg["k_atr"])
    atr = out["ATR"]
    fwd_close = out["Close"].shift(-H)
    fwd_move = (fwd_close - out["Close"])
    thr = k * atr
    y = np.where(fwd_move > thr, 1, np.where(fwd_move < -thr, -1, 0))
    out["Y"] = y

    # forward returns for impact analysis
    out["FWD_RET_1"] = out["Close"].shift(-1) / out["Close"] - 1.0
    out["FWD_RET_3"] = out["Close"].shift(-3) / out["Close"] - 1.0
    out["FWD_RET_5"] = out["Close"].shift(-5) / out["Close"] - 1.0
    return out


# ----------------------------
# feature list
# ----------------------------
def feature_cols(df):
    cand = [
        # technical
        "EMA20","EMA50","EMA200","RSI14","MACD","MACDh","MACDs","BBP",
        "ATR_PCT","ADX14","ROC10","CCI20","WILLR14","STOCHk","STOCHd",
        "OBV","CMF20","MFI14","RET1","RET5",

        # core flow
        "Total_NetLot","Total_Abs_Flow","Buyer_Count","Seller_Count",
        "NetLot_to_Volume","AbsFlow_to_Volume",
        "Total_NetLot_Lag1","Total_NetLot_Lag2","Total_NetLot_Lag3",
        "BofA_NetLot","BofA_Lag1","BofA_Lag2","BofA_Lag3","BofA_to_Volume",
        "Citibank_NetLot","Deutsche_Bank_NetLot","Goldman_Sachs_NetLot",
        "HSBC_NetLot","JP_Morgan_NetLot","UBS_NetLot","Foreign_Selected_Total",

        # zscore / cumulative
        "Total_NetLot_Cum5","BofA_Cum5","Foreign_Selected_Cum5",
        "Total_NetLot_Z20","BofA_Z20","Foreign_Selected_Z20",
        "Total_NetLot_Cum5_Z20","BofA_Cum5_Z20","Foreign_Selected_Cum5_Z20",
        "Total_NetLot_Z60","BofA_Z60","Foreign_Selected_Z60",

        # regime features
        "IS_BULL","IS_BEAR","IS_RANGE",
        "Flow_in_BULL","Flow_in_BEAR","Flow_in_RANGE",
        "BofA_in_BULL","BofA_in_BEAR","Foreign_in_BEAR",
        "BuyPressure_vs_Return1","BofA_vs_Return1",
    ]
    cols = [c for c in cand if c in df.columns]
    keep = [c for c in cols if df[c].notna().mean() >= 0.75]
    return keep


# ----------------------------
# walk-forward
# ----------------------------
def walk_forward_3class(df, feats, cfg):
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["Y", "ATR", "ATR_PCT", "Regime"]).copy()
    data = data.iloc[:-int(cfg["H"])].copy()
    n = len(data)
    if n < 600:
        raise ValueError(f"Veri az: n={n}")

    min_train = max(300, int(cfg["walk_train_min"]))
    min_train = min(min_train, n - 100)
    step = max(1, int(cfg["walk_retrain_step"]))

    pL = np.full(n, np.nan)
    pS = np.full(n, np.nan)
    pN = np.full(n, np.nan)

    model = None
    last_fit = -10**9

    for i in range(min_train, n):
        if model is None or (i - last_fit) >= step:
            X = data[feats].iloc[:i].values
            y = data["Y"].iloc[:i].values
            model = RandomForestClassifier(
                n_estimators=int(cfg["rf_n_estimators"]),
                max_depth=int(cfg["rf_max_depth"]),
                min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
                max_features=cfg["rf_max_features"],
                random_state=int(cfg["seed"]),
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
            model.fit(X, y)
            last_fit = i

        proba = model.predict_proba(data[feats].iloc[i:i+1].values)[0]
        cls = model.classes_

        def getp(target):
            if target in cls:
                return float(proba[np.where(cls == target)[0][0]])
            return 0.0

        pS[i] = getp(-1)
        pN[i] = getp(0)
        pL[i] = getp(1)

    out = data.copy()
    out["P_LONG"] = pL
    out["P_SHORT"] = pS
    out["P_NONE"] = pN
    return out


# ----------------------------
# sizing / signal
# ----------------------------
def edge_position_multiplier(p_long, p_short, cfg):
    edge = float(p_long) - float(p_short)
    thr = float(cfg["edge_thr"])
    if thr <= 0:
        return 1.0
    raw = abs(edge) / thr
    mult = 0.95 + 0.25 * raw
    cap = float(cfg.get("max_size_mult", 1.6))
    return float(np.clip(mult, 0.85, cap))

def compute_signal(row, cfg):
    pL = float(row["P_LONG"])
    pS = float(row["P_SHORT"])
    pN = float(row.get("P_NONE", 0.0))
    regime = str(row.get("Regime", "RANGE"))

    if cfg["use_regime"] and (not cfg["allow_range"]) and regime == "RANGE":
        return 0
    if pN > float(cfg["none_max"]):
        return 0

    edge = pL - pS

    # institution-aware bias
    flow_bias = 0.0
    flow_bias += 0.010 * float(row.get("Total_NetLot_Z20", 0.0))
    flow_bias += 0.015 * float(row.get("BofA_Z20", 0.0))
    flow_bias += 0.012 * float(row.get("Foreign_Selected_Z20", 0.0))
    flow_bias += 0.010 * float(row.get("Flow_in_BEAR", 0.0))
    flow_bias += 0.010 * float(row.get("BofA_in_BEAR", 0.0))
    flow_bias += 0.008 * float(row.get("BuyPressure_vs_Return1", 0.0))

    edge_adj = edge + flow_bias
    thr = float(cfg["edge_thr"])

    sig = 0
    if edge_adj >= thr:
        sig = +1
    elif edge_adj <= -thr:
        sig = -1

    if regime == "TREND_BEAR" and sig != 0:
        if abs(edge_adj) < float(cfg["bear_min_edge"]):
            return 0
        if sig == +1 and pL < float(cfg["bear_long_min_p"]):
            return 0
        if sig == -1 and pS < float(cfg["bear_short_min_p"]):
            return 0

    return sig


# ----------------------------
# backtest
# ----------------------------
def backtest(df, cfg):
    fee = (cfg["fee_bps"] + cfg["slippage_bps"]) / 10000.0

    cash = float(cfg["initial_cash"])
    equity_curve = []
    trades = []

    pos = 0
    entry = np.nan
    entry_i = None
    qty = 0.0
    stop = np.nan
    tp = np.nan

    last_signal_i = -10**9
    dates = df.index.to_list()

    for i in range(len(df)):
        row = df.iloc[i]
        close = float(row["Close"])
        atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
        atr_pct = float(row["ATR_PCT"]) if pd.notna(row["ATR_PCT"]) else np.nan

        equity = cash if pos == 0 else (cash + pos * qty * (close - entry))
        equity_curve.append(equity)

        if pos != 0:
            if entry_i is not None and (i - entry_i) >= int(cfg["time_stop"]):
                exitp = close
                pnl = pos * qty * (exitp - entry)
                cost = fee * (abs(qty) * entry + abs(qty) * exitp)
                cash = cash + pnl - cost
                trades.append((dates[i], "TIME_EXIT", pos, exitp, qty, stop, tp, cash))
                pos = 0; entry = np.nan; qty = 0; stop = np.nan; tp = np.nan; entry_i = None
                continue

            hi = float(row["High"])
            lo = float(row["Low"])
            hit_stop = (lo <= stop) if pos == 1 else (hi >= stop)
            hit_tp = (hi >= tp) if pos == 1 else (lo <= tp)

            if hit_stop or hit_tp:
                exitp = stop if hit_stop else tp
                pnl = pos * qty * (exitp - entry)
                cost = fee * (abs(qty) * entry + abs(qty) * exitp)
                cash = cash + pnl - cost
                trades.append((dates[i], "STOP" if hit_stop else "TAKEPROFIT", pos, exitp, qty, stop, tp, cash))
                pos = 0; entry = np.nan; qty = 0; stop = np.nan; tp = np.nan; entry_i = None
                continue

        if pd.isna(row.get("P_LONG", np.nan)) or pd.isna(row.get("P_SHORT", np.nan)) or pd.isna(atr) or pd.isna(atr_pct):
            continue
        if atr_pct < float(cfg["min_atr_pct"]):
            continue

        sig = compute_signal(row, cfg)

        if sig != 0 and (i - last_signal_i) < int(cfg["cooldown"]):
            sig = 0

        if cfg["allow_flip"] and pos != 0 and sig != 0 and sig != pos:
            pL = float(row["P_LONG"])
            pS = float(row["P_SHORT"])
            edge = pL - pS
            thr = float(cfg["edge_thr"]) + float(cfg["flip_edge_boost"])
            ok = (edge >= thr) if sig == +1 else (edge <= -thr)

            if ok:
                exitp = close
                pnl = pos * qty * (exitp - entry)
                cost = fee * (abs(qty) * entry + abs(qty) * exitp)
                cash = cash + pnl - cost
                trades.append((dates[i], "FLIP_EXIT", pos, exitp, qty, stop, tp, cash))

                last_signal_i = i
                pos = 0; entry = np.nan; qty = 0; stop = np.nan; tp = np.nan; entry_i = None

                if i + 1 < len(df):
                    next_open = float(df.iloc[i + 1]["Open"])
                    stop_dist = float(cfg["atr_stop_mult"]) * atr
                    if stop_dist > 0:
                        mult = edge_position_multiplier(pL, pS, cfg)
                        risk_cash = (cash * float(cfg["risk_per_trade"])) * mult
                        qty = risk_cash / stop_dist
                        entry = next_open
                        entry_i = i + 1
                        pos = sig

                        if pos == 1:
                            stop = entry - stop_dist
                            tp = entry + float(cfg["atr_tp_mult"]) * atr
                        else:
                            stop = entry + stop_dist
                            tp = entry - float(cfg["atr_tp_mult"]) * atr

                        cash = cash - fee * (abs(qty) * entry)
                        trades.append((dates[i + 1], "OPEN_LONG" if pos == 1 else "OPEN_SHORT", pos, entry, qty, stop, tp, cash))
                continue

        if pos == 0 and sig != 0:
            last_signal_i = i
            if i + 1 >= len(df):
                continue

            next_open = float(df.iloc[i + 1]["Open"])
            pL = float(row["P_LONG"])
            pS = float(row["P_SHORT"])
            mult = edge_position_multiplier(pL, pS, cfg)
            risk_cash = (cash * float(cfg["risk_per_trade"])) * mult

            stop_dist = float(cfg["atr_stop_mult"]) * atr
            if stop_dist <= 0:
                continue

            qty = risk_cash / stop_dist
            entry = next_open
            entry_i = i + 1
            pos = sig

            if pos == 1:
                stop = entry - stop_dist
                tp = entry + float(cfg["atr_tp_mult"]) * atr
            else:
                stop = entry + stop_dist
                tp = entry - float(cfg["atr_tp_mult"]) * atr

            cash = cash - fee * (abs(qty) * entry)
            trades.append((dates[i + 1], "OPEN_LONG" if pos == 1 else "OPEN_SHORT", pos, entry, qty, stop, tp, cash))

    eq = pd.Series(equity_curve, index=df.index, name="Equity")
    tr = pd.DataFrame(
        trades,
        columns=["Date", "Action", "Side", "Price", "Qty", "Stop", "TP", "Cash"]
    ).set_index("Date") if trades else pd.DataFrame(columns=["Action", "Side", "Price", "Qty", "Stop", "TP", "Cash"])
    return eq, tr

def perf(eq):
    if len(eq) < 50:
        return {}
    rets = eq.pct_change().dropna()
    sharpe = np.sqrt(252) * rets.mean() / (rets.std() + 1e-12)
    dd = (eq / eq.cummax() - 1.0).min()
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0])**(1 / years) - 1 if years > 0 else np.nan
    return {"Sharpe": float(sharpe), "MaxDD": float(dd), "TotalRet": float(total), "CAGR": float(cagr)}


# ----------------------------
# calibration
# ----------------------------
def calibrate(oos_df, cfg):
    lo = int(cfg["target_trades_lo"])
    hi = int(cfg["target_trades_hi"])

    edge_grid = [0.015, 0.02, 0.025, 0.03]
    none_grid = [0.60, 0.65, 0.70]
    hold_grid = [2, 3]
    cool_grid = [0, 1]
    flip_boost_grid = [0.003, 0.005, 0.01]

    base = dict(cfg)
    best = None

    for edge_thr in edge_grid:
        for none_max in none_grid:
            for time_stop in hold_grid:
                for cooldown in cool_grid:
                    for fb in flip_boost_grid:
                        c = dict(base)
                        c["edge_thr"] = float(edge_thr)
                        c["none_max"] = float(none_max)
                        c["time_stop"] = int(time_stop)
                        c["cooldown"] = int(cooldown)
                        c["flip_edge_boost"] = float(fb)

                        eq, tr = backtest(oos_df, c)
                        m = perf(eq)
                        opens = int((tr["Action"].astype(str).str.startswith("OPEN")).sum()) if len(tr) else 0

                        if opens < lo:
                            pen = (lo - opens) / max(1, lo)
                        elif opens > hi:
                            pen = (opens - hi) / max(1, hi)
                        else:
                            pen = 0.0

                        score = (m.get("Sharpe", -9e9)
                                 - 0.60 * abs(m.get("MaxDD", 0.0))
                                 - 6.0 * pen)

                        if best is None or score > best["score"]:
                            best = {"score": score, "cfg": c, "m": m, "opens": opens, "rows": len(tr)}
    return best


# ----------------------------
# tomorrow probs
# ----------------------------
def tomorrow_probs(full_df, feats, cfg):
    data = full_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + ["Y"]).copy()
    data = data.iloc[:-int(cfg["H"])].copy()
    if len(data) < 600:
        return None

    X = data[feats].values
    y = data["Y"].values
    model = RandomForestClassifier(
        n_estimators=int(cfg["rf_n_estimators"]),
        max_depth=int(cfg["rf_max_depth"]),
        min_samples_leaf=int(cfg["rf_min_samples_leaf"]),
        max_features=cfg["rf_max_features"],
        random_state=int(cfg["seed"]),
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X, y)

    last = full_df.dropna(subset=feats).iloc[-1]
    proba = model.predict_proba(last[feats].values.reshape(1, -1))[0]
    cls = model.classes_

    def getp(target):
        if target in cls:
            return float(proba[np.where(cls == target)[0][0]])
        return 0.0

    return {
        "date": full_df.index[-1],
        "P_LONG": getp(1),
        "P_SHORT": getp(-1),
        "P_NONE": getp(0),
        "Regime": str(last.get("Regime", "RANGE")),
        "ATR_PCT": float(last.get("ATR_PCT", np.nan)),
        "Close": float(full_df["Close"].iloc[-1]),
        "BofA_NetLot": float(last.get("BofA_NetLot", 0.0)),
        "Total_NetLot": float(last.get("Total_NetLot", 0.0)),
        "Foreign_Selected_Total": float(last.get("Foreign_Selected_Total", 0.0)),
        "BofA_Z20": float(last.get("BofA_Z20", 0.0)),
        "Total_NetLot_Z20": float(last.get("Total_NetLot_Z20", 0.0)),
        "Foreign_Selected_Z20": float(last.get("Foreign_Selected_Z20", 0.0)),
    }


# ----------------------------
# open position
# ----------------------------
def open_position_pnl(oos_df, tr, cfg):
    if tr is None or len(tr) == 0:
        return None
    last = tr.iloc[-1]
    if not str(last["Action"]).startswith("OPEN"):
        return None

    side = int(last["Side"])
    entry = float(last["Price"])
    qty = float(last["Qty"])
    cur = float(oos_df["Close"].iloc[-1])
    fee = (cfg["fee_bps"] + cfg["slippage_bps"]) / 10000.0

    pnl = side * qty * (cur - entry)
    exit_cost = fee * (abs(qty) * cur)

    return {
        "entry_date": last.name,
        "side": side,
        "entry": entry,
        "current": cur,
        "qty": qty,
        "unreal_pnl_before_exit_cost": pnl,
        "exit_cost_est": exit_cost,
        "unreal_pnl_after_exit_cost": pnl - exit_cost,
        "stop": float(last["Stop"]) if pd.notna(last["Stop"]) else np.nan,
        "tp": float(last["TP"]) if pd.notna(last["TP"]) else np.nan,
    }


# ----------------------------
# impact analysis
# ----------------------------
def impact_after_buy_analysis(df):
    tmp = df.copy()

    conditions = {
        "TOTAL_NET_BUY": tmp["Total_NetLot"] > 0,
        "BOFA_NET_BUY": tmp["BofA_NetLot"] > 0,
        "FOREIGN_SELECTED_BUY": tmp["Foreign_Selected_Total"] > 0,
        "TOTAL_NET_BUY_ZPOS": tmp["Total_NetLot_Z20"] > 0,
        "BOFA_BUY_ZPOS": tmp["BofA_Z20"] > 0,
        "FOREIGN_BUY_ZPOS": tmp["Foreign_Selected_Z20"] > 0,
        "BOFA_BUY_IN_BEAR": (tmp["BofA_NetLot"] > 0) & (tmp["Regime"] == "TREND_BEAR"),
        "FOREIGN_BUY_IN_BEAR": (tmp["Foreign_Selected_Total"] > 0) & (tmp["Regime"] == "TREND_BEAR"),
    }

    rows = []
    for name, cond in conditions.items():
        sub = tmp.loc[cond, ["FWD_RET_1", "FWD_RET_3", "FWD_RET_5"]].copy()
        n = len(sub)
        if n == 0:
            rows.append({
                "Condition": name, "Count": 0,
                "Avg_Fwd1_%": np.nan, "Avg_Fwd3_%": np.nan, "Avg_Fwd5_%": np.nan
            })
            continue

        rows.append({
            "Condition": name,
            "Count": int(n),
            "Avg_Fwd1_%": float(sub["FWD_RET_1"].mean() * 100),
            "Avg_Fwd3_%": float(sub["FWD_RET_3"].mean() * 100),
            "Avg_Fwd5_%": float(sub["FWD_RET_5"].mean() * 100),
        })

    return pd.DataFrame(rows).sort_values(["Avg_Fwd5_%", "Avg_Fwd3_%"], ascending=False)


# ============================================================
# RUN
# ============================================================
raw = load_yfinance(CFG["ticker"], CFG["period"], CFG["interval"])
print("Loaded:", raw.shape, "| Date:", raw.index.min().date(), "->", raw.index.max().date())
print("Last Close:", float(raw["Close"].iloc[-1]))

raw = add_indicators(raw, CFG)
raw = add_regime(raw, CFG)
raw = add_flow_to_main_df(raw, CFG)
raw = add_target(raw, CFG)

F = feature_cols(raw)
print("Features used:", len(F), F)

flow_feats_used = [c for c in F if any(k in c for k in ["NetLot", "BofA", "Foreign", "Buyer_Count", "Seller_Count", "Flow_", "_Z20", "_Z60", "_Cum5"])]
print("Flow/ZScore/Regime features used:", len(flow_feats_used), flow_feats_used)

oos = walk_forward_3class(raw, F, CFG)
print("\nOOS rows:", oos.shape)

oos_bt = oos.dropna(subset=["P_LONG", "P_SHORT", "P_NONE", "ATR", "ATR_PCT", "Regime"]).copy()

if CFG["do_calibrate"]:
    best = calibrate(oos_bt, CFG)
    if best:
        CFG.update(best["cfg"])
        print("\n=== CALIBRATION RESULT (TARGET 400-600 OPENS) ===")
        print("OPEN trades:", best["opens"], "| Log rows:", best["rows"],
              "| Sharpe:", f"{best['m'].get('Sharpe', np.nan):.2f}",
              "| MaxDD:", f"{best['m'].get('MaxDD', np.nan) * 100:.2f}%",
              "| TotalRet:", f"{best['m'].get('TotalRet', np.nan) * 100:.2f}%",
              "| CAGR:", f"{best['m'].get('CAGR', np.nan) * 100:.2f}%")
        print("Chosen params:",
              f"edge_thr={CFG['edge_thr']}, none_max={CFG['none_max']}, time_stop={CFG['time_stop']}, cooldown={CFG['cooldown']}, flip_boost={CFG['flip_edge_boost']}")

eq, tr = backtest(oos_bt, CFG)
m = perf(eq)
opens = int((tr["Action"].astype(str).str.startswith("OPEN")).sum()) if len(tr) else 0

print("\n=== OOS BACKTEST (flow + zscore + regime features) ===")
print("OPEN trades:", opens, "| Log rows:", len(tr),
      "| Final Equity:", float(eq.iloc[-1]),
      "| MaxDD:", f"{m.get('MaxDD', np.nan) * 100:.2f}%")
print("Sharpe:", f"{m.get('Sharpe', np.nan):.2f}",
      "| TotalRet:", f"{m.get('TotalRet', np.nan) * 100:.2f}%",
      "| CAGR:", f"{m.get('CAGR', np.nan) * 100:.2f}%")

tp = tomorrow_probs(raw, F, CFG)
print("\n=== TOMORROW PROBABILITIES (NEXT", CFG["H"], "DAYS MOVE) ===")
if tp is None:
    print("Yeterli veri yok / feature NaN çok.")
else:
    print("Last date:", tp["date"].date(), "| Close:", f"{tp['Close']:.2f}", "| Regime:", tp["Regime"], "| ATR_PCT:", f"{tp['ATR_PCT']:.4f}")
    print(f"P_LONG  : {tp['P_LONG']:.4f}")
    print(f"P_SHORT : {tp['P_SHORT']:.4f}")
    print(f"P_NONE  : {tp['P_NONE']:.4f}")

    edge = tp["P_LONG"] - tp["P_SHORT"]
    stance = "BEKLE"

    if (not np.isnan(tp["ATR_PCT"])) and tp["ATR_PCT"] >= CFG["min_atr_pct"] and tp["P_NONE"] <= CFG["none_max"]:
        if tp["Regime"] == "TREND_BEAR":
            if abs(edge) >= CFG["bear_min_edge"]:
                if edge >= CFG["edge_thr"] and tp["P_LONG"] >= CFG["bear_long_min_p"]:
                    stance = "LONG"
                elif edge <= -CFG["edge_thr"] and tp["P_SHORT"] >= CFG["bear_short_min_p"]:
                    stance = "SHORT"
        else:
            if edge >= CFG["edge_thr"]:
                stance = "LONG"
            elif edge <= -CFG["edge_thr"]:
                stance = "SHORT"

    mult = edge_position_multiplier(tp["P_LONG"], tp["P_SHORT"], CFG)
    print("EDGE:", f"{edge:.4f}", "| edge_thr:", CFG["edge_thr"], "| none_max:", CFG["none_max"], "| size_mult:", f"{mult:.2f}x")
    print("Total_NetLot:", f"{tp['Total_NetLot']:.0f}", "| Total_NetLot_Z20:", f"{tp['Total_NetLot_Z20']:.3f}")
    print("BofA_NetLot:", f"{tp['BofA_NetLot']:.0f}", "| BofA_Z20:", f"{tp['BofA_Z20']:.3f}")
    print("Foreign_Selected_Total:", f"{tp['Foreign_Selected_Total']:.0f}", "| Foreign_Selected_Z20:", f"{tp['Foreign_Selected_Z20']:.3f}")
    print("Bear filter:", f"bear_min_edge={CFG['bear_min_edge']}, bear_long_min_p={CFG['bear_long_min_p']}, bear_short_min_p={CFG['bear_short_min_p']}")
    print("Recommended stance:", stance)

print("\n=== BUY IMPACT ANALYSIS (AVERAGE PRICE EFFECT AFTER BUYS) ===")
impact_tbl = impact_after_buy_analysis(raw)
display(impact_tbl.round(3))

print("\nLast trades (tail 20):")
display(tr.tail(20))

op = open_position_pnl(oos_bt, tr, CFG)
print("\n=== OPEN POSITION (IF ANY) ===")
if op is None:
    print("Açık pozisyon yok (ya da son işlem EXIT).")
else:
    side_txt = "LONG" if op["side"] == 1 else "SHORT"
    print(f"Open position: {side_txt} | Entry date: {op['entry_date'].date()} | Entry: {op['entry']:.2f} | Current: {op['current']:.2f}")
    print(f"Qty: {op['qty']:.2f} | Stop: {op['stop']:.2f} | TP: {op['tp']:.2f}")
    print(f"Unrealized PnL (before exit cost): {op['unreal_pnl_before_exit_cost']:.2f}")
    print(f"Exit cost est: {op['exit_cost_est']:.2f}")
    print(f"Unrealized PnL (after exit cost): {op['unreal_pnl_after_exit_cost']:.2f}")

if CFG["plot"] and _HAS_PLOTLY:
    d = oos_bt.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["P_LONG"], name="P_LONG"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["P_SHORT"], name="P_SHORT"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["P_NONE"], name="P_NONE"), row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_layout(height=850, title=f"{CFG['ticker']} - OOS probabilities + flow zscore + regime features")
    fig.show()
elif CFG["plot"]:
    print("\nPlotly yoksa grafik çizilmez.")
