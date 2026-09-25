# bist-quant-system
AI-assisted quantitative trading and institutional flow analysis system for BIST stocks using Python and machine learning.

- `analysis.py`: SASA/BIST günlük sistem (kurum akışı, z-skor, rejim; tek hücre).
- `meta_labeling/`: Meta-Labeling + Triple Barrier Method ile uçtan uca, aşırı öğrenmeye dirençli ML pipeline'ı (aşağıda).
- `meta_labeling/research/`: backtest sonucunun güvenilirliğini test eden araştırma çerçevesi (`config.yaml`, aşağıda).

## Meta-Labeling Pipeline (`meta_labeling/`)

Marcos López de Prado'nun *Advances in Financial Machine Learning* (AFML) kitabındaki
metodolojiyi izleyen, modüler bir araştırma/üretim pipeline'ı. Bağımlılıklar yalnızca
`numpy`, `pandas`, `scikit-learn` ve `lightgbm`.

```bash
pip install -r requirements.txt
python -m meta_labeling                                   # sentetik veri, varsayılan ayarlar
python -m meta_labeling --primary bollinger --model rf    # alternatif birincil model / meta-model
python -m meta_labeling --csv SASA.csv --threshold 0.55   # gerçek veri (Date,Open,High,Low,Close,Volume)
python -m pytest                                          # testler
```

**Jupyter Lab:** `notebooks/meta_labeling_pipeline.ipynb` (adım adım hücreler + equity grafiği)
veya tek hücreye yapıştırmak için `notebooks/meta_labeling_tek_hucre.py`. İkisi de kendi kendine
yeter (paket import'u gerektirmez) ve `python scripts/build_notebook.py` ile paketten üretilir.

```python
from meta_labeling import MetaLabelingPipeline, PipelineConfig

pipe = MetaLabelingPipeline(PipelineConfig())
result = pipe.run()                     # veya pipe.run(ohlcv_df)
print(result.summary())
result.events                           # t0, t1, bariyer, meta-etiket, ağırlık, OOS P(Y=1), bet size
pipe.score_events(ohlcv_df, result.final_model, last_n=1)   # canlı karar: side, P(Y=1), size
```

### Akış

| Adım | Modül | AFML | Açıklama |
|---|---|---|---|
| 1. Veri & birincil model | `data.py`, `primary.py`, `sampling.py` | 2.5, 3.6 | Rejim değiştiren GARCH simülasyonu; EMA kesişimi / Bollinger kırılımı ile `side ∈ {-1,0,+1}`; CUSUM filtresiyle olay örnekleme (yüksek recall) |
| 2. Triple Barrier | `volatility.py`, `labeling.py` | 3.1–3.4 | EWMA log-getiri volatilitesi; `pt·σ` kâr al, `sl·σ` zarar kes, `h` bar dikey bariyer; ilk temas `t1` |
| 3. Meta-etiket | `labeling.py` | 3.6 | `Y=1` ⇔ `side·(P_t1/P_t0−1) − maliyet > 0` |
| 4. Öznitelikler | `features.py` | — | Yönden bağımsız rejim göstergeleri: RSI, ATR%, ATR/vol oranı, vol oranı, BB genişliği, otokorelasyon, verimlilik oranı, |momentum| z-skoru, çarpıklık, hacim z-skoru |
| 5. Doğrulama | `cv.py`, `sample_weights.py` | 4, 7 | Benzersizlik ağırlıkları; Purged K-Fold (teşhis) + Purged walk-forward (OOS olasılıklar) |
| 6. Meta-model & karar | `model.py`, `sizing.py`, `backtest.py` | 6, 10, 14 | Dengelenmiş LightGBM/RF; `P(Y=1) > 0.55` filtresi; `m = 2Φ(z)−1` bet sizing; Win Rate, Sharpe, PSR karşılaştırması |

### Sızıntı (leakage) kontrolleri

- **Purging:** Etiket aralığı `[t0, t1]` test setinin `[min t0, max t1]` aralığıyla kesişen eğitim gözlemleri atılır.
- **Embargo:** Test bitişinden sonraki `embargo_pct × toplam süre` içinde başlayan eğitim gözlemleri atılır (rolling özniteliklerin seri korelasyonu).
- **Walk-forward OOS:** Backtest'te kullanılan her olasılık, yalnızca o andan önce *kapanmış* etiketlerle eğitilmiş bir modelden gelir.
- **Look-ahead testi:** `tests/test_features_weights_sizing.py` öznitelikleri kesilmiş veriyle yeniden hesaplayıp değişmediklerini doğrular.
- **Örtüşme:** Eğitim ağırlıkları ve ağaç alt örneklem oranı ortalama benzersizliğe göre ayarlanır.
- **Ex-ante parametreler:** Eşik ve hiperparametreler OOS sonuçlarına göre ayarlanmaz (`config.py`).

### Örnek çıktı (sentetik veri, varsayılan ayarlar)

```
[2] Purged walk-forward OOS
  OOS olay: 1262 | AUC: 0.554 | birincil precision (baz): 50.5% → meta precision: 56.1% | kapsama: 34.2%

[3] Strateji karşılaştırması (OOS, maliyet 5 bps/yön)
                                   İşlem WinRate Ort.İşlem İşlemSR Sharpe     PSR Yıl.Getiri   MaxDD
Birincil (filtresiz)                1262   50.5%    -0.13%   -0.34  -0.01   48.2%      -0.3%  -73.6%
Meta filtre (p>0.55)                 431   56.1%    +0.29%    0.48   0.53   97.4%       7.8%  -45.7%
Meta filtre + bet sizing (p>0.55)    431   56.1%    +0.12%    0.79   0.74   99.7%       2.5%   -9.3%
```

Tek bir fiyat yolu tek bir gerçekleşmedir: 10 farklı simülasyon tohumunda filtre, win rate'i
10'un 9'unda artırmış (ort. %48.7 → %52.9), portföy Sharpe'ını 9'unda iyileştirmiştir
(ort. −0.29 → +0.16). Sentetik verideki iyileşme gerçek piyasada garanti değildir; gerçek
veride aynı pipeline'ı çalıştırıp sonuçları PSR/Deflated Sharpe ile değerlendirin.

## Research Framework (`meta_labeling/research/`)

Amaç backtest performansını maksimize etmek değil, **sonucun güvenilir olup olmadığını test etmek**.
Tüm parametreler `config.yaml` içindedir; bilinmeyen anahtarlar hata verir.

```bash
python -m meta_labeling.research --config config.yaml             # tam rapor (~30 sn)
python -m meta_labeling.research --config config.yaml --universe  # config'deki tüm hisseler
```

Jupyter: `notebooks/research_framework.ipynb` (01_config … 24_final_report). Çıktılar
`research_output/<TICKER>/` altına yazılır: `research_report.md` (25 bölüm), `final_oos_results.csv`,
`robustness_summary.csv`, grafikler ve `research_output/experiments/` deney kayıtları.

| Modül | İçerik |
|---|---|
| `leakage.py` | `run_leakage_audit()`: öznitelik/vol/sinyal/CUSUM/ADV/rejim kesme testi, etiket sırası, CV/CPCV bölmeleri, kalibrasyon; PASS/FAIL + feature + timestamp + neden |
| `execution.py` | Sinyal close(t) → işlem open(t+1) (varsayılan) veya close; `komisyon + spread/2 + slippage + k·sqrt(emir/ADV)`; risk limitleri |
| `modeling.py` | Execution fiyatlı meta-etiketler, purged walk-forward, geçmişe dayalı Platt/Isotonic kalibrasyon, sizing (equal, vol, Prado, prob×vol, fraksiyonel Kelly) |
| `session.py` | Aşama aşama `ResearchSession` (benchmark A-I, maliyet, eşik ızgarası, long/short, rejim, dönem, önem + ablation, bootstrap, CPCV, placebo) |
| `metrics.py` | CAGR, Sharpe, Sortino, Calmar, PF, turnover, exposure, PSR, DSR, drawdown dönemleri, blok bootstrap |
| `cpcv.py` | Combinatorial Purged CV ve backtest yolu birleştirme |
| `regimes.py` | Genişleyen kantillerle geleceğe bakmayan vol / trend / piyasa rejimleri |
| `universe.py` | Çoklu hisse (simulated / csv / yfinance), veri doğrulama, tarihsel endeks üyeliği (survivorship) |
| `tracking.py`, `report.py` | Deney kaydı + `reproduce()`, otomatik rapor ve ex-ante robustness kriterleri |

Robustness kriterleri (`config.yaml → criteria`) sonuçlara bakmadan tanımlıdır. Placebo testlerinde
en küçük p-değeri 1/(1+tekrar) olduğundan, tekrar sayısı α'ya ulaşmaya yetmiyorsa test FAIL değil
**INCONCLUSIVE** olarak raporlanır.
