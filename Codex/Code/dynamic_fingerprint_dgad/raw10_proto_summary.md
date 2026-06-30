# Raw10 No-context Prototype-router MoE Summary

## 1. Method Name

Current short name:

**raw10 proto** = raw 10-dimensional dynamic fingerprint + no-context mechanism prototype-router MoE.

Code model type:

```bash
--model-type mechanism_proto_router_moe
--no-mechanism-context
```

## 2. Input Fingerprint

For each temporal edge \(e_t=(u,v,t)\), the model uses 10 dynamic fingerprint features:

| index | feature | meaning |
|---:|---|---|
| 0 | nCN | normalized common neighbors, local closure strength |
| 1 | CP1 | first-order Chebyshev/spectral proximity |
| 2 | CP2 | second-order spectral proximity |
| 3 | CP3 | third-order spectral proximity |
| 4 | CP4 | fourth-order spectral proximity |
| 5 | d_nCN | temporal change of nCN |
| 6 | d_CP1 | temporal change of CP1 |
| 7 | d_CP2 | temporal change of CP2 |
| 8 | d_CP3 | temporal change of CP3 |
| 9 | d_CP4 | temporal change of CP4 |

This version does **not** use:

- edge surprise
- node activity
- snapshot-relative features
- Atlas local static features
- high-order multiview features
- full context embedding

So it is a clean 10-feature mechanism diagnosis baseline.

## 3. Four Mechanism Experts

The model has 4 experts. Each expert only sees the fingerprint group that matches its anomaly mechanism.

| expert | input features | intended mechanism |
|---|---|---|
| Local Structure | nCN, d_nCN | weak local closure or local relation change |
| Low-order Spectral | CP1, CP2, d_CP1, d_CP2 | low-order / near-neighborhood structural abnormality |
| High-order Spectral | CP3, CP4, d_CP3, d_CP4 | high-order / more global structural abnormality |
| Temporal Delta | d_nCN, d_CP1, d_CP2, d_CP3, d_CP4 | overall temporal shift abnormality |

Each expert outputs:

1. a representation \(h_i\)
2. an expert anomaly logit \(s_i\)

## 4. Prototype Evidence Router

Each expert maintains two EMA prototypes:

- normal prototype \(p_i^N\)
- anomaly prototype \(p_i^A\)

The prototypes are registered as buffers, not optimized parameters.

For expert \(i\), router evidence is:

\[
e_i = \cos(h_i, p_i^A) - \cos(h_i, p_i^N)
\]

Router weights are:

\[
w_i = \operatorname{softmax}(e_i / \tau)
\]

Current default:

\[
\tau = 1.0
\]

Final anomaly logit:

\[
s = \sum_i w_i s_i
\]

Final anomaly score:

\[
\sigma(s)
\]

## 5. Prototype Update

During training only, after each training batch:

1. compute expert representations
2. split batch by label
3. compute mean representation of normal and anomaly samples per expert
4. update prototype by EMA

\[
p \leftarrow m p + (1-m)\bar{h}
\]

Current default:

\[
m = 0.9
\]

Validation and target test do not update prototypes.

## 6. Training Setting

Main clean setting:

```bash
python -m dynamic_fingerprint_dgad.training.run_experiment \
  --source MOOC Wikipedia \
  --target btc_otc_all_injection \
  --num-snapshots 50 \
  --history-window 5 \
  --cheb-order 4 \
  --epochs 50 \
  --hidden-dim 64 \
  --num-layers 2 \
  --num-heads 4 \
  --chunk-size 512 \
  --model-type mechanism_proto_router_moe \
  --router-temperature 1.0 \
  --prototype-momentum 0.9 \
  --no-mechanism-context \
  --loss-type bce \
  --sampler balanced \
  --balanced-neg-ratio 1 \
  --feature-ablation full \
  --feature-view full \
  --val-ratio 0.2 \
  --early-stop-metric auroc \
  --patience 8 \
  --seed 0
```

Loss:

\[
L = \operatorname{BCEWithLogitsLoss}(s, y)
\]

Balanced sampler:

- anomaly : normal = 1 : 1
- batch size = 512
- about 256 anomaly edges and 256 normal edges per batch
- if one class is insufficient, sampling uses replacement

## 7. Source Data

Training source datasets:

| source | rows | nodes | snapshots | anomalies | anomaly rate |
|---|---:|---:|---:|---:|---:|
| MOOC | 411,749 | 7,144 | 50 | 4,066 | 0.9875% |
| Wikipedia | 157,474 | 9,227 | 50 | 217 | 0.1378% |

Source split:

| split | edges | anomalies |
|---|---:|---:|
| train | 455,378 | 3,426 |
| validation | 113,845 | 857 |

## 8. Main BTC-OTC Result

Target:

`btc_otc_all_injection`

Output file:

`results/sampler_ratio_proto_router_btc_otc_seed0/ratio_1/metrics.json`

| metric | value |
|---|---:|
| best epoch | 12 |
| source val AUROC | 0.6277 |
| source val AUPRC | 0.0104 |
| target AUROC | 0.6968 |
| target AUPRC | 0.1433 |
| P@anom | 0.1318 |
| P@1% | 0.0485 |

Router mean:

| Local | Low-order | High-order | Temporal |
|---:|---:|---:|---:|
| 0.2500 | 0.2472 | 0.2516 | 0.2511 |

Per-expert target metrics:

| expert | AUROC | AUPRC | P@1% |
|---|---:|---:|---:|
| Local Structure | 0.4441 | 0.0907 | 0.0153 |
| Low-order Spectral | 0.6029 | 0.1123 | 0.0561 |
| High-order Spectral | 0.6990 | 0.1438 | 0.0842 |
| Temporal Delta | 0.4970 | 0.1091 | 0.0051 |

Key observation:

**High-order Spectral is the strongest expert on BTC-OTC.** The final MoE is close to the High-order expert, but does not exceed it.

## 9. Target-wise Raw10 Proto Results

These are seed 0, all-injection targets, AUROC early stopping.

Output directory:

`results/mechanism_proto_router_moe_nocontext_bestval50_bce/`

| target | final AUROC | final AUPRC | best expert | best expert AUROC | best expert AUPRC |
|---|---:|---:|---|---:|---:|
| uci_all | 0.7059 | 0.1513 | High-order | 0.7067 | 0.1516 |
| btc_otc_all | 0.6968 | 0.1433 | High-order | 0.6990 | 0.1438 |
| btc_alpha_all | 0.7206 | 0.1534 | High-order | 0.7231 | 0.1540 |
| email_dnc_all | 0.9142 | 0.3904 | High-order | 0.9174 | 0.4019 |
| as_topology_all | 0.7501 | 0.1746 | High-order | 0.7779 | 0.1848 |

Per-target expert AUROC/AUPRC:

| target | Local | Low-order | High-order | Temporal |
|---|---:|---:|---:|---:|
| uci_all | 0.4752 / 0.0905 | 0.5461 / 0.0993 | 0.7067 / 0.1516 | 0.4855 / 0.1042 |
| btc_otc_all | 0.4441 / 0.0907 | 0.6029 / 0.1123 | 0.6990 / 0.1438 | 0.4970 / 0.1091 |
| btc_alpha_all | 0.4331 / 0.0908 | 0.6183 / 0.1161 | 0.7231 / 0.1540 | 0.4999 / 0.1135 |
| email_dnc_all | 0.1858 / 0.0905 | 0.8976 / 0.3446 | 0.9174 / 0.4019 | 0.5966 / 0.1626 |
| as_topology_all | 0.3207 / 0.0909 | 0.6985 / 0.1478 | 0.7779 / 0.1848 | 0.5027 / 0.1208 |

## 10. Main Conclusion

1. Raw10 proto is a clean and interpretable mechanism baseline.
2. Across all current targets, the strongest mechanism is consistently **High-order Spectral Dynamics**.
3. The prototype router is close to uniform, so it does not strongly select the best expert.
4. Final MoE performance is usually close to, but slightly below, the best High-order expert.
5. Local and Temporal experts are weak in raw10.
6. Later Atlas-local variants can improve Local expert, but they often damage High-order behavior and do not improve final performance.

Current practical takeaway:

**The stable signal is not “more handcrafted features”; it is CP3/CP4 and their temporal changes.**

This suggests that the next strong direction should either:

- keep High-order Spectral Dynamics as the main mechanism, or
- design a better training objective / scoring rule for top anomaly retrieval,

rather than continuing to add many local or relative features.

