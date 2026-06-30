# Context-aware Representation MoE 实验指令

请实现一个新的最小实验线：**Context-aware Representation MoE**。

目标是验证组会反馈里的核心问题：当前 prototype-router 主要是 source-prototype + score-level fusion，不够 target-adaptive，也很难超过 best expert。因此这次要做 **representation-level fusion**，并引入 **target unlabeled context**。

先只做：

```text
target: btc_otc
seed: 0
feature: raw10
```

不要加 Atlas-local，不要加新 loss，不要扩展到 5 个 target。

## 1. 实验目标

当前 raw10 no-context prototype-router baseline：

```text
target: btc_otc
seed: 0

raw10 proto-router final:
AUROC 0.6968
AUPRC 0.1433

raw10 High-order expert:
AUROC 0.6990
AUPRC 0.1438
```

当前问题：

```text
prototype-router 的权重主要来自 source normal/anomaly prototypes
target 测试时没有显式利用 target 自身的无标签分布
score-level MoE 基本只能追平 best expert，很难真正超过 High-order expert
```

这次新模型要回答：

> 如果把 MoE 从“分数加权”改成“representation 协同”，并加入 target unlabeled context，final 是否能超过 High-order single expert？

## 2. 输入特征

使用现有 raw10，不改 feature extractor：

```text
x(u,v) =
[
  nCN,
  CP1, CP2, CP3, CP4,
  d_nCN,
  d_CP1, d_CP2, d_CP3, d_CP4
]
```

不要加：

```text
Atlas-local
rel20
snapshot-relative
high-order multiview
residual
edge_surprise
node_activity
```

## 3. Expert 分组

这次去掉单独的 Temporal expert，改成 3 个 expert。

delta features 保留在各自机制 expert 里。

```text
Local expert input:
[nCN, d_nCN]
input dim = 2
```

```text
Low-order expert input:
[CP1, CP2, d_CP1, d_CP2]
input dim = 4
```

```text
High-order expert input:
[CP3, CP4, d_CP3, d_CP4]
input dim = 4
```

## 4. Expert 模型

每个 expert 是一个 mechanism encoder，不再主要作为单独打分器。

建议结构：

```text
Expert_i:
Linear(input_dim, hidden_dim)
LayerNorm(hidden_dim)
GELU
Dropout
Linear(hidden_dim, hidden_dim)
LayerNorm(hidden_dim)
GELU
```

默认：

```text
hidden_dim = 64
dropout = 0.1
```

输出：

```text
h_local ∈ R^64
h_low   ∈ R^64
h_high  ∈ R^64
```

也就是说：

```text
Local: 2 -> 64
Low-order: 4 -> 64
High-order: 4 -> 64
```

## 5. Representation-level fusion

不要做：

```text
final_logit = weighted sum of expert logits
```

而是做 representation-level interaction。

对每条边，先得到 3 个 expert tokens：

```text
tokens_edge = [
  h_local,
  h_low,
  h_high
]
```

然后送入一个小 Transformer encoder 或 attention fusion layer。

建议第一版：

```text
num_layers = 1
num_heads = 4
hidden_dim = 64
feedforward_dim = 128
dropout = 0.1
```

输出后，可以取：

```text
h_fused = mean(tokens_after_transformer, dim=expert_token)
```

或者加一个 learnable `[CLS]` token，取 CLS 输出。为了最小实现，建议先用 mean pooling。

最终分类头：

```text
final_head:
LayerNorm(64)
Linear(64, 32)
GELU
Dropout
Linear(32, 1)
```

输出：

```text
final_logit
```

训练 loss：

```text
BCEWithLogitsLoss(final_logit, label)
```

## 6. Target unlabeled context 版本

除了 no-context representation MoE，还要做 context-aware 版本。

请实现参数：

```text
--context-mode none|target_mean
```

### 6.1 context-mode=none

只有 3 个 expert tokens：

```text
[h_local, h_low, h_high]
```

进入 Transformer fusion。

这是 ablation，用来判断 representation-level fusion 本身是否有效。

### 6.2 context-mode=target_mean

使用 target unlabeled feature distribution 构造 context token。

注意：不能使用 target label，只能使用 target features。

建议做法：

1. 在提取完 target raw10 features 后，取 target train/test 全部或 evaluation pool 的 `x`，不看标签。

2. 计算：

```text
target_mean = mean(x_target, dim=0)
target_std  = std(x_target, dim=0)
```

3. 拼接：

```text
context_stats = [target_mean, target_std]
dim = 20
```

4. 用一个 MLP 映射成 context token：

```text
context_encoder:
Linear(20, 64)
LayerNorm(64)
GELU
Linear(64, 64)
```

得到：

```text
h_context ∈ R^64
```

5. 对每条边，把 context token prepend 到 expert tokens：

```text
tokens = [
  h_context,
  h_local,
  h_low,
  h_high
]
```

然后进入 Transformer fusion。

pooling 时建议：

```text
h_fused = mean(tokens_after_transformer, dim=token)
```

或者如果加 CLS，则取 CLS。第一版用 mean 即可。

重要：`h_context` 对同一个 target dataset 是固定的，不随 label 变化。它只表示 target 的无标签宏观分布。

## 7. 为什么这样仍然是 zero-shot

请在日志、README 或 metrics 里注明：

```text
target labels are not used for training or context construction.
Only unlabeled target feature distribution is used as context.
```

这属于 unsupervised target context / transductive zero-shot adaptation，不是 supervised target training。

如果担心严格 inductive 设置，可以之后再做 per-snapshot past-only context。第一版先做 target_mean，验证方向。

## 8. 必须实现两个实验

先只跑 btc_otc seed=0。

### A. No-context representation MoE

```text
model: context_rep_moe
context-mode: none
feature: raw10
target: btc_otc
seed: 0
```

输出目录：

```text
results/context_rep_moe_none_raw10_bestval50_bce/btc_otc_seed_0
```

### B. Target-mean context representation MoE

```text
model: context_rep_moe
context-mode: target_mean
feature: raw10
target: btc_otc
seed: 0
```

输出目录：

```text
results/context_rep_moe_targetmean_raw10_bestval50_bce/btc_otc_seed_0
```

## 9. 训练设置

和 raw10 prototype-router baseline 对齐：

```text
source: MOOC + Wikipedia
target: btc_otc
seed: 0
history-window: 5
num-snapshots: 50
sampler: balanced
balanced-neg-ratio: 1
loss: BCEWithLogitsLoss
early stopping: 和 baseline 保持一致，先用 source val AUROC
```

不要加：

```text
prototype-router
evidence loss
hard rank loss
AP loss
router warmup
Atlas-local features
```

这次只验证 architecture change。

## 10. Diagnostics

`metrics.json` 至少输出：

```text
test AUROC
test AUPRC
best epoch
val AUROC
val AUPRC
P@anom
P@0.1%
P@0.5%
P@1%
```

模型诊断：

```text
context_mode
hidden_dim
num_experts
expert_input_dims
```

如果方便，请输出 auxiliary per-expert logits。

虽然新模型最终不用 score-level fusion，但可以给每个 expert 加一个 diagnostic head，只用于诊断，不参与 final loss 或可选参与 very small auxiliary loss。

第一版建议：

```text
diagnostic_head_local
diagnostic_head_low
diagnostic_head_high
```

输出 per-expert AUROC/AUPRC，方便比较：

```text
Local expert diagnostic AUROC/AUPRC
Low-order expert diagnostic AUROC/AUPRC
High-order expert diagnostic AUROC/AUPRC
```

注意：diagnostic head 默认不要参与训练 loss，除非实现上必须；如果参与，请单独说明。

Context 诊断：

```text
target_context_mean
target_context_std
context_token_norm
```

如果 Transformer 可以输出 attention weights，最好输出：

```text
mean attention to context token
mean attention among expert tokens
```

但这个不是必须，别为了 attention diagnostics 大改结构。

## 11. 判断标准

和 baseline 对比：

```text
raw10 proto-router final:
0.6968 AUROC / 0.1433 AUPRC

raw10 High-order expert:
0.6990 AUROC / 0.1438 AUPRC
```

主要看：

### 11.1 Representation fusion 是否有效

```text
context-mode none
vs
raw10 proto-router final
```

如果 no-context rep MoE 已经提升，说明 score-level fusion 确实限制了模型。

### 11.2 Target context 是否有效

```text
context-mode target_mean
vs
context-mode none
```

如果 target_mean 明显提升，说明 target unlabeled distribution 有用。

### 11.3 是否超过 High-order expert

最关键：

```text
target_mean final > 0.6990 / 0.1438
```

如果超过，说明 representation-level cooperation + target context 真的带来了超过单 expert 的收益。

如果没有超过，但比 prototype-router final 稳定，也可以作为正向信号。

## 12. Smoke test

正式提交前请先做 smoke test：

1. raw10 input dim = 10。

2. expert dims 正确：

```text
Local = 2
Low = 4
High = 4
```

3. expert output shape：

```text
[batch, 64]
```

4. context-mode none tokens shape：

```text
[batch, 3, 64]
```

5. context-mode target_mean tokens shape：

```text
[batch, 4, 64]
```

6. final_logit shape：

```text
[batch]
```

7. target context construction 不使用 target labels。

8. training / validation / test 正常落盘 `metrics.json`。

请先跑 `btc_otc seed=0`，不要扩到 5 个 target，等结果出来后再决定下一步。
