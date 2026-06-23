# DP-DGAD 论文方法与当前代码实现不一致分析

本文档对照论文 `DP-DGAD: A Generalist Dynamic Graph Anomaly Detector with Dynamic Prototypes` 的方法描述，与当前工作区代码实现进行核对。重点关注：

- `train_source.py`：source datasets 预训练流程。
- `infer_target.py`：target datasets 无标签适配与测试流程。
- `model/Transformer.py`、`model/DGG.py`：与训练和推理直接相关的 scoring/prototype 逻辑。

说明：这里比较的是当前工作区代码，不一定等同于论文作者发布时的原始代码状态。我们前面已经做过一些工程修改，例如路径参数化、设备硬编码修复、标签约定修复等。

---

## 1. 总体结论

当前代码和论文描述有多处重要不一致，其中有几处会直接影响复现实验结果：

1. **损失权重方向疑似写反。**
   论文说最佳设置是 `lambda_A = 0.1, lambda_BCE = 0.9`，即 BCE 为主、alignment 为辅；当前 `train_source.py` 实际是 `0.1 * BCE + 0.9 * alignment`。

2. **target 伪标签选择逻辑与论文不一致。**
   论文说选择低熵、高置信的 normal 和 abnormal detections；当前代码把最低熵样本当 normal，把最高熵样本当 abnormal。最高熵其实是低置信样本。

3. **target buffer size 计算方式与论文不一致。**
   论文说 memory buffer size 是数据规模的 10%；当前 `infer_target.py` 里是 `config.buffer_size * len(prototype_buffer)`，默认 `buffer_size=30`，即 source buffer 的 30 倍，而不是 target 数据大小的 10%。

4. **论文说 target adaptation 是 parameter-free，只更新 memory buffer；当前代码并不是真正用梯度更新 buffer。**
   当前代码冻结模型参数，但把 `normal_mean/cov/abnormal_mean/cov` 放进 optimizer；同时 `Transformer.py` 又大量 `.detach()` 和 `torch.no_grad()`，导致这些统计量基本不是论文意义上的可学习更新。buffer 本身是 Python list 里的 CPU tensor，只是 append/replace，不是通过 loss 反传更新。

5. **anomaly scoring 公式与论文不一致。**
   论文公式是 `score = z^T mu - lambda * z^T Sigma z`；当前代码是 `dot + 0.5 * lambda * covariance_term`，符号相反，并且 `lambda` 是固定 `1e-3`，不是论文所说 learnable parameter。

6. **模型返回的是 sigmoid 后概率，但训练用的是 `binary_cross_entropy_with_logits`。**
   `BCEWithLogits` 期望输入 raw logits，当前却给了 sigmoid 概率，等于又套了一层 sigmoid，训练目标会偏。

7. **source 训练不是保存最佳模型，而是保存最后模型。**
   论文算法说预训练得到 `Psi_pretrain` 和 buffer，但实验复现通常应保存最佳或稳定 checkpoint。当前代码里虽然有 `best_loss/counter`，但最后保存的是训练结束时的模型。我们已经观察到 epoch=2 比 epoch=50 好，很大程度与这一点有关。

---

## 2. 论文方法摘要

论文方法可以拆成两阶段。

### 2.1 Source pretraining

论文 Algorithm 1 描述：

- 初始化 memory buffer `B`。
- 对每个 source dataset 训练。
- 对每条边提取 temporal ego-graph，经过 GNN 和 Transformer 得到 edge representation。
- 使用 normal prototype `p_n` 和 abnormal prototype `p_a` 进行 anomaly scoring。
- 使用 BCE loss 与 alignment loss 联合训练。
- 对第一个 source dataset，主要根据 prototype pair difference score `s_d` 更新 buffer。
- 对后续 source dataset，同时考虑 prototype difference `s_d` 和跨域相似性 `s_e`：
  - `s_r = lambda_d * s_d - lambda_e * s_e`
  - `s_e` 越低表示越相似，`s_r` 越高越应保留。

论文实现细节写到：

- source datasets 是 `Wikipedia` 和 `MOOC`。
- memory buffer size `M` 是数据规模的 10%。
- momentum `alpha=0.9`。
- loss ratio：`lambda_A=0.1`，`lambda_BCE=0.9`。
- cross-domain 权重：`lambda_d=0.3`，`lambda_e=0.7`。

### 2.2 Target adaptation / inference

论文 Algorithm 2 描述：

- target dataset 无标签。
- 使用 pretrained scorer 得到 anomaly probability。
- 用 entropy 衡量置信度。
- 选择 top `N_con` 低熵 normal 和 abnormal detections，作为 pseudo-label。
- 冻结 generalist backbone，只用 pseudo-label 通过 alignment loss 更新 memory buffer。
- 最终用 frozen model + updated memory buffer 进行 target anomaly detection。

论文强调：

- target 阶段是 parameter-free adaptation。
- backbone frozen。
- 只更新 memory buffer 中的 prototypes。

---

## 3. `train_source.py` 中的不一致

### 3.1 Loss 权重和论文相反

论文描述：

- `L = lambda_BCE * L_BCE + lambda_A * L_A`
- 实验设置：`lambda_A=0.1`，`lambda_BCE=0.9`

当前代码：

```python
loss = args.ratio * criterion(logits, y) + (1 - args.ratio) * loss_alignment
```

相关文件：

- `train_source.py` 第 471 行
- `option_source_train.py` 第 19 行：`ratio=0.1`

因此当前实际为：

```text
0.1 * BCE + 0.9 * alignment
```

这与论文设置的：

```text
0.9 * BCE + 0.1 * alignment
```

方向相反。

影响：

- 代码过度强调 prototype alignment，弱化分类判别能力。
- 这和我们观察到的现象一致：长 epoch 训练 loss 下降，但 AUROC/AUPRC 反而下降。
- 论文附录还说 pure alignment 表现很差，说明这个不一致影响非常大。

建议：

如果按论文实现，应改为：

```python
loss = (1 - args.ratio) * criterion(logits, y) + args.ratio * loss_alignment
```

或者显式改成：

```python
loss = config.lambda_bce * bce_loss + config.lambda_align * align_loss
```

避免 `ratio` 语义混乱。

---

### 3.2 `BCEWithLogits` 输入不是 raw logits

当前 `criterion`：

```python
F.binary_cross_entropy_with_logits(logits, labels)
```

但 `model/Transformer.py` 中：

```python
raw_logits = abnormal_scores - normal_scores
logits = torch.sigmoid(raw_logits)
```

相关文件：

- `train_source.py` 第 31-35 行
- `infer_target.py` 第 28-32 行
- `model/Transformer.py` 第 152-154 行

问题：

- `binary_cross_entropy_with_logits` 期望输入未经过 sigmoid 的 raw logits。
- 当前传入的是 sigmoid 后的概率。
- 这等于把概率当 logits 再做 sigmoid，损失函数不符合论文 Eq.17-18 的概率/BCE定义。

影响：

- 分类 loss 数值和梯度都会不对。
- 这会影响 source pretraining，也会影响 target eval loss。

建议：

二选一：

1. `Transformer.py` 返回 `raw_logits`，保留 `binary_cross_entropy_with_logits`。
2. `Transformer.py` 返回 sigmoid probability，把 loss 改成 `binary_cross_entropy`。

更建议第 1 种，因为 PyTorch 的 logits 版更稳定。

---

### 3.3 保存的是最后模型，不是最佳模型

当前代码虽然初始化了：

```python
best_loss = float('inf')
counter = 0
early_stopper = EarlyStopMonitor(...)
```

但实际保存模型在训练所有 dataset 后才执行：

```python
save_model(...)
```

相关文件：

- `train_source.py` 第 371-377 行
- `train_source.py` 第 487-496 行
- `train_source.py` 第 507-509 行

问题：

- 代码没有在 best validation metric 或 best average loss 时保存 checkpoint。
- `best_val_aucs`、`best_val_aps` 定义了但没有真正参与保存。
- `early_stopper` 创建了但没有使用。
- `best_loss` 比较的是最后一个 batch 的 `loss`，不是 epoch 平均 loss，也不是 validation loss。
- 即使触发 early stopping，也只是跳出当前 dataset 的 epoch loop，最终仍保存当前状态，不一定是最佳状态。

实验证据：

我们的训练日志显示：

```text
epoch=2 source checkpoint:
best auc: 0.5526
best ap: 0.1966

epoch=50 source checkpoint:
best auc: 0.4894
best ap: 0.1214
```

这说明 epoch=50 的最后模型明显退化。

影响：

- 长训练不一定更好。
- 当前 `epoch=2` 比 `epoch=50` 好，很可能就是因为 `epoch=50` 过拟合或 buffer 污染后仍保存最后状态。

建议：

- 每个 epoch 后在 held-out source validation 或指定 target validation 上评估。
- 保存 best AUROC/AP 或 best validation loss 的 checkpoint。
- 至少保存 `best_state_dict` 和对应 `prototype_buffer`。

---

### 3.4 Difference score 公式不完全一致

论文 Eq.6：

```text
s_d = mean_m ||p_a^m - p_n^m||^2
```

当前代码：

```python
proto_diff_raw = torch.mean(torch.norm(normal_prompt - abnormal_prompt))
proto_diff = torch.sigmoid(0.1 * proto_diff_raw).item()
```

相关文件：

- `train_source.py` 第 408-412 行
- `infer_target.py` 第 379-380 行

问题：

- 论文是 prototype pair 的均值欧氏距离/平方距离形式。
- 当前代码对整个向量做 `torch.norm`，再 `mean`；由于 `torch.norm` 返回标量，`torch.mean` 实际没有意义。
- 当前代码又套了 `sigmoid(0.1 * score)`，论文没有这个归一化步骤。

影响：

- buffer replacement 中的排序分数和论文不一致。
- `sigmoid` 会压缩大差异，使不同 prototype 的区分度降低。

---

### 3.5 跨 source domain 的 `s_r` 实现和论文公式不一致

论文公式：

```text
s_r = lambda_d * s_d - lambda_e * s_e
```

其中：

- `s_d`：normal/abnormal prototype pair 的差异，越大越好。
- `s_e`：prototype 与新 domain representation 的距离，越小越好。
- 所以公式中对 `s_e` 是减号。

当前 `train_source.py`：

```python
prototype_relevance_scores = calculate_prototype_relevance_scores(...)
combined_scores = [
    config.relevance * rel + config.difference * proto[2]
    for rel, proto in zip(prototype_relevance_scores, prototype_buffer)
]
```

相关文件：

- `train_source.py` 第 359-368 行
- `train_source.py` 第 425-438 行
- `option_source_train.py` 第 20-22 行：`relevance=0.7`, `difference=0.3`

代码里的 `rel` 是由负距离归一化得到，和论文的 `-s_e` 有一点相似，但仍有差异：

- 论文明确是 `lambda_d * s_d - lambda_e * s_e`。
- 代码使用归一化后的 `rel` 加上 `difference score`。
- 新加入的 target/source prototype 直接赋 `relevance=1.0`，不重新计算和新 domain 的真实相似度。
- `relevance_threshold` 定义了但没有使用。

影响：

- buffer 保留的“domain-agnostic prototypes”可能不符合论文定义。
- 新 domain 的 prototype 很容易因为 `relevance=1.0` 被偏爱。

---

### 3.6 buffer 增长和 epoch 数强耦合，论文没有明确说明保存最后大 buffer

论文说 buffer size 是数据规模 10%，并保留最 discriminative/general prototypes。

当前 source 代码：

```python
buffer_size = max(1, int(config.buffer_size * len(dataset_train)))
```

这一点在 source 阶段和论文一致，因为 `config.buffer_size=0.1`。

但实际训练日志显示：

```text
epoch=2 model: prototype_buffer_len = 959
epoch=50 model: prototype_buffer_len = 20587
```

原因是：

- buffer size 上限以第二个 source dataset MOOC 的大小为准，最大可到 20587。
- epoch 越多，buffer 越接近填满。
- 论文虽然说 larger buffer 有时会提升，但也承认 oversized buffer 会引入 irrelevant prototypes。

影响：

- epoch=50 的 buffer 很大，target inference 时每个 target embedding 都要和 20587 个 prototypes 做匹配。
- 更重要的是，过大的 buffer 可能包含大量 source-specific / noisy prototypes，降低 target generalization。

---

### 3.7 Source dataset 顺序不稳定

论文实现细节说 source datasets 是 Wikipedia 和 MOOC。

当前配置：

- `option_source_train.py` 默认：`['MOOC', 'Wikipedia']`
- 我们实际 Slurm 脚本使用：`Wikipedia MOOC`

相关文件：

- `option_source_train.py` 第 9 行
- `slurm_scripts/train_source_gpu.slurm`
- `slurm_scripts/train_source_epoch2_gpu.slurm`

影响：

- DP-DGAD 是 sequential training，buffer 会受 dataset 顺序影响。
- 如果论文结果使用固定顺序，而代码默认和运行脚本不一致，结果可能不可比。

建议：

- 固定 source order，并在文档/脚本中注明。
- 分别跑 `Wikipedia->MOOC` 与 `MOOC->Wikipedia` 做 ablation。

---

### 3.8 开启 `torch.autograd.set_detect_anomaly(True)` 不属于论文方法

当前代码：

```python
torch.autograd.set_detect_anomaly(True)
```

相关文件：

- `train_source.py` 第 278 行

影响：

- 这是调试工具，不是论文方法。
- 会显著拖慢 backward。
- 不直接改变理论方法，但会影响训练效率和复现实验时间。

---

## 4. `infer_target.py` 中的不一致

### 4.1 Target buffer size 不是 target 数据规模的 10%

论文实现细节：

- memory buffer size `M` 是数据规模的 10%。
- `N_con` 是数据规模的 10%。

当前 `infer_target.py`：

```python
buffer_size = max(1, config.buffer_size * len(prototype_buffer))
```

相关文件：

- `infer_target.py` 第 193-194 行
- `option_infer_target.py` 第 42 行：`buffer_size=30`

这意味着：

```text
target buffer 上限 = 30 * source prototype_buffer_len
```

例如：

- epoch=50 source buffer 是 20587，则 target buffer 上限约 617610。
- epoch=2 source buffer 是 959，则 target buffer 上限约 28770。

这和论文的“target data size 的 10%”完全不同。

影响：

- target 阶段几乎不会触发 buffer full replacement。
- 代码更像是不断 append target pseudo prototypes，而不是在固定-size buffer 中选择/替换。
- epoch=50 和 epoch=2 的 infer 差异会被 source buffer 大小进一步放大。

建议：

应改成类似：

```python
buffer_size = max(1, int(config.buffer_size * len(dataset_test)))
```

并让 `option_infer_target.py` 的 `buffer_size` 默认值与 source 一致，例如 `0.1`。

---

### 4.2 Target 伪标签选择逻辑与论文相反/不完整

论文说：

- 用 entropy 表示置信度。
- 低 entropy 表示高置信。
- 选择 top `N_con` normal 和 abnormal low-entropy detections。
- pseudo-label 来自 pretrained scorer 的 detection result。

当前代码：

```python
edge_probs = torch.sigmoid(logits)
edge_entropy = calculate_class_ranking_reliability(edge_probs)
sorted_entropy, sorted_indices = torch.sort(edge_entropy)
normal_indices = sorted_indices[:normal_count]
abnormal_indices = sorted_indices[-abnormal_count:]
```

相关文件：

- `infer_target.py` 第 351-365 行

问题：

1. `normal_indices` 取最低熵，符合“高置信”。
2. `abnormal_indices` 取最高熵，代表最低置信，而不是论文说的低熵 abnormal detections。
3. 代码没有根据预测概率判断样本是 normal 还是 abnormal。
4. 论文说“detection result will serve as pseudo label”，代码没有用 detection result 生成 normal/abnormal，而是用 entropy 排序位置硬分。

正确方向更可能是：

- 先计算 probability。
- 根据 probability 判断 predicted label。
- 在 predicted normal 中取最低 entropy 的 top `N_con`。
- 在 predicted abnormal 中取最低 entropy 的 top `N_con`。

影响：

- 当前 abnormal pseudo-label 可能是最不确定的一批样本。
- 这会污染 target buffer，尤其在 source model 初始性能较差时更严重。

---

### 4.3 `option_infer_target.py` 里有 confidence strategy 参数，但代码没有使用

当前配置：

```python
parser.add_argument('--confident_detection_method', default='random',
                    choices=['entropy', 'random', 'threshold', 'distance', 'similarity'])
```

但 `infer_target.py` 实际固定使用 entropy：

```python
edge_entropy = calculate_class_ranking_reliability(edge_probs)
sorted_entropy, sorted_indices = torch.sort(edge_entropy)
```

相关文件：

- `option_infer_target.py` 第 23-26 行
- `infer_target.py` 第 351-356 行

问题：

- 论文 Table 4/消融讨论了不同 confidence generation strategies。
- 代码参数暴露了这些 strategy，但没有实现分支。
- 默认值还是 `random`，但实际运行不是 random。

影响：

- 命令行设置 `--confident_detection_method threshold` 等不会生效。
- 做消融实验时容易误判。

---

### 4.4 Target adaptation 不是论文意义上的 parameter-free buffer update

论文说：

- generalist backbone frozen。
- only prototypes in memory buffer are updated。
- parameter-free adaptation。

当前代码：

```python
for param in model.parameters():
    param.requires_grad = False

normal_mean.requires_grad = True
normal_cov.requires_grad = True
abnormal_mean.requires_grad = True
abnormal_cov.requires_grad = True

optimizer = torch.optim.Adam([normal_mean, normal_cov, abnormal_mean, abnormal_cov], ...)
```

相关文件：

- `infer_target.py` 第 185-220 行

但是 `Transformer.py` 中：

```python
updated_normal_mean = normal_mean.clone().detach()
...
with torch.no_grad():
    updated_normal_mean = ...
...
nm = normal_mean.clone().detach()
...
```

相关文件：

- `model/Transformer.py` 第 89-117 行
- `model/Transformer.py` 第 127-132 行

问题：

- mean/cov 被设为 requires_grad，但 forward 中又 detach。
- buffer 里的 prototype 是 CPU tensor list，不是 optimizer 管理的 parameter。
- target loss 只是在代码层面 append/replace buffer，而不是“通过 loss 更新 buffer 中 prototypes”。

影响：

- target adaptation 的实际可学习性很弱。
- `if loss.requires_grad` 保护能避免报错，但也可能导致 optimizer step 根本没有有效梯度。
- 这与论文描述的“通过 alignment loss 更新 memory buffer”不一致。

---

### 4.5 Target 阶段仍读取 labels，且保存 confident 样本时用到了真值

当前 target train loader：

```python
dataset_test = dataset.DygDataset(config, 'train')
...
y, logits, output, ...
...
show_confident.append({'fn': y[normal_indices], ...})
```

相关文件：

- `infer_target.py` 第 229-235 行
- `infer_target.py` 第 346-366 行

说明：

- loss 和 buffer update 没有直接使用 `y` 作为监督。
- 但是代码在 target adaptation 阶段仍读取 labels，并把 labels 保存到 `show_confident`。

与论文关系：

- 论文设定 target unlabeled。
- 如果只是为了离线分析 pseudo-label 质量，可以接受；但从方法实现角度，这不是纯粹的 unlabeled pipeline。

建议：

- target train 阶段不应依赖 label 字段存在。
- 如果需要分析，单独加 debug flag 保存。

---

### 4.6 Target train/test split 与论文描述需要核对

论文实现细节说：

- target datasets 只含 normal edges 用于 fine-tune/update。
- testing 在同一 target datasets 上注入 anomalies。

当前 `infer_target.py`：

```python
dataset_train = DygDataset(config, 'train')
dataset_test = DygDataset(config, 'test')
```

相关文件：

- `infer_target.py` 第 230-235 行
- `infer_target.py` 第 453-460 行

是否一致取决于 `data/*.pkl` 内部 train/test 如何构造。

风险：

- 如果我们准备的 pkl 里 train split 已经包含 injected anomalies，那么 target adaptation 就不符合论文“target train only normal”的设定。
- 如果 pkl 没有显式 train/test split，而 `DygDataset` 只是按同一数据读取，可能也和论文不一致。

建议：

- 单独检查 `datasets.py` 和每个 pkl 的 split/label 分布。
- 确保 target train split labels 全为 normal，test split 才有 injected anomalies。

---

### 4.7 Final testing 使用的 best prototype 逻辑和论文不完全一致

论文说：

- updated memory buffer + frozen model 用于 final anomaly detection。
- anomaly scoring 依赖 buffer prototypes 构建 normal/abnormal distributions。

当前代码：

```python
if prototype_buffer and prototype_relevance_scores:
    best_prototype_idx = np.argmax(combined_scores)
else:
    best_prototype_idx = 0
```

然后每个 batch 使用单个 best prototype pair：

```python
normal_prompt_raw, abnormal_prompt_raw = prototype_buffer[best_pair_idx]
```

相关文件：

- `infer_target.py` 第 437-443 行
- `infer_target.py` 第 56-60 行

问题：

- 论文描述更像是从 buffer 中 top prototype pairs 更新 normal/abnormal distribution。
- 当前实现主要选择单个 `best_prototype_idx` 作为 prompt 输入。
- 虽然 checkpoint 中保存了 mean/cov，但更新统计量的实现也不是基于 top M prototypes。

影响：

- buffer 的多数 prototype 只参与 relevance 排名，不直接参与 scoring。
- 这削弱了论文中“memory buffer 覆盖多样 domain patterns”的作用。

---

## 5. `model/Transformer.py` 与 `model/DGG.py` 中的不一致

### 5.1 Scoring 公式符号与论文相反

论文 Eq.15-16：

```text
s_n = z^T mu_n - lambda_n * z^T Sigma_n z
s_a = z^T mu_a - lambda_a * z^T Sigma_a z
s_i = s_a - s_n
```

当前代码：

```python
cov_term_normal = 0.5 * lambda_val * torch.dot(f, cov_product)
normal_score = mu_term_normal + cov_term_normal

cov_term_abnormal = 0.5 * lambda_val * torch.dot(f, cov_product)
abnormal_score = mu_term_abnormal + cov_term_abnormal
```

相关文件：

- `model/Transformer.py` 第 120-143 行

差异：

- 论文是减去 covariance penalty。
- 代码是加上 covariance term。
- 论文说 `lambda_a/lambda_n` 是 learnable parameter。
- 代码固定 `lambda_val = 1e-3`。
- 代码多了 `0.5` 系数，论文没有。

影响：

- anomaly score 的统计意义改变。
- covariance 越大在论文中应降低相似性，在代码中却提高 score。

---

### 5.2 mean/cov 更新不是基于 buffer 的 top prototype pairs

论文 Eq.9-14：

- 用 memory buffer 中的多个 prototype pairs 更新 mean/cov。
- `M` 是 prototype 数量。

当前代码：

```python
batch_normal_mean = normal_prompt.mean(0)
batch_abnormal_mean = abnormal_prompt.mean(0)
normal_centered = (normal_prompt - normal_mean).unsqueeze(0)
batch_normal_cov = torch.mm(normal_centered.t(), normal_centered) / max(1, normal_centered.size(0) - 1)
```

相关文件：

- `model/Transformer.py` 第 95-117 行

问题：

- `normal_prompt` 通常是单个 prototype vector，不是 `M` 个 prototypes。
- `unsqueeze(0)` 后 `normal_centered.size(0)` 是 1，因此 covariance 是单样本外积。
- 这不等价于论文中对 top prototype pairs 的 covariance 估计。

影响：

- distribution-based scoring 的 mean/cov 与论文设想不同。
- covariance 估计不稳定，且又被 detach。

---

### 5.3 PrototypeAttention 实际未使用

`DGG.py` 中：

```python
# normal_prompt = self.act(self.Transformer_d(normal_prompt_raw, ra))
# abnormal_prompt = self.act(self.Transformer_d(abnormal_prompt_raw, ra))

normal_prompt = normal_prompt_raw
abnormal_prompt = abnormal_prompt_raw
```

相关文件：

- `model/DGG.py` 第 29-33 行

问题：

- `PrototypeAttention` 类存在，但在 forward 中被注释掉。
- 论文描述中 dynamic prototypes 会和 ego-graph/representations 交互、演化；当前实现直接使用 raw prototype。

影响：

- prototypes 没有通过当前 batch 的 subgraph embedding 做 attention 更新。
- “dynamic prototype”的动态性主要来自 buffer append/replace 和统计量更新，不来自 attention 交互。

---

## 6. 参数设置与论文不一致

| 项目 | 论文描述 | 当前代码 |
|---|---|---|
| source loss ratio | `lambda_A=0.1`, `lambda_BCE=0.9` | `ratio=0.1` 且 `loss=ratio*BCE+(1-ratio)*alignment`，实际 BCE=0.1, alignment=0.9 |
| source buffer size | 数据规模 10% | `option_source_train.buffer_size=0.1`，基本一致 |
| target buffer size | 数据规模 10% | `option_infer_target.buffer_size=30`，且乘 source buffer size |
| `N_con` | target 数据规模 10% | 每个 batch 选 10% normal + 10% abnormal，且 abnormal 用最高熵 |
| cross-domain weights | `lambda_d=0.3`, `lambda_e=0.7` | source 默认 `difference=0.3`, `relevance=0.7`；infer 默认 `difference=0.9`, `relevance=0.1` |
| confidence strategy | entropy-based 是 DP-DGAD 主策略 | 参数默认 `random`，但代码实际固定 entropy 排序 |
| target adaptation | parameter-free，只更新 buffer | 冻结 model，但 optimizer 试图更新 mean/cov；buffer 通过 append/replace 更新 |
| scoring lambda | learnable `lambda_a/lambda_n` | 固定 `lambda_val=1e-3` |
| scoring covariance sign | `- lambda * z^T Sigma z` | `+ 0.5 * lambda * z^T Sigma z` |

---

## 7. 与我们实验现象的关系

我们已经观察到：

```text
epoch=2 source checkpoint:
prototype_buffer_len = 959
source test auc/ap = 0.5526 / 0.1966

epoch=50 source checkpoint:
prototype_buffer_len = 20587
source test auc/ap = 0.4894 / 0.1214
```

target inference 对比中，epoch=2 大多数数据集也更好。

这和上述不一致能对应上：

1. 训练 loss 权重偏向 alignment，训练越久越可能让 prototype/embedding 空间过度贴合 source。
2. 保存最后模型而非最佳模型，使 epoch=50 直接保存退化状态。
3. buffer 变得巨大，target 阶段选 best prototype 更容易受 noisy/source-specific prototypes 干扰。
4. target pseudo-label abnormal 选择了最高熵样本，进一步增加 target buffer 噪声。
5. scoring 公式与论文不同，使 prototype distribution 的意义偏离论文。

因此，epoch=2 比 epoch=50 好并不是论文方法本身的结论，而是当前实现细节下很可能出现的结果。

---

## 8. 建议的修复/验证顺序

### 第一优先级：直接影响指标

1. 修正 source loss 权重：

```python
loss = 0.9 * bce_loss + 0.1 * alignment_loss
```

2. 修正 `BCEWithLogits` 与 sigmoid 的关系：

- 让模型返回 raw logits。
- eval 时再 `sigmoid` 得到 probability。

3. 修正 target pseudo-label selection：

- 用 predicted probability 判定 pseudo normal / pseudo abnormal。
- 各自内部取 lowest entropy top `N_con`。
- 不要用最高 entropy 当 abnormal。

4. 修正 target buffer size：

```python
buffer_size = int(0.1 * len(target_train_dataset))
```

5. 保存 best checkpoint：

- 每个 epoch 评估 source validation。
- 保存 best AUROC/AP 或 best validation loss 时的 model + buffer。

### 第二优先级：贴近论文公式

6. 修正 scoring formula：

```python
normal_score = dot(z, mu_n) - lambda_n * z.T @ Sigma_n @ z
abnormal_score = dot(z, mu_a) - lambda_a * z.T @ Sigma_a @ z
```

并考虑让 `lambda_n/lambda_a` 可学习。

7. mean/cov 应从 top prototype pairs 或 buffer 中多个 prototypes 估计，而不是单个 prompt vector。

8. 实现或移除 `PrototypeAttention`：

- 如果论文需要 prototype 与 subgraph representation 交互，应启用并验证。
- 如果不用，应在文档中说明这是简化实现。

### 第三优先级：实验复现严谨性

9. 固定 source dataset 顺序，并记录在脚本里。
10. 检查 target train split 是否只有 normal。
11. 实现 `confident_detection_method` 的各分支，支持论文 ablation。
12. 关闭 `torch.autograd.set_detect_anomaly(True)` 用于正式训练。

---

## 9. 结论

当前代码可以跑通 DP-DGAD 的大致框架：source pretraining、prototype buffer、target pseudo-label update、target inference 都存在。但它和论文方法并不是严格一致实现，尤其在 loss 权重、伪标签选择、buffer size、scoring formula、best checkpoint 保存这几处存在高影响差异。

如果目标是“复现论文结果”，建议先修复第 8 节的第一优先级问题，再重新训练 source model，并重新跑 target inference。否则继续调 epoch 或数据集处理，很可能只是在当前偏离论文的实现上调参，难以解释和论文表格之间的差距。
