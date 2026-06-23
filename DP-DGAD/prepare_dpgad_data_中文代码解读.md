# `prepare_dpgad_data.py` 中文代码解读

本文档对应当前版本的 [prepare_dpgad_data.py](./prepare_dpgad_data.py)，逐段解释它如何把原始动态图数据转成 DP-DGAD 的 `.pkl` 格式。

这版脚本有两条数据路线：

1. 无标签数据集：`uci/digg/btc_alpha/btc_otc`，参考 TADDY 注入随机 non-edge 异常；可选择先清洗成唯一无向边，也可保留原始时间事件。
2. 有标签源数据集：`wikipedia/wiki/mooc`，直接读取原始标签，不做 TADDY 异常注入。

## 参考来源总览

| 来源 | 文件 | 被借鉴/适配的内容 |
|---|---|---|
| DP-DGAD | [datasets.py](./datasets.py) | 最终 `.pkl` 必须包含 `nodefeatures`, `edgefeatures`, `labels`, `Tmats`, `adjs`, `eadjs`, `ra`。 |
| TADDY | [../TADDY/codes/AnomalyGeneration.py](../TADDY/codes/AnomalyGeneration.py) | 无标签数据的异常注入：划分 train/test，采样原图中不存在的边，把异常随机插入测试时间序列。 |
| TADDY | [../TADDY/0_prepare_data.py](../TADDY/0_prepare_data.py) | 读取 `uci/digg/btc_alpha/btc_otc` 原始数据，按时间排序，去 self-loop 和重复边。 |
| GeneralDyG | [../GeneralDyG/generate_datasets.py](../GeneralDyG/generate_datasets.py) | 对每条边抽取 k-hop ego-graph，构造节点邻接、节点-边关联矩阵、边邻接矩阵。 |
| GeneralDyG | [../GeneralDyG/datasets.py](../GeneralDyG/datasets.py) | DP-DGAD 的 `datasets.py` 基本沿用 GeneralDyG 的数据结构。 |
| GeneralDyG 原始数据 | [../Data/Dynamic_Graph_data](../Data/Dynamic_Graph_data) | Wikipedia 和 MOOC 的原始带标签动态图数据。 |

## 对应检查表

| 当前代码结构 | 代码行号 | 文档是否覆盖 |
|---|---:|---|
| 导入、常量、数据集配置 | 1-44 | 已覆盖 |
| `load_raw_edges` | 47-74 | 已覆盖 |
| `load_labeled_events` | 77-120 | 已补充覆盖 |
| `sample_random_nonedges` | 123-161 | 已覆盖 |
| `inject_anomalies` | 164-217 | 已覆盖 |
| `EgoGraphBuilder.__init__` | 220-233 | 已覆盖 |
| `extract_k_hop_nodes` | 235-260 | 已覆盖 |
| `reorder_nodes` | 262-274 | 已覆盖 |
| `replace_subgraph` | 276-301 | 已覆盖 |
| `prune_edges` | 303-318 | 已补充覆盖 |
| 矩阵工具函数 | 320-357 | 已覆盖 |
| `build` | 359-412 | 已覆盖 |
| `compact_edge_ids` | 414-419 | 已覆盖 |
| `parse_args` | 422-444 | 已更新覆盖 |
| `main` | 448-484 | 已更新覆盖 |
| 脚本入口 | 487-488 | 已覆盖 |

## 整体流程

| 步骤 | 无标签数据路线 | 有标签源数据路线 |
|---|---|---|
| 1 | 从 TADDY raw 文件读取时间边列表。 | 从 `Data/Dynamic_Graph_data` 读取 Wikipedia/MOOC 原始文件。 |
| 2 | 按时间排序，去 self-loop；`unique_undirected` 会转无向边并去重，`temporal_events` 会保留重复时间交互。 | 按时间排序，保留原始事件标签。 |
| 3 | 按 `train_per` 分 train/test。 | 不重新切分；只生成完整事件序列的样本。 |
| 4 | 采样原图中不存在的 non-edge 作为异常。 | 不注入异常。 |
| 5 | 把异常随机插入测试序列。 | 使用原始 `LABEL` 或 `state_label`。 |
| 6 | 对每条事件构造 ego-graph。 | 对每条事件构造 ego-graph。 |
| 7 | 生成 DP-DGAD 所需字段并保存 `.pkl`。 | 生成 DP-DGAD 所需字段并保存 `.pkl`。 |

## 文件头、导入与常量

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 1-12 | 文件顶部 docstring，说明脚本目标：准备 DP-DGAD pickle 数据，并列出输出字段。 | 输出字段来自 DP-DGAD `datasets.py`。 |
| 14 | `from __future__ import annotations`，让 Python 类型注解延迟求值。 | 工程适配。 |
| 16-20 | 导入命令行参数、深拷贝、pickle、随机数、路径工具。 | Python 标准库。 |
| 22 | 导入 `networkx`，用于建图、抽子图、生成邻接矩阵。 | GeneralDyG 数据生成逻辑。 |
| 23 | 导入 `numpy`，用于数组处理、排序、采样、ID 重映射。 | TADDY 和 GeneralDyG 都使用。 |
| 24 | 导入 `pandas`，用于组织边事件表。 | GeneralDyG 使用 `graph_df` 风格表结构。 |
| 25 | 导入 `scipy.sparse`，用于稀疏矩阵。 | GeneralDyG 矩阵构造。 |
| 26 | 导入 `torch`，用于把矩阵转成 tensor。 | DP-DGAD 模型输入。 |
| 27 | 导入 `tqdm`，用于显示构造 ego-graph 的进度条。 | GeneralDyG 也使用进度条。 |
| 30-35 | `UNLABELED_DATASET_FILES`，定义无标签数据集名到 TADDY raw 文件名的映射。 | TADDY raw 数据命名。 |
| 37 | `LABELED_DATASETS`，定义直接带标签的数据集：Wikipedia/Wiki/MOOC。 | 本脚本为 DP-DGAD 源数据准备新增。 |
| 38 | `DATASET_CHOICES`，把无标签和有标签数据集合并成命令行可选项。 | 本脚本新增。 |
| 39-43 | `DATASET_OUTPUT_NAMES`，为 Wikipedia/MOOC 设置默认输出名。 | DP-DGAD 常用数据命名。 |
| 44 | `EVENT_MODES`，定义无标签 raw 数据的事件处理模式。 | 本脚本新增，用于区分 TADDY 唯一边模式和 DP-DGAD 时间事件模式。 |

## `load_raw_edges`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 47 | 定义 `load_raw_edges(dataset, raw_dir, event_mode)`，读取无标签 raw 数据并返回事件边数组。 | TADDY `0_prepare_data.py` + 本脚本新增模式。 |
| 48 | 函数说明：读入 raw temporal edges，并把节点 ID 标准化成从 0 开始。 | 本脚本说明。 |
| 49-50 | 如果数据集不是无标签集合之一，就报错。 | 参数保护。 |
| 51-52 | 如果 `event_mode` 不是支持模式，就报错。 | 本脚本新增参数保护。 |
| 54 | 拼出 raw 文件路径。 | TADDY raw 目录结构。 |
| 55-56 | 文件不存在时报 `FileNotFoundError`。 | 参数保护。 |
| 58-61 | 对 `uci/digg`：按空格分隔读取，按第 4 列时间戳排序，取前两列端点。 | TADDY 读取逻辑。 |
| 62-65 | 对 Bitcoin Alpha/OTC：按逗号读取，按第 4 列时间戳排序，取前两列端点。 | TADDY 读取逻辑。 |
| 67 | 删除 self-loop，即 `u == v` 的边。 | TADDY 清洗逻辑。 |
| 69-72 | 如果 `event_mode=unique_undirected`，则端点排序并按端点去重，只保留每对节点第一次出现的边。 | TADDY `0_prepare_data.py` 风格。 |
| 73 | 对所有节点 ID 做连续重映射。 | TADDY `return_inverse` 思路。 |
| 74 | 返回形状为 `[num_events, 2]` 的整型边/事件数组。 | 后续异常注入输入。 |

注意：这个函数只用于无标签数据集。`unique_undirected` 会按端点去重，所以同一对端点在不同时间发生多次交互时，只保留最早一次；`temporal_events` 不按端点去重，会保留原始时间事件，更贴近 DP-DGAD 论文里 "random timestamps" 和 "samples" 的表述。

## `load_labeled_events`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 74 | 定义 `load_labeled_events(dataset, dynamic_data_dir)`，读取已经带标签的源数据。 | 本脚本为 Wikipedia/MOOC 新增。 |
| 75 | 函数说明：读取自带标签的数据集。 | 本脚本说明。 |
| 76 | 如果数据集是 `wikipedia` 或 `wiki`，进入 Wikipedia 分支。 | DP-DGAD 源训练数据。 |
| 77 | 原始文件路径是 `dynamic_data_dir / "Wiki" / "wikipedia.csv"`。 | GeneralDyG 原始数据目录。 |
| 78-79 | 如果 Wikipedia 原始文件不存在，报错。 | 参数保护。 |
| 80 | 用 `np.loadtxt` 读取 CSV，跳过表头，只取第 0、1、2、3 列：用户、物品、时间戳、标签。CSV 后面的原始 event feature 向量当前没有读入。 | 当前 DP-DGAD `datasets.py` 会重新随机初始化节点/边特征。 |
| 81 | 按第 3 列时间戳排序。 | 动态图事件序列要求。 |
| 82 | 取用户 ID。 | Wikipedia 二部图用户端。 |
| 83 | 取物品/页面 ID。 | Wikipedia 二部图物品端。 |
| 84 | 计算 `item_offset = max(user_id) + 1`，让用户节点和物品节点 ID 空间分开。 | 二部图常见处理；避免 user 1 和 item 1 被当成同一节点。 |
| 85 | 读取原始标签。这里约定 `0=正常，1=异常`。 | Wikipedia 原始标签。 |
| 86-91 | 构造标准事件表字段：`u`, `i`, `label`, `id`。其中 `i` 加 offset，`id` 是时间顺序事件编号。 | GeneralDyG/DP-DGAD 需要 `u/i/label/id`。 |
| 92 | 返回 pandas DataFrame。 | 后续 ego-graph 构造输入。 |
| 94 | 如果数据集是 `mooc`，进入 MOOC 分支。 | DP-DGAD 源训练数据。 |
| 95-97 | 拼出 MOOC actions 文件和 labels 文件路径。 | GeneralDyG 原始数据目录。 |
| 98-101 | 如果 actions 或 labels 文件不存在，分别报错。 | 参数保护。 |
| 103 | 读取 `mooc_actions.tsv`。 | MOOC 原始行为事件。 |
| 104 | 读取 `mooc_action_labels.tsv`。 | MOOC 原始事件标签。 |
| 105 | 按 `ACTIONID` 把行为表和标签表合并。 | MOOC 数据格式。 |
| 106 | 按 `TIMESTAMP` 排序并重置行索引。 | 动态图事件序列要求。 |
| 107 | 计算 `TARGETID` 的 offset，让用户节点和目标节点 ID 空间分开。 | 二部图常见处理。 |
| 108-115 | 返回标准事件表：`u=USERID`，`i=TARGETID+offset`，`label=LABEL`，`id=时间顺序编号`。 | GeneralDyG/DP-DGAD 输入格式。 |
| 117 | 如果不是支持的有标签数据集，报错。 | 参数保护。 |

注意：这个函数不会按端点去重，也不会注入异常。Wikipedia 和 MOOC 中同一对节点在不同时间的多次交互会作为多条事件保留下来。

这里的 `u/i/id` 后面会变成 `.pkl` 里的 `nodefeatures/edgefeatures` 索引。名字里虽然有 `features`，但它们不是原始 CSV 的 dense 特征向量，而是“用来索引特征矩阵的节点 ID/边事件 ID”。

## `sample_random_nonedges`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 120-125 | 定义随机采样 non-edge 的函数，输入节点数、已有边集合、采样数量和随机数生成器。 | TADDY `AnomalyGeneration.py` 的 fake edge 生成逻辑；更接近 `anomaly_generation2()` 中不做谱聚类的随机生成。 |
| 126 | 函数说明：采样原图里不存在的无向节点对。 | 对 TADDY `processEdges` 的清晰化改写。TADDY 实际入口调用 `anomaly_generation()`，其中也会调用 `processEdges()` 过滤 fake edges。 |
| 127 | 创建空集合 `sampled`，避免重复采样。 | 本脚本实现细节。 |
| 128 | 计算理论上最多可采多少条 non-edge。 | 本脚本边界检查。 |
| 129-130 | 如果请求的异常数量超过可能数量，直接报错。 | 鲁棒性处理。 |
| 132-133 | 初始化拒绝采样次数，并设置最大尝试次数。 | 鲁棒性处理。 |
| 134 | 数量不够时持续采样。 | TADDY fake edge 思路。 |
| 135-146 | 如果随机拒绝采样太慢，就枚举全部候选 non-edge，再随机补齐。 | 本脚本增强。 |
| 148-150 | 随机抽两个节点 ID，并增加尝试次数。 | TADDY 随机采样。 |
| 151-152 | 如果两个端点相同，跳过 self-loop。 | TADDY 清洗逻辑。 |
| 153 | 标准化为 `u < v` 的无向边。 | TADDY 端点排序。 |
| 154-155 | 如果边已存在于原图或已被采过，跳过。 | TADDY fake edge 过滤。 |
| 156 | 把合法 non-edge 加入集合。 | TADDY 异常边思想。 |
| 158 | 返回排序后的异常边数组。 | 本脚本实现细节。 |

## `inject_anomalies`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 161-167 | 定义异常注入函数，输入清洗后的边、训练比例、异常比例、异常数量基准和随机种子。 | TADDY 实际入口 `0_prepare_data.py` 调用的 `anomaly_generation()`；插入异常的主流程与 `anomaly_generation()`/`anomaly_generation2()` 基本一致。 |
| 168-172 | 函数说明：输出时间有序边表；标签约定 `0=正常，1=注入异常`。 | TADDY 标签约定。 |
| 173 | 创建 NumPy 随机数生成器，保证可复现。 | 本脚本工程化。 |
| 174 | 统计总边数。 | TADDY `m = len(edges)`。 |
| 175 | 用最大节点 ID 推出节点数。 | TADDY 需要节点数采 fake edge。 |
| 176 | 根据 `train_per` 计算训练边数量。 | TADDY `ini_graph_percent`。 |
| 178 | 前 `train_num` 条边作为训练边。 | TADDY。 |
| 179 | 剩余边作为测试边。 | TADDY。 |
| 181-182 | 如果 `anomaly_base=total`，异常数按总边数计算。 | DP-DGAD Appendix A.1 的 `p*m`。 |
| 183-184 | 如果 `anomaly_base=test`，异常数按测试边数计算。 | TADDY 代码写法。 |
| 185-186 | 其他取值报错。 | 参数保护。 |
| 188 | 把已有真实边按无向形式放入集合。即使 `temporal_events` 保留方向和重复时间事件，采异常时仍按“两个节点是否连接过”判断 disconnected nodes。 | TADDY `processEdges(fake_edges, data)` + DP-DGAD A.1 disconnected nodes 表述。 |
| 189 | 调用 `sample_random_nonedges` 生成异常边。 | TADDY fake edge 思路；这里没有复刻 `anomaly_generation()` 的谱聚类筛选。 |
| 191 | 测试序列长度等于真实测试边数加异常边数。 | TADDY 插入异常流程。 |
| 192 | 在测试序列中随机选择异常出现的位置。 | TADDY `anomaly_pos`。 |
| 193 | 初始化输出行列表。 | 本脚本实现细节。 |
| 195 | 初始化事件 ID。 | GeneralDyG 用 `id` 表示边事件编号。 |
| 196-198 | 把训练边加入输出表，标签都是 0。 | TADDY 训练部分全正常。 |
| 200-201 | 初始化真实测试边指针和异常边指针。 | 本脚本实现细节。 |
| 202 | 遍历完整测试序列位置。 | TADDY 插入异常流程。 |
| 203-206 | 如果当前位置是异常位置，取一条 fake edge，标签为 1。 | TADDY。 |
| 207-210 | 否则取一条真实测试边，标签为 0。 | TADDY。 |
| 211-212 | 写入当前事件，并递增事件 ID。 | GeneralDyG 表结构。 |
| 214 | 返回列名为 `u`, `i`, `label`, `id` 的 DataFrame。 | 后续 ego-graph 构造输入。 |

## `EgoGraphBuilder.__init__`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 217 | 定义 `EgoGraphBuilder` 类，负责把边事件表转成 DP-DGAD 样本。 | GeneralDyG `BatchGraphSample`。 |
| 218 | 构造函数输入：事件表、k-hop、最大节点数、最大边数、随机种子。 | GeneralDyG 基础逻辑，本脚本参数化。 |
| 219-222 | 保存输入参数。 | 本脚本实现细节。 |
| 223 | 创建 Python 随机数对象，用于邻居采样、边裁剪、边权选择。 | GeneralDyG 使用随机选择。 |
| 224 | 创建无向图。 | GeneralDyG `nx.Graph()`。 |
| 225 | 遍历事件表的每一行。 | GeneralDyG。 |
| 226 | 取出源节点、目标节点和事件 ID。 | GeneralDyG 的 `u/i/id`。 |
| 227-228 | 如果图里已有这对节点的边，把新的事件 ID 加入 `weight` 列表。 | GeneralDyG 用 weight 保存多次交互事件 ID。 |
| 229-230 | 如果图里没有这条边，就新增边，并把当前事件 ID 放进 `weight`。 | GeneralDyG。 |

## `extract_k_hop_nodes`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 232 | 定义函数：从一个端点出发抽取 k-hop 邻居节点。 | GeneralDyG `extract_k_hop_subgraph`。 |
| 233-235 | 初始化节点集合、访问集合和当前扩展边界。 | 本脚本清晰化实现。 |
| 236 | 扩展 `k_hop` 轮。 | GeneralDyG。 |
| 237 | 初始化当前层的新节点集合。 | 本脚本实现细节。 |
| 238 | 遍历当前边界节点。 | BFS 风格邻居扩展。 |
| 239 | 计算还能加入多少节点，避免超过 `max_nodes`。 | 本脚本内存控制。 |
| 240-241 | 如果已经没有容量，停止扩展。 | 本脚本内存控制。 |
| 242 | 找出还没有访问过的邻居。 | GeneralDyG 邻居扩展。 |
| 243-244 | 如果邻居太多，就随机采样到剩余容量。 | 本脚本把节点截断前置到采样阶段。 |
| 245 | 加入本层新节点。 | BFS 风格扩展。 |
| 246-248 | 更新访问集合、总节点集合和下一轮边界。 | BFS 风格扩展。 |
| 250 | 强制把另一端点 `dst` 加入 ego-graph。 | 保证 focal edge 两端点都存在。 |
| 251 | 转成 list，方便后续随机采样。 | 本脚本实现细节。 |
| 252-256 | 如果节点数仍超过 `max_nodes`，保留 focal edge 两端点，再随机保留其他节点。 | GeneralDyG `max_mask_len` 思路。 |
| 257 | 返回最终节点集合。 | 后续 induced subgraph 输入。 |

## `reorder_nodes`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 259-260 | 定义静态方法：对子图节点重新编号。 | GeneralDyG `reorder_nodes`。 |
| 261 | 创建字典保存每个节点的最早关联事件时间。 | GeneralDyG。 |
| 262 | 遍历子图节点。 | GeneralDyG。 |
| 263 | 取出该节点关联边的 `weight`。 | GeneralDyG。 |
| 264 | 初始化权重列表。 | 本脚本实现细节。 |
| 265-269 | 如果边权是列表，取其中最小事件 ID；否则直接用权重。 | 适配多次交互。 |
| 270 | 该节点用最早关联事件 ID 表示时间顺序；孤立节点用无穷大。 | GeneralDyG 时间排序思想。 |
| 271 | 按最早事件 ID 排序，生成旧节点 ID 到局部连续 ID 的映射。 | GeneralDyG。 |

## `replace_subgraph`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 273-280 | 定义函数：把原始子图替换成局部编号子图，并生成节点/边特征查找表。 | GeneralDyG `replace_subgraph`。 |
| 281 | 创建新的局部子图。 | GeneralDyG。 |
| 282 | 初始化节点特征数组，长度等于子图节点数。 | GeneralDyG。 |
| 283 | 初始化边特征字典，key 为局部边，value 为事件 ID。 | GeneralDyG。 |
| 285-287 | 遍历节点映射，保存原始节点 ID，并把局部节点加入新图。 | GeneralDyG。 |
| 289 | 遍历原始子图的每条边。 | GeneralDyG。 |
| 290 | 把原始端点转换成局部端点。 | GeneralDyG。 |
| 291 | 如果同一对端点有多次交互，随机选一个事件 ID 作为这条局部边的边特征。 | GeneralDyG。 |
| 292-293 | 如果这条边正是当前要预测的 focal edge，强制使用当前事件 ID。 | GeneralDyG focal edge 处理。 |
| 294 | 在局部图中添加边，权重设为 1。 | 后续邻接矩阵只需要结构。 |
| 295 | 标准化局部无向边顺序。 | 本脚本实现细节。 |
| 296 | 保存局部边到事件 ID 的映射。 | DP-DGAD `edgefeatures` 来源。 |
| 298 | 返回局部子图、节点特征数组、边特征查找表。 | 后续矩阵构造输入。 |

## `prune_edges`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 300 | 定义函数：可选地限制每个 ego-graph 的边数。 | 本脚本工程参数。 |
| 301-302 | 如果没有设置 `max_edges`，或当前边数已经不超限，就原样返回。 | 用户当前运行 Wikipedia/MOOC 时未设置 `--max-edges`。 |
| 304 | 如果子图里存在 focal edge，就记录它。 | 保证当前样本目标边不被删。 |
| 305 | 收集除 focal edge 以外的其他边。 | 本脚本实现细节。 |
| 306 | 计算还需要保留多少其他边。 | 本脚本实现细节。 |
| 307 | 从其他边里随机采样保留边。 | 工程内存控制。 |
| 308-309 | 如果有 focal edge，把它加入保留集合。 | 保证样本有效。 |
| 311-314 | 新建裁剪后的图，保留原节点和被选中的边及其属性。 | NetworkX 子图处理。 |
| 315 | 返回裁剪后的子图。 | 后续矩阵构造输入。 |

注意：`max_edges` 默认是 `None`。所以默认不会删边；只有命令行显式传 `--max-edges` 时才会裁剪。

## 矩阵构造工具函数

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 317-319 | `sparse_to_dense_tensor`：把 SciPy 稀疏矩阵转成 PyTorch dense tensor。 | DP-DGAD 当前模型使用 dense tensor。 |
| 321-326 | `normalize`：按行归一化稀疏矩阵，空行的无穷大倒数置 0。 | GeneralDyG `normalize`。 |
| 328-338 | `create_transition_matrix`：从节点邻接矩阵构造节点-边关联矩阵 `T`。 | GeneralDyG `create_transition_matrix`。 |
| 330-332 | 拷贝邻接矩阵、去掉自环、只取上三角边，避免无向边重复。 | GeneralDyG。 |
| 333-338 | 为每条边的两个端点填 1，得到形状 `[num_nodes, num_edges]` 的 CSR 矩阵。 | DP-DGAD `CensNet` 需要。 |
| 340-354 | `create_edge_adj`：构造边邻接矩阵，两条边共享节点则相邻，并给边加自连接。 | GeneralDyG `create_edge_adj`。 |

## `build`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 356 | 定义 `build`，正式生成 DP-DGAD 的数据字典。 | GeneralDyG `get_batch_data` + DP-DGAD `datasets.py`。 |
| 357-362 | 初始化输出字段列表：节点特征、边特征、`T`、节点邻接、边邻接、`ra`。 | DP-DGAD 所需字段。 |
| 364 | 如果传入 `limit`，只处理前 N 条事件，用于 smoke test。 | 本脚本调试功能。 |
| 365 | 遍历每条事件，并显示 `Building ego-graphs` 进度条。 | GeneralDyG。 |
| 366 | 取当前事件的源节点、目标节点、事件 ID。 | GeneralDyG `u/i/id`。 |
| 367-368 | 分别从两个端点抽取 k-hop 节点。 | GeneralDyG ego-graph。 |
| 369 | 合并两侧节点，得到当前样本的节点集合。 | GeneralDyG。 |
| 370 | 从全图中抽取 induced subgraph。 | NetworkX/GeneralDyG。 |
| 372-373 | 如果 focal edge 不在子图里，补进去。异常 non-edge 可能需要这一步。 | 为 TADDY 异常注入适配。 |
| 374-375 | 如果 focal edge 已存在，确保当前事件 ID 在该边的 `weight` 中。 | 多次交互事件适配。 |
| 376 | 可选裁剪边数；默认不裁剪。 | 本脚本工程参数。 |
| 378 | 对子图节点按时间顺序重新编号。 | GeneralDyG。 |
| 379-381 | 替换成局部编号子图，并得到节点特征和边特征查找表。 | GeneralDyG。 |
| 383 | 生成局部节点邻接矩阵。 | NetworkX。 |
| 384 | 构造节点-边关联矩阵 `T`。 | GeneralDyG。 |
| 385 | 构造边邻接矩阵和局部边名列表。 | GeneralDyG。 |
| 387-390 | 按边名顺序生成当前样本的边特征数组。 | DP-DGAD `edgefeatures`。 |
| 392 | 保存 `Tmat`。 | DP-DGAD 输入。 |
| 393 | 保存归一化后的节点邻接矩阵，并加自环。 | GeneralDyG。 |
| 394 | 保存归一化后的边邻接矩阵。 | GeneralDyG。 |
| 395 | 保存节点特征。 | DP-DGAD/GeneralDyG。 |
| 396 | 保存边特征。 | DP-DGAD/GeneralDyG。 |
| 397 | 保存 `ra` 占位数组。 | DP-DGAD 的 `DGG.forward` 接收 `ra`。 |
| 399 | 把所有边 ID 压缩成连续编号。 | 适配 DP-DGAD loader。 |
| 400 | 从事件表取标签数组。 | DP-DGAD `labels`。 |
| 401-409 | 返回最终数据字典，字段名与 DP-DGAD `datasets.py` 对齐。 | DP-DGAD 数据格式。 |

## `compact_edge_ids`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 411-412 | 定义静态方法，把局部边 ID 压缩为连续 ID。 | 本脚本新增。 |
| 413 | 函数说明：DP-DGAD loader 需要边 ID 能索引边特征矩阵。 | DP-DGAD `datasets.py`。 |
| 414 | 找到所有出现过的边 ID。 | 本脚本新增。 |
| 415 | 建立旧 ID 到新连续 ID 的映射。 | 本脚本新增。 |
| 416 | 对每个样本的边 ID 数组执行重映射。 | 本脚本新增。 |

## `parse_args`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 423 | 定义命令行参数解析函数。 | Python `argparse`。 |
| 424 | 创建 parser，并写明脚本用途。 | 本脚本工程入口。 |
| 425 | `--dataset`：可选无标签数据或有标签源数据。 | `DATASET_CHOICES`。 |
| 426 | `--raw-dir`：TADDY raw 数据目录，默认 `../TADDY/data/raw`。 | TADDY 数据。 |
| 427 | `--dynamic-data-dir`：Wikipedia/MOOC 原始数据目录，默认 `../Data/Dynamic_Graph_data`。 | GeneralDyG 原始数据。 |
| 428 | `--output-dir`：输出目录，默认 DP-DGAD 的 `data`。 | DP-DGAD 默认读取 `./data`。 |
| 429 | `--output-name`：自定义输出文件名前缀。 | 本脚本新增。 |
| 430 | `--train-per`：无标签异常注入时训练边比例，默认 0.5。 | TADDY。 |
| 431 | `--anomaly-per`：无标签异常注入比例，限制为 1%、5%、10%。 | DP-DGAD 实验设置。 |
| 432 | `--anomaly-base`：异常数量按总边数或测试边数计算。 | DP-DGAD 论文和 TADDY 代码差异。 |
| 433-438 | `--event-mode`：无标签数据的事件处理模式，`unique_undirected` 为 TADDY 风格唯一无向边，`temporal_events` 为保留原始时间事件。 | 本脚本新增。 |
| 439 | `--k-hop`：ego-graph 跳数，默认 1。 | GeneralDyG。 |
| 440 | `--max-nodes`：每个 ego-graph 最多节点数，默认 26。 | GeneralDyG `max_mask_len` 思路。 |
| 441 | `--max-edges`：每个 ego-graph 最多边数，默认 `None`，即不限制。 | 本脚本工程参数。 |
| 442 | `--seed`：随机种子。 | 可复现实验。 |
| 443 | `--limit`：只构造前 N 条事件，用于小规模测试。 | 本脚本新增。 |
| 444 | `--save-csv`：同时保存标准事件表 CSV。 | 便于检查标签和事件顺序。 |
| 445 | 返回解析后的参数对象。 | `argparse`。 |

## `main`

| 行号 | 代码在做什么 | 参考来源 |
|---:|---|---|
| 448 | 定义主函数。 | Python 常规入口。 |
| 449 | 解析命令行参数。 | `parse_args`。 |
| 450 | 把数据集名转小写，统一匹配。 | 本脚本新增。 |
| 451-452 | 设置 Python 和 NumPy 随机种子。 | 可复现实验。 |
| 454-455 | 如果是有标签数据集，调用 `load_labeled_events`，直接读取原始标签。 | Wikipedia/MOOC 源数据路线。 |
| 456-458 | 否则走无标签路线：先按 `event_mode` 调用 `load_raw_edges`，再 `inject_anomalies`。 | TADDY 异常注入路线 + DP-DGAD 时间事件模式。 |
| 460-466 | 创建 `EgoGraphBuilder`，传入事件表和 ego-graph 参数。 | GeneralDyG 数据构造。 |
| 467 | 调用 `build` 生成 DP-DGAD 数据字典。 | DP-DGAD `.pkl` 格式。 |
| 469 | 创建输出目录。 | 本脚本工程处理。 |
| 470 | 生成默认输出名：有标签数据用 `Wikipedia/MOOC`，无标签数据用 `dataset_异常百分比`。 | 本脚本新增。 |
| 471-472 | 如果是小样本测试，在输出名里加 `limitN`。 | 本脚本新增。 |
| 473 | 如果用户传了 `--output-name`，优先使用用户指定名称。 | 本脚本新增。 |
| 474 | 拼出 `.pkl` 输出路径。 | DP-DGAD 数据文件。 |
| 475-476 | 用 pickle 保存数据字典。 | DP-DGAD/GeneralDyG 数据保存方式。 |
| 478-481 | 如果启用 `--save-csv`，保存标准事件表 CSV，并打印路径。 | 便于人工检查。 |
| 483 | 打印 `.pkl` 保存路径。 | 本脚本运行提示。 |
| 484 | 打印样本数和异常数。 | 本脚本运行提示。 |
| 487-488 | 当脚本被直接运行时，调用 `main()`。 | Python 常规入口。 |

## 输出文件字段说明

| 字段 | 类型 | 含义 | DP-DGAD 中如何使用 |
|---|---|---|---|
| `nodefeatures` | `np.array(dtype=object)` | 每个样本 ego-graph 的局部节点 ID 列表，不是原始节点属性向量。 | `datasets.py` 用这些 ID 索引随机初始化的节点特征矩阵。 |
| `edgefeatures` | `np.array(dtype=object)` | 每个样本 ego-graph 的局部边事件 ID，已经压缩为连续 ID，不是原始 CSV 后面的 event feature 向量。 | `datasets.py` 用这些 ID 索引随机初始化的边特征矩阵。 |
| `labels` | `np.ndarray` | 样本标签，当前统一为 `0=正常`，`1=异常`。 | 训练和评估时计算 BCE、AUC、AP。 |
| `Tmats` | `list[torch.Tensor]` | 节点-边关联矩阵，形状 `[num_nodes, num_edges]`。 | `CensNet` 中节点和边相互传信息。 |
| `adjs` | `list[torch.Tensor]` | 归一化后的节点邻接矩阵，带自环。 | 节点卷积使用。 |
| `eadjs` | `list[torch.Tensor]` | 归一化后的边邻接矩阵。两条边共享节点则相邻。 | 边卷积使用。 |
| `ra` | `np.array(dtype=object)` | 当前是占位数组。 | DP-DGAD 的 `DGG.forward` 接收该字段，但主路径中基本未实际使用。 |

## 常用运行命令

生成 Wikipedia 源数据：

```bash
cd /home/qfu/bx82_scratch2/qfu/[A]GAD_202606_learning/DP-DGAD
source /home/qfu/bx82_scratch2/qfu/conda_envs/DPGAD/bin/activate

python prepare_dpgad_data.py \
  --dataset wikipedia \
  --output-name Wikipedia \
  --save-csv
```

生成 MOOC 源数据：

```bash
python prepare_dpgad_data.py \
  --dataset mooc \
  --output-name MOOC \
  --save-csv
```

生成 10% 异常比例的 Bitcoin-Alpha 数据：

```bash
python prepare_dpgad_data.py \
  --dataset btc_alpha \
  --event-mode temporal_events \
  --anomaly-per 0.1 \
  --max-edges 24 \
  --output-name btc_alpha \
  --save-csv
```

如果要复刻 TADDY 唯一无向边风格，可以显式指定：

```bash
python prepare_dpgad_data.py \
  --dataset btc_alpha \
  --event-mode unique_undirected \
  --anomaly-per 0.1 \
  --max-edges 24 \
  --output-name btc_alpha_taddy_unique \
  --save-csv
```

只生成 100 条样本做格式检查：

```bash
python prepare_dpgad_data.py \
  --dataset btc_alpha \
  --anomaly-per 0.01 \
  --limit 100 \
  --output-name smoke_btc_alpha \
  --save-csv
```

## 与参考代码的主要差异

1. TADDY 的异常数量默认按测试边数计算；DP-DGAD 论文 A.1 写的是 `p*m`。本脚本默认按论文写法，并提供 `--anomaly-base test` 复刻 TADDY 代码。
2. 无标签数据现在支持两种事件模式：`unique_undirected` 会按端点去重；`temporal_events` 会保留原始时间交互。Wikipedia/MOOC 有标签事件不会按端点去重。
3. Wikipedia/MOOC 是二部图，本脚本给 item/target 节点加 offset，避免它们和 user 节点 ID 冲突。
4. GeneralDyG 原始数据构造逻辑被整理到 `EgoGraphBuilder`，并新增 `max_edges=None` 的可选边裁剪。
5. DP-DGAD 比 GeneralDyG 多要求 `ra` 字段，本脚本先填充占位值。
6. DP-DGAD 的 `datasets.py` 要求 `edgefeatures` 里的 ID 能连续索引边特征矩阵，因此本脚本用 `compact_edge_ids` 做额外重映射。
7. TADDY 仓库实际运行入口调用 `anomaly_generation()`，它会先做谱聚类，再优先选择跨簇 fake edges；本脚本当前采用更直接的随机 non-edge 采样，没有做谱聚类。
