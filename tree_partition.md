# `tree_partition.cpp` 说明

本文档说明当前 C++ 版 [`src/tree_partition.cpp`](./src/tree_partition.cpp) 的职责、完成进度、与 Julia `tree_partition.jl` 的对应关系，以及各函数的输入输出数据格式与含义。

## 当前进度

- 当前 `tree_partition.cpp` 已经是主流程可用模块。
- 它已经实际接入 `specpart.cpp`，会在 two-way / k-way 初始化与 refine 阶段反复调用。
- 目前已经完成：
  - embedding 子集枚举
  - graph reweight
  - LSST tree / MST tree 构造
  - two-way tree sweep
  - k-way recursive tree sweep
  - cut-distillation 驱动的 METIS cost-tree 构造
  - 候选分区去重、评分、筛选
  - 最终 best partition 选择
- 目前仍保留的非完全 Julia 一致点：
  - `local_refine_partition` 是 C++ 侧补充的局部搬移 refine，不是 Julia `tree_partition.jl` 的逐函数移植。
  - 
  - `construct_mst_tree` 已经按 Julia `degrees_aware_prim_mst(g, 10)` 的控制流贴近实现，但为了避免断连图导致后续流程失效，C++ 仍保留了多连通分量桥接的安全兜底。

## 模块在整体流程中的位置

`tree_partition.cpp` 的职责是：

1. 从 graphification 阶段得到的普通图和 embedding 出发，重新赋边权。
2. 基于重赋权后的图构造树结构：
   - `tree_type = 1`：LSST
   - `tree_type = 2`：MST
3. 在树上运行 `cut_distillation`，得到 `CutProfile`。
4. 用 tree sweep 和/或 METIS 生成候选分区。
5. 对候选分区做合法性检查、平衡检查、cutsize 评分和筛选。
6. 输出当前最优的候选分区，供 `specpart.cpp` 继续 refine / overlay / projection。

## 关键数据结构

### `Hypergraph`

定义位置：[`include/kspecpart/definitions.hpp`](./include/kspecpart/definitions.hpp)

- `num_vertices`：顶点数。
- `num_hyperedges`：超边数。
- `eptr`：长度 `num_hyperedges + 1`，超边到顶点列表的 CSR 指针。
- `eind`：所有超边成员顶点拼接后的数组，元素是 `0` 基顶点编号。
- `vptr`：长度 `num_vertices + 1`，顶点到 incident 超边列表的 CSR 指针。
- `vind`：所有顶点 incident 超边拼接后的数组，元素是 `0` 基超边编号。
- `fixed`：长度 `num_vertices`。`-1` 表示未固定；否则表示该顶点固定到哪个 partition。
- `vwts`：长度 `num_vertices`，顶点权重。
- `hwts`：长度 `num_hyperedges`，超边权重。

### `WeightedGraph`

定义位置：[`include/kspecpart/definitions.hpp`](./include/kspecpart/definitions.hpp)

- `num_vertices`：图顶点数。
- `adjacency`：邻接表，`adjacency[u]` 是若干 `(v, weight)` 二元组。
- `degrees`：每个顶点的加权度。

### `PartitionIndex`

定义位置：[`include/kspecpart/cut_distillation.hpp`](./include/kspecpart/cut_distillation.hpp)

- `p1`：固定到 part `0` 的顶点编号列表。
- `p2`：固定到 part `1` 的顶点编号列表。

说明：

- 当前主要用于 embedding 的 biclique 约束和 tree cut distillation。
- 顶点编号为 `0` 基。

### `CutProfile`

定义位置：[`include/kspecpart/cut_distillation.hpp`](./include/kspecpart/cut_distillation.hpp)

这是 `distill_cuts_on_tree(...)` 的输出。`tree_partition.cpp` 里主要会使用：

- `vtx_cuts[i]`：以树边 `i - pred[i]` 切开时，一侧子树的顶点总权重。
- `edge_cuts[i]`：该切法对应的基础 cut 统计。
- `pred[i]`：树上父节点编号；根满足 `pred[root] == root`。
- `forced_0` / `forced_1` / `forced_01`：受 fixed vertices 影响被强制归类的超边索引集合。
- `FB0` / `FB1`：固定顶点带来的补偿项。
- `edge_cuts_0` / `edge_cuts_1`：极性相关的 cut 补充统计。

### `BalanceLimits`

定义位置：[`include/kspecpart/tree_partition.hpp`](./include/kspecpart/tree_partition.hpp)

- `min_capacity`：每个 partition 允许的最小总顶点权重。
- `max_capacity`：每个 partition 允许的最大总顶点权重。

### `TreePartitionOptions`

定义位置：[`include/kspecpart/tree_partition.hpp`](./include/kspecpart/tree_partition.hpp)

- `num_parts`：目标 partition 数。
- `imb`：不平衡容忍度，单位与 Julia 保持一致，按百分比解释。
- `eigvecs`：embedding 维数上限。
- `solver_iters`：求特征向量时的迭代参数，会传给 `solve_eigs`。
- `cycles`：graphification 时的 cycle 数。
- `best_solns`：保留候选解数量参数。
- `seed`：随机种子。
- `gpmetis_executable`：`gpmetis` 路径。
- `enable_metis`：是否允许使用 `gpmetis` 候选。
- `gpmetis_explicit`：是否要求该路径必须显式可解析。

### `TreePartitionCandidate`

定义位置：[`include/kspecpart/tree_partition.hpp`](./include/kspecpart/tree_partition.hpp)

- `partition`：长度 `num_vertices` 的 partition 向量。
- `cutsize`：该 partition 在超图上的 cutsize。
- `balance`：每个 partition 的总顶点权重。

### Partition 向量格式

`tree_partition.cpp` 所有分区结果都用：

- 类型：`std::vector<int>`
- 长度：`hypergraph.num_vertices`
- 取值范围：`[0, num_parts - 1]`
- 含义：`partition[v] = p` 表示顶点 `v` 被分到 partition `p`

## 私有辅助结构

### `VectorHasher`

- 输入：`std::vector<int>`
- 输出：`std::size_t`
- 含义：用于把 partition 向量放进 `unordered_set` 去重。

### `ScoredPartition`

- `partition`：候选分区。
- `metrics`：`evaluate_partition(...)` 的结果。
- `penalty`：平衡惩罚值，越小越好。

### `TreeSweepResult`

- `partition`：由 tree sweep 得到的 partition。
- `cutsize`：该 partition 的 cutsize。
- `cut_point`：树上被切掉的子树根节点编号；若失败则为 `-1`。

## 私有函数说明

### `make_edge_key(int u, int v)`

- 输入：
  - `u`, `v`：无向边两端点，`0` 基顶点编号。
- 输出：
  - `std::uint64_t`
- 含义：
  - 把无向边 `(u, v)` 编码为唯一 key，用于 `removed_edges` 集合。

### `total_vertex_weight(const Hypergraph& hypergraph)`

- 输入：
  - `hypergraph`
- 输出：
  - 所有顶点权重之和。

### `partition_complete(const std::vector<int>& partition, int num_vertices)`

- 输入：
  - `partition`：候选 partition 向量。
  - `num_vertices`：应有顶点数。
- 输出：
  - `bool`
- 含义：
  - 检查 partition 长度是否正确，且每个元素都已赋值为非负整数。

### `fixed_vertices_satisfied(const Hypergraph& hypergraph, const std::vector<int>& partition, int num_parts)`

- 输入：
  - `hypergraph`
  - `partition`
  - `num_parts`
- 输出：
  - `bool`
- 含义：
  - 检查 partition 是否满足 fixed vertex 约束。

### `edge_is_cut_after_move(...)`

- 输入：
  - `hypergraph`
  - `partition`
  - `edge`：超边编号。
  - `moved_vertex`：尝试移动的顶点编号；若 `< 0` 则表示“不移动”。
  - `new_part`：移动后目标 partition。
- 输出：
  - `bool`
- 含义：
  - 判断指定移动前后，某条超边是否成为 cut edge。

### `delta_cut_for_move(...)`

- 输入：
  - `hypergraph`
  - `partition`
  - `vertex`：待移动顶点。
  - `to_part`：目标 partition。
- 输出：
  - `int`
- 含义：
  - 估计把一个顶点从当前 partition 移到 `to_part` 后，cutsize 的增量。
- 说明：
  - 该函数被 `local_refine_partition` 使用。

### `compute_balance(...)`

- 输入：
  - `hypergraph`
  - `partition`
  - `num_parts`
- 输出：
  - 长度为 `num_parts` 的整型数组。
- 含义：
  - 统计每个 partition 当前累积的顶点权重。

### `greedy_assign_unset_vertices(...)`

- 输入：
  - `hypergraph`
  - `partition`：可被原地修改，允许存在 `< 0` 的“未分配顶点”。
  - `limits`
  - `num_parts`
- 输出：
  - 无返回值，原地写回 `partition`。
- 含义：
  - 用简单平衡优先策略给尚未赋值的顶点补 partition。

### `coordinate_distance(const Eigen::MatrixXd& embedding, int lhs, int rhs, bool lst)`

- 输入：
  - `embedding`：形状为 `num_vertices x dims` 的矩阵。
  - `lhs`, `rhs`：两个顶点编号。
  - `lst`：
    - `true`：使用 LSST 风格的 `sum(1 / span^2)`
    - `false`：使用 `L1` 距离 `sum(abs(span))`
- 输出：
  - `double`
- 含义：
  - 计算 reweight 之后的图边权。

### `add_undirected_edge(WeightedGraph& graph, int u, int v, double weight)`

- 输入：
  - `graph`
  - `u`, `v`
  - `weight`
- 输出：
  - 无返回值，原地更新 `graph`
- 含义：
  - 往 `WeightedGraph` 中加入一条无向加权边。

### `make_empty_graph(int num_vertices)`

- 输入：
  - `num_vertices`
- 输出：
  - 初始化完成的 `WeightedGraph`
- 含义：
  - 创建空图并初始化邻接表与加权度数组。

### `spectral_order(const Eigen::MatrixXd& embedding, int num_vertices)`

- 输入：
  - `embedding`
  - `num_vertices`
- 输出：
  - 长度为 `num_vertices` 的顶点顺序。
- 含义：
  - 按第一列 embedding 坐标从小到大排序，作为 path / LSST bridge 的顺序依据。

### `reweight_graph(const WeightedGraph& graph, const Eigen::MatrixXd& embedding, bool lst)`

- 输入：
  - `graph`：graphification 输出的普通图。
  - `embedding`
  - `lst`
- 输出：
  - 新的 `WeightedGraph`
- 含义：
  - 将原图边权替换为由 embedding 坐标诱导的距离。
- 对应 Julia：
  - `reweigh_graph(adj, X, lst)`

### `construct_mst_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding)`

- 输入：
  - `graph`：已经完成 reweight 的普通图。
  - `embedding`：主要用于断连图时的桥接顺序和桥边权。
- 输出：
  - `WeightedGraph` 形式的树。
- 含义：
  - 构造 `tree_type = 2` 的树。
- 当前实现细节：
  - 主控制流已贴近 Julia `degrees_aware_prim_mst(g, 10)`。
  - 由于 Julia 原函数里 `degrees` 未更新，当前 C++ 也保持相同控制流。
  - 若输入图存在多个连通分量，C++ 会额外按 embedding 顺序桥接各分量，保证后续 cut distillation 可以在单棵树上运行。

### `construct_path_tree(const Eigen::MatrixXd& embedding)`

- 输入：
  - `embedding`
- 输出：
  - 路径树 `WeightedGraph`
- 含义：
  - 按第一维 embedding 顺序把顶点串成路径。
- 说明：
  - 当前 `generate_tree_candidates(...)` 不枚举 `tree_type = 3`，因此该函数暂未进入主流程。

### `construct_tree(const WeightedGraph& graph, const Eigen::MatrixXd& embedding, int tree_type)`

- 输入：
  - `graph`
  - `embedding`
  - `tree_type`
- 输出：
  - `WeightedGraph`
- 含义：
  - `tree_type = 1` 返回 LSST。
  - `tree_type = 2` 返回 MST。
  - `tree_type = 3` 返回 path tree。

### `compact_partition_labels(const std::vector<int>& components)`

- 输入：
  - 连通分量标签数组，标签值本身可以不连续。
- 输出：
  - 压缩后的 partition 向量，标签重新映射到 `0..k-1`。

### `connected_components(const WeightedGraph& graph, const std::unordered_set<std::uint64_t>& removed_edges)`

- 输入：
  - `graph`
  - `removed_edges`：需要视作“已删除”的树边集合。
- 输出：
  - 每个顶点所属连通分量标签。
- 含义：
  - 在“切掉若干树边”之后重新计算连通分量。

### `count_parts(const std::vector<int>& partition)`

- 输入：
  - `partition`
- 输出：
  - partition 数量。
- 含义：
  - 返回 `max(partition) + 1`。

### `sum_hyperedge_weights(const Hypergraph& hypergraph, const std::vector<int>& edges)`

- 输入：
  - `hypergraph`
  - `edges`：超边编号列表。
- 输出：
  - 这些超边的权重和。

### `build_metis_cost_tree(...)`

- 输入：
  - `tree`
  - `distilled_cuts`
  - `hypergraph`
- 输出：
  - 一个新的 `WeightedGraph`
- 含义：
  - 把树上每条父子边赋权为与 tree cut 代价相关的 cost，用于交给 `gpmetis`。
- 对应 Julia：
  - `METIS_tree_partition(...)` 内部构造 `T_matrix` 的那一段。

### `metis_tree_partition_candidate(...)`

- 输入：
  - `tree`
  - `distilled_cuts`
  - `hypergraph`
  - `num_parts`
  - `options`
  - `gpmetis_executable`
- 输出：
  - `std::optional<std::vector<int>>`
- 含义：
  - 若可调用 `gpmetis`，返回基于 METIS 的候选 partition；否则返回空。

### `partition_index_from_fixed(const Hypergraph& hypergraph)`

- 输入：
  - `hypergraph`
- 输出：
  - `PartitionIndex`
- 含义：
  - 从 `hypergraph.fixed` 中抽取固定到 `0/1` 两侧的顶点集合。

### `tree_root(const CutProfile& distilled_cuts)`

- 输入：
  - `distilled_cuts`
- 输出：
  - 根节点编号。
- 含义：
  - 在 `pred[v] == v` 的位置寻找 tree root。

### `evaluate_removed_edges(...)`

- 输入：
  - `tree`
  - `hypergraph`
  - `removed_edges`
  - `num_parts`
  - `cut_point`
- 输出：
  - `TreeSweepResult`
- 含义：
  - 在树上切掉若干边后，重新生成 partition，并计算 cutsize。
- 失败条件：
  - 生成的连通分量数不等于 `num_parts`
  - 不满足 fixed vertex 约束

### `two_way_linear_tree_sweep(...)`

- 输入：
  - `tree`
  - `distilled_cuts`
  - `hypergraph`
  - `limits`
- 输出：
  - `std::optional<TreeSweepResult>`
- 含义：
  - 对每个候选 cut edge 计算：
    - tree cut cost
    - 容量是否合法
    - ratio fallback cost
  - 选出最佳 cut point，并返回对应的 two-way partition。
- 对应 Julia：
  - `two_way_linear_tree_sweep(...)`

### `k_way_linear_tree_sweep(...)`

- 输入：
  - `tree`
  - `fixed_vertices`
  - `hypergraph`
  - `limits`
  - `num_parts`
- 输出：
  - `std::optional<std::vector<int>>`
- 含义：
  - 通过反复调用 two-way sweep，记录多个 cut point，最后一次性删去这些树边得到 k-way partition。
- 对应 Julia：
  - `k_way_linear_tree_sweep(...)`

### `select_columns(const Eigen::MatrixXd& embedding, const std::vector<int>& cols)`

- 输入：
  - `embedding`
  - `cols`：待选取的列号列表。
- 输出：
  - 过滤后的合法列号列表。

### `slice_embedding(const Eigen::MatrixXd& embedding, const std::vector<int>& cols)`

- 输入：
  - `embedding`
  - `cols`
- 输出：
  - 形状为 `num_vertices x |cols|` 的新矩阵。
- 含义：
  - 从完整 embedding 中抽取指定列集合，作为当前候选子 embedding。

### `enumerate_subsets_recursive(...)`

- 输入：
  - `dims`：总维数。
  - `start`：递归起点。
  - `current`：当前组合。
  - `subsets`：输出缓冲区。
- 输出：
  - 无返回值，原地往 `subsets` 添加所有组合。

### `enumerate_embedding_subsets(int dims)`

- 输入：
  - `dims`
- 输出：
  - 若干列号子集。
- 含义：
  - 枚举 embedding 的列组合。
- 当前策略：
  - `dims <= 8`：枚举所有非空子集。
  - `dims > 8`：只保留每个单维和“全维”。

### `score_candidates(...)`

- 输入：
  - `hypergraph`
  - `candidates`：候选 partition 列表。
  - `num_parts`
  - `limits`
- 输出：
  - `std::vector<ScoredPartition>`
- 含义：
  - 去重、过滤非法 partition、计算 cutsize 和 balance penalty，并排序。

### `generate_tree_candidates(...)`

- 输入：
  - `hypergraph`
  - `graph`
  - `embedding`
  - `fixed_vertices`
  - `num_parts`
  - `limits`
  - `options`
  - `gpmetis_executable`
- 输出：
  - 候选 partition 列表。
- 含义：
  - 枚举 embedding 子集
  - 对每个子集尝试 `tree_type = 1, 2`
  - 生成 LSST/MST
  - 调 `distill_cuts_on_tree(...)`
  - 再调用：
    - `metis_tree_partition_candidate(...)`
    - `two_way_linear_tree_sweep(...)` 或 `k_way_linear_tree_sweep(...)`

### `generate_two_way_candidates(...)`

- 输入：
  - 与 `generate_tree_candidates(...)` 基本相同，额外含：
    - `base_partition`
    - `best_solns`
    - `rng`
- 输出：
  - two-way 候选 partition 列表。
- 当前状态：
  - 目前只是转发到 `generate_tree_candidates(...)`。
  - `base_partition`、`best_solns`、`rng` 尚未影响逻辑。

### `generate_k_way_candidates(...)`

- 输入：
  - 与 `generate_two_way_candidates(...)` 类似，但包含 `num_parts`
- 输出：
  - k-way 候选 partition 列表。
- 当前状态：
  - 目前只是转发到 `generate_tree_candidates(...)`。
  - `base_partition`、`best_solns`、`rng` 尚未影响逻辑。

### `fallback_partition(const Hypergraph& hypergraph, int num_parts)`

- 输入：
  - `hypergraph`
  - `num_parts`
- 输出：
  - 一个保底 partition。
- 含义：
  - fixed vertex 按 fixed 保留；
  - 其余顶点按 `vertex % num_parts` 平铺。
- 用途：
  - 当前没有任何合法候选分区时的兜底。

## 公共 API 说明

### `compute_balance_limits(const Hypergraph& hypergraph, int num_parts, int imb)`

- 输入：
  - `hypergraph`
  - `num_parts`
  - `imb`
- 输出：
  - `BalanceLimits`
- 含义：
  - 计算每个 partition 的最小/最大容量限制。
- 对应 Julia：
  - `specpart.jl` 中构造 `capacities = [min_capacity, max_capacity]` 的逻辑。

### `balance_penalty(const std::vector<int>& balance, const BalanceLimits& limits)`

- 输入：
  - `balance`
  - `limits`
- 输出：
  - `long long`
- 含义：
  - 计算 balance 违反上下界时的平方惩罚。
  - `0` 表示满足容量限制。

### `local_refine_partition(...)`

- 输入：
  - `hypergraph`
  - `partition`
  - `num_parts`
  - `limits`
  - `rng`
- 输出：
  - refine 后的 partition
- 含义：
  - 先补齐未赋值顶点，再做局部顶点搬移以降低 balance penalty，并尽量降低 cutsize。
- 说明：
  - 这是 C++ 版本加入的辅助 refine，不是 `tree_partition.jl` 的逐函数直译。

### `tree_partition_with_embedding(...)`

- 输入：
  - `hypergraph`
  - `graph`：普通图
  - `embedding`
  - `fixed_vertices`
  - `options`
  - `base_partition`
  - `rng`
- 输出：
  - `std::vector<TreePartitionCandidate>`
- 含义：
  - 在给定 graph 和 embedding 的前提下，直接生成并评分所有候选树划分结果。

### `tree_partition(...)`

- 输入：
  - `hypergraph`
  - `options`
  - `base_partition`
  - `rng`
- 输出：
  - `std::vector<TreePartitionCandidate>`
- 含义：
  - 完整入口：
    - 调 `hypergraph_to_graph(...)`
    - 调 `solve_eigs(...)`
    - 再调用 `tree_partition_with_embedding(...)`

### `tree_partition_best_with_embedding(...)`

- 输入：
  - 与 `tree_partition_with_embedding(...)` 相同
- 输出：
  - 当前最优 partition
- 含义：
  - 从候选集中选择 penalty 最小、cutsize 最小的 partition，并再经过 `local_refine_partition(...)`。

### `tree_partition_best(...)`

- 输入：
  - `hypergraph`
  - `options`
  - `base_partition`
  - `rng`
- 输出：
  - 当前最优 partition
- 含义：
  - `tree_partition(...)` 的 best-only 包装版本。

### `partition_two_way_hypergraph(...)`

- 输入：
  - `hypergraph`
  - `imb`
  - `eigvecs`
  - `solver_iters`
  - `cycles`
  - `best_solns`
  - `base_partition`
  - `rng`
- 输出：
  - two-way partition
- 含义：
  - 为 `num_parts = 2` 填充 `TreePartitionOptions`，再调用 `tree_partition_best(...)`。

### `partition_k_way_hypergraph(...)`

- 输入：
  - `hypergraph`
  - `num_parts`
  - `imb`
  - `eigvecs`
  - `solver_iters`
  - `cycles`
  - `best_solns`
  - `base_partition`
  - `rng`
- 输出：
  - k-way partition
- 含义：
  - 为一般 `k` 填充 `TreePartitionOptions`，再调用 `tree_partition_best(...)`。

## 当前与 Julia 的对应关系总结

大体对应关系如下：

- Julia `reweigh_graph(...)`
  - 对应 C++ `reweight_graph(...)`
- Julia `construct_tree(...)`
  - 对应 C++ `construct_tree(...)`
- Julia `two_way_linear_tree_sweep(...)`
  - 对应 C++ `two_way_linear_tree_sweep(...)`
- Julia `k_way_linear_tree_sweep(...)`
  - 对应 C++ `k_way_linear_tree_sweep(...)`
- Julia `METIS_tree_partition(...)`
  - 对应 C++ `build_metis_cost_tree(...) + metis_tree_partition_candidate(...)`
- Julia `tree_partition(...)`
  - 对应 C++ `tree_partition(...) / tree_partition_best(...)`
