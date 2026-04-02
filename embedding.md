# `embedding.cpp` 说明

本文档对应实现文件 [src/embedding.cpp](/home/norising/K_SpecPart_C/src/embedding.cpp) 和声明文件 [include/kspecpart/embedding.hpp](/home/norising/K_SpecPart_C/include/kspecpart/embedding.hpp)。

## 当前进度

- 当前 `embedding.cpp` 已经是可运行模块。
- 它已经接入当前 C++ 主流程，会为 `tree_partition.cpp` 和 `specpart.cpp` 提供谱嵌入矩阵。
- 当前已经完成：
  - `solve_eigs(...)` 主接口
  - 小图分支的图 Laplacian 稠密特征分解
  - 大图分支的超图广义特征值问题
  - 超图算子 `A`
  - clique / biclique 算子 `B`
  - 去掉平凡零空间的 Helmert 基
  - 按维数截取所需特征向量
- 当前仍保留的非完全 Julia 一致点：
  - Julia 用的是 `LinearMap + lobpcg + CMG preconditioner`，C++ 目前是 Eigen 的稠密特征值求解。
  - `iterations` 参数目前尚未实际控制求解过程。
  - Julia 里的 `projection_step!` 还没有单独移植。
  - 当前实现的“数学目标”已经接近 Julia，但“数值求解路径”仍然是替代实现。

## 模块在整体流程中的位置

`embedding.cpp` 的职责是：

1. 接收图化后的普通图 `WeightedGraph`。
2. 根据超图 `Hypergraph` 构造对应的谱算子。
3. 根据 fixed vertices 构造额外的 biclique 约束。
4. 求出若干维 embedding 矩阵。
5. 把 embedding 交给 `tree_partition.cpp` 做 graph reweight 和 tree candidate 生成。

在当前 C++ 管线中，它处于：

1. 读入超图
2. isolate / preprocess
3. graphification
4. `solve_eigs(...)`
5. tree partition / refine

## 关键数据结构

### `Hypergraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

- `num_vertices`：顶点数 `n`
- `num_hyperedges`：超边数 `m`
- `eptr`：长度为 `m + 1` 的偏移数组
- `eind`：所有超边 pin 的顶点编号拼接数组
- `vptr`：顶点到 incident 超边的偏移数组
- `vind`：incident 超边编号拼接数组
- `fixed`：固定顶点信息，`-1` 表示未固定
- `vwts`：顶点权重
- `hwts`：超边权重

索引约定：

- C++ 内部全部使用 `0` 基索引。

### `WeightedGraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

- `num_vertices`：图顶点数
- `adjacency`：邻接表，`adjacency[u]` 中的每个元素 `(v, w)` 表示边 `u-v`
- `degrees`：每个顶点的 weighted degree

### `PartitionIndex`

定义位置：[include/kspecpart/cut_distillation.hpp](/home/norising/K_SpecPart_C/include/kspecpart/cut_distillation.hpp)

- `p1`：固定到 part `0` 的顶点列表
- `p2`：固定到 part `1` 的顶点列表

说明：

- 在 `embedding.cpp` 中，它主要用于构造 biclique 约束矩阵。
- 如果 `p1` 和 `p2` 都为空，则不加该约束项。

## 输出数据格式

### embedding 矩阵

`solve_eigs(...)` 返回值类型为：

```cpp
Eigen::MatrixXd
```

含义：

- 行数：`hypergraph.num_vertices`
- 列数：实际返回的 embedding 维数，通常不超过 `requested_dims`
- 第 `v` 行：顶点 `v` 的 embedding 坐标
- 第 `d` 列：第 `d` 个特征向量

例如：

- 返回形状 `n x 1`：表示一维谱坐标
- 返回形状 `n x k`：表示 `k` 维谱嵌入

## 私有函数说明

### `dense_graph_laplacian(const WeightedGraph& graph)`

- 输入：
  - `graph`
- 输出：
  - `n x n` 的稠密实对称矩阵
- 含义：
  - 构造普通图的 Laplacian：

```text
L = D - A + 1e-6 I
```

- 细节：
  - 对角线项来自 `graph.degrees`
  - 非对角项来自邻接边权
  - 添加 `1e-6` 是为了数值稳定

### `hyperedge_scale(int edge_size)`

- 输入：
  - `edge_size`：某条超边的 pin 数
- 输出：
  - `double`
- 含义：
  - 计算超边在超图算子中的缩放因子：

```text
floor(edge_size / 2) * ceil(edge_size / 2) / (edge_size - 1)
```

- 作用：
  - 用于把超边贡献归一到 `dense_hypergraph_operator(...)`

### `dense_hypergraph_operator(const Hypergraph& hypergraph, int epsilon)`

- 输入：
  - `hypergraph`
  - `epsilon`
- 输出：
  - `n x n` 的稠密矩阵
- 含义：
  - 构造当前 C++ 版的超图算子 `A`
- 细节：
  - 对每条超边逐个累加贡献
  - 若超边大小 `<= 1`，则忽略
  - `epsilon` 至少被夹到 `1`
- 对应 Julia：
  - 近似对应 `make_a_func(...)` / `hypl(H, X, epsilon)` 的线性算子作用

### `dense_clique_operator(const Hypergraph& hypergraph)`

- 输入：
  - `hypergraph`
- 输出：
  - `n x n` 的稠密矩阵
- 含义：
  - 构造 clique 风格的 `B` 矩阵
- 公式直观上是：

```text
B = Diag(vwts) - vwts * vwts^T / sum(vwts)
```

- 作用：
  - 作为广义特征值问题中的右侧矩阵基底

### `dense_biclique_operator(int num_vertices, const PartitionIndex& pindex)`

- 输入：
  - `num_vertices`
  - `pindex`
- 输出：
  - `n x n` 稠密矩阵
- 含义：
  - 根据 fixed vertices 的两侧集合 `p1/p2` 构造 biclique 约束
- 作用：
  - 若某些点被固定到两侧 partition，这个矩阵会鼓励两侧被拉开

### `helmert_complement_basis(int n)`

- 输入：
  - `n`
- 输出：
  - 形状为 `n x (n - 1)` 的矩阵
- 含义：
  - 构造 Helmert complement basis，用于去掉常数向量对应的平凡零空间
- 作用：
  - 把广义特征值问题投影到 `1` 向量的正交补空间

### `take_eigenvectors(...)`

- 输入：
  - `eigenvectors`：已经求出的特征向量矩阵
  - `largest`：是否取最大的特征向量
  - `requested_dims`：想要的维数
  - `skip_smallest`：需要跳过的最小特征向量个数
- 输出：
  - 截断后的 `Eigen::MatrixXd`
- 含义：
  - 从完整特征向量矩阵中选出最终要返回的列
- 常见用法：
  - 对图 Laplacian 的小图分支，会跳过最小的平凡特征向量

### `solve_small_graph_problem(const WeightedGraph& graph, bool largest, int requested_dims)`

- 输入：
  - `graph`
  - `largest`
  - `requested_dims`
- 输出：
  - embedding 矩阵
- 含义：
  - 当 `n < 100` 时，直接对普通图 Laplacian 做稠密对称特征分解
- 当前逻辑：
  - 若 `largest == false`，跳过最小特征向量后返回后续列
  - 若 `largest == true`，从最大特征值方向返回
- 说明：
  - 这是当前 C++ 的“小图直解”路径
  - 对应 Julia 小图分支的 `lobpcg(lap_matrix, ...)`

## 公共 API 说明

### `solve_eigs(...)`

声明位置：[include/kspecpart/embedding.hpp](/home/norising/K_SpecPart_C/include/kspecpart/embedding.hpp)

定义位置：[src/embedding.cpp](/home/norising/K_SpecPart_C/src/embedding.cpp)

函数签名：

```cpp
Eigen::MatrixXd solve_eigs(const Hypergraph& hypergraph,
                           const WeightedGraph& graph,
                           const PartitionIndex& pindex,
                           bool largest,
                           int requested_dims,
                           int iterations,
                           int epsilon = 1);
```

输入：

- `hypergraph`
  - 含义：原始超图，用于构造超图算子
- `graph`
  - 含义：图化后的普通图
  - 用途：
    - 小图分支构造图 Laplacian
    - 大图分支提供规模判定和必要图信息
- `pindex`
  - 含义：固定顶点两侧索引
  - 用途：在大图分支中决定是否叠加 biclique 约束
- `largest`
  - 含义：是否取“大特征值方向”
  - 当前主流程里通常传 `false`
- `requested_dims`
  - 含义：希望返回的 embedding 维数
  - 实际返回维数不会超过 `n - 1`
- `iterations`
  - 含义：与 Julia 的 `solver_iters` 对应
  - 当前状态：接口保留，但当前实现还没有真正用它控制 Eigen 求解过程
- `epsilon`
  - 含义：超图算子缩放参数
  - 在 two-way / k-way 路径中由上层传入

输出：

- 返回 `Eigen::MatrixXd`
- 形状通常为 `num_vertices x dims`
- 若输入为空图或 `requested_dims <= 0`，可能返回空矩阵或 `n x 0` 矩阵

当前执行路径：

1. 若 `n == 0` 或 `requested_dims <= 0`，直接返回空结果。
2. 若 `n == 1`，返回 `1 x 1` 零矩阵。
3. 若 `n < 100`：
   - 调 `solve_small_graph_problem(...)`
4. 若 `n >= 100`：
   - 构造 `A = dense_hypergraph_operator(...)`
   - 构造 `B = dense_clique_operator(...)`
   - 若存在 fixed vertices，再加上 `500 * dense_biclique_operator(...)`
   - 用 `helmert_complement_basis(...)` 投影到约束子空间
   - 调 `Eigen::GeneralizedSelfAdjointEigenSolver`
   - 把 reduced eigenvectors 映射回原空间

失败回退行为：

- 若大图分支的广义特征值求解失败，会回退到 `solve_small_graph_problem(...)`

## 当前与 Julia 的对应关系

对应 Julia 文件：

- [embedding.jl](/home/norising/K_SpecPart/embedding.jl)

主要对应关系：

- Julia `make_a_func(...)`
  - 对应 C++ `dense_hypergraph_operator(...)`
- Julia `make_b_func(...)`
  - 对应 C++ `dense_clique_operator(...) + dense_biclique_operator(...)`
- Julia `solve_eigs(...)`
  - 对应 C++ `solve_eigs(...)`

当前一致点：

- 都以超图算子和 clique/biclique 约束为核心
- 都输出顶点级 embedding 矩阵
- 都把 fixed vertices 信息接入 embedding 阶段
- 都区分小规模和大规模问题

当前差异：

- Julia：
  - 使用 `LinearMap`
  - 使用 `lobpcg`
  - 大图时使用 `CMG` 预条件器
- C++：
  - 直接显式构造稠密矩阵
  - 小图用 `SelfAdjointEigenSolver`
  - 大图用 `GeneralizedSelfAdjointEigenSolver`

因此当前更准确的说法是：

- `embedding.cpp` 已经实现了“可运行的谱嵌入模块”
- 但尚未完成对 Julia 数值求解器层的逐步等价移植

## 适合汇报时的简述

可以把 `embedding.cpp` 概括成：

- 当前已经完成可运行的谱嵌入模块，能为后续树划分提供 embedding。
- 它已经实现了超图算子、clique/biclique 约束和特征向量提取。
- 目前与 Julia 版最大的差异不在接口，而在求解器内部：Julia 用 `lobpcg + CMG`，C++ 目前还是 Eigen 的稠密特征分解替代实现。

