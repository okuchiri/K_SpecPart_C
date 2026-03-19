# `graphification.cpp` 说明

本文档对应实现文件 [src/graphification.cpp](/home/norising/K_SpecPart_C/src/graphification.cpp) 和声明文件 [include/kspecpart/graphification.hpp](/home/norising/K_SpecPart_C/include/kspecpart/graphification.hpp)。

## 模块职责

这个模块负责把超图 `Hypergraph` 转成普通加权图 `WeightedGraph`，供后续谱嵌入使用。

它对应 Julia 版的 `graphification.jl`，当前 C++ 版实现的是一个可运行的基础图化版本。

## 相关数据结构

### `Hypergraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

字段含义：

- `num_vertices`
  含义：顶点数 `n`
- `num_hyperedges`
  含义：超边数 `m`
- `eptr`
  含义：超边 CSR 风格偏移数组，长度为 `m + 1`
  解释：第 `e` 条超边的 pin 位于 `eind[eptr[e] ... eptr[e+1]-1]`
- `eind`
  含义：所有超边 pin 的顶点编号拼接数组
- `vptr`
  含义：顶点到 incident hyperedge 的偏移数组
- `vind`
  含义：每个顶点关联的超边编号拼接数组
- `fixed`
  含义：固定顶点信息，`-1` 表示未固定，其它值表示固定到对应 partition
- `vwts`
  含义：顶点权重
- `hwts`
  含义：超边权重

索引约定：

- C++ 内部全部使用 0-based 索引
- `eind` 中存放的顶点编号也是 0-based

### `WeightedGraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

字段含义：

- `num_vertices`
  含义：图顶点数
- `adjacency`
  类型：`std::vector<std::vector<std::pair<int, double>>>`
  含义：邻接表
  解释：`adjacency[u]` 中每个元素 `(v, w)` 表示无向边 `u-v` 的权重为 `w`
- `degrees`
  含义：每个顶点的 weighted degree
  解释：`degrees[u] = sum_{(v,w) in adjacency[u]} w`

## 函数说明

### `add_undirected_edge`

定义位置：[src/graphification.cpp](/home/norising/K_SpecPart_C/src/graphification.cpp)

函数签名：

```cpp
void add_undirected_edge(std::vector<std::unordered_map<int, double>>& accum,
                         int u,
                         int v,
                         double weight)
```

作用：

- 往临时累加结构 `accum` 中加入一条无向边 `u-v`
- 如果该边已经存在，则把权重累加上去

输入：

- `accum`
  含义：长度为 `n` 的数组，`accum[u][v]` 表示当前已累计的边权
- `u`
  含义：边一端顶点编号，0-based
- `v`
  含义：边另一端顶点编号，0-based
- `weight`
  含义：要增加的边权

输出：

- 无返回值
- 副作用是修改 `accum`

边界行为：

- 如果 `u == v`，直接忽略，不加自环
- 如果 `weight == 0.0`，直接忽略

### `hypergraph_to_graph`

声明位置：[include/kspecpart/graphification.hpp](/home/norising/K_SpecPart_C/include/kspecpart/graphification.hpp)

定义位置：[src/graphification.cpp](/home/norising/K_SpecPart_C/src/graphification.cpp)

函数签名：

```cpp
WeightedGraph hypergraph_to_graph(const Hypergraph& hypergraph, int cycles, std::mt19937& rng);
```

作用：

- 把超图转成普通加权图
- 输出的图用于后续谱嵌入

输入：

- `hypergraph`
  含义：待图化的超图
- `cycles`
  含义：对大超边进行 cycle-based 近似时，随机采样多少轮 cycle
  约束：如果传入小于 1 的值，函数内部会强制改成 1
- `rng`
  含义：随机数生成器
  用途：对大超边 pin 顺序进行随机打乱

输出：

- 返回一个 `WeightedGraph`
- 图是无向图
- `degrees` 已同步计算完成

## 图化规则

当前实现按超边大小分三种情况处理。

### 情况 1：超边大小 `size == 2`

做法：

- 直接把这条超边转成一条普通边

边权：

- 普通边权重 = 超边权重 `hwts[e]`

示例：

```text
hyperedge: {u, v}, weight = 5
graph edge: u-v, weight = 5
```

### 情况 2：超边大小 `size == 3`

做法：

- 把三元超边转成一个三角形

边权：

- 三条边每条边的权重 = `hwts[e] / 2`

示例：

```text
hyperedge: {a, b, c}, weight = 6
graph edges:
a-b = 3
b-c = 3
c-a = 3
```

### 情况 3：超边大小 `size >= 4`

做法：

- 先取该超边所有 pin
- 每轮随机打乱 pin 顺序
- 依次连接成一条环
- 重复 `cycles` 次

也就是说，如果随机排列后为：

```text
p0, p1, p2, ..., pk-1
```

则该轮加边：

```text
p0-p1, p1-p2, ..., p(k-2)-p(k-1), p(k-1)-p0
```

边权计算：

```text
scale = floor(size/2) * ceil(size/2) / (size - 1)
cycle_weight = hyperedge_weight / (cycles * 2 * scale)
```

每条被加入的环边使用 `cycle_weight`

含义：

- 当前实现不是 clique expansion
- 而是 Julia 版也使用过的一种随机 cycle-based 近似
- 目标是在大超边上避免生成过多普通边

## 输出图的数据格式

函数返回的 `WeightedGraph` 满足：

- `num_vertices == hypergraph.num_vertices`
- `adjacency.size() == num_vertices`
- 对任意边 `u-v`：
  - `v` 会出现在 `adjacency[u]`
  - `u` 会出现在 `adjacency[v]`
- `degrees[u]` 为 `adjacency[u]` 所有边权之和

注意：

- 邻接表里顶点编号为 0-based
- 边权类型为 `double`

## 这个模块在整体流程中的位置

当前 C++ 管线中，这个模块位于：

1. 读入超图
2. `isolate_islands`
3. `remove_single_hyperedges`
4. `hypergraph_to_graph`
5. `leading_eigenvectors`

因此，它的输出直接决定了谱嵌入看到的图拉普拉斯/邻接结构。

## 当前实现与 Julia 版的关系

对应 Julia 文件：

- [graphification.jl](/home/norising/K_SpecPart/graphification.jl)

当前一致点：

- 都是把超图映射成普通图
- 都对大超边使用随机 cycle 近似
- 都区分 `size == 2`、`size == 3`、`size >= 4`

当前差异：

- 当前文档描述的是当前 C++ 实现的精确行为
- 是否与 Julia 版每个权重细节完全一致，后续还需要继续逐行比对确认

## 适合汇报时的简述

可以把 `graphification.cpp` 概括成：

- 输入一个带顶点权重和超边权重的超图
- 按超边大小把它近似成普通加权图
- 小超边直接变成边或三角形
- 大超边用随机 cycle 方式近似
- 输出用于谱嵌入的 `WeightedGraph`
