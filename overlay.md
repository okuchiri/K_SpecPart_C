# `overlay.cpp` 说明

本文档对应实现文件 [src/overlay.cpp](/home/norising/K_SpecPart_C/src/overlay.cpp) 和声明文件 [include/kspecpart/overlay.hpp](/home/norising/K_SpecPart_C/include/kspecpart/overlay.hpp)。

## 模块职责

这个模块负责把多组 partition 结果叠加起来，找出“哪些超边在这些 partition 中被切开过”，然后把这些超边视为边界，把其余未被切开的部分收缩成 cluster，最终构造 coarse hypergraph。

这就是当前 C++ 版中的 overlay 流程。

对应 Julia 版的：

- `overlay.jl`

## 相关数据结构

### `Hypergraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

含义与数据格式：

- `num_vertices`
  顶点数
- `num_hyperedges`
  超边数
- `eptr`
  长度为 `num_hyperedges + 1` 的偏移数组
- `eind`
  所有超边 pin 的顶点编号拼接数组
- `fixed`
  顶点是否固定到某个 partition
- `vwts`
  顶点权重
- `hwts`
  超边权重

索引约定：

- C++ 内部一律是 0-based

### `OverlayResult`

定义位置：[include/kspecpart/overlay.hpp](/home/norising/K_SpecPart_C/include/kspecpart/overlay.hpp)

字段含义：

- `hypergraph`
  含义：overlay 后收缩得到的 coarse hypergraph
- `clusters`
  类型：`std::vector<int>`
  长度：等于原超图顶点数
  含义：`clusters[v]` 表示原顶点 `v` 属于哪个 coarse cluster

注意：

- `clusters` 中的 cluster id 是 0-based
- `clusters[v]` 可直接作为 coarse graph / coarse hypergraph 的顶点编号

## 函数说明

### `cut_edges_for_partition`

定义位置：[src/overlay.cpp](/home/norising/K_SpecPart_C/src/overlay.cpp)

函数签名：

```cpp
std::vector<int> cut_edges_for_partition(const Hypergraph& hypergraph,
                                         const std::vector<int>& partition);
```

作用：

- 给定一组 partition，找出哪些超边被切开

输入：

- `hypergraph`
  含义：原始超图
- `partition`
  含义：每个顶点所属的 partition 编号
  长度：应等于 `hypergraph.num_vertices`
  格式：`partition[v] = p`

输出：

- 返回 `std::vector<int>`
- 每个元素是一个超边编号 `edge`
- 含义：该超边在当前 partition 中是 cut edge

判定规则：

- 取该超边第一个 pin 的 partition 作为 `base_part`
- 如果该超边中存在任意 pin 的 partition 与 `base_part` 不同，则该超边是 cut edge

### `contract_hypergraph`

声明位置：[include/kspecpart/overlay.hpp](/home/norising/K_SpecPart_C/include/kspecpart/overlay.hpp)

定义位置：[src/overlay.cpp](/home/norising/K_SpecPart_C/src/overlay.cpp)

函数签名：

```cpp
Hypergraph contract_hypergraph(const Hypergraph& hypergraph,
                               const std::vector<int>& clusters);
```

作用：

- 按 `clusters` 提供的顶点聚类结果，把原超图收缩成 coarse hypergraph

输入：

- `hypergraph`
  含义：待收缩的原始超图
- `clusters`
  类型：`std::vector<int>`
  长度：应等于 `hypergraph.num_vertices`
  含义：`clusters[v]` 表示原顶点 `v` 映射到哪个 coarse 顶点

输出：

- 返回一个新的 `Hypergraph`
- 顶点数变为 cluster 数
- 超边会按 cluster 重新映射并合并

### `overlay_partitions`

声明位置：[include/kspecpart/overlay.hpp](/home/norising/K_SpecPart_C/include/kspecpart/overlay.hpp)

定义位置：[src/overlay.cpp](/home/norising/K_SpecPart_C/src/overlay.cpp)

函数签名：

```cpp
OverlayResult overlay_partitions(const std::vector<std::vector<int>>& partitions,
                                 const Hypergraph& hypergraph);
```

作用：

- 输入多组 partition
- 收集这些 partition 中所有出现过的 cut edge
- 把这些 cut edge 从超图连通关系中“排除”
- 然后调用 `island_removal(...)` 找到剩下的连通块
- 把每个连通块视为一个 cluster
- 最后把原图收缩成 coarse hypergraph

输入：

- `partitions`
  类型：`std::vector<std::vector<int>>`
  含义：多组候选 partition
  其中 `partitions[i][v]` 表示第 `i` 组划分中顶点 `v` 的 partition id
- `hypergraph`
  含义：原始超图

输出：

- 返回 `OverlayResult`
- `result.clusters`
  原图顶点到 coarse cluster 的映射
- `result.hypergraph`
  收缩后的 coarse hypergraph

## `contract_hypergraph` 的详细行为

这个函数做了三件事。

### 1. 收缩顶点权重

代码逻辑：

```cpp
vwts[cluster] += hypergraph.vwts[vertex];
```

含义：

- 同一 cluster 中所有原顶点的顶点权重相加
- 形成 coarse 顶点的 `vwts`

### 2. 收缩 fixed 约束

代码逻辑：

- 如果 cluster 中某个原顶点是 fixed
- 那么 coarse 顶点也继承该 fixed 值

当前实现行为：

- 如果同一 cluster 中出现多个 fixed 值不一致，当前代码会直接覆盖成最后一次看到的值
- 这是一种过渡实现，后续如需严格对齐 Julia 版，可能还需要更细致的冲突处理

### 3. 收缩超边

步骤如下：

1. 遍历原图每条超边
2. 把超边中的所有 pin `v` 映射成 `clusters[v]`
3. 对映射后的 cluster id 去重
4. 如果去重后只剩 1 个 cluster，说明该超边已完全内部化，丢弃
5. 否则保留该 coarse hyperedge
6. 如果不同原超边收缩后得到相同的 cluster 集合，则把它们的 `hwts` 累加

这一步使用了：

```cpp
std::unordered_map<std::vector<int>, int, VectorHasher> edge_weights;
```

其含义是：

- key：一个 coarse hyperedge 的 pin 列表
- value：该 coarse hyperedge 的累计权重

## `overlay_partitions` 的详细行为

### 第一步：求所有 partition 的 cut edge 并集

代码逻辑：

```cpp
std::unordered_set<int> union_cut;
for (const auto& partition : partitions) {
    for (int edge : cut_edges_for_partition(hypergraph, partition)) {
        union_cut.insert(edge);
    }
}
```

含义：

- 只要某条超边在任意一组 partition 中被切开，就把它记入 `union_cut`

### 第二步：把这些超边排除后，求剩余连通块

代码逻辑：

```cpp
const auto [clusters, sizes] = island_removal(hypergraph, union_cut);
```

含义：

- `union_cut` 中的超边不再参与连通
- 其余超边保持连通关系
- 得到的每个连通块就是一个 overlay cluster

### 第三步：按 cluster 收缩超图

代码逻辑：

```cpp
result.hypergraph = contract_hypergraph(hypergraph, clusters);
result.clusters = clusters;
```

含义：

- 返回 coarse hypergraph
- 同时保留原顶点到 coarse 顶点的映射，供后续 partition 回投影使用

## 输入输出数据格式总结

### `cut_edges_for_partition`

输入：

- `partition[v] = p`

输出：

- `cut_edges = {e0, e1, ...}`

### `contract_hypergraph`

输入：

- `clusters[v] = c`

输出：

- `Hypergraph`
  其中 coarse 顶点编号即 `c`

### `overlay_partitions`

输入：

- `partitions[i][v] = p`

输出：

- `OverlayResult.hypergraph`
  coarse hypergraph
- `OverlayResult.clusters[v] = c`
  原顶点到 coarse 顶点映射

## 这个模块在整体流程中的位置

当前 C++ 管线里，overlay 主要用于：

1. 先生成多组候选 partition
2. 取其中较优的一部分
3. 调 `overlay_partitions(...)`
4. 得到 coarse hypergraph
5. 在 coarse hypergraph 上再做一轮分区
6. 用 `clusters` 把 coarse partition 投回原图

因此 overlay 的主要目的是：

- 找到不同候选结果中“相对稳定、不容易被切开”的内部区域
- 把这些区域收缩成 coarse 顶点
- 缩小后续分区问题规模

## 当前实现与 Julia 版的关系

对应 Julia 文件：

- [overlay.jl](/home/norising/K_SpecPart/overlay.jl)

当前一致点：

- 都基于多组 partition 的 cut edge 信息做 cluster
- 都会形成 coarse hypergraph

当前差异：

- 当前 C++ 版实现的是基础 overlay 版本
- 后续如要完全对齐 Julia 版，还需要继续核对 cluster 构造和 coarse partition 细节

## 适合汇报时的简述

可以把 `overlay.cpp` 概括成：

- 输入多组候选 partition
- 收集所有被切开的超边
- 把这些超边看作 cluster 边界
- 在剩余超边诱导的连通关系上形成 cluster
- 再把超图收缩成 coarse hypergraph，供下一轮更小规模分区使用
