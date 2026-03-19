# `isolate_islands.cpp` 说明

本文档对应实现文件 [src/isolate_islands.cpp](/home/norising/K_SpecPart_C/src/isolate_islands.cpp) 和声明文件 [include/kspecpart/isolate_islands.hpp](/home/norising/K_SpecPart_C/include/kspecpart/isolate_islands.hpp)。

## 模块职责

这个模块负责识别超图中的连通块，并在需要时只保留最大的连通块，把其它“孤岛”从后续主流程中剥离出来。

它对应 Julia 版的：

- `isolate_islands.jl`

## 为什么需要它

很多超图并不是单个大连通块，而是由多个互不连通的部分组成。

当前 C++ 管线的策略是：

- 先找出所有连通块
- 选出最大的那个连通块作为主处理对象
- 只在这个主块上继续做谱划分
- 其它小连通块后面再根据已有 partition 结果补回去

这样做的好处：

- 降低问题规模
- 避免小孤岛影响主图的谱结构

## 相关数据结构

### `Hypergraph`

定义位置：[include/kspecpart/definitions.hpp](/home/norising/K_SpecPart_C/include/kspecpart/definitions.hpp)

重要字段：

- `num_vertices`
  顶点数
- `num_hyperedges`
  超边数
- `eptr`
  超边偏移数组
- `eind`
  超边 pin 顶点数组
- `fixed`
  fixed 顶点信息
- `vwts`
  顶点权重
- `hwts`
  超边权重

索引约定：

- 全部为 0-based

### `IsolateResult`

定义位置：[include/kspecpart/isolate_islands.hpp](/home/norising/K_SpecPart_C/include/kspecpart/isolate_islands.hpp)

字段含义：

- `hypergraph`
  含义：只包含最大连通块的新超图
- `original_indices`
  含义：新图顶点编号到原图顶点编号的映射
  解释：`original_indices[new_v] = old_v`
- `new_indices`
  含义：原图顶点编号到新图顶点编号的映射
  解释：`new_indices[old_v] = new_v`
  如果原顶点不在主块中，则值为 `-1`
- `component_labels`
  含义：原图每个顶点属于哪个连通块
  解释：`component_labels[v] = component_id`
- `component_sizes`
  含义：每个连通块的顶点数
- `main_component`
  含义：最大连通块的编号

## 内部辅助结构

### `UnionFind`

定义位置：[src/isolate_islands.cpp](/home/norising/K_SpecPart_C/src/isolate_islands.cpp)

作用：

- 用并查集维护顶点连通关系

字段：

- `parent`
  并查集父节点数组
- `size`
  每个集合的大小

接口：

- `find(x)`
  查找代表元
- `unite(a, b)`
  合并两个集合

## 函数说明

### `island_removal`

声明位置：[include/kspecpart/isolate_islands.hpp](/home/norising/K_SpecPart_C/include/kspecpart/isolate_islands.hpp)

定义位置：[src/isolate_islands.cpp](/home/norising/K_SpecPart_C/src/isolate_islands.cpp)

函数签名：

```cpp
std::pair<std::vector<int>, std::vector<int>> island_removal(
    const Hypergraph& hypergraph,
    const std::unordered_set<int>& excluded_hyperedges);
```

作用：

- 在“排除某些超边”的条件下，计算超图的连通块

输入：

- `hypergraph`
  含义：原始超图
- `excluded_hyperedges`
  类型：`std::unordered_set<int>`
  含义：这些超边不参与连通性分析

输出：

- 返回一个 `pair`

第一个返回值：

- `clusters`
  类型：`std::vector<int>`
  长度：`hypergraph.num_vertices`
  含义：`clusters[v] = component_id`

第二个返回值：

- `sizes`
  类型：`std::vector<int>`
  长度：连通块个数
  含义：`sizes[c] = 第 c 个连通块的顶点数`

### `isolate_islands`

声明位置：[include/kspecpart/isolate_islands.hpp](/home/norising/K_SpecPart_C/include/kspecpart/isolate_islands.hpp)

定义位置：[src/isolate_islands.cpp](/home/norising/K_SpecPart_C/src/isolate_islands.cpp)

函数签名：

```cpp
IsolateResult isolate_islands(const Hypergraph& hypergraph);
```

作用：

- 找到超图中的最大连通块
- 构造只包含该连通块的新超图
- 记录新旧图顶点编号映射

输入：

- `hypergraph`
  含义：原始超图

输出：

- 返回 `IsolateResult`

## `island_removal` 的详细行为

### 第一步：初始化并查集

代码逻辑：

```cpp
UnionFind uf(hypergraph.num_vertices);
```

含义：

- 开始时每个顶点单独是一个集合

### 第二步：遍历所有未排除的超边，把同一超边内的 pin 合并

代码逻辑：

```cpp
for (int edge = 0; edge < hypergraph.num_hyperedges; ++edge) {
    if (excluded_hyperedges.count(edge) > 0) {
        continue;
    }
    ...
    for (int idx = start + 1; idx < end; ++idx) {
        uf.unite(hypergraph.eind[start], hypergraph.eind[idx]);
    }
}
```

含义：

- 对每条超边，取第一个 pin 作为代表
- 把该超边中其它 pin 都与它合并

效果：

- 同一条超边中的所有顶点最终会处于同一并查集集合

### 第三步：重新编号连通块

代码逻辑：

```cpp
std::unordered_map<int, int> remap;
...
clusters[vertex] = it->second;
sizes[it->second] += 1;
```

含义：

- 并查集的根节点编号不一定连续
- 所以函数会把这些根重新映射成连续的 `0, 1, 2, ...`

输出格式：

- `clusters[v]` 一定是从 `0` 开始的连续 component id

## `isolate_islands` 的详细行为

### 第一步：先求全部连通块

代码逻辑：

```cpp
const auto [clusters, cluster_sizes] = island_removal(hypergraph, {});
```

含义：

- 这里没有排除任何超边
- 即按整个超图的自然连通性求连通块

### 第二步：找最大的连通块

代码逻辑：

```cpp
const int main_component = ... max_element(cluster_sizes.begin(), cluster_sizes.end())
```

含义：

- 顶点数最多的那个连通块被视为主块

### 第三步：建立原图和新图顶点编号映射

当前实现同时建立两种映射。

#### `original_indices`

含义：

- 新图顶点编号 -> 原图顶点编号

格式：

```text
original_indices[new_v] = old_v
```

#### `new_indices`

含义：

- 原图顶点编号 -> 新图顶点编号

格式：

```text
new_indices[old_v] = new_v
```

如果原顶点不在主块中：

```text
new_indices[old_v] = -1
```

### 第四步：抽取主块中的顶点属性

被保留到新图中的内容：

- 顶点权重 `vwts`
- fixed 信息 `fixed`

做法：

- 只保留属于 `main_component` 的原顶点
- 按新编号顺序写入 `vwts_processed` 和 `fixed_processed`

### 第五步：抽取主块中的超边

代码行为：

- 遍历原图每条超边
- 只保留其 pin 中位于主块的部分
- 把这些原顶点编号映射成新图顶点编号

当前实现的具体规则是：

- 如果一条超边在主块里没有任何 pin，则丢弃
- 如果一条超边在主块中有 pin，则把这些 pin 保留下来，形成新图中的对应超边

这意味着：

- 新图超边数通常小于等于原图超边数
- 新图只保留主块相关的超边内容

### 第六步：构造新超图

最后调用：

```cpp
build_hypergraph(...)
```

输出一个新的 `Hypergraph`

该新图满足：

- 顶点只包含主块中的顶点
- 超边只包含主块相关部分
- 顶点编号从 `0` 连续开始

## 输入输出数据格式总结

### `island_removal`

输入：

- `excluded_hyperedges` 是一个超边编号集合

输出：

- `clusters[v] = c`
- `sizes[c] = component size`

### `isolate_islands`

输出 `IsolateResult`：

- `hypergraph`
  主块对应的新超图
- `original_indices[new_v] = old_v`
- `new_indices[old_v] = new_v or -1`
- `component_labels[old_v] = component_id`
- `component_sizes[component_id] = size`
- `main_component = 最大连通块编号`

## 这个模块在整体流程中的位置

当前 C++ 管线里，该模块通常在以下位置调用：

1. 读入原始超图
2. 调 `isolate_islands(...)`
3. 后续谱划分只对 `IsolateResult.hypergraph` 进行
4. 最终再通过 `new_indices / original_indices / component_labels` 把结果补回原图

## 当前实现与 Julia 版的关系

对应 Julia 文件：

- [isolate_islands.jl](/home/norising/K_SpecPart/isolate_islands.jl)

当前一致点：

- 都先识别连通块
- 都保留最大的连通块供主流程使用
- 都维护新旧索引映射

当前需要继续核对的点：

- 某些超边保留/截断细节是否与 Julia 实现完全一致
- 小孤岛回填策略是否与 Julia 版完全一致

## 适合汇报时的简述

可以把 `isolate_islands.cpp` 概括成：

- 用并查集找出超图连通块
- 选最大连通块作为主处理对象
- 生成只包含该主块的新超图
- 同时保存原图和新图之间的顶点编号映射
- 为后续主流程降规模，并为最后把 partition 映射回原图提供索引基础
