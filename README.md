# K-SpecPart C++

## 当前状态

当前项目已经完成基础模块拆分，并进一步把 `tree_partition` 提升成独立的高层划分模块入口。

当前主要模块：

- `src/io.cpp`：`.hgr` / fixed / hint / partition 文件读写
- `src/hypergraph.cpp`：超图基础构造与单 pin 超边移除
- `src/golden_evaluator.cpp`：cutsize 与 balance 评估
- `src/isolate_islands.cpp`：连通块识别与主块提取
- `src/graphification.cpp`：超图转普通加权图
- `src/embedding.cpp`：当前过渡版谱嵌入
- `src/overlay.cpp`：overlay 与 coarse hypergraph contraction
- `src/tree_partition.cpp`：当前 tree partition 主入口、树候选生成、局部平衡修复
- `src/specpart.cpp`：高层 two-way / k-way refine 流程编排与结果回投影

## 本轮完成内容

本轮已经完成下面这一步结构对齐：

- 新增 `TreePartitionOptions` 与 `tree_partition(...)` / `tree_partition_best(...)` 接口
- `specpart.cpp` 不再直接管理底层候选划分细节，而是统一通过 `tree_partition` 模块获取候选分区
- `tree_partition.cpp` 中新增了更接近 Julia `tree_partition.jl` 模块边界的实现骨架，包括：
  - 图重加权入口 `reweight_graph(...)`
  - 树构造入口 `construct_tree(...)`
  - two-way tree sweep 候选生成
  - k-way tree edge cut 贪心候选生成
  - 谱 sweep / balanced k-means / local refine 的统一候选汇总

## 当前 `tree_partition.cpp` 实现了什么

当前 `tree_partition.cpp` 已经不仅是“几个辅助函数”，而是当前 C++ 版的实际划分模块入口。

它目前负责：

- `compute_balance_limits(...)`
- `balance_penalty(...)`
- `local_refine_partition(...)`
- `tree_partition(...)`
- `tree_partition_best(...)`
- `partition_two_way_hypergraph(...)`
- `partition_k_way_hypergraph(...)`
- 谱嵌入列组合枚举
- 基于重加权图的树候选构造
- two-way / k-way 候选分区打分与筛选

当前候选来源包括：

- tree-based 候选
  - 基于谱坐标的 path-tree proxy
  - 基于重加权图的 MST-tree proxy
- 原有过渡版候选
  - one-dimensional sweep
  - balanced k-means
  - base partition

## 与 Julia 原版的一致性说明

当前版本比之前更接近 `tree_partition.jl` 的模块边界，但仍然不是 Julia 原版等价实现。

已经对齐的部分：

- `specpart.cpp -> tree_partition.cpp` 的调用关系更接近 Julia `specpart.jl -> tree_partition.jl`
- 已经引入“重加权图 -> 构树 -> 基于树生成候选分区”的处理思路
- two-way 与 k-way 仍然分开处理

仍未完全对齐的部分：

- 还没有实现 Julia 的 `cut_distillation.jl`
- 还没有实现 Julia 的 `METIS_tree_partition(...)`
- 当前 `construct_tree(...)` 是 proxy 版本，不是 Julia 里的 LSST / degree-aware Prim 等完整实现
- 当前 k-way tree candidate 是贪心删边版本，不是 Julia 原版递归切树流程
- 还没有接入 hMETIS / METIS / TritonPart / ILP 外部求解链路

## 已验证

```bash
cmake -S . -B build
cmake --build build -j4
./build/K_SpecPart --hypergraph /home/norising/K_SpecPart/test.graph --num-parts 2 --seed 1 --output /tmp/tree_split_two.part
./build/K_SpecPart --hypergraph /home/norising/K_SpecPart/test.graph --num-parts 3 --seed 1 --output /tmp/tree_split_three.part
```

当前验证结果：

- `num-parts=2`：`cutsize=7`，`balance=[524, 493]`
- `num-parts=3`：`cutsize=6`，`balance=[334, 348, 335]`

## 下一步建议

为了继续向 Julia 版靠拢，接下来建议按这个顺序推进：

1. `cut_distillation.cpp`
2. 更贴近 Julia 的 `tree_partition.cpp` 内部实现
3. `rmq/*`
4. `projection.cpp`
5. 外部 partitioner/refiner 接口
