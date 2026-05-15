# K_SpecPart_C 环境配置与运行说明

本文档重点说明基础编译、可选外部工具配置以及实验脚本的使用方式。当前仓库可以单独编译 C++ 主程序，但如果要完整复现实验环境，还需要额外准备 Julia 参考实现、hMETIS、KaHyPar、ILP 分区器、Triton/OpenROAD 等外部组件。

需要先说明的一点是：本仓库中仍保留了一些开发机绝对路径，例如 `/home/norising/...`。因此，在新设备上部署时，建议优先采用“显式传参”或“修改脚本中的候选路径”两种方式，而不要完全依赖自动发现。

## 1. 推荐运行环境

- 操作系统：64 位 Linux，优先推荐 Ubuntu 22.04/24.04，WSL2 也可以
- 编译器：支持 C++17 的 `g++` 或 `clang++`
- CMake：3.16 及以上
- Python：3.10 及以上
- Julia：1.11.x（仅在需要运行 Julia 参考实现时）
- 硬件建议：大图和大规模 benchmark 会占用较多内存，建议至少 16 GB 内存

## 2. 基础系统依赖

如果只想先把 C++ 主程序编译并跑通，可以先安装下面这组依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  pkg-config \
  git \
  python3 \
  python3-pip \
  libeigen3-dev \
  libssl-dev \
  libopenblas-dev \
  liblapack-dev
```

其中，本项目的 `CMakeLists.txt` 直接依赖以下库：

- Eigen3
- OpenSSL
- Threads
- BLAS
- LAPACK

如果后续准备使用老版本 hMETIS 二进制，通常还需要 32 位运行时支持：

```bash
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y libc6:i386 libstdc++6:i386
```

## 3. 获取代码

先获取 C++ 仓库本身：

```bash
git clone <your-repo-url> K_SpecPart_C
cd K_SpecPart_C
```

如果你还需要运行 Julia 参考实现，请另外准备原始 Julia 代码仓库。当前 `scripts/run_julia_specpart.jl` 中写死了：

```julia
const JULIA_SOURCE_ROOT = "/home/norising/K_SpecPart"
```

这意味着新设备上有两种处理方式：

1. 将 Julia 原始仓库放到完全相同的路径。
2. 手动修改 [scripts/run_julia_specpart.jl](/home/norising/K_SpecPart_C/scripts/run_julia_specpart.jl) 中的 `JULIA_SOURCE_ROOT`。

如果只使用 C++ 主程序而不运行 Julia 对照，这一步可以跳过。

## 4. 编译 C++ 主程序

推荐使用 Release 模式构建：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

编译成功后，主程序位于：

```bash
build/K_SpecPart
```

可以先检查帮助信息：

```bash
./build/K_SpecPart --help
```

如果 `cmake` 报找不到 Eigen3、BLAS 或 LAPACK，优先检查上一节依赖是否安装完整；如果这些库安装在非系统默认目录，也可以通过 `CMAKE_PREFIX_PATH` 补充路径。

## 5. 最小可运行模式

C++ 主程序可以先绕过外部优化器和 Triton 精化：

```bash
./build/K_SpecPart \
  --hypergraph /path/to/your.graph \
  --num-parts 2 \
  --imb 2 \
  --output partition.part \
  --disable-optimal-partitioner \
  --disable-triton-refiner
```

这里的 `/path/to/your.graph` 需要替换成实际超图文件路径。本仓库本身不附带 `processed_graph` 数据集，因此图数据需要你自行准备，或者来自配套的 Julia 仓库 / 实验数据目录。

## 6. 可选外部工具配置

### 6.1 hMETIS

hMETIS 主要用于初始 hint 和 overlay 阶段的外部分区。当前仓库中与 hMETIS 相关的默认入口有两处：

- [scripts/hmetis_wrapper.sh](/home/norising/K_SpecPart_C/scripts/hmetis_wrapper.sh)
- C++ 主程序的 `--hmetis <path>` 参数

需要注意的是，`hmetis_wrapper.sh` 目前只会在少量固定路径中查找二进制，例如：

- `/home/norising/hmetis-1.5-linux/hmetis`
- `/home/fetzfs_projects/SpecPart/K_SpecPart/hmetis`

因此，新设备上建议采用以下任一种方式：

1. 直接运行 C++ 主程序时，显式传入 `--hmetis /path/to/hmetis`。
2. 运行 benchmark 时，修改 [scripts/hmetis_wrapper.sh](/home/norising/K_SpecPart_C/scripts/hmetis_wrapper.sh) 中的候选路径。
3. 自己写一个新的 wrapper，并在 benchmark 中通过 `--hmetis-wrapper /path/to/your_wrapper.sh` 指定。

如果你使用的是老的 32 位 hMETIS，可执行文件即使没有执行权限，wrapper 也会尝试通过 `/lib/ld-linux.so.2` 等 32 位 loader 启动；若系统中没有该 loader，就需要安装前面提到的 i386 运行时。

### 6.2 Julia 参考环境

Julia 参考环境定义在 [julia_ref_env](/home/norising/K_SpecPart_C/julia_ref_env) 中。当前 `Manifest.toml` 记录的 Julia 版本为 1.11.1，因此建议直接安装 Julia 1.11.x。

第一次使用前可以执行：

```bash
julia --project=./julia_ref_env -e 'using Pkg; Pkg.instantiate()'
```

随后可通过下面的脚本运行 Julia 参考实现：

```bash
./scripts/run_julia_specpart.sh --hypergraph /path/to/your.graph --num-parts 2 --imb 2 --output julia.part
```

如果 Julia 参考环境中的外部工具不在默认位置，可使用环境变量覆盖：

```bash
export K_SPECPART_JULIA_PROJECT=/home/yourname/K_SpecPart_C/julia_ref_env
export K_SPECPART_HMETIS_PATH=/path/to/hmetis_or_wrapper
export K_SPECPART_ILP_PATH=/path/to/ilp_wrapper_or_binary
export K_SPECPART_TRITON_PATH=/path/to/openroad
```

这里还要特别注意一点：Julia 运行脚本本身依赖原始 Julia 仓库中的 `specpart.jl`，因此仅仅装好 Julia 和 `julia_ref_env` 还不够，仍需保证上一节提到的 `JULIA_SOURCE_ROOT` 问题已经处理好。

### 6.3 ILP 分区器

本仓库默认通过 [scripts/julia_ilp_wrapper.sh](/home/norising/K_SpecPart_C/scripts/julia_ilp_wrapper.sh) 调用 ILP 分区器。这个 wrapper 会优先查找：

- `/home/norising/K_SpecPart_C/.external_build/ilp_part/ilp_part`
- `/home/norising/K_SpecPart/ilp_partitioner/build/ilp_part`

ILP 可执行文件并不由本仓库直接生成，而是来自外部 `ilp_partitioner` 工程，且通常依赖：

- OR-Tools
- CPLEX

如果新设备上暂时不准备这套 ILP 环境，有三种处理方式：

1. 对 C++ 主程序使用 `--disable-ilp`。
2. 对 benchmark 保留默认设置，让脚本记录该工具不可用或回退行为。
3. 自行准备外部 ILP 工程，并把 `ilp_part` 放到上面两个默认路径之一，或修改 wrapper 路径，建议参考原Julia仓库给出的说明获取该外部工具和hMETIS。

### 6.4 KaHyPar

KaHyPar 已放在当前项目子目录 [third_party/KaHyPar](/home/norising/K_SpecPart_C/third_party/KaHyPar) 下，通常可以单独编译。推荐命令如下：

```bash
cmake -S third_party/KaHyPar -B third_party/KaHyPar/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKAHYPAR_USE_MINIMAL_BOOST=ON
cmake --build third_party/KaHyPar/build -j"$(nproc)" --target KaHyPar
```

编译完成后，benchmark 脚本默认查找：

```bash
third_party/KaHyPar/build/kahypar/application/KaHyPar
```

并默认使用：

```bash
third_party/KaHyPar/config/cut_kKaHyPar_sea20.ini
```

如果你不使用 `KAHYPAR_USE_MINIMAL_BOOST=ON`，则需要系统中提供 Boost.Program_options。

### 6.5 Triton / OpenROAD

Triton/OpenROAD 精化属于可选后处理步骤。C++ 代码默认会尝试查找：

- `/home/norising/TritonPart_OpenROAD/build/src/openroad`
- `/home/norising/OpenROAD/build/src/openroad`

如果找不到，C++ 主程序会打印提示，并保持当前分区结果不变。新设备上有两种常见做法：

1. 显式关闭：

```bash
--disable-triton-refiner
```

2. 显式指定路径：

```bash
--triton-refiner /path/to/openroad
```

Julia 侧则可以通过 `K_SPECPART_TRITON_PATH` 覆盖。

### 6.6 gpmetis

`gpmetis` 只是树划分阶段的一个辅助候选生成器，不是核心必需组件。若系统中没有 `gpmetis`，程序仍可使用内部候选策略继续运行。

若你已经安装了 `gpmetis`，可以通过：

```bash
--gpmetis /path/to/gpmetis
```

显式指定路径。

## 7. benchmark 脚本使用方法

批量实验入口为 [scripts/benchmark_partitioners.py](/home/norising/K_SpecPart_C/scripts/benchmark_partitioners.py)。它可以同时比较：

- C++ 版本
- Julia 版本
- hMETIS
- KaHyPar

基础示例如下：

```bash
python3 scripts/benchmark_partitioners.py \
  --hypergraphs /path/to/1.graph /path/to/2.graph \
  --num-parts-list 2,3,4 \
  --imb-list 2,5,10 \
  --repeats 3 \
  --repeat-seed-mode fixed \
  --timeout-seconds 1800 \
  --out-dir $HOME/kspecpart_runs/bench_v1
```

建议注意以下几点：

- `--timeout-seconds 1800` 表示每个算法单次运行最多 1800 秒，不是整批实验总时长。
- 长时间正式实验不要把 `--out-dir` 放在 `/tmp` 下。系统重启后，`/tmp` 中的结果很可能被清空。
- 如果某些工具尚未部署完整，可以先使用 `--skip-julia`、`--skip-hmetis`、`--skip-kahypar` 缩小实验范围。
- 如果要严格对齐 C++ 与 Julia 的输入 hint，可加 `--shared-initial-hint`。
- benchmark 脚本会把 `timeout`、非零退出、启动失败和内部异常都记录到对应 case 目录下的日志与 `summary` 中，正常情况下不会因为单个算法失败而整批中断。

例如，只跑 C++ 与 KaHyPar：

```bash
python3 scripts/benchmark_partitioners.py \
  --hypergraphs /path/to/1.graph \
  --num-parts-list 2,3 \
  --imb-list 2,5 \
  --repeats 3 \
  --skip-julia \
  --skip-hmetis \
  --out-dir $HOME/kspecpart_runs/cpp_vs_kahypar
```

## 8. 一次较完整的推荐配置顺序

在新设备上，比较稳妥的配置顺序通常是：

1. 安装基础编译依赖，先编出 `build/K_SpecPart`。
2. 用 `--disable-optimal-partitioner --disable-triton-refiner` 跑通一个最小样例。
3. 配置 hMETIS，并验证 C++ 主程序能正常识别它。
4. 配置 Julia 1.11.x 与 `julia_ref_env`，再处理 `JULIA_SOURCE_ROOT`。
5. 编译 KaHyPar，并验证 benchmark 能识别它。
6. 如确有需要，再补 ILP 分区器与 Triton/OpenROAD。
7. 正式大规模实验时，把输出目录放到持久路径，而不是 `/tmp`。

## 9. 常见问题

### 9.1 `cmake` 提示找不到 Eigen3 / BLAS / LAPACK

先检查第二节中的系统依赖是否已经安装；若库在自定义目录中，可以补充 `CMAKE_PREFIX_PATH`。

### 9.2 `run_julia_specpart.sh` 能启动，但 Julia 内部报找不到 `specpart.jl`

这通常不是 Julia 包环境的问题，而是 [scripts/run_julia_specpart.jl](/home/norising/K_SpecPart_C/scripts/run_julia_specpart.jl) 中 `JULIA_SOURCE_ROOT` 与你机器上的实际 Julia 仓库路径不一致。

### 9.3 hMETIS wrapper 报 binary not found

说明 [scripts/hmetis_wrapper.sh](/home/norising/K_SpecPart_C/scripts/hmetis_wrapper.sh) 中的候选路径没有命中。可以直接改脚本、传 `--hmetis`，或者为 benchmark 单独指定新的 wrapper。

### 9.4 外部工具没有全部装好，程序还能不能先跑

可以。对 C++ 主程序而言，最稳妥的做法是先关闭：

```bash
--disable-optimal-partitioner --disable-triton-refiner
```

等基础链路确认无误后，再逐个接入 hMETIS、KaHyPar、Julia、ILP 和 Triton。

## 10. 绝对路径约束

检查并按需修改以下文件中的绝对路径：

- [scripts/hmetis_wrapper.sh](/home/norising/K_SpecPart_C/scripts/hmetis_wrapper.sh)
- [scripts/julia_ilp_wrapper.sh](/home/norising/K_SpecPart_C/scripts/julia_ilp_wrapper.sh)
- [scripts/run_julia_specpart.sh](/home/norising/K_SpecPart_C/scripts/run_julia_specpart.sh)
- [scripts/run_julia_specpart.jl](/home/norising/K_SpecPart_C/scripts/run_julia_specpart.jl)
- [src/optimal_partitioner.cpp](/home/norising/K_SpecPart_C/src/optimal_partitioner.cpp)
- [src/triton_refiner.cpp](/home/norising/K_SpecPart_C/src/triton_refiner.cpp)
- [src/metis.cpp](/home/norising/K_SpecPart_C/src/metis.cpp)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
