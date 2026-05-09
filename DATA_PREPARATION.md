# 训练数据准备规格说明

> **适用范围**：本文档面向准备使用其他数值仿真软件（ASPECT、Underworld、StagYY、COMSOL、自研代码等）的输出来训练本仓库代理模型的研究团队。它详细说明数据的目录布局、文件格式、物理约定、参数空间设计，以及从主流求解器到本代码所需 `.pt` 文件的转换流程。
>
> **配套工具**：根目录的 [`validate_sim_data.py`](./validate_sim_data.py) 是与本规格配套的自动校验脚本。

---

## 目录

- [0. 总览与硬性前提](#0-总览与硬性前提)
- [1. 目录结构](#1-目录结构)
- [2. 文件格式逐个详解](#2-文件格式逐个详解)
  - [2.1 `sims.pt`](#21-simspt--全局元数据)
  - [2.2 `xc.pt`, `yc.pt`](#22-xcpt-ycpt--空间网格坐标每条仿真一份)
  - [2.3 `times.pt`](#23-timespt--时间戳数组每条仿真一份)
  - [2.4 `e1_Tprev_data_select_snaps.pt`](#24-e1_tprev_data_select_snapspt--温度场快照)
  - [2.5 `e1_uprev_data_select_snaps.pt`, `e1_vprev_data_select_snaps.pt`](#25-e1_uprev_data_select_snapspt-e1_vprev_data_select_snapspt--速度场)
- [3. 数据量建议](#3-数据量建议)
- [4. 参数空间设计](#4-参数空间设计撒-50-个点的具体方法)
- [5. 工作流：从仿真求解器到 `.pt`](#5-工作流从仿真求解器到-pt)
  - [5.1 通用模板](#51-通用模板)
  - [5.2 各主流求解器的具体读取代码](#52-各主流求解器的具体读取代码)
  - [5.3 完整批量转换 + 校验脚本](#53-完整批量转换--校验脚本)
- [6. 校验流程](#6-校验流程)
- [7. 常见坑 - cheat sheet](#7-常见坑---cheat-sheet)
- [8. 一个完整的 minimal 工作示例](#8-一个完整的-minimal-工作示例)
- [9. 后续工作](#9-后续工作)

---

## 0. 总览与硬性前提

这套代码训练的是一个 **输入温度场 → 输出速度场** 的代理网络，本质是用神经网络替代 Stokes 方程求解器。**所有"硬契约"都建立在和作者的物理设定一致的前提下**。

如果你打算复用作者的预训练权重或 `scaler.py` 里的 scaling law，下表的物理设定**必须严格匹配**；如果你打算从零训练，可以放宽，但注意要同步改 `scaler.py` 和 `eta_torch`。

| 物理项 | 必须值（用作者权重时） | 改了之后的影响 |
|---|---|---|
| 维度 | 2D 笛卡尔 | 改 3D = 重写整个网络架构 |
| 域几何 | `[0, 4] × [0, 1]`（AR=4） | 改 AR = 改 `xc` 边界值 + 网络感受野要重训 |
| 网格分辨率 | **128 × 506**（y × x）| 改了 = 网络输入尺寸全变 = 必须重训 |
| 加热模式 | **纯内热**（RaQ 仅）| 改底部加热 = `scaler.py` 要重新拟合 |
| 流变 | Frank-Kamenetskii `η(T, z)` | 改流变 = 黏度计算公式 (`eta_torch`) 要重写 |
| 近似 | Boussinesq | 改可压缩 = 物理损失要换 |
| 边界 | 上下 free-slip + 上下等温 + 左右反射 | 改 free surface = 需要 sticky air 等额外通道 |
| 标度 | 长度=D，时间=D²/κ，温度=ΔT | 改了所有量级都对不上 |
| 参数范围 | RaQ ∈ [0.13, 9.98], log₁₀FKt ∈ [6, 9.9], log₁₀FKp ∈ [0.005, 2] | 超出 = 训练分布外推 |

**判断准则**：如果你的研究问题在上面任何一项**显著偏离**，建议**完全重新训练**而不是微调。下面的格式契约对两种情况都适用，因为代码读数据的方式是一样的。

---

## 1. 目录结构

```
my_data_dir/                    ← 这就是 paths.py 里的 data_dir
├── sims.pt                     ← 全局元数据（必须）
├── train/
│   ├── sim_0/                  ← 一条仿真一个目录，子目录名固定 sim_<num>
│   │   ├── xc.pt
│   │   ├── yc.pt
│   │   ├── times.pt
│   │   ├── e1_Tprev_data_select_snaps.pt
│   │   ├── e1_uprev_data_select_snaps.pt
│   │   └── e1_vprev_data_select_snaps.pt
│   ├── sim_1/  ...
│   └── sim_<N>/
├── cv/
│   └── sim_<X>/   ...          ← 同样 6 个文件
└── test/
    └── sim_<Y>/   ...          ← 同样 6 个文件
```

**几个硬约束**：

- 子目录名 **必须** 是 `sim_<num>`，`<num>` 是 `sims.pt` 里的整数 ID。
- 子目录所在的父目录名 **必须** 是 `train` / `cv` / `test`，与 `sims.pt` 里第二个字段一致。
- 文件名前缀 `e1_` 来自 `take_every=1`（每步都采样）。代码里 `take_every` 硬编码 = 1，所以 **你只能用 `e1_`**。

---

## 2. 文件格式逐个详解

### 2.1 `sims.pt` — 全局元数据

```python
sims = [
    [num, dataset, raq, fkt, fkp, gr, ar, source_path],
    ...
]
```

| 字段 | 类型 | 含义 | 取值约束 |
|---|---|---|---|
| `num` | `int` | 全局唯一仿真 ID | 在整个 list 里不重复 |
| `dataset` | `str` | 数据划分 | `"train"` / `"cv"` / `"test"`（必须三选一）|
| `raq` | `float` | 内热瑞利数 RaQ | 用作者权重时 ∈ [0.13, 10] |
| `fkt` | `float` | FK 温度黏度对比 ηₜ = η(T=0)/η(T=1) | 用作者权重时 ∈ [10⁶, 10¹⁰] |
| `fkp` | `float` | FK 深度黏度对比 ηₚ = η(z=1)/η(z=0) | 用作者权重时 ∈ [1, 100] |
| `gr` | `int` | 垂直分辨率 (levels) | **必须 = 128** |
| `ar` | `int` | 水平/垂直比 | **必须 = 4** |
| `source_path` | `str` | 原始数据出处 | 自由字符串，代码不读 |

**保存方式**：

```python
import torch
sims = []
for i, (raq, fkt, fkp, ds) in enumerate(my_param_list):
    sims.append([i, ds, float(raq), float(fkt), float(fkp), 128, 4,
                 f"my_simulation_{i}"])
torch.save(sims, "my_data_dir/sims.pt")
```

**踩坑点**：

- 必须是 Python `list` of `list`，**不要存成 ndarray 或 tuple of tuple** —— 代码里用 `for si, sim in enumerate(sims)` 然后 `sim[1] == "train"`，要支持索引和字符串比较。
- `raq, fkt, fkp` 必须是 Python `float` 或 ndarray scalar，不能是 0-d tensor（后续 `np.log10(fkt)` 会出错）。
- `dataset` 字段大小写敏感，必须小写。
- 代码里有硬编码的 `ignore_inds = [8, 39]` 跳过，作者数据里的 sim 8 和 sim 39 被排除。**你的数据要么避免用这两个 ID，要么手动改 `datasetio.py` 把它清空**：

```python
# datasetio.py 多处出现
ignore_inds = []   # 原来是 [8, 39]
```

### 2.2 `xc.pt`, `yc.pt` — 空间网格坐标（每条仿真一份）

```python
xc.shape == (128, 506)    # [y_idx, x_idx]
xc.dtype == torch.float64
xc[i, j] = (i, j) 格点的 x 坐标，范围 [0, 4]

yc.shape == (128, 506)
yc.dtype == torch.float64
yc[i, j] = (i, j) 格点的 y 坐标，范围 [0, 1]
```

**方向约定**（极易踩坑）：

```
yc[0,    :]  ≈ 0    ← y_idx=0   是底
yc[127,  :]  ≈ 1    ← y_idx=127 是顶
xc[:,    0]  ≈ 0    ← x_idx=0   是左
xc[:,  505]  ≈ 4    ← x_idx=505 是右
```

代码加载后会**强制覆盖边界**：

```python
self.xc[:, :, 0]  = 0.0
self.xc[:, :, -1] = 4.0
self.yc[:, 0, :]  = 0.0
self.yc[:, -1, :] = 1.0
```

这意味着中间网格非均匀没事，代码只精确对齐 4 个边。

**生成方式**（均匀网格的占位 —— 你应该从仿真求解器导出真实坐标）：

```python
import torch
xc_1d = torch.linspace(0.0, 4.0, 506, dtype=torch.float64)
yc_1d = torch.linspace(0.0, 1.0, 128, dtype=torch.float64)
xc = xc_1d.unsqueeze(0).expand(128, 506).contiguous()
yc = yc_1d.unsqueeze(1).expand(128, 506).contiguous()
torch.save(xc, "xc.pt")
torch.save(yc, "yc.pt")
```

> **注意**：作者用的是**非均匀网格**（边界附近加密），用真实坐标准确度更高。如果你的求解器输出是均匀网格，用上面的占位代码就行；如果是非均匀网格，从求解器把每个 cell 中心坐标导出来。

### 2.3 `times.pt` — 时间戳数组（每条仿真一份）

```python
times.shape == (N_t,)             # 一维
times.dtype == torch.float64
times[i] = 第 i 个完整时间步的无量纲时间
```

| 属性 | 要求 |
|---|---|
| 单调性 | **严格递增** |
| 起点 | 一般是小正数，比如 `1e-7`（不能为 0，避免 `t**(-0.25)` 数值问题）|
| 单位 | 无量纲扩散时间（`t / (D²/κ)`）|
| 长度 | 等于完整仿真步数 `N_t`（典型 1000–6000）|
| 与 snaps 关系 | **`N_t ≥ N_snap`**；times 是完整序列时间，不是 snaps 子集时间 |

**踩坑点**：

- 即使你的仿真是固定步长，也要把每步的时间存下来（不只是 `dt`）。
- 单位必须是无量纲，不要存秒或年。如果你的求解器用 SI 单位输出 `t [秒]`，要除以 `D²/κ` 换算。
- 长度 `N_t` **不必**等于 snaps 帧数 `N_snap`，但 `N_t ≥ N_snap`。

### 2.4 `e1_Tprev_data_select_snaps.pt` — 温度场快照

```python
T.shape == (N_snap, 1, 128, 506)    # [batch, channel=1, y, x]
T.dtype == torch.float64
T 取值 ≈ [0, 1.05]                   # 无量纲，Boussinesq
```

| 物理项 | 约定 |
|---|---|
| 温标 | 上边界 T = 0，下边界 T = 1（无量纲化 `T_phys → (T_phys - T_top)/(T_bot - T_top)`）|
| 加热 | 内部 T ≈ 1.005 略高于 1 是 **纯内热的标志**（典型超热 0.5–2%）|
| 顶部边界层 | 厚度典型 5–10% 域厚度（表征对流强度）|
| 底部边界层 | 内热下非常薄，几乎贴着底面 |
| `T[:, 0, 0, :]` | 必须 ≈ 1（底）|
| `T[:, 0, -1, :]` | 必须 ≈ 0（顶）|

**示例**（实测 `sim_1` 第 0 帧垂直平均剖面）：

```
y_idx=  0  y=0.000   T=1.0000   ← 底界
y_idx=  4  y=0.031   T=1.0036   ← 底层很薄
y_idx= 16  y=0.126   T=1.0059   ← 已进入"近似等温"内部
y_idx= 64  y=0.504   T=1.0059   ← 中部
y_idx=112  y=0.882   T=0.9761   ← 顶层开始
y_idx=119  y=0.937   T=0.7455   ← 顶层中段
y_idx=123  y=0.969   T=0.4049   ← 顶层下段
y_idx=127  y=1.000   T=0.0000   ← 顶界
```

**踩坑点**：

- 如果你用的是**底部加热**（不是内热），中部 T ≈ 0.5，顶底边界层对称。这种数据**不能**用作者的预训练权重，`scaler.py` 的标度公式也对不上。
- 如果你输出的 T 是 SI 单位（K），先归一化：`T_nd = (T - T_top_K) / (T_bot_K - T_top_K)`。
- y 轴必须是底→顶递增；很多求解器（ASPECT、Underworld 默认输出）是顶→底，**忘了 flip 是最常见 bug**。

### 2.5 `e1_uprev_data_select_snaps.pt`, `e1_vprev_data_select_snaps.pt` — 速度场

```python
u.shape == (N_snap, 1, 128, 506)    # 同 T
u.dtype == torch.float64
u, v 都是有量纲值，量级 10²–10⁵（按你的参数）
```

| 物理项 | 约定 |
|---|---|
| `u` | 水平速度（沿 x 方向）|
| `v` | 垂直速度（沿 y 方向，**正向上**）|
| 单位 | **无量纲速度** = `u_phys × D / κ` |
| 范围 | 由 scaler 公式决定：`5 · exp(0.18·RaQ + 0.43·ln ηₜ − 0.46·ln ηₚ)` 量级 |
| 是否预 scale | **不要预先除 scaler**，dataloader 在 `__getitem__` 里现算 |
| 散度 | `∇·u ≈ 0`（不可压缩 Stokes）；实测 `|∇·u|/|u| < 1e-13` |
| 顶面 | `v[:, 0, -1, :] ≈ 0`（free-slip 顶界面）|
| 底面 | `v[:, 0, 0, :]` 在 cell-center 上**可以非零**（face-staggered 边界 v=0 不等于 cell-center v=0）|

**踩坑点**（最容易出问题的地方）：

1. **方向**：`u` 沿 x 方向，正向右；`v` 沿 y 方向，正向上。如果你的求解器把 v 定义为正向下（很多 geodynamics 代码因为深度递增的传统），**必须翻号**：`v = -v_solver`。

2. **Staggered grid → cell center**：
   - GAIA / ASPECT / StagYY 默认把速度存在 cell face 上（u 在 vertical face，v 在 horizontal face）。
   - 你的 `xc.pt`, `yc.pt` 是 cell **center** 坐标。
   - **必须把 face 速度插值到 cell center**，否则散度不为零，作者的 curl loss 监督就毁了。
   - 简单插值：`u_cell[i, j] = (u_face[i, j] + u_face[i, j+1]) / 2`，类似 `v`。

3. **速度量纲**：作者用无量纲 `u_nd = u_SI × D / κ`，其中 `D` = 域厚度（米），`κ` = 热扩散率（m²/s）。地幔典型 `D ≈ 2.9e6 m`，`κ ≈ 1e-6 m²/s`，所以 `D/κ ≈ 3e12 s/m`。物理 u ~ cm/年 = 3e-10 m/s → 无量纲 u ~ 1000，正好对上 `sim_1` 实测 ~3000。

4. **不要预先做任何归一化**：dataloader 里会自己除 scaler 做训练时归一化。你只要保证保存的是 `D/κ` 制下的纯量。

---

## 3. 数据量建议

| 项 | 作者 | 建议你最少 | 理想 |
|---|---|---|---|
| 仿真总数 | 130 | **50** | 200+ |
| Train / CV / Test | 96 / 16 / 18 | 35 / 7 / 8 | 140 / 30 / 30 |
| 每条帧数 | ~700 (子集) | ≥ 200 | 全部步数 (1000+) |
| 总训练样本 | ~50000 | ~7000 | ~100000+ |

> **少于 50 条仿真**：模型可以收敛但泛化弱，参数空间外推风险大。

**子集采样策略**：

```python
# 推荐：保留所有时间步
N_snap = N_t

# 或者：作者风格，前 200 全保留 + 后续随机采样
import numpy as np, random
if N_t > 200:
    rest = list(range(200, N_t))
    if N_t > 700:
        rest = random.choices(rest, k=min(500, N_t - 200))
    i_vec = list(range(N_t)) + rest
else:
    i_vec = list(range(N_t))
T_snaps = T_full[i_vec]   # [N_snap, 1, 128, 506]
```

> **建议**：直接保存所有时间步，磁盘多用一点没关系，训练时用 `max_examples_percent_per_epoch` 控制 epoch 内采样比例。

---

## 4. 参数空间设计（撒 50 个点的具体方法）

如果要复用作者标度律，参数范围保持：

```python
import numpy as np
from scipy.stats import qmc

n_sims = 50
sampler = qmc.LatinHypercube(d=3, seed=42)
X = sampler.random(n=n_sims)   # [n, 3] in [0, 1]^3

# 映射到物理范围
RaQ_min, RaQ_max         = 0.13, 9.98
log_FKt_min, log_FKt_max = 6.0, 9.9
log_FKp_min, log_FKp_max = 0.005, 2.0

raq = X[:, 0] * (RaQ_max - RaQ_min) + RaQ_min
fkt = 10 ** (X[:, 1] * (log_FKt_max - log_FKt_min) + log_FKt_min)
fkp = 10 ** (X[:, 2] * (log_FKp_max - log_FKp_min) + log_FKp_min)

# 划分 train/cv/test
labels = ["train"]*35 + ["cv"]*7 + ["test"]*8
np.random.RandomState(42).shuffle(labels)
```

> **注意**：Latin Hypercube 在每个维度做分层，比纯均匀采样在低 N 下覆盖更好。地幔对流的训练数据成本很高（每条仿真上千 CPU·小时），50 条 LHS 的效果接近 100+ 条均匀采样。

---

## 5. 工作流：从仿真求解器到 `.pt`

### 5.1 通用模板

不管你用哪个求解器，转换脚本结构如下：

```python
import os, torch, numpy as np

def convert_one_sim(sim_id, split, raq, fkt, fkp,
                    T_full, u_full, v_full,         # numpy [N_t, 128, 506]
                    times_full,                     # numpy [N_t]
                    xc_2d, yc_2d,                   # numpy [128, 506]
                    out_root):
    """
    把一条仿真的原生数据转成代码期望的 .pt 文件。
    """
    out_dir = os.path.join(out_root, split, f"sim_{sim_id}")
    os.makedirs(out_dir, exist_ok=True)

    # === 检查 1：方向 ===
    # y 方向：底应该是 0，顶应该是 1
    if yc_2d[0, 0] > yc_2d[-1, 0]:
        # 顶→底 reversed，需要 flip
        print(f"  [convert] flipping y axis")
        yc_2d  = yc_2d[::-1]
        T_full = T_full[:, ::-1, :]
        u_full = u_full[:, ::-1, :]
        v_full = v_full[:, ::-1, :]

    # === 检查 2：v 方向 ===
    # 一些求解器 v 正向下；需要的是正向上
    # 如果你的求解器约定 v 朝下，启用下面这行：
    # v_full = -v_full

    # === 检查 3：温度归一化 ===
    if T_full.max() > 100:
        # 看起来是 SI K，需要归一化
        T_top, T_bot = 273.0, 3500.0    # 改成你的实际值
        T_full = (T_full - T_top) / (T_bot - T_top)
        print(f"  [convert] normalized T from K to [0, 1]")

    # === 检查 4：速度无量纲化 ===
    # u_phys [m/s] → u_nd = u_phys × D / κ
    if abs(u_full).max() < 1e-3:
        # 看起来是 SI m/s，需要无量纲化
        D     = 2.89e6      # 改成你的实际域厚度 (m)
        kappa = 1e-6        # 改成你的实际热扩散率 (m²/s)
        u_full = u_full * D / kappa
        v_full = v_full * D / kappa
        print(f"  [convert] non-dimensionalized velocity (×{D/kappa:.2e})")

    # === 检查 5：时间无量纲化 ===
    if times_full.max() > 1e5:
        # 看起来是秒
        D, kappa = 2.89e6, 1e-6
        times_full = times_full * kappa / (D * D)
        print(f"  [convert] non-dimensionalized times")
    if times_full[0] == 0:
        times_full[0] = 1e-7   # 避免后续 t**(-0.25)

    # === 添加 channel 维度 ===
    T = torch.from_numpy(np.ascontiguousarray(T_full)).double().unsqueeze(1)
    u = torch.from_numpy(np.ascontiguousarray(u_full)).double().unsqueeze(1)
    v = torch.from_numpy(np.ascontiguousarray(v_full)).double().unsqueeze(1)
    assert T.shape == (T.shape[0], 1, 128, 506), f"shape {T.shape}"

    # === 保存 ===
    torch.save(T,  f"{out_dir}/e1_Tprev_data_select_snaps.pt")
    torch.save(u,  f"{out_dir}/e1_uprev_data_select_snaps.pt")
    torch.save(v,  f"{out_dir}/e1_vprev_data_select_snaps.pt")
    torch.save(torch.from_numpy(times_full).double(), f"{out_dir}/times.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(xc_2d)).double(),
               f"{out_dir}/xc.pt")
    torch.save(torch.from_numpy(np.ascontiguousarray(yc_2d)).double(),
               f"{out_dir}/yc.pt")
```

### 5.2 各主流求解器的具体读取代码

#### ASPECT (HDF5 + XDMF)

```python
import h5py, glob, numpy as np
from scipy.interpolate import griddata

def load_aspect_sim(sim_dir):
    """ASPECT 输出 solution-XXXXX.h5 + .xdmf。"""
    h5_files = sorted(glob.glob(f"{sim_dir}/solution/solution-*.h5"))
    T_list, u_list, v_list, times = [], [], [], []
    for f in h5_files:
        with h5py.File(f, 'r') as h:
            # ASPECT 是非结构网格，需要先插到规则 128×506
            nodes   = h['nodes'][:]              # [N_nodes, 2]
            T_unstr = h['T'][:]                  # [N_nodes]
            v_unstr = h['velocity'][:]           # [N_nodes, 2]
            t       = h.attrs['time']            # 标量
        # 规则网格目标
        xq = np.linspace(0, 4, 506)
        yq = np.linspace(0, 1, 128)
        Xq, Yq = np.meshgrid(xq, yq)
        T_reg = griddata(nodes, T_unstr,        (Xq, Yq), method='cubic')
        u_reg = griddata(nodes, v_unstr[:, 0],  (Xq, Yq), method='cubic')
        v_reg = griddata(nodes, v_unstr[:, 1],  (Xq, Yq), method='cubic')
        T_list.append(T_reg)
        u_list.append(u_reg)
        v_list.append(v_reg)
        times.append(t)
    return (np.stack(T_list), np.stack(u_list), np.stack(v_list),
            np.array(times), Xq, Yq)
```

> ASPECT 的非结构网格用插值会引入数值误差，更好的做法是**让 ASPECT 直接输出在均匀 128×506 网格上**。在 `prm` 文件里设置：
>
> ```
> subsection Mesh refinement
>   set Initial global refinement = 7   # 2^7 = 128 in y
>   ...
> ```

#### Underworld 2 (HDF5)

```python
import h5py, glob

def load_underworld_sim(sim_dir):
    h5_files = sorted(glob.glob(f"{sim_dir}/temperature_*.h5"))
    T_list, u_list, v_list, times = [], [], [], []
    for f in h5_files:
        step = f.split('_')[-1].split('.')[0]
        with h5py.File(f, 'r') as h:
            T = h['data'][:].reshape(128, 506)   # 取决于你保存的 swarm
        with h5py.File(f"{sim_dir}/velocity_{step}.h5", 'r') as h:
            uv = h['data'][:].reshape(128, 506, 2)
        T_list.append(T)
        u_list.append(uv[..., 0])
        v_list.append(uv[..., 1])
        # times 需要从 metadata 或 separate file 读
    ...
```

> Underworld 默认 v 正向上，符合代码约定，不用翻号。

#### StagYY (binary)

StagYY 输出是自定义二进制，需要用 `stagpy` 或 `staglib`：

```python
import stagpy

def load_stagyy_sim(sim_dir):
    sdat = stagpy.StagyyData(sim_dir)
    T_list, u_list, v_list, times = [], [], [], []
    for snap in sdat.snaps:
        T = snap.fields['T'].values.squeeze().T   # 转置到 [y, x]
        u = snap.fields['u'].values.squeeze().T
        v = snap.fields['v'].values.squeeze().T
        # StagYY 默认 z 朝上但格点顺序可能反，需要测试
        T_list.append(T)
        ...
```

#### COMSOL / FEniCS / Firedrake / 其它 FEM

跟 ASPECT 类似，FEM 输出是非结构的，需要插到均匀网格。可以用：

- ParaView Python API (`paraview.simple.ResampleToImage`)
- `scipy.interpolate.griddata`（慢但简单）
- `meshio` + `scipy.spatial.cKDTree` 自己写

### 5.3 完整批量转换 + 校验脚本

```python
import os, glob, torch, pandas as pd

# === 配置 ===
SOLVER      = "aspect"   # 你用的求解器
RAW_DIR     = "/path/to/raw_simulations"
OUT_DIR     = "my_data_dir"
PARAM_TABLE = "params.csv"   # 含 sim_id, raq, fkt, fkp, split

# === 主循环 ===
df = pd.read_csv(PARAM_TABLE)
sims_meta = []

for _, row in df.iterrows():
    sim_id = int(row['sim_id'])
    split  = row['split']
    raq    = float(row['raq'])
    fkt    = float(row['fkt'])
    fkp    = float(row['fkp'])
    raw_sim_dir = f"{RAW_DIR}/sim_{sim_id}"

    print(f"\n=== sim_{sim_id} ({split}) ===")
    if SOLVER == "aspect":
        T, u, v, times, xc, yc = load_aspect_sim(raw_sim_dir)
    elif SOLVER == "underworld":
        T, u, v, times, xc, yc = load_underworld_sim(raw_sim_dir)
    # ...

    convert_one_sim(sim_id, split, raq, fkt, fkp,
                    T, u, v, times, xc, yc, OUT_DIR)
    sims_meta.append([sim_id, split, raq, fkt, fkp, 128, 4,
                      f"{SOLVER}_sim_{sim_id}"])

# === 写 sims.pt ===
torch.save(sims_meta, f"{OUT_DIR}/sims.pt")
print(f"\nWrote {len(sims_meta)} simulations to {OUT_DIR}")

# === 校验 ===
os.system(f"python validate_sim_data.py {OUT_DIR}")
```

---

## 6. 校验流程

每生成一条仿真就跑一次校验（用根目录的 `validate_sim_data.py`）：

```bash
# 单条
python validate_sim_data.py my_data_dir train/sim_0

# 全部
python validate_sim_data.py my_data_dir
```

校验会跑这些检查：

| 类别 | 项 |
|---|---|
| 文件存在 | 6 个文件齐全 |
| Shape | snaps `[N, 1, 128, 506]`, coords `[128, 506]`, times `[N_t]` |
| dtype | float64 |
| 数值健全 | 无 NaN / Inf |
| 温度范围 | T ∈ [0, 1.05] |
| 温度边界 | T[底] ≈ 1, T[顶] ≈ 0（**容易踩 y 翻转**）|
| 坐标方向 | yc[底]=0, yc[顶]=1, xc[左]=0, xc[右]=4 |
| 速度量级 | `|u|max ∈ [1e-3, 1e7]` |
| 散度 | `|∇·u| / |u| < 1e-2`（**容易踩 face-stagger 没插值**）|
| 时间单调 | times 严格递增 |
| sims.pt schema | 字段类型、split 名、gr=128, ar=4 |

零 errors → 训练。有 errors → 按提示修。

---

## 7. 常见坑 - cheat sheet

按出现频率排序，复制保存当 cheat sheet：

| # | 现象 | 原因 | 修法 |
|---|---|---|---|
| 1 | 校验失败：T[底]≈0, T[顶]≈1 | y 轴方向反了（求解器顶→底）| `T_full = T_full[:, ::-1, :]`；同步 flip u, v, yc |
| 2 | 校验失败：`|∇·u|/|u| ≈ 0.1` | 速度是 face-stagger，没插到 cell center | 插值到 cell center，见 5.1 |
| 3 | 训练 loss 爆 NaN | 速度量纲是 m/s 没无量纲化 | 乘 `D/κ` |
| 4 | T 全是 0 或全是 1 | 温度无量纲化时除以了错的范围 | 用 `(T - T_top) / (T_bot - T_top)` |
| 5 | 训练 loss 不下降 | scaler 公式不匹配（你用的不是 GAIA 的物理设定）| 在你自己数据上重新拟合 `scaler.py` 三个系数 |
| 6 | 训练慢得离谱 | 数据放在网络盘 / 慢盘，每 epoch 都 IO 重读 | 数据放本地 NVMe；机器内存够大就 cache 进 RAM |
| 7 | OOM | dataset 在 `__init__` 一次性加载所有帧到内存 | 切小 sim 数量；或重写 `__getitem__` 改成 lazy load |
| 8 | sims_vec 长度 0，dataset 空 | sim 子目录命名错（不是 `sim_<num>` 格式）| 检查目录名 |
| 9 | dim 不对 (`expected 4, got 3`) | 忘了 `.unsqueeze(1)` 加 channel 维度 | T/u/v 必须 `[N, 1, 128, 506]` 不是 `[N, 128, 506]` |
| 10 | xc/yc 报 shape error | 存成了 1D 而不是 2D | 必须 `[128, 506]` 网格点坐标 |
| 11 | 校验过了但训练时 v 越来越大 | v 方向反了（求解器正向下）| `v = -v` |
| 12 | 训练能跑但推理时网络外推 | 你的参数超过了 `non_dimensionalize_*` 的 [0, 1] 区间 | 在 `calculate_profiles.py` 里改区间，或限制参数空间 |

---

## 8. 一个完整的 minimal 工作示例

假设你刚跑完 1 条仿真，验证整个 pipeline：

```bash
# 1. 准备目录和参数
mkdir -p my_data/train/sim_0

# 2. 跑你的转换脚本（伪代码）
python convert_aspect_to_pt.py \
    --raw /path/to/aspect/sim_0 \
    --out my_data/train/sim_0 \
    --raq 5.0 --fkt 1e8 --fkp 30.0

# 3. 写 sims.pt
python -c "
import torch
torch.save([[0, 'train', 5.0, 1e8, 30.0, 128, 4, 'aspect_test']],
           'my_data/sims.pt')"

# 4. 校验
python validate_sim_data.py my_data
# 期望输出: Errors=0, Warnings=0

# 5. 试跑训练（小 epoch）
# 首先把 paths.py 里 data_dir 指向 my_data
python multigpu.py -deb 1 -p_pred 0 ...
# debug=True 跑 1-2 epoch 验证 loss 下降
```

如果第 5 步跑通且 loss 在下降，再产剩下 49 条仿真，跑真训练。

---

## 9. 后续工作

按这个顺序推进：

1. **确定仿真求解器** —— ASPECT、Underworld、StagYY、COMSOL、自研代码或别的。本文档第 5.2 节给出了主流求解器的读取片段，但完整的 `convert_*_to_pt.py` 转换脚本（特别是处理坐标方向、staggered grid 插值、单位换算）需要按你团队的具体配置定制。

2. **确认物理设定**：

   - 是不是 2D Cartesian、AR=4？
   - 是纯内热还是底加热（或两者都有）？
   - 流变是 FK 还是其它（位错/扩散蠕变、塑性、复合）？
   - 边界条件是 free-slip 还是别的？

   如果有任何项偏离，需要判断是"复用预训练权重"还是"重训练 + 改 `scaler.py`"。

3. **跑 1 条小仿真做 pipeline 验证** —— 用 `validate_sim_data.py` 校验，再用 `debug=True` 跑 1–2 epoch 看 loss 是否下降。这一步成功后再扩到 50+ 条做真训练。

---

## 附录 A：与本仓库其它 PR 的关系

- 本文档对应的代码改动在 PR [`cursor/snaps-only-training-bb41`](https://github.com/hongliang-shl/PBML_Mantle_Convection/pull/1)：让 `debug=False` 训练能在仅有 `_select_snaps.pt` 文件时跑通。如果你按本文档准备数据，**可以选择**只生成 `_select_snaps.pt` 而不是 `_select_init.pt` + `_select.pt`，配合该 PR 即可训练。
- 校验脚本 `validate_sim_data.py` 在 PR [`cursor/add-sim-data-validator-bb41`](https://github.com/hongliang-shl/PBML_Mantle_Convection/pull/2)。

## 附录 B：参考文献与外部资源

- **作者论文与数据集**：DOI [10.5281/zenodo.15088589](https://doi.org/10.5281/zenodo.15088589)
- **GAIA**（作者使用的求解器）：DLR / WWU 联合开发，需要联系作者获取。
- **ASPECT**：[https://aspect.geodynamics.org](https://aspect.geodynamics.org)
- **Underworld 2**：[https://www.underworldcode.org](https://www.underworldcode.org)
- **StagYY** + **stagpy**：[https://stagpy.readthedocs.io](https://stagpy.readthedocs.io)
- **Latin Hypercube Sampling** in scipy：`scipy.stats.qmc.LatinHypercube`
