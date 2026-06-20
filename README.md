# Student Kramers 科研项目

这个仓库用于继续研究部分观测的 Student Kramers 模型。它不是论文旧结果的展示仓库，而是一套可以反复修改模型、重新计算、检查中间结果并生成新实验结果的科研工作流。

核心原则：

1. 数学公式和可复用计算写在 `student_kramers/*.py`。
2. `notebooks/new project.ipynb` 是当前研究主界面，负责按研究顺序调用函数、查看结果和画图。
3. 可复用研究图写在 `student_kramers/figures.py`；notebook 只加载结果并调用短函数。
4. 昂贵计算由短小的 `run_*.py` 命令行入口执行，可以中断后继续。
5. 每次新实验使用独立的 `RUN_NAME`，结果写入 `results/runs/<RUN_NAME>/`，避免覆盖或误用旧实验。

## 1. 项目结构

```text
项目/
├── data/
│   └── official_ice_data.xlsx  # 首次运行时自动下载，本地文件不上传
├── notebooks/
│   └── new project.ipynb
├── results/
│   └── runs/
├── student_kramers/
│   ├── config.py
│   ├── models.py
│   ├── data_loading.py
│   ├── likelihoods.py
│   ├── estimation.py
│   ├── simulation.py
│   ├── bootstrap.py
│   ├── bootstrap_analysis.py
│   ├── ios_analysis.py
│   ├── run_single.py
│   ├── run_validation.py
│   ├── run_application.py
│   ├── run_bootstrap.py
│   ├── run_bootstrap_analysis.py
│   ├── run_modelwise_ios_bootstrap.py
│   ├── run_modelwise_ios_analysis.py
│   ├── run_ios_analysis.py
│   └── run_analysis.py
├── tests/
├── pyproject.toml
├── M4_RESEARCH_WORKFLOW_ZH.md
├── M4_RESEARCH_WORKFLOW_EN.md
├── PYTHON_BEGINNER_GUIDE.md
├── PYTHON_BEGINNER_GUIDE_EN.md
└── README.md
```

旧论文运行产生的文件可以保留在本地 `results/archive_completed_paper/`，但它们不会被新代码自动读取，也不会上传 GitHub。

## 2. 第一次安装

所有命令都从项目根目录执行，也就是包含 `pyproject.toml` 的目录。

```bash
cd "/Users/sal/Desktop/Statistic Project/项目"

python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

`-e` 表示 editable installation。修改 `student_kramers/*.py` 后不需要重新安装。

第一次读取数据时，`data_loading.py` 会从项目中记录的官方 URL 下载 Excel 文件到 `data/`。该本地数据文件不会上传 GitHub。

运行测试：

```bash
python3 -m unittest discover tests -v
```

启动 notebook：

```bash
jupyter lab "notebooks/new project.ipynb"
```

## 3. 文档入口

| 文档 | 中文 | English |
|---|---|---|
| M4 研究工作流程、失败诊断、当前结果与审查表 | [`M4_RESEARCH_WORKFLOW_ZH.md`](M4_RESEARCH_WORKFLOW_ZH.md) | [`M4_RESEARCH_WORKFLOW_EN.md`](M4_RESEARCH_WORKFLOW_EN.md) |
| Python 初学者教材第六版（中文已更新） | [`PYTHON_BEGINNER_GUIDE.md`](PYTHON_BEGINNER_GUIDE.md) | [`PYTHON_BEGINNER_GUIDE_EN.md`](PYTHON_BEGINNER_GUIDE_EN.md) |
| 当前已执行研究 notebook | [`notebooks/new project.ipynb`](notebooks/new%20project.ipynb) | same notebook |

研究过程、失败尝试和决策理由记录在 M4 workflow 中。Python Guide 解释代码本身。
两类文档不要混用：workflow 回答“我们做了什么和为什么”，guide 回答“代码怎样工作”。

## 4. 当前 M4 状态

第一次 M4 真实数据拟合只从 M3 边界
`delta = epsilon = zeta = 0` 开始，因此硬惩罚优化器被锁在边界。当前正式实现把
`q(x,v) - floor` 的增广矩阵写成 `L @ L.T`，使 M4 的每个优化试探都全局有效；同时
保留 M3 边界、全局内部起点和已有最佳 M4 warm start 用于审查。

当前正式真实数据结果保存在：

```text
results/runs/m4_real_data_cholesky/
```

| 模型 | NLL | AIC | BIC | 全局最小 q |
|---|---:|---:|---:|---:|
| M2 | 8524.348 | 17060.695 | 17095.637 | 4176.668 |
| M3 | 8524.059 | 17064.118 | 17110.707 | 4167.946 |
| M4 | **8499.312** | **17020.623** | **17084.683** | **0.000276** |

M4 的全局最小扩散接近 floor，但该最小点位于离观测区域极远的位置；观测矩形上的
最小扩散约为 3867。增广矩阵最小特征值接近零，因此 M4 的常规 IOS 渐近参考值不应
作正式校准使用。

当前研究里程碑是：**正式 M2/M3/M4 real-data fit、sampled IOS pilot、observed
exact IOS 和 M4 model-wise IOS bootstrap 均已完成。** observed exact IOS 保存在
`results/runs/ios_observed/`，有限样本 M4 calibration 保存在
`results/runs/m4_modelwise_ios_bootstrap/`。

| 模型 | observed exact $T_N$ | 有效 transition | 渐近参考 | 当前解释 |
|---|---:|---:|---:|---|
| M2 | 8.589 | 2499/2499 | 9.75 | 参考可描述比较，正式有限样本校准未做 |
| M3 | 11.549 | 2499/2499 | 11.75 | 参考可描述比较，正式有限样本校准未做 |
| M4 | 21.876 | 2499/2499 | 18.50 | 使用 200 次 model-wise bootstrap 校准 |

当前 M4 parametric bootstrap 已扩展到 500 次，496/500 次成功，用于参数和扩散曲面
不确定性检查。严格的 M4 model-wise IOS bootstrap 已完成 200/200 次：

```text
results/runs/m4_parametric_bootstrap/
results/runs/m4_modelwise_ios_bootstrap/
```

前者回答“参数和 $q(x,v)$ 的 bootstrap uncertainty 是什么”；后者回答
“真实 observed $T_N$ 在 fitted-M4 有限样本分布里是否异常”。这两个结果不能混用。

M4 model-wise IOS 结果：

| 指标 | 数值 |
|---|---:|
| observed $T_N$ | 21.876 |
| bootstrap median | 39.829 |
| bootstrap 95% interval | [21.089, 154.801] |
| upper-tail $p$ | 0.965 |
| lower-tail $p$ | 0.040 |

planned upper-tail IOS 检查不拒绝 M4。lower-tail 结果说明 observed statistic 位于
bootstrap 分布低端，应作为校准诊断保留，不能把“不拒绝”改写成“证明 M4 正确”。

## 5. Notebook 应该怎样使用

`notebooks/new project.ipynb` 是当前最常打开的研究文件。它已经本地从头执行并保存输出。

顶部固定指向当前正式结果：

```python
FIT_RUN = "m4_real_data_cholesky"
PILOT_RUN = "ios_pilot_cholesky"
IOS_RUN = "ios_observed"
BOOTSTRAP_RUN = "m4_modelwise_ios_bootstrap"
```

notebook 按 A-D 顺序读取并展示：

- 正式 M2/M3/M4 real-data fit 与机制图；
- M4 Cholesky 优化审查；
- 三模型共同 sampled IOS pilot；
- 三模型 observed exact IOS；
- 影响集中度、模型间一致性、状态空间位置和参数移动；
- M4 200 次 model-wise IOS bootstrap 校准和累计 p-value 稳定性。

它不会重新运行 2499 次 leave-one-out 拟合，而是读取已经缓存的 CSV 并调用
`student_kramers` 中的分析与绘图函数。当前 notebook 共 28 个 cells，已经从头执行，
17 个 code cells 全部无错误且输出已内嵌保存。

## 6. 推荐的完整研究顺序

以下示例使用同一个实验名称：

```text
m4_trial_01
```

### 第 0 步：运行测试

```bash
python3 -m unittest discover tests -v
```

修改数学公式后必须先运行测试。

### 第 1 步：可选的模拟验证

验证从已知参数模拟数据后，完整观测和部分观测估计器是否工作：

```bash
python3 -m student_kramers.run_validation \
  --generate-model M4 \
  --fit-models M3 M4 \
  --run-name m4_trial_01 \
  --n-starts 2
```

`--generate-model` 选择用于模拟数据的模型，`--fit-models` 选择随后拟合的模型。
这一步不是每次实验都必须运行，但修改 likelihood 或 simulator 后应该运行。

### 第 2 步：拟合真实数据

拟合全部 registry 模型：

```bash
python3 -m student_kramers.run_application \
  --run-name m4_trial_01 \
  --n-starts 8
```

只拟合指定模型：

```bash
python3 -m student_kramers.run_application \
  --run-name m4_trial_01 \
  --models M2 M4 \
  --n-starts 8
```

结果保存为：

```text
results/runs/m4_trial_01/model_fits.csv
```

### 第 3 步：快速检查一个拟合结果

```bash
python3 -m student_kramers.run_single \
  --run-name m4_trial_01 \
  --model M2
```

它会重新计算保存参数对应的 NLL。保存值和重新计算值应一致。

### 第 4 步：计算 exact IOS

```bash
env VECLIB_MAXIMUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  python3 -m student_kramers.run_bootstrap \
  --fit-run m4_real_data_cholesky \
  --run-name ios_observed \
  --mode ios \
  --model M2
```

IOS 会为每个 transition 重新拟合模型，通常是非常昂贵的步骤。程序会持续保存
checkpoint。这里限制 BLAS 为单线程，是因为当前 likelihood 包含大量很小的矩阵
指数；实测单线程明显快于每次小运算都启动多个 BLAS 线程。

全部模型完成后生成比较表和图：

```bash
python3 -m student_kramers.run_ios_analysis \
  --fit-run m4_real_data_cholesky \
  --run-name ios_observed
```

### 第 5 步：nested-model contrast bootstrap

例如检验 M1 与 M2：

```bash
python3 -m student_kramers.run_bootstrap \
  --run-name m4_trial_01 \
  --mode contrast \
  --model M1 \
  --contrast-alt M2 \
  --n-boot 100
```

其中 `--model` 是 null model，`--contrast-alt` 是 alternative model。

### 第 6 步：parametric bootstrap

只重新拟合参数：

```bash
python3 -m student_kramers.run_bootstrap \
  --run-name m4_trial_01 \
  --mode parametric \
  --model M2 \
  --n-boot 100
```

每个 bootstrap replication 内同时重新计算 exact IOS：

```bash
env VECLIB_MAXIMUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  python3 -m student_kramers.run_modelwise_ios_bootstrap \
  --fit-run m4_trial_01 \
  --run-name m4_trial_01_modelwise_ios \
  --model M4 \
  --n-boot 200 \
  --n-workers 6
```

如果刚修改过 IOS 判定但想保留已有成功 replication 的 row checkpoint，可加：

```bash
  --retry-failed --no-resume
```

普通同代码版本扩展到 300 或 500 次时，不需要 `--no-resume`，只改 `--n-boot`。

最后一种运行会非常昂贵，因为每个 replication 都要重新做一次 exact IOS。完成后：

```bash
python3 -m student_kramers.run_modelwise_ios_analysis \
  --run-name m4_trial_01_modelwise_ios \
  --observed-run ios_observed \
  --model M4
```

如果只做普通 M4 parametric bootstrap，完成后用：

```bash
python3 -m student_kramers.run_bootstrap_analysis \
  --fit-run m4_trial_01 \
  --run-name m4_trial_01 \
  --model M4
```

### 第 7 步：模拟诊断

```bash
python3 -m student_kramers.run_analysis \
  --run-name m4_trial_01 \
  --model M2 \
  --simulate \
  --n-rep 100 \
  --n-first-passage 1000
```

该脚本只计算并保存诊断表格，不画图。打开 notebook 后直接使用可见的绘图 cell 画图。

### 第 8 步：在 notebook 中检查和画图

将 notebook 顶部设置为：

```python
RUN_NAME = "m4_trial_01"
FOCUS_MODEL = "M2"
```

关闭昂贵开关后执行 `Run All`。Notebook 会读取该实验已经生成的结果并画图。

## 7. 各个 Python 文件负责什么

### 正式数学与计算模块

| 文件 | 用途 | 通常是否直接运行 |
|---|---|---|
| `config.py` | 数据路径、步长、随机种子、优化和模拟规模 | 否 |
| `models.py` | 参数顺序、模型 registry、force、potential、diffusion、约束 | 否 |
| `data_loading.py` | 下载/读取数据、预处理、结果路径、CSV 和 checkpoint provenance | 否 |
| `likelihoods.py` | 完整观测和部分观测 Strang pseudo-likelihood | 否 |
| `estimation.py` | L-BFGS-B、多起点优化、批量模型拟合 | 否 |
| `simulation.py` | SDE 模拟、waiting time、first passage、model check | 否 |
| `recovery.py` | complete/partial repeated parameter recovery | 否 |
| `discrimination.py` | M3/M4 discrimination scenarios | 否 |
| `bootstrap.py` | exact IOS、parametric bootstrap、contrast bootstrap | 否 |
| `bootstrap_analysis.py` | bootstrap 汇总表、分位数和扩散诊断 | 否 |

这些文件由 notebook 或 `run_*.py` 调用。修改数学模型时主要修改这里。

### 可以直接执行的入口

| 文件 | 用途 | 是否昂贵 |
|---|---|---|
| `run_single.py` | 检查环境并重新计算一个已拟合模型的 NLL | 便宜 |
| `run_validation.py` | 已知参数下的模拟恢复实验 | 中等或昂贵 |
| `run_recovery.py` | 重复参数恢复实验 | 昂贵 |
| `run_discrimination.py` | M3/M4 区分实验 | 昂贵 |
| `run_application.py` | 对真实数据拟合一个或多个模型 | 昂贵 |
| `run_bootstrap.py` | IOS、parametric bootstrap、contrast bootstrap | 非常昂贵 |
| `run_bootstrap_analysis.py` | 汇总普通 parametric bootstrap 并画参数/扩散图 | 便宜 |
| `run_modelwise_ios_bootstrap.py` | 并行运行严格 model-wise IOS bootstrap | 最昂贵 |
| `run_modelwise_ios_analysis.py` | 汇总 model-wise IOS bootstrap 并画校准图 | 便宜 |
| `run_analysis.py` | 计算 waiting time、model check、first passage 表格 | 昂贵 |

不要在 `student_kramers/` 目录内直接执行 `python3 run_application.py`。统一从项目根目录使用：

```bash
python3 -m student_kramers.run_application
```

这种方式能保证 package import 行为一致。

## 8. 当前统一的 M1-M4 模型族

四个模型现在共享同一个十一维完整参数向量：

```text
[eta, a, b, c, d, alpha, beta, gamma, delta, epsilon, zeta]
```

共同 SDE 为：

\[
dX_t=V_tdt,
\]

\[
dV_t=[-\eta V_t + ax^3+bx^2+cx+d]dt
      +\sqrt{\alpha v^2+\beta v+\gamma
      +\delta x^2+\epsilon xv+\zeta x}\,dW_t.
\]

M1-M3 将新增的 `delta`、`epsilon`、`zeta` 固定为 0。M4 释放全部十一
个参数。因此当三个新增参数为 0 时，M4 必须退化成 M3。测试会同时检查
扩散函数、模拟路径、完整观测 likelihood 和部分观测 likelihood 的退化关系。

M4 的位置相关扩散改变了二阶矩传播。`likelihoods.py` 使用老师代码中的
`check_alpha`、`check_beta`、`check_gamma` 和 `I1-I5` 结构实现它，并用直接
的 7×7 矩方程作为独立数值校验。

### 参数约束

所有模型保留：

\[
\eta>0,\qquad a<0,\qquad 0\leq\alpha<2\eta.
\]

扩散方差写成：

\[
q(x,v)=
\begin{bmatrix}x&v&1\end{bmatrix}
H
\begin{bmatrix}x&v&1\end{bmatrix}^{\mathsf T},
\]

\[
H=
\begin{bmatrix}
\delta & \epsilon/2 & \zeta/2\\
\epsilon/2 & \alpha & \beta/2\\
\zeta/2 & \beta/2 & \gamma
\end{bmatrix}.
\]

代码检查减去一个很小正数后的矩阵是否半正定，从而保证所有实数
`x, v` 上 `q(x,v)` 都保持为正。使用半正定而不是严格正定很重要，因为
M1-M3 位于 M4 的边界；强制 `delta > 0` 会错误地破坏这一嵌套关系。

### M4 优化

批量拟合按 `M1 M2 M3 M4` 顺序运行时，M4 会保留 M3 边界结果作为基准，并
自动生成非零、全局正定的 M4 内部起点。也可以使用 `--warm-start-run` 加入
上一轮最佳 M4。所有起点最终仍联合优化全部十一个参数。

然后：

```bash
python3 -m unittest discover tests -v

python3 -m student_kramers.run_application \
  --run-name m4_trial_01 \
  --models M1 M2 M3 M4 \
  --n-starts 8
```

在 notebook 顶部可以把 M4 加入需要的列表：

```python
FOCUS_MODEL = "M4"
IOS_MODELS = ["M2", "M4"]
BOOTSTRAP_MODELS = ["M4"]
```

### 将来添加 M5

如果后续模型改变以下任意内容，仅添加 registry entry 不够：

- force 不再是 cubic polynomial；
- diffusion variance 不再是 quadratic polynomial；
- 状态维数改变；
- 观测方式改变；
- Strang splitting 公式改变；
- 参数约束或稳定性条件改变。

此时至少需要同步检查：

1. `models.py` 中的数学函数和约束；
2. `likelihoods.py` 中的 splitting 与 transition likelihood；
3. `simulation.py` 中的 SDE simulator；
4. `tests/` 中的新模型数值测试；
5. notebook 中的解释文字和绘图范围。

## 9. 结果目录与 checkpoint

所有新结果都保存在：

```text
results/runs/<RUN_NAME>/
```

请不要让两个含义不同的实验共用同一个 `RUN_NAME`。

模型拟合、IOS 和 bootstrap CSV 旁边会保存 `.meta.json` provenance 文件，其中包含：

- workflow 类型；
- model；
- 参数 hash；
- 数据 hash；
- 代码 hash；
- 时间步长；
- bootstrap 或 IOS 设置。

继续运行时，程序会先比较 provenance。如果模型、数据、配置或代码已经改变，程序会拒绝继续旧 checkpoint，并要求使用新的 `RUN_NAME`。这样可以避免把旧结果误认为新模型结果。

正常继续未完成计算：

```bash
python3 -m student_kramers.run_bootstrap \
  --run-name m4_trial_01 \
  --mode ios \
  --model M4
```

明确不继续旧 checkpoint、重新开始：

```bash
python3 -m student_kramers.run_bootstrap \
  --run-name m4_trial_01 \
  --mode ios \
  --model M4 \
  --no-resume
```

更推荐为真正不同的实验创建新的 `RUN_NAME`，而不是覆盖已有结果。

## 10. 运行时间的相对大小

| 工作流 | 相对成本 |
|---|---|
| 数据读取、测试、`run_single` | 秒级或较快 |
| 单模型多起点拟合 | 中等，取决于起点数量 |
| simulation validation | 中等到昂贵 |
| exact IOS | 非常昂贵，需要逐 transition 重拟合 |
| parametric bootstrap | 非常昂贵，需要逐 replication 重拟合 |
| bootstrap 内再做 exact IOS | 最昂贵 |

先使用小规模参数验证工作流，再进行正式运行。

## 11. 上传 GitHub 前

`results/`、虚拟环境、缓存和 notebook checkpoint 已经被 `.gitignore` 忽略。上传前运行：

```bash
python3 -m unittest discover tests -v
```

建议公开仓库前再由项目负责人决定 LICENSE 和数据引用文字。
