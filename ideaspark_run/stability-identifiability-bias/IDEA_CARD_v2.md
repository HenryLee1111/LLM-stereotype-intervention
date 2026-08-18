# ResearchStudio IdeaSpark — Idea Card (v2, completed)

> **v1 是 `IDEA_CARD.md`（保留未改动）。** v2 补足了 v1 缺失的 Phase 2.2 候选契约字段，并修复了 2.3 coherence gate 用可执行数值实验查出的三个 blocking 缺陷。改动清单见文末 §What changed。
>
> **Run dir**: `ideaspark_run/stability-identifiability-bias`
> **Canonical candidate**: `phase2_generate/phase2_generate_output.json`
> **Executed gate**: `phase2_coherence/coherence_trace.json` · `blocking_findings.json`
> **Audit**: `phase3_critique/phase3_critique_output.json` — verdict `advance`
> **Patterns**: `controlled_diagnostic_design` (anchor, sub-pattern **C02**) × `assumption_audit_and_pivot` (sibling, sub-pattern **C01**) — Composition: CHAIN

---

## Title

**Stable but Not Privileged: Separating Estimator Artifacts from Causal Privilege in LLM Bias Directions**

*中文：稳定但不特权 —— 区分 LLM 偏见方向中的「估计器假象」与「因果特权性」*

**Method name**: `PrivGap-DL` — 方向 / 位置双通道特权性诊断

---

## Hook

在同一批激活上，两个都被本领域用来「求偏见方向」的估计器，彼此的分歧远大于同一估计器在 bootstrap 重采样下的分歧。因此当前论文报告的那个 stability 数字，认证的是**估计器**，而不是**表征**。我们预言：真正能预测「这个方向是否比一个受同等优化的随机方向更有因果特权」的，是**跨估计器一致性 Stab_est**，而不是大家在报的 bootstrap 一致性 Stab_boot。

> Hook 形状 = (a) surprising empirical reveal。命名指标 Stab_est，其预测方向对熟悉 Kim et al. 与 stability 文献的读者是非显然的。(b) structural reframe 亦可用但被否决——(b) 是最廉价的形状，本候选的实质内容是一个可测的方向性预言，而不是换个说法。

---

## Motivation

### 背景

Kim et al. (arxiv:2503.02080, ICLR 2025) 在 attention head 激活里用 ridge probe 找到预测 DW-NOMINATE 的政治立场线性方向，并用 `x + α·σ·θ` 启发式干预 steer 输出。你的 Pilot 把这套范式移植到 gender-occupation 偏见，引入 Bootstrap Stability 作为可信度判据，在 Qwen2.5-0.5B 上得到 best layer stability = 0.979、CV accuracy = 1.00（`summary.txt`）。

### 瓶颈（Bottleneck）

领域内有一个未经检验的隐含假设链：**stability + heuristic steerability ⇒ identifiability**。Wang & Veitch (arxiv:2502.11447) 用 IPO + 局部 LoRA 重参数化构造最优局部干预，证明随机 head 的最优干预与 probe 筛选 head 效果相当，因此「编辑成功」不能作为定位证据。

但这里有**两个**瓶颈，而 v1 只处理了一个：

1. **因果侧**（v1 已识别）：编辑成功 ≠ 定位。
2. **估计侧**（v1 遗漏，Aaron 会议记录 §5–§7 的核心问题）：θ̂ = f(表征, 数据, **估计器**)。Bootstrap 固定估计器、变动数据；Aaron 问的是变动**估计器**会怎样。这一维在 v1 的 Stability 分解里完全不存在。

第 2 点不是锦上添花，它决定第 1 点是否可解释：如果 θ̂ 只是估计器的产物，那么 PrivGap(θ̂) 测的就是估计器，不是表征。

**已执行的证据**（`coherence_trace.json` probe 2）：mass-mean 估计器（ITI，arxiv:2305.14250 的 θ = μ⁺ − μ⁻）与 Fisher/LDA 估计器（Σ⁻¹(μ⁺ − μ⁻)）在解析上差一个逆协方差。在坐标尺度离散度 κ = 5 时，两者在**同一批数据**上相距 55°，|cos| = 0.57 —— 远低于 v1 定义的 0.90「稳定」线。LLM residual stream 的各向异性远超 κ = 10，所以这是一个保守下界。

| 各向异性 κ | cos(mass-mean, LDA) | 夹角 |
|---|---|---|
| 1 | 1.000 | 0.0° |
| 2 | 0.798 | 37.0° |
| 5 | 0.568 | 55.4° |
| 10 | 0.442 | 63.8° |

也就是说：**Stab_boot = 0.979 与「两个标准估计器对方向的分歧超过 60°」是完全相容的。**

### Gap

尚无工作为偏见方向同时提供：

1. **可操作的三层区分** —— 可复现性（数据轴）/ 表征内在性（估计器轴）/ 因果特权性（干预轴）；
2. **混淆隔离诊断** —— 在同一目标函数、同一优化预算下，一次只让一个因子变动；
3. **可分离的两类特权断言** —— 方向特权 vs 位置特权。这两者在现有文献里被当成一件事，而 Wang & Veitch 实际只测了后者。

---

## Core Mechanism — `PrivGap-DL`

把偏见方向当作**被估计量** θ̂ = f(表征, 数据, 估计器)，把领域现在用一个数字回答的三个问题拆开。

### 固定设定

- 模型 M：`Qwen2.5-0.5B-Instruct`、`Llama-2-7b-chat`（开源权重，需要梯度）
- 位点 s：residual layer ℓ，或 attention head (ℓ,h)；取最后一个 prompt token 的激活 x_s
- **估计器类 E** = {L2-logistic、linear SVM、Fisher LDA、ridge、mass-mean}，5 个成员，均为闭式解或 scikit-learn，每一个都已在本文献里被用来产出「偏见方向」
- **两套匹配 prompt 集**（同一份 OccuGender BLS 职业表，arxiv:2212.10678）：
  - `P_name`：性别信号由显式名字 token 承载（Pilot 现有的 4 个模板）
  - `P_free`：性别信号只由共指代词承载，**无名字 token**

### Level 1 — 可复现性（数据轴）

对估计器 e、分层 bootstrap 重复 b（**B = 20**，重采样比例 **ρ = 0.8**，沿用 Pilot 默认）：

$$\text{Stab}_{\text{boot}}(e,s) = \mathbb{E}_{i \neq j}\big|\cos(\hat\theta^{e,i}_s, \hat\theta^{e,j}_s)\big|$$

### Level 2 — 表征内在性（估计器轴）← v1 缺失

令 $\bar\theta^e_s$ 为估计器 e 的 bootstrap 均值方向：

$$\text{Stab}_{\text{est}}(s) = \mathbb{E}_{e \neq e'}\big|\cos(\bar\theta^{e}_s, \bar\theta^{e'}_s)\big| \qquad
\text{Stab}_{\text{lex}}(s) = \big|\cos(\bar\theta_s^{P_\text{name}},\ \bar\theta_s^{P_\text{free}})\big|$$

> 用 |cos| 而非 cos：判别方向的符号是标签编码约定，跨估计器不一致。D = 896 时各向同性随机向量的 |cos| 零均值 ≈ √(2/πD) ≈ 0.027，所以无符号一致性几乎不损失零分布校准。

### Level 3 — 因果特权性（干预轴）

**结果度量**：$M = \mathbb{E}_{P_\text{free}}\big[\text{logit}(\texttt{" she"}) - \text{logit}(\texttt{" he"})\big]$，在留出中性探针 `The {occupation} said that` 的续写位置上取；职业取自 **O_test**（种子固定的 30% 职业，任何 probe 都没见过）。$\Delta M = M(\text{intervened}) - M(\text{base})$。

**干预族**（Wang & Veitch 重参数化）：$W^\ell \leftarrow W^\ell (I + b a^\top)$，b 的支撑限制在目标位点。用 IPO 目标在 WinoBias 偏好对 $(x, y^+, y^-)$ 上拟合。

**唯一的设计动作：让 b 出现在两个嵌套因子层级上。**

| 通道 | b 的约束 | 承载因子 | 对应文献 |
|---|---|---|---|
| **DIRECTION (D)** | $b = c\,\hat\theta$，只学标量 c 与读出 a | **方向** | 无人做过 |
| **LOCATION (L)** | b 在该位点 $d_h$ 维内自由 | **位点** | Wang & Veitch 测的就是这格 |

> 这是可行的，因为 b 只通过外积 $b a^\top$ 进入，所以「固定 b 的方向、只学它的尺度」是自由 b 问题的**严格嵌套子模型**，目标函数与预算完全一致。

**特权性 = 对「受同等优化的零分布」的单侧随机化对比**：抽 **R_n = 200** 个对照臂（D 通道抽球面均匀随机单位向量，L 通道抽随机 head），**每个对照臂获得与处理臂完全相同的 IPO 预算**，然后

$$\text{PrivGap-D}(s) = \big|\Delta M(\hat\theta)\big| - Q_{1-q}\Big(\big\{|\Delta M(\theta_r)|\big\}_{r=1}^{R_n}\Big), \qquad
p = \frac{1 + \#\{r : |\Delta M(\theta_r)| \geq |\Delta M(\hat\theta)|\}}{1 + R_n}$$

PrivGap-L 同式，对照臂换成随机 head。

**为什么是分位数而不是最大值** —— 这是 v1 的一个致命错误，已用 Monte Carlo 执行验证（`coherence_trace.json` probe 1，3000 trials/pool，处理臂固定在真实 2σ 特权效应，零分布为半正态）：

| R_n | E[max\|ΔM_r\|] | E[PrivGap] | **P(PrivGap > 0)** | E[p_rank] |
|---|---|---|---|---|
| 20 | 2.169 | −0.169 | **0.388** | 0.0919 |
| 100 | 2.746 | −0.746 | **0.008** | 0.0544 |
| 500 | 3.243 | −1.243 | **0.000** | 0.0474 |
| 2000 | 3.631 | −1.631 | **0.000** | 0.0460 |

R_n 个抽样的最大值随 R_n 无界增长，所以 max 型 PrivGap 的**符号由池大小决定，而不是由方向决定**。两个后果都是致命的：(a) 在 v1 自己的 R_n = 20 下，一个真实的 2σ 特权方向有 **61% 的概率被判为「不可识别」**；(b) 取 R_n = 2000，任何方向都必被判为「不可识别」—— 预言可以靠调池大小人为制造。同一批抽样上的 rank p 值则**收敛**到真值 P(|Z| ≥ 2) = 0.0455。分位数/秩形式是相合的，最大值形式不是。

### 已命名参数（全部带默认值与选择规则）

| 符号 | 默认 | 选择规则 |
|---|---|---|
| `B` | 20 | Pilot 既有默认 |
| `ρ` | 0.8 | 分层 bootstrap 重采样比例，Pilot 既有默认 |
| `R_n` | 200 | `R_n ≥ 1/α_target − 1`，使最小可达 rank p 触及 `α_target = 0.005`；亦是首选的缩减旋钮 |
| `q` | 0.05 | 零分布尾部水平，扫 0.01–0.10 |
| `K` | 16 | 与 arxiv:2502.11447 的 top-16 臂对齐，使 L 通道可直接比较 |
| `R_h` | 16 | L 通道随机 head 对照池，与 K 匹配 |
| `stab_null_max` | 0.30 | 打乱标签臂的 Stab_boot 上限；越线则仪器本身有偏，停机检修而非报告零结果 |
| `η_floor` | 0.2 logits | 效率比 η = \|ΔM\|/\|ΔM_full\| 的分母下限，防止除零噪声主导 |

### 估计目标（Estimand）

估计目标**不是**「那个偏见方向」。对固定的 M、位点 s、prompt 分布 P（OccuGender 中性模板 × 留出 BLS 职业），估计目标是：

> 把干预约束到「在 s 处估计出的单一方向」上，相对于「在同一位点用受同等优化的球面均匀随机方向」所能达到的效应，在 M 上的**超额因果效应** —— 一个定义在 P 与 θ_r 的随机化分布上的总体量。

PrivGap-D 是它的 plug-in 估计量；PrivGap-L 是把随机化取在位点上的对应估计目标。**把两者分开声明，正是这两个断言能各自独立被证伪的原因。**

### 实施步骤

1. 建两套匹配 prompt 集 + O_test 留出划分。
2. 前向一遍，缓存每层 residual 与每个 (ℓ,h) 的末 token 激活（后续所有步骤复用，不再跑 encoder）。
3. 拟合估计器类：每个 (s, e, b) 得 θ̂，另加「职业内打乱性别标签」臂。
4. 算三个 stability 量 + 打乱标签校准。**校准闸门**：任一位点打乱臂 Stab_boot > `stab_null_max` → 停机检修。
5. 选位点：按 **P_free 上的留出 AUC**（绝不用 P_name，否则位点选择本身就继承了词汇捷径）取 top-K head，另抽 R_h 个随机 head。
6. 拟合 **D 通道**：`b = c·θ`，只学 (c, a)，IPO，2 epochs，rank-1，**各臂预算完全一致**。权重写回模型 → 下一次前向产出不同续写（这才是第 8 步测的东西，不是中间量的范数）。
7. 拟合 **L 通道**：b 在位点内自由；top-K head、R_h 随机 head、外加一个全层臂作为 full-model 上界。
8. 测下游：留出中性探针上的 ΔM；并行做 200 词续写 + LLM judge 刻板印象打分，**judge 分必须与流畅度守卫联合报告**，防止某臂靠「把模型弄坏」取胜。
9. 算 PrivGap-D / PrivGap-L 与 rank p 值，**报告完整对照分布**而非仅摘要，使读者可在别的 q 上自行重算。
10. 合表检验赌注：把 PrivGap-D 分别对 Stab_boot 与 Stab_est 回归，报告两个关联及其置信区间；效率比 η 仅在 full-model 臂过 `η_floor` 时报告。

---

## Falsification Prediction

> **最小实验**：在 Qwen2.5-0.5B-Instruct 与 Llama-2-7b-chat 上，用 5 成员估计器类在 OccuGender 衍生的 name-cued / name-free prompt 上、于 top-16 head 拟合方向；跑 D 通道（span-constrained IPO）对 **R_n = 200** 个受同等优化的球面随机方向；测 ΔM = 留出中性职业探针上 logit(" she") − logit(" he") 的均值变化。
>
> **预言方向**：(i) 同一位点上 **Stab_est 显著低于 Stab_boot** —— 同表征换估计器的分歧，大于同估计器换样本的分歧；(ii) **Stab_est 与 PrivGap-D 正相关且强于 Stab_boot 与 PrivGap-D 的关联，后者我们预言接近 0** —— 即大家在报的那个 stability 不预测因果特权，而没人报的那个预测。
>
> **承载变量（load-bearing variable）**：θ̂ 本身，即 D 通道里被供给的那个方向。
>
> **负对照**：置换它的**生产者** —— 在职业内打乱性别标签后重新拟合 θ̂，位点、目标函数、优化预算全部固定。预测效应落在**下游结果指标**上而非 θ̂ 自身：打乱臂的 |ΔM| 回落进受同等优化的随机方向带内，其 rank p 值变为 [0,1] 上的均匀分布。**若打乱臂仍能 steer 代词差，则该仪器测的是优化预算而不是方向，本诊断被证伪。**
>
> **正对照**：只学标量 c、把读出 a 冻结在基线值的精简臂，应能复现真实臂大部分的 |ΔM| —— 这把「证伪器」转成对「方向确为有效载体」的建设性识别。
>
> **独立复现检查**：预言 PrivGap-L 与 0 不可区分，即在刻板印象域复现 arxiv:2502.11447 在 truthfulness 上 `measured` 的结果；若 PrivGap-L 显著非零，则说明偏见设定不受同一「无特权位点」机制支配。

**双结局皆可发表**：预言成立 → 刻画线性偏见方向的因果边界，并给出「stability 该怎么报」的可操作结论；预言被证伪（PrivGap-D 显著且与 Stab_est 强相关）→ 首次建立 stability → causality 的桥梁。

> v1 里的 `PrivGap_heuristic > 0.5σ` 已删除。该 0.5 可追溯到 `outputs_bias_v2/v2_null_check.json` 中 `priv_gap_heuristic: 0.5` 及其旁注 *"Replace with real bootstrap..."* —— 那是占位符，不是推导值也不是实测值，把它写进 kill-switch 字段是 fabrication。可证伪性由「指标 + 方向 + 对照」提供，本就不需要这个数。

---

## Compute Budget

| 项 | 估计 |
|---|---|
| 激活抽取 + 全部 5 估计器 × 20 bootstrap × 2 prompt 集探针扫 | ~4 GPU-hours（一次 encoder pass + CPU 线代） |
| D 通道 Qwen2.5-0.5B：(1 true + 1 shuffled + 200 random) rank-1 IPO | ~5 GPU-hours |
| D 通道 Llama-2-7b-chat：同上 | ~40 GPU-hours |
| L 通道：(16 top + 16 random + 1 full-model) × 2 模型 | ~7 GPU-hours |
| 生成评估 judge | ~$200 |
| **合计** | **≈ 2.4 GPU-days + ~$200 API** |

对照 intake 声明的 40 GPU-hours + $50 / 单张 80GB / 一周：**tight，不是 comfortable** —— GPU 线约 1.4×，API 线约 4×。

**缩减杠杆（已命名）**：`R_n` 从 200 降到 60，省约 28 GPU-hours，使总量落回 40 GPU-hours 内；代价是最小可达 rank p 从 0.005 升到 0.016，对 0.05 水平的单侧检验仍然够用。**更推荐的做法**：pilot 轮用 R_n = 60，只对通过 pilot 的位点升到 R_n = 200 —— 把 p 值分辨率花在有效应的地方。砍掉 Llama-2-7b-chat 也能省，但会丢掉回应「单模型 artifact」质疑的跨模型复现，所以砍 R_n 优先。

所有数据集（OccuGender、WinoBias）与模型均为公开现有 artifact，无需自建标注。

---

## Differentiation from Literature

| 论文 | 他们做了什么 | 我们多做了什么 |
|---|---|---|
| **arxiv:2503.02080**<br>Kim et al., ICLR 2025 | ridge probe 方向预测 DW-NOMINATE；固定 α 的启发式编辑能改变生成的政治文本 | 构造他们的固定 α 设计**产不出**的「受同等优化随机化零分布」；测当对照臂拿到相同优化预算后，probe 方向是否还保有效应优势 |
| **arxiv:2502.11447**<br>Wang & Veitch | 自由 b 的局部最优编辑在随机 head 上匹配 top head 并逼近 full-model 上界（truthfulness）→ 关于**位点**的断言 | 加入 span-constrained 子模型 `b = c·θ`，把位点断言与方向断言分开；测他们的单因子设计从未实例化的方向级对比 |
| **arxiv:2511.19166**<br>Representational Stability of Truth | 单一估计器族内、标签扰动下的 probe 方向旋转，跨 16 个模型 | 测**跨估计器类**在固定数据与标签下的分歧；并检验两个 stability 量哪一个预测因果特权对比 —— 他们的旋转指标从未与之配对 |
| **arxiv:2212.10678**<br>OccuGender | BLS 锚定、模板受控的行为级偏见测量，无表征层对象 | 把他们的中性探针构造用作 name-free 臂，使 probe 的**观测模型**对「职业刻板印象」而非「名字性别」无偏；并把两臂之差作为一等量报告 |
| **arxiv:2305.14250**<br>ITI, Li et al. | 用 mass-mean θ = μ⁺ − μ⁻ 作干预方向，把这个选择当建模细节 | 把这个**选择本身**变成被测对象：mass-mean 只是 5 成员估计器类的一员，其与 LDA / logistic 解在同一批激活上的夹角本身就是「是否存在良定义的单一方向」的证据 |

**Almost-prior**：`arxiv:2502.11447`

**What step was missed**：他们构造了自由 b 的局部最优，其中**位点是唯一被操纵的因子**，并把由此得到的零结果读作「反对定位」的证据。要对**方向**下任何结论，还需要在同一目标函数、同一预算下的 span-constrained 子模型 `b = c·θ` —— 这是他们的重参数化本就容纳（因为 b 只经 `b aᵀ` 进入）却从未实例化的第二个因子层级。没有这个子模型，位点上的零结果对「固定位点上的方向是否特权」没有任何蕴含，而这两个断言一直被当成一个在读。

---

## Why This Composition

- **`controlled_diagnostic_design` / C02**（86 篇）：ICLR corpus 中处理「测量本身就是 artifact」问题的最强模式。其签名动作 —— *构造一组受控实例，只让一个被命名的混淆变动或被阻断，其余全固定，用独立于被测系统的 oracle 打分，把「表观性能与去混淆性能之差」本身作为头条结果* —— 在本候选中被具体实例化：双因子层级干预族让方向与位点各自单独变动，oracle 是随机化分布（独立于被测模型），头条正是 Level 1 认证的与 Level 3 测到的之间的背离。
- **`assumption_audit_and_pivot` / C01**（181 篇）：直击「编辑成功 = 定位」这一全领域隐含假设。C01 的结构动作是*把性质的执行/证据点从假定位点搬到被忽视的位点，并对「知道新位点的最强对手」做验证* —— 这里就是把定位证据从「在此处编辑能否改变输出」搬到「固定位点上的方向特权」与「自由 b 下的位点特权」两个可分别证伪的位点，对手是拿到相同优化预算的随机臂。
  > **Corpus 支持在本域偏薄（已声明）**：C01 的范例来自对抗安全与检测场景（后门触发器、模型溯源、恶意更新检测），`assumption_audit_and_pivot` 下没有任何子簇带表征探针范例。本条应按 C01 的抽象 Step-by-Step 评判，而非按范例表面相似度。

**链式**：先审计假设（为什么 heuristic 不可信、为什么单因子随机化不可解释），再设计诊断（如何受控地证伪）。

---

## Reviewer Concerns & Responses

| # | 质疑 | 严重度 | 回应 |
|---|---|---|---|
| 1 | **生态效度**：所有 prompt 集都是模板化的，C02 的头号失败模式正是「只在合成数据上验证合成 ground truth 的仪器」 | non-blocking | 把生成臂做到与 logit 臂同等标准：固定解码参数、写明 judge 模型与 rubric、对小规模人工标注子集报 inter-rater agreement、加流畅度守卫。或者显式把结论 scope 到模板化探针并在标题的 setting 里说清楚 |
| 2 | **C01 第五步偏薄**：「为什么估计器这个位点承载信号」目前只是各向异性论证，不是理论 | non-blocking | Park et al. (arxiv:2403.03867) 的 causal inner product 构造本身就是「有意义的几何是白化后的几何」这一陈述，它**预言** mass-mean 与 LDA 在未白化时必然分歧。把 Stab_est 框成对该预言的检验，可把一个否定性测量变成肯定性主张 |
| 3 | **「bias 只是 truth 换皮」** | non-blocking | L 通道本身就是这个质疑的检验：若 PrivGap-L 显著非零，则偏见域**不**受与 truthfulness 相同的机制支配，这本身是结果。再加跨模型（Qwen/Llama）与跨偏见轴（age）复现 |
| 4 | **文献底座是降级检索** | **blocking before submission** | `phase0/.connectors_degraded` 记录 openreview 跳过，`lit_table.md` 每行 `retrieved_via` 都是 `web_search` / `web_search+model_recall`，按 skill 自己的闸门这至多是 `webfallback`；其中两行（`arxiv:2605.05715`、`arxiv:2607.04439`）无法从 repo 内 artifact 核验。三篇承重引用（2503.02080 / 2502.11447 / 2212.10678）可独立核验且论证不依赖存疑行，所以这限制的是**查新完整度的信心**而非机制本身。投稿前须用可用连接器重跑 Phase 0 + 3.1 |
| 5 | **统计侧 reviewer**：「这不就是 specification curve / placebo arm 吗？」 | non-blocking | 承认同源并主动引用。领域特异结构有二：(i) 零分布抽样**改变的是优化问题**而非仅标签，每个对照臂必须在同一目标同一预算下重拟合，所以这是对「带冗余参数优化的统计量」做随机化检验，预算本身进入零假设规格；(ii) 方向与位点纠缠在同一个 `W^ℓ(I + baᵀ)` 参数化里，随机化必须施加在**两个嵌套因子层级**上才可解释 —— 单因子随机化恰恰就是「无特权位点」与「无特权方向」被混为一谈的成因。这个嵌套就是贡献。若这两条结构不存在，我们会诚实地把它标成 application-grade |
| 6 | **compute 超出声明 envelope** | non-blocking | 约 1.4× GPU 线、4× API 线，已在 §Compute Budget 明写。`R_n` 是命名杠杆，pilot 用 60、过筛位点升 200 |

---

## Validator Checklist

- [x] 每个 claim 有可追溯文献或被标为 model-supplied
- [x] 方法参数皆具名、带默认值与选择规则（B, ρ, R_n, q, K, R_h, stab_null_max, η_floor）
- [x] 数据 / 模型皆为现有 artifact（OccuGender, WinoBias, Qwen2.5-0.5B-Instruct, Llama-2-7b-chat）
- [x] Falsification 含**下游指标**上的负对照（非同义反复）+ 正对照 + 独立复现检查
- [x] Falsification 无杜撰数值门槛；唯一数值锚 tagged `measured in arxiv:2502.11447`
- [x] 观测模型前提已声明（P2：标签由名字 token 决定，故对「职业刻板印象」有偏）
- [x] Estimand 已精确声明（PrivGap-D 与 PrivGap-L 分别声明）
- [x] Naive-baseline audit 走 branch (i)，并完成 standard-tool follow-up
- [x] Sub-pattern 引用与父模式一致（C02 → controlled_diagnostic_design；C01 → assumption_audit_and_pivot）
- [x] Compute budget 具名，且与 intake envelope 的偏差被诚实标为 tight 并给出缩减杠杆
- [ ] **文献底座为 connector-verified** —— 未通过，见 Reviewer Concern #4
- [ ] **method_lineage 回归检查** —— 未通过：`phase1_output.json` 无 `method_lineage` 字段，hard rule 2 无法对照祖先树执行；`alias_terms` 因此纯由参数化知识构建。投稿前手查三个族：randomization inference over nuisance-optimized statistics、specification-curve / multiverse analysis、causal mediation 里的 sham control

---

## What changed from v1

### 🔴 修复的 blocking 缺陷（2.3 coherence gate，附可执行证据）

1. **`BLOCKING-1` — v1 的头条预言按构造为真，因而不可证伪。**
   v1 的 optimal 轨写「搜最优 b*（即最优 θ）」，同时又写「同一 b* 下对比 θ_true vs θ_rand vs θ_shuffled」。这两句不相容：若 b 在 head 内自由搜索，则 b* **就是**方向，供给的 θ 根本不进入优化，三个臂是同一个优化问题，PrivGap_optimal ≡ 0 是代数事实而非发现。同时 v1 把「无特权**位点**」（Wang & Veitch 实测的）与「无特权**方向**」（v1 想得出的）混为一谈 —— 正是 v1 批评 Kim et al. 的那个混淆。
   **修复**：拆成 D / L 两个嵌套因子层级；`b = c·θ` 是自由 b 的严格子模型（因为 b 只经 `b aᵀ` 进入）。

2. **`BLOCKING-2` — v1 的 PrivGap 是发散统计量，判决由池大小而非方向决定。**
   见上方 Monte Carlo 表。**修复**：max → (1−q) 分位数 + 精确 rank p 值；`R_n` 具名、默认 200、给出选择规则。

3. **`BLOCKING-3` — v1 falsification 里的 `0.5σ` 是占位符。**
   可追溯到 `v2_null_check.json` 的 `priv_gap_heuristic: 0.5` + *"Replace with real bootstrap..."* 旁注。**修复**：删除；改为「指标 + 方向 + 对照」，唯一数值锚 tagged `measured in arxiv:2502.11447`。

### 🟡 补足的 Phase 2.2 契约字段（v1 完全缺失）

`hook_shape_rationale` · `core_mechanism_reasoning`（PREMISES ledger 含强制的**观测模型前提** + **ESTIMAND** 声明 + **NAIVE-BASELINE AUDIT** 含 standard-tool follow-up + 设计理由与被否决的备选） · `gap_closure[].how_closed`（含 C01 的 thin-corpus 声明） · `almost_prior_paper_id` + `what_step_was_missed` · `signature_terms[]` + `alias_terms[]`（Phase 3.1 collision 检索所需） · `differentiation_from_lit[]` 扩到 5 篇并全部改写为**实质性**差异 · 全部参数具名化。

### 🟢 新增的研究内容

**估计器轴（Level 2）** —— 直接对应 Aaron 会议记录 §5「What if you change the probe?」、§6「θ = f(representation, data, estimator)」、§7「Does the representation define the direction, or does the probe?」、§8「Too stable can be suspicious」。v1 的 Stability 分解只有数据轴与模板轴，缺了 Aaron 称为「本项目最深问题」的那一维。`Stab_lex` 则对应 §4（John→Male 的词汇记忆）与 §10（控制实验清单）。

同时修正：**sub-pattern 引用 C03 → C02**。C03 的父模式是 `adapt_via_conditioning`（18 篇），不是 `controlled_diagnostic_design`；C02 才是（86 篇 —— 也正是 v1 自己在引的那个数字）。这一项会被 `subpattern_citation_consistency` validator 判失败。

### ⚪ 未生成（环境限制，非省略）

`phase4_skeleton.py` / `phase4_assemble` / `phase4_render` 需要 Python，本机 `python` 仅有 Microsoft Store 占位符（exit 49），故 `phase4_expansion.json` 与 LaTeX/PDF 渲染未生成。本卡片与 `idea.std.en.md` 以人工方式承载了同一份记录的内容。Phase 0 / 3.1 的真实连接器检索同样不可用 —— 见 Reviewer Concern #4。
