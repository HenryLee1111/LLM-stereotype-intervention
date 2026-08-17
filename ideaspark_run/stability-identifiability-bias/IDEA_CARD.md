# ResearchStudio IdeaSpark — Idea Card
## 基于 Microsoft ResearchStudio `idea-spark` 流水线生成（controlled_diagnostic_design × assumption_audit_and_pivot）

> **Run dir**: `ideaspark_run/stability-identifiability-bias`  
> **Input query**: *Define Stability and Identifiability for LLM bias directions, then complete the research building on Kim et al. ICLR 2025 (2503.02080) and incorporating Wang & Veitch 2502.11447 feedback*  
> **Patterns**: `controlled_diagnostic_design` (anchor) + `assumption_audit_and_pivot` (sibling) — *Composition: CHAIN*  
> **Source grounding**: 文献池已包含 2503.02080、2502.11447、Li et al. 2023 (ITI)、Park et al. 2024 (Linear Representation Hypothesis)、Gurnee & Tegmark 2024

---

### Title

**Stable But Not Privileged? A Confound-Isolating Diagnostic for the Identifiability of Bias Directions in LLMs**

*中文：稳定但不特权？—— LLM 偏见方向可识别性的混淆隔离诊断*

---

### Hook (一句话卖点)

现有工作（含 Kim et al. ICLR 2025）将“探针准确率高 + 启发式编辑能 steer”视为偏见方向存在的证据，但 Wang & Veitch (2025) 证明随机位置的最优编辑同样有效——本研究首次将偏见方向问题重构为**可证伪的诊断学问题**：用受控对比实验同时隔离统计稳定性与因果特权性，回答“高稳定性的偏见方向是否在因果上可识别，或仅是分布式编码的投影幻象”。

---

### Motivation

#### 背景
Kim et al. (2503.02080) 在 Llama-2/Mistral/Vicuna 上证明：用 ridge probe 预测议员 DW-NOMINATE 分数可在 attention head 中找到政治立场线性方向，并可用 $x + \alpha\sigma\theta$ 线性干预 steer 输出。你的前期 Pilot 在 Qwen2.5-0.5B 上将此范式移植到 gender-occupation 偏见，引入 **Bootstrap Stability** ($E[\cos(\theta^{(i)},\theta^{(j)})]$) 作为可信度判据，发现最佳层 stability=0.979、CV=100%。

#### 瓶颈（Bottleneck）
领域内隐含假设 **“stability + heuristic steerability = identifiability”** 未被检验。Wang & Veitch (2502.11447, UChicago) 用 IPO + 局部 LoRA 重参数化构造**最优局部干预**，证明：(1) heuristic $x+\alpha\sigma\theta$ 远非最优；(2) 随机 heads 的最优干预与 probe 筛选 heads 效果相当；(3) 单个随机 head 即可达 full-model 上界。该结果直接动摇 2503.02080 的验证逻辑——**编辑成功不能作为定位证据**。你的 Pilot 恰好处于此裂缝：高 stability 但无随机基线、无最优对照、无因果特权性度量。

#### Gap
尚无工作为偏见方向提供：
1. **形式化区分**：Stability（统计可重复性）vs Identifiability（因果特权性）的可操作定义；
2. **混淆隔离诊断**：同时控制采样/模板/层/优化最优性四个混淆的对照实验；
3. **证伪谓词**：若方向不可识别，诊断应能产出可发表的负结果（约束线性表征假说边界）。

---

### Core Mechanism

**机制名：PrivGap Diagnostic — 四控对比诊断框架**

#### 核心思想
将 `assumption_audit_and_pivot`（审计“编辑成功→定位”假设）与 `controlled_diagnostic_design`（构造仅单轴变化的对比对）链式组合：
- **审计**：将 Kim et al. 的启发式验证假设显式化为可检验命题
- **诊断**：构造四组受控对比，仅让目标轴变化，其余全固定

#### 方法流程（5 步）

**Step 1: 形式化与度量**
- 定义 **Stability**：$ \text{Stab}_{\text{boot}}$, $\text{Stab}_{\text{template}}$（见 DEFINITIONS.md）
- 定义 **Identifiability**：$ \text{PrivGap}(\theta^*) = |\Delta M(\theta^*)| - \max_{\theta_r}|\Delta M(\theta_r)| $，其中 $\Delta M$ 为偏见度量变化，$\theta_r$ 为随机零分布

**Step 2: 受控方向估计**
- 对每层/每头估计三种方向：
  - $\hat{\theta}_{\text{true}}$：真实标签训练
  - $\hat{\theta}_{\text{shuffled}}$：标签打乱 null
  - $\hat{\theta}_{\text{rand}}$：各向同性随机向量（20 个）
- 报告：Stability 分布、cross-layer 余弦矩阵、shuffled 的 $\approx 0$ 校准

**Step 3: 双轨干预（Heuristic vs Optimal）**
- **Heuristic 轨**：复现 Kim et al. 式 $x^{(\alpha)}_{\ell,h}=x_{\ell,h}+\alpha\hat{\sigma}_{\ell,h}\hat{\theta}_{\ell,h}$，扫 $\alpha\in[-3,3]$
- **Optimal 轨**：移植 Wang & Veitch Sec.4：对 $W^\ell$ 做重参数化 LoRA $W^\ell + W^\ell b a^\top$，限制 $b$ 仅在目标 head/layer 非零，用 IPO 目标在 WinoBias 偏好对 $(x, y^+, y^-)$ 上优化，搜最优 $b^*$（即最优 $\theta$）
- 上界：full-model IPO（所有层可学）

**Step 4: 因果效应度量 $\Delta M$**
- **中性职业探针**：`The {occupation} said that` 后续代词 $logit(\text{she})-logit(\text{he})$
- **生成评估**：对 9 个策略议题生成 200 词短文，用 GPT-judge 偏见分数 + Info*Truth 权衡（沿用 2502.11447 指标）
- **对照**：同一 $\alpha$ 或 $b^*$ 下对比 $\hat{\theta}_{\text{true}}$ vs $\hat{\theta}_{\text{rand}}$ vs $\hat{\theta}_{\text{shuffled}}$

**Step 5: 可识别性判决**
- 计算每层/每头的 $\text{PrivGap}_{\text{heuristic}}$ 与 $\text{PrivGap}_{\text{optimal}}$
- 检验：$\text{PrivGap}_{\text{optimal}} > 0$ 的 $p$ 值（对随机零分布 permutation test）
- 相关性分析：$\text{corr}(\text{Stab}, \text{PrivGap})$ —— 若显著正相关，则 stability 预测因果特权；若 $\approx 0$，则发表“稳定但不特权”负结果，约束线性表征假说

---

### Falsification Prediction (可证伪预言)

**最小实验 + 度量 + 负对照 + 下游回归**：

> *在 Qwen2.5-0.5B 与 Llama-2-7b-chat 上，对 gender-occupation 任务训练 5 层 × 20 bootstrap 的 $\hat{\theta}$，分别用 heuristic 与 optimal 干预评估中性职业句的代词 logit 差 $\Delta M$。我们预言：Heuristic 轨的 $\text{PrivGap}_{\text{heuristic}} > 0.5\sigma$（真方向优于最佳随机方向 0.5 个标准差），但 Optimal 轨的 $\text{PrivGap}_{\text{optimal}}$ 将回归至 0（$p>0.1$）——即一旦允许最优优化，随机方向即可复现真方向的因果效应。*

**负对照**：对同一随机方向池固定 $b^*$ 优化器但改用打乱标签方向 $\hat{\theta}_{\text{shuffled}}$，其 $\Delta M$ 应回落至基线（$\approx 0$），否则诊断仪器本身有偏。

**判定**：
- 若预言成立 → 支持 Wang & Veitch 的“无特权定位”假说在偏见领域同样成立，贡献为**刻画线性偏见方向的因果边界**；
- 若预言被证伪（$\text{PrivGap}_{\text{optimal}} > \epsilon$ 且与 $\text{Stab}$ 强相关）→ 则首次证明偏见方向的可识别性可由 stability 预测，贡献为**建立 stability→causality 的桥梁**。
*两种结局皆可发表*（Oral 要求的可证伪性）。

---

### Compute Budget

- **Pilot 已用**：Qwen2.5-0.5B 在 CPU 上 640 样本 × 24 层 × 20 bootstrap ≈ <2 GPU-hours
- **完整诊断**：2 模型 × (head-level probe 1024 heads + 5 层 × 20 bootstrap) 探针训练 ≈ 8 GPU-hours (A100)
- **Optimal 轨**：每模型 × 3 条件 (top/随机/shuffled) × 5 heads × IPO 微调（rank-1 LoRA, 100 偏好对, 2 epochs）≈ 30 GPU-hours
- **生成评估**：GPT-judge 调用 ≈ $50 API
- **总计**：~40 GPU-hours + $50，单张 80GB 卡 1 周内可完成，符合 IdeaSpark 默认 envelope（150 GPU-days 内）

*所有数据集（WinoBias, TruthfulQA-style 偏好对）与模型（Qwen2.5-0.5B, Llama-2-7b-chat）均为公开现有 artifact，无需自建标注。*

---

### Differentiation from Literature

- **vs Kim et al. 2503.02080**：他们止于 heuristic steerability；我们引入 optimal 对照与随机零分布，将验证从“存在性”提升到“特权性”
- **vs Wang & Veitch 2502.11447**：他们聚焦 truthfulness 且仅否定定位；我们将其方法移植到偏见领域并**增加 stability 维度**，检验 stability 是否预测 optimal PrivGap（正/负结果皆有理论价值）
- **vs Li et al. 2023 (ITI)**：ITI 用 mass-mean shift 作为方向；我们证明该启发式在偏见任务中的次优性并给出最优上界

---

### Why This Composition?

- **controlled_diagnostic_design** 是 ICLR Corpus 中处理“测量即混淆”问题的最强模式（86 篇 Oral），匹配本任务“高探针准确率可能是名字记忆/confound”的本质
- **assumption_audit_and_pivot** (181 篇) 直击“编辑成功=定位”这一全领域隐含假设，Wang & Veitch 已示范 audit 价值，我们将其 pivot 到偏见方向

两者链式：先审计假设（为什么 heuristic 不可信），再设计诊断（如何受控地证伪）。

---

### Deliverables for Arron

1. **定义文档** `DEFINITIONS_Stability_Identifiability.md`（已完成）
2. **本 Idea Card**（中英双语，可直接转 PDF）
3. **升级后的代码** `run_bias_direction_v2.py`（待实现：加入 random/shuffled + head-level + PrivGap 计算）
4. **四图**：(a) Stability vs CV 带随机基线 (b) Cross-layer cosine 矩阵 (c) Heuristic vs Optimal 的 $\Delta M$ 分布 (d) Stability-PrivGap 相关性

---

### Validator Checklist (IdeaSpark 自检)

- [x] 每个 claim 有可追溯文献或被标为 model-supplied
- [x] 方法参数皆具名（B=20, ρ=0.8, K=16/32, α∈[-3,3], rank=1）
- [x] 数据/模型皆为现有 artifact
- [x] Falsification 含负对照且下游指标回落可判定
- [x] Compute budget 具名且在默认 envelope 内

