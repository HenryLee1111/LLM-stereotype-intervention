# Stability 与 Identifiability 的形式化定义
## —— 为《Toward Stable and Identifiable Bias Directions》建立可证伪基础
### 回应 Arron Schein 要求：先定义，再做研究

> 对应 ResearchStudio IdeaSpark **Phase 1: Bottleneck identification** 的前置工作：若核心概念未形式化，后续方法无法被评审证伪。

---

## 0. 符号设定

设 LLM 在处理 prompt $w$ 时，在位置 $p$（residual stream 层 $\ell$ 或 attention head $(\ell,h)$）产生激活 $x_{\ell} \in \mathbb{R}^D$ 或 $x_{\ell,h} \in \mathbb{R}^d$。

候选偏见方向为单位向量 $\theta \in \mathbb{S}^{D-1}$（或 $\mathbb{S}^{d-1}$）。

通过在标签 $y \in \{0,1\}$（如 male/female 上下文）上训练线性探针得到：

$$\hat{\theta} = \arg\min_{\theta} \sum_i \mathcal{L}(y^{(i)}, \sigma(\theta^\top x^{(i)})) + \lambda \|\theta\|_2^2,\quad \theta \leftarrow \theta / \|\theta\|$$

---

## 1. Stability（稳定性）：统计可重复性

### 1.1 直觉
若 $\theta$ 是模型内在结构的真实反映，则用不同数据子集重新估计时应得到几乎相同的方向；若每次重采样都得到不同方向，则它是数据/提示的偶然 artifact。

### 1.2 形式定义

**定义 1 (Bootstrap Stability)**：给定数据集 $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$，对 $b=1\dots B$ 做分层 bootstrap（male/female 分别有放回采样比例 $\rho=0.8$），得到 $\hat{\theta}^{(b)}$。则

$$ \text{Stab}_{\text{boot}}(\hat{\theta}) = \mathbb{E}_{i \neq j}[\cos(\hat{\theta}^{(i)}, \hat{\theta}^{(j)})] = \frac{1}{B(B-1)}\sum_{i\neq j} \frac{\langle \hat{\theta}^{(i)}, \hat{\theta}^{(j)}\rangle}{\|\hat{\theta}^{(i)}\|\|\hat{\theta}^{(j)}\|} \in [-1,1] $$

- $\text{Stab}=1$：所有重采样方向完全一致（理想稳定）
- $\text{Stab}\approx 0$：方向随机
- 阈值（经验，来自 MVP）：$>0.90$ 稳定，$<0.80$ 不稳定

**定义 2 (Cross-Template Stability)**：留一模板（leave-one-template-out）训练 $\hat{\theta}_{-t}$，在留出模板上测试：

$$ \text{Stab}_{\text{template}} = \min_{t} \cos(\hat{\theta}_{-t}, \hat{\theta}_{\text{full}}) $$

衡量方向是否依赖特定表述。

**定义 3 (Cross-Layer Consistency)**：

$$ C_{\ell,\ell'} = \cos(\hat{\theta}_\ell, P_{\ell\to\ell'}\hat{\theta}_{\ell'}) $$

其中 $P$ 为维度对齐投影（若维度相同则恒等）。若多层一致性高，说明编码分布式。

### 1.3 性质
- **必要非充分**：高 Stability 是可信方向的必要条件，但不能证明因果有效性（Wang & Veitch 2025 的核心批判）。
- **可证伪**：若打乱标签 $y \to \tilde{y}$ 后 $\text{Stab}_{\text{boot}} > 0.5$，则指标失效（应 $<0.2$）。

---

## 2. Identifiability（可识别性）：因果特权性

### 2.1 为什么 Stability 不够？
Wang & Veitch (arXiv:2502.11447) 证明：对 TruthfulQA 任务，**最优干预（IPO + 局部 LoRA）**在随机 16 heads 上与在 probe 选出的 top-16 heads 上效果相当，且单随机 head 即可达到 full-model finetune 的上界。结论：

> *Heuristic editing success provides little to no evidence for localization.*

类比到偏见方向：即使你的 $\hat{\theta}$ 的 $\text{Stab}=0.98$ 且能用 $x+\alpha\sigma\hat{\theta}$ steer，也不能说它是**特权的（privileged）**偏见轴——随机方向经最优优化后可能同样能 steer。

### 2.2 形式定义（反事实）

设因果效应为干预后偏见度量的变化。选一偏见度量 $M$（如中性职业句生成中代词 $P(\text{she})/P(\text{he})$ 的 logit 差，或 GPT-judge 偏见分数）：

$$ \Delta M(\theta, \alpha) = M(x + \alpha\sigma\theta) - M(x) $$

设 $\mathcal{T}$ 为候选方向集合，$\mathcal{R}$ 为随机基线方向集合（20 个各向同性随机单位向量，或随机 head 的 probe 方向）。

**定义 4 (Privileged Identifiability)**：

$\theta^*$ 是 **$\epsilon$-identifiable** 当且仅当在最优干预强度 $\alpha^*$ 下：

1. **存在性**：$|\Delta M(\theta^*, \alpha^*)| > \tau$ （阈值，如 Cohen's $d>0.8$）
2. **特权性（Privilegedness）**：
   $$ \text{PrivGap}(\theta^*) = |\Delta M(\theta^*, \alpha^*)| - \max_{\theta_r \in \mathcal{R}} |\Delta M(\theta_r, \alpha^*_r)| > \epsilon $$
   即真实方向的最优效果显著超过最佳随机方向。
3. **效率**：
   $$ \frac{|\Delta M(\theta^*, \alpha^*)|}{|\Delta M_{\text{full}}|} > 0.8 $$
   其中 $\Delta M_{\text{full}}$ 为 full-model（所有层）最优干预的上界。

若条件 2 不满足，则 $\theta^*$ 是 **discoverable but not identifiable**——可被发现但非特权，编码是分布式的。

**定义 5 (Heuristic vs Optimal Identifiability)**：

- **Heuristic Identifiability**：用固定启发式 $x+\alpha\sigma\theta$ 时的 PrivGap
- **Optimal Identifiability**：用 IPO 局部 LoRA 搜到的最优 $b$ 向量（Wang & Veitch Sec.4 重参数化 $W^l b$）时的 PrivGap

Wang & Veitch 发现：Truthfulness 在 heuristic 下 PrivGap 大（似乎可识别），在 optimal 下 PrivGap $\approx 0$（不可识别）。我们的研究需同时报告两者。

### 2.3 操作化检验（Falsification）

- **Shuffled-label null**：$\tilde{\theta}$ 的 $\text{Stab}$ 应 $<0.3$ 且 $\Delta M(\tilde{\theta}) \approx 0$
- **Random-direction null**：$\max_{\theta_r} |\Delta M(\theta_r)|$ 的分布（bootstrap 100 次）作为零分布，真 $\theta^*$ 的 $p$ 值应 $<0.01$
- **Single-head multiplicity**：若存在 $\geq 2$ 个不重叠单头各自 PrivGap $>0$ 且彼此 $\cos <0.3$，则无单一特权位置

---

## 3. 两者的关系图

```
Discoverable (probe accuracy > chance)
    ↓
Stable (Stab_boot > 0.9, cross-template > 0.8)  ← 你已完成
    ↓  (必要非充分)
Identifiable (PrivGap > ε under OPTIMAL intervention)  ← 下一步核心
    ↓
Causally Valid (stability 预测 PrivGap, 跨模型复现)
```

**一句话区分**：
- **Stability** 回答：*这个方向是可重复估计的吗？*（统计问题）
- **Identifiability** 回答：*这个方向是因果上特权的吗？随机方向不能替代它吗？*（因果/反事实问题）

---

## 4. 与 ResearchStudio IdeaSpark 模式的对接

| 定义要素 | 对应 IdeaSpark Pattern | 作用 |
|---|---|---|
| Stability 的 bootstrap / cross-template | `controlled_diagnostic_design` (设计混淆隔离诊断) | 隔离"数据采样/模板措辞"混淆 |
| Identifiability 的 random vs top 对比 | `assumption_audit_and_pivot` (审计并转轴假设) | 审计"editing success = localization"的隐含假设并转轴 |
| PrivGap 与 full-model 上界 | `characterize_limit_then_surpass` (刻画极限再超越) | 形式化 heuristic 的极限，再用 optimal 检验是否可超越 |

> 下一步的 Idea Card 将基于此定义，提出一个同时满足两个模式的**可证伪实验**。

---

## 5. 给 Arron 的定义确认句（可直接放入邮件）

> *Following your suggestion, we now formally separate **Stability** (bootstrap cosine >0.9, cross-template consistency) as statistical replicability from **Identifiability** (PrivGap under optimal localized intervention, vs random-direction null) as causal privilegedness. Stability is necessary but not sufficient; identifiability requires that no random direction matches the causal effect under optimal optimization (Wang & Veitch 2025).*

