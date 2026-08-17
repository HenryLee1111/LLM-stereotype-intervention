# 稳定与可识别的偏见方向 — 课本
## Stable and Identifiable Bias Directions in LLMs: From Linear Representation to Privileged Causality
### 为《Toward Stable and Identifiable Bias Directions》定制的精读课本

> **使用说明：** 本课本不是论文原文的复制，而是按 Arron 要求的 `先定义再做研究` 和 ResearchStudio 的 `controlled_diagnostic_design × assumption_audit_and_pivot` 逻辑，将 13 篇核心文献重写为一本可直接研读的教材。每章末尾标注 **原文页码**，你应对照 arXiv 原文精读对应页，本课本帮你搭梯子。
> 全书约 18,000 字，10天读完，每天一章。

---

## 第0章 导航：我们要回答什么

**核心问题（Research Question）：**
> LLM 里那条“性别-职业偏见轴” `θ`，到底是模型里一条特权的、因果有效的旋钮，还是我们用次优方法时看到的、随机方向也能复现的幻觉？

**两条公理 vs 一个审判：**
- **公理1 Kim 2503.02080：** 政治立场是线性的，能被探针找到，能被 `x+ασθ` steer。
- **公理2 Park LRH：** 高层概念天生就是线性的。
- **审判 Wang 2502.11447：** 别信编辑能成功，随机位置的最优编辑和你挑的位置一样好。

**本书结构：**
```
Ch1 线代地基 → Ch2 Transformer解剖 → Ch3 探针 → Ch4 LRH公理 → Ch5 Kim的抄法 → Ch6 Wang的打脸 → Ch7 偏见测量 → Ch8 诊断设计 → Ch9 统计检验
```

---

## 第1章 线性代数地基：方向、投影、余弦

### 1.1 为什么偏见是一条线

我们说“偏见方向” `θ ∈ ℝ^D, ||θ||=1`，意思是：把任意激活 `x` 投影到 `θ` 上

```
score = θᵀx = ||x||·cos(x,θ)
```

越大越“女性语境”，越小越“男性语境”。这就是 Park 所说的 `sparse linear combination Σ c_i·f_i` 中，`f_i=θ`。

**关键操作1：余弦相似度**

```
cos(θ₁,θ₂) = ⟨θ₁,θ₂⟩ / (||θ₁||·||θ₂||) ∈ [-1,1]
```

- `1` 完全同向，`0` 正交，`-1` 反向。
- 你的 `Stab_boot = E[cos(θ^i,θ^j)]` 就是 20次 bootstrap 方向的平均两两余弦。

**关键操作2：单位化与投影**

探针学到的 `w` 要单位化 `θ = w/||w||`，否则 `ασθ` 的 `α` 失去意义。投影 `P_θ(x)= (θᵀx)θ`。

**关键操作3：秩1更新（为 Ch6 准备）**

LoRA `W + ΔW, ΔW = b·aᵀ` 是秩1矩阵。Wang 的 `W + W·b·aᵀ` 也是秩1，但 `W·b` 让更新落在 `W` 的列空间，才能定位到 head。

> **原文页码：** Elhage p.1-6，Park p.4-6 公式(2)

### 1.2 练习

- 证明：若 `θ₁,θ₂` 各向同性随机，则 `E[cos]=0, Var≈1/D`。所以 `D=896` 时随机余弦 ~0.03，你的 0.979 远超随机。
- 手算：`μ⁺=[1,0], μ⁻=[-1,0]`，`θ=μ⁺-μ⁻=[2,0]` 单位化后 `θ=[1,0]`，它就是 ITI 的 mass-mean 方向。

---

## 第2章 Transformer 解剖：Residual Stream 与 Attention Head

### 2.1 一句话结构

```
r₀ = Embed(tokens)
for ℓ=1..L:
  o_{ℓ,h} = Attn_{ℓ,h}(r_{ℓ-1})  ∈ ℝ^d      // 每个 head 的输出
  o_ℓ = concat_h o_{ℓ,h} ∈ ℝ^{D·H}
  r_{ℓ} = r_{ℓ-1} + W_ℓ·o_ℓ + MLP(...)    // Residual Stream 累加
r_L → Unembed → logits
```

- `r_ℓ` 是 **Residual Stream**，所有 head+MLP 的累加和，你的 MVP 探的就是它（`features[:,ℓ,:]`）。
- `x_{ℓ,h}=o_{ℓ,h}` 是 **Attention Head 输出**，Kim 探的是它（1024个探针）。

**为什么结果不同？** `r_ℓ` 是叠加，信息冗余，最后一层最易分；`x_{ℓ,h}` 是细粒度，middle layer 最有语义（Kim Fig.2）。你的 `best layer 24` vs Kim `layer 16` 正说明 **分布式 vs 定位** 的分歧。

> **原文页码：** Elhage p.12-14 Fig.1；Kim Sec.2 公式(1)-(3) p.2-4

### 2.2 必看图

- **Elhage Fig.1：** 横向 residual 箭头，纵向 head 分支。记住 `W_{ℓ,h}` 是 `D×d` 的列块，`W_ℓ = [W_{ℓ,1}...W_{ℓ,H}]`。

---

## 第3章 探针：从相关到因果的陷阱

### 3.1 探针三步

1.  **收集：** `D={(xⁱ,yⁱ)}`，`y=0` male, `1` female。
2.  **拟合：** `min Σ CE(y,σ(θᵀx)) + λ||θ||²`，`θ = coef_`
3.  **评估：** 准确率 + 交叉验证。

### 3.2 控制任务（Hewitt & Liang 的警告）

> 探针准，≠ 模型真的用这个信息。

检验：把 `y` 打乱为 `ỹ` 再训，若探针仍准，说明探针在记数据集噪音。你的 `shuffled null` 就来自此：**打乱后 Stab 应 <0.3**，否则你的 0.979 是过拟合。

### 3.3 线性 vs 非线性

Nanda 的 Othello 例子：非线性探针 1.7% 错，线性 20% 错，说明线性假设可能错。Park 通过 `non-linear probe ≤ linear probe` 来检验 LRH。

> **原文页码：** Hewitt p.4-5 Control Task；Nanda Othello-GPT 全篇

---

## 第4章 线性表征假说 LRH：为什么会线性

### 4.1 Park 的形式化

**假设：** 概念 `c` 存在方向 `f_c`，使得
```
x = Σ c_i·f_i + noise,  c_i 稀疏
p(next token | x) ∝ exp( u_vᵀ·r_L )
```
且 `f_c` 可通过 `c` 的 counterfactual pair（如 `James/Mary`）恢复。

**起源解释（直觉版）：**
1.  **Log-Odds：** 训练目标 `softmax CE` 逼迫模型让 `logit` 差等于真实 log-odds，而 log-odds 是线性的。
2.  **梯度下降隐式偏置：** GD 会让相似概念的 `f` 对齐，正交概念分离（Park Theorem 1-2）。

### 4.2 可证伪条件

Park 给出检验：若 `linear probe ≈ non-linear probe` 且 `steering` 有效，则 LRH 成立；否则是 superposition（多个概念挤一个方向）。

> **原文页码：** Park p.1-3, p.4-6 Theorem；Gurnee p.2-4 探针方法

---

## 第5章 Kim 的抄法：政治立场的线性探针与 Steer

### 5.1 数据与探针

- **数据：** 议员发言 prompt → DW-NOMINATE 分数 `y`（-1自由, +1保守）。
- **探针：** 对每个 `x_{ℓ,h}` 做 **Ridge 回归** `min Σ(y-θᵀx)²+λ||θ||²`，按 Spearman 排序选 Top-K=32 个 head。

### 5.2 关键结果

- **Fig.2：** Top Head 集中在 **Middle Layer (ℓ≈16)**，预测相关性 `ρ≈0.7`。
- **公式(8)：** `x^{(α)}_{ℓ,h}=x_{ℓ,h}+α·σ_{ℓ,h}·θ_{ℓ,h}`，`σ` 是该 head 激活标准差，`α∈[-3,3]` 负偏自由，正偏保守。
- **Fig.4：** `α` 越大越保守，但生成长度也变长（副作用）。

### 5.3 隐含假设

Kim 假设 `θ` 是特权的，且 `α·σ·θ` 是最优干预。这正是 Wang 要打的。

> **原文页码：** Kim p.2-4 Sec.2, p.4-8 Sec.3, p.11-14 Sec.6 公式(8) Fig.4

---

## 第6章 Wang 的打脸：最优编辑与随机零分布

### 6.1 ITI 的次优

ITI 的 `θ=μ⁺-μ⁻` 是 heuristic，Wang 证明它远非最优。

### 6.2 最优的构造（全书最难，必须手推）

**目标：** 找到在指定 head 上，能让 TruthfulQA 最真的 **最优局部编辑**。

**技巧：把编辑变成对齐问题**

TruthfulQA 的 `(x, y⁺, y⁻)` 可重构成偏好对，用 **IPO**：
```
max Σ [ log(π_φ(y⁺|x)/π₀(y⁺|x) / π_φ(y⁻|x)/π₀(y⁻|x)) - τ⁻¹/2 ]²
```

**重参数化 LoRA：**

1.  **朴素 LoRA：** `r_{LoRA}= r+ (W+ b aᵀ)o = r_orig + ⟨a,o⟩·b`
    -  `b` 是编辑，但 `b` 不在 `W` 空间，无法定位到 head。

2.  **Wang 的重参数化：** `b → W·b`
    ```
    r_{reparam}= r_orig + ⟨a,o⟩·W·b = r_orig + ⟨a,o⟩· Σ_h W_{ℓ,h}·b_{ℓ,h}
    ```
    - 此时 `b_{ℓ,h}` 就是 `θ_{ℓ,h}`，`⟨a,o⟩` 就是自适应 `α`。
    - **限制 `b_{ℓ,h}=0` 对非目标 head**，就实现了“只在指定 head 上最优”。

**实验设计（四步）：**

1.  Top-16 最优 ≈ Full-model 最优 (Fig.2) —— 看似最强证据
2.  **Random-16 最优 ≈ Top-16 最优 (Fig.3)** —— 证据崩塌
3.  单随机 Head 最优 ≈ Full-model，且有**多个**这样的 head (Fig.4) —— 无唯一性
4.  结论：`heuristic 下 Top>>Random` 是假象，`optimal 下 Top≈Random` 才是真相。

> **原文页码：** Wang p.3-4 Sec.2, **p.6-9 Sec.4-6 Fig.2-4 公式(7)(8)**，必逐行推

### 6.3 必看图

- **Fig.1：** Heuristic 下 Top 的 Info*Truth 远超 Random（p=1.6e-8）——假阳性
- **Fig.3：** Optimal 下 Random 分布紧贴 Top——真阴性
- **Fig.4a：** 24个单 head 最优分数直方图，5个 ≈Top-16

---

## 第7章 偏见测量：从行为到表征

### 7.1 行为测量

- **WinoBias：** `The nurse said she...` 指代消解，测模型选 `he/she`。
- **OccuGender 五条军规（必背）：** 1) 无模板混淆 2) BLS客观标签 3) 小预测空间（预测性别而非职业）4) 显式/隐式 5) 非二元。你的模板 `At work, {name} the {occupation}...` 需做 `leave-one-template-out` 检验第1条。

### 7.2 表征测量

行为准 ≠ 表征线性。需同时做：
- **探针准确率** + **生成 logit 差 `ΔM`** + **擦除实验 LEACE**（擦掉 `θ` 看准确率掉多少）

> **原文页码：** OccuGender p.2-3 Desiderata, p.4 Fig.2；Zhao WinoBias p.2

---

## 第8章 诊断设计：四控×双轨

### 8.1 受控诊断三要素（ResearchStudio C03）

1.  **同源对比对：** 同一职业，`James vs Mary`，仅性别变。
2.  **单轴变化：** 每次只换模板/采样/优化器其一。
3.  **零分布：** `20×random` + `shuffled` + `full-model上界`。

### 8.2 假设审计（C01）

审计 `编辑成功 ⇒ 定位`：若审计失败（Wang Fig.3），则 Kim Fig.2 的结论需收回。

### 8.3 你的 PrivGap 诊断流程图

```
D_true → θ_true ─┬─→ Heuristic ΔM_true ─┐
D_shuf → θ_shuf ─┼─→ Heuristic ΔM_shuf  ├─→ PrivGap_heu
20×rand → θ_rand─┴─→ Heuristic ΔM_rand ┘
                  └─→ Optimal ΔM_* (LoRA-IPO) ─→ PrivGap_opt
```

---

## 第9章 统计检验：Bootstrap 与 Permutation

### 9.1 Bootstrap Stability

```
for b=1..20:
  sample 80% male + 80% female with replacement
  θ^b = probe(X_b, y_b)
Stab = mean_{i≠j} cos(θ^i,θ^j)
```

- 分层采样防失衡（male/female 分别采）。
- 你的 `0.979` 是 `mean off-diag`，需同时报 `std` 和 `shuffled null`。

### 9.2 Permutation 检验 PrivGap

```
null = { PrivGap_rand^1 ... PrivGap_rand^20 }
p = (# null ≥ PrivGap_true +1)/(20+1)
```

- `p<0.01` 才算特权。
- 还要报 `corr(Stab, PrivGap)`，检验 Stability 是否预测 Identifiability。

> **原文页码：** Wasserman Ch.8 Bootstrap；Efron 1979

---

## 附录：如何使用本课本

1.  每章对照原文页码精读，先看本课本搭梯子，再读原文验证。
2.  每章末做练习，能白板推导再下一章。
3.  读完 Ch6 必须能回答：`为什么 Wang Fig.3 能证伪 Kim Fig.2？` 答案：`Fig.2 heuristic假赢，Fig.3 optimal真比`。
4.  全书读完，`DEFINITIONS.md` 和 `IDEA_CARD.md` 的每行你都能指出处。

> 全文完。去 `reading_pack/` 按 `A1_A2...` 重命名 PDF，开读吧。

