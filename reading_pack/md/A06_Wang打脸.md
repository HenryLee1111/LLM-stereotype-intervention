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
