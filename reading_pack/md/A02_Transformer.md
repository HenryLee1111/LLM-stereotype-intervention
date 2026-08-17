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
