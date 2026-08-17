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
