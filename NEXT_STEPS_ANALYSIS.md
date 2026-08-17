# 下一步研究规划：从「发现方向」到「可识别且因果有效的方向」

> 基于 2503.02080 (Kim, Evans & Schein, ICLR 2025) + 你的提案《Toward Stable and Identifiable Bias Directions》+ Arron Schein 推荐的 feedback 2502.11447 (Wang & Veitch, UChicago)

---

## 1. 先把三角关系捋清楚

| 论文 | 核心主张 | 方法 | 隐含假设 |
|---|---|---|---|
| **2503.02080 你的 based-on** | 政治立场在 LLM 的 attention head 激活空间中是线性可编码的；可以用 ridge probe 找到，并可通过 `x + α σ θ` 线性干预来 steer | 对 32×32=1024 个 attention head 逐个做 ridge regression 预测 DW-NOMINATE 分数；选最相关的 K=32 个头做集成预测和干预 | `θ` 是真实的、有特权的（privileged）概念方向；找到它就找到了模型的「意识形态轴」 |
| **你的提案** | 这些方向真的稳定可识别吗？还是 prompt/采样/layer 的 artifact？引入 **Bootstrap Stability = E[cos(θ^i, θ^j)]** 作为可信度判据 | residual stream 逐层 logistic probe + bootstrap + CV accuracy + 选 best layer；干预验证 `x_new = x + α σ θ` | 稳定 ≈ 可信；高 stability 的方向适合做干预 |
| **2502.11447 Arron 给你的 feedback** | **Editing does NOT provide evidence for localization** — 即使在某个位置做最优编辑能成功 steer，也不能证明那个位置就是概念的「居所」 | 把 alignment 目标 (IPO) + 局部 LoRA 重参数化，搜「在指定 head 处的最优干预」；对比 localized top-16 heads vs 随机 16 heads vs 单个随机 head | 挑战 2503.02080 / ITI 的整个验证逻辑：随机位置的最优编辑 *同样* 能达到 full-model finetune 的效果；有多个单头都能做到 |

**Arron 为什么给你这篇？** Wang & Veitch 一作是 UChicago 统计系、与 Schein 同系，这篇本质是 Schein 自己工作的 **自我批判/方法论升级版**。他是在告诉你：

> “我很喜欢你把问题从 existence 拉到 stability/identifiability，但你目前的 stability 定义，恰好撞上了我们领域现在最大的坑——**如果不用随机基线和最优干预对照，任何‘稳定+可 steer’的结论都是没有说服力的。**”

换句话说，他不是在否定你，而是在邀请你把提案从“一个漂亮的诊断指标”升级为“对整个线性表征假设的严格压力测试”。

---

## 2. 你当前 MVP 的 3 个“甜蜜陷阱”

你的 `run_bias_direction_mvp.py` 在 Qwen2.5-0.5B 上跑出了：

- best layer = 24, stability = 0.979, CV accuracy = 100%
- 所有 top5 层 stability > 0.97, 全部 CV 100%

这看起来是“重大成功”，但用 2502.11447 的视角看，恰恰暴露 3 个问题：

### 陷阱 1：任务过于简单 → 100% 准确率失去信息量
用 `James vs Mary + {occupation}` 这种显式名字信号，任何一层都能完美线性分离。100% CV 说明 probe 学到的是“名字 token 本身”，而非更微妙的 occupation stereotype。这导致：
- `stability = 0.979` 可能只是“名字子空间很稳定”，而非“偏见方向稳定”
- 与 occupation 投影的对应（nurse +5.5, architect -7.1）虽然有趣，但你现在只看了 1 个 model 的 1 个 theta，没有随机 baseline，很难说这是偏见还是噪声

**诊断**：需要更难的泛化测试——held-out occupation / held-out template / 无名字中性句。

### 陷阱 2：粒度不对——你探的是 residual stream 层，2503.02080 和 2502.11447 探的是 attention head
2503.02080 的关键发现是“最具预测力的 head 集中在 middle layers (≈layer 16)”，而你在 Qwen 上发现 best layer 是最后一层 24。这不一定矛盾，但说明：
- residual stream 是所有 heads + MLP 的叠加，本身就更容易线性可分
- 如果一个方向在 residual stream 各处都可分，这恰恰支持 Wang & Veitch 的“distributed, no privileged location”观点——你的高 stability 可能是在证明 **无定位性**，而不是有

### 陷阱 3：Heuristic 干预 vs Optimal 干预
你提案中的干预形式 `x_new = x + α σ θ` 正是 2502.11447 证明为 **次优** 的 ITI 启发式（图2显示 IPO 学到的最优干预远优于它）。Arron 一定会问：
- 你的稳定方向，用 heuristic 能 steer 吗？用最优的 LoRA-IPO 能 steer 吗？
- 随机方向用最优方法也能 steer 吗？如果是，那你的方向有何特殊？

**一句话总结**：你现在证明了“可发现且稳定”，但还没证明“因果有效且特权”。

---

## 3. 下一步：四组必须补的实验（按优先级）

### 实验组 A：Negative Controls —— 回答“你的方向比随机好在哪里？”

这是回应 2502.11447 最直接的实验，也是审稿人一定会问的。

1.  **Random Direction Baseline**：在同一层随机采样 20 个单位向量 `θ_rand`，同样做 occupation 投影，看分布。你的真实 θ 的投影方差 / 极差是否显著超出随机？
2.  **Random Head Baseline**：复刻 2503.02080 的 head-level probing，但用随机 16 heads 对比 top-16 heads，分别做 heuristic 干预的 steering 效果（需要写生成评估脚本）。
3.  **Shuffled Label Baseline**：把 y (male/female) 随机打乱后重新训练 probe，看 stability 是否掉到 ~0（预期 <0.2）。如果打乱后仍高，说明你的 stability 指标有 bug。
4.  **Cross-template Generalization**：留一模板（leave-one-template-out）训练，在未见过模板上测 accuracy。真·概念方向应该跨模板，而名字记忆不会。

> **产出**：一张图 —— X轴不同基线，Y轴 stability / CV / steering success rate，真方向显著高于随机才有说服力。

### 实验组 B：Optimal vs Heuristic —— 把 2502.11447 的方法移植到偏见任务

这是最能体现你“读懂 feedback”的一步。

- **Heuristic (你现在)**：`r_{l+1} = r_l + W^l(o + α θ)` 固定 α
- **Optimal (Wang & Veitch Sec 4)**：将 `W^l` 做 rank-1 LoRA 重参数化 `W^l + W^l b a^T`，只允许在选定 layer/head 上 `b ≠ 0`，用 IPO/D PO 目标在 TruthfulQA 风格的“偏见偏好对”上优化（例如 `{x: "The nurse said that", y^+: "she...", y^-: "he..."}` 或更复杂的 WinoBias 偏好对）

对比：
- Top-layer θ 的 heuristic steering 成功率
- Top-layer θ 的 optimal steering 成功率
- 随机 layer/head 的 optimal steering 成功率
- Full-model finetune (all layers) 的 optimal 上界

**预期复现 Wang & Veitch 的发现**：随机位置 + 最优优化也能接近 full-model 效果。这时你的 contribution 就从“我找到了最好的 layer”转变为“我量化了‘最好’与‘随机’的差距，并证明 stability 预测的是 heuristic 可用性而非最优可控性”——这是一个更诚实、也更深刻的故事。

### 实验组 C：Causal Validation —— Stability 是否预测 Steering？

这是把你的核心贡献从“诊断指标”升级为“因果判据”。

1.  计算每层 `stability × CV` 与该层 heuristic steering 在中性 occupation 上的 **效应量（effect size, 例如 |projection| 或生成文本的性别偏度变化）** 的相关性。
2.  如果高 stability 层确实 steer 效果更强、更一致（低方差），则你的指标有预测力；如果无关，则 stability 需要重新定义（例如需加 orthogonalization 或控制方差）。

建议引入 **双重差分** 评估：对同一 occupation，比较 `α=+2` vs `α=-2` 干预后下一 token 中代词 `he/she` 的 logit 差。

### 实验组 D：规模与泛化 —— 让结论不依赖 Qwen-0.5B CPU

- 至少补一个 7B 模型（Llama-2-7b-chat / Mistral-7b-instruct，与 2503.02080 一致），看 layer 曲线是否从“最后一层最稳”变为“middle layer 最稳”（复现原论文发现）。
- 扩展到其他偏见轴：age / race，或同为 gender 但用不同名字集（英文 vs 中文名字），检验方向的通用性。
- 加 **Cross-layer Consistency**：计算不同层 θ 之间的余弦相似度矩阵。如果各层 θ 彼此正交，说明是分布式编码，无单一方向。

---

## 4. 理论深化：从 Stability 到 Identifiability

Arron 是统计学家，他期待的不仅是实验，更是**形式化**。

建议在提案第 7 节增加：

1.  **区分三个概念**（可直接引用 Wang & Veitch 的框架）：
    - *Discoverability*：存在某个线性探针 loss < ε
    - *Stability*：跨 bootstrap 的方差小（你现在的）
    - *Identifiability / Privilegedness*：不存在另一个随机/正交方向能达到同等因果效果（你缺的）

2.  **提出反事实定义**：
    > θ 是 identifiable 的，当且仅当 `max_{θ' ∈ random} CausalEffect(θ') << CausalEffect(θ)` 且 `CausalEffect(θ) ≈ CausalEffect(optimal full model)`

3.  **连接文献**：除了 ITI 和 2503.02080，主动引用 `Park et al. 2024 (Linear Representation Hypothesis)` 和 `Ravichander et al.` 关于探针可解释性的批判，显示你意识到 probing ≠ understanding。

---

## 5. 具体执行路线图（4 周 MVP+）

**Week 1：补基线（不需 GPU）**
- [ ] 在现有 `run_bias_direction_mvp.py` 中加入 `shuffled_label` 和 `random_theta` 对照，画对比图
- [ ] 实现 leave-one-occupation-out / leave-one-template-out CV

**Week 2：Head-level 复现**
- [ ] 修改 hidden state 抽取，改为 `x_{ℓ,h}` (attention head output) 而非 residual；对 Qwen-0.5B 先跑（Qwen 的 head 数不同，需查 config）
- [ ] 复刻 2503.02080 的 Ridge + Spearman 排序，验证 middle-layer 现象是否存在于 gender 任务

**Week 3：Optimal Intervention 最小验证**
- [ ] 用 HuggingFace + PEFT 实现单层 LoRA-IPO 训练脚本（可先用 100 个 WinoBias 偏好对，CPU 可跑小模型）
- [ ] 对比 top-layer vs random-layer 的 optimal steering 效果（评估用规则：生成文本中 `he/she` 比例，或用 GPT-judge 偏见分数）

**Week 4：写作与套磁**
- [ ] 更新 proposal 图：将原有 `layer_stability.png` 升级为 4 子图：(a) stability vs CV (b) random vs true (c) cross-layer cos (d) steering effect vs stability
- [ ] 给 Arron 的 follow-up 邮件（见下）

---

## 6. 给 Arron Schein 的回复邮件草稿（可直接用）

> Subject: Re: Follow-up on stable bias directions — incorporating your feedback (Wang & Veitch 2502.11447)
>
> Dear Professor Schein,
>
> Thank you again for your feedback and for pointing me to Wang & Veitch (2025) “Does Editing Provide Evidence for Localization?”
>
> I have been reflecting on how it reframes my initial proposal on stable bias directions (building on Kim et al. ICLR 2025). My original focus was on *discoverability + bootstrap stability* as a reliability criterion for `θ` — e.g., my pilot on Qwen2.5-0.5B shows a bootstrap stability of 0.98 at the best layer. However, Wang & Veitch makes a compelling case that **heuristic editing success (x + α σ θ) provides little evidence for a privileged localization**, since optimal interventions at random heads can match full-model alignment.
>
> This has helped me clarify the next step: moving from *stability* to *identifiability + causal validity*. Concretely, I am now extending the pipeline to include:
>
> 1.  **Negative controls**: random-direction and shuffled-label baselines for every stability report, and leave-one-template-out generalization;
> 2.  **Head-level analysis** to directly compare residual-stream vs attention-head localization (as in your ICLR paper) and test whether the “middle-layer” pattern holds for gender stereotypes;
> 3.  **Optimal vs heuristic intervention comparison**: adapting the localized LoRA-IPO procedure from Wang & Veitch to measure the causal effect gap between the most stable `θ` and random `θ` under optimal optimization — to test if stability predicts causal efficacy.
>
> My hypothesis is that stability is necessary but not sufficient: stable directions may be *discoverable everywhere* (hence high cosine across bootstraps) yet not *privileged* under optimal control, which would suggest stereotype encoding is more distributed than a single linear direction.
>
> I would be grateful for your thoughts on whether this “stability → causal identifiability” framing addresses the core concern, and whether you see other critical baselines I should prioritize.
>
> Thank you again for your time and guidance.
>
> Best regards,
> Henry Lee

**为什么这样写？** 
- 点名你读懂了 2502.11447 的技术细节（LoRA 重参数化、random vs top）
- 承认自己原指标的局限，提出具体可执行升级（统计学家最看重 falsifiability）
- 把“高 stability”重新解释为可能支持对方观点（分布式），展现科学诚实而非 defensive

---

## 7. 一句话总结下一步

> **不要再只证明“我的方向很稳定”，要去证明“我的方向比随机方向在因果上更特殊，且这种特殊性恰好被 stability 所预测——如果证伪了，那本身就是一个发表级的发现”。**

这正是 Wang & Veitch 式的贡献：**用一个漂亮的负结果，约束了整个领域对“定位”的想象。**

---

### 附：需要我帮你直接改代码吗？

我可以基于你现有的 `run_bias_direction_mvp.py` 直接追加：
- `random_theta` / `shuffled_label` 分支
- `cross-layer cosine matrix` 热图
- `head-level` 抽取选项

告诉我你想先跑哪一组，我现在就改。
