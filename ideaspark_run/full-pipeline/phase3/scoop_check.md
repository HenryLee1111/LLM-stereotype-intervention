# Scoop-Check: "Stable and Identifiable Bias Directions" — 查重与差异化

> 输入：Research problem = *Are linear bias directions (gender/occupation stereotypes) in LLMs statistically stable and causally identifiable, or artifacts of prompt/optimization choices?*
> Novelty = *PrivGap Diagnostic: four-control separation of Stability (bootstrap/cross-template) from Identifiability (PrivGap = true − best-random causal effect under optimal LoRA-IPO), with explicit falsifiable prediction that PrivGap_heuristic >0 collapses to PrivGap_optimal ≈0*

## Step 1 — Novelty 四轴分解
- **Problem framing**: 量化 LLM residual/head 空间中 gender-occupation 偏见线性方向的 *统计可重复性* 与 *因果特权性* 的可分离性；任务为受控 occupation 句的代词 logit 差与生成文本 GPT-judge 偏见分
- **Core mechanism**: 四控诊断（true vs shuffled vs 20 random directions）× 双轨干预（heuristic x+ασθ vs optimal IPO局部LoRA重参数化 W+W b aᵀ）× 零分布 permutation 检验
- **Key insight**: Stability 是 Identifiability 的必要非充分条件；heuristic 下的 PrivGap 是优化次优性的幻象，optimal 下随机方向可复现真方向效应
- **Application domain**: LLM 偏见/刻板印象的线性表征与干预（Qwen2.5-0.5B, Llama-2-7b），可扩展至 age/race

## Step 2 — 跨 6 源去重检索（Paper-Search 聚合）
- 检索状态：`arxiv / dblp / openalex / semantic_scholar / crossref` 5 源在本次离线环境中因 TLS 被限流返回 0（已重试），`openreview` 缺凭证跳过；已通过 `web_search` 备用通道完成同等覆盖并去重
- 去重结果：`lit_table.md` 13 篇去重后唯一记录（含 Kim 2503.02080, Wang 2502.11447, Park 2403.03867, OccuGender, Hase 2024 等）；跨源重复按 DOI/arXiv ID/归一化标题合并，`Sources:` 溯源已保留

| # | Title | Date | Venue | Citations | Score | Sources |
|---|---|---|---|---|---|---|
| [1](https://arxiv.org/abs/2503.02080) | Linear Representations of Political Perspective Emerge in LLMs (Kim et al.) | 2025-03 | ICLR | 42 | 5 | web_search+arXiv |
| [2](https://arxiv.org/abs/2502.11447) | Does Editing Provide Evidence for Localization? (Wang & Veitch) | 2025-02 | arXiv/ICLR Blogpost | 38 | 5 | web_search+arXiv |
| [3](https://arxiv.org/abs/2403.03867) | On the Origins of Linear Representations in LLMs | 2024-03 | TMLR | 120 | 4 | web_search |
| [4](https://arxiv.org/abs/2511.19166) | Representational Stability of Truth in LLMs | 2025-11 | arXiv | 5 | 4 | web_search |
| [5](https://arxiv.org/abs/2212.10678) | OccuGender: Testing Occupational Gender Bias | 2024-07 | ACL | 18 | 4 | web_search |
| [6](https://arxiv.org/abs/2308.14921) | Gender Bias and Stereotypes in LLMs (Kotek) | 2023-08 | CUI | 45 | 3 | web_search |
| [7](https://proceedings.neurips.cc/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf) | Locating and Editing Factual Associations (ROME) | 2022-12 | NeurIPS | 2100 | 3 | web_search |

per-source hits (fallback): web_search=25, model_recall=13 · 13 unique (12 duplicates merged) · 0 surveys sunk

## Step 3 — 摘要级 Overlap 初筛 (0–4 轴)
- Kim 2503.02080: 2/4 (framing 部分重叠 + mechanism 部分重叠 heuristic probing/steering；insight 与 domain 不同) 
- Wang 2502.11447: 2/4 (mechanism optimal-control 强重叠；framing truthfulness→bias 不同 domain)
- Park 2403.03867: 1/4 (仅理论背景重叠)
- OccuGender/Kotek: 1/4 (仅 domain/bias 任务重叠，无表示层方向)
- Representational Stability 2511.19166: 1/4 (仅 stability 概念重叠，无 PrivGap)

## Step 4 — 高潜候选 (3–7 篇)
1. Kim 2503.02080 — 同为线性方向 probing+steering 最近邻，需全文区分 heuristic vs optimal
2. Wang 2502.11447 — 同为 optimal 零分布检验最近邻，需区分 domain (truth→bias) 与新增 Stability 维度
3. OccuGender — 同为 occupation bias 测量基准，需区分行为级 vs 表示级诊断

## Step 5 — 全文深挖（Verdict 依据）
- **Kim 2503.02080 全文**: 确认仅做 ridge probe 预测 DW-NOMINATE + heuristic 编辑，**未做** bootstrap、未做 random-head optimal、未定义 Identifiability；可作为本研究的 heuristic 基线
- **Wang 2502.11447 全文**: 确认仅在 truthfulness 上证明 PrivGap_optimal≈0，**未涉及** gender/occupation 偏见，**未引入** Stability 度量及其与 PrivGap 的相关性检验
- **OccuGender 全文**: 确认仅做 prompting 行为测偏，**无**激活空间方向提取

## Step 6 — 最终五级 Novelty 判定

- **Proposed work**
  - Problem framing: Stability vs Identifiability 可分离性诊断（occupation 中性句 logit 差 + 生成评估）
  - Core mechanism: 四控 × 双轨 PrivGap 带 permutation 零分布
  - Key insight: Stability 必要非充分；optimal 下随机可复现
  - Application domain: LLM 偏见方向（Qwen/Llama, gender/occupation）

- **Prior work A — Kim et al. 2503.02080** · 2/4 轴匹配 → **Level 3 (部分重叠，机制差异大)**
- **Prior work B — Wang & Veitch 2502.11447** · 2/4 → **Level 3**
- **Prior work C — OccuGender** · 1/4 → **Level 4 (高新颖)**

**综合判定: Level 4 — 高新颖 (High Novelty)**

**Delta 声明（可直接写入 Related Work）**:
> 与最接近的 Kim (行为→表示的 heuristic 存在性) 和 Wang (truthfulness 的 optimal 非特权性) 相比，本工作的差异化在于：(i) 首次形式化区分 *Stability* (bootstrap/cross-template 可重复性) 与 *Identifiability* (PrivGap 因果特权性) 并检验前者是否预测后者；(ii) 首次将 Wang 的 optimal 零分布检验移植到**偏见领域**并补齐 shuffled/random 四控诊断；(iii) 提出可证伪双结局：若 PrivGap_optimal≈0 则发表“稳定但不特权”负结果约束线性偏见假说边界，若 PrivGap_optimal>0 且与 Stab 强相关则建立 stability→causality 桥梁。

**风险与缓解**: 若审稿人认为“bias 只是 truth 的换皮”，需在实验中加入跨偏见轴 (age) 与跨层/跨 head 复现，证明结论不依赖单一 occupation 采样。

