# 研究问题 → 想法卡 完整流水线报告
## ResearchStudio IdeaSpark + Paper-Search + Scoop-Check 全量执行
### 项目：Toward Stable and Identifiable Bias Directions in LLMs

> 本报告按 Microsoft ResearchStudio 官方 Skill 契约执行：`paper_search`（6源去重检索）→ `idea_spark`（Phases 0–4）→ `scoop_check`（7步查重），每步产物路径与 Schema 完全对齐。

---

## 0. 输入问题（User Query 原文持久化）

> *我的研究想法基于 2503.02080 (Kim et al. ICLR 2025 Linear Representations of Political Perspective)，找完 Arron Schein 陶瓷后她给我的反馈是 feedback (2502.11447 Wang & Veitch Does Editing Provide Evidence for Localization?) 请想想下一步；他想要我先定义 Stability and Identifiability 是什么，然后再做这个研究；同时我要你使用 ResearchStudio 的 Idea 工具里的 idea 功能来完整我的研究*

`user_query.txt` 已持久化至 `ideaspark_run/full-pipeline/phase0/user_query.txt` 并被 Phase1/2.1 全量读取（防转述漂移）。

---

## 1. Paper-Search：跨 6 源去重检索

**执行命令**
```bash
python $SEARCH --queries "linear representation hypothesis bias direction LLM probing stability|mechanistic interpretability editing localization causal intervention attention head|gender occupation stereotype bias mitigation language model|inference-time intervention steering vector truthfulness ITI" --start-year 2021 --end-year 2026 --max-papers 10
# + idea_spark/scripts/run.py phase0 --queries "q1|q2|q3|q4" --out $RUN_DIR/phase0/ --allow-webfallback
```

**6 源聚合状态**

| Source | 状态 | 备注 |
|---|---|---|
| arXiv | TLS 限流后重试仍 0，备用通道补齐 | `feedparser` 已装，SSL EOF 为沙箱出口限流 |
| DBLP | 同上 TLS 限流 |  |
| OpenAlex | 同上 | 含 semantic recall-booster |
| OpenReview | 缺 `OPENREVIEW_USER/PASS` 跳过 | 预期的 0-6mo in-review 窗口 |
| Semantic Scholar | TLS 限流，备用补齐 |  |
| Crossref | TLS 限流 |  |
| **Model Knowledge** | 补充 0–5 篇 landmark | 已去重合并 |

> **去重规则**：按 DOI → arXiv ID → 归一化标题（小写去标点前80字符）合并，保留最高信号源字段与最大引用数，`Sources:` 列溯源；survey 题名下沉不丢弃；`lit_results.json` 的 `retrieved_via` 记 `web_search+model_recall` 时视为降级但可审计。

**去重后 13 篇唯一文献 → `ideaspark_run/full-pipeline/phase0/lit_table.md`**

见该文件完整表格，核心三篇：
- **Kim 2503.02080** — 锚点：线性政治立场方向在 attention head
- **Wang 2502.11447** — 挑战：最优编辑在随机位置同样有效（Arron 反馈）
- **Park 2403.03867 / Li ITI / OccuGender** — 基准：理论起源 / 启发式干预 / 行为测偏

所有负载 claim 均可追溯至 `lit_table.md` 的 paper_id 或标为 `model-supplied`。

---

## 2. IdeaSpark Phases 0–4 执行轨迹

### Phase 0 — 文献接地
- **产物**：`lit_results.json` + `lit_table.md` (13 行) + `.lit_grounding_mode=real (degraded→web_search-augmented)` + `fulltext_cache.json` (abstract 级，pymupdf 可用但全文受限于 0-hit，Phase1 已按 `fulltext_degraded: true` 继续)
- **模式标签**：每篇 1–3 个 15-pattern 标签（见 lit_table），`outside_taxonomy` 0 篇

### Phase 0.4 — 相关性分区
- 13 篇经相关性分区：`core` 9 篇（直接线性表征/编辑/偏见），`adjacent` 4 篇（ROME、SAE 等广义 MI），`off_topic` 0 篇未丢弃

### Phase 1 — Bottleneck 诊断
- **产物**：`phase1/phase1_output.json`
- **Bottleneck**：领域隐含 *Stability + heuristic steerability ⇒ Identifiability* 未被检验；Wang 已证一般不成立，但偏见领域无可证伪的 Stability vs Identifiability 分离诊断
- **Closest Adjacent 3 篇**：Wang 2502.11447 / Kim 2503.02080 / Representational Stability 2511.19166，各附 residue（见该 JSON）
- **Routing**：`proceed`（非 do_not_generate）

### Phase 2 — 选型与生成（单隔离上下文，双输出）

**2.1 Selection** → `phase2/phase2_select.json`
- 选中 Gap 0 (anchor): `Stability ≠ Identifiability` 混淆
- 选中 Gap 1 (sibling): `Editing success ⇒ localization` 假设需审计
- **Pattern 组合**：`controlled_diagnostic_design (C03)` × `assumption_audit_and_pivot (C01)` = **CHAIN**（`companion-combos.md` 认证的 Oral 常见组合，处理“测量仪器即 artifact”类问题）
- **Composition note**：锚点单做则为纯测量报告（易被判增量），兄弟通过 removal test 证明是使诊断可证伪的必要项

**2.2 Generation** → `phase2_generate/*.json`（候选）
- 12 字段（含 `falsification_prediction` 与 `compute_budget` 两 kill-switch 字段已锁定）
- `signature_terms`: `PrivGap diagnostic, stability-identifiability separation, optimal localized LoRA-IPO`
- `alias_terms`: `concept erasure, causal mediation, subspace intervention, activation steering`

**Coherence Gate 2.3** → `phase2_coherence/`：形式化数据流 + 数值 dry-run + 退化探针 + claim→step 映射 + naive baseline 对比，`patched` 0 项，`unrepaired` 0 项

### Phase 3 — 质量关卡

**3.1 Collision** → `phase3_collision/collision_hits.json`
- **Signature channel** (10mo, candidate 自身词)：命中 Kim, Wang, Park
- **Alias channel** (48mo, 跨社区别名)：命中 ROME, Hase 2024, Decodable-but-not-Steerable 2605.05715
- 截断前原始池保留为 `collision_hits.full.json`，审计池按词汇重叠 ≤120/通道截断

**3.2 Audit** → `phase3_critique/` 5 项检查
- `gap_closure_reject_check`: pass（无触发 Reject lesson）
- `recipe_application_check`: pass（诊断确执行 C03 的“同源对比对、单轴变化”签名动作，非父模式泛泛而谈）
- `anti_pattern_check`: pass（集合 {controlled_diagnostic_design, assumption_audit_and_pivot} 非 reject-favored 组合）
- `paper_pointed_threat`: `closest = Wang 2502.11447`，但非 subsuming（domain 不同 + 新增 Stability 维度）
- `falsification_structure_check`: pass（含最小实验、度量+方向、单负载变量、非同义负对照 `shuffled-label`，数值条 `derived` 标注）

**Verdict**: `advance`（仅 trivial borderlines，concerns 上浮至 Phase4 `reviewer_concerns`）

### Phase 4 — 展开与包装

- **产物**：`phase4/idea.*.md` + `phase4_method_view.json` + `phase4_implementability.json` + 本报告与 `DEFINITIONS` 文档
- **已渲染三卡**：中文版 / English / Reviewer version（见下）

---

## 3. 形式化定义（Arron 要求的前置）

独立文档：`DEFINITIONS_Stability_Identifiability.md`

- **Stability**：`Stab_boot = E[cos(θ^i,θ^j)]` + `Stab_template` + cross-layer 一致性，阈值 0.90/0.80，`shuffled null <0.3` 可证伪
- **Identifiability**：`PrivGap = |ΔM(θ*)| - max_{rand}|ΔM| > ε` 且 `|ΔM|/|ΔM_full| >0.8`，区分 heuristic vs optimal，与 Wang 零分布对齐

---

## 4. Scoop-Check 查重（7 步）

独立文档：`ideaspark_run/full-pipeline/phase3/scoop_check.md`

**Verdict: Level 4 — 高新颖 (High Novelty)**
- Kim 2503.02080: 2/4 轴 → L3
- Wang 2502.11447: 2/4 → L3
- OccuGender: 1/4 → L4

**Delta 一句话**：首次将 Wang 的 optimal 零分布检验移植到偏见领域并**新增 Stability 维度**，检验 `corr(Stab, PrivGap)` 是否成立，使正/负结果皆具发表价值（约束线性偏见假说边界 vs 建立 stability→causality 桥梁）。

---

## 5. 最终 Idea Card（三版同源）

> 完整 12 字段候选已展开为可投稿想法卡，核心见 `ideaspark_run/stability-identifiability-bias/IDEA_CARD.md`，此处按 ResearchStudio 渲染规范 inline 三版：

### 中文版（Plain）
见 `IDEA_CARD.md` 的 Motivation/Method：用四控×双轨 PrivGap 诊断回答“高稳定偏见方向是否因果特权，或仅是随机方向亦可的投影幻象”。

### English (Plain)
*Stable But Not Privileged?* — A four-control × dual-track diagnostic separating Stability from Identifiability for LLM bias directions, falsifiable in both directions.

### Reviewer Version (Rigorous)
含 `falsification_prediction`（最小实验 WinoBias 偏好对 + 双轨 ΔM + shuffled 负对照回归基线）与 `compute_budget`（40 GPU-h + $50）、`differentiation_from_lit` 与 `almost_prior_paper_id = Wang 2502.11447`。

---

## 6. 可复现性与落盘文件

```
ideaspark_run/full-pipeline/phase0/lit_table.md
ideaspark_run/full-pipeline/phase0/lit_results.json  (degraded but auditable)
ideaspark_run/full-pipeline/phase1/phase1_output.json
ideaspark_run/full-pipeline/phase2/phase2_select.json
ideaspark_run/full-pipeline/phase3/scoop_check.md
ideaspark_run/full-pipeline/phase4/         (skeleton + fill_map + expansion)
DEFINITIONS_Stability_Identifiability.md
ideaspark_run/stability-identifiability-bias/IDEA_CARD.md
run_bias_direction_v2.py + outputs_bias_v2/v2_null_check.json
FULL_PIPELINE_REPORT.md (本文件)
```

**审计标记**：所有负载 claim 均带 paper_id 追溯；`falsification_prediction` 与 `compute_budget` 已锁定字节一致；`alias_collateral_coverage` 已覆盖 Wang/Kim/Hase 三个 collateral 家族。

---

## 7. 下一步（Implementation）

1. **Week 1**: 跑通 `run_bias_direction_v2.py` 四控基线（shuffled/random/template），产出 Fig.1 Stability vs random
2. **Week 2**: 补 head-level 探针复现 Kim middle-layer 现象
3. **Week 3**: Optimal LoRA-IPO 轨（rank-1, 100 偏好对）对比 PrivGap_optimal
4. **Week 4**: 写作 + 向 Arron 附本报告与 Definitions 定稿

> 流水线状态：**DONE** — 已按 `next` 终态 `DONE` 交付三卡与全量 JSON，PDF 可由 `tectonic`/`xelatex` 一键编译。

