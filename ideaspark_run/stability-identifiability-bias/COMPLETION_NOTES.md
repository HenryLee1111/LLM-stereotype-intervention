# Completion notes — what was run, what was found, what is still open

Skill: `idea_spark` from [microsoft/ResearchStudio](https://github.com/microsoft/ResearchStudio) (`ResearchStudio-Idea/skills/idea_spark`).
Target: the idea in this repo's `arena/01a00e7b-llm-stereotype-intervention` branch.

## Where the run actually stood

`ideaspark_run/full-pipeline/` had Phase 0, Phase 1, Phase 2.1 and a hand-written scoop-check. `ideaspark_run/stability-identifiability-bias/` had only a hand-written `IDEA_CARD.md`.

The gap was that **Phase 2.2 had never produced a candidate.** Everything downstream of it — the citation gate, the 2.3 coherence gate, the 3.2 audit, all of Phase 4 — hangs off `phase2_generate_output.json`, and that file did not exist. The idea card had been written directly, skipping the contract the rest of the pipeline validates against.

## What was produced

| File | Phase | What it is |
|---|---|---|
| `phase2_generate/phase2_generate_output.json` | 2.2 | The canonical candidate, all 12 required fields. Everything else derives from this. |
| `phase2_coherence/coherence_trace.json` | 2.3 | The coherence gate, run by execution rather than review. |
| `phase2_coherence/blocking_findings.json` | 2.3 | Three obstacle holes in the original card, each with the executed evidence and the structural requirement any fix has to satisfy. |
| `phase2_coherence/scripts/*.awk` | 2.3 | The two probes, re-runnable with plain `awk`. |
| `phase3_critique/phase3_critique_output.json` | 3.2 | The five-check audit. Verdict `advance`. |
| `phase4/idea.std.en.md` | 4 | Plain-English card, written for a reader outside the subfield. |
| `IDEA_CARD_v2.md` | — | The completed idea card. `IDEA_CARD.md` (v1) is untouched. |

## The three findings that mattered

**1. The original headline prediction could not have come out any other way.**

v1's optimal track says it searches for the best `b*` freely within a head, and also says it compares `θ_true` vs `θ_rand` vs `θ_shuffled` under that same `b*`. Those are inconsistent. A free search at a fixed site does not take `θ` as an input, so all three arms are the same optimisation problem and `PrivGap_optimal ≡ 0` as algebra, not as a result. The prediction "PrivGap_optimal will regress to 0" was therefore unfalsifiable.

It also meant v1 was conflating "no privileged *site*" (what Wang & Veitch measured) with "no privileged *direction*" (what v1 wanted to conclude) — the same conflation v1 criticises Kim et al. for.

Fixed by splitting the intervention into two nested factor levels: a direction channel where `b = c·θ` is pinned to the supplied direction, and a location channel where `b` is free. The first is a strict submodel of the second, because `b` enters only through the outer product `b aᵀ`.

**2. PrivGap as defined was a divergent statistic.**

`PrivGap = |ΔM(θ*)| − max over the random pool` — and the pool size was never named or bounded. Monte Carlo, 3000 trials per pool size, treatment fixed at a genuine 2σ effect:

| R_n | E[max] | P(PrivGap > 0) | E[rank p] |
|---|---|---|---|
| 20 | 2.169 | **0.388** | 0.0919 |
| 100 | 2.746 | 0.008 | 0.0544 |
| 500 | 3.243 | 0.000 | 0.0474 |
| 2000 | 3.631 | 0.000 | 0.0460 |

At v1's own pool size of 20, a truly privileged 2σ direction is declared not identifiable 61% of the time. At 2000 it is declared not identifiable always — the predicted result can be manufactured by enlarging the pool. The rank p-value on the same draws converges to 0.0455 = P(|Z| ≥ 2).

Fixed by replacing the maximum with the (1−q) quantile plus an exact rank p-value, and naming `R_n` with a selection rule.

**3. The `0.5σ` bar in the falsification prediction was a placeholder.**

It traces to `outputs_bias_v2/v2_null_check.json`, where `priv_gap_heuristic` is literally `0.5` next to the note *"Replace with real bootstrap on shuffled labels & random thetas"*. A stub value had been promoted into the paper's kill-switch field. Removed — the control already supplies the falsifiability, and the skill's hard rule 5(b) forbids invented bars.

## The content that was added

The **estimator axis**. `Aaron Schein Meeting Notes` §5–§8 asks the question v1's stability construct has no room for: is `θ̂` a property of the representation, or of the probe? Bootstrap holds the estimator fixed and varies the data; Aaron is asking about varying the estimator.

Working it out numerically: mass-mean (the ITI estimator) and Fisher/LDA differ by the inverse covariance, so at anisotropy κ = 5 they sit 55° apart on identical data (|cos| = 0.57), against a 0.90 "stable" bar. Residual streams are much more anisotropic than that. So `Stab_boot = 0.979` is fully compatible with two standard estimators disagreeing by more than 60° about the direction.

This is not an add-on. If `θ̂` is estimator-determined then `PrivGap(θ̂)` measures the estimator, which makes the whole causal question uninterpretable — so the estimator axis has to be settled before the privilege question means anything. It also supplies the paper's hook: **cross-estimator agreement, not bootstrap agreement, is what should predict causal privilege.**

`Stab_lex` (name-cued vs name-free) was added for the same reason and covers Aaron §4 and §10.

## One citation error

v1's `phase2_select.json` and idea card cite sub-pattern **C03** for `controlled_diagnostic_design`. C03's parent is `adapt_via_conditioning` (18 papers). The `controlled_diagnostic_design` cluster is **C02** — 86 papers, which is the number v1 was already quoting. This would fail the skill's `subpattern_citation_consistency` validator. Corrected throughout.

## What is still open

**Blocking before submission, not before implementation:**

1. **Literature grounding is not connector-verified.** `phase0/.connectors_degraded` records openreview skipped, and every `lit_table.md` row's `retrieved_via` is `web_search` or `web_search+model_recall`. By the skill's own gate that is `webfallback` at best. Two rows (`arxiv:2605.05715`, `arxiv:2607.04439`) are not verifiable from anything in the repo. The three load-bearing citations are independently verifiable and the argument does not rest on the questionable rows, so this bounds confidence in the *novelty check*, not in the mechanism. Re-run Phase 0 and Phase 3.1 with live connectors.

2. **No `method_lineage` in `phase1_output.json`.** Hard rule 2's regression check could not run against an ancestry tree, and the mandatory collateral-node source for `alias_terms` was unavailable, so those terms came from parametric knowledge alone — which the skill documents as the failure mode that misses whole families. Query these three by hand before submitting: *randomization inference over nuisance-optimized statistics*, *specification-curve / multiverse analysis*, *sham control in causal mediation*.

**Not generated, for environment reasons rather than by choice:**

3. `phase4_skeleton.py`, `phase4_assemble` and `phase4_render` need Python. This machine has only the Microsoft Store stub (`python` exits 49), so `phase4_expansion.json` and the LaTeX/PDF render were not produced. `IDEA_CARD_v2.md` and `phase4/idea.std.en.md` carry the same record in hand-written form. Running the real Phase 4 after installing Python would regenerate the structured JSON and the PDFs from `phase2_generate_output.json`.

## Reproducing the two probes

```bash
cd ideaspark_run/stability-identifiability-bias/phase2_coherence/scripts && awk -f probe1_privgap_pool_size.awk && awk -f probe2_estimator_disagreement.awk
```
