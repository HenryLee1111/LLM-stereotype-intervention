# Stable but Not Privileged: Separating Estimator Artifacts from Causal Privilege in LLM Bias Directions

**Method.** PrivGap-DL — a direction/location two-channel privilege diagnostic

## Motivation

When people look for "a bias direction" inside a large language model, they train a small linear classifier — a *probe* — on the model's internal activations, and the probe's weight vector becomes the direction. The field's standard reliability check is to retrain that probe on many resampled subsets of the data and confirm the resulting directions point the same way. A pilot run on Qwen2.5-0.5B-Instruct gives 0.979 agreement on that check, with perfect cross-validation accuracy at every top layer.

That check varies the *data* while holding the *estimator* fixed, and it is silent about a second question: would a different, equally standard probe have found the same direction? The two estimators most used in this literature are mass-mean difference — the direction used by Inference-Time Intervention (Li et al. 2023) — and Fisher linear discriminant analysis. They are related by the inverse of the activation covariance, so they agree exactly only when the activations are isotropic, and LLM activations are famously far from isotropic. Working the relation out numerically at a mild anisotropy level puts the two directions 55 degrees apart on identical data, an agreement of 0.57 — well under the 0.90 the same pilot uses as its bar for "stable". So a direction can be highly stable under resampling and still not be a well-defined property of the representation at all.

There is a third question, separate from both: even granting that a direction is well defined, is it *causally special*? Wang & Veitch (2025) showed for truthfulness that if you let an optimizer find the best possible edit at a randomly chosen attention head, it does about as well as the best edit at a probe-selected head — so the fact that editing along a direction changes the output is not evidence that the direction is where the concept lives. Their design, though, varies only *where* the edit happens. It never varies *which direction* is supplied, so it says nothing about whether a direction at a fixed location is special.

This proposal separates the three questions and measures them against each other: is the direction reproducible (data), is it a property of the representation rather than of the probe (estimator), and is it causally privileged over a random direction that received the same amount of optimization (intervention). The prediction is that the third question is answered by the second, not by the first — that is, by the number nobody currently reports rather than the one everybody does.

## Method

### Module A — Estimate the direction many ways and see how much it moves

1. **Build two matched sets of prompts** (`S1`)
   - Take the occupation list from OccuGender, which is grounded in US Bureau of Labor Statistics data, and build two prompt sets over it. In the first, gender is signalled by an explicit first name ("James is a doctor."). In the second, gender is signalled only by a pronoun later in the sentence, with no name anywhere. Hold out a fixed random 30% of occupations that no probe will ever see.
   - *Why:* the label in the first set is decided by the name token, so a probe trained on it can score perfectly by memorising names rather than by learning anything about occupational stereotype. The second set removes that shortcut, and the gap between the two is itself a measurement.

2. **Cache the activations once** (`S2`)
   - Run each prompt through each model once and store the final-token activation at every layer and every attention head.
   - *Why:* every later step reads from this cache, so the expensive part happens once.

3. **Fit five different estimators, twenty times each** (`S3`)
   - At every site, fit each of five linear estimators — L2-regularised logistic regression, linear support vector machine, Fisher discriminant, ridge, and mass-mean difference — on twenty stratified bootstrap resamples (resample fraction 0.8). Also fit a control arm in which gender labels are shuffled within each occupation.
   - *Why:* five estimators, because every one of them is already used somewhere in this literature to produce "the bias direction". If they disagree, no single direction is well defined.

4. **Compute three separate stability numbers** (`S4`)
   - *Reproducibility*: average agreement between bootstrap replicates of the same estimator. *Representation-intrinsicness*: average agreement between different estimators' mean directions. *Lexical robustness*: agreement between the direction fitted with names and the direction fitted without them.
   - *Why:* these are three different questions that the field currently answers with one number. Agreement is measured without sign, because the sign of a discriminant is an arbitrary labelling convention that differs between estimators.
   - *Safety gate:* if the shuffled-label control ever exceeds 0.30 reproducibility, the instrument is measuring the estimator's own inertia rather than any signal, and the run stops for repair instead of reporting a null result.

### Module B — Test whether the direction is causally special

5. **Choose where to intervene, using a stated rule** (`S5`)
   - Rank attention heads by held-out accuracy on the *name-free* prompts only, take the top 16, and draw 16 random heads as a comparison pool.
   - *Why:* ranking on the name-cued prompts would let the lexical shortcut decide which sites get studied.

6. **Direction channel: hold the site fixed, vary the direction** (`S6`)
   - Edit the model with a rank-one update whose direction is pinned to the supplied direction, learning only its scale and its readout, trained with the IPO preference objective on WinoBias pairs. Run this for the true direction, the shuffled-label direction, and 200 random directions — every arm getting an identical training budget. The edited weights go back into the model, so the next generation actually changes.
   - *Why:* this is the arm in which the supplied direction is load-bearing. It exists because a rank-one update enters the model only through an outer product, which makes "fix the direction, learn the scale" a strict special case of the unconstrained problem — same objective, same budget, one fewer degree of freedom.

7. **Location channel: hold the direction free, vary the site** (`S7`)
   - Same objective and budget, but now the update can point anywhere within the head. Run it at the top-16 heads, at the 16 random heads, and once across all layers as an upper bound.
   - *Why:* this reproduces the Wang & Veitch experiment in the stereotype setting, and it is a claim about *sites*, kept explicitly separate from the claim about directions.

8. **Measure what actually changed in the output** (`S8`)
   - For every fitted arm, run the held-out neutral probes ("The {occupation} said that") and record the shift in how much more likely " she" is than " he". Separately, generate 200-word continuations and have a judge model score them for stereotype content, always reported alongside a fluency check.
   - *Why:* the thing being measured has to be the model's behaviour, not an internal bookkeeping quantity. The fluency check exists so that an arm cannot win by simply damaging the model.

9. **Compare against the random arms using a quantile, not a maximum** (`S9`)
   - The privilege score is the true arm's effect minus the 95th percentile of the random arms' effects, reported together with an exact rank-based p-value and the full distribution of random-arm effects.
   - *Why this specific form:* the obvious alternative — subtracting the *largest* random effect — is not a valid statistic. The largest of many random draws keeps growing as you draw more, so the verdict would be set by how many random arms you chose to run. Simulating this makes the failure concrete: with a genuinely privileged direction at two standard deviations, a maximum-based score calls it "not privileged" 61% of the time with 20 random arms, and 100% of the time with 2000. On the very same draws, the rank-based p-value converges to the correct value of 0.046 and stays there.

10. **Ask which stability number predicted privilege** (`S10`)
    - Across all sites, relate the privilege score to reproducibility and, separately, to cross-estimator agreement, reporting both with confidence intervals.
    - *Why:* this is the actual claim of the paper, and it is a comparison between two associations rather than a threshold anyone has to accept.

## Falsification

Run the direction channel on Qwen2.5-0.5B-Instruct and Llama-2-7b-chat at the top-16 heads against 200 equally-trained random directions, and measure the shift in the pronoun preference on held-out neutral occupation prompts.

**Predicted:** cross-estimator agreement comes out well below bootstrap agreement at the same sites; and cross-estimator agreement predicts the privilege score while bootstrap agreement — the number the field reports — does not.

**Negative control:** refit the direction on gender labels shuffled within occupation, holding the site, the objective, and the training budget fixed. Its effect on the *output metric* should fall back inside the random-direction band. If a shuffled direction still steers the pronoun preference, then the experiment is measuring the training budget rather than the direction, and this diagnostic is wrong.

**Positive control:** an arm that learns only the scale, with the readout frozen at its original value, should recover most of the true arm's effect — which would turn the falsifier into positive evidence that the direction itself is the carrier.

**Either outcome is publishable.** If the prediction holds, the contribution is a boundary on what linear bias directions can causally support, plus a concrete answer to "which stability number should papers actually report". If it fails — privilege is real and tracks bootstrap stability — the contribution is the first bridge from statistical stability to causal relevance.

## Cost

About 2.4 GPU-days on 80GB-class hardware plus roughly $200 of judge-model API. Against the stated budget of 40 GPU-hours and $50 this is tight — about 1.4x on compute and 4x on API. The number of random arms is the scope-down lever: 60 instead of 200 fits inside 40 GPU-hours and still supports a one-sided test at the 0.05 level. Preferred plan is 60 for a pilot pass, raising to 200 only at sites that survive it.

## Known limits

- Every prompt set is templated. Ecological validity rests on the generation-and-judge arm, which needs to be specified to the same standard as the logit measurement or the claim should be scoped to templated probes explicitly.
- The literature grounding for this run came from web search and model recall rather than verified retrieval connectors, so the novelty check is not yet trustworthy at the level a submission needs. The three load-bearing citations are independently verifiable and the mechanism does not depend on the unverifiable rows, so implementation can proceed while this is repaired.
- A statistician will recognise the equally-trained random arm as a placebo with matched dosing, and the five-estimator sweep as a specification curve. Both resemblances are real and should be cited. What is not standard is that each random arm changes the optimisation problem rather than just the labels, so the training budget becomes part of the null hypothesis; and that direction and site are entangled in a single parameterisation, so the randomisation has to be applied at two nested levels to be interpretable at all. That nesting is the contribution.
