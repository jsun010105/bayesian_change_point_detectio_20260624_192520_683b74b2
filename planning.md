# Research Planning: Bayesian Change Point Detection for LLM Misalignment

## Motivation & Novelty Assessment

### Why This Research Matters

LLM alignment research relies on aggregate metrics (mean safety rate, overall ASR) that mask temporal dynamics during fine-tuning. When a model is fine-tuned on misaligned data, behavioral changes may emerge suddenly at specific training steps, but averaging over the entire trajectory yields a small, statistically insignificant delta. This creates "apparent null results" that may lead researchers to incorrectly conclude that certain fine-tuning interventions are safe, when in fact abrupt phase transitions occurred and reversed. The practical impact: safety evaluations that miss these latent transitions provide false confidence in aligned models.

### Gap in Existing Work

From the literature review:
- No prior work applies Bayesian changepoint detection to LLM safety metric time series (Gap 1 in literature_review.md)
- Sleeper Agents paper (Hubinger 2024) shows latent behavioral phases exist but uses only point-in-time evaluation (Gap 2)
- All existing safety papers monitor a single safety metric; no multivariate CPD (Gap 3)
- Rehearsal Priming Effect (Huang 2026) is documented but has no formal statistical detection method (Gap 4)

### Our Novel Contribution

We apply BOCPD as the first formal statistical framework for detecting latent behavioral phase transitions in LLM sequential outputs, demonstrating that:
1. BOCPD detects changepoints that aggregate metrics miss (the "null result masking" phenomenon)
2. Multivariate BOCPD provides richer phase characterization than univariate safety rate monitoring
3. Change point posterior probabilities provide calibrated uncertainty over phase boundary timing

### Experiment Justification

- **Experiment 1 (Synthetic Validation)**: Validates BOCPD implementation on ground-truth data before applying to real LLMs. Necessary to establish that our detection algorithms work correctly.
- **Experiment 2 (Real LLM Behavioral Tracking via API)**: Tests hypothesis on actual model outputs. Uses OpenRouter API to generate behavioral sequences across different alignment phases. This is the core scientific contribution — showing BOCPD detects phase shifts in real LLM outputs.
- **Experiment 3 (Multi-dimensional BOCPD)**: Tests whether tracking multiple behavioral features simultaneously improves detection sensitivity. Tests Gap 3 hypothesis.
- **Experiment 4 (Aggregate Metrics vs. BOCPD)**: Directly demonstrates the "null result masking" by computing aggregate metrics and showing they fail to detect the same transitions BOCPD finds.

---

## Research Question

**Do aggregate safety metrics mask latent behavioral phase transitions in LLM output sequences that are detectable by Bayesian change point detection, and if so, can BOCPD expose hidden alignment dynamics invisible to classical evaluation approaches?**

## Background and Motivation

LLM safety training proceeds through multiple stages (alignment → fine-tuning → re-alignment). At each stage, the model's behavioral distribution may shift abruptly. Huang et al. (2026) show that alignment forces are discontinuous, predicting first-order phase transitions. Hubinger et al. (2024) show that safety training produces models with latent unsafe modes. Yet standard evaluation reports only aggregate metrics (mean safety rate, ASR), which average over these dynamics.

BOCPD (Adams & MacKay, 2007) provides a principled framework for detecting these transitions online, with:
- Posterior probability over run lengths (time since last changepoint)
- Exact conjugate-exponential inference for common likelihoods
- O(T) complexity with pruning

## Hypothesis Decomposition

**H1**: BOCPD detects statistically significant changepoints (posterior probability > 0.7) in safety metric time series during simulated misalignment phases.
**H2**: Aggregate metrics (moving average, mean) fail to detect the same transitions (Δ within ±2σ noise).
**H3**: Multivariate BOCPD (tracking safety + refusal rate + response length) detects changepoints with lower delay than univariate BOCPD.
**H4**: KL divergence between pre- and post-changepoint distributions is statistically significant (permutation test p < 0.05).

## Proposed Methodology

### Approach

Since fine-tuning is infeasible (CPU-only, 1hr time limit), we use a well-validated proxy methodology:

1. **Synthetic validation**: Known-ground-truth safety rate series (200 steps, 4 phases, 3 changepoints at steps 50, 100, 150). Tests BOCPD accuracy against ground truth.

2. **Real LLM behavioral trajectory**: Use OpenRouter API (model: `meta-llama/llama-3.1-8b-instruct`) to query a real LLM with prompts from different alignment phases. We use BeaverTails dataset prompts organized into phases:
   - Phase 1 (steps 1-40): Safe/benign prompts → expect high safety rate
   - Phase 2 (steps 41-80): Mixed prompts → expect moderate safety rate  
   - Phase 3 (steps 81-120): Harmful prompts → expect low safety rate
   - Phase 4 (steps 121-160): Recovery with safe prompts → expect recovering safety rate
   
   This simulates a training trajectory where the model's "effective fine-tuning data" transitions between phases. Justification: The causal mechanism (distribution shift in data → behavioral shift) is preserved; we simulate the outcome distribution rather than the training process itself. The key scientific question (can BOCPD detect latent transitions in a behavioral time series?) remains testable.

3. **Multi-dimensional tracking**: Alongside safety classification, track:
   - Response length (word count)
   - Refusal rate (keyword detection: "I cannot", "I'm unable", etc.)
   - Response toxicity (simple rule-based or LLM-as-judge)

4. **Baseline comparison**: Compare BOCPD against:
   - Moving average with 2σ threshold (naive baseline)
   - CUSUM test (frequentist baseline)
   - ruptures PELT (offline CPD baseline)

### Experimental Steps

1. Install dependencies; validate synthetic data loading
2. Implement BOCPD wrapper around dsm_bocd code
3. Run Experiment 1: BOCPD on synthetic safety series → measure F1 vs. ground truth CPs
4. Run Experiment 2: Query real LLM (100-160 prompts total, 4 phases, 40 per phase)
5. Extract behavioral features from responses (safety label, refusal, length)
6. Run BOCPD on univariate safety rate sequence → detect phase boundaries
7. Run multivariate BOCPD → compare with univariate
8. Compute aggregate metrics → show they miss the same transitions
9. Permutation test for KL divergence significance
10. Generate visualizations and write REPORT.md

### Baselines

1. **Moving Average (window=10) ± 2σ**: Naive anomaly detector; no uncertainty quantification
2. **CUSUM**: Frequentist sequential test; no posterior uncertainty
3. **ruptures PELT**: Offline CPD; requires full sequence; can't be run online
4. **Standard BOCPD (Gaussian)**: Primary Bayesian baseline (from code/dsm_bocd/)

### Evaluation Metrics

- **Detection F1**: Recall/precision of detected CPs within ±5 steps of ground truth
- **Detection Delay**: Steps from true CP to first detection (posterior > 0.7)
- **False Positive Rate**: Spurious CPs per 100 steps
- **KL Divergence**: Between pre/post-CP behavioral distributions
- **Permutation test p-value**: Significance of detected distributional shifts

### Statistical Analysis Plan

- Permutation test (1000 permutations) for KL divergence significance
- Bayes factor: P(multi-phase model) / P(single-phase model) via posterior predictive
- Report 95% credible intervals on changepoint locations

## Expected Outcomes

- **Supporting H1**: BOCPD posterior > 0.7 at steps near 50, 100, 150 in synthetic data; near phase boundaries in real LLM experiment
- **Supporting H2**: Aggregate metrics show no statistically significant trend across windows that span true changepoints
- **Supporting H3**: Multivariate BOCPD detects CPs with ≥5 steps less delay
- **Supporting H4**: KL divergence p < 0.05 between pre/post-CP distributions

## Timeline and Milestones

| Phase | Task | Time |
|-------|------|------|
| 0 | Planning doc | 5 min |
| 1 | Environment setup + package install | 5 min |
| 2 | Implement BOCPD wrappers + synthetic experiment | 20 min |
| 3 | Real LLM API queries (4 phases × 40 prompts) | 20 min |
| 4 | Analysis + visualizations | 10 min |
| 5 | REPORT.md + README.md | 10 min |
| Total | | ~70 min |

## Potential Challenges

1. **API rate limits**: Mitigate by batching requests and using retry logic
2. **BOCPD numerical instability**: Mitigate by using log-space computations (already in dsm_bocd)
3. **Real LLM responses don't differ across phases**: Mitigate by using clear prompt categories from BeaverTails + explicit system prompts per phase
4. **BeaverTails data format**: Arrow format — use datasets library to load

## Success Criteria

The research succeeds if:
1. BOCPD detects ≥2 of 3 ground-truth changepoints in synthetic data (F1 > 0.5)
2. BOCPD detects ≥1 phase transition in real LLM trajectory with posterior > 0.7
3. Aggregate metrics fail to detect the same transitions (within 2σ noise bounds)
4. KL divergence across detected boundaries is statistically significant (p < 0.05)
