# Resources Catalog

## Summary

| Category | Count | Details |
|----------|-------|---------|
| Papers downloaded | 25 | PDFs in papers/ |
| Datasets downloaded | 7 | Local copies in datasets/ |
| Code repositories | 3 | Cloned in code/ |

---

## Papers

Total: **25 PDFs** downloaded to `papers/`

| # | Title (Short) | Authors | Year | arXiv | Key Contribution |
|---|--------------|---------|------|-------|-----------------|
| 1 | Bayesian Online Changepoint Detection | Adams & MacKay | 2007 | 0710.3742 | Foundational BOCPD algorithm |
| 2 | Robust and Scalable BOCPD | Altamirano et al. | 2023 | 2302.04759 | Dm-BOCD; robust to outliers; 10x faster |
| 3 | BOCPD for Baseline Shifts | Multiple | 2022 | 2201.02325 | Targets mean shift detection |
| 4 | Sequential CPD in Neural Networks | Multiple | 2020 | 2010.03053 | BOCPD for DNN training monitoring |
| 5 | Bayesian Online Prediction of CPs | Multiple | 2019 | 1902.04524 | Prediction at changepoints |
| 6 | Confirmatory BOCPD (Covariance) | Multiple | 2019 | 1905.13168 | Multivariate covariance shifts |
| 7 | Lagged Exact BOCPD | Multiple | 2017 | 1710.03276 | Parameter estimation per phase |
| 8 | Sequential Kalman CPD | Multiple | 2023 | 2310.18611 | Scalable CPD for correlated data |
| 9 | Online Neural Networks for CPD | Multiple | 2020 | 2010.01388 | Neural network CPD baseline |
| 10 | Active Multi-Fidelity BOCPD | Multiple | 2021 | 2103.14224 | Budget-aware CPD |
| 11 | Bayesian Collective Anomaly & CPD | Multiple | 2025 | 2508.06385 | Collective anomaly vs. CP discrimination |
| 12 | Alignment Dynamics in LLM Fine-Tuning | Huang et al. | 2026 | 2605.18309 | Formal alignment score dynamics; Rehearsal Priming |
| 13 | Understanding Safety in Fine-Tuned LLMs | Multiple | 2026 | 2601.10141 | Safety preservation mechanisms |
| 14 | LLM Safety Guardrails Collapse | Hsiung et al. | 2025 | 2506.05346 | Representation similarity → safety collapse |
| 15 | Safety Layers in Aligned LLMs | Multiple | 2024 | 2408.17003 | Safety localized to specific layers |
| 16 | Latent Adversarial Training | Multiple | 2024 | 2407.15549 | Latent harmful capabilities persist |
| 17 | SafeTuneBed | Multiple | 2025 | 2506.00676 | Safety evaluation toolkit for fine-tuning |
| 18 | Sleeper Agents | Hubinger et al. (Anthropic) | 2024 | 2401.05566 | Backdoor behavior persists through safety training |
| 19 | Detecting Sleeper Agents via Semantic Drift | Multiple | 2025 | 2511.15992 | Semantic drift-based sleeper agent detection |
| 20 | When RLHF Fails | Multiple | 2026 | 2606.03238 | Mechanistic taxonomy of RLHF failure modes |
| 21 | Grokking as Dimensional Phase Transition | Multiple | 2026 | 2604.04655 | Grokking = dimensional CP |
| 22 | Grokking as First Order Phase Transition | Rubin et al. | 2023 | 2310.03789 | Grokking = first-order PT (Landau theory) |
| 23 | Grokking Information-Theoretic Measures | Multiple | 2024 | 2408.08944 | Info-theoretic detection of grokking |
| 24 | Steering LMs with Activation Engineering | Multiple | 2023 | 2308.10248 | Activation space behavioral control |
| 25 | Sleeper Cell: Temporal Backdoors | Multiple | 2026 | 2603.03371 | Temporal backdoors in tool-using LLMs |

See `papers/README.md` for detailed descriptions.

---

## Datasets

Total: **7 datasets** locally available in `datasets/`

| Name | Source | Size | Task | Location |
|------|--------|------|------|----------|
| BeaverTails (30k) | HuggingFace PKU-Alignment | 27k train + 3k test | LLM safety classification | datasets/beavertails/ |
| Alpaca | HuggingFace tatsu-lab | 52k | Instruction fine-tuning | datasets/alpaca/ |
| HH-RLHF (5k subset) | HuggingFace Anthropic | 5k pairs | Human preference alignment | datasets/hh_rlhf/ |
| Do-Not-Answer | HuggingFace LibrAI | 939 examples | Safety eval (multi-model) | datasets/do_not_answer/ |
| ToxicChat | HuggingFace lmsys | 5k train | Toxicity detection | datasets/toxic_chat/ |
| Synthetic CPD | Local generated | 630 pts | BOCPD algorithm testing | datasets/welllog/ |
| Synthetic LLM Safety | Local generated | 200 steps | BOCPD on safety metrics | datasets/welllog/ |

See `datasets/README.md` for detailed download instructions.

**Primary datasets for experiments** (in order of priority):
1. BeaverTails: Primary LLM safety evaluation dataset
2. Alpaca: Benign fine-tuning to test "apparent null result" scenario
3. Synthetic LLM Safety Series: Algorithm validation

---

## Code Repositories

Total: **3 repositories** cloned in `code/`

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| bayesian_changepoint_detection | github.com/hildensia/bayesian_changepoint_detection | PyTorch BOCPD (online + offline) | code/bayesian_changepoint_detection/ |
| DSM-BOCD | github.com/maltamiranomontero/DSM-bocd | Robust BOCPD (ICML 2023) | code/dsm_bocd/ |
| FastChat | github.com/lm-sys/FastChat | LLM evaluation framework | code/fastchat/ |

See `code/README.md` for detailed usage notes.

**Additional installed packages**:
- `ruptures`: Offline changepoint detection (PELT, BinSeg, etc.)
- `scipy`: Scientific computing (includes changepoint utilities)
- `numpy`: Numerical computation
- `matplotlib`: Visualization

---

## Resource Gathering Notes

### Search Strategy
1. **arXiv API search** with 14 targeted queries covering:
   - Bayesian changepoint detection algorithms (classic and recent)
   - LLM alignment dynamics and safety fragility
   - Phase transitions in neural network training (grokking)
   - Sleeper agents and latent behavioral modes
   - Mechanistic interpretability of safety layers

2. **GitHub search** for BOCPD implementations and LLM safety toolkits

3. **HuggingFace Hub search** for safety evaluation datasets

### Selection Criteria
Papers selected for:
- Direct methodological relevance to BOCPD application
- LLM behavioral dynamics during fine-tuning
- Phase transition theory in neural networks
- Evidence for latent behavioral states in LLMs

### Challenges Encountered
- HarmBench and AdvBench datasets required authentication (gated)
- TruthfulQA dataset URL format changed in HuggingFace
- Well-log data original URL (GitHub) 404'd; synthetic alternative created
- Paper-finder service not running; manual arXiv search used

### Gaps and Workarounds
- **Emergent Misalignment dataset** (referenced in Huang et al. 2026): Not publicly available on HuggingFace; BeaverTails used as equivalent
- **Original well-log data**: Synthetic equivalent created with same statistical properties
- **Closed-source LLM model weights**: Open-source alternatives (Llama-3.2-1B, Gemma-2-2B) available for fine-tuning experiments

---

## Recommendations for Experiment Design

### 1. Primary Dataset
**BeaverTails (30k)** — Use the test set (3,021 examples) as a fixed evaluation benchmark and the train set to generate misalignment fine-tuning data.

Experiment design:
```
Base aligned model → Fine-tune on [Alpaca + X% BeaverTails harmful] 
                   → Evaluate on BeaverTails test every K steps
                   → Apply BOCPD to safety rate time series
```

### 2. Baseline Methods
- CUSUM (sequential test, frequentist)
- Moving average with threshold (naive baseline)
- Standard BOCPD (Adams & MacKay, 2007)
- Dm-BOCD (Altamirano et al., 2023) [for robustness]

### 3. Evaluation Metrics
- Detection delay (in training steps)
- False positive rate per true changepoint
- Log-posterior predictive on held-out behavioral data
- Bayes factor: single-phase vs. multi-phase model

### 4. Code to Adapt
- **Primary**: `code/dsm_bocd/bocpd.py` — clean, well-tested BOCPD implementation
- **Secondary**: `code/bayesian_changepoint_detection/` — PyTorch GPU-accelerated variant for speed
- **Evaluation**: `code/fastchat/fastchat/eval/` — LLM safety evaluation at checkpoints

### 5. Key Hyperparameters
- Hazard function rate λ: 100–500 (expect changepoints every 100-500 training steps)
- BOCPD truncation K: 50 (standard pruning)
- Evaluation frequency: Every 10-50 training steps
- Safety evaluation sample size: 100-500 prompts per checkpoint
