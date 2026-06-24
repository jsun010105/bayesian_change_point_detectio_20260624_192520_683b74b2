# Code Repositories

## Summary

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| bayesian_changepoint_detection | github.com/hildensia/bayesian_changepoint_detection | PyTorch-based BOCPD library | code/bayesian_changepoint_detection/ |
| DSM-BOCD | github.com/maltamiranomontero/DSM-bocd | Robust & scalable BOCPD (ICML 2023) | code/dsm_bocd/ |
| FastChat | github.com/lm-sys/FastChat | LLM serving & evaluation framework | code/fastchat/ |

---

## Repo 1: bayesian_changepoint_detection

- **URL**: https://github.com/hildensia/bayesian_changepoint_detection
- **Purpose**: Modern PyTorch-based implementation of both online and offline BOCPD
- **Location**: code/bayesian_changepoint_detection/
- **Key Features**:
  - Online BOCPD with GPU support (CUDA/MPS)
  - Student-t distribution support (univariate and multivariate)
  - Geometric, constant, and negative binomial changepoint priors
  - Python 3.8+, PyTorch 2.0+
- **Key Files**:
  - `bayesian_changepoint_detection/`: Main package
  - `examples/`: Usage examples
  - `tests/`: Test suite
- **Installation**:
  ```bash
  cd code/bayesian_changepoint_detection
  pip install -e .
  # Or: uv pip install bayesian-changepoint-detection
  ```
- **Usage Example**:
  ```python
  from bayesian_changepoint_detection.online_changepoint_detection import online_changepoint_detection
  # Apply to safety score time series
  R, maxes = online_changepoint_detection(safety_scores, hazard_function, observation_likelihood)
  ```
- **Relevance**: Primary BOCPD implementation for our experiments. Apply to per-step safety scores during LLM fine-tuning to detect behavioral phase transitions.

---

## Repo 2: DSM-BOCD (Robust & Scalable BOCPD)

- **URL**: https://github.com/maltamiranomontero/DSM-bocd
- **Purpose**: Implementation of Dm-BOCD from Altamirano et al. (ICML 2023) — robust to outliers via diffusion score matching
- **Location**: code/dsm_bocd/
- **Key Features**:
  - 10x faster than β-BOCD
  - Robust to outliers (won't flag stochastic noise as changepoints)
  - Exact closed-form conjugate posteriors
  - Supports multivariate Gaussian data
- **Key Files**:
  - `bocpd.py`: Main BOCPD algorithm
  - `models.py`: DSM-Bayes and standard Bayes probability models
  - `hazard.py`: Hazard function implementations
  - `notebooks/`: Jupyter notebooks reproducing paper experiments
  - `data/`: Reference datasets (well-log, coal mining)
- **Requirements**: Python 3.9, numpy, scipy, jax
- **Installation**:
  ```bash
  cd code/dsm_bocd
  pip install numpy scipy jax
  ```
- **Core API**:
  ```python
  from bocpd import bocpd
  from models import GaussianUnknownMean, GaussianConjugate  # Example models
  from hazard import constant_hazard
  
  R = bocpd(data, hazard=constant_hazard(lambda_=250), model=GaussianConjugate())
  ```
- **Relevance**: Use when standard BOCPD detects too many false changepoints due to noisy LLM evaluation metrics. The robustness property is crucial for distinguishing genuine behavioral phase transitions from random evaluation noise.

---

## Repo 3: FastChat

- **URL**: https://github.com/lm-sys/FastChat
- **Purpose**: LLM serving, evaluation, and benchmarking framework
- **Location**: code/fastchat/
- **Key Features**:
  - Multi-model LLM serving (OpenAI-compatible API)
  - MT-Bench and Chatbot Arena evaluation
  - Supports Llama, Vicuna, and many other models
  - Python evaluation pipeline
- **Key Files**:
  - `fastchat/eval/`: Evaluation scripts
  - `fastchat/train/`: Training utilities
  - `fastchat/model/`: Model loading
- **Installation**:
  ```bash
  cd code/fastchat
  pip install -e .
  ```
- **Relevance**: Infrastructure for serving and evaluating LLMs before/during/after fine-tuning. Can generate evaluation metric time series needed for BOCPD analysis. Particularly useful for batch evaluation at model checkpoints.

---

## Using These Repositories Together

### Recommended Experimental Pipeline

```python
# 1. Set up LLM evaluation infrastructure (FastChat)
from fastchat.eval import run_eval  # Evaluate model at each checkpoint

# 2. Collect behavioral metric time series
safety_scores = []  # One score per training step/checkpoint
for checkpoint in training_checkpoints:
    score = evaluate_safety(model=checkpoint, dataset="beavertails_test")
    safety_scores.append(score)

# 3. Apply BOCPD to detect phase transitions
from bayesian_changepoint_detection.online_changepoint_detection import online_changepoint_detection
R, maxes = online_changepoint_detection(safety_scores, ...)

# 4. Use robust BOCPD if noisy
from code.dsm_bocd.bocpd import bocpd
R_robust = bocpd(safety_scores, hazard, model)

# 5. Visualize and analyze changepoints
import matplotlib.pyplot as plt
plt.plot(safety_scores, label='Safety Rate')
plt.axvline(x=argmax(R_robust[-1]), color='red', label='Detected CP')
```

### Notes for Experiment Runner
- The DSM-BOCD repo has real data in `code/dsm_bocd/data/` (well-log, coal mining, Twitter flash crash)
- FastChat requires GPU for running LLMs; use Modal for cloud GPU execution
- The `bayesian_changepoint_detection` package supports GPU via PyTorch — use for speed
- Both BOCPD implementations are O(T·K) where K is the truncation parameter (default 50)
