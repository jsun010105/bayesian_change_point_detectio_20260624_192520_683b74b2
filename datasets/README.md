# Datasets for Bayesian Change Point Detection in LLM Misalignment Training

Data files are NOT committed to git due to size. Follow download instructions below.

## Summary

| Dataset | Source | Size | Purpose | Location |
|---------|--------|------|---------|----------|
| BeaverTails | HuggingFace (PKU-Alignment) | 30K | LLM safety/harmfulness labels | datasets/beavertails/ |
| Alpaca | HuggingFace (tatsu-lab) | 52K | Instruction fine-tuning data | datasets/alpaca/ |
| HH-RLHF | HuggingFace (Anthropic) | 160K | Human preference data for RLHF | datasets/hh_rlhf/ |
| Do-Not-Answer | HuggingFace (LibrAI) | 939 | Safety evaluation prompts | datasets/do_not_answer/ |
| ToxicChat | HuggingFace (lmsys) | 5K | Toxic/jailbreak detection | datasets/toxic_chat/ |
| Synthetic CPD | Local | 630 pts | Known changepoints for BOCPD testing | datasets/welllog/ |
| Synthetic LLM Safety | Local | 200 steps | LLM safety rate with 3 known CPs | datasets/welllog/ |

---

## Dataset 1: BeaverTails

### Overview
- **Source**: HuggingFace `PKU-Alignment/BeaverTails`
- **Size**: 27,186 train / 3,021 test examples (30k split)
- **Format**: HuggingFace Dataset
- **Task**: Safety classification of LLM responses (safe/unsafe + 14 harm categories)
- **License**: CC-BY-NC-4.0
- **Key Features**: `prompt`, `response`, `is_safe`, `category` (14 harm types)

### Why Relevant
This dataset provides **labeled sequences of safe and unsafe LLM responses** to prompts. It is used in the Alignment Dynamics (2026) paper as a primary benchmark. For our BOCPD research, it enables:
- Constructing time series of safety scores during fine-tuning
- Simulating misalignment emergence by training on harmful subsets
- Evaluating whether BOCPD detects the safety phase transition

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/BeaverTails")
train = dataset["30k_train"]  # 27,186 examples
test = dataset["30k_test"]    # 3,021 examples
train.save_to_disk("datasets/beavertails/train")
test.save_to_disk("datasets/beavertails/test")
```

### Loading
```python
from datasets import load_from_disk
train = load_from_disk("datasets/beavertails/train")
```

---

## Dataset 2: Alpaca

### Overview
- **Source**: HuggingFace `tatsu-lab/alpaca`
- **Size**: 52,002 examples
- **Format**: HuggingFace Dataset
- **Task**: Instruction following (fine-tuning data)
- **License**: CC-BY-NC-4.0
- **Key Features**: `instruction`, `input`, `output`, `text`

### Why Relevant
Used extensively in LLM safety fine-tuning research. The Alignment Dynamics paper uses Alpaca-cleaned for warm-up fine-tuning. Critical for our research because Qi et al. (2024) showed that even benign Alpaca fine-tuning can degrade safety guardrails — making it ideal for simulating the apparent "null result" scenario our hypothesis addresses.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset.save_to_disk("datasets/alpaca")
```

---

## Dataset 3: HH-RLHF (Anthropic Human Feedback)

### Overview
- **Source**: HuggingFace `Anthropic/hh-rlhf`
- **Size**: 160,800 train / 8,552 test pairs
- **Format**: HuggingFace Dataset (chosen/rejected pairs)
- **Task**: Helpfulness and harmlessness alignment via RLHF
- **License**: MIT
- **Key Features**: `chosen` (preferred response), `rejected` (dispreferred response)

### Why Relevant
Real human preference data for LLM alignment training. Can be used to construct sequences of alignment training steps and observe behavioral changes over time — the core time series for BOCPD analysis.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset.save_to_disk("datasets/hh_rlhf/train")
```

---

## Dataset 4: Do-Not-Answer

### Overview
- **Source**: HuggingFace `LibrAI/do-not-answer`
- **Size**: 939 examples
- **Format**: HuggingFace Dataset
- **Task**: LLM safety evaluation (harmful question prompts)
- **License**: Apache 2.0
- **Key Features**: `risk_area`, `types_of_harm`, `question`, model responses + safety labels for multiple models

### Why Relevant
Contains pre-labeled model responses from multiple LLMs (GPT-4, ChatGPT, Claude, Llama2, etc.) with safety judgments. Provides a **multi-model reference** for what constitutes harmful behavior across different architectures.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("LibrAI/do-not-answer", split="train")
dataset.save_to_disk("datasets/do_not_answer")
```

---

## Dataset 5: ToxicChat

### Overview
- **Source**: HuggingFace `lmsys/toxic-chat`
- **Size**: 5,082 train / 5,083 test examples
- **Format**: HuggingFace Dataset
- **Task**: Toxicity detection in user-LLM interactions
- **License**: CC-BY-NC-4.0

### Why Relevant
Real-world LLM interaction data with toxicity labels, useful for evaluating whether behavioral phase transitions correlate with toxicity metrics.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
dataset.save_to_disk("datasets/toxic_chat")
```

---

## Dataset 6: Synthetic Changepoint Data (Local)

### Overview
- **Source**: Generated locally (see datasets/welllog/)
- **Files**: 
  - `synthetic_changepoint_data.json`: Well-log-like numerical data, 630 points, 4 true CPs
  - `synthetic_llm_safety_series.json`: LLM safety rate time series, 200 steps, 3 true CPs
- **Purpose**: Benchmarking BOCPD implementations before applying to real LLM data

### Loading
```python
import json
with open("datasets/welllog/synthetic_llm_safety_series.json") as f:
    data = json.load(f)
# data["safety_rate_sequence"]: list of 200 safety rates
# data["true_phase_boundaries"]: [50, 100, 150]
# data["phases"]: ["aligned", "degrading", "misaligned", "re-aligned"]
```

---

## Notes for Experiment Runner

1. **Primary dataset for BOCPD experiment**: BeaverTails + Alpaca
   - Use Alpaca to fine-tune a small aligned LLM
   - Inject BeaverTails harmful examples at varying rates
   - Track safety score on BeaverTails test set per training step
   - Apply BOCPD to the resulting safety score time series

2. **Synthetic data for validation**: Start with `synthetic_llm_safety_series.json` to validate BOCPD implementation before scaling to real LLM experiments

3. **Multi-model evaluation**: Do-Not-Answer provides pre-computed responses across multiple models for cross-architecture comparison

4. **HH-RLHF for RLHF dynamics**: Use chosen/rejected pairs to simulate realistic alignment training followed by counter-alignment fine-tuning
