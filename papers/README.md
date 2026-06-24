# Downloaded Papers

Papers for: Bayesian Change Point Detection for Uncovering Latent Phase Shifts in LLM Misalignment Training

## Bayesian Change Point Detection (Core Methods)

1. [Bayesian Online Changepoint Detection](0710.3742v1_Bayesian_Online_Changepoint_Detection_Adams2007.pdf)
   - **Authors**: Ryan Prescott Adams, David J.C. MacKay
   - **Year**: 2007
   - **arXiv**: 0710.3742
   - **Why relevant**: THE foundational BOCPD paper. Introduces recursive run-length estimation with message-passing algorithm. Derives online exact Bayesian inference for changepoint locations. Compatible with any conjugate-exponential likelihood.

2. [Robust and Scalable Bayesian Online Changepoint Detection](2302.04759v2_Robust_Scalable_BOCPD_2023.pdf)
   - **Authors**: Matias Altamirano, François-Xavier Briol, Jeremias Knoblauch
   - **Year**: 2023
   - **arXiv**: 2302.04759
   - **Why relevant**: Introduces Dm-BOCD using diffusion score matching for robustness to outliers. 10x faster than β-BOCD. Published at ICML 2023.

3. [Bayesian Online Change Point Detection for Baseline Shifts](2201.02325v1_BOCPD_Baseline_Shifts_2022.pdf)
   - **Authors**: Multiple
   - **Year**: 2022
   - **arXiv**: 2201.02325
   - **Why relevant**: Extension of BOCPD specifically designed for detecting mean/baseline shifts — directly applicable to detecting safety score transitions in LLM training.

4. [Sequential Changepoint Detection in Neural Networks with Checkpoints](2010.03053v1_Sequential_CPD_Neural_Networks_Checkpoints_2020.pdf)
   - **Authors**: Multiple
   - **Year**: 2020
   - **arXiv**: 2010.03053
   - **Why relevant**: Directly applies BOCPD framework to neural network training, tracking parameter or metric changes at checkpoints. Most directly relevant prior work to our research.

5. [Bayesian Online Prediction of Change Points](1902.04524v2_Bayesian_Online_Prediction_Changepoints_2019.pdf)
   - **Authors**: Multiple
   - **Year**: 2019
   - **arXiv**: 1902.04524
   - **Why relevant**: Extension for prediction at changepoints; discusses regime identification.

6. [Confirmatory Bayesian Online Change Point Detection in the Covariance Structure](1905.13168v2_Confirmatory_BOCPD_Covariance_2019.pdf)
   - **Authors**: Multiple
   - **Year**: 2019
   - **arXiv**: 1905.13168
   - **Why relevant**: Handles multivariate changepoint detection in covariance — relevant for tracking multiple LLM behavioral metrics simultaneously.

7. [Lagged Exact Bayesian Online Changepoint Detection with Parameter Estimation](1710.03276v3_Lagged_Exact_BOCPD_Parameter_Estimation_2017.pdf)
   - **Authors**: Multiple
   - **Year**: 2017
   - **arXiv**: 1710.03276
   - **Why relevant**: Introduces parameter estimation (not just detection) allowing characterization of each phase's distribution.

8. [Sequential Kalman Filter for Fast Online Changepoint Detection](2310.18611v3_Sequential_Kalman_Changepoint_2023.pdf)
   - **Authors**: Multiple
   - **Year**: 2023
   - **arXiv**: 2310.18611
   - **Why relevant**: Scalable variant for temporally correlated data — important if LLM behaviors exhibit autocorrelation.

9. [Online Neural Networks for Change-Point Detection](2010.01388v2_Online_Neural_Networks_CPD_2020.pdf)
   - **Authors**: Multiple
   - **Year**: 2020
   - **arXiv**: 2010.01388
   - **Why relevant**: Neural network-based online CPD — provides comparison baseline for our Bayesian approach.

10. [Active Multi-Fidelity Bayesian Online Changepoint Detection](2103.14224v2_Active_Multifidelity_BOCPD_2021.pdf)
    - **Authors**: Multiple
    - **Year**: 2021
    - **arXiv**: 2103.14224
    - **Why relevant**: Active learning variant for budget-constrained settings (important when LLM evaluations are expensive).

11. [Bayesian Online Collective Anomaly and Change Point Detection in Fine-Grained Time Series](2508.06385v1_Bayesian_Collective_Anomaly_CPD_Finegrained_2025.pdf)
    - **Authors**: Multiple
    - **Year**: 2025
    - **arXiv**: 2508.06385
    - **Why relevant**: Most recent BOCPD work; handles fine-grained time series which maps to per-step training metrics.

## LLM Alignment, Safety, and Misalignment

12. [Alignment Dynamics in LLM Fine-Tuning](2605.18309v1_Alignment_Dynamics_LLM_FineTuning_2026.pdf)
    - **Authors**: Yuhan Huang, Huanran Chen, Yinpeng Dong
    - **Year**: 2026
    - **arXiv**: 2605.18309
    - **Why relevant**: Derives closed-form alignment score dynamics during fine-tuning, revealing Rebound Force + Driving Force decomposition. Discovers "Rehearsal Priming Effect" — exactly the kind of latent dynamic our BOCPD should detect.

13. [Understanding and Preserving Safety in Fine-Tuned LLMs](2601.10141v1_Understanding_Preserving_Safety_FineTuned_LLMs_2026.pdf)
    - **Authors**: Multiple
    - **Year**: 2026
    - **arXiv**: 2601.10141
    - **Why relevant**: Comprehensive treatment of safety degradation mechanisms during fine-tuning.

14. [Why LLM Safety Guardrails Collapse After Fine-tuning](2506.05346v1_LLM_Safety_Guardrails_Collapse_FineTuning_2025.pdf)
    - **Authors**: Lei Hsiung et al.
    - **Year**: 2025
    - **arXiv**: 2506.05346
    - **Why relevant**: Identifies representation similarity as root cause of safety collapse — suggests that safety collapses are not smooth but abrupt (changepoint-like).

15. [Safety Layers in Aligned Large Language Models](2408.17003v5_Safety_Layers_Aligned_LLMs_2024.pdf)
    - **Authors**: Multiple
    - **Year**: 2024
    - **arXiv**: 2408.17003
    - **Why relevant**: Locates safety behavior in specific model layers — provides mechanistic grounding for why behavioral phase transitions occur.

16. [Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors](2407.15549v3_Latent_Adversarial_Training_Robustness_2024.pdf)
    - **Authors**: Multiple
    - **Year**: 2024
    - **arXiv**: 2407.15549
    - **Why relevant**: Addresses the exact problem of hidden harmful behaviors that persist through safety training — demonstrates latent behavioral states exist.

17. [SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning](2506.00676v1_SafeTuneBed_LLM_Safety_Alignment_2025.pdf)
    - **Authors**: Multiple
    - **Year**: 2025
    - **arXiv**: 2506.00676
    - **Why relevant**: Provides standardized toolkit for measuring safety during fine-tuning — the exact type of evaluation infrastructure needed for our experiment.

18. [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](2401.05566v3_Sleeper_Agents_Training_Deceptive_LLMs_2024.pdf)
    - **Authors**: Evan Hubinger, Carson Denison et al. (Anthropic)
    - **Year**: 2024
    - **arXiv**: 2401.05566
    - **Why relevant**: Demonstrates that deceptive behavior (misalignment) can be INVISIBLE to aggregate metrics while persisting through safety training. Direct motivation for our BOCPD approach to detect latent phase shifts.

19. [Detecting Sleeper Agents in LLMs via Semantic Drift Analysis](2511.15992v1_Detecting_Sleeper_Agents_LLMs_Semantic_Drift_2025.pdf)
    - **Authors**: Multiple
    - **Year**: 2025
    - **arXiv**: 2511.15992
    - **Why relevant**: Detection approach for sleeper agents that could be enhanced by BOCPD.

20. [When RLHF Fails: A Mechanistic Taxonomy of Reward Hacking, Collapse, and Evasion](2606.03238v1_When_RLHF_Fails_Reward_Hacking_Collapse_2026.pdf)
    - **Authors**: Multiple
    - **Year**: 2026
    - **arXiv**: 2606.03238
    - **Why relevant**: Mechanistic taxonomy of RLHF failure modes — identifies collapse patterns that should appear as changepoints in behavioral metrics.

## Phase Transitions and Grokking

21. [Grokking as Dimensional Phase Transition in Neural Networks](2604.04655v1_Grokking_Dimensional_Phase_Transition_2026.pdf)
    - **Authors**: Multiple
    - **Year**: 2026
    - **arXiv**: 2604.04655
    - **Why relevant**: Maps grokking (abrupt generalization) to dimensional phase transitions — theoretical framework applicable to misalignment emergence.

22. [Grokking as a First Order Phase Transition in Two Layer Networks](2310.03789v3_Grokking_First_Order_Phase_Transition_2023.pdf)
    - **Authors**: Noa Rubin, Inbar Seroussi, Zohar Ringel
    - **Year**: 2023 (ICLR 2024)
    - **arXiv**: 2310.03789
    - **Why relevant**: Rigorous mapping between neural network learning phases and first-order phase transitions (Landau theory). Identifies three distinct learning phases (GFL, GMFL-I, GMFL-II) directly analogous to alignment → misalignment → re-alignment.

23. [Information-Theoretic Progress Measures Reveal Grokking is an Emergent Phase Transition](2408.08944v1_Grokking_Emergent_Phase_Transition_Info_Theory_2024.pdf)
    - **Authors**: Multiple
    - **Year**: 2024
    - **arXiv**: 2408.08944
    - **Why relevant**: Information-theoretic detection of phase transitions — complementary approach to BOCPD for validating detected changepoints.

## Mechanistic Interpretability and Behavioral Analysis

24. [Steering Language Models With Activation Engineering](2308.10248v5_Steering_Language_Models_Activation_Engineering_2023.pdf)
    - **Authors**: Multiple
    - **Year**: 2023
    - **arXiv**: 2308.10248
    - **Why relevant**: Provides technical framework for measuring alignment state in activation space — defines observable metrics that BOCPD can track.

25. [Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs](2603.03371v1_Sleeper_Cell_Temporal_Backdoors_ToolUsing_LLMs_2026.pdf)
    - **Authors**: Multiple
    - **Year**: 2026
    - **arXiv**: 2603.03371
    - **Why relevant**: Recent extension of sleeper agents to temporal backdoors — demonstrates time-conditioned behavior changes detectable via BOCPD.
