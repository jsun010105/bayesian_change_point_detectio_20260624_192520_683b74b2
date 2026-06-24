"""
Bayesian Online Changepoint Detection core implementation.
Implements BOCPD with Gaussian (Normal-Inverse-Gamma) likelihood.
"""

import numpy as np
from scipy.special import logsumexp, gammaln
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class NormalInverseGamma:
    """
    Normal-Inverse-Gamma conjugate prior for univariate Gaussian data with unknown mean and variance.
    Parameters: mu0 (prior mean), kappa (prior observations), alpha (shape), beta (rate).
    """
    mu0: float = 0.0
    kappa: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0

    def log_pred_prob(self, x: float) -> float:
        """Log predictive probability of x under Student-t predictive distribution."""
        mu0, kappa, alpha, beta = self.mu0, self.kappa, self.alpha, self.beta
        # Predictive is Student-t with 2*alpha degrees of freedom
        df = 2 * alpha
        scale_sq = beta * (kappa + 1) / (kappa * alpha)
        log_prob = (
            gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df * scale_sq)
            - ((df + 1) / 2) * np.log(1 + (x - mu0) ** 2 / (df * scale_sq))
        )
        return log_prob

    def update(self, x: float) -> "NormalInverseGamma":
        """Bayesian update with observation x. Returns posterior parameters."""
        mu0, kappa, alpha, beta = self.mu0, self.kappa, self.alpha, self.beta
        kappa_new = kappa + 1
        mu0_new = (kappa * mu0 + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + (kappa * (x - mu0) ** 2) / (2 * kappa_new)
        return NormalInverseGamma(mu0_new, kappa_new, alpha_new, beta_new)


class BocpdGaussian:
    """
    Bayesian Online Changepoint Detection with Normal-Inverse-Gamma likelihood.
    Tracks run-length distribution P(r_t | x_{1:t}).
    """

    def __init__(self, hazard_rate: float = 0.01, prior: Optional[NormalInverseGamma] = None):
        """
        Args:
            hazard_rate: Probability of a changepoint at each step (1/expected_segment_length).
            prior: Prior parameters for the Normal-Inverse-Gamma model.
        """
        self.hazard_rate = hazard_rate
        self.prior = prior or NormalInverseGamma()
        self._reset()

    def _reset(self):
        self.t = 0
        # Log run-length distribution: log_R[t, r] = log P(r_t = r | x_{1:t})
        # We store as a list of (run_length -> log_prob) pairs for efficiency
        self.log_R = np.array([0.0])  # At t=0, r=0 with prob 1
        self.params = [self.prior]    # One NIG param set per run length
        self.max_run = 0

    def step(self, x: float) -> np.ndarray:
        """
        Process one observation. Returns the run-length posterior at time t.

        Correct BOCPD (Adams & MacKay 2007) formulation:
          P(r_t=0, x_{1:t}) = H * P(x_t | fresh_prior) * sum_r P(r_{t-1}=r)
                             = H * P(x_t | params[0])   [since sum_r P(r_{t-1}=r) = 1]

          P(r_t=r+1, x_{1:t}) = (1-H) * P(x_t | params[r]) * P(r_{t-1}=r)

        The changepoint term uses ONLY the fresh prior (params[0]), not segment-
        specific posteriors. This is what makes BOCPD detect distribution shifts.
        """
        self.t += 1
        n = len(self.log_R)  # Current number of possible run lengths

        # Compute log predictive probabilities for each run length
        log_preds = np.array([p.log_pred_prob(x) for p in self.params])

        # Growth probabilities: r_t = r_{t-1} + 1 (for each current run length r)
        log_growth = log_preds + self.log_R + np.log(1 - self.hazard_rate)

        # Changepoint probability: r_t = 0
        # Uses ONLY fresh prior predictive (log_preds[0] = P(x_t | prior))
        # because after a changepoint, x_t is the first observation of the new segment
        log_cp = np.log(self.hazard_rate) + log_preds[0]

        # New run-length distribution
        new_log_R = np.empty(n + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth

        # Normalize
        log_Z = logsumexp(new_log_R)
        new_log_R -= log_Z
        self.log_R = new_log_R

        # Update sufficient statistics for each run length
        new_params = [self.prior]  # New segment starts fresh with prior
        for p in self.params:
            new_params.append(p.update(x))
        self.params = new_params

        return np.exp(self.log_R)

    def run_length_posterior(self) -> np.ndarray:
        """Return current run-length posterior P(r_t | x_{1:t})."""
        return np.exp(self.log_R)

    def changepoint_probability(self) -> float:
        """Probability that a changepoint occurred at current step (r_t = 0 or 1)."""
        return float(np.exp(self.log_R[0]))

    def most_likely_run_length(self) -> int:
        """Return the most likely current run length."""
        return int(np.argmax(self.log_R))


@dataclass
class BetaBernoulli:
    """
    Beta-Bernoulli conjugate model for binary (0/1) Bernoulli data.
    alpha = prior successes, beta = prior failures.
    """
    alpha: float = 1.0
    beta: float = 1.0

    def log_pred_prob(self, x: float) -> float:
        """Log predictive prob of binary x under Beta-Bernoulli model."""
        x = float(round(x))  # ensure binary
        if x == 1.0:
            return np.log(self.alpha / (self.alpha + self.beta))
        else:
            return np.log(self.beta / (self.alpha + self.beta))

    def update(self, x: float) -> "BetaBernoulli":
        x = float(round(x))
        if x == 1.0:
            return BetaBernoulli(self.alpha + 1, self.beta)
        else:
            return BetaBernoulli(self.alpha, self.beta + 1)


def run_bocpd_bernoulli(series: np.ndarray, hazard_rate: float = 0.02,
                         alpha0: float = 1.0, beta0: float = 1.0) -> Tuple[list, np.ndarray, List[int]]:
    """Run BOCPD with Beta-Bernoulli likelihood for binary series."""
    prior = BetaBernoulli(alpha0, beta0)
    T = len(series)

    log_R = np.array([0.0])
    params = [prior]
    log_R_all = []
    cp_probs = np.zeros(T)

    for t, x in enumerate(series):
        n = len(log_R)
        log_preds = np.array([p.log_pred_prob(x) for p in params])

        log_growth = log_preds + log_R + np.log(1 - hazard_rate)
        log_cp = np.log(hazard_rate) + log_preds[0]

        new_log_R = np.empty(n + 1)
        new_log_R[0] = log_cp
        new_log_R[1:] = log_growth
        log_Z = logsumexp(new_log_R)
        new_log_R -= log_Z
        log_R = new_log_R
        log_R_all.append(np.exp(log_R).copy())
        cp_probs[t] = np.exp(log_R[0])

        new_params = [prior]
        for p in params:
            new_params.append(p.update(x))
        params = new_params

    detected = list(np.where(cp_probs > 0.5)[0])
    return log_R_all, cp_probs, detected


def run_bocpd(series: np.ndarray, hazard_rate: float = 0.01,
              prior: Optional[NormalInverseGamma] = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Run BOCPD on a time series.

    Returns:
        R: run-length posterior matrix (T x T+1)
        cp_probs: changepoint probability at each step
        detected_cps: list of step indices where changepoint probability > 0.5
    """
    T = len(series)
    detector = BocpdGaussian(hazard_rate=hazard_rate, prior=prior)

    R_list = []
    cp_probs = np.zeros(T)

    for t, x in enumerate(series):
        r_post = detector.step(x)
        R_list.append(r_post)
        cp_probs[t] = detector.changepoint_probability()

    detected_cps = list(np.where(cp_probs > 0.5)[0])
    return R_list, cp_probs, detected_cps


def cusum_test(series: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, List[int]]:
    """
    CUSUM sequential changepoint test.
    Detects mean shifts using cumulative sum of standardized residuals.
    """
    mu = np.mean(series)
    sigma = np.std(series)
    if sigma < 1e-10:
        return np.zeros(len(series)), []

    z = (series - mu) / sigma
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))

    for t in range(1, len(series)):
        cusum_pos[t] = max(0, cusum_pos[t-1] + z[t] - 0.5)
        cusum_neg[t] = max(0, cusum_neg[t-1] - z[t] - 0.5)

    cusum = np.maximum(cusum_pos, cusum_neg)
    detected = list(np.where(cusum > threshold)[0])
    # Deduplicate consecutive detections
    deduplicated = []
    prev = -10
    for d in detected:
        if d - prev > 5:
            deduplicated.append(d)
            prev = d
    return cusum, deduplicated


def moving_average_detector(series: np.ndarray, window: int = 10,
                              threshold_sigma: float = 2.0) -> Tuple[np.ndarray, List[int]]:
    """
    Moving average baseline: flag points where series deviates from local MA by > threshold_sigma.
    """
    n = len(series)
    ma = np.zeros(n)
    detected = []

    for t in range(n):
        start = max(0, t - window)
        window_data = series[start:t] if t > 0 else series[0:1]
        if len(window_data) < 2:
            ma[t] = series[t]
            continue
        ma[t] = np.mean(window_data)
        sigma = np.std(window_data)
        if sigma > 1e-10 and abs(series[t] - ma[t]) > threshold_sigma * sigma:
            detected.append(t)

    # Deduplicate
    deduplicated = []
    prev = -10
    for d in detected:
        if d - prev > 5:
            deduplicated.append(d)
            prev = d
    return ma, deduplicated


def kl_divergence_empirical(samples_a: np.ndarray, samples_b: np.ndarray,
                             n_bins: int = 20) -> float:
    """
    Empirical KL divergence between two sample sets using binning.
    """
    all_data = np.concatenate([samples_a, samples_b])
    bins = np.linspace(all_data.min() - 0.01, all_data.max() + 0.01, n_bins + 1)

    hist_a, _ = np.histogram(samples_a, bins=bins, density=True)
    hist_b, _ = np.histogram(samples_b, bins=bins, density=True)

    # Add small epsilon for numerical stability
    eps = 1e-10
    hist_a = hist_a + eps
    hist_b = hist_b + eps

    # Normalize
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()

    return float(np.sum(hist_a * np.log(hist_a / hist_b)))


def permutation_test_kl(series: np.ndarray, cp: int, n_permutations: int = 1000,
                         seed: int = 42) -> Tuple[float, float]:
    """
    Permutation test for significance of KL divergence at a detected changepoint.

    Returns: (observed_kl, p_value)
    """
    rng = np.random.default_rng(seed)
    pre = series[:cp]
    post = series[cp:]

    if len(pre) < 5 or len(post) < 5:
        return 0.0, 1.0

    observed_kl = kl_divergence_empirical(pre, post)

    # Permutation distribution
    combined = np.concatenate([pre, post])
    null_kls = []
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        null_pre = perm[:len(pre)]
        null_post = perm[len(pre):]
        null_kls.append(kl_divergence_empirical(null_pre, null_post))

    p_value = float(np.mean(np.array(null_kls) >= observed_kl))
    return observed_kl, p_value


def detection_metrics(detected: List[int], true_cps: List[int],
                       tolerance: int = 5) -> dict:
    """
    Compute detection F1, precision, recall, and average delay.
    """
    tp = 0
    matched_true = set()
    delays = []

    for d in detected:
        for t in true_cps:
            if t not in matched_true and abs(d - t) <= tolerance:
                tp += 1
                matched_true.add(t)
                delays.append(d - t)
                break

    fp = len(detected) - tp
    fn = len(true_cps) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_delay = float(np.mean(delays)) if delays else float('inf')

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'avg_delay': avg_delay,
        'n_detected': len(detected)
    }
