"""Statistical utilities for evaluation comparisons.

Provides bootstrap confidence intervals and paired permutation tests for
comparing model capability scores. No external dependencies — stdlib only.
"""

import math
import random


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        scores: Per-case scores (one per test case).
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 = 95% CI).
        seed: Optional RNG seed for reproducibility.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not scores:
        return 0.0, 0.0, 0.0
    if len(scores) == 1:
        return scores[0], scores[0], scores[0]

    rng = random.Random(seed)
    n = len(scores)
    mean = sum(scores) / n

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(scores, k=n)
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return (
        round(mean, 4),
        round(bootstrap_means[lo_idx], 4),
        round(bootstrap_means[hi_idx], 4),
    )


def scores_significantly_different(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
    n_permutations: int = 10000,
    seed: int | None = None,
) -> tuple[bool, float]:
    """Paired permutation test for difference in means.

    Better than t-test for small samples (n=7-10). Tests the null hypothesis
    that the two sets of scores come from the same distribution.

    Args:
        scores_a: Per-case scores for model A.
        scores_b: Per-case scores for model B (same cases, same order).
        alpha: Significance level.
        n_permutations: Number of random permutations.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (is_significant, p_value)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score lists must have same length: {len(scores_a)} vs {len(scores_b)}"
        )
    n = len(scores_a)
    if n == 0:
        return False, 1.0

    # Observed difference in means
    observed_diff = abs(sum(scores_a) / n - sum(scores_b) / n)

    # Paired differences
    diffs = [a - b for a, b in zip(scores_a, scores_b)]

    rng = random.Random(seed)
    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly flip the sign of each paired difference
        perm_diff = sum(d * rng.choice([-1, 1]) for d in diffs) / n
        if abs(perm_diff) >= observed_diff:
            count_extreme += 1

    p_value = round(count_extreme / n_permutations, 4)
    return p_value <= alpha, p_value


def format_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format a confidence interval for display.

    Example: "0.87 [0.79, 0.94]"
    """
    return f"{mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
