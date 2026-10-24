"""Small statistics helpers for comparing NFN scores and placements."""

import math

from scipy.stats import spearmanr


def spearman(a, b):
    """Rank correlation of two equal-length score lists (1 = same order, -1 = reversed)."""
    return float(spearmanr(a, b).statistic)


def pick(scores, k=3):
    """The k lowest-scoring types, as a set (PLoP's choice)."""
    return set(sorted(scores, key=scores.get)[:k])


def pick_change_rate(score_sets, k=3):
    """Fraction of pairs among `score_sets` (list of {type: score}) whose PLoP picks differ."""
    picks = [pick(s, k) for s in score_sets]
    pairs = [(i, j) for i in range(len(picks)) for j in range(i + 1, len(picks))]
    return sum(picks[i] != picks[j] for i, j in pairs) / len(pairs)


def mean_se(xs):
    """Mean and standard error of the mean."""
    n = len(xs)
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float("nan"))
