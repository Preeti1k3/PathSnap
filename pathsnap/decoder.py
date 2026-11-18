from __future__ import annotations
import numpy as np


def viterbi(candidate_ids, probabilities, arcs, gap_seconds=None, transition_weight=1.8):
    """Find the maximum-score candidate-edge sequence with Viterbi dynamic programming.

    Hidden state is a road-arc candidate for each observation. Emissions are the
    classifier's normalized candidate probabilities. For each prior state,
    transition logits are 0 for staying on an arc or moving to an edge whose
    ``from`` node matches the prior edge's ``to`` node; disconnected transitions
    receive ``-transition_weight``. Row-wise softmax turns these into conditional
    transition probabilities. A gap over 30 seconds scales the penalty by 0.25 so
    long GPS outages can recover. Disconnected transitions remain possible because
    local road graphs can be incomplete.
    """
    n, k = probabilities.shape
    if n == 0:
        return np.array([], dtype=int)
    if candidate_ids.shape != probabilities.shape:
        raise ValueError("candidate_ids and probabilities must have the same (observations, candidates) shape")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("emission probabilities must be finite and nonnegative")
    if (candidate_ids < 0).any() or (candidate_ids >= len(arcs)).any():
        raise ValueError("candidate arc ID is outside the road-network arc array")
    endpoints = arcs[candidate_ids]
    logs = np.log(np.clip(probabilities, 1e-8, 1.0))
    score = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    score[0] = logs[0]
    for t in range(1, n):
        gap_scale = 0.25 if gap_seconds is not None and gap_seconds[t] > 30 else 1.0
        previous_ids = candidate_ids[t-1]
        current_ids = candidate_ids[t]
        previous = endpoints[t-1]
        current = endpoints[t]
        connected = ((previous[:, 1, None] == current[None, :, 0]) |
                     (previous_ids[:, None] == current_ids[None, :]))
        logits = np.where(connected, 0.0, -transition_weight * gap_scale)
        max_logits = np.max(logits, axis=1, keepdims=True)
        log_transition = logits - max_logits - np.log(
            np.exp(logits-max_logits).sum(axis=1, keepdims=True)
        )
        values = score[t-1, :, None] + log_transition
        best_prev = np.argmax(values, axis=0)
        score[t] = values[best_prev, np.arange(k)] + logs[t]
        back[t] = best_prev
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n-1, 0, -1):
        path[t-1] = back[t, path[t]]
    return candidate_ids[np.arange(n), path]
