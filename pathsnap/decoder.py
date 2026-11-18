from __future__ import annotations
import numpy as np


def viterbi(candidate_ids, probabilities, arcs, gap_seconds=None, transition_weight=1.8):
    """Decode one candidate edge per observation using network connectivity.

    Connected directed edges receive a small transition cost, disconnected jumps
    receive a larger cost. Large GPS gaps reduce the continuity penalty.
    """
    n, k = probabilities.shape
    if n == 0:
        return np.array([], dtype=int)
    edge_map = {tuple(map(int, edge)): i for i, edge in enumerate(arcs)}
    endpoints = arcs[candidate_ids]
    logs = np.log(np.clip(probabilities, 1e-8, 1.0))
    score = np.full((n, k), -np.inf)
    back = np.zeros((n, k), dtype=int)
    score[0] = logs[0]
    for t in range(1, n):
        gap_scale = 0.25 if gap_seconds is not None and gap_seconds[t] > 30 else 1.0
        for j in range(k):
            prev = endpoints[t-1]
            curr = endpoints[t, j]
            connected = (prev[:, 1] == curr[0])
            # A legal edge transition is preferred. Disconnected transitions remain
            # possible to recover from GPS gaps and incomplete local graphs.
            cost = np.where(connected, 0.0, transition_weight * gap_scale)
            values = score[t-1] - cost
            best = int(np.argmax(values))
            score[t, j] = values[best] + logs[t, j]
            back[t, j] = best
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n-1, 0, -1):
        path[t-1] = back[t, path[t]]
    return candidate_ids[np.arange(n), path]

