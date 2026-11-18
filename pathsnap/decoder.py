"""
pathsnap/decoder.py - Sequence-based HMM / Viterbi route reconstruction for Map-Matching.

Responsible for:
- Formulating candidate road segments as hidden states.
- Modeling emissions from candidate classifier probabilities or spatial/heading distances.
- Modeling transitions using road-network topology, continuity, temporal gaps, and U-turn penalties.
- Executing the Viterbi dynamic programming recurrence in log-space.
- Reconstructing the optimal continuous sequence of road segments (arcs).
- Evaluating pointwise accuracy, sequence accuracy, and route continuity.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class ViterbiDecoder:
    """
    Hidden Markov Model (HMM) Viterbi Decoder for GPS Map-Matching.

    Attributes:
        same_arc_prob: Prior probability of remaining on the current road segment (vehicle driving along edge).
        forward_arc_prob: Probability of transitioning to a directly connected forward segment (prev_v == curr_u).
        u_turn_penalty: Penalty applied when switching to the reverse direction on the same road (prev_u == curr_v and prev_v == curr_u).
        two_hop_prob: Probability of transition via a 1-hop intermediate node.
        continuity_penalty: Minimum transition floor applied when candidate segments are disconnected.
        gap_threshold_s: Time gap in seconds beyond which continuity constraints are relaxed.
        min_prob: Numerical epsilon floor to prevent log(0) in dynamic programming.
    """

    def __init__(
        self,
        same_arc_prob: float = 0.65,
        forward_arc_prob: float = 0.30,
        u_turn_penalty: float = 1e-5,
        two_hop_prob: float = 0.04,
        continuity_penalty: float = 1e-5,
        gap_threshold_s: float = 5.0,
        min_prob: float = 1e-12,
    ):
        self.same_arc_prob = same_arc_prob
        self.forward_arc_prob = forward_arc_prob
        self.u_turn_penalty = u_turn_penalty
        self.two_hop_prob = two_hop_prob
        self.continuity_penalty = continuity_penalty
        self.gap_threshold_s = gap_threshold_s
        self.min_prob = min_prob

    def build_adjacency(self, arcs: np.ndarray) -> Dict[int, List[int]]:
        """
        Builds a map from from_node -> list of outgoing arc_ids.
        
        Args:
            arcs: (N, 2) array of [from_node, to_node] indexed by arc_id.
        """
        adj: Dict[int, List[int]] = {}
        for arc_id in range(len(arcs)):
            u = int(arcs[arc_id, 0])
            if u not in adj:
                adj[u] = []
            adj[u].append(arc_id)
        return adj

    def transition_probability(
        self,
        prev_arc: int,
        curr_arc: int,
        arcs: np.ndarray,
        dt: float,
        adj: Optional[Dict[int, List[int]]] = None,
    ) -> float:
        """
        Calculates P(curr_arc at t | prev_arc at t-1) using road network topology,
        heading continuity, and temporal gaps between GPS observations.
        """
        # Case 1: Continuing along the same road segment
        if prev_arc == curr_arc:
            return self.same_arc_prob

        prev_u = int(arcs[prev_arc, 0])
        prev_v = int(arcs[prev_arc, 1])
        curr_u = int(arcs[curr_arc, 0])
        curr_v = int(arcs[curr_arc, 1])

        # Case 2: Immediate 180-degree U-turn reversal on a bidirectional road segment
        # In physical vehicle tracking, oscillating back-and-forth between opposing lanes is invalid.
        if prev_u == curr_v and prev_v == curr_u:
            return self.u_turn_penalty

        # Case 3: Direct topological forward connection (end node of prev == start node of curr)
        if prev_v == curr_u:
            return self.forward_arc_prob

        # Case 4: Temporal gap (GPS loss, tunnel, signal dropout) -> relax topological constraint
        if dt > self.gap_threshold_s:
            gap_factor = min(1.0, dt / 30.0)
            return max(self.continuity_penalty, 0.05 * gap_factor)

        # Case 5: 2-hop adjacency (prev_arc -> intermediate -> curr_arc)
        if adj is not None and prev_v in adj:
            for intermediate_arc in adj[prev_v]:
                inter_v = int(arcs[intermediate_arc, 1])
                if inter_v == curr_u:
                    return self.two_hop_prob

        # Case 6: Disconnected jump
        return self.continuity_penalty

    def decode(
        self,
        candidate_arcs_per_step: List[List[int]],
        candidate_probs_per_step: List[List[float]],
        arcs: np.ndarray,
        timestamps: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Performs Viterbi dynamic programming in log-space to find the most
        probable sequence of road arcs.

        Args:
            candidate_arcs_per_step: Outer list length T, inner list containing candidate arc IDs.
            candidate_probs_per_step: Outer list length T, inner list containing candidate emission probabilities.
            arcs: (N, 2) array of [from_node, to_node].
            timestamps: Optional list of timestamps (seconds) length T.

        Returns:
            List of length T containing the optimal road arc ID for each observation.
        """
        T = len(candidate_arcs_per_step)
        if T == 0:
            return []

        adj = self.build_adjacency(arcs)

        # dp[t][i] stores the log probability of the most likely path ending at candidate i at time t
        dp: List[np.ndarray] = []
        backpointer: List[np.ndarray] = []

        # Step 0: Initialization
        init_cands = candidate_arcs_per_step[0]
        init_probs = np.array(candidate_probs_per_step[0], dtype=float)
        
        # Normalize initial probabilities
        p_sum = init_probs.sum()
        if p_sum > 0:
            init_probs = init_probs / p_sum
        init_probs = np.maximum(init_probs, self.min_prob)
        
        dp.append(np.log(init_probs))
        backpointer.append(np.zeros(len(init_cands), dtype=int))

        # Forward Viterbi recursion
        for t in range(1, T):
            dt = 1.0
            if timestamps is not None and len(timestamps) > t:
                dt = max(float(timestamps[t] - timestamps[t - 1]), 0.1)

            prev_cands = candidate_arcs_per_step[t - 1]
            curr_cands = candidate_arcs_per_step[t]
            curr_probs = np.array(candidate_probs_per_step[t], dtype=float)

            # Normalize emission probabilities
            c_sum = curr_probs.sum()
            if c_sum > 0:
                curr_probs = curr_probs / c_sum
            curr_probs = np.maximum(curr_probs, self.min_prob)
            log_emissions = np.log(curr_probs)

            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            # Compute transition matrix between candidates at t-1 and t
            log_transitions = np.empty((K_prev, K_curr), dtype=float)
            for i, p_arc in enumerate(prev_cands):
                for j, c_arc in enumerate(curr_cands):
                    trans_p = self.transition_probability(p_arc, c_arc, arcs, dt, adj)
                    log_transitions[i, j] = np.log(max(trans_p, self.min_prob))

            # Vectorized Viterbi step:
            candidate_scores = dp[t - 1][:, np.newaxis] + log_transitions
            best_prev_indices = np.argmax(candidate_scores, axis=0)
            max_scores = candidate_scores[best_prev_indices, np.arange(K_curr)]

            dp.append(max_scores + log_emissions)
            backpointer.append(best_prev_indices)

        # Backtracking
        best_last_idx = int(np.argmax(dp[-1]))
        optimal_indices = [best_last_idx]

        for t in range(T - 1, 0, -1):
            prev_idx = int(backpointer[t][optimal_indices[-1]])
            optimal_indices.append(prev_idx)

        optimal_indices.reverse()

        # Map state indices back to arc IDs
        decoded_path = [
            candidate_arcs_per_step[t][optimal_indices[t]]
            for t in range(T)
        ]
        return decoded_path

    @staticmethod
    def compress_route(arc_sequence: List[int]) -> List[int]:
        """
        Compresses an observation-wise arc sequence into a unique chronological
        route by removing consecutive duplicate arcs, matching the ground-truth
        .route format.
        """
        if not arc_sequence:
            return []
        compressed = [arc_sequence[0]]
        for arc in arc_sequence[1:]:
            if arc != compressed[-1]:
                compressed.append(arc)
        return compressed

    @staticmethod
    def calculate_continuity(arc_sequence: List[int], arcs: np.ndarray) -> float:
        """
        Computes the route continuity score: the proportion of consecutive steps
        that are either identical or topologically connected in the road network.
        
        Returns:
            float in [0.0, 1.0].
        """
        if len(arc_sequence) <= 1:
            return 1.0
        connected = 0
        for i in range(len(arc_sequence) - 1):
            a1, a2 = arc_sequence[i], arc_sequence[i + 1]
            if a1 == a2:
                connected += 1
            elif int(arcs[a1, 1]) == int(arcs[a2, 0]):
                connected += 1
        return float(connected / (len(arc_sequence) - 1))

    @staticmethod
    def evaluate_pointwise(predicted_arcs: List[int], reference_arcs: List[int]) -> float:
        """
        Computes pointwise Top-1 accuracy against observation-level reference arcs.
        """
        if not predicted_arcs or not reference_arcs:
            return 0.0
        n = min(len(predicted_arcs), len(reference_arcs))
        matches = sum(1 for i in range(n) if predicted_arcs[i] == reference_arcs[i])
        return float(matches / n)

    @staticmethod
    def evaluate_route(
        predicted_route: List[int],
        reference_route: List[int],
    ) -> Dict[str, float]:
        """
        Computes sequence-level evaluation metrics:
        - Precision: fraction of predicted route arcs present in true route.
        - Recall: fraction of true route arcs captured by predicted route.
        - F1 Score: harmonic mean of Precision and Recall.
        - LCS Ratio: Longest Common Subsequence match ratio preserving order.
        """
        if not predicted_route or not reference_route:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "lcs_ratio": 0.0}

        pred_set = set(predicted_route)
        ref_set = set(reference_route)

        tp = len(pred_set & ref_set)
        precision = float(tp / len(pred_set)) if pred_set else 0.0
        recall = float(tp / len(ref_set)) if ref_set else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Longest Common Subsequence (LCS) ratio
        n, m = len(predicted_route), len(reference_route)
        dp = np.zeros((n + 1, m + 1), dtype=int)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if predicted_route[i - 1] == reference_route[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1] + 1
                else:
                    dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
        lcs_length = dp[n, m]
        lcs_ratio = float(lcs_length / max(m, 1))

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "lcs_ratio": round(lcs_ratio, 4),
        }

    def compare(
        self,
        raw_classifier_arcs: List[int],
        decoded_arcs: List[int],
        reference_route: List[int],
        arcs: np.ndarray,
        reference_pointwise_arcs: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Produces a side-by-side comparison between raw classifier predictions
        and HMM/Viterbi-decoded route predictions, fulfilling report section 5.
        """
        raw_route = self.compress_route(raw_classifier_arcs)
        dec_route = self.compress_route(decoded_arcs)

        raw_cont = self.calculate_continuity(raw_classifier_arcs, arcs)
        dec_cont = self.calculate_continuity(decoded_arcs, arcs)

        raw_metrics = self.evaluate_route(raw_route, reference_route)
        dec_metrics = self.evaluate_route(dec_route, reference_route)

        res = {
            "continuity": {
                "classifier_only": round(raw_cont, 4),
                "viterbi_decoded": round(dec_cont, 4),
                "improvement": round(dec_cont - raw_cont, 4),
            },
            "route_f1": {
                "classifier_only": raw_metrics["f1"],
                "viterbi_decoded": dec_metrics["f1"],
                "improvement": round(dec_metrics["f1"] - raw_metrics["f1"], 4),
            },
            "lcs_ratio": {
                "classifier_only": raw_metrics["lcs_ratio"],
                "viterbi_decoded": dec_metrics["lcs_ratio"],
                "improvement": round(dec_metrics["lcs_ratio"] - raw_metrics["lcs_ratio"], 4),
            },
            "route_length": {
                "ground_truth": len(reference_route),
                "classifier_only": len(raw_route),
                "viterbi_decoded": len(dec_route),
            },
        }

        if reference_pointwise_arcs:
            raw_pt = self.evaluate_pointwise(raw_classifier_arcs, reference_pointwise_arcs)
            dec_pt = self.evaluate_pointwise(decoded_arcs, reference_pointwise_arcs)
            res["pointwise_accuracy"] = {
                "classifier_only": round(raw_pt, 4),
                "viterbi_decoded": round(dec_pt, 4),
                "improvement": round(dec_pt - raw_pt, 4),
            }

        return res
