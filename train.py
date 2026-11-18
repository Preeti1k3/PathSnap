from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

from pathsnap.data import candidate_table, inventory, load_trip
from pathsnap.decoder import viterbi
from pathsnap.models import FEATURES, MODEL_CONFIGS, model_zoo
from pathsnap.plots import model_comparison_plot, confusion_plot, decoder_plot

LOGGER = logging.getLogger("pathsnap.train")


def _sample_observations(table: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Uniformly subsample observations while retaining all K candidates per fix."""
    if limit is None:
        return table
    observations = np.sort(table["obs"].unique())
    if len(observations) <= limit:
        return table
    chosen = observations[np.linspace(0, len(observations) - 1, limit, dtype=int)]
    return table[table["obs"].isin(chosen)].reset_index(drop=True)


def load_partition(root: Path, name: str, max_observations: int | None = None):
    parent = root / name
    if not parent.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {parent}")
    result = []
    for folder in sorted(p for p in parent.iterdir() if p.is_dir()):
        trip = load_trip(folder, include_route=True)
        if trip.route is None or not len(trip.route):
            raise ValueError(f"Trip {trip.trip_id} has no usable reference route")
        table = _sample_observations(candidate_table(trip), max_observations)
        table["trip_id"] = trip.trip_id
        result.append((trip, table))
    if not result:
        raise ValueError(f"Partition contains no trip folders: {parent}")
    return result


def _connected(a: np.ndarray, b: np.ndarray, arcs: np.ndarray) -> np.ndarray:
    return (a == b) | (arcs[a, 1] == arcs[b, 0])


def _observation_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def evaluate(model, trips, transition_weight: float = 1.8):
    """Evaluate binary candidate scoring and route-level predictions."""
    candidate_true, candidate_pred, candidate_prob = [], [], []
    point_true, point_pred, hmm_pred = [], [], []
    point_losses, hmm_losses, misses = [], [], []
    classifier_continuity, hmm_continuity = [], []
    trip_macro, hmm_trip_macro, exact_classifier, exact_hmm = [], [], [], []
    sequence_rows = []
    inference_seconds = 0.0

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Candidate classifier did not learn the positive class")
    positive_col = classes.index(1)

    for trip, table in trips:
        predict_start = time.perf_counter()
        x = table[FEATURES]
        raw_p = model.predict_proba(x)[:, positive_col]
        inference_seconds += time.perf_counter() - predict_start
        groups = list(table.groupby("obs", sort=True))
        cand = np.stack([g.arc_id.to_numpy(int) for _, g in groups])
        p = np.stack([raw_p[g.index.to_numpy()] for _, g in groups])
        true_route = np.array([int(g.reference_arc_id.iloc[0]) for _, g in groups])
        denom = p.sum(axis=1, keepdims=True)
        emissions = np.divide(p, denom, out=np.full_like(p, 1.0 / p.shape[1]), where=denom > 1e-12)
        top = cand[np.arange(len(cand)), np.argmax(p, axis=1)]
        gaps = np.r_[0.0, np.diff(trip.track.time.to_numpy(float))]
        decoded = viterbi(cand, emissions, trip.arcs, gaps, transition_weight)

        candidate_true.extend(table.label.to_numpy(int).tolist())
        candidate_pred.extend((raw_p >= 0.5).astype(int).tolist())
        candidate_prob.extend(raw_p.tolist())
        point_true.extend(true_route.tolist())
        point_pred.extend(top.tolist())
        hmm_pred.extend(decoded.tolist())
        valid = true_route >= 0
        misses.extend((~np.any(cand == true_route[:, None], axis=1))[valid].astype(int).tolist())
        for row, target in enumerate(true_route):
            if target < 0:
                continue
            ids = cand[row]
            loc = np.flatnonzero(ids == target)
            prob_true = float(emissions[row, loc[0]]) if len(loc) else 1e-12
            point_losses.append(-np.log(max(prob_true, 1e-12)))
            loc_hmm = np.flatnonzero(ids == decoded[row])
            # Decoder score for predicted state is the emission probability selected
            # by the path; use a tiny floor for route-level sequence log loss.
            prob_hmm = float(emissions[row, loc_hmm[0]]) if len(loc_hmm) else 1e-12
            hmm_losses.append(-np.log(max(prob_hmm, 1e-12)) if decoded[row] != target else -np.log(max(prob_true, 1e-12)))
        if len(top) > 1:
            classifier_continuity.extend(_connected(top[:-1], top[1:], trip.arcs).astype(int).tolist())
            hmm_continuity.extend(_connected(decoded[:-1], decoded[1:], trip.arcs).astype(int).tolist())
        trip_macro.append(_observation_macro_f1(true_route[valid], top[valid]))
        hmm_trip_macro.append(_observation_macro_f1(true_route[valid], decoded[valid]))
        exact_classifier.append(int(valid.all() and np.array_equal(true_route, top)))
        exact_hmm.append(int(valid.all() and np.array_equal(true_route, decoded)))
        sequence_rows.append({
            "trip_id": trip.trip_id,
            "observations": int(valid.sum()),
            "classifier_top1": float(np.mean(top[valid] == true_route[valid])) if valid.any() else np.nan,
            "hmm_top1": float(np.mean(decoded[valid] == true_route[valid])) if valid.any() else np.nan,
            "classifier_continuity": float(np.mean(_connected(top[:-1], top[1:], trip.arcs))) if len(top) > 1 else 1.0,
            "hmm_continuity": float(np.mean(_connected(decoded[:-1], decoded[1:], trip.arcs))) if len(decoded) > 1 else 1.0,
            "classifier_exact_sequence": int(valid.all() and np.array_equal(true_route, top)),
            "hmm_exact_sequence": int(valid.all() and np.array_equal(true_route, decoded)),
        })

    candidate_cm = confusion_matrix(candidate_true, candidate_pred, labels=[0, 1])
    top_acc = accuracy_score(point_true, point_pred)
    hmm_acc = accuracy_score(point_true, hmm_pred)
    candidate_logloss = log_loss(candidate_true, np.clip(candidate_prob, 1e-12, 1-1e-12), labels=[0, 1])
    result = {
        "candidate_accuracy": float(accuracy_score(candidate_true, candidate_pred)),
        "candidate_precision": float(precision_score(candidate_true, candidate_pred, zero_division=0)),
        "candidate_recall": float(recall_score(candidate_true, candidate_pred, zero_division=0)),
        "candidate_f1": float(f1_score(candidate_true, candidate_pred, zero_division=0)),
        "candidate_macro_f1": float(f1_score(candidate_true, candidate_pred, average="macro", zero_division=0)),
        "candidate_log_loss": float(candidate_logloss),
        "tn": int(candidate_cm[0, 0]), "fp": int(candidate_cm[0, 1]),
        "fn": int(candidate_cm[1, 0]), "tp": int(candidate_cm[1, 1]),
        "top1_accuracy": float(top_acc),
        "observation_macro_f1": float(np.mean(trip_macro)),
        "observation_log_loss": float(np.mean(point_losses)),
        "candidate_miss_rate": float(np.mean(misses)) if misses else np.nan,
        "classifier_route_continuity": float(np.mean(classifier_continuity)) if classifier_continuity else np.nan,
        "hmm_top1_accuracy": float(hmm_acc),
        "hmm_macro_f1": float(np.mean(hmm_trip_macro)),
        "hmm_route_continuity": float(np.mean(hmm_continuity)) if hmm_continuity else np.nan,
        "classifier_exact_sequence_rate": float(np.mean(exact_classifier)),
        "hmm_exact_sequence_rate": float(np.mean(exact_hmm)),
        "inference_seconds": float(inference_seconds),
        "inference_ms_per_observation": float(1000 * inference_seconds / max(len(point_true), 1)),
    }
    return result, sequence_rows


def cv_evaluate(model, train_data, x, y, groups, folds=5, transition_weight=1.8):
    splitter = GroupKFold(n_splits=folds)
    fold_rows = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups), start=1):
        fold_model = clone(model)
        start = time.perf_counter()
        fold_model.fit(x.iloc[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - start
        val_ids = set(groups[val_idx])
        val_trips = [(trip, table) for trip, table in train_data if trip.trip_id in val_ids]
        metrics, _ = evaluate(fold_model, val_trips, transition_weight)
        fold_rows.append({"fold": fold, "fit_seconds": fit_seconds, **metrics})
        LOGGER.info("CV fold %s: %s", fold, json.dumps(fold_rows[-1]))
    keys = [k for k in fold_rows[0] if k not in {"fold"}]
    summary = {}
    for key in keys:
        vals = np.asarray([row[key] for row in fold_rows], dtype=float)
        summary[f"cv_{key}_mean"] = float(np.nanmean(vals))
        summary[f"cv_{key}_std"] = float(np.nanstd(vals, ddof=1))
    return summary, fold_rows


def search_space(name: str):
    return {
        "logistic_regression": {"logisticregression__C": Real(0.03, 30.0, prior="log-uniform")},
        "gaussian_nb": {"var_smoothing": Real(1e-11, 1e-7, prior="log-uniform")},
        "decision_tree": {"max_depth": Integer(5, 24), "min_samples_leaf": Integer(2, 16)},
        "extra_trees": {"n_estimators": Integer(50, 140), "min_samples_leaf": Integer(1, 8)},
        "hist_gradient_boosting": {
            "max_leaf_nodes": Integer(7, 31), "learning_rate": Real(0.025, 0.2, prior="log-uniform"),
            "l2_regularization": Real(1e-4, 10.0, prior="log-uniform"),
        },
    }[name]


def search_space_description(name: str) -> str:
    descriptions = {
        "logistic_regression": "C: log-uniform [0.03, 30]",
        "gaussian_nb": "var_smoothing: log-uniform [1e-11, 1e-7]",
        "decision_tree": "max_depth: integer [5, 24]; min_samples_leaf: integer [2, 16]",
        "extra_trees": "n_estimators: integer [50, 140]; min_samples_leaf: integer [1, 8]",
        "hist_gradient_boosting": "max_leaf_nodes: integer [7, 31]; learning_rate: log-uniform [0.025, 0.2]; l2_regularization: log-uniform [1e-4, 10]",
    }
    return descriptions[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../Student folder")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-observations-per-trip", type=int, default=500,
                        help="Uniform temporal subsample for model fitting; CV still groups all 80 trips")
    parser.add_argument("--bayes-iterations", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    root, out = Path(args.data).resolve(), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "training.log"), logging.StreamHandler()])
    LOGGER.info("Python/sklearn versions: %s / %s", __import__("sys").version, sklearn.__version__)
    LOGGER.info("Seed=%d, train observation cap/trip=%d", args.seed, args.max_train_observations_per_trip)

    inv = inventory(root)
    inv.to_csv(out / "dataset_inventory.csv", index=False)
    train_data = load_partition(root, "train", args.max_train_observations_per_trip)
    test_data = load_partition(root, "student_test", None)
    x = pd.concat([table[FEATURES] for _, table in train_data], ignore_index=True)
    y = pd.concat([table.label for _, table in train_data], ignore_index=True).to_numpy(int)
    groups = np.concatenate([np.repeat(trip.trip_id, len(table)) for trip, table in train_data])
    if len(np.unique(groups)) != 80:
        LOGGER.warning("Expected 80 training trips per brief; found %d", len(np.unique(groups)))

    all_rows, confusion_rows, fold_rows_all = [], [], []
    fitted, cv_summaries = {}, {}
    for name, model in model_zoo(args.seed).items():
        LOGGER.info("Starting model %s", name)
        cv_summary, fold_rows = cv_evaluate(model, train_data, x, y, groups,
                                            folds=args.cv_folds, transition_weight=1.8)
        fold_rows_all.extend({"model": name, **row} for row in fold_rows)
        fit_start = time.perf_counter()
        model.fit(x, y)
        train_seconds = time.perf_counter() - fit_start
        model_dir = out / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{name}.joblib",
                    compress=("lzma", 6) if name == "extra_trees" else 3)
        train_metrics, _ = evaluate(model, train_data, 1.8)
        test_metrics, sequence_rows = evaluate(model, test_data, 1.8)
        cv_summaries[name] = cv_summary
        fitted[name] = model
        row = {"model": name, "algorithm_family": MODEL_CONFIGS[name]["family"],
               "initial_configuration": json.dumps(MODEL_CONFIGS[name]["initial_parameters"], sort_keys=True),
               "cv_fit_seconds_total": float(sum(fr["fit_seconds"] for fr in fold_rows)),
               "training_seconds_full": float(train_seconds),
               **{f"training_{k}": v for k, v in train_metrics.items()},
               **cv_summary, **{f"student_{k}": v for k, v in test_metrics.items()}}
        all_rows.append(row)
        confusion_rows.append({"model": name, "tn": test_metrics["tn"], "fp": test_metrics["fp"],
                               "fn": test_metrics["fn"], "tp": test_metrics["tp"]})
        LOGGER.info("Saved trained baseline %s to %s", name, model_dir / f"{name}.joblib")
        LOGGER.info("Student-test %s: %s", name, json.dumps(test_metrics))

    results = pd.DataFrame(all_rows).sort_values("cv_observation_macro_f1_mean", ascending=False)
    results.to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows_all).to_csv(out / "cv_fold_results.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(out / "student_test_confusion_matrices.csv", index=False)
    model_comparison_plot(out / "model_comparison.csv", out / "model_comparison.png")
    confusion_plot(out / "student_test_confusion_matrices.csv", out / "student_test_confusion_matrices.png")
    winner = str(results.iloc[0].model)
    LOGGER.info("Selected by mean 5-fold observation macro F1: %s", winner)

    # Bayesian optimisation with trip-grouped CV. Objective is candidate-level macro F1.
    bayes_cv = GroupKFold(n_splits=min(4, args.cv_folds))
    search = BayesSearchCV(
        estimator=model_zoo(args.seed)[winner], search_spaces=search_space(winner),
        n_iter=args.bayes_iterations, scoring="f1_macro", cv=bayes_cv,
        n_jobs=1, random_state=args.seed, refit=True, return_train_score=True,
    )
    search_start = time.perf_counter()
    search.fit(x, y, groups=groups)
    search_seconds = time.perf_counter() - search_start
    bayes_frame = pd.DataFrame(search.cv_results_)
    bayes_frame.to_csv(out / "bayes_trials.csv", index=False)

    untuned_metrics, _ = evaluate(fitted[winner], test_data)
    tuned_metrics, sequence_rows = evaluate(search.best_estimator_, test_data)
    pd.DataFrame([
        {"version": "untuned", **untuned_metrics},
        {"version": "tuned", **tuned_metrics},
    ]).to_csv(out / "tuned_vs_untuned_student_test.csv", index=False)
    pd.DataFrame(sequence_rows).to_csv(out / "student_test_sequence_metrics.csv", index=False)

    # Development-only sweep for decoder continuity weight, restricted to permitted
    # training/student-test data. Final evaluator routes are never read here.
    decoder_rows = []
    decoder_weights = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1.8, 3.0, 6.0)
    for weight in decoder_weights:
        metrics, seq = evaluate(search.best_estimator_, test_data, transition_weight=weight)
        decoder_rows.append({"transition_weight": weight, **metrics})
    decoder_results = pd.DataFrame(decoder_rows)
    baseline_accuracy = float(decoder_results.loc[decoder_results.transition_weight.eq(0.0), "hmm_top1_accuracy"].iloc[0])
    decoder_results["top1_drop_from_no_transition"] = baseline_accuracy - decoder_results.hmm_top1_accuracy
    # Keep at most a three-point Top-1 loss relative to independent classifier
    # ranking, then choose the strongest continuity among settings that pass.
    decoder_results["passes_top1_guardrail"] = decoder_results.hmm_top1_accuracy >= baseline_accuracy - 0.03
    decoder_results.to_csv(out / "decoder_parameter_sweep.csv", index=False)
    decoder_plot(out / "decoder_parameter_sweep.csv", out / "decoder_parameter_sweep.png")
    eligible = decoder_results[decoder_results.passes_top1_guardrail]
    best_decoder = float(eligible.sort_values(
        ["hmm_route_continuity", "hmm_macro_f1"], ascending=False
    ).iloc[0].transition_weight)
    final_metrics, final_sequences = evaluate(search.best_estimator_, test_data, best_decoder)
    pd.DataFrame(final_sequences).to_csv(out / "student_test_sequence_metrics.csv", index=False)
    joblib.dump(search.best_estimator_, out / "model.joblib", compress=3)
    config = {
        "selected_model": winner,
        "algorithm_family": MODEL_CONFIGS[winner]["family"],
        "initial_configuration": MODEL_CONFIGS[winner]["initial_parameters"],
        "model_explanations": MODEL_CONFIGS,
        "best_bayesian_parameters": search.best_params_,
        "bayesian_search_space": search_space_description(winner),
        "bayesian_objective": "candidate-level macro F1",
        "bayesian_best_cv_score": float(search.best_score_),
        "bayesian_trials": args.bayes_iterations,
        "bayesian_search_seconds": search_seconds,
        "bayesian_cv_folds": min(4, args.cv_folds),
        "cv_folds": args.cv_folds,
        "cv_group": "trip_id",
        "seed": args.seed,
        "max_train_observations_per_trip": args.max_train_observations_per_trip,
        "decoder_transition_weight": best_decoder,
        "decoder_selection_rule": "Maximize route continuity subject to no more than a 0.03 Top-1 accuracy drop from independent candidate ranking, using student-test development data.",
        "student_test_tuned_metrics": final_metrics,
        "python_version": __import__("sys").version,
        "sklearn_version": sklearn.__version__,
    }
    (out / "model_config.json").write_text(json.dumps(config, indent=2, default=float))
    pd.DataFrame([
        {"model": name, **summary}
        for name, summary in cv_summaries.items()
    ]).to_csv(out / "cv_summary.csv", index=False)
    LOGGER.info("Saved tuned model %s. Best params=%s. Search time=%.1fs", winner, search.best_params_, search_seconds)
    LOGGER.info("Best decoder weight on student-test development sweep: %s", best_decoder)


if __name__ == "__main__":
    main()
