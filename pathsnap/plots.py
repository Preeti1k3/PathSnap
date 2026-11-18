from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def model_comparison_plot(results_csv: str | Path, output: str | Path) -> None:
    df = pd.read_csv(results_csv).sort_values("cv_observation_macro_f1_mean", ascending=False)
    names = [str(x).replace("_", " ") for x in df.model]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [
        ("cv_observation_macro_f1_mean", "cv_observation_macro_f1_std", "5-fold CV observation Macro F1", "Macro F1"),
        ("student_top1_accuracy", None, "Student-test Top-1 accuracy", "Accuracy"),
        ("student_observation_macro_f1", None, "Student-test observation Macro F1", "Macro F1"),
        ("student_observation_log_loss", None, "Student-test route log loss", "Log loss (lower is better)"),
    ]
    for ax, (metric, err, title, ylabel) in zip(axes.ravel(), panels):
        values = df[metric].to_numpy(float)
        errors = df[err].fillna(0).to_numpy(float) if err else None
        ax.bar(x, values, yerr=errors, capsize=4 if errors is not None else 0,
               color=["#0f766e" if i == 0 else "#3b82f6" for i in x])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Measured PathSnap model comparison", fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def confusion_plot(confusion_csv: str | Path, output: str | Path) -> None:
    df = pd.read_csv(confusion_csv)
    fig, axes = plt.subplots(1, len(df), figsize=(3.1 * len(df), 3.5), constrained_layout=True)
    if len(df) == 1:
        axes = [axes]
    for ax, row in zip(axes, df.itertuples(index=False)):
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(str(row.model).replace("_", " "), fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"], rotation=25)
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="black", fontsize=8)
    fig.suptitle("Student-test candidate confusion matrices", fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def decoder_plot(decoder_csv: str | Path, output: str | Path) -> None:
    df = pd.read_csv(decoder_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(df.transition_weight, df.top1_accuracy, marker="o", label="Classifier-only")
    ax.plot(df.transition_weight, df.hmm_top1_accuracy, marker="o", label="Classifier + Viterbi")
    ax.plot(df.transition_weight, df.classifier_route_continuity, marker="s", linestyle="--", label="Classifier continuity")
    ax.plot(df.transition_weight, df.hmm_route_continuity, marker="s", linestyle="--", label="Viterbi continuity")
    ax.set_xlabel("Disconnected-transition penalty")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.25)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Measured decoder development sweep (student-test trips)")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
