from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)

from pathsnap.models import MODEL_CONFIGS


def fmt(value, digits=3):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def para(text, style):
    return Paragraph(str(text).replace("&", "&amp;"), style)


def add_table(story, data, widths, font_size=7.5):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123d45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#b7c5ca")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f5f6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 5*mm))


def build_report(project: Path, output: Path) -> None:
    art = project / "artifacts"
    exp = art / "exploration"
    required = [art/"model_comparison.csv", art/"tuned_vs_untuned_student_test.csv",
                art/"bayes_trials.csv", art/"decoder_parameter_sweep.csv",
                art/"model_config.json", exp/"partition_summary.csv",
                exp/"sample_trajectory.png", art/"model_comparison.png"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Run explore.py and train.py first; missing: " + ", ".join(missing))

    results = pd.read_csv(art / "model_comparison.csv")
    if len(results) < 5:
        raise ValueError(f"Expected at least five measured models; found {len(results)}")
    results = results.sort_values("cv_observation_macro_f1_mean", ascending=False)
    config = json.loads((art / "model_config.json").read_text())
    partitions = pd.read_csv(exp / "partition_summary.csv")
    tuned = pd.read_csv(art / "tuned_vs_untuned_student_test.csv")
    bayes = pd.read_csv(art / "bayes_trials.csv")
    decoder = pd.read_csv(art / "decoder_parameter_sweep.csv")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=24,
                              leading=29, textColor=colors.HexColor("#123d45"), alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=11,
                              leading=15, textColor=colors.HexColor("#52636b"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=16,
                              leading=20, textColor=colors.HexColor("#087e78"), spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11,
                              leading=14, textColor=colors.HexColor("#123d45"), spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5,
                              leading=12, spaceAfter=5))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7,
                              leading=9, textColor=colors.HexColor("#52636b")))

    page_size = landscape(A4)
    doc = SimpleDocTemplate(str(output), pagesize=page_size, rightMargin=15*mm,
                            leftMargin=15*mm, topMargin=14*mm, bottomMargin=14*mm,
                            title="PathSnap Technical Report", author="PathSnap Team")
    usable_w = page_size[0] - 30*mm
    story = [Spacer(1, 18*mm), para("PathSnap", styles["Title2"]),
             para("Distributed map matching and route reconstruction", styles["SubTitle"]),
             Spacer(1, 9*mm),
             para("Technical report generated from the project source and measured outputs.", styles["Bodyx"]),
             para("AI assistance disclosure: Codex assisted with code, analysis scaffolding, and report generation. The team must review the implementation, add its own explanation of design choices, and disclose assistance according to the course policy.", styles["Bodyx"]),
             para(f"This report covers {int(partitions.loc[partitions.partition.eq('train'),'trips'].sum())} training trips and {int(partitions.loc[partitions.partition.eq('student_test'),'trips'].sum())} student-test trips. Evaluator trip records and evaluator-derived scores are kept out of this public report; evaluator labels stay on Laptop 1 during the demo.", styles["Bodyx"]),
             PageBreak()]

    story.append(para("1. Dataset and exploration", styles["H1x"]))
    story.append(para("The raw data is organized by trip. Each trip contains headerless whitespace-separated GPS observations (longitude, latitude, seconds), road nodes (longitude, latitude), directed arcs (from-node, to-node), and a reference route (arc row IDs). Preprocessing coerces numeric fields, drops invalid GPS rows, removes duplicate timestamps, sorts by time, validates arc endpoints, and remaps route arc IDs if invalid arc rows are removed.", styles["Bodyx"]))
    story.append(para("The inventory and exploration summaries cover train and student_test. Route labels for supervised development use the closest segment in the supplied route by local point-to-segment distance; these labels are treated as reference annotations, not independently verified ground truth. Evaluator route files remain on Laptop 1 and are read only after prediction.", styles["Bodyx"]))
    part_rows = [["Partition", "Trips", "Raw GPS rows", "Clean fixes", "Nodes", "Arcs", "Route arcs", "Invalid GPS rows", "Gaps >5s", "Gaps >30s"]]
    for row in partitions.itertuples(index=False):
        part_rows.append([row.partition, int(row.trips), int(row.gps_rows), int(row.clean_observations),
                          int(row.nodes), int(row.arcs), int(row.route_arcs), int(row.invalid_gps_rows),
                          int(row.gaps_over_5s), int(row.gaps_over_30s)])
    add_table(story, part_rows, [22*mm, 13*mm, 23*mm, 23*mm, 20*mm, 20*mm, 22*mm, 26*mm, 20*mm, 22*mm])
    map_img = Image(str(exp / "sample_trajectory.png"), width=usable_w*.54, height=usable_w*.54*0.62)
    story.append(KeepTogether([
        para("Figure 1. Exported view of the same GPS, nearby-road, and reference-route layers generated in the interactive Folium map. Blue shows GPS fixes; green shows the supplied reference route. The interactive HTML is committed separately.", styles["Smallx"]),
        map_img,
    ]))
    story.append(para("Files: artifacts/exploration/raw_inventory.csv, partition_summary.csv, dataset_exploration.png, sample_trajectory.html, and sample_trip_summary.json.", styles["Smallx"]))

    story.append(PageBreak())
    story.append(para("2. Feature engineering and model design", styles["H1x"]))
    story.append(para("For each GPS fix, a haversine BallTree retrieves eight nearest road-segment midpoints. Candidate features are point-to-segment distance in metres (local equirectangular projection), absolute heading difference, segment length, GPS speed, timestamp gap, and candidate rank. Candidate labels are binary: 1 if the candidate is the closest supplied-route segment for that observation, else 0. The route arc ID is retained only as an evaluation target and is excluded from model features.", styles["Bodyx"]))
    story.append(para("Training is capped at a uniform temporal sample of 400 observations per training trip to control runtime while retaining all 80 trip groups. Cross-validation uses GroupKFold by trip, so observations from one trip cannot appear in both train and validation folds. Student-test trips are used for development comparison only.", styles["Bodyx"]))
    story.append(para("3. Five-model comparison", styles["H1x"]))
    story.append(para(f"Selection criterion: the highest mean five-fold CV observation-level Macro F1. The selected baseline is {results.iloc[0].model}, with CV observation Macro F1 {fmt(results.iloc[0].cv_observation_macro_f1_mean)} ± {fmt(results.iloc[0].cv_observation_macro_f1_std)}. This selection uses group-aware CV rather than training accuracy alone.", styles["Bodyx"]))
    table1 = [["Model / family", "Initial configuration", "CV obs Macro F1 ± SD", "CV Top-1", "Train obs Macro F1", "Student Top-1", "Student obs Macro F1"]]
    table2 = [["Model", "Candidate precision", "Candidate recall", "Candidate Macro F1", "Route log loss", "Candidate miss rate", "Route continuity", "Classifier ms/fix"]]
    for row in results.itertuples(index=False):
        initial = {
            "logistic_regression": "StandardScaler; C=1; max_iter=500; balanced",
            "gaussian_nb": "var_smoothing=1e-9",
            "decision_tree": "Gini; depth=12; leaf=5; balanced",
            "extra_trees": "100 trees; sqrt features; leaf=2; balanced",
            "hist_gradient_boosting": "100 iters; lr=0.08; 15 leaves; L2=1",
        }[row.model]
        table1.append([para(row.model.replace("_", " "), styles["Smallx"]), para(initial, styles["Smallx"]),
                       f"{fmt(row.cv_observation_macro_f1_mean)} ± {fmt(row.cv_observation_macro_f1_std)}",
                       fmt(row.cv_top1_accuracy_mean), fmt(row.training_observation_macro_f1),
                       fmt(row.student_top1_accuracy), fmt(row.student_observation_macro_f1)])
        table2.append([row.model.replace("_", " "), fmt(row.student_candidate_precision), fmt(row.student_candidate_recall),
                       fmt(row.student_candidate_macro_f1), fmt(row.student_observation_log_loss),
                       fmt(row.student_candidate_miss_rate), fmt(row.student_classifier_route_continuity),
                       fmt(row.student_inference_ms_per_observation, 5)])
    add_table(story, table1, [30*mm, 52*mm, 32*mm, 23*mm, 28*mm, 25*mm, 34*mm], 7)
    add_table(story, table2, [34*mm, 28*mm, 25*mm, 31*mm, 27*mm, 30*mm, 30*mm, 26*mm], 7)
    story.append(para("All five baseline classifiers are serialized under artifacts/models/ for direct reuse. The Bayesian-tuned HGB model selected for deployment is saved separately as artifacts/model.joblib.", styles["Smallx"]))
    for row in results.itertuples(index=False):
        info = MODEL_CONFIGS[row.model]
        story.append(KeepTogether([
            para(f"{row.model.replace('_', ' ').title()} - {info['family']}", styles["H2x"]),
            para(f"Suitable because: {info['description']} Strength: {info['strengths']} Limitation: {info['limitations']}", styles["Bodyx"]),
            para(f"Objective: {info['objective']}. Initial parameters: {row.initial_configuration}. Measured training candidate Macro F1 {fmt(row.training_candidate_macro_f1)}; CV candidate Macro F1 {fmt(row.cv_candidate_macro_f1_mean)} ± {fmt(row.cv_candidate_macro_f1_std)} (gap {fmt(row.training_candidate_macro_f1-row.cv_candidate_macro_f1_mean)}); student-test Top-1 {fmt(row.student_top1_accuracy)}; student-test observation Macro F1 {fmt(row.student_observation_macro_f1)}; student-test observation log loss {fmt(row.student_observation_log_loss)}. Training time {fmt(row.training_seconds_full, 2)} s; classifier scoring time {fmt(row.student_inference_ms_per_observation, 5)} ms/fix. The training-to-CV gap is reported as an overfit diagnostic; it should be interpreted alongside student-test results.", styles["Bodyx"]),
        ]))
    story.append(Image(str(art / "model_comparison.png"), width=usable_w*.78, height=usable_w*.78*.53))
    if (art / "student_test_confusion_matrices.png").exists():
        story.append(Image(str(art / "student_test_confusion_matrices.png"), width=usable_w*.83, height=usable_w*.83*.22))
    story.append(para("Figures 2-3 are generated directly from model_comparison.csv and student_test_confusion_matrices.csv. No synthetic training curves are used; these models do not expose epoch-wise histories in this implementation.", styles["Smallx"]))

    story.append(PageBreak())
    story.append(para("4. Bayesian hyperparameter optimisation", styles["H1x"]))
    story.append(para(f"The best CV baseline ({config['selected_model']}) was tuned with BayesSearchCV. The search maximized candidate-level Macro F1 using {config['bayesian_cv_folds']}-fold GroupKFold grouped by trip. Search space: {config.get('bayesian_search_space', 'See pathsnap/train.py and artifacts/bayes_trials.csv')}. {config['bayesian_trials']} trials were retained; elapsed search time was {fmt(config['bayesian_search_seconds'], 1)} seconds. Best parameters: {json.dumps(config['best_bayesian_parameters'], default=str)}.", styles["Bodyx"]))
    trial_rows = [["Rank", "CV Macro F1", "CV SD", "Mean fit sec", "Parameters"]]
    for row in bayes.sort_values("rank_test_score").itertuples(index=False):
        params = json.dumps(row.params, sort_keys=True, default=str)
        trial_rows.append([int(row.rank_test_score), fmt(row.mean_test_score), fmt(row.std_test_score),
                           fmt(row.mean_fit_time, 2), para(params, styles["Smallx"])])
    add_table(story, trial_rows, [14*mm, 28*mm, 22*mm, 26*mm, 185*mm], 7)
    tuned_rows = [["Version", "Candidate Macro F1", "Top-1 accuracy", "Observation Macro F1", "Observation log loss", "Route continuity", "Inference ms/fix"]]
    for row in tuned.itertuples(index=False):
        tuned_rows.append([row.version, fmt(row.candidate_macro_f1), fmt(row.top1_accuracy),
                           fmt(row.observation_macro_f1), fmt(row.observation_log_loss),
                           fmt(row.classifier_route_continuity), fmt(row.inference_ms_per_observation, 5)])
    add_table(story, tuned_rows, [27*mm, 34*mm, 28*mm, 38*mm, 34*mm, 32*mm, 32*mm])
    story.append(para("The table uses the same student-test trips and metrics for the untuned and tuned estimators. The complete trial record, best parameters, seed, software versions, and selected model are saved in artifacts/.", styles["Bodyx"]))

    story.append(PageBreak())
    story.append(para("5. HMM/Viterbi route reconstruction", styles["H1x"]))
    story.append(para("Hidden state: one of the eight road-arc candidates for each GPS fix. Emission: the selected classifier's positive-candidate probability, normalized across candidates at that fix. Transition: for each previous candidate, legal moves (same edge or a directed edge whose start node matches the previous edge's end node) receive logit 0; other moves receive a negative penalty. A row-wise softmax turns these logits into conditional transition probabilities. A gap over 30 seconds reduces the penalty to one quarter. Candidate generation always returns the nearest available edges; disconnected transitions remain possible to recover from incomplete local networks. Viterbi stores the best previous state and backtracks the maximum sequence score.", styles["Bodyx"]))
    story.append(para(f"The selected transition penalty is {config['decoder_transition_weight']}. It was selected on student-test development data by maximizing route continuity subject to no more than a 3 percentage-point Top-1 loss relative to independent classifier ranking. The full sweep and selection criterion are saved in artifacts/decoder_parameter_sweep.csv. The comparison below is measured, including cases where continuity improves while point accuracy declines.", styles["Bodyx"]))
    seq_path = art / "student_test_sequence_metrics.csv"
    if seq_path.exists():
        seq = pd.read_csv(seq_path)
        summary = [["Measure", "Classifier only", "Classifier + Viterbi"]]
        summary.extend([
            ["Mean observation Top-1", fmt(seq.classifier_top1.mean()), fmt(seq.hmm_top1.mean())],
            ["Mean route continuity", fmt(seq.classifier_continuity.mean()), fmt(seq.hmm_continuity.mean())],
            ["Exact whole-trip sequence rate", fmt(seq.classifier_exact_sequence.mean()), fmt(seq.hmm_exact_sequence.mean())],
        ])
        add_table(story, summary, [65*mm, 45*mm, 50*mm])
    story.append(Image(str(art / "decoder_parameter_sweep.png"), width=usable_w*.55, height=usable_w*.55*.43))
    story.append(para("Reference route IDs are never model features. For final evaluator requests, API code constructs candidates without labels, runs prediction, then loads and scores route references only inside Laptop 1. The API response contains predictions and aggregate metrics, not reference route labels.", styles["Bodyx"]))

    story.append(PageBreak())
    story.append(para("6. Two-laptop deployment", styles["H1x"]))
    story.append(para("Laptop 1 - model and API", styles["H2x"]))
    story.append(para("Install Python 3.10-3.12 and project dependencies. Place artifacts/model.joblib and the separate evaluator_test directory on Laptop 1. From the repository root, configure paths and start the API:", styles["Bodyx"]))
    code1 = "export PATHSNAP_DATA=\"/path/to/Student folder\"<br/>export PATHSNAP_EVALUATOR_DIR=\"/path/to/evaluator_test\"<br/>export PATHSNAP_MODEL=\"/path/to/artifacts/model.joblib\"<br/>uvicorn pathsnap.service:app --host 0.0.0.0 --port 8000"
    story.append(para(code1, styles["Code"]))
    story.append(para("Laptop 2 - evaluator UI", styles["H2x"]))
    story.append(para("Start the standard-library UI server on Laptop 2 with python3 run_ui.py and open http://localhost:8080. In the Evaluator UI tab, enter Laptop 1's LAN address, for example http://192.168.1.20:8000, then load the trip list, select a trip, and run inference. Confirm both laptops are on a network where Laptop 2 can reach port 8000 on Laptop 1. Laptop 1's terminal shows Uvicorn access logs. The local preview tab is for development, not model inference.", styles["Bodyx"]))
    story.append(para("7. Limitations and final evaluation status", styles["H1x"]))
    story.append(para("Evaluator scores are produced by the API on Laptop 1 during the demo. Evaluator trip records and derived metrics are not included in this public report or repository. The team should capture the live two-laptop UI and review the server-side results during its demonstration.", styles["Bodyx"]))
    story.append(para("Known modeling limitations include midpoint-based nearest-candidate retrieval (with exact point-to-segment feature distances), a route target defined by closest reference-route geometry, and a simplified connectivity transition model that may trade point accuracy for route continuity. Candidate miss rate quantifies when the reference edge is absent from the candidate set; no classifier can recover a missing candidate.", styles["Bodyx"]))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#52636b"))
        canvas.drawString(15*mm, 8*mm, "PathSnap - generated from measured supplied-data runs; team review required")
        canvas.drawRightString(page_size[0]-15*mm, 8*mm, f"Page {doc_obj.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--output", default="artifacts/PathSnap_Technical_Report.pdf")
    args = parser.parse_args()
    project = Path(args.project).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_report(project, output)
    print(output.resolve())


if __name__ == "__main__":
    main()
