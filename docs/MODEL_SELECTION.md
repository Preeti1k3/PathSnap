# Model Selection and Performance Summary

Based on 5-fold GroupKFold cross-validation across 80 training trips (232,942 candidate samples) and evaluation on the 10 student-test trips:

## 1. Candidate Classifiers Comparison
- **Random Forest (Ensemble):** Won top cross-validation score (Macro F1: 0.7892, Test Top-1 Accuracy: 82.89%). Its balanced class weighting and ensemble variance reduction prevented overfitting on complex urban intersections.
- **XGBoost (Boosting):** High competitive performance (Macro F1: 0.7857, Test Top-1 Accuracy: 82.10%) with ultra-fast inference (< 0.02s per trip).
- **MLP Neural Net:** Strong probability calibration (Log Loss: 0.0943, Test Macro F1: 0.9185), but longer training convergence.
- **Extra Trees & Decision Tree:** Fast tree-based baselines, but slightly lower generalisation on unseen student-test trips.
- **Logistic Regression & KNN:** Linear baseline performed reasonably on distance features (Top-1 Acc: 80.71%), but failed to capture nonlinear heading-speed interactions. KNN showed highest latency and lowest recall.

## 2. Bayesian Hyperparameter Tuning
Optuna Bayesian optimization tuned the leading Random Forest model over 12 trials:
- 
_estimators: 60
- max_depth: 14
- min_samples_split: 3
- max_features: 0.6571
- Final CV Macro F1: **0.7892**

## 3. Impact of HMM & Viterbi Sequential Decoding
- Direct classifier Top-1 Accuracy: 82.89% (Route continuity: 41.2%)
- Classifier + HMM Viterbi Decoding: Top-1 Accuracy: **84.15%** (Route continuity: **89.5%**)
- Demonstrates a **+48.3% increase in route continuity** by enforcing road network graph transitions.
