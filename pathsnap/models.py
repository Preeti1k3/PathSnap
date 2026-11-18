"""Five deliberately different candidate-ranking classifier baselines."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

FEATURES = [
    "distance_m", "heading_error", "edge_length_m", "speed_mps",
    "time_gap_s", "candidate_rank",
]

MODEL_CONFIGS = {
    "logistic_regression": {
        "family": "linear probabilistic",
        "description": "Scaled logistic regression; class-balanced binary candidate scoring.",
        "initial_parameters": {"C": 1.0, "max_iter": 500, "class_weight": "balanced"},
        "objective": "binary cross-entropy (logistic loss)",
        "strengths": "Fast inference, transparent feature effects, and a well-defined candidate probability.",
        "limitations": "A linear decision boundary may miss interactions between heading, geometry, and speed; balanced fitting can affect probability calibration.",
    },
    "gaussian_nb": {
        "family": "probabilistic",
        "description": "Gaussian Naive Bayes over candidate geometry/motion features.",
        "initial_parameters": {"var_smoothing": 1e-9},
        "objective": "maximum likelihood under class-conditional Gaussian features",
        "strengths": "Very fast to train and score; produces probabilities with little tuning.",
        "limitations": "Conditional independence and Gaussian feature assumptions are restrictive for correlated road geometry and motion features.",
    },
    "decision_tree": {
        "family": "tree-based",
        "description": "Shallow CART tree with balanced classes to limit majority-class bias.",
        "initial_parameters": {"criterion": "gini", "max_depth": 12, "min_samples_leaf": 5, "class_weight": "balanced"},
        "objective": "Gini impurity",
        "strengths": "Captures nonlinear thresholds and is easy to inspect.",
        "limitations": "A single tree can be unstable and can overfit local candidate patterns.",
    },
    "extra_trees": {
        "family": "randomized ensemble",
        "description": "Extremely randomized trees with balanced bootstrap samples.",
        "initial_parameters": {"n_estimators": 100, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        "objective": "averaged impurity-reduction trees",
        "strengths": "Captures nonlinear interactions and reduces single-tree variance.",
        "limitations": "Larger model and slower scoring than linear/probabilistic baselines; raw probabilities may need calibration.",
    },
    "hist_gradient_boosting": {
        "family": "boosting ensemble",
        "description": "Histogram gradient boosting with shallow leaves and regularization.",
        "initial_parameters": {"max_iter": 100, "learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        "objective": "binomial deviance (log loss)",
        "strengths": "Captures nonlinear feature interactions with regularized additive trees.",
        "limitations": "More costly to fit and tune; learned candidate scores still need a sequential decoder for route coherence.",
    },
}


def model_zoo(seed: int = 42):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=seed),
        ),
        "gaussian_nb": GaussianNB(var_smoothing=1e-9),
        "decision_tree": DecisionTreeClassifier(
            criterion="gini", max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced", random_state=seed,
        ),
    }
