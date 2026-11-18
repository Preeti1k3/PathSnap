from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

FEATURES = ["distance_m", "heading_error", "edge_length_m", "speed_mps", "time_gap_s", "candidate_rank"]


def model_zoo(seed=42):
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, random_state=seed),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
    }

