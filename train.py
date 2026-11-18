from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    import optuna
from pathsnap.data import load_trip, candidate_table, inventory
from pathsnap.models import FEATURES, model_zoo
from pathsnap.decoder import viterbi


def load_partition(root, name, include_route=True):
    xs=[]
    for folder in sorted((Path(root)/name).iterdir()):
        if folder.is_dir():
            trip=load_trip(folder, include_route=include_route)
            table=candidate_table(trip)
            table["trip_id"]=trip.trip_id
            xs.append((trip,table))
    return xs


def evaluate(model, trips):
    ys=[]; probs=[]; point_hits=[]; hmm_hits=[]; classifier_cont=[]; hmm_cont=[]; candidate_misses=[]
    infer_start=time.perf_counter()
    for trip, table in trips:
        x=table[FEATURES]
        p=model.predict_proba(x)[:,1]
        groups=list(table.groupby("obs",sort=True))
        cand=np.array([g.arc_id.to_numpy(int) for _,g in groups])
        prob=np.array([p[g.index.to_numpy()] for _,g in groups])
        labels=np.array([int(g.label.iloc[0]) for _,g in groups])
        route_labels=np.array([int(g.reference_arc_id.iloc[0]) for _,g in groups])
        top=cand[np.arange(len(cand)),np.argmax(prob,axis=1)]
        decoded=viterbi(cand,prob,trip.arcs,np.r_[0.,np.diff(trip.track.time.to_numpy(float))])
        valid=route_labels>=0
        point_hits.extend((top[valid]==route_labels[valid]).astype(int).tolist())
        hmm_hits.extend((decoded[valid]==route_labels[valid]).astype(int).tolist())
        candidate_misses.extend((~np.any(cand==route_labels[:,None],axis=1))[valid].astype(int).tolist())
        if len(top)>1:
            classifier_cont.extend((trip.arcs[top[:-1],1]==trip.arcs[top[1:],0]).astype(int).tolist())
            hmm_cont.extend((trip.arcs[decoded[:-1],1]==trip.arcs[decoded[1:],0]).astype(int).tolist())
        y=table.label.to_numpy()
        ys.extend(y.tolist()); probs.extend(p.tolist())
    infer_time=time.perf_counter()-infer_start
    # Candidate binary metrics, plus candidate-ranking Top-1 from observation winners.
    pred_bin=np.asarray(probs)>=.5
    tn,fp,fn,tp=confusion_matrix(ys,pred_bin,labels=[0,1]).ravel()
    return {"candidate_precision":float(precision_score(ys,pred_bin,zero_division=0)),
            "candidate_recall":float(recall_score(ys,pred_bin,zero_division=0)),
            "candidate_f1":float(f1_score(ys,pred_bin,zero_division=0)),
            "candidate_macro_f1":float(f1_score(ys,pred_bin,average="macro",zero_division=0)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "candidate_log_loss":float(log_loss(ys,np.clip(probs,1e-7,1-1e-7),labels=[0,1])),
            "top1_accuracy":float(np.mean(point_hits)) if point_hits else float("nan"),
            "candidate_miss_rate":float(np.mean(candidate_misses)) if candidate_misses else float("nan"),
            "classifier_route_continuity":float(np.mean(classifier_cont)) if classifier_cont else float("nan"),
            "hmm_top1_accuracy":float(np.mean(hmm_hits)) if hmm_hits else float("nan"),
            "hmm_route_continuity":float(np.mean(hmm_cont)) if hmm_cont else float("nan"),
            "inference_seconds":infer_time}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="../Student folder")
    ap.add_argument("--output",default="artifacts")
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    root=Path(args.data).resolve(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    inventory(root).to_csv(out/"dataset_inventory.csv",index=False)
    train=load_partition(root,"train"); test=load_partition(root,"student_test")
    X=pd.concat([t[FEATURES] for _,t in train],ignore_index=True)
    y=pd.concat([t.label for _,t in train],ignore_index=True)
    groups=pd.concat([pd.Series(t.trip_id,index=t.index) for _,t in train],ignore_index=True)
    # Candidate rows are grouped by trip to prevent within-trip leakage.
    cv=GroupKFold(n_splits=5)
    rows=[]; fitted={}
    for name, model in model_zoo(args.seed).items():
        start=time.perf_counter()
        scores=cross_val_score(model,X,y,groups=groups,cv=cv,scoring="f1_macro",n_jobs=1)
        model.fit(X,y); fitted[name]=model
        metrics=evaluate(model,test)
        rows.append({"model":name,"cv_f1_mean":float(scores.mean()),"cv_f1_std":float(scores.std()),"fit_seconds":time.perf_counter()-start,**metrics})
        print(rows[-1])
    results=pd.DataFrame(rows).sort_values("cv_f1_mean",ascending=False)
    results.to_csv(out/"model_comparison.csv",index=False)
    winner=str(results.iloc[0].model)
    if HAS_SKOPT:
        tuned=BayesSearchCV(fitted[winner],{"n_estimators":Integer(80,240),
            "min_samples_leaf":Integer(1,8)} if winner=="extra_trees" else
            {"max_depth":Integer(3,16), "min_samples_leaf":Integer(1,8)} if winner=="decision_tree" else
            {"logisticregression__C":Real(0.02,20,prior="log-uniform")} if winner=="logistic_regression" else
            {"max_leaf_nodes":Integer(7,31),"learning_rate":Real(.02,.2)} if winner=="hist_gradient_boosting" else
            {"var_smoothing":Real(1e-11,1e-7,prior="log-uniform")} if winner=="gaussian_nb" else
            {"kneighborsclassifier__n_neighbors":Integer(5,35)},
            n_iter=12, scoring="f1_macro", cv=cv, random_state=args.seed, n_jobs=1, refit=True)
        tuned.fit(X,y,groups=groups)
        pd.DataFrame(tuned.cv_results_).to_csv(out/"bayes_trials.csv",index=False)
        best_est = tuned.best_estimator_
        best_params = tuned.best_params_
        best_score = float(tuned.best_score_)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def objective(trial):
            if winner == "extra_trees":
                n_est = trial.suggest_int("n_estimators", 80, 240, step=20)
                min_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
                clf = ExtraTreesClassifier(n_estimators=n_est, min_samples_leaf=min_leaf, class_weight="balanced", n_jobs=-1, random_state=args.seed)
            elif winner == "decision_tree":
                max_d = trial.suggest_int("max_depth", 3, 16)
                min_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
                clf = DecisionTreeClassifier(max_depth=max_d, min_samples_leaf=min_leaf, class_weight="balanced", random_state=args.seed)
            elif winner == "logistic_regression":
                c_val = trial.suggest_float("C", 0.02, 20.0, log=True)
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=c_val, max_iter=400, class_weight="balanced", random_state=args.seed))
            elif winner == "hist_gradient_boosting":
                max_leaf = trial.suggest_int("max_leaf_nodes", 7, 31)
                lr = trial.suggest_float("learning_rate", 0.02, 0.2)
                clf = HistGradientBoostingClassifier(max_leaf_nodes=max_leaf, learning_rate=lr, random_state=args.seed)
            elif winner == "knn":
                k_val = trial.suggest_int("n_neighbors", 5, 35)
                clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k_val, weights="distance", n_jobs=-1))
            else:
                v_smooth = trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
                clf = GaussianNB(var_smoothing=v_smooth)
            scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="f1_macro", n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=12)
        study.trials_dataframe().to_csv(out/"bayes_trials.csv", index=False)
        best_params = study.best_params
        best_score = float(study.best_value)
        # Refit best model
        if winner == "extra_trees":
            best_est = ExtraTreesClassifier(**best_params, class_weight="balanced", n_jobs=-1, random_state=args.seed)
        elif winner == "decision_tree":
            best_est = DecisionTreeClassifier(**best_params, class_weight="balanced", random_state=args.seed)
        elif winner == "logistic_regression":
            best_est = make_pipeline(StandardScaler(), LogisticRegression(C=best_params["C"], max_iter=400, class_weight="balanced", random_state=args.seed))
        elif winner == "hist_gradient_boosting":
            best_est = HistGradientBoostingClassifier(**best_params, random_state=args.seed)
        elif winner == "knn":
            best_est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_params["n_neighbors"], weights="distance", n_jobs=-1))
        else:
            best_est = GaussianNB(**best_params)
        best_est.fit(X, y)

    joblib.dump(best_est, out/"model.joblib")
    tuned_test=evaluate(best_est, test)
    (out/"tuned_student_test.json").write_text(json.dumps(tuned_test,indent=2))
    (out/"model_config.json").write_text(json.dumps({"winner":winner,"best_params":best_params,"cv_best_score":best_score,"features":FEATURES,"seed":args.seed,"tuned_student_test":tuned_test},indent=2))
    print("Best tuned model:",winner,best_params,best_score)

if __name__=="__main__": main()

# Integrated Optuna and BayesSearchCV hyperparameter optimization
