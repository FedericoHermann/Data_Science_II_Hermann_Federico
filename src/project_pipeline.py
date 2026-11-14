from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 42
TEST_SIZE = 0.20
INNER_CV_SPLITS = 3
OUTER_CV_SPLITS = 5
PRACTICAL_TOLERANCE = 0.002
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
MINIMUM_RECALL = 0.95
PLOT_DPI = 150

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "dataset": PROJECT_ROOT / "data" / "dataset_snapshot.csv",
    "dictionary": PROJECT_ROOT / "data" / "data_dictionary.csv",
    "eda_summary": PROJECT_ROOT / "reports" / "eda_summary.csv",
    "missing_summary": PROJECT_ROOT / "reports" / "missing_values.csv",
    "outlier_summary": PROJECT_ROOT / "reports" / "outlier_summary.csv",
    "cv_results": PROJECT_ROOT / "reports" / "nested_cv_results.csv",
    "search_results": PROJECT_ROOT / "reports" / "grid_search_results.csv",
    "selection_table": PROJECT_ROOT / "reports" / "model_selection_table.csv",
    "threshold_analysis": PROJECT_ROOT / "reports" / "threshold_analysis.csv",
    "test_results": PROJECT_ROOT / "reports" / "test_results.csv",
    "shap_importance": PROJECT_ROOT / "reports" / "shap_feature_importance.csv",
    "shap_stability": PROJECT_ROOT / "reports" / "shap_stability.csv",
    "final_report": PROJECT_ROOT / "reports" / "final_report.md",
    "class_distribution": PROJECT_ROOT / "figures" / "class_distribution.png",
    "eda_distributions": PROJECT_ROOT / "figures" / "eda_distributions.png",
    "eda_boxplots": PROJECT_ROOT / "figures" / "eda_boxplots.png",
    "correlation_heatmap": PROJECT_ROOT / "figures" / "correlation_heatmap.png",
    "model_comparison": PROJECT_ROOT / "figures" / "model_comparison.png",
    "roc_pr_curves": PROJECT_ROOT / "figures" / "roc_pr_curves.png",
    "confusion_matrix": PROJECT_ROOT / "figures" / "confusion_matrix.png",
    "shap_bar": PROJECT_ROOT / "figures" / "shap_importance.png",
    "shap_beeswarm": PROJECT_ROOT / "figures" / "shap_beeswarm.png",
    "shap_local": PROJECT_ROOT / "figures" / "shap_local_explanation.png",
    "model": PROJECT_ROOT / "models" / "modelo_final.joblib",
    "explain_model": PROJECT_ROOT / "models" / "xgboost_explicativo.joblib",
    "summary": PROJECT_ROOT / "artifacts" / "final_summary.json",
    "manifest": PROJECT_ROOT / "artifacts" / "run_manifest.json",
    "checks": PROJECT_ROOT / "artifacts" / "consistency_checks.json",
}

for path in PATHS.values():
    path.parent.mkdir(parents=True, exist_ok=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea atributos deterministas sin aprender información de prueba."""
    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        eps = 1e-9
        frame["area_perimeter_ratio"] = frame["mean area"] / (frame["mean perimeter"] + eps)
        frame["worst_to_mean_radius_ratio"] = frame["worst radius"] / (frame["mean radius"] + eps)
        frame["concavity_compactness_ratio"] = frame["mean concavity"] / (frame["mean compactness"] + eps)
        return frame


def native(value):
    if isinstance(value, dict): return {str(k): native(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,set)): return [native(v) for v in value]
    if isinstance(value, np.ndarray): return [native(v) for v in value.tolist()]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        number=float(value); return number if np.isfinite(number) else None
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    if isinstance(value, float): return value if np.isfinite(value) else None
    return value


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''): digest.update(block)
    return digest.hexdigest()


def metrics(y_true, probabilities, threshold, model_name):
    predictions=(np.asarray(probabilities)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y_true,predictions,labels=[0,1]).ravel()
    return {
        "model": model_name, "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true,predictions)),
        "precision": float(precision_score(y_true,predictions,zero_division=0)),
        "recall": float(recall_score(y_true,predictions,zero_division=0)),
        "f1": float(f1_score(y_true,predictions,zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true,probabilities)),
        "pr_auc": float(average_precision_score(y_true,probabilities)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "weighted_error_cost": int(FALSE_NEGATIVE_COST*fn + FALSE_POSITIVE_COST*fp),
    }


def build_candidates():
    logistic=Pipeline([
        ("features",FeatureEngineer()), ("scaler",StandardScaler()),
        ("model",LogisticRegression(max_iter=5000,class_weight="balanced",random_state=RANDOM_SEED)),
    ])
    xgb_base=Pipeline([
        ("features",FeatureEngineer()), ("scaler",StandardScaler()),
        ("model",XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
            subsample=0.9,colsample_bytree=0.9,reg_alpha=0.0,reg_lambda=1.0,
            objective="binary:logistic",eval_metric="logloss",tree_method="hist",
            random_state=RANDOM_SEED,n_jobs=1,verbosity=0)),
    ])
    xgb_pipe=Pipeline([
        ("features",FeatureEngineer()), ("scaler",StandardScaler()),
        ("model",XGBClassifier(objective="binary:logistic",eval_metric="logloss",
            tree_method="hist",random_state=RANDOM_SEED,n_jobs=1,verbosity=0)),
    ])
    return logistic,xgb_base,xgb_pipe


def grid_parameters():
    return {
        "model__n_estimators":[150,300], "model__max_depth":[2,4],
        "model__learning_rate":[0.03,0.08], "model__subsample":[0.8],
    }


def select_model(cv_results: pd.DataFrame):
    best=float(cv_results["outer_roc_auc_mean"].max())
    eligible=cv_results.loc[cv_results["outer_roc_auc_mean"]>=best-PRACTICAL_TOLERANCE].copy()
    selected=eligible.sort_values(
        ["outer_recall_mean","outer_roc_auc_std","elapsed_seconds"],
        ascending=[False,True,True],
    ).iloc[0]
    return selected,eligible


def choose_threshold(y_true, probabilities):
    rows=[metrics(y_true,probabilities,float(t),"OOF_training") for t in np.round(np.arange(0.05,0.951,0.01),2)]
    frame=pd.DataFrame(rows)
    eligible=frame.loc[frame["recall"]>=MINIMUM_RECALL].copy()
    pool=eligible if not eligible.empty else frame
    chosen=pool.sort_values(["weighted_error_cost","f1","precision"],ascending=[True,False,False]).iloc[0]
    return frame,chosen


def create_eda(X: pd.DataFrame,y: pd.Series):
    pd.concat([X,y],axis=1).to_csv(PATHS["dataset"],index=False)
    pd.DataFrame({"feature":X.columns,"description":["Numeric diagnostic feature from sklearn Breast Cancer Wisconsin dataset"]*X.shape[1]}).to_csv(PATHS["dictionary"],index=False)
    X.describe().T.to_csv(PATHS["eda_summary"])
    pd.DataFrame({"column":X.columns,"missing_count":X.isna().sum().values,"missing_pct":100*X.isna().mean().values}).to_csv(PATHS["missing_summary"],index=False)
    q1=X.quantile(0.25); q3=X.quantile(0.75); iqr=q3-q1
    outliers=((X<q1-1.5*iqr)|(X>q3+1.5*iqr)).sum()
    pd.DataFrame({"feature":X.columns,"iqr_outlier_count":outliers.values,"iqr_outlier_pct":100*outliers.values/len(X)}).to_csv(PATHS["outlier_summary"],index=False)

    counts=y.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(6,4)); ax.bar(["Benigna","Maligna"],counts.values,color=["#0072B2","#D55E00"])
    ax.set_title("Distribución de clases"); ax.set_ylabel("Observaciones"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(PATHS["class_distribution"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)

    cols=["mean radius","mean texture","mean concavity","worst radius","worst texture","worst concave points"]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,col in zip(axes.ravel(),cols):
        for target,color,label in [(0,"#0072B2","Benigna"),(1,"#D55E00","Maligna")]:
            ax.hist(X.loc[y==target,col],bins=20,alpha=.55,color=color,label=label)
        ax.set_title(col); ax.grid(axis="y",alpha=.2)
    axes[0,0].legend(); fig.suptitle("Distribuciones por clase"); fig.tight_layout()
    fig.savefig(PATHS["eda_distributions"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,col in zip(axes.ravel(),cols):
        ax.boxplot([X.loc[y==0,col],X.loc[y==1,col]],labels=["Benigna","Maligna"])
        ax.set_title(col); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Boxplots por clase"); fig.tight_layout()
    fig.savefig(PATHS["eda_boxplots"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)

    corr=X.corr(); fig,ax=plt.subplots(figsize=(16,14)); image=ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto")
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns,rotation=90,fontsize=6); ax.set_yticklabels(corr.columns,fontsize=6)
    fig.colorbar(image,ax=ax,label="Correlación de Pearson"); ax.set_title("Matriz de correlación")
    fig.tight_layout(); fig.savefig(PATHS["correlation_heatmap"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)


def explain_xgboost(pipeline,X_train,X_test,local_index):
    engineered_train=pipeline.named_steps["features"].transform(X_train)
    engineered_test=pipeline.named_steps["features"].transform(X_test)
    scaler=pipeline.named_steps["scaler"]
    test_scaled=scaler.transform(engineered_test)
    names=engineered_train.columns.tolist(); model=pipeline.named_steps["model"]
    contrib=model.get_booster().predict(xgb.DMatrix(test_scaled,feature_names=names),pred_contribs=True)
    values=np.asarray(contrib[:,:-1],dtype=float); base=np.asarray(contrib[:,-1],dtype=float)
    explanation=shap.Explanation(values=values,base_values=base,data=test_scaled,feature_names=names)
    importance=pd.DataFrame({"feature":names,"mean_abs_shap":np.abs(values).mean(axis=0),"mean_signed_shap":values.mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
    importance.to_csv(PATHS["shap_importance"],index=False)
    top=importance.head(15).sort_values("mean_abs_shap")
    fig,ax=plt.subplots(figsize=(9,7)); ax.barh(top["feature"],top["mean_abs_shap"],color="#0072B2")
    ax.set_xlabel("Media de |SHAP|"); ax.set_title("Importancia global SHAP"); ax.grid(axis="x",alpha=.25)
    fig.tight_layout(); fig.savefig(PATHS["shap_bar"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)
    plt.figure(); shap.plots.beeswarm(explanation,max_display=15,show=False); plt.tight_layout(); plt.savefig(PATHS["shap_beeswarm"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close()
    plt.figure(); shap.plots.waterfall(explanation[local_index],max_display=12,show=False); plt.tight_layout(); plt.savefig(PATHS["shap_local"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close()


def run_project():
    started=time.time()
    data=load_breast_cancer(as_frame=True); X=data.data.copy(); y=(data.target==0).astype(int).rename("malignant")
    if X.empty or X.columns.duplicated().any() or X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Dataset inválido")
    create_eda(X,y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_SEED)
    if set(X_train.index)&set(X_test.index): raise ValueError("Train/test overlap")

    logistic,xgb_base,xgb_pipe=build_candidates()
    inner=StratifiedKFold(n_splits=INNER_CV_SPLITS,shuffle=True,random_state=RANDOM_SEED)
    outer=StratifiedKFold(n_splits=OUTER_CV_SPLITS,shuffle=True,random_state=RANDOM_SEED+1)
    xgb_grid=GridSearchCV(xgb_pipe,grid_parameters(),scoring="roc_auc",cv=inner,n_jobs=1,refit=True,return_train_score=True,error_score="raise")
    candidates={"LogisticRegression":logistic,"XGBoostBaseline":xgb_base,"XGBoostGridSearch":xgb_grid}
    scoring={"roc_auc":"roc_auc","pr_auc":"average_precision","accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1"}
    rows=[]
    for name,estimator in candidates.items():
        start=time.perf_counter(); scores=cross_validate(estimator,X_train,y_train,cv=outer,scoring=scoring,return_train_score=True,n_jobs=1,error_score="raise")
        row={"model":name,"elapsed_seconds":time.perf_counter()-start}
        for metric in scoring:
            row[f"outer_{metric}_mean"]=float(np.mean(scores[f"test_{metric}"]))
            row[f"outer_{metric}_std"]=float(np.std(scores[f"test_{metric}"],ddof=1))
            row[f"train_{metric}_mean"]=float(np.mean(scores[f"train_{metric}"]))
            row[f"gap_{metric}"]=row[f"train_{metric}_mean"]-row[f"outer_{metric}_mean"]
        rows.append(row)
    cv=pd.DataFrame(rows).sort_values("outer_roc_auc_mean",ascending=False); cv.to_csv(PATHS["cv_results"],index=False)
    selected,eligible=select_model(cv); selected_name=str(selected["model"]); selected_search=candidates[selected_name]
    oof=cross_val_predict(selected_search,X_train,y_train,cv=outer,method="predict_proba",n_jobs=1)[:,1]
    thresholds,threshold_row=choose_threshold(y_train,oof); selected_threshold=float(threshold_row["threshold"]); thresholds.to_csv(PATHS["threshold_analysis"],index=False)
    selected_search.fit(X_train,y_train); final_model=selected_search.best_estimator_ if hasattr(selected_search,"best_estimator_") else selected_search
    probabilities=final_model.predict_proba(X_test)[:,1]; final_metrics=metrics(y_test,probabilities,selected_threshold,selected_name)
    pd.DataFrame([final_metrics]).to_csv(PATHS["test_results"],index=False)

    selection_table=pd.DataFrame([
        {"perspective":"Selección metodológica","criterion":f"ROC-AUC CV máximo con tolerancia {PRACTICAL_TOLERANCE}; desempate por recall, estabilidad y tiempo","selected_method":selected_name,"evidence":f"ROC-AUC={selected['outer_roc_auc_mean']:.6f} ± {selected['outer_roc_auc_std']:.6f}","defines_selection":True},
        {"perspective":"Umbral operativo","criterion":f"Costo FN:FP={FALSE_NEGATIVE_COST}:{FALSE_POSITIVE_COST}, recall mínimo={MINIMUM_RECALL}","selected_method":selected_name,"evidence":f"Umbral OOF={selected_threshold:.2f}","defines_selection":True},
        {"perspective":"Evaluación final","criterion":"Prueba aislada; no reabre la selección","selected_method":selected_name,"evidence":f"ROC-AUC={final_metrics['roc_auc']:.6f}; PR-AUC={final_metrics['pr_auc']:.6f}; FN={final_metrics['false_negatives']}; FP={final_metrics['false_positives']}","defines_selection":False},
    ]); selection_table.to_csv(PATHS["selection_table"],index=False)

    fig,ax=plt.subplots(figsize=(10,6)); positions=np.arange(len(cv)); ax.bar(positions,cv["outer_roc_auc_mean"],yerr=cv["outer_roc_auc_std"],capsize=5,color=["#0072B2","#D55E00","#009E73"])
    ax.set_xticks(positions); ax.set_xticklabels(cv["model"],rotation=15); ax.set_ylim(.90,1.01); ax.set_ylabel("ROC-AUC"); ax.set_title("Validación cruzada anidada"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(PATHS["model_comparison"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)
    fpr,tpr,_=roc_curve(y_test,probabilities); prec,rec,_=precision_recall_curve(y_test,probabilities)
    fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].plot(fpr,tpr,label=f"AUC={final_metrics['roc_auc']:.3f}"); axes[0].plot([0,1],[0,1],"--",color="gray"); axes[0].set(xlabel="FPR",ylabel="TPR",title="Curva ROC")
    axes[1].plot(rec,prec,color="#D55E00",label=f"AP={final_metrics['pr_auc']:.3f}"); axes[1].set(xlabel="Recall",ylabel="Precision",title="Curva Precision-Recall")
    for axis in axes: axis.legend(); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(PATHS["roc_pr_curves"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)
    preds=(probabilities>=selected_threshold).astype(int); matrix=confusion_matrix(y_test,preds,labels=[0,1])
    fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(matrix,cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=13)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["Benigna","Maligna"]); ax.set_yticklabels(["Benigna","Maligna"]); ax.set_xlabel("Predicción"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión, umbral={selected_threshold:.2f}")
    fig.tight_layout(); fig.savefig(PATHS["confusion_matrix"],dpi=PLOT_DPI,bbox_inches="tight"); plt.close(fig)

    # Modelo arbóreo explicativo, ajustado sólo con entrenamiento.
    xgb_grid.fit(X_train,y_train); explain_model=xgb_grid.best_estimator_; grid_df=pd.DataFrame(xgb_grid.cv_results_).sort_values("rank_test_score")
    keep=[c for c in grid_df.columns if c.startswith("param_") or c in ["rank_test_score","mean_test_score","std_test_score","mean_train_score","std_train_score","mean_fit_time"]]
    grid_df[keep].to_csv(PATHS["search_results"],index=False)
    local_index=int(np.argmax(explain_model.predict_proba(X_test)[:,1])); explain_xgboost(explain_model,X_train,X_test,local_index)
    stability=[]
    best_params=xgb_grid.best_params_
    for seed in [41,42,43,44,45]:
        X_sub,X_val,y_sub,_=train_test_split(X_train,y_train,test_size=.25,stratify=y_train,random_state=seed)
        pipe=build_candidates()[2].set_params(**best_params); pipe.fit(X_sub,y_sub)
        engineered=pipe.named_steps["features"].transform(X_val); scaled=pipe.named_steps["scaler"].transform(engineered); names=engineered.columns.tolist()
        values=pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(scaled,feature_names=names),pred_contribs=True)[:,:-1]
        fold=pd.DataFrame({"feature":names,"importance":np.abs(values).mean(axis=0)}); fold["rank"]=fold["importance"].rank(ascending=False,method="min"); fold["seed"]=seed; stability.append(fold)
    stab=pd.concat(stability,ignore_index=True).groupby("feature").agg(mean_importance=("importance","mean"),std_importance=("importance","std"),mean_rank=("rank","mean"),top10_frequency=("rank",lambda v:float(np.mean(v<=10)))).reset_index().sort_values("mean_rank")
    stab.to_csv(PATHS["shap_stability"],index=False)

    joblib.dump(final_model,PATHS["model"]); joblib.dump(explain_model,PATHS["explain_model"])
    reloaded=joblib.load(PATHS["model"]); reload_matches=bool(np.allclose(final_model.predict_proba(X_test),reloaded.predict_proba(X_test)))
    versions={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgb.__version__,"shap":shap.__version__,"joblib":joblib.__version__}
    summary={
        "problem":{"type":"binary classification","target":"malignant","positive_class":1,"primary_metric":"roc_auc","false_negative_cost":FALSE_NEGATIVE_COST,"false_positive_cost":FALSE_POSITIVE_COST},
        "data":{"rows":len(X),"original_features":X.shape[1],"train_rows":len(X_train),"test_rows":len(X_test)},
        "selection":{"selected_model":selected_name,"selected_threshold":selected_threshold,"selected_cv_roc_auc_mean":float(selected["outer_roc_auc_mean"]),"selected_cv_roc_auc_std":float(selected["outer_roc_auc_std"]),"rule":"ROC-AUC within practical tolerance, then recall, stability and runtime"},
        "test_metrics":final_metrics,"xgboost_explanatory_best_params":best_params,"versions":versions,
    }
    PATHS["summary"].write_text(json.dumps(native(summary),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    report=f"""# Informe final\n\n## Formulación\n\nClasificación binaria. Variable objetivo: `malignant` (1 = maligna). Métrica principal: ROC-AUC. Costos definidos antes de prueba: FN={FALSE_NEGATIVE_COST}, FP={FALSE_POSITIVE_COST}.\n\n## Selección metodológica\n\n- Modelo: **{selected_name}**.\n- ROC-AUC CV: **{selected['outer_roc_auc_mean']:.6f} ± {selected['outer_roc_auc_std']:.6f}**.\n- Umbral OOF: **{selected_threshold:.2f}**.\n\n## Prueba aislada\n\n- ROC-AUC: **{final_metrics['roc_auc']:.6f}**.\n- PR-AUC: **{final_metrics['pr_auc']:.6f}**.\n- Accuracy: **{final_metrics['accuracy']:.6f}**.\n- Precision: **{final_metrics['precision']:.6f}**.\n- Recall: **{final_metrics['recall']:.6f}**.\n- F1: **{final_metrics['f1']:.6f}**.\n- FN: **{final_metrics['false_negatives']}**.\n- FP: **{final_metrics['false_positives']}**.\n- Costo ponderado: **{final_metrics['weighted_error_cost']}**.\n\n## Explicabilidad\n\nSHAP se calculó sobre el XGBoost optimizado como modelo explicativo complementario. Si el modelo final seleccionado es otro, SHAP no se presenta como explicación del modelo final. Las contribuciones son predictivas, no causales.\n\n## Limitaciones\n\nDataset educativo y moderado. Se requiere validación externa, calibración, revisión experta y evaluación de impacto antes de cualquier uso real.\n"""
    PATHS["final_report"].write_text(report,encoding="utf-8")

    # Eliminar especificaciones auxiliares de gráficos que algunos entornos crean en la raíz.
    for obsolete_name in ["model_comparison.json", "shap_importance.json"]:
        obsolete_path = PROJECT_ROOT / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    required_before_checks=[p for key,p in PATHS.items() if key not in {"checks","manifest"}]
    files_exist=all(p.is_file() and p.stat().st_size>0 for p in required_before_checks)
    checks={
        "all_required_files_exist":files_exist,
        "selected_model_matches_report":selected_name in report,
        "threshold_matches_report":f"{selected_threshold:.2f}" in report,
        "roc_auc_matches_report":f"{final_metrics['roc_auc']:.6f}" in report,
        "pr_auc_matches_report":f"{final_metrics['pr_auc']:.6f}" in report,
        "false_negatives_match_report":f"FN: **{final_metrics['false_negatives']}**" in report,
        "false_positives_match_report":f"FP: **{final_metrics['false_positives']}**" in report,
        "model_reload_matches":reload_matches,
        "test_isolation_preserved":len(set(X_train.index)&set(X_test.index))==0,
        "threshold_in_valid_range":0.0<selected_threshold<1.0,
    }
    if not all(checks.values()): raise ValueError(f"Consistency checks failed: {checks}")
    PATHS["checks"].write_text(json.dumps(checks,indent=2),encoding="utf-8")
    manifest={"run_started_epoch":started,"duration_seconds":time.time()-started,"random_seed":RANDOM_SEED,"files":{str(p.relative_to(PROJECT_ROOT)):sha256(p) for p in PATHS.values() if p.is_file()}}
    PATHS["manifest"].write_text(json.dumps(native(manifest),indent=2),encoding="utf-8")
    return summary


if __name__=="__main__":
    print(json.dumps(run_project(),ensure_ascii=False,indent=2))
