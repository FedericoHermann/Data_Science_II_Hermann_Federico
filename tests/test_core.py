import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from src.project_pipeline import FeatureEngineer, metrics, choose_threshold, PROJECT_ROOT


def test_feature_engineer_adds_expected_columns():
    X=load_breast_cancer(as_frame=True).data.head(5)
    transformed=FeatureEngineer().fit_transform(X)
    expected={"area_perimeter_ratio","worst_to_mean_radius_ratio","concavity_compactness_ratio"}
    assert expected.issubset(transformed.columns)
    assert transformed.shape[1]==X.shape[1]+3
    assert np.isfinite(transformed.to_numpy()).all()


def test_weighted_cost_math():
    y=pd.Series([0,0,1,1]); probabilities=np.array([0.1,0.8,0.2,0.9])
    result=metrics(y,probabilities,0.5,"demo")
    assert result["false_positives"]==1
    assert result["false_negatives"]==1
    assert result["weighted_error_cost"]==6


def test_threshold_selection_is_valid():
    y=pd.Series([0,0,0,1,1,1]); p=np.array([.05,.2,.3,.6,.8,.95])
    frame,row=choose_threshold(y,p)
    assert len(frame)>1
    assert 0.0<float(row["threshold"])<1.0


def test_generated_consistency_checks_are_all_true():
    path=PROJECT_ROOT/"artifacts"/"consistency_checks.json"
    checks=json.loads(path.read_text(encoding="utf-8"))
    assert checks
    assert all(checks.values())


def test_summary_matches_test_results():
    summary=json.loads((PROJECT_ROOT/"artifacts"/"final_summary.json").read_text(encoding="utf-8"))
    test=pd.read_csv(PROJECT_ROOT/"reports"/"test_results.csv").iloc[0]
    assert summary["selection"]["selected_model"]==test["model"]
    assert np.isclose(summary["selection"]["selected_threshold"],test["threshold"])
    assert np.isclose(summary["test_metrics"]["roc_auc"],test["roc_auc"])
    assert np.isclose(summary["test_metrics"]["pr_auc"],test["pr_auc"])
