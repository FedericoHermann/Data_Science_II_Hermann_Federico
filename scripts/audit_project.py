import ast
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
errors=[]
required=["README.md","requirements.txt","notebooks/proyecto_final_ml.ipynb","src/project_pipeline.py","artifacts/final_summary.json","artifacts/consistency_checks.json","reports/final_report.md","reports/model_selection_table.csv","models/modelo_final.joblib"]
for relative in required:
    path=ROOT/relative
    if not path.is_file() or path.stat().st_size==0: errors.append(f"Missing or empty: {relative}")
for obsolete in ["model_comparison.json","shap_importance.json"]:
    if (ROOT/obsolete).exists(): errors.append(f"Obsolete root artifact: {obsolete}")
ast.parse((ROOT/"src/project_pipeline.py").read_text(encoding="utf-8"))
nb=json.loads((ROOT/"notebooks/proyecto_final_ml.ipynb").read_text(encoding="utf-8"))
if nb.get("nbformat")!=4: errors.append("Notebook is not nbformat 4")
checks=json.loads((ROOT/"artifacts/consistency_checks.json").read_text(encoding="utf-8"))
if not checks or not all(checks.values()): errors.append(f"Consistency checks failed: {checks}")
summary=json.loads((ROOT/"artifacts/final_summary.json").read_text(encoding="utf-8"))
test=pd.read_csv(ROOT/"reports/test_results.csv").iloc[0]
selection=pd.read_csv(ROOT/"reports/model_selection_table.csv")
if summary["selection"]["selected_model"]!=test["model"]: errors.append("Selected model mismatch")
if abs(float(summary["selection"]["selected_threshold"])-float(test["threshold"]))>1e-12: errors.append("Threshold mismatch")
if selection["selected_method"].nunique()!=1: errors.append("Selection table has multiple methods")
if errors:
    print("AUDIT FAILED")
    print("\n".join(f"- {e}" for e in errors))
    sys.exit(1)
print("AUDIT PASSED")
print(f"Model: {summary['selection']['selected_model']}")
print(f"Threshold: {summary['selection']['selected_threshold']:.2f}")
print(f"ROC-AUC test: {summary['test_metrics']['roc_auc']:.6f}")
print(f"PR-AUC test: {summary['test_metrics']['pr_auc']:.6f}")
