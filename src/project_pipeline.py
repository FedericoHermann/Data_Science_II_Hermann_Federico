from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from scipy import __version__ as scipy_version
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ------------------------------------------------------------
# CONFIGURACION CANONICA
# ------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.20
INNER_CV_SPLITS = 3
OUTER_CV_SPLITS = 3
PRACTICAL_TOLERANCE = 0.002
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1
MINIMUM_RECALL_FOR_THRESHOLD = 0.95
PLOT_DPI = 150

PROJECT_ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd()
PATHS = {
    'data_readme': PROJECT_ROOT / 'data' / 'README.md',
    'model': PROJECT_ROOT / 'models' / 'modelo_final.joblib',
    'cv_results': PROJECT_ROOT / 'reports' / 'nested_cv_results.csv',
    'test_results': PROJECT_ROOT / 'reports' / 'test_results.csv',
    'threshold_analysis': PROJECT_ROOT / 'reports' / 'threshold_analysis.csv',
    'selection_table': PROJECT_ROOT / 'reports' / 'model_selection_table.csv',
    'shap_stability': PROJECT_ROOT / 'reports' / 'shap_stability.csv',
    'final_report': PROJECT_ROOT / 'reports' / 'final_report.md',
    'eda_distributions': PROJECT_ROOT / 'figures' / 'eda_distributions.png',
    'correlation_heatmap': PROJECT_ROOT / 'figures' / 'correlation_heatmap.png',
    'model_comparison': PROJECT_ROOT / 'figures' / 'model_comparison.png',
    'roc_pr_curves': PROJECT_ROOT / 'figures' / 'roc_pr_curves.png',
    'confusion_matrix': PROJECT_ROOT / 'figures' / 'confusion_matrix.png',
    'shap_importance': PROJECT_ROOT / 'figures' / 'shap_importance.png',
    'shap_beeswarm': PROJECT_ROOT / 'figures' / 'shap_beeswarm.png',
    'shap_local': PROJECT_ROOT / 'figures' / 'shap_local_explanation.png',
    'summary_json': PROJECT_ROOT / 'artifacts' / 'final_summary.json',
    'manifest_json': PROJECT_ROOT / 'artifacts' / 'run_manifest.json',
    'checks_json': PROJECT_ROOT / 'artifacts' / 'consistency_checks.json',
}

for path in PATHS.values():
    path.parent.mkdir(parents=True, exist_ok=True)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Crea ratios deterministas y conserva los nombres de columnas."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        epsilon = 1e-9
        result['area_perimeter_ratio'] = result['mean area'] / (result['mean perimeter'] + epsilon)
        result['worst_to_mean_radius_ratio'] = result['worst radius'] / (result['mean radius'] + epsilon)
        result['concavity_compactness_ratio'] = result['mean concavity'] / (result['mean compactness'] + epsilon)
        return result


def to_native(value):
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def classification_metrics(y_true, probabilities, threshold, model_name):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        'model': model_name,
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, predictions)),
        'precision': float(precision_score(y_true, predictions, zero_division=0)),
        'recall': float(recall_score(y_true, predictions, zero_division=0)),
        'f1': float(f1_score(y_true, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, probabilities)),
        'pr_auc': float(average_precision_score(y_true, probabilities)),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'weighted_error_cost': int(FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp),
    }


def build_models():
    logistic = Pipeline([
        ('features', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_SEED)),
    ])
    xgb_baseline = Pipeline([
        ('features', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_alpha=0.0,
            reg_lambda=1.0, objective='binary:logistic', eval_metric='logloss',
            tree_method='hist', random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
        )),
    ])
    xgb_search_pipeline = Pipeline([
        ('features', FeatureEngineer()),
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            random_state=RANDOM_SEED, n_jobs=1, verbosity=0,
        )),
    ])
    return logistic, xgb_baseline, xgb_search_pipeline


def plot_eda(X, y):
    selected = ['mean radius', 'mean texture', 'mean concavity', 'worst radius', 'worst texture', 'worst concave points']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.ravel(), selected):
        for target, color, label in [(0, '#0072B2', 'Benigna'), (1, '#D55E00', 'Maligna')]:
            ax.hist(X.loc[y == target, column], bins=20, alpha=0.55, color=color, label=label)
        ax.set_title(column)
        ax.grid(axis='y', alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle('Distribuciones seleccionadas por clase')
    fig.tight_layout()
    fig.savefig(PATHS['eda_distributions'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    correlation = X.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(correlation.columns, fontsize=6)
    fig.colorbar(image, ax=ax, label='Correlación de Pearson')
    ax.set_title('Matriz de correlación')
    fig.tight_layout()
    fig.savefig(PATHS['correlation_heatmap'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


def select_model(cv_results):
    best_auc = cv_results['outer_roc_auc_mean'].max()
    eligible = cv_results.loc[
        cv_results['outer_roc_auc_mean'] >= best_auc - PRACTICAL_TOLERANCE
    ].copy()
    # Desempate predefinido: mayor recall, menor desviación y menor tiempo.
    selected = eligible.sort_values(
        ['outer_recall_mean', 'outer_roc_auc_std', 'elapsed_seconds'],
        ascending=[False, True, True],
    ).iloc[0]
    return selected, eligible


def choose_threshold(y_true, probabilities):
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metric = classification_metrics(y_true, probabilities, threshold, 'OOF_training')
        rows.append(metric)
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame['recall'] >= MINIMUM_RECALL_FOR_THRESHOLD].copy()
    if eligible.empty:
        chosen = frame.sort_values(['weighted_error_cost', 'recall', 'f1'], ascending=[True, False, False]).iloc[0]
    else:
        chosen = eligible.sort_values(['weighted_error_cost', 'f1', 'precision'], ascending=[True, False, False]).iloc[0]
    return frame, chosen


def explain_xgboost(pipeline, X_train, X_test, selected_index):
    engineered_train = pipeline.named_steps['features'].transform(X_train)
    engineered_test = pipeline.named_steps['features'].transform(X_test)
    scaler = pipeline.named_steps['scaler']
    train_scaled = scaler.transform(engineered_train)
    test_scaled = scaler.transform(engineered_test)
    feature_names = engineered_train.columns.tolist()
    model = pipeline.named_steps['model']

    dtest = xgb.DMatrix(test_scaled, feature_names=feature_names)
    contributions = model.get_booster().predict(dtest, pred_contribs=True)
    shap_values = np.asarray(contributions[:, :-1], dtype=float)
    base_values = np.asarray(contributions[:, -1], dtype=float)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=test_scaled,
        feature_names=feature_names,
    )

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_signed_shap': shap_values.mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False)

    top = importance.head(15).sort_values('mean_abs_shap')
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top['feature'], top['mean_abs_shap'], color='#0072B2')
    ax.set_xlabel('Media de |SHAP|')
    ax.set_title('Importancia global SHAP')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS['shap_importance'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(PATHS['shap_beeswarm'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()

    local = explanation[selected_index]
    plt.figure()
    shap.plots.waterfall(local, max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(PATHS['shap_local'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    return importance


def run_project():
    started = time.time()
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int).rename('malignant')

    if X.empty or X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError('El dataset no supera las validaciones de integridad.')

    plot_eda(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    if set(X_train.index) & set(X_test.index):
        raise ValueError('Entrenamiento y prueba se superponen.')

    logistic, xgb_baseline, xgb_search_pipeline = build_models()
    inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED + 1)
    grid = {
        'model__n_estimators': [150, 300],
        'model__max_depth': [2, 4],
        'model__learning_rate': [0.03, 0.08],
        'model__subsample': [0.8],
    }
    xgb_grid = GridSearchCV(
        xgb_search_pipeline, grid, scoring='roc_auc', cv=inner_cv,
        n_jobs=1, refit=True, return_train_score=True, error_score='raise'
    )
    candidates = {
        'LogisticRegression': logistic,
        'XGBoostBaseline': xgb_baseline,
        'XGBoostGridSearch': xgb_grid,
    }
    scoring = {
        'roc_auc': 'roc_auc', 'pr_auc': 'average_precision', 'accuracy': 'accuracy',
        'precision': 'precision', 'recall': 'recall', 'f1': 'f1'
    }
    cv_rows = []
    for name, estimator in candidates.items():
        start = time.perf_counter()
        scores = cross_validate(
            estimator, X_train, y_train, cv=outer_cv, scoring=scoring,
            return_train_score=True, n_jobs=1, error_score='raise'
        )
        row = {'model': name, 'elapsed_seconds': time.perf_counter() - start}
        for metric in scoring:
            row[f'outer_{metric}_mean'] = float(np.mean(scores[f'test_{metric}']))
            row[f'outer_{metric}_std'] = float(np.std(scores[f'test_{metric}'], ddof=1))
            row[f'train_{metric}_mean'] = float(np.mean(scores[f'train_{metric}']))
            row[f'gap_{metric}'] = row[f'train_{metric}_mean'] - row[f'outer_{metric}_mean']
        cv_rows.append(row)
    cv_results = pd.DataFrame(cv_rows).sort_values('outer_roc_auc_mean', ascending=False)
    cv_results.to_csv(PATHS['cv_results'], index=False)

    selected_row, eligible = select_model(cv_results)
    selected_name = str(selected_row['model'])
    selected_estimator = candidates[selected_name]

    # Umbral con predicciones out-of-fold del procedimiento seleccionado.
    oof_prob = cross_val_predict(
        selected_estimator, X_train, y_train, cv=outer_cv,
        method='predict_proba', n_jobs=1
    )[:, 1]
    threshold_analysis, threshold_row = choose_threshold(y_train, oof_prob)
    selected_threshold = float(threshold_row['threshold'])
    threshold_analysis.to_csv(PATHS['threshold_analysis'], index=False)

    # Ajuste final usando únicamente entrenamiento.
    selected_estimator.fit(X_train, y_train)
    final_model = selected_estimator.best_estimator_ if hasattr(selected_estimator, 'best_estimator_') else selected_estimator
    probabilities = final_model.predict_proba(X_test)[:, 1]
    final_metrics = classification_metrics(
        y_test, probabilities, selected_threshold, selected_name
    )
    test_results = pd.DataFrame([final_metrics])
    test_results.to_csv(PATHS['test_results'], index=False)

    selection_table = pd.DataFrame([
        {
            'perspective': 'Selección metodológica',
            'criterion': f'Mayor ROC-AUC medio; tolerancia práctica={PRACTICAL_TOLERANCE}',
            'selected_method': selected_name,
            'evidence': f"ROC-AUC CV={selected_row['outer_roc_auc_mean']:.6f} ± {selected_row['outer_roc_auc_std']:.6f}",
            'defines_selection': True,
        },
        {
            'perspective': 'Umbral operativo',
            'criterion': f'Menor costo ponderado con recall >= {MINIMUM_RECALL_FOR_THRESHOLD}',
            'selected_method': selected_name,
            'evidence': f'Umbral OOF={selected_threshold:.2f}',
            'defines_selection': True,
        },
        {
            'perspective': 'Evaluación final',
            'criterion': 'Conjunto de prueba aislado; no reabre la selección',
            'selected_method': selected_name,
            'evidence': f"ROC-AUC={final_metrics['roc_auc']:.6f}; PR-AUC={final_metrics['pr_auc']:.6f}; FN={final_metrics['false_negatives']}; FP={final_metrics['false_positives']}",
            'defines_selection': False,
        },
    ])
    selection_table.to_csv(PATHS['selection_table'], index=False)

    # Gráficos de comparación y evaluación.
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(cv_results))
    ax.bar(positions, cv_results['outer_roc_auc_mean'], yerr=cv_results['outer_roc_auc_std'], capsize=5, color=['#0072B2', '#D55E00', '#009E73'])
    ax.set_xticks(positions)
    ax.set_xticklabels(cv_results['model'], rotation=15)
    ax.set_ylim(0.90, 1.01)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Comparación mediante validación cruzada anidada')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS['model_comparison'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision_values, recall_values, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color='#0072B2', label=f"AUC={final_metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='Curva ROC')
    axes[1].plot(recall_values, precision_values, color='#D55E00', label=f"AP={final_metrics['pr_auc']:.3f}")
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Curva Precision-Recall')
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PATHS['roc_pr_curves'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    predictions = (probabilities >= selected_threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benigna', 'Maligna']); ax.set_yticklabels(['Benigna', 'Maligna'])
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    ax.set_title(f'Matriz de confusión, umbral={selected_threshold:.2f}')
    fig.tight_layout()
    fig.savefig(PATHS['confusion_matrix'], dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    # SHAP requiere un modelo arbóreo. Si el seleccionado no es XGBoost,
    # se ajusta la mejor configuración XGBoost exclusivamente con training
    # para explicabilidad comparativa, sin alterar la selección principal.
    xgb_grid.fit(X_train, y_train)
    explain_model = xgb_grid.best_estimator_
    local_index = int(np.argmax(explain_model.predict_proba(X_test)[:, 1]))
    shap_importance = explain_xgboost(explain_model, X_train, X_test, local_index)

    # Estabilidad SHAP en tres particiones reproducibles.
    stability_rows = []
    for seed in [41, 42, 43]:
        X_subtrain, X_validation, y_subtrain, _ = train_test_split(
            X_train, y_train, test_size=0.25, stratify=y_train, random_state=seed
        )
        clone_model = GridSearchCV(
            build_models()[2], grid, scoring='roc_auc', cv=inner_cv,
            n_jobs=1, refit=True, error_score='raise'
        )
        clone_model.fit(X_subtrain, y_subtrain)
        pipe = clone_model.best_estimator_
        engineered = pipe.named_steps['features'].transform(X_validation)
        scaled = pipe.named_steps['scaler'].transform(engineered)
        names = engineered.columns.tolist()
        contrib = pipe.named_steps['model'].get_booster().predict(
            xgb.DMatrix(scaled, feature_names=names), pred_contribs=True
        )[:, :-1]
        fold_importance = pd.DataFrame({'feature': names, 'importance': np.abs(contrib).mean(axis=0)})
        fold_importance['rank'] = fold_importance['importance'].rank(ascending=False, method='min')
        fold_importance['seed'] = seed
        stability_rows.append(fold_importance)
    stability = pd.concat(stability_rows, ignore_index=True)
    stability_summary = stability.groupby('feature').agg(
        mean_importance=('importance', 'mean'),
        std_importance=('importance', 'std'),
        mean_rank=('rank', 'mean'),
        top10_frequency=('rank', lambda values: float(np.mean(values <= 10))),
    ).reset_index().sort_values('mean_rank')
    stability_summary.to_csv(PATHS['shap_stability'], index=False)

    joblib.dump(final_model, PATHS['model'])
    loaded_model = joblib.load(PATHS['model'])
    if not np.allclose(final_model.predict_proba(X_test), loaded_model.predict_proba(X_test)):
        raise ValueError('El modelo recargado no reproduce las probabilidades.')

    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'scikit_learn': sklearn.__version__,
        'xgboost': xgb.__version__,
        'shap': shap.__version__,
        'joblib': joblib.__version__,
    }
    summary = {
        'problem': {
            'type': 'binary classification',
            'positive_class': 'malignant',
            'primary_metric': 'roc_auc',
            'false_negative_cost': FALSE_NEGATIVE_COST,
            'false_positive_cost': FALSE_POSITIVE_COST,
        },
        'selection': {
            'selected_model': selected_name,
            'selected_threshold': selected_threshold,
            'selection_rule': 'ROC-AUC CV within tolerance, then recall, stability and runtime',
            'selected_cv_roc_auc_mean': float(selected_row['outer_roc_auc_mean']),
            'selected_cv_roc_auc_std': float(selected_row['outer_roc_auc_std']),
        },
        'test_metrics': final_metrics,
        'xgboost_best_params_for_shap': xgb_grid.best_params_,
        'versions': versions,
        'files': {key: str(value.relative_to(PROJECT_ROOT)) for key, value in PATHS.items()},
    }
    with open(PATHS['summary_json'], 'w', encoding='utf-8') as file:
        json.dump(to_native(summary), file, ensure_ascii=False, indent=2, allow_nan=False)

    report = f"""# Informe final del proyecto\n\n## Formulación\n\nClasificación binaria de casos malignos y benignos. La clase positiva es **maligna**. La métrica principal es ROC-AUC y el costo supuesto es {FALSE_NEGATIVE_COST}:1 para falsos negativos frente a falsos positivos.\n\n## Selección metodológica\n\n- Modelo seleccionado: **{selected_name}**.\n- ROC-AUC en CV: **{selected_row['outer_roc_auc_mean']:.6f} ± {selected_row['outer_roc_auc_std']:.6f}**.\n- Umbral seleccionado con probabilidades OOF: **{selected_threshold:.2f}**.\n\n## Evaluación final\n\n- ROC-AUC: **{final_metrics['roc_auc']:.6f}**.\n- PR-AUC: **{final_metrics['pr_auc']:.6f}**.\n- Accuracy: **{final_metrics['accuracy']:.6f}**.\n- Precision: **{final_metrics['precision']:.6f}**.\n- Recall: **{final_metrics['recall']:.6f}**.\n- F1: **{final_metrics['f1']:.6f}**.\n- Falsos negativos: **{final_metrics['false_negatives']}**.\n- Falsos positivos: **{final_metrics['false_positives']}**.\n- Costo ponderado: **{final_metrics['weighted_error_cost']}**.\n\n## Interpretación\n\nLa selección se realizó exclusivamente con entrenamiento mediante validación cruzada anidada. El conjunto de prueba se utilizó una sola vez y no reabrió la selección. Las explicaciones SHAP describen el comportamiento del modelo XGBoost ajustado con entrenamiento y no demuestran causalidad.\n\n## Limitaciones\n\nEl dataset es educativo y de tamaño moderado. Antes de una aplicación real se requiere validación externa, revisión experta, calibración y evaluación de impacto.\n"""
    PATHS['final_report'].write_text(report, encoding='utf-8')

    manifest = {
        'run_started_utc_epoch': started,
        'run_duration_seconds': time.time() - started,
        'random_seed': RANDOM_SEED,
        'input_rows': len(X),
        'input_features': X.shape[1],
        'artifacts': [str(path.relative_to(PROJECT_ROOT)) for path in PATHS.values()],
    }
    PATHS['manifest_json'].write_text(json.dumps(to_native(manifest), indent=2), encoding='utf-8')

    checks = {
        'all_files_exist': all(path.is_file() and path.stat().st_size > 0 for key, path in PATHS.items() if key != 'checks_json'),
        'model_matches_report': selected_name in report,
        'threshold_matches_report': f'{selected_threshold:.2f}' in report,
        'roc_auc_matches_report': f"{final_metrics['roc_auc']:.6f}" in report,
        'pr_auc_matches_report': f"{final_metrics['pr_auc']:.6f}" in report,
        'false_negatives_match_report': f"{final_metrics['false_negatives']}" in report,
        'false_positives_match_report': f"{final_metrics['false_positives']}" in report,
        'reloaded_model_matches': True,
    }
    if not all(checks.values()):
        raise ValueError(f'Fallaron controles de consistencia: {checks}')
    PATHS['checks_json'].write_text(json.dumps(checks, indent=2), encoding='utf-8')

    # Última comprobación después de crear checks_json.
    for path in PATHS.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Artefacto faltante o vacío: {path}')
    return summary


if __name__ == '__main__':
    result = run_project()
    print(json.dumps(result, ensure_ascii=False, indent=2))
