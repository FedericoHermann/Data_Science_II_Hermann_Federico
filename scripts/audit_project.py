import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


errors = []


# ------------------------------------------------------------
# 1. ARCHIVOS OBLIGATORIOS
# ------------------------------------------------------------

required_files = [
    "README.md",
    "requirements.txt",
    "notebooks/proyecto_final_ml.ipynb",
    "src/project_pipeline.py",

    "artifacts/final_summary.json",
    "artifacts/consistency_checks.json",

    "reports/final_report.md",
    "reports/model_selection_table.csv",
    "reports/nested_cv_results.csv",
    "reports/test_results.csv",
    "reports/threshold_analysis.csv",
    "reports/shap_stability.csv",

    "reports/clinical_descriptive_associations.csv",
    "reports/morphological_group_analysis.csv",
    "reports/high_feature_correlations.csv",
    "reports/bootstrap_metric_intervals.csv",
    "reports/priority_case_review.csv",

    "figures/clinical_descriptive_associations.png",

    "models/modelo_final.joblib"
]


for relative_path in required_files:

    file_path = (
        PROJECT_ROOT
        / relative_path
    )

    if not file_path.is_file():

        errors.append(
            "Archivo ausente: "
            + relative_path
        )

    elif file_path.stat().st_size == 0:

        errors.append(
            "Archivo vacío: "
            + relative_path
        )


# ------------------------------------------------------------
# 2. SINTAXIS DEL PIPELINE
# ------------------------------------------------------------

pipeline_path = (
    PROJECT_ROOT
    / "src"
    / "project_pipeline.py"
)


if pipeline_path.is_file():

    try:

        ast.parse(
            pipeline_path.read_text(
                encoding="utf-8"
            )
        )

    except SyntaxError as syntax_error:

        errors.append(
            "Error de sintaxis en el pipeline: "
            + str(
                syntax_error
            )
        )


# ------------------------------------------------------------
# 3. FORMATO DEL NOTEBOOK
# ------------------------------------------------------------

notebook_path = (
    PROJECT_ROOT
    / "notebooks"
    / "proyecto_final_ml.ipynb"
)


if notebook_path.is_file():

    try:

        notebook_content = json.loads(
            notebook_path.read_text(
                encoding="utf-8"
            )
        )

        if notebook_content.get(
            "nbformat"
        ) != 4:

            errors.append(
                "El notebook no utiliza nbformat 4."
            )

    except json.JSONDecodeError as json_error:

        errors.append(
            "El notebook no contiene JSON válido: "
            + str(
                json_error
            )
        )


# ------------------------------------------------------------
# 4. CONTROLES DEL PIPELINE
# ------------------------------------------------------------

checks_path = (
    PROJECT_ROOT
    / "artifacts"
    / "consistency_checks.json"
)


if checks_path.is_file():

    try:

        consistency_checks = json.loads(
            checks_path.read_text(
                encoding="utf-8"
            )
        )

        if not consistency_checks:

            errors.append(
                "El archivo de controles está vacío."
            )

        failed_checks = [
            check_name
            for check_name, check_result
            in consistency_checks.items()
            if not check_result
        ]

        if failed_checks:

            errors.append(
                "Controles fallidos: "
                + ", ".join(
                    failed_checks
                )
            )

    except json.JSONDecodeError as json_error:

        errors.append(
            "El archivo de controles no es válido: "
            + str(
                json_error
            )
        )


# ------------------------------------------------------------
# 5. COHERENCIA ENTRE RESUMEN Y PRUEBA
# ------------------------------------------------------------

summary_path = (
    PROJECT_ROOT
    / "artifacts"
    / "final_summary.json"
)


test_results_path = (
    PROJECT_ROOT
    / "reports"
    / "test_results.csv"
)


selection_table_path = (
    PROJECT_ROOT
    / "reports"
    / "model_selection_table.csv"
)


if (
    summary_path.is_file()
    and test_results_path.is_file()
    and selection_table_path.is_file()
):

    try:

        final_summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        test_results = pd.read_csv(
            test_results_path
        ).iloc[0]

        selection_table = pd.read_csv(
            selection_table_path
        )

        summarized_model = final_summary[
            "selection"
        ][
            "selected_model"
        ]

        summarized_threshold = float(
            final_summary[
                "selection"
            ][
                "selected_threshold"
            ]
        )

        summarized_roc_auc = float(
            final_summary[
                "test_metrics"
            ][
                "roc_auc"
            ]
        )

        summarized_pr_auc = float(
            final_summary[
                "test_metrics"
            ][
                "pr_auc"
            ]
        )

        if (
            summarized_model
            != test_results["model"]
        ):

            errors.append(
                "El modelo seleccionado no coincide "
                "entre final_summary.json y "
                "test_results.csv."
            )

        if not np.isclose(
            summarized_threshold,
            float(
                test_results[
                    "threshold"
                ]
            )
        ):

            errors.append(
                "El umbral no coincide entre "
                "final_summary.json y "
                "test_results.csv."
            )

        if not np.isclose(
            summarized_roc_auc,
            float(
                test_results[
                    "roc_auc"
                ]
            )
        ):

            errors.append(
                "ROC-AUC no coincide entre "
                "final_summary.json y "
                "test_results.csv."
            )

        if not np.isclose(
            summarized_pr_auc,
            float(
                test_results[
                    "pr_auc"
                ]
            )
        ):

            errors.append(
                "PR-AUC no coincide entre "
                "final_summary.json y "
                "test_results.csv."
            )

        if (
            selection_table[
                "selected_method"
            ].nunique()
            != 1
        ):

            errors.append(
                "La tabla de selección contiene "
                "más de un método seleccionado."
            )

        if (
            selection_table[
                "selected_method"
            ].iloc[0]
            != summarized_model
        ):

            errors.append(
                "La tabla de selección no coincide "
                "con el modelo final."
            )

    except Exception as audit_error:

        errors.append(
            "No fue posible auditar los resultados: "
            + str(
                audit_error
            )
        )


# ------------------------------------------------------------
# 6. RESULTADO DE LA AUDITORÍA
# ------------------------------------------------------------

if errors:

    print(
        "AUDIT FAILED"
    )

    for error_message in errors:

        print(
            "-",
            error_message
        )

    sys.exit(
        1
    )


print(
    "AUDIT PASSED"
)


print(
    "Modelo:",
    final_summary[
        "selection"
    ][
        "selected_model"
    ]
)


print(
    "Umbral:",
    f"{final_summary['selection']['selected_threshold']:.2f}"
)


print(
    "ROC-AUC de prueba:",
    f"{final_summary['test_metrics']['roc_auc']:.6f}"
)


print(
    "PR-AUC de prueba:",
    f"{final_summary['test_metrics']['pr_auc']:.6f}"
)
