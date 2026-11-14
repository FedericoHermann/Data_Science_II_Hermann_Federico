# Proyecto Final Data Science II

[![CI](https://github.com/FedericoHermann/Data_Science_II_Hermann_Federico/actions/workflows/ci.yml/badge.svg)](https://github.com/FedericoHermann/Data_Science_II_Hermann_Federico/actions/workflows/ci.yml)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedericoHermann/Data_Science_II_Hermann_Federico/blob/main/notebooks/proyecto_final_ml.ipynb)

Proyecto profesional y reproducible de clasificación binaria sobre Breast Cancer Wisconsin, con EDA, ingeniería de atributos, pipelines, optimización, validación cruzada anidada, umbral operativo, SHAP, persistencia y controles de consistencia.

## Objetivo

Predecir la clase `malignant` (`1 = maligna`) priorizando ROC-AUC y explicitando el costo relativo de falsos negativos y falsos positivos.

## Reproducción

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile src\project_pipeline.py
python src\project_pipeline.py
python scripts\audit_project.py
pytest -q
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m py_compile src/project_pipeline.py
python src/project_pipeline.py
python scripts/audit_project.py
pytest -q
```

## Fuente única de verdad

- `artifacts/final_summary.json`
- `reports/final_report.md`
- `reports/model_selection_table.csv`
- `artifacts/consistency_checks.json`

No editar manualmente resultados generados.

## Estructura

- `notebooks/`: notebook final para Colab.
- `src/`: pipeline canónico.
- `data/`: snapshot y diccionario.
- `reports/`: tablas e informe.
- `figures/`: EDA, evaluación y SHAP.
- `models/`: modelos serializados.
- `artifacts/`: resumen, manifiesto y checks.
- `tests/`: pruebas automáticas.
- `.github/workflows/`: integración continua.

## Nota ética

Proyecto educativo. No constituye una herramienta clínica ni un sistema de diagnóstico.

## Autor

Federico Hermann
