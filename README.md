# Proyecto Final Data Science II

## Clasificación reproducible de casos malignos y benignos

Proyecto final de Machine Learning desarrollado por Federico Hermann. El repositorio implementa un flujo profesional y auditable sobre el dataset Breast Cancer Wisconsin incluido en scikit-learn.

## Objetivo

Construir un clasificador binario que identifique casos malignos. La clase positiva es `malignant = 1`. La métrica principal es ROC-AUC y la selección operativa también considera recall, falsos negativos y un costo supuesto de 5:1 para falsos negativos frente a falsos positivos.

## Características del proyecto

- EDA descriptivo y visual.
- Ingeniería de atributos reproducible.
- Pipelines de scikit-learn.
- Benchmark con regresión logística y XGBoost.
- Optimización con GridSearchCV.
- Validación cruzada anidada.
- Selección del umbral con probabilidades out-of-fold.
- Curvas ROC y Precision-Recall.
- Explicaciones SHAP globales, direccionales y locales.
- Estabilidad SHAP en múltiples particiones.
- Persistencia y validación del modelo.
- Informe final generado automáticamente.
- Controles de consistencia entre tablas, métricas e informe.

## Estructura

```text
notebooks/   Notebook principal reproducible
src/         Código fuente canónico
models/      Modelo final serializado
reports/     Resultados tabulares e informe final
figures/     Visualizaciones del EDA, evaluación y SHAP
artifacts/   Resumen JSON, manifiesto y controles
```

## Reproducción local

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python src/project_pipeline.py
```

## Google Colab

Abrir `notebooks/proyecto_final_ml.ipynb`, ejecutar la celda de instalación y luego **Runtime > Run all**.

## Dataset

No se distribuye un archivo de datos externo. El código carga `load_breast_cancer(as_frame=True)` desde scikit-learn. Consultar `data/README.md`.

## Criterio de selección

1. Maximizar ROC-AUC medio mediante validación cruzada anidada.
2. Considerar equivalentes los métodos dentro de una tolerancia de 0.002.
3. Desempatar por recall, estabilidad y tiempo de ejecución.
4. Elegir el umbral con predicciones out-of-fold y costos definidos antes de evaluar prueba.
5. Usar el conjunto de prueba una sola vez, sin reabrir la selección.

## Limitaciones

Este proyecto es educativo. No constituye una herramienta clínica ni debe utilizarse para diagnóstico. Una aplicación real requeriría validación externa, revisión experta, calibración, monitoreo y evaluación de impacto.

## Autor

Federico Hermann
