# Informe final del proyecto

## Formulación

Clasificación binaria de casos malignos y benignos. La clase positiva es **maligna**. La métrica principal es ROC-AUC y el costo supuesto es 5:1 para falsos negativos frente a falsos positivos.

## Selección metodológica

- Modelo seleccionado: **LogisticRegression**.
- ROC-AUC en CV: **0.995831 ± 0.003485**.
- Umbral seleccionado con probabilidades OOF: **0.43**.

## Evaluación final

- ROC-AUC: **0.995370**.
- PR-AUC: **0.993151**.
- Accuracy: **0.973684**.
- Precision: **0.975610**.
- Recall: **0.952381**.
- F1: **0.963855**.
- Falsos negativos: **2**.
- Falsos positivos: **1**.
- Costo ponderado: **11**.

## Interpretación

La selección se realizó exclusivamente con entrenamiento mediante validación cruzada anidada. El conjunto de prueba se utilizó una sola vez y no reabrió la selección. Las explicaciones SHAP describen el comportamiento del modelo XGBoost ajustado con entrenamiento y no demuestran causalidad.

## Limitaciones

El dataset es educativo y de tamaño moderado. Antes de una aplicación real se requiere validación externa, revisión experta, calibración y evaluación de impacto.
