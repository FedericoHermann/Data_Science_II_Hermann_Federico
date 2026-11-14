# Informe de Pipeline

## Formulación

Clasificación binaria. Variable objetivo: `malignant` (1 = maligna). Métrica principal: ROC-AUC. Costos definidos antes de prueba: FN=5, FP=1.

## Selección metodológica

- Modelo: **LogisticRegression**.
- ROC-AUC CV: **0.995666 ± 0.003872**.
- Umbral OOF: **0.31**.

## Prueba aislada

- ROC-AUC: **0.995370**.
- PR-AUC: **0.993151**.
- Accuracy: **0.964912**.
- Precision: **0.952381**.
- Recall: **0.952381**.
- F1: **0.952381**.
- FN: **2**.
- FP: **2**.
- Costo ponderado: **12**.

## Explicabilidad

SHAP se calculó sobre el XGBoost optimizado como modelo explicativo complementario. Si el modelo final seleccionado es otro, SHAP no se presenta como explicación del modelo final. Las contribuciones son predictivas, no causales.

## Limitaciones

Dataset educativo y moderado.
