# Model Card

## Uso previsto
Proyecto educativo de clasificación binaria reproducible.

## Uso no previsto
No utilizar para diagnóstico ni decisiones clínicas.

## Métricas
ROC-AUC principal. PR-AUC, recall, precision, F1 y costo ponderado como apoyo.

## Riesgos
Dataset moderado, ausencia de validación externa y posible falta de representatividad.

## Explicabilidad
SHAP explica el XGBoost optimizado complementario. No implica causalidad y no debe atribuirse a otro modelo seleccionado.
