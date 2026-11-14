# Informe profesional del proyecto final de Data Science II

## Clasificación reproducible de casos malignos y benignos

**Autor:** Federico Hermann  
**Tipo de problema:** Clasificación binaria  
**Variable objetivo:** `malignant`, con `1 = maligna`  
**Métrica principal de selección:** ROC-AUC  
**Modelo seleccionado:** `LogisticRegression`  
**Umbral operativo seleccionado:** `0.31`

---

## 1. Introducción

Este proyecto desarrolla un flujo integral de Machine Learning orientado a clasificar observaciones como benignas o malignas a partir del dataset Breast Cancer Wisconsin incluido en scikit-learn. El trabajo no se limita al entrenamiento de un modelo: incorpora formulación del problema, análisis exploratorio, validación de datos, ingeniería de atributos, comparación de algoritmos, optimización de hiperparámetros, selección del umbral operativo, evaluación independiente, explicabilidad mediante SHAP, persistencia de modelos y controles automáticos de consistencia.

La clase positiva se definió como **maligna**, porque el error operativo más sensible es clasificar un caso maligno como benigno. Antes de evaluar el conjunto de prueba se estableció un costo relativo de **5 a 1** para falsos negativos frente a falsos positivos. Esta decisión permite complementar las métricas estadísticas con un criterio operativo explícito.

El dataset contiene **569 observaciones** y **30 atributos originales**. La partición reservó **455 registros para entrenamiento** y **114 para prueba**, manteniendo la prueba aislada durante la selección del modelo y del umbral.

---

## 2. Análisis funcional del pipeline

### 2.1 Carga, validación y trazabilidad de los datos

El pipeline carga los datos desde `sklearn.datasets.load_breast_cancer`, transforma la codificación de la variable objetivo para que `1` represente la clase maligna y ejecuta controles de integridad antes del modelado. Se comprueba que el dataset no esté vacío, que no existan columnas duplicadas, valores faltantes ni valores no finitos. También se genera un snapshot de los datos, un diccionario de variables y resúmenes de valores faltantes y posibles outliers mediante el criterio IQR.

La separación entre entrenamiento y prueba se realiza de forma estratificada y con semilla fija. El código verifica que las dos particiones no compartan índices, evitando una fuga directa de observaciones.

### 2.2 Análisis exploratorio

El EDA produce evidencia descriptiva y visual mediante:

- distribución de las clases;
- histogramas comparativos por clase;
- boxplots para observar dispersión y valores extremos;
- matriz de correlación;
- resumen estadístico de las variables;
- conteo y porcentaje de posibles outliers.

Este análisis permite identificar diferencias de escala, relaciones entre variables y señales potencialmente útiles para la clasificación. Los posibles outliers se documentan en lugar de eliminarse automáticamente, evitando descartar observaciones que podrían representar variabilidad real del problema.

### 2.3 Ingeniería de atributos

La ingeniería de atributos se encapsula en un transformador compatible con scikit-learn. Se crean tres relaciones:

- `area_perimeter_ratio`;
- `worst_to_mean_radius_ratio`;
- `concavity_compactness_ratio`.

Al encontrarse dentro de los pipelines, estas transformaciones se aplican de forma consistente durante entrenamiento, validación, prueba y predicción futura. Esto reduce el riesgo de diferencias entre el desarrollo y el uso posterior del modelo.

### 2.4 Modelos comparados

El proyecto compara tres procedimientos:

1. **Logistic Regression**, como alternativa lineal regularizada e interpretable.
2. **XGBoost baseline**, con una configuración fija y reproducible.
3. **XGBoost con GridSearchCV**, optimizado exclusivamente con el conjunto de entrenamiento.

La comparación utiliza validación cruzada anidada. La capa interna optimiza hiperparámetros y la capa externa estima el rendimiento del procedimiento completo, reduciendo el optimismo que se produciría al seleccionar y evaluar parámetros sobre las mismas particiones.

### 2.5 Selección del modelo

La regla de selección fue definida antes de consultar el conjunto de prueba:

1. maximizar el ROC-AUC medio de validación cruzada;
2. considerar equivalentes los modelos dentro de una tolerancia práctica de `0.002`;
3. desempatar mediante recall, estabilidad y tiempo de ejecución;
4. seleccionar el umbral con probabilidades out-of-fold del entrenamiento;
5. evaluar el conjunto de prueba una única vez, sin reabrir la selección.

### 2.6 Explicabilidad y persistencia

El modelo elegido mediante la regla metodológica es Logistic Regression. Para cumplir el análisis SHAP con un modelo de árboles, el proyecto conserva un XGBoost optimizado como **modelo explicativo complementario**. Esta separación está documentada para evitar atribuir explicaciones del XGBoost al modelo final seleccionado.

El pipeline final y el XGBoost explicativo se serializan con `joblib`. Después de recargar el modelo final, el proyecto verifica que las probabilidades coincidan con las originales.

---

## 3. Análisis de resultados del pipeline

### 3.1 Validación cruzada anidada

| model | outer_roc_auc_mean | outer_roc_auc_std | outer_pr_auc_mean | outer_recall_mean | outer_f1_mean | gap_roc_auc | elapsed_seconds |
|---|---|---|---|---|---|---|---|
| LogisticRegression | 0.995666 | 0.003872 | 0.994150 | 0.958824 | 0.958352 | 0.002773 | 0.323023 |
| XGBoostGridSearch | 0.994014 | 0.004957 | 0.992336 | 0.941176 | 0.955216 | 0.005986 | 22.257922 |
| XGBoostBaseline | 0.993395 | 0.005178 | 0.991376 | 0.947059 | 0.958281 | 0.006605 | 1.210809 |

Logistic Regression alcanzó el mayor ROC-AUC medio, con **0.995666 ± 0.003872**. La brecha de ROC-AUC entre entrenamiento y validación fue reducida, lo que aporta evidencia de una generalización estable dentro de las particiones evaluadas.

Los resultados de XGBoost también fueron elevados, pero sus métricas de entrenamiento se acercaron o llegaron al máximo, mientras que las métricas externas fueron inferiores. Esta diferencia es consistente con una mayor capacidad de ajuste y con un riesgo de sobreajuste superior al del modelo lineal.

### 3.2 Umbral operativo

El umbral seleccionado fue **0.31**. La selección se realizó con probabilidades out-of-fold del entrenamiento, buscando minimizar el costo ponderado y conservar un recall mínimo de `0.95`. El umbral no se eligió observando el conjunto de prueba.

### 3.3 Evaluación final sobre prueba

- **Accuracy:** 0.964912
- **Precision:** 0.952381
- **Recall:** 0.952381
- **F1-score:** 0.952381
- **ROC-AUC:** 0.995370
- **PR-AUC:** 0.993151
- **Verdaderos negativos:** 70
- **Falsos positivos:** 2
- **Falsos negativos:** 2
- **Verdaderos positivos:** 40
- **Costo ponderado observado:** 12

El modelo clasificó correctamente **40 de 42 casos malignos** y **70 de 72 casos benignos**. Se registraron **2 falsos negativos** y **2 falsos positivos**. Esto representa un equilibrio alto entre sensibilidad y precisión, aunque los falsos negativos restantes continúan siendo el error prioritario desde la perspectiva operativa definida.

### 3.4 Estabilidad de SHAP

| Variable | Importancia media | Desviación | Ranking medio | Frecuencia top 10 |
|---|---:|---:|---:|---:|
| worst_to_mean_radius_ratio | 1.266898 | 0.221058 | 1.20 | 1.00 |
| mean concave points | 0.940254 | 0.285700 | 3.60 | 1.00 |
| worst perimeter | 0.770242 | 0.234049 | 4.60 | 1.00 |
| worst texture | 0.793345 | 0.276527 | 4.80 | 1.00 |
| mean texture | 0.773083 | 0.279873 | 5.40 | 1.00 |
| concavity_compactness_ratio | 0.694668 | 0.148166 | 6.00 | 1.00 |
| area error | 0.577947 | 0.263279 | 7.00 | 0.80 |
| worst concave points | 0.592617 | 0.112855 | 7.60 | 0.80 |
| worst radius | 0.430881 | 0.257336 | 10.20 | 0.60 |
| concave points error | 0.370075 | 0.112377 | 11.20 | 0.20 |

El atributo derivado `worst_to_mean_radius_ratio` obtuvo el mejor ranking medio y apareció dentro del top 10 en todas las particiones analizadas. También mostraron estabilidad `mean concave points`, `worst perimeter`, `worst texture`, `mean texture` y `concavity_compactness_ratio`.

La presencia estable de dos atributos derivados entre las variables más relevantes sugiere que las relaciones construidas aportan señal al XGBoost explicativo. Sin embargo, esta importancia no demuestra causalidad ni implica que esos atributos sean indispensables para Logistic Regression, que fue el modelo seleccionado mediante validación cruzada.

---

## 4. Observaciones sobre funcionalidad y funcionamiento

### 4.1 Reproducibilidad

El proyecto fija semillas, registra versiones, conserva dependencias, serializa los modelos y genera un manifiesto de ejecución. La lógica principal reside en `src/project_pipeline.py`, mientras que el notebook ejecuta ese archivo en lugar de mantener una segunda implementación independiente.

### 4.2 Coherencia entre código e informe

El informe, las tablas, los gráficos y el resumen JSON se generan desde los mismos objetos finales. Los controles automáticos comprobaron:

- **all_required_files_exist:** Superado
- **selected_model_matches_report:** Superado
- **threshold_matches_report:** Superado
- **roc_auc_matches_report:** Superado
- **pr_auc_matches_report:** Superado
- **false_negatives_match_report:** Superado
- **false_positives_match_report:** Superado
- **model_reload_matches:** Superado
- **test_isolation_preserved:** Superado
- **threshold_in_valid_range:** Superado

Esta arquitectura reduce el riesgo de que el texto reporte métricas diferentes de las tablas o de que el umbral documentado no coincida con el utilizado para producir la matriz de confusión.

### 4.3 Utilidad de los resultados

El resultado más útil para un cliente no es únicamente el ROC-AUC elevado. El proyecto entrega un mecanismo de decisión completo:

- una regla previa para seleccionar el modelo;
- un umbral alineado con el costo de los errores;
- una matriz de confusión que traduce probabilidades en decisiones;
- un costo ponderado que permite comparar escenarios operativos;
- explicaciones globales y locales para inspeccionar el comportamiento del modelo arbóreo;
- artefactos reproducibles para auditoría y mantenimiento.

### 4.4 Limitaciones funcionales

El pipeline es funcional como demostración profesional y reproducible, pero no está preparado para uso clínico real. El dataset es educativo, no existe validación externa y la relación de costos `5:1` es un supuesto analítico. Antes de cualquier implementación real se necesitarían datos independientes, calibración de probabilidades, definición del costo con especialistas, evaluación por subgrupos, monitoreo de deriva y revisión del impacto de las decisiones.

---

## 5. Insights para la interpretación y utilidad para el cliente

### Insight 1: un modelo más complejo no fue necesario

La selección de Logistic Regression muestra que la complejidad adicional de XGBoost no produjo una mejora suficiente bajo la regla establecida. Para un cliente, esto favorece una solución más rápida, más simple de mantener y con menor riesgo de sobreajuste.

### Insight 2: la métrica principal debe complementarse con errores concretos

Un ROC-AUC cercano a 1 indica una discriminación muy alta, pero no informa por sí solo cuántos casos malignos se perderán con un umbral determinado. La matriz de confusión y el recall permiten convertir la evaluación técnica en una lectura operativa.

### Insight 3: el umbral es una decisión de negocio

El umbral `0.31` refleja la prioridad otorgada a los falsos negativos. Si el cliente modificara el costo relativo de errores o la capacidad de revisión manual, el umbral podría recalcularse sin cambiar necesariamente el modelo.

### Insight 4: los falsos negativos restantes requieren un proceso de respaldo

Incluso con un recall de **95.24%**, permanecen **2 casos malignos no detectados** en la prueba. En un escenario real, el modelo debería funcionar como apoyo y no como único decisor. Una revisión humana o una segunda regla de control sería recomendable para casos cercanos al umbral.

### Insight 5: SHAP sirve para auditoría, no para causalidad

Las explicaciones SHAP ayudan a identificar qué variables utiliza el XGBoost para modificar sus predicciones y permiten revisar observaciones concretas. Sin embargo, no prueban que una característica cause malignidad. La utilidad para el cliente se encuentra en la trazabilidad de la decisión y la detección de comportamientos inesperados.

### Insight 6: la estabilidad de variables aporta confianza adicional

Las variables que mantienen una posición alta en diferentes particiones son candidatas más confiables para monitoreo y revisión. Las variables cuya importancia cambia de forma considerable deberían interpretarse con mayor prudencia.

---

## 6. Conclusión final

El proyecto cumple el objetivo de construir una solución de clasificación binaria completa, reproducible y auditable. Logistic Regression fue seleccionada por su ROC-AUC medio, su estabilidad y su rendimiento operativo dentro de la tolerancia predefinida. El umbral se eligió sin consultar la prueba y la evaluación final mostró una discriminación elevada, con **2 falsos negativos** y **2 falsos positivos**.

Desde la perspectiva funcional, la solución ofrece más que una predicción: proporciona trazabilidad desde los datos hasta la recomendación, controles de consistencia, persistencia, auditoría, documentación y explicaciones complementarias. Esto facilita que un cliente comprenda por qué se eligió el modelo, qué errores comete, cómo fue definido el umbral y qué condiciones deberían cumplirse antes de considerar un uso real.

La conclusión principal es que el modelo resulta adecuado como demostración técnica y como base para un sistema de apoyo a decisiones, pero no debe presentarse como herramienta clínica terminada. La siguiente etapa profesional requeriría validación externa, calibración, definición formal de costos con stakeholders, evaluación de equidad y monitoreo continuo.

---

## 7. Evidencia técnica y rutas del repositorio

- Código canónico: `src/project_pipeline.py`
- Notebook reproducible: `notebooks/proyecto_final_ml.ipynb`
- Resumen final: `artifacts/final_summary.json`
- Controles: `artifacts/consistency_checks.json`
- Tabla de selección: `reports/model_selection_table.csv`
- Resultados de validación: `reports/nested_cv_results.csv`
- Resultados de prueba: `reports/test_results.csv`
- Análisis de umbral: `reports/threshold_analysis.csv`
- Estabilidad SHAP: `reports/shap_stability.csv`
- Modelo final: `models/modelo_final.joblib`
- Modelo explicativo: `models/xgboost_explicativo.joblib`

## 8. Entorno validado

- Python: `3.12.16`
- NumPy: `1.26.4`
- pandas: `2.2.2`
- matplotlib: `3.8.4`
- scikit-learn: `1.5.1`
- XGBoost: `3.1.2`
- SHAP: `0.49.1`
- joblib: `1.6.0`
