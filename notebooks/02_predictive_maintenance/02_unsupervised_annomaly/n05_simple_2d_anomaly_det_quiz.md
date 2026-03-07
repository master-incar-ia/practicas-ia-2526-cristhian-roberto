# Simple 2D Anomaly Detection Tutorial - Evaluation Questionnaire

## Student Information
- **Name**: Roberto del Horno y Cristhian Pinzón
- **Date**: 06/03/2026
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: Simple 2D Gaussian Anomaly Detection

---

## Section 1: Theoretical Understanding (25 points)

### 1.1 Gaussian Model Fundamentals (15 points)

**Question 1** (5 points): What are the main parameters that define a Gaussian multivariate distribution for 2D data?

- c) Mean vector and covariance matrix


**Question 2** (5 points): In the tutorial dataset, what were the three types of anomalies generated?

- b) High temperature, High pressure, Low temperature-pressure


**Question 3** (5 points): True/False: The Gaussian anomaly detector requires labeled anomaly data during training.

Answer: Falso

Explanation: El detector de anomalías gaussiano normalmente se entrena con datos normales, sin necesidad de contar con anomalías etiquetadas durante el entrenamiento.

### 1.2 Anomaly Scoring (10 points)

**Question 4** (5 points): What does a higher anomaly score indicate?

- b) Lower probability of being normal


**Question 5** (5 points): Fill in the blank: The anomaly score is calculated as the __________ of the probability density, which means lower probability results in __________ anomaly score.

Answer: negative logarithm and Higher

---

## Section 2: Visualization and Interpretation (25 points)

### 2.1 2D Visualization (25 points)

**Question 6** (10 points): Explain what probability contours represent in the 2D visualization and how they help in understanding the anomaly detection model.

_Your answer:_
Los contornos de probabilidad representan líneas de igual densidad de probabilidad en el espacio 2D. Muestran qué regiones del plano son más probables según el modelo gaussiano aprendido a partir de los datos normales. Cerca del centro, donde está la media, la probabilidad suele ser mayor, y al alejarse disminuye. Estos contornos ayudan a entender visualmente qué zonas el modelo considera normales y cuáles podrían clasificarse como anómalas si quedan fuera del límite o umbral definido.

**Question 7** (8 points): In the tutorial, what happened to the decision boundary when the threshold percentile was increased from 85% to 95%?

- a) The boundary became more restrictive (smaller normal region)


**Question 8** (7 points): Describe the difference between True Positives, False Positives, True Negatives, and False Negatives in the context of anomaly detection.

- True Positives: Anomalías reales que el modelo detecta correctamente como anomalías.
- False Positives: Datos normales que el modelo clasifica incorrectamente como anomalías.
- True Negatives: Datos normales que el modelo clasifica correctamente como normales.
- False Negatives: Anomalías reales que el modelo no detecta y clasifica como normales.

---

## Section 3: Performance Metrics (25 points)

### 3.1 Metrics Calculation (25 points)

**Question 9** (8 points): What does the F1-score represent in anomaly detection?

- b) The harmonic mean of precision and recall


**Question 10** (9 points): Given the following confusion matrix for an anomaly detection model:

```
              Predicted
              Normal  Anomaly
Actual Normal   450      50
       Anomaly   15      35
```

Calculate:
- Precision: 0.412
- Recall: 0.7
- F1-Score: 0.519

Show your calculations:
Precisión = TP / (TP + FP) = 35 / (35 + 50) = 35 / 85 = 0.412

Recall = TP / (TP + FN) = 35 / (35 + 15) = 35 / 50 = 0.7

F1 = 2 × (Precisión × Recall) / (Precisión + Recall) = 2 × (0.412 × 0.700) / (0.412 + 0.700)=           = 0.577 / 1.112 ≈ 0.519

**Question 11** (8 points): What is the trade-off when selecting threshold percentiles?

- b) High percentile: Fewer false positives, more false negatives


---

## Section 4: Industrial Applications (15 points)

### 4.1 Real-World Applications (15 points)

**Question 12** (8 points): List three real-world industrial applications where 2D Gaussian anomaly detection could be effectively used. For each application, specify the two variables that would be monitored.

Applications:
1. Monitoreo de calderas industriales: temperatura y presión.
2. Supervisión de motores eléctricos: temperatura del motor y vibración.
3. Control de sistemas hidráulicos: presión del fluido y caudal.

**Question 13** (7 points): Which of the following is NOT a limitation of the Gaussian multivariate method?

- c) Requires large amounts of labeled training data


---

## Section 5: Critical Thinking (10 points)

### 5.1 Implementation Strategy (10 points)

**Question 14** (10 points): You are implementing anomaly detection for a manufacturing process that monitors motor temperature and vibration. Based on the tutorial concepts:

1. How would you collect and prepare training data?
2. What threshold selection strategy would you use and why?
3. How would you handle the trade-off between false alarms and missed anomalies?
4. What additional considerations would you have for a real industrial deployment?

_Your answer:_
_______________________________________________________________________________
En primer lugar, recogería datos históricos del proceso en condiciones normales de funcionamiento, midiendo temperatura del motor y vibración durante distintos turnos, cargas y condiciones operativas. Después limpiaría los datos, eliminaría errores de sensor, trataría valores faltantes y comprobaría si las variables siguen aproximadamente una distribución gaussiana.

Para el umbral, empezaría con un umbral basado en percentiles, por ejemplo 90% o 95%, y lo ajustaría usando datos de validación o eventos conocidos. Esto permite controlar qué tan estricto será el detector frente a desviaciones del comportamiento normal.

El equilibrio entre falsas alarmas y anomalías no detectadas dependería del coste de cada error. Si un fallo del motor puede ser grave, preferiría un umbral más sensible, aceptando más falsas alarmas. Si las paradas por falsas alarmas son muy costosas, usaría un umbral más conservador.

Por ultimo, en un despliegue industrial real también consideraría la recalibración periódica del modelo, el envejecimiento de sensores, cambios en el proceso, mantenimiento preventivo, integración con sistemas SCADA o alarmas en tiempo real, y la validación continua del rendimiento del modelo para evitar degradación con el tiempo.
---

## Scoring Summary

| Section | Points Earned | Total Points |
|---------|---------------|--------------|
| 1. Theoretical Understanding | _____ / 25 | 25 |
| 2. Visualization and Interpretation | _____ / 25 | 25 |
| 3. Performance Metrics | _____ / 25 | 25 |
| 4. Industrial Applications | _____ / 15 | 15 |
| 5. Critical Thinking | _____ / 10 | 10 |
| **Total** | **_____ / 100** | **100** |

**Grade**: ____________

**Comments**:
_______________________________________________________________________________
_______________________________________________________________________________
