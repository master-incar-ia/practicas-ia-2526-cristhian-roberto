# K-means Clustering Tutorial - Evaluation Questionnaire

## Student Information

- **Name**: Roberto del Horno y Cristhian Pinzón
- **Date**: 13/03/2026
- **Course**: Applications of AI for Industrial Control
- **Tutorial**: K-means Clustering for Industrial Applications

---

## Section 1: Theoretical Understanding (20 points)

### 1.1 Basic Concepts (20 points)

**Question 1** (4 points): What type of machine learning algorithm is K-means clustering?

- a) Supervised learning
- **b) Unsupervised learning**
- c) Semi-supervised learning
- d) Reinforcement learning

Answer: Es aprendizaje no supervisado debido a que se pasa un dataset sin elementos etiquetados para que descubra caracteristicas que asocien grupos (clusters) que describan a cierto tipo de elementos del dataset.

**Question 2** (4 points): What is the main objective of the K-means algorithm?

- a) To classify data points into predefined categories
- **b) To minimize the sum of squared distances from data points to cluster centroids**
- c) To maximize the distance between different clusters
- d) To predict future values based on historical data

Answer: El objetivo matemático del algoritmo K-means es minimizar la «suma de cuadrados intraclúster»  o inercia. Su objetivo es hacer que los clústeres sean lo más compactos posible minimizando la distancia entre cada punto y el centro (centroide) del clúster al que pertenece.

**Question 3** (4 points): In the iris dataset tutorial, how many features were used for clustering?

- a) 2 features
- b) 3 features
- **c) 4 features**
- d) 5 features

**Question 4** (4 points): Which of the following is NOT a step in the K-means algorithm?

- a) Initialize cluster centroids randomly
- b) Assign each data point to the nearest centroid
- c) Update centroids by calculating the mean of assigned points
- **d) Remove outliers from the dataset**

Answer: Esto no es un paso del algoritmo en sí. Aunque a menudo se recomienda eliminar los valores atípicos como paso de preprocesamiento antes de ejecutar el algoritmo K-means, el algoritmo K-means en sí mismo no identifica ni elimina los valores atípicos; simplemente intenta incluirlos en el clúster más cercano, lo que puede sesgar los resultados.

**Question 5** (4 points): What does the 'k' parameter represent in K-means clustering?

- a) The number of features in the dataset
- b) The number of iterations to run
- **c) The number of clusters to create**
- d) The number of data points in each cluster

Answer: la «K» se refiere precisamente al número de clusters distintos y que no se superponen en los que se desea que el algoritmo divida el conjunto de datos. Se trata de un hiperparámetro que el usuario debe especificar antes de ejecutar el algoritmo.

---

## Section 2: Data Preprocessing and Scaling (15 points)

### 2.1 Feature Engineering (15 points)

**Question 6** (5 points): Why is feature scaling (standardization) important for K-means clustering?

- a) It makes the algorithm run faster
- **b) It ensures all features contribute equally to distance calculations**
- c) It reduces the memory requirements
- d) It automatically determines the optimal number of clusters

Answer: Este método se basa en gran medida en el cálculo de distancias (normalmente la distancia euclidiana) entre los puntos de datos y los centroides de los clústeres. Si el conjunto de datos tiene una característica que se mide en miles (como el salario) y otra que se mide en decimales (como una proporción de 0 a 1), la característica con los números más grandes dominará por completo el cálculo de la distancia. Escalar o estandarizar las características las pone a todas en igualdad de condiciones, asegurando que el algoritmo considere cada característica de manera proporcional.

**Question 7** (5 points): In the tutorial, what was the effect of standardization on the iris dataset features?

- **a) Mean became 0 and standard deviation became 1**
- b) All values became positive
- c) The range of all features became [0,1]
- d) The features were sorted in ascending order

**Question 8** (5 points): Which iris features showed the strongest discriminative power for clustering?

- a) Sepal length and sepal width
- **b) Petal length and petal width**
- c) Sepal length and petal length
- d) All features contributed equally

---

## Section 3: Industrial Applications (25 points)

### 3.1 Real-World Applications (25 points)

**Question 9** (5 points): Which of the following is a valid industrial application of K-means clustering? (Select all that apply)

- a) Equipment health monitoring
- b) Product defect classification
- c) Customer segmentation
- d) Energy consumption pattern analysis
- **e) All of the above**

**Question 10** (5 points): In predictive maintenance applications, K-means clustering can be used to:

- a) Predict exact failure times
- **b) Group similar failure patterns**
- c) Identify normal vs abnormal operating conditions
- d) Both b and c

**Question 11** (5 points): When would you NOT recommend using K-means clustering?

- a) When clusters are expected to be spherical
- b) When you have continuous numerical features
- **c) When clusters have very different sizes**
- d) When you need fast processing

Answer: 

**Question 12** (5 points): In manufacturing quality control, K-means clustering could help with:

- a) Automatically setting quality thresholds
- b) Grouping products by quality characteristics
- c) Identifying defect patterns
- **d) All of the above**

**Question 13** (5 points): In the tutorial's 3D visualization, what was the purpose of using PCA?

- a) To reduce the dataset size
- b) To speed up the clustering algorithm
- **c) To visualize high-dimensional data in 3D space**
- d) To improve clustering accuracy

---

## Section 4: Practical Considerations (20 points)

### 4.1 Validation and Evaluation (20 points)

**Question 14** (10 points): Explain the difference between internal and external validation metrics for clustering. Give one example of each from the tutorial.

_Your answer:_

Las **métricas de validación interna** evaluan la calidad del cluster usando solo caracteristicas de los datos del propio Dataset (evaluando la cohesión y separación de los clusteres sin conocer los Labels reales). Un ejemplo usado en el tutorial es el **Silhouette Score**, el cual mide cuán similar es un elemento a su propio cluster comparado a otros clusteres, indicando cuán buen separados los clusteres están y cuán separados o estrechos los clusteres están empaquetados.

Las **métricas de validación externa** evaluan la calidad del cluster mediante la comparación de los agrupamientos creados por el algoritmo vs etiquetas de clase conocidas y verdaderas (verdad fundamental). Un ejemplo usado en el tutorial es el **Adjusted Rand Index (ARI)**, el cual se usa para medir desempeño de un cluster. Esta métrica mide cuán perfecto los agrupamientos generados por el algoritmo emparejan con los Labels o categorias reales.

---

**Question 15** (10 points): Describe a specific industrial scenario where you would apply K-means clustering. Include: (1) the type of data you would cluster, (2) what the clusters would represent, and (3) how the results would be used for decision-making.

_Your answer:_

Un escenario industrial donde aplicar K-means clustering puede ser en el mantenimiento predictivo de equipos de manufactura pesada (e.g. Maquinas de mecanizado CNC).

En grandes plantas de fabricación, las averías inesperadas de las máquinas provocan enormes pérdidas económicas debido a la interrupción de las líneas de producción. Para evitarlo, se instalan sensores en la maquinaria para supervisar su comportamiento físico, pero los datos brutos son demasiados para que el personal los analicen de forma continua. Es ahí donde se puede utilizar el K-means clustering para identificar automáticamente los distintos «estados de salud» de las máquinas.

**Tipos de datos para el Cluster:**

Se recolectarían datos de los sensores multi-dimensionales y continuos registrandose de manera regular (e.g. cada minuto) de la maquinaria. Los features (variables) que alimentan el algoritmo K-means incluirían:

* Amplitud de vibración (mm/s): Cuanto la maquina se está sacudiendo fisicamente.
* Frecuencia de vibración (Hz)
* Temperatura de operación (°C): Calor generada por fricción o por la carga en el motor.
* Intensidad de corriente en el motor (Amps): Energía eléctrica requerida para mantener la maquina funcionando.
* Emisiones Acústicas (dB): Perfiles de ruido que genera la maquina.

**Lo que los Clusteres representarian:**

Ya que K-means es un algoritmo no supervisado, agrupará de forma natural las lecturas de los sensores en distintos clústeres matemáticos basados en la similitud. Los clústeres representan distintos estados operativos o «perfiles de salud» de la maquinaria.

Si se ajustara k = 4, el algoritmo podria identificar:

* **Clúster 0** (estado «normal»): probablemente sea el clúster más grande y compacto. Representa las operaciones básicas: temperaturas estables, bajas vibraciones y consumo energético normal.
* **Clúster 1** (estado de «desalineación mecánica»): los puntos de datos mostrarían temperaturas normales, pero amplitudes de vibración inusualmente altas en frecuencias específicas, lo que indica que las piezas están ligeramente desalineadas.
* **Clúster 2** (estado de «problema de fricción/lubricación»): los datos mostrarían temperaturas elevadas, emisiones acústicas más altas (chirridos) y un consumo de corriente ligeramente superior, pero con vibraciones normales.
* **Clúster 3** (estado de «fallo inminente»): un clúster disperso y escaso que se caracteriza por picos extremos en todas las métricas: calor elevado, vibraciones intensas y subidas de tensión erráticas.

**¿Cómo los resultados se usarían para una toma de decisión?**

Los equipos de mantenimiento y operaciones de la fábrica utilizarían estos resultados de clustering para tomar decisiones críticas basadas en datos:

* **Programación dinámica del mantenimiento:** en lugar de cambiar los cojinetes de las máquinas cada seis meses (mantenimiento preventivo, lo cual es un desperdicio), el mantenimiento solo se activa cuando los puntos de datos en tiempo real de una máquina comienzan a desviarse del «Clúster 0» (Normal) al «Clúster 2» (Problema de fricción).
* **Asignación inteligente de recursos:** si el gerente de la planta ve que tres máquinas pertenecen al «Clúster 1» y una al «Clúster 3», sabrá exactamente cómo priorizar a su limitado equipo de técnicos: apagar y reparar inmediatamente la máquina del grupo 3 para evitar una avería catastrófica, y programar las máquinas del grupo 1 para realizar comprobaciones de alineación durante el siguiente cambio de turno.
* **Solución de problemas específicos:** dado que los clusters representan perfiles específicos, los técnicos saben qué buscar incluso antes de tocar la máquina. Si los datos de una máquina entran en el cluster de fricción, saben que deben llevar lubricante y revisar los cojinetes; si entran en el cluster de desalineación, llevan herramientas.

## Scoring Summary

| Section                           | Points Earned        | Total Points |
| --------------------------------- | -------------------- | ------------ |
| 1. Theoretical Understanding      | _____ / 20           | 20           |
| 2. Data Preprocessing and Scaling | _____ / 15           | 15           |
| 3. Industrial Applications        | _____ / 25           | 25           |
| 4. Practical Considerations       | _____ / 20           | 20           |
| **Total**                   | **_____ / 80** | **80** |

**Grade**: ____________

**Comments**:

---

---
