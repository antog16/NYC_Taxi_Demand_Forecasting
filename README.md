# 🚕 Forecast de Demanda de Taxis en New York

## 🗽 Descripción General

Nueva York, una metrópolis vibrante con más de 8 millones de residentes, alberga uno de los sistemas de transporte más dinámicos del mundo. Millones de viajes en taxis amarillos y vehículos de alquiler se registran mensualmente, reflejando la intensa actividad de la ciudad. Sin embargo, esta demanda fluctúa significativamente según la hora del día, las condiciones climáticas, la agenda de eventos locales y la distribución geográfica. Esta variabilidad inherente presenta desafíos considerables, como la escasez de taxis en momentos pico, tiempos de espera prolongados para los usuarios y una distribución de la flota que a menudo resulta ineficiente.

Anticipar con precisión estos patrones de demanda se vuelve no solo crucial para optimizar la eficiencia operativa de las empresas de transporte, sino también para mejorar significativamente la experiencia del usuario, garantizando la disponibilidad de taxis cuando y donde se necesiten.

## ❓ Planteamiento del Problema

Las empresas de transporte en Nueva York se enfrentan a una volatilidad constante en la demanda de sus servicios. Dada una flota limitada de aproximadamente 8.000 taxis amarillos en toda la ciudad, la necesidad de distribuir sus recursos de manera inteligente es primordial. La incapacidad de predecir con exactitud los picos y valles de demanda, tanto en tiempo como en ubicación, no solo conduce a la pérdida de oportunidades de negocio y al aumento de costos operativos, sino que también impacta negativamente en la satisfacción del usuario al generar frustrantes tiempos de espera.

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es desarrollar un modelo robusto de Machine Learning capaz de predecir la demanda de taxis en la ciudad de Nueva York, especificando la ubicación y la hora con una ventana de anticipación de hasta 24 horas. Para lograr esto, se ha realizado un análisis exhaustivo de datos históricos de viajes de taxi, incorporando variables temporales cruciales y factores externos relevantes como las condiciones meteorológicas y la influencia de días festivos. Se espera que este modelo proporcione a las empresas de transporte una herramienta valiosa para la toma de decisiones estratégicas, optimizando la asignación de recursos y mejorando la calidad del servicio ofrecido a los ciudadanos y visitantes de Nueva York.

## 🛠️ Tecnologías Utilizadas

**Lenguaje:** Python

**Entorno de trabajo:** Jupyter Lab

**Principales librerías:**

* `pandas`: Para la manipulación y análisis de datos tabulares.
* `numpy`: Para operaciones numéricas eficientes.
* `geopandas`: Para el manejo y análisis de datos geoespaciales, crucial para trabajar con la ubicación de los viajes.
* `streamlit`: Para la creación y despliegue de la aplicación web interactiva.
* `urllib` y `glob`: Para la automatización de la descarga y gestión de archivos de datos.
* `pydeck` y `folium`: Para la visualización interactiva de datos geográficos en mapas.
* `seaborn` y `matplotlib`: Para la creación de visualizaciones estadísticas y exploratorias.
* `beautifulsoup4`: Para el web scraping de los enlaces de archivos parquet incluyendo el histórico de viajes de taxis.
* `optuna`: Para la optimización eficiente de los hiperparámetros del modelo de machine learning.
* `scikit-learn`: Para algoritmos de machine learning (K-means clustering), preprocesamiento de datos y evaluación de modelos.
* `xgboost`: Para la implementación del modelo de gradient boosting, conocido por su alto rendimiento en tareas de regresión.
* `lightgbm`: Otra implementación eficiente del algoritmo de gradient boosting, que a menudo ofrece un rendimiento competitivo y tiempos de entrenamiento más rápidos.
* `catboost`: Un algoritmo de gradient boosting que se destaca por su manejo robusto de variables categóricas y su menor necesidad de ajuste de hiperparámetros.
* `plotly`: Para la creación de visualizaciones interactivas y dashboards dentro de la aplicación Streamlit.

## 🗃️ Conjuntos de Datos

**Viajes en taxis amarillos (Enero 2022 - Enero 2025):**

* **Fuente:** NYC Taxi & Limousine Commission (TLC)
* **Descripción:** Este conjunto de datos contiene información detallada sobre cada viaje realizado por los taxis amarillos de Nueva York, incluyendo las ubicaciones (barrios) de inicio y fin, la hora de inicio y fin del viaje, la duración, la distancia recorrida y el número de pasajeros. La granularidad de los datos es a nivel de cada viaje individual.
* **Variables clave utilizadas:** `tpep_pickup_hour`, `PULocationID`.

**Datos históricos de clima en NYC:**

* **Fuente:** Open-Meteo
* **Descripción:** Esta API proporciona información horaria sobre las condiciones climáticas en la ciudad de Nueva York, incluyendo temperatura, precipitación, velocidad del viento y otros factores relevantes.
* **Variables clave utilizadas:** `temperature_2m`, `precipitation`, `wind_speed_10m`, `rain`, `snow`, `snow_depth`.

## 🔍 Proceso seguido

1.  **Descarga automatizada de datos de viajes:** Se utilizaron las librerías `BeautifulSoup`, `glob` y `urllib.request` para automatizar la descarga y organización de los archivos de datos históricos de viajes.
2.  **Descarga de datos climáticos con conexión a API:** Se descargan los datos climáticos correspondientes al periodo de viajes.
2.  **Limpieza y transformación de datos:** Se aplicaron técnicas de limpieza para manejar valores faltantes y asegurar la consistencia de los datos. Se realizaron transformaciones necesarias, como la conversión de formatos de fecha y hora y la agrupación de los viajes por fecha, hora y barrio.
3.  **Análisis exploratorio (EDA):** Se llevó a cabo un análisis exploratorio exhaustivo utilizando librerías como `seaborn` y `matplotlib` para comprender las distribuciones de las variables, identificar patrones de demanda a lo largo del tiempo y el espacio, y explorar las relaciones entre las diferentes variables.
4.  **Feature engineering:** Se crearon nuevas variables relevantes para el modelo a partir de los datos existentes. Esto incluyó la generación de variables horarias (hora del día, día de la semana, fin de semana), variables meteorológicas (temperatura, lluvia) y variables contextuales (indicadores de días festivos).
5.  **Entrenamiento del modelo:** Se seleccionó y entrenó el modelo `XGBoost Regressor` utilizando los datos preparados y las características generadas. Se empleó la librería `optuna` para la optimización de los hiperparámetros del modelo, buscando la configuración que ofreciera el mejor rendimiento.
6.  **Despliegue en app web:** Se desarrolló una aplicación web interactiva utilizando la librería `streamlit` para permitir a los usuarios visualizar las predicciones de demanda en diferentes ubicaciones y horas. Se utilizaron librerías como `pydeck` y `folium` para la visualización de los resultados en mapas.

## 🔗 Accede a la app en vivo

[https://taxidemandforecastnyc.streamlit.app/]

## 🤖 Modelo y Métricas

**Modelo principal:** XGBoost Regressor. Se eligió este modelo debido a su capacidad para manejar grandes conjuntos de datos, su robustez frente a la multicolinealidad y su alto rendimiento en tareas de regresión, especialmente cuando se trata de datos estructurados y series de tiempo con patrones complejos.

**Métricas de evaluación:**

* **RMSE (Root Mean Squared Error):** Esta métrica mide la raíz cuadrada del promedio de los errores al cuadrado entre las predicciones del modelo y los valores reales de demanda. Un valor más bajo de RMSE indica una mayor precisión del modelo en términos de la magnitud de los errores.
* **R² (Coeficiente de determinación):** Esta métrica representa la proporción de la varianza en la variable dependiente (demanda de taxis) que es predecible a partir de las variables independientes (características utilizadas en el modelo). Un valor de R² cercano a 1 indica que el modelo explica una gran parte de la variabilidad en la demanda.

## 📈 Principales hallazgos

* El modelo demuestra una **alta precisión en la predicción de la demanda de taxis para la siguiente hora por zona**, lo que permite una respuesta operativa casi en tiempo real.
* La capacidad de **planificar la distribución de vehículos con hasta 24 horas de anticipación** ofrece a las empresas la oportunidad de optimizar la asignación de recursos, anticipándose a los picos de demanda y reduciendo los tiempos de espera para los usuarios.
* Se identificó que las **variables históricas**, que capturan los patrones de demanda recurrentes en días y horas anteriores, son factores **esenciales para el rendimiento del modelo**, superando en importancia a otras variables como el clima en la mayoría de los casos.
* El **clima**, en general, **no demostró ser un factor determinante en la predicción de la demanda**, excepto en situaciones de condiciones meteorológicas extremas.
