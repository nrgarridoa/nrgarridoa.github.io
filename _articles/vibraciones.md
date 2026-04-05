---
layout: article
title: "Predicción de vibraciones con Scikit-learn"
subtitle: "Tutorial reproducible con datos sintéticos y código."
date: 2025-10-31

# Hero + card
cover: "/assets/articles/vibraciones/cover.jpg"
hero: "/assets/articles/vibraciones/hero.jpg"
card: /assets/articles/vibraciones/card.jpg

# Metadatos / filtros
series: "Vibraciones"
tags: [Python, Blasting, PPV, USBM]
reading_time: "25 min"

# Resumen destacado
summary: >
  <p>En este artículo ajustamos un modelo empírico de vibraciones para voladuras. Compararemos la <strong>PPV medida</strong> con la <strong>PPV estimada</strong> usando la relación clásica con la <em>distancia escalada</em> (SD). Esto permite prever niveles vibratorios, evaluar cumplimiento y apoyar el diseño de disparos.</p>
  <p></p>
  <div class="ppv-note">Modelo base: <strong>PPV = k · SD<sup>b</sup></strong> &nbsp;⇔&nbsp; <code>log(PPV) = log(k) + b · log(SD)</code></div>

# Índice de contenidos
contents:
  - { anchor: "#intr", title: "Introducción al análisis de la PPV" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos y parámetros de entrada" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#eda", title: "Análisis Exploratorio (EDA)" }
  - { anchor: "#ajuste", title: "Ajuste del modelo USBM" }
  - { anchor: "#validacion", title: "Validación estadística" }
  - { anchor: "#prediccion", title: "Predicción operativa" }
  - { anchor: "#conclusiones", title: "Conclusiones operativas" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Variable objetivo", body: "PPV (mm/s) a 10–400 m del frente." }
  - { title: "Modelo", body: <div class="ppv-card__value">PPV = k · SD<sup>b</sup></div> }
  - { title: "Stack", body: "Python · Pandas · Matplotlib · Scikit-learn · SciPy · Jupyter Lab" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/article-vibraciones-ppv/blob/main/notebooks/vibraciones-ppv.ipynb" }
  - { type: "data",     label: "Dataset CSV",      url: "https://github.com/nrgarridoa/article-vibraciones-ppv/raw/main/data/raw/blasting_vibration_data.csv" }
  - { type: "repo",     label: "Repositorio",      url: "https://github.com/nrgarridoa/article-vibraciones-ppv" }

# Comentarios
comments: true
giscus:
  repo: "nrgarridoa/nrgarridoa.github.io"
  repo_id: ""
  category: "General"
  category_id: ""
  mapping: "pathname"
  reactions: "1"
  theme: "dark"
---

<a id="intr" class="anchor-clean"></a>
## 1) Introducción al análisis de la PPV

### 1.1) Importancia de las vibraciones en minería superficial

Cada vez que se ejecuta una voladura, la detonación genera una liberación súbita de energía. Parte de esa energía se utiliza para fragmentar la roca (según el diseño de perforación y carga), pero otra parte se propaga en forma de ondas sísmicas a través del macizo rocoso y del suelo circundante. Esas ondas inducen vibraciones que se transmiten a estructuras cercanas (taludes, bermas, equipos, edificaciones, zonas pobladas).

En la industria se monitorea estas vibraciones típicamente mediante geófonos o sismógrafos instalados a distintas distancias del punto de voladura. El parámetro operativo estándar es la **Velocidad Pico Partícula (PPV, Peak Particle Velocity)**, medida usualmente en mm/s. Un valor alto de PPV implica una vibración más intensa del suelo en ese punto de medición.

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} Control operacional</div>
  La PPV se usa para responder preguntas como:
  <ul>
    <li>¿Cuál es la vibración máxima que llegará a una casa o instalación a 200 m de la voladura?</li>
    <li>¿Existe riesgo de agrietamiento estructural o inestabilidad física por fatiga vibracional acumulada?</li>
    <li>¿Estamos cumpliendo los límites regulatorios / contractuales de vibración?</li>
  </ul>
</div>

Por ello, se vuelve crítico contar con un modelo que **prediga la PPV antes de disparar la voladura**. Esto habilita a ingeniería de perforación y voladura a ajustar variables de diseño:

- Carga máxima por retardo
- Número de taladros que detonan simultáneamente
- Secuencias de retardo
- Distancia a receptores sensibles

### 1.2) Aplicación de analítica avanzada

Desde el punto de vista de analítica, lo que buscamos es aprender una relación cuantitativa entre variables controlables / medibles (distancia al punto de monitoreo, carga explosiva por retardo) y la respuesta observada (PPV). Esa relación se usa luego para pronosticar PPV bajo nuevos escenarios.

El **modelo USBM** es el estándar clásico de este tipo de predicción y se basa en **ajustar ecuaciones empíricas** obtenidas mediante **regresión lineal** en escala logarítmica a datos medidos en campo a distintas distancias y con distintas cargas explosivas. Este enfoque tiene su origen en estudios sistemáticos de vibración de minas a cielo abierto y demoliciones controladas realizados por la U.S. Bureau of Mines (USBM) a partir de la década de 1960 y publicados extensamente alrededor de 1980.

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Metas pedagógicas y operativas para comprender, calibrar y aplicar un modelo de vibraciones por voladura con Python y la interpretación operativa en campo.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Explicar el fundamento físico-minero del problema de vibraciones y la importancia de la PPV como indicador clave en minería superficial.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Describir el modelo USBM y los parámetros K y β asociados al método de Distancia Escalada.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Construir un dataset sintético con valores realistas basados en literatura técnica de vibraciones de voladura.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Ejecutar un flujo reproducible en Python: EDA, transformación log-log, regresión, validación cruzada e intervalos de confianza.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Interpretar los resultados desde una perspectiva operativa: tablas de distancias seguras y cargas máximas por retardo.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) Definición de PPV

La **Velocidad Pico de Partícula (PPV)** es el máximo valor instantáneo de la velocidad de vibración de una partícula del suelo/roca medida por el sensor durante el evento de voladura. Se expresa típicamente en mm/s. Es preferida frente a aceleración o desplazamiento porque correlaciona bien con daño estructural observado en edificaciones convencionales y resulta relativamente robusta frente a diferencias locales de litología.

Los geófonos triaxiales registran la velocidad de partícula en tres componentes ortogonales (longitudinal, transversal y vertical). La PPV se reporta como el máximo de la **resultante vectorial** o, más comúnmente en normativa, como el máximo de cualquiera de las tres componentes individuales.

### 2.2) Distancia escalada (Scaled Distance)

Una de las ideas empíricas más influyentes en la predicción de vibraciones es la **Distancia Escalada**, también conocida como *Scaled Distance (SD)*. Esta variable combina:

- `D`: la distancia entre la voladura y el punto de medición (m), y
- `W`: la carga máxima de explosivo que detona en un mismo retardo (kg).

La formulación más utilizada es la de raíz cuadrada (square-root scaling):

<div class="formula">
  <code>SD = D / √W</code>
</div>

La lógica es que, para una misma distancia, mayor carga produce mayor vibración; y para una misma carga, mayor distancia produce menor vibración. La distancia escalada normaliza ambos efectos en una sola variable, lo que permite comparar mediciones de diferentes voladuras en un mismo gráfico.

Existen otras formulaciones de distancia escalada (cúbica, 2/3), pero la de raíz cuadrada propuesta por la USBM es la más ampliamente adoptada en minería superficial.

### 2.3) Ecuación USBM clásica

Históricamente, la USBM presentó que la PPV puede aproximarse mediante una relación potencial del tipo:

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Ecuación USBM</div>
  <div class="formula">
    <code>PPV = K · (SD)<sup>-β</sup></code>
  </div>
  <p>donde:</p>
  <ul>
    <li><code>PPV</code>: Velocidad Pico Partícula (mm/s).</li>
    <li><code>SD</code>: Distancia Escalada (m/kg<sup>0.5</sup>).</li>
    <li><code>K</code>: Constante empírica asociada al sitio / condiciones locales (a menudo interpretada como una medida de la "intensidad" inicial de vibración).</li>
    <li><code>β</code>: Exponente de atenuación del sitio, relacionado con la rapidez con la que decae la vibración con la distancia escalada. β suele ser positivo.</li>
  </ul>

  <p>Tomando logaritmos en ambos lados:</p>

  <div class="formula">
    <code>log<sub>10</sub>(PPV) = log<sub>10</sub>(K) - β · log<sub>10</sub>(SD)</code>
  </div>
</div>

Esto es una relación lineal entre `log10(PPV)` y `log10(SD)`. En consecuencia, ajustar el modelo USBM equivale a hacer una **regresión lineal** en el espacio **log-log**, para estimar los parámetros **log<sub>10</sub>(K) (intercepto)** y **β (pendiente negativa)**. Este procedimiento se ha convertido en estándar en ingeniería de voladura superficial y en control de vibraciones en minería y túneles.

### 2.4) Interpretación física de K y β

- **K** captura, entre otros factores, la eficiencia con la que la energía explosiva se acopla al macizo rocoso local. Depende de la geología, del confinamiento del explosivo, de si la voladura está superficial o enterrada, del desacople del explosivo en el taladro, y de las condiciones de disparo. Valores típicos de K oscilan entre 100 y 5000 mm/s según el sitio.
- **β** describe la atenuación de las ondas sísmicas / vibracionales con la distancia. Puede verse como un resumen empírico de la geometría de propagación, disipación y heterogeneidad estructural. Valores mayores de β implican que la vibración cae más rápido con la distancia escalada. Valores típicos: 1.0 a 2.0.

La combinación (K, β) es específica de cada sitio y debe calibrarse con datos de campo. No son transferibles entre minas sin validación.

### 2.5) Valores de referencia normativos

Para contextualizar los resultados, es útil conocer los límites regulatorios más utilizados:

| Norma / Referencia | Límite PPV (mm/s) | Aplicación |
|---|---|---|
| USBM RI 8507 (conservador) | 12.5 | Estructuras residenciales |
| USBM RI 8507 (general) | 50.8 | Estructuras comerciales/industriales |
| NTP 350.004 (Perú) | 50 | Vibraciones por voladura |
| DIN 4150-3 (Alemania) | 5 – 50 | Según tipo de estructura y frecuencia |
| BS 7385-2 (UK) | 15 – 50 | Según tipo de edificación |

### 2.6) Limitaciones del modelo USBM

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} Limitaciones</div>
  <ul>
    <li>El modelo asume que la vibración medida en un punto está dominada por un único bloque de carga que detona en un retardo específico y que la propagación es aproximadamente radial.</li>
    <li>En la práctica, las voladuras de producción involucran <em>múltiples taladros</em> y las ondas pueden superponerse constructivamente, produciendo PPV mayores que los estimados por una sola <strong>"carga equivalente"</strong>.</li>
    <li>No incorpora variables geológicas explícitamente (tipo de roca, fracturas, agua subterránea). Estos efectos quedan implícitos en K y β.</li>
    <li>Asume homogeneidad del macizo entre el punto de voladura y el receptor.</li>
    <li>No modela efectos topográficos (amplificación en crestas, atenuación en valles).</li>
  </ul>
</div>


<a id="datos" class="anchor-clean"></a>
## 3) Datos y parámetros de entrada

### 3.1) Variables del dataset

Para este tutorial generamos un dataset sintético que emula una campaña real de monitoreo de vibraciones. Las variables son:

| Variable | Símbolo | Unidad | Rango simulado | Descripción |
|---|---|---|---|---|
| Distancia | D | m | 30 – 400 | Distancia del punto de voladura al geófono |
| Carga por retardo | W | kg | 5 – 60 | Carga máxima de explosivo que detona simultáneamente |
| Distancia escalada | SD | m/kg<sup>0.5</sup> | ~4 – 180 | SD = D / √W |
| PPV medida | PPV | mm/s | variable | Velocidad pico de partícula registrada |

### 3.2) Parámetros del sitio simulado

Los parámetros "verdaderos" del sitio permiten validar el modelo ajustado:

| Parámetro | Valor | Justificación |
|---|---|---|
| K<sub>sitio</sub> | 1000.0 mm/s | Valor medio para roca competente (Siskind et al., 1980) |
| β<sub>sitio</sub> | 1.60 | Atenuación moderada, típica de macizos medianamente fracturados |
| σ<sub>ruido</sub> | 0.25 | Dispersión lognormal que replica variabilidad de campo |
| n | 150 | Cantidad de registros de monitoreo |

### 3.3) Generación de datos sintéticos físicamente plausibles

El ruido se aplica como **factor multiplicativo lognormal** sobre la PPV ideal. Esto refleja la naturaleza de la variabilidad real en campo:

- En escala logarítmica, el ruido se distribuye normalmente (aditivo)
- En escala original, el ruido es multiplicativo (un registro puede ser 1.3× o 0.7× el valor teórico)
- La dispersión σ = 0.25 produce un coeficiente de variación de ~25%, consistente con campañas de monitoreo reales

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Fuentes de variabilidad en campo</div>
  <ul>
    <li>Heterogeneidad del macizo rocoso (litología, fracturas, agua)</li>
    <li>Variaciones en el acoplamiento explosivo-roca</li>
    <li>Diferencias en la secuencia real de detonación vs diseño</li>
    <li>Condiciones locales del terreno en el punto de medición</li>
    <li>Interferencia constructiva/destructiva entre ondas de diferentes taladros</li>
  </ul>
</div>


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

### 4.1) Librerías utilizadas

En esta sección se importan las librerías base para análisis numérico, manipulación tabular, visualización y modelado estadístico.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

from scipy import stats

# Reproducibilidad
rng = np.random.default_rng(seed=42)
```

### 4.2) Generación de datos sintéticos representativos

A continuación se genera un conjunto de datos sintético que emula una campaña real de monitoreo de vibraciones por voladura.

```python
# Parámetros del sitio (ground truth)
K_SITE = 1000.0      # constante de sitio (mm/s)
BETA_SITE = 1.60     # exponente de atenuación
SIGMA_NOISE = 0.25   # dispersión lognormal
N_SAMPLES = 150      # número de registros

# Variables operativas
distance_m = rng.uniform(30, 400, size=N_SAMPLES)
charge_kg = rng.uniform(5, 60, size=N_SAMPLES)
scaled_distance = distance_m / np.sqrt(charge_kg)

# PPV ideal según USBM + ruido lognormal multiplicativo
ppv_ideal = K_SITE * (scaled_distance ** (-BETA_SITE))
noise_factor = rng.lognormal(mean=0.0, sigma=SIGMA_NOISE, size=N_SAMPLES)
ppv_measured = ppv_ideal * noise_factor

# DataFrame
data = pd.DataFrame({
    "distance_m": np.round(distance_m, 1),
    "charge_kg": np.round(charge_kg, 1),
    "scaled_distance": np.round(scaled_distance, 2),
    "ppv_ideal": np.round(ppv_ideal, 4),
    "ppv_mm_s": np.round(ppv_measured, 4)
})

data = data.sort_values("scaled_distance").reset_index(drop=True)
data.head(10)
```

El dataset resultante contiene 150 registros con distancias entre 30 y 400 m, cargas entre 5 y 60 kg/retardo, y PPV que varían desde fracciones de mm/s (lejos, poca carga) hasta decenas de mm/s (cerca, mucha carga).


<a id="eda" class="anchor-clean"></a>
## 5) Análisis Exploratorio de Datos (EDA)

### 5.1) Estadística descriptiva

Revisamos las distribuciones básicas para validar que las simulaciones están dentro de rangos físicamente razonables para minería superficial.

```python
desc_stats = data[["distance_m", "charge_kg", "scaled_distance", "ppv_mm_s"]].describe()
desc_stats.round(2)
```

### 5.2) Distribuciones de las variables

Se grafican histogramas de las cuatro variables principales para verificar la cobertura del espacio de parámetros. Una buena cobertura es importante para que la regresión sea representativa.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(data["distance_m"], bins=20, color="#0ea5e9", edgecolor="white")
axes[0, 0].set_xlabel("Distancia D (m)")
axes[0, 0].set_title("Distribución de distancias")

axes[0, 1].hist(data["charge_kg"], bins=20, color="#f59e0b", edgecolor="white")
axes[0, 1].set_xlabel("Carga máxima por retardo W (kg)")
axes[0, 1].set_title("Distribución de cargas")

axes[1, 0].hist(data["scaled_distance"], bins=25, color="#10b981", edgecolor="white")
axes[1, 0].set_xlabel("Distancia Escalada SD (m/kg^0.5)")
axes[1, 0].set_title("Distribución de SD")

axes[1, 1].hist(data["ppv_mm_s"], bins=25, color="#ef4444", edgecolor="white")
axes[1, 1].set_xlabel("PPV medida (mm/s)")
axes[1, 1].set_title("Distribución de PPV")

plt.tight_layout()
plt.show()
```

### 5.3) Matriz de correlación

La matriz de correlación de Pearson muestra la relación lineal entre variables. Esperamos una correlación negativa fuerte entre SD y PPV.

```python
corr_matrix = data[["distance_m", "charge_kg", "scaled_distance", "ppv_mm_s"]].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)

labels = ["Distancia", "Carga", "SD", "PPV"]
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr_matrix.values[i, j]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.8, label="Correlación de Pearson")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.show()
```

### 5.4) Visualización: PPV vs Distancia y PPV vs SD

Se grafican dos relaciones clave, coloreando los puntos por la carga explosiva:

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# PPV vs Distancia física
scatter1 = axes[0].scatter(
    data["distance_m"], data["ppv_mm_s"],
    c=data["charge_kg"], cmap="YlOrRd", alpha=0.7, edgecolor="white", s=50
)
axes[0].set_xlabel("Distancia D (m)")
axes[0].set_ylabel("PPV (mm/s)")
axes[0].set_title("PPV vs Distancia física")
plt.colorbar(scatter1, ax=axes[0], label="Carga W (kg)")

# PPV vs SD (escala log-log)
scatter2 = axes[1].scatter(
    data["scaled_distance"], data["ppv_mm_s"],
    c=data["charge_kg"], cmap="YlOrRd", alpha=0.7, edgecolor="white", s=50
)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Distancia Escalada SD (m/kg^0.5)")
axes[1].set_ylabel("PPV (mm/s)")
axes[1].set_title("PPV vs SD (escala log-log)")
plt.colorbar(scatter2, ax=axes[1], label="Carga W (kg)")

plt.tight_layout()
plt.show()
```

Observaciones clave:
- En el gráfico izquierdo, la dispersión es alta porque la distancia sola no explica la PPV — falta incorporar la carga.
- En el gráfico derecho (log-log), los puntos se alinean claramente en una tendencia lineal, validando el uso del modelo USBM.


<a id="ajuste" class="anchor-clean"></a>
## 6) Ajuste del modelo USBM

### 6.1) Transformación logarítmica y regresión lineal

Ajustamos la relación `PPV = K · (SD)^(-β)` tomando logaritmos en base 10:

`log₁₀(PPV) = log₁₀(K) - β · log₁₀(SD)`

Esto se reescribe como `Y = A + B · X` donde:

- `Y = log₁₀(PPV)`, `X = log₁₀(SD)`
- `A = log₁₀(K)` (intercepto), `B = -β` (pendiente)

```python
# Transformación log10
X_log = np.log10(data["scaled_distance"].values).reshape(-1, 1)
Y_log = np.log10(data["ppv_mm_s"].values).reshape(-1, 1)

# Regresión lineal con Scikit-learn
model = LinearRegression()
model.fit(X_log, Y_log)

Y_pred = model.predict(X_log)

# Extraer parámetros USBM
A = float(model.intercept_[0])     # log10(K)
B = float(model.coef_[0][0])       # -beta

K_est = 10 ** A
beta_est = -B

# Métricas
r2 = r2_score(Y_log, Y_pred)
rmse_log = np.sqrt(mean_squared_error(Y_log, Y_pred))
mae_log = mean_absolute_error(Y_log, Y_pred)

print(f"K estimado:    {K_est:.2f} mm/s   (real: {K_SITE})")
print(f"β estimado:    {beta_est:.4f}        (real: {BETA_SITE})")
print(f"R² (log-log):  {r2:.4f}")
print(f"RMSE (log₁₀):  {rmse_log:.4f}")
```

### 6.2) Visualización del ajuste log-log

```python
sd_range = np.linspace(data["scaled_distance"].min(), data["scaled_distance"].max(), 300)
ppv_fit_line = K_est * (sd_range ** (-beta_est))

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.scatter(
    data["scaled_distance"], data["ppv_mm_s"],
    alpha=0.6, color="#0ea5e9", edgecolor="white", s=45, label="Datos medidos"
)

ax.plot(
    sd_range, ppv_fit_line,
    color="#ef4444", linewidth=2.5,
    label=f"Ajuste USBM: PPV = {K_est:.0f} · SD^(-{beta_est:.2f})  R²={r2:.4f}"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Distancia Escalada SD (m/kg^0.5)")
ax.set_ylabel("PPV (mm/s)")
ax.set_title("Modelo USBM ajustado — PPV vs Distancia Escalada (log-log)")
ax.legend()
plt.tight_layout()
plt.show()
```

### 6.3) PPV medida vs PPV predicha (gráfico 1:1)

El gráfico 1:1 es una herramienta de validación clave. Si el modelo es perfecto, todos los puntos caen sobre la diagonal. La dispersión alrededor de la diagonal refleja la variabilidad no explicada por el modelo.

```python
ppv_pred_original = 10 ** Y_pred.flatten()
ppv_real_original = data["ppv_mm_s"].values

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(ppv_real_original, ppv_pred_original,
           alpha=0.6, color="#0ea5e9", edgecolor="white", s=40)

lim_min = min(ppv_real_original.min(), ppv_pred_original.min()) * 0.8
lim_max = max(ppv_real_original.max(), ppv_pred_original.max()) * 1.2
ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#ef4444", linewidth=2,
        label="Línea 1:1 (predicción perfecta)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("PPV medida (mm/s)")
ax.set_ylabel("PPV predicha (mm/s)")
ax.set_title("PPV medida vs PPV predicha")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.show()
```


<a id="validacion" class="anchor-clean"></a>
## 7) Validación estadística

### 7.1) Análisis de residuos

Un modelo bien calibrado debe tener residuos que:
- Se distribuyan aleatoriamente alrededor de cero (sin patrón sistemático)
- Tengan varianza constante (homocedasticidad)
- Se aproximen a una distribución normal

```python
residuals = (Y_log - Y_pred).flatten()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Residuos vs X
axes[0].scatter(X_log.flatten(), residuals, alpha=0.6, color="#0ea5e9", edgecolor="white")
axes[0].axhline(0, color="#ef4444", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("log₁₀(SD)")
axes[0].set_ylabel("Residuo (log₁₀)")
axes[0].set_title("Residuos vs log₁₀(SD)")

# Histograma
axes[1].hist(residuals, bins=20, color="#10b981", edgecolor="white", density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[1].plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
             color="#ef4444", linewidth=2, label="Normal teórica")
axes[1].set_xlabel("Residuo (log₁₀)")
axes[1].set_title("Histograma de residuos")
axes[1].legend()

# QQ-Plot
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("QQ-Plot de residuos")

plt.suptitle("Diagnóstico de residuos del modelo USBM", fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
```

### 7.2) Test de normalidad (Shapiro-Wilk)

```python
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Test de Shapiro-Wilk:")
print(f"  Estadístico W: {shapiro_stat:.4f}")
print(f"  p-value:       {shapiro_p:.4f}")
print(f"  Conclusión:    {'Residuos normales (p > 0.05)' if shapiro_p > 0.05 else 'Residuos NO normales (p < 0.05)'}")
```

Si los residuos son normales, los intervalos de predicción basados en la distribución t-student son válidos. Si no, se podrían usar métodos bootstrap.

### 7.3) Validación cruzada (K-Fold)

Para evaluar la estabilidad del modelo y asegurar que no está sobreajustado, realizamos una validación cruzada con **k=5 folds**.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(LinearRegression(), X_log, Y_log.ravel(), cv=kf, scoring="r2")
cv_neg_mse = cross_val_score(LinearRegression(), X_log, Y_log.ravel(), cv=kf, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_neg_mse)

print("Validación cruzada (5-Fold):")
print("-" * 45)
for i, (r2_i, rmse_i) in enumerate(zip(cv_r2, cv_rmse), 1):
    print(f"  Fold {i}:  R² = {r2_i:.4f}  |  RMSE = {rmse_i:.4f}")
print("-" * 45)
print(f"  Media:   R² = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  Media:   RMSE = {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
```

La baja varianza entre folds confirma que el modelo es estable y generaliza bien a datos no vistos. Esto es esperado dado que la relación es genuinamente lineal en espacio log-log.

### 7.4) Intervalos de confianza al 95%

Para uso operativo, no basta con la predicción puntual — necesitamos **bandas de predicción** que indiquen el rango probable de PPV.

```python
se_residuals = np.std(residuals, ddof=2)
n = len(residuals)
t_crit = stats.t.ppf(0.975, df=n - 2)

Y_pred_flat = Y_pred.flatten()
upper_log = Y_pred_flat + t_crit * se_residuals
lower_log = Y_pred_flat - t_crit * se_residuals

ppv_upper = 10 ** upper_log
ppv_lower = 10 ** lower_log

order = np.argsort(data["scaled_distance"].values)
sd_sorted = data["scaled_distance"].values[order]

fig, ax = plt.subplots(figsize=(10, 6.5))

ax.fill_between(sd_sorted, ppv_lower[order], ppv_upper[order],
                alpha=0.2, color="#f59e0b", label="Intervalo de predicción 95%")
ax.scatter(data["scaled_distance"], data["ppv_mm_s"],
           alpha=0.5, color="#0ea5e9", edgecolor="white", s=35, label="Datos medidos")
ax.plot(sd_range, ppv_fit_line,
        color="#ef4444", linewidth=2.5, label=f"Modelo USBM")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Distancia Escalada SD (m/kg^0.5)")
ax.set_ylabel("PPV (mm/s)")
ax.set_title("Modelo USBM con intervalos de predicción al 95%")
ax.legend()
plt.tight_layout()
plt.show()
```

La banda de predicción al 95% define el rango dentro del cual se espera que caiga el 95% de las mediciones futuras. El **límite superior** de esta banda es lo que el ingeniero de voladura debería usar para diseño conservador.


<a id="prediccion" class="anchor-clean"></a>
## 8) Predicción operativa

### 8.1) Curvas PPV vs Distancia para diferentes cargas

Esta es la herramienta más práctica del análisis. Dado el modelo calibrado, se generan curvas de predicción para distintas cargas por retardo, junto con los límites regulatorios.

```python
distances = np.linspace(10, 500, 300)
charges = [10, 20, 30, 50]
colors_charge = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]

fig, ax = plt.subplots(figsize=(10, 6.5))

for W, color in zip(charges, colors_charge):
    sd = distances / np.sqrt(W)
    ppv_pred = K_est * (sd ** (-beta_est))
    ax.plot(distances, ppv_pred, linewidth=2.5, color=color, label=f"W = {W} kg/retardo")

# Límites operativos
ax.axhline(50, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(15, 55, "Límite NTP (50 mm/s)", fontsize=9, color="gray")

ax.axhline(12.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(15, 14, "Límite USBM conservador (12.5 mm/s)", fontsize=9, color="gray")

ax.set_xlabel("Distancia al punto de monitoreo D (m)")
ax.set_ylabel("PPV predicha (mm/s)")
ax.set_title("Predicción de PPV según distancia y carga por retardo")
ax.set_yscale("log")
ax.legend(title="Carga por retardo")
ax.set_xlim(10, 500)
plt.tight_layout()
plt.show()
```

### 8.2) Tabla de distancias mínimas seguras

Dado un límite de PPV y una carga por retardo, la distancia mínima segura se calcula como:

<div class="formula">
  <code>D<sub>min</sub> = √W · (PPV<sub>max</sub> / K)<sup>-1/β</sup></code>
</div>

```python
ppv_limits = [50, 25, 12.5, 5]  # mm/s
charges_table = [5, 10, 20, 30, 40, 50, 60]  # kg

print("Distancia mínima segura D_min (m)")
print("=" * 70)

header = f"{'W (kg)':>8}"
for ppv_lim in ppv_limits:
    header += f"  PPV<{ppv_lim:>5} mm/s"
print(header)
print("-" * 70)

for W in charges_table:
    row = f"{W:>8}"
    for ppv_lim in ppv_limits:
        d_min = np.sqrt(W) * (ppv_lim / K_est) ** (-1 / beta_est)
        row += f"{d_min:>16.1f}"
    print(row)
```

### 8.3) Tabla de carga máxima permitida por retardo

Inversamente, dado un límite de PPV y una distancia fija al receptor:

<div class="formula">
  <code>W<sub>max</sub> = (D / SD<sub>min</sub>)²</code> &nbsp;&nbsp; donde &nbsp;&nbsp; <code>SD<sub>min</sub> = (PPV<sub>max</sub> / K)<sup>-1/β</sup></code>
</div>

```python
distances_table = [50, 100, 150, 200, 300, 400]
ppv_limit = 50  # mm/s (NTP)

print(f"Carga máxima por retardo W_max (kg) para PPV < {ppv_limit} mm/s")
print("=" * 45)
print(f"{'D (m)':>8}  {'W_max (kg)':>12}  {'SD mínimo':>12}")
print("-" * 45)

for D in distances_table:
    sd_min = (ppv_limit / K_est) ** (-1 / beta_est)
    w_max = (D / sd_min) ** 2
    print(f"{D:>8}  {w_max:>12.1f}  {sd_min:>12.2f}")
```

Esta tabla es de uso directo en campo: el supervisor de voladura consulta la distancia al receptor más cercano y determina la carga máxima por retardo que puede usar.


<a id="conclusiones" class="anchor-clean"></a>
## 9) Conclusiones operativas

### 9.1) Lectura de los parámetros calibrados

Los parámetros estimados por el modelo (K ≈ 1000 mm/s, β ≈ 1.60) coinciden con los valores simulados, validando el flujo de calibración. En una aplicación real, estos valores serían específicos del sitio y deberían recalcularse con cada nueva campaña de monitoreo o cuando cambien las condiciones geológicas.

- **K ≈ 1000 mm/s** indica un sitio con acoplamiento medio entre la carga explosiva y el macizo. Valores más altos (>2000) sugieren macizo muy competente con buen confinamiento. Valores más bajos (<500) sugieren roca fracturada o mala práctica de cargado.
- **β ≈ 1.60** indica una atenuación moderada. Valores más altos (>1.8) se observan en macizos muy fracturados o con alto contenido de arcilla. Valores más bajos (<1.3) se observan en macizos masivos y homogéneos.

### 9.2) Uso práctico del modelo

El modelo calibrado permite tres aplicaciones operativas directas:

1. **Predicción pre-voladura**: antes de disparar, estimar la PPV esperada en receptores sensibles y verificar cumplimiento normativo.
2. **Diseño inverso**: dado un límite de PPV, determinar la carga máxima por retardo o la distancia mínima segura.
3. **Control de calidad**: comparar la PPV medida post-voladura con la predicción. Desviaciones sistemáticas indican cambios en las condiciones del sitio.

### 9.3) Recomendaciones para implementación en campo

- Recalibrar K y β al menos cada 6 meses o cuando cambie el nivel de operación (nuevo banco, nueva zona geológica).
- Usar el **límite superior del intervalo de predicción al 95%** para diseño conservador, no la predicción media.
- Mantener un registro histórico de (D, W, PPV) para cada voladura. La acumulación de datos mejora la precisión del modelo.
- Complementar con monitoreo de frecuencia: la norma DIN 4150-3 establece límites diferenciados según la frecuencia dominante de la vibración.

### 9.4) Trabajo futuro

- Aplicar el modelo a datos reales de una mina específica
- Comparar con modelos de machine learning (Random Forest, XGBoost, SVR) que capturen no-linealidades
- Incorporar variables adicionales: tipo de roca, tipo de explosivo, burden, espaciamiento, orientación del frente
- Implementar predicción probabilista con modelos bayesianos
- Desarrollar un dashboard interactivo con Streamlit para uso operativo en campo


<a id="refs" class="anchor-clean"></a>
## 10) Referencias

<div class="references" markdown="1">

Siskind, D. E., Stagg, M. S., Kopp, J. W., & Dowding, C. H. (1980). **Structure response and damage produced by ground vibration from surface mine blasting.** U.S. Bureau of Mines, Report of Investigations RI 8507. Este trabajo estableció relaciones empíricas ampliamente adoptadas para estimar PPV a partir de distancia escalada en minería superficial.

Agrawal, H., & Mishra, A. K. (2019). **Modified scaled distance regression analysis approach for prediction of blast-induced ground vibration in multi-hole blasting.** Journal of Rock Mechanics and Geotechnical Engineering, 11, 202–207. El artículo discute cómo la interferencia entre ondas de múltiples taladros puede producir PPV mayores que los previstos por la ecuación USBM estándar, y propone una corrección para escenarios multihoyo.

Grobbelaar, M., Molea, T., & Durrheim, R. (2020). **Measurement of air and ground vibrations produced by explosions situated on the Earth's surface.** Journal of the Southern African Institute of Mining and Metallurgy, 120(9), 521–528. Analiza vibraciones y sobrepresión de explosiones superficiales no confinadas, y evalúa la aplicabilidad de ecuaciones tipo USBM en escenarios de demolición / superficie.

Sambuelli, L. (2009). **Theoretical derivation of a peak particle velocity–distance law for the prediction of vibrations from blasting.** Rock Mechanics and Rock Engineering, 42, 547–556. Propone una justificación teórica para la forma funcional de las leyes tipo PPV = K · SD^(-β), vinculando explícitamente energía explosiva, propagación de ondas y parámetros del macizo rocoso.

Dowding, C. H. (1985). **Blast Vibration Monitoring and Control.** Prentice-Hall. Texto de referencia para la ingeniería de control de vibraciones por voladura, incluyendo instrumentación, criterios de daño y métodos de predicción.

</div>
