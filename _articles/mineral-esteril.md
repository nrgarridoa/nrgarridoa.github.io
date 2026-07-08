---
layout: article
title: "Clasificar mineral y estéril con regresión logística"
subtitle: "Tutorial reproducible: de Cu y Mo a una decisión de ley de corte."
date: 2026-07-01

# Hero + card
cover: "/assets/articles/mineral-esteril/cover.jpg"
hero: "/assets/articles/mineral-esteril/hero.jpg"
card: /assets/articles/mineral-esteril/card.jpg

# Metadatos / filtros
series: "Ore Control"
series_order: 1
tags: [Python, Machine Learning, Ley de corte, Porfido]
reading_time: "20 min"

# Resumen destacado
summary: >
  <p>En este artículo entrenamos una <strong>regresión logística</strong> que clasifica muestras como <strong>mineral</strong> o <strong>estéril</strong> en un pórfido de Cu-Mo, a partir de dos mediciones geoquímicas rápidas: la <em>ley de Cu (%)</em> y el <em>Mo (ppm)</em>. El modelo estima una probabilidad de mineralización y la convierte en una decisión de destino (planta o botadero).</p>
  <p></p>
  <div class="ppv-note">Modelo base: <strong>p(mineral) = σ(z)</strong> &nbsp;⇔&nbsp; <code>z = β0 + β1·Cu + β2·Mo</code></div>

# Índice de contenidos
contents:
  - { anchor: "#intr", title: "La decisión mineral / estéril" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos y parámetros de entrada" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#eda", title: "Análisis Exploratorio (EDA)" }
  - { anchor: "#ajuste", title: "Ajuste del modelo" }
  - { anchor: "#validacion", title: "Validación estadística" }
  - { anchor: "#uso", title: "Uso operativo: umbral y costo" }
  - { anchor: "#conclusiones", title: "Conclusiones operativas" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Variable objetivo", body: "Clase mineral / estéril (etiqueta binaria del geólogo)." }
  - { title: "Modelo", body: <div class="ppv-card__value">p = σ(β0 + β1·Cu + β2·Mo)</div> }
  - { title: "Stack", body: "Python · pandas · scikit-learn · Matplotlib" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/mineral-esteril/notebooks/mineral-esteril.ipynb" }
  - { type: "data",     label: "Dataset CSV",      url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/mineral-esteril/data/raw/mineral_esteril.csv" }
  - { type: "repo",     label: "Repositorio",      url: "https://github.com/nrgarridoa/talleres-mineria-python" }

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
## 1) La decisión mineral / estéril

### 1.1) Importancia del control de leyes

El control de leyes (*ore control*) decide, banco a banco y muestra a muestra, el destino de la roca extraída: si va a la planta de procesamiento (**mineral**) o al botadero (**estéril**). Esa decisión se toma miles de veces al día en una operación a cielo abierto, y cada equivocación cuesta dinero en una de dos direcciones opuestas. Enviar estéril a la planta **diluye** la ley alimentada (se procesa roca sin valor y se gasta energía, reactivos y capacidad de molienda en ella). Botar mineral bueno al botadero es **pérdida** directa de metal que nunca se recupera.

La regla clásica es un **corte por ley**: si la ley de la muestra supera la ley de corte económica, la roca es mineral. El problema operativo es de tiempo. La ley de laboratorio tarda horas o días, y en el frente de mina se necesitan decisiones inmediatas para dirigir palas y camiones.

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} Decisión operacional</div>
  El control de leyes responde preguntas como:
  <ul>
    <li>¿Este bloque va a planta o al botadero?</li>
    <li>¿Cuánto estéril estamos enviando a planta (dilución) y cuánto mineral estamos perdiendo?</li>
    <li>¿Podemos clasificar en el frente sin esperar el laboratorio, sin perder criterio geológico?</li>
  </ul>
</div>

### 1.2) Clasificación a partir de mediciones rápidas

Una alternativa práctica es **predecir la clasificación del geólogo** a partir de mediciones inmediatas de campo. Un lector portátil de fluorescencia de rayos X (*pXRF*) entrega la ley de Cu y el contenido de Mo en segundos, directamente sobre la muestra. Si un modelo reproduce con confianza la clasificación experta desde esas dos variables, se gana velocidad sin sacrificar el criterio.

Este es un problema de **clasificación binaria**, y la regresión logística es la herramienta canónica para resolverlo: es interpretable, entrega probabilidades calibradas y es sencilla de auditar. Desde el punto de vista de analítica, buscamos aprender una relación cuantitativa entre variables medibles (Cu y Mo) y la respuesta observada (la etiqueta mineral / estéril), y luego usar esa relación para clasificar muestras nuevas.

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Metas pedagógicas y operativas para comprender, calibrar y aplicar un clasificador de mineral / estéril con regresión logística en Python, y traducir sus salidas en una decisión de ley de corte.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Explicar el problema minero de clasificar mineral y estéril, y los dos costos opuestos de equivocarse: dilución y pérdida.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Describir la regresión logística: la función sigmoide, el log-odds y la interpretación de sus coeficientes.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Construir un dataset sintético físicamente plausible a partir de un modelo logístico conocido (*ground truth*) para un pórfido de Cu-Mo.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Ejecutar un flujo reproducible en Python: EDA, estandarización, ajuste, matriz de confusión, curva ROC y validación cruzada.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Elegir el umbral de decisión minimizando el costo total del error según la relación dilución / pérdida del proyecto.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) La función sigmoide

La regresión logística modela la **probabilidad** de que una muestra pertenezca a la clase positiva (mineral) como una función *sigmoide* de una combinación lineal de las variables. La sigmoide toma cualquier número real y lo comprime al intervalo (0, 1), lo que la vuelve idónea para representar una probabilidad:

<div class="formula">
  <code>p(mineral) = σ(z) = 1 / (1 + e<sup>−z</sup>)</code>
</div>

donde el término lineal `z` combina las variables geoquímicas:

<div class="formula">
  <code>z = β<sub>0</sub> + β<sub>1</sub>·Cu + β<sub>2</sub>·Mo</code>
</div>

A diferencia de la regresión lineal, que estima un valor continuo, la logística estima una probabilidad acotada entre 0 y 1. Se clasifica una muestra como mineral cuando `p ≥` un **umbral** (por defecto 0.5). Ese umbral no es sagrado: se ajusta según el costo de cada tipo de error (Sección 8).

### 2.2) Log-odds y razón de momios (odds ratio)

El término `z` tiene una interpretación directa: es el **log-odds**, el logaritmo de la razón de momios (odds) entre ser mineral y ser estéril.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} El log-odds</div>
  <div class="formula">
    <code>ln( p / (1−p) ) = β<sub>0</sub> + β<sub>1</sub>·Cu + β<sub>2</sub>·Mo</code>
  </div>
  <p>Es decir, la regresión logística es <strong>lineal en el espacio del log-odds</strong>. Un incremento de una unidad en una variable suma su coeficiente al log-odds, y multiplica los odds por <code>e<sup>β</sup></code> (el <em>odds ratio</em>).</p>
</div>

Esto entrega una lectura cuantitativa de cada coeficiente. Un `β` positivo empuja la clasificación hacia mineral. Su exponencial `e^β` indica cuánto se multiplican los odds de ser mineral por cada unidad adicional de la variable.

### 2.3) Interpretación de los coeficientes

Con las variables **estandarizadas** (media 0, desviación 1), el signo y la magnitud de cada `β` dicen cuánto y en qué dirección pesa cada variable, en unidades comparables. Un `β` positivo grande para el Cu significa que más cobre empuja la clasificación hacia mineral. Comparar `|β_Cu|` y `|β_Mo|` revela cuál variable domina la decisión.

| Situación | Signo de β | Efecto sobre p(mineral) | Odds ratio e<sup>β</sup> |
|---|---|---|---|
| Variable favorece mineral | β > 0 | Aumenta al subir la variable | > 1 |
| Variable sin efecto | β ≈ 0 | Indiferente | ≈ 1 |
| Variable favorece estéril | β < 0 | Disminuye al subir la variable | < 1 |
| Variable domina la decisión | \|β\| grande | Cambio abrupto de p | Lejos de 1 |

### 2.4) Geoquímica: por qué Cu y Mo

En los **pórfidos de cobre**, el molibdeno (Mo) acompaña al cobre como subproducto: ambos se concentran en el núcleo mineralizado y decaen hacia la periferia estéril. El Mo actúa como *pathfinder* (indicador geoquímico) del sistema porfídico. Por eso Cu y Mo **co-varían** positivamente y, juntos, separan mejor mineral de estéril que cualquiera de ellos por separado.

| Variable | Rol geoquímico | Comportamiento en el pórfido |
|---|---|---|
| Cu (%) | Metal de interés económico | Alto en el núcleo, driver dominante de la clasificación |
| Mo (ppm) | Subproducto y *pathfinder* | Co-varía con el Cu, refuerza la señal de mineralización |
| Correlación Cu-Mo | Estructura del yacimiento | Positiva (~0.5), ambos suben juntos hacia el núcleo |

### 2.5) Limitaciones del enfoque

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} Limitaciones</div>
  <ul>
    <li>La frontera de decisión de la regresión logística es <strong>lineal</strong> (en el espacio de las variables estandarizadas). Yacimientos con relaciones no lineales entre geoquímica y mineralización requieren modelos más flexibles o variables transformadas.</li>
    <li>El modelo aprende de <em>etiquetas</em> del geólogo. Si esas etiquetas tienen sesgo o error, el modelo lo hereda: no reemplaza el criterio experto, lo reproduce.</li>
    <li>Solo usa dos variables (Cu y Mo). Contactos, alteración, litología y estructura no entran explícitamente y quedan implícitos en el ruido.</li>
    <li>La probabilidad estimada supone que el pXRF mide sin sesgo sistemático. Errores de calibración del instrumento se propagan a la clasificación.</li>
    <li>El modelo es específico del sitio y del rango de leyes con que se entrenó. Extrapolar a otra zona o campaña exige re-calibrar.</li>
  </ul>
</div>


<a id="datos" class="anchor-clean"></a>
## 3) Datos y parámetros de entrada

### 3.1) Variables del dataset

Para este tutorial generamos un dataset sintético que emula una campaña de muestreo geoquímico de un pórfido de Cu-Mo. Las variables son:

| Variable | Símbolo | Unidad | Rango simulado | Descripción |
|---|---|---|---|---|
| Ley de cobre | Cu | % | 0.09 – 1.90 | Ley de Cu de la muestra (pXRF) |
| Molibdeno | Mo | ppm | 11 – 289 | Contenido de Mo (*pathfinder* de pórfido) |
| Clase | mineral | 0/1 | (adimensional) | 1 = mineral, 0 = estéril (etiqueta del geólogo) |

### 3.2) Parámetros del *ground truth*

La etiqueta proviene de un modelo logístico **conocido**, lo que permite validar que el modelo ajustado recupera la dirección verdadera de los coeficientes. Estos son los parámetros que generan las etiquetas:

| Parámetro | Valor | Justificación |
|---|---|---|
| β<sub>1</sub> (Cu) | 2.30 | Cu como driver dominante de la mineralización |
| β<sub>2</sub> (Mo) | 1.15 | Mo como co-indicador, la mitad del peso del Cu |
| β<sub>0</sub> | −0.15 | Intercepto (más estéril que mineral, típico de una campaña) |
| n | 240 | Muestras de la campaña |
| semilla | 2026 | Reproducibilidad |

### 3.3) Generación de datos sintéticos físicamente plausibles

La construcción respeta la física del yacimiento. Un **factor latente de mineralización** correlaciona Cu y Mo (ambos suben juntos hacia el núcleo del pórfido), y la etiqueta se **muestrea** del modelo logístico verdadero (`B0, B1, B2`) aplicado sobre las variables estandarizadas.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Realismo del dataset</div>
  <ul>
    <li>Cu y Mo se generan a partir de un factor común, reproduciendo su co-variación (~0.5) en el pórfido.</li>
    <li>La etiqueta es <strong>probabilística</strong>, no determinista: no todas las muestras de alta ley se clasifican como mineral (dilución geológica, contactos, muestras de borde).</li>
    <li>Las dos poblaciones se <strong>solapan</strong> cerca de la ley de corte, de modo que el problema no se resuelve con un umbral trivial en una sola variable.</li>
    <li>El resultado es 240 muestras: 100 mineral (42%) y 140 estéril, con Cu mediana ≈ 0.38% y Mo mediana ≈ 54 ppm.</li>
  </ul>
</div>


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

### 4.1) Librerías utilizadas

Se importan las librerías base para análisis numérico, manipulación tabular, visualización y modelado. El bloque de métricas de `sklearn` cubre toda la validación posterior (matriz de confusión, ROC, precisión, recall, F1).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, precision_score, recall_score, f1_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold

SEMILLA = 2026
rng = np.random.default_rng(SEMILLA)
```

### 4.2) Generación del dataset

Un factor latente de mineralización correlaciona Cu y Mo; la etiqueta se muestrea del modelo logístico verdadero (`B0, B1, B2`). El ruido de etiqueta reproduce la realidad de que no todas las muestras de alta ley se clasifican como mineral (dilución geológica, contactos, etc.).

```python
# Ground truth
N = 240
B0, B1, B2 = -0.15, 2.30, 1.15      # logit = B0 + B1*z_cu + B2*z_mo

fac = rng.normal(size=N)            # factor latente de mineralizacion
cu = np.clip(np.exp(-1.00 + 0.52 * (0.80 * fac + 0.62 * rng.normal(size=N))), 0.03, 3.0)   # %
mo = np.clip(np.exp(3.90 + 0.55 * (0.70 * fac + 0.70 * rng.normal(size=N))), 3.0, 800.0)   # ppm

zc = (cu - cu.mean()) / cu.std()
zm = (mo - mo.mean()) / mo.std()
p_true = 1.0 / (1.0 + np.exp(-(B0 + B1 * zc + B2 * zm)))
mineral = (rng.random(N) < p_true).astype(int)   # 1 = mineral, 0 = esteril

df = pd.DataFrame({'cu': np.round(cu, 3), 'mo': np.round(mo, 1), 'mineral': mineral})
df = df.sort_values('cu').reset_index(drop=True)
df.head(8)
```

El conjunto resultante tiene 240 muestras: 100 mineral (42%) y 140 estéril, un desbalance moderado y realista para una campaña donde predomina el estéril.

```python
# (opcional) guardar el dataset para el repositorio
import os
os.makedirs('../data/raw', exist_ok=True)
df.to_csv('../data/raw/mineral_esteril.csv', index=False)
print(f'{len(df)} muestras | mineral={df.mineral.sum()} ({df.mineral.mean():.0%}) | esteril={(df.mineral==0).sum()}')
```


<a id="eda" class="anchor-clean"></a>
## 5) Análisis Exploratorio de Datos (EDA)

### 5.1) Estadística descriptiva

Revisamos las distribuciones básicas para confirmar que las simulaciones caen en rangos físicamente razonables para un pórfido de Cu-Mo. La mediana de Cu (≈ 0.38%) y de Mo (≈ 54 ppm) son coherentes con leyes de campaña de este tipo de yacimiento.

```python
df[['cu', 'mo']].describe().round(2)
```

### 5.2) Distribuciones por clase

Se grafican los histogramas de Cu y Mo separados por clase. La superposición entre mineral y estéril es la señal de que el problema no es trivial: existen muestras de ambas clases en el mismo rango de valores.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, col, unidad in zip(axes, ['cu', 'mo'], ['Ley de Cu (%)', 'Mo (ppm)']):
    ax.hist(df[df.mineral == 1][col], bins=20, alpha=0.7, color='#0f766e', label='Mineral', edgecolor='white')
    ax.hist(df[df.mineral == 0][col], bins=20, alpha=0.7, color='#b4312a', label='Estéril', edgecolor='white')
    ax.set_xlabel(unidad); ax.set_ylabel('Frecuencia'); ax.legend()
fig.suptitle('Distribución de Cu y Mo por clase'); plt.tight_layout(); plt.show()
```

![Distribución de Cu y Mo por clase](/assets/articles/mineral-esteril/distribuciones_por_clase.png)

Las dos poblaciones se **solapan** cerca de la ley de corte: hay estéril con algo de Cu y mineral de ley moderada. Ese solape es lo que hace útil combinar ambas variables en lugar de aplicar un corte simple sobre una sola.

### 5.3) Dispersión Cu-Mo y correlación

Se proyectan las muestras en el plano Cu-Mo, coloreadas por clase, y se calcula la correlación de Pearson entre ambas variables.

```python
corr = df[['cu', 'mo']].corr().iloc[0, 1]
fig, ax = plt.subplots(figsize=(7.5, 6))
for cls, c, m, lab in [(1, '#065f46', 'o', 'Mineral'), (0, '#b4312a', 's', 'Estéril')]:
    d = df[df.mineral == cls]
    ax.scatter(d.cu, d.mo, c=c, marker=m, s=28, alpha=0.75, edgecolors='white', linewidths=0.4, label=lab)
ax.set_xlabel('Ley de Cu (%)'); ax.set_ylabel('Mo (ppm)')
ax.set_title(f'Mineral vs estéril en el espacio Cu-Mo  (correlación Cu-Mo = {corr:.2f})')
ax.legend(); plt.tight_layout(); plt.show()
```

![Dispersión Cu-Mo por clase](/assets/articles/mineral-esteril/dispersion_cu_mo.png)

La correlación Cu-Mo es **0.52**, positiva y moderada, tal como se espera del pórfido: ambos metales suben juntos hacia el núcleo. El mineral ocupa la esquina de alto Cu y alto Mo; el estéril, la de bajo Cu y bajo Mo. La frontera entre ambos es lo que el modelo debe aprender.


<a id="ajuste" class="anchor-clean"></a>
## 6) Ajuste del modelo

### 6.1) Estandarizar y entrenar

Se estandarizan las variables (media 0, desviación 1) para que Cu y Mo, con escalas muy distintas (% frente a ppm), pesen en igualdad de condiciones y sus coeficientes sean directamente comparables. Luego se ajusta la regresión logística.

```python
X = df[['cu', 'mo']].values
y = df['mineral'].values

scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

modelo = LogisticRegression().fit(Xs, y)

coef = modelo.coef_[0]
print(f'Coeficiente Cu: {coef[0]:+.3f}   (ground truth {B1:+.2f})')
print(f'Coeficiente Mo: {coef[1]:+.3f}   (ground truth {B2:+.2f})')
print(f'Intercepto:     {modelo.intercept_[0]:+.3f}   (ground truth {B0:+.2f})')
print(f'Accuracy (in-sample): {accuracy_score(y, modelo.predict(Xs)):.3f}')
```

El modelo **recupera la dirección verdadera**: ambos coeficientes son positivos y el de **Cu (+2.99) domina** al de Mo (+0.88), aproximadamente 3 veces su magnitud, consistente con el *ground truth* del pórfido (β1 = 2.30, β2 = 1.15). El acierto in-sample es de **0.887**.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} ¿Por qué no coinciden exactamente los coeficientes?</div>
  Que las magnitudes estimadas no igualen los valores verdaderos es esperable: la regularización por defecto de scikit-learn (<code>C=1</code>) encoge los coeficientes hacia cero, y el ruido de etiqueta introduce sesgo y varianza. Lo relevante es que la <strong>dirección</strong> y el <strong>orden de importancia</strong> (Cu &gt; Mo) se recuperan con claridad.
</div>

### 6.2) La frontera de decisión

Se evalúa la probabilidad del modelo sobre una malla del plano Cu-Mo y se dibuja el contorno donde `p = 0.5`, que es la frontera que separa las dos clases con el umbral por defecto.

```python
xx, yy = np.meshgrid(np.linspace(df.cu.min() - 0.05, df.cu.max() + 0.05, 300),
                     np.linspace(df.mo.min() - 10, df.mo.max() + 10, 300))
grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
Z = modelo.predict_proba(grid)[:, 1].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['#fdeeee', '#eafaf3'])
ax.contour(xx, yy, Z, levels=[0.5], colors=['#0f766e'], linewidths=2)
for cls, c, m, lab in [(1, '#065f46', 'o', 'Mineral'), (0, '#b4312a', 's', 'Estéril')]:
    d = df[df.mineral == cls]
    ax.scatter(d.cu, d.mo, c=c, marker=m, s=26, alpha=0.8, edgecolors='white', linewidths=0.4, label=lab)
ax.set_xlabel('Ley de Cu (%)'); ax.set_ylabel('Mo (ppm)')
ax.set_title('Frontera de decisión aprendida (probabilidad = 0.5)')
ax.legend(); plt.tight_layout(); plt.show()
```

![Frontera de decisión aprendida](/assets/articles/mineral-esteril/frontera_decision.png)

La frontera es una recta (en el espacio estandarizado) que corta el plano dejando el mineral en la región de alto Cu y alto Mo. Su inclinación refleja que el Cu pesa más que el Mo: la decisión se mueve principalmente al variar el eje horizontal.


<a id="validacion" class="anchor-clean"></a>
## 7) Validación estadística

### 7.1) Matriz de confusión

La matriz de confusión cruza lo real contra lo predicho. En minería, los dos errores tienen precios distintos: un **falso positivo** (estéril clasificado como mineral) va a planta y **diluye**; un **falso negativo** (mineral clasificado como estéril) se bota y es **pérdida** de metal.

```python
pred = modelo.predict(Xs)
cm = confusion_matrix(y, pred)   # [[TN, FP], [FN, TP]]

fig, ax = plt.subplots(figsize=(5.6, 5))
ax.imshow(cm, cmap='Greens', vmin=0, vmax=cm.max())
labs = ['Estéril', 'Mineral']; notas = [['OK', 'dilución'], ['pérdida', 'OK']]
ax.set_xticks([0, 1], labels=labs); ax.set_yticks([0, 1], labels=labs)
ax.set_xlabel('Predicho por el modelo'); ax.set_ylabel('Real (geólogo)')
for i in range(2):
    for j in range(2):
        ax.text(j, i - 0.08, cm[i, j], ha='center', fontsize=18, fontweight='bold',
                color='white' if cm[i, j] > cm.max() * 0.5 else '#16241d')
        ax.text(j, i + 0.22, notas[i][j], ha='center', fontsize=10, style='italic',
                color='white' if cm[i, j] > cm.max() * 0.5 else '#5f6f67')
plt.title('Matriz de confusión'); plt.tight_layout(); plt.show()

print(f'Falsos positivos (dilución): {cm[0,1]}   Falsos negativos (pérdida): {cm[1,0]}')
print(f'Precisión: {precision_score(y, pred):.3f}   Recall: {recall_score(y, pred):.3f}   F1: {f1_score(y, pred):.3f}')
```

![Matriz de confusión](/assets/articles/mineral-esteril/matriz_confusion.png)

La matriz es `[[132, 8], [19, 81]]`: 8 falsos positivos (**dilución**) y 19 falsos negativos (**pérdida**). La **precisión** es 0.910 (de lo clasificado como mineral, el 91% lo es de verdad), el **recall** es 0.810 (se recupera el 81% del mineral real) y el **F1** es 0.857. Con el umbral por defecto, el modelo es conservador al declarar mineral: peca más de perder mineral que de diluir.

### 7.2) Curva ROC y AUC

La curva ROC recorre todos los umbrales posibles y grafica la tasa de verdaderos positivos contra la de falsos positivos. El área bajo la curva (**AUC**) resume la capacidad discriminativa del modelo en un solo número, independiente del umbral elegido.

```python
proba = modelo.predict_proba(Xs)[:, 1]
fpr, tpr, _ = roc_curve(y, proba)
auc = roc_auc_score(y, proba)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(fpr, tpr, color='#0f766e', lw=2.5, label=f'Modelo (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#94a3a0', label='Azar (AUC = 0.5)')
ax.set_xlabel('Tasa de falsos positivos'); ax.set_ylabel('Tasa de verdaderos positivos')
ax.set_title('Curva ROC'); ax.legend(loc='lower right'); plt.tight_layout(); plt.show()
```

![Curva ROC](/assets/articles/mineral-esteril/curva_roc.png)

El **AUC es 0.935**, muy por encima del 0.5 del azar: el modelo separa las clases con solidez. Un AUC de 0.935 significa que, tomada una muestra mineral y una estéril al azar, el modelo asigna mayor probabilidad de mineral a la correcta el 93.5% de las veces.

### 7.3) Validación cruzada

El AUC in-sample puede ser optimista, porque se evalúa sobre los mismos datos con que se entrenó. La validación cruzada estratificada (k = 5) estima el desempeño en datos no vistos, preservando la proporción de clases en cada partición.

```python
cv = cross_val_score(LogisticRegression(), Xs, y,
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA), scoring='roc_auc')
print(f'AUC por fold: {np.round(cv, 3)}')
print(f'AUC 5-fold CV: {cv.mean():.3f} +/- {cv.std():.3f}')
```

El **AUC 5-fold es 0.929 ± 0.043**, prácticamente igual al valor in-sample (0.935). La cercanía entre ambos y la baja dispersión entre folds confirman que el modelo es estable y generaliza bien: no hay sobreajuste apreciable.


<a id="uso" class="anchor-clean"></a>
## 8) Uso operativo: el umbral y el costo del error

### 8.1) Barrido de umbral: dilución frente a pérdida

El umbral por defecto (0.5) equilibra ambos errores por igual. Pero en operación el costo de diluir y el de perder mineral rara vez son iguales. Barremos el umbral de decisión y contamos, en cada punto, cuántos falsos positivos (dilución) y cuántos falsos negativos (pérdida) se producen.

```python
umbrales = np.linspace(0.15, 0.85, 15)
fp = np.array([int(((y == 0) & (proba >= t)).sum()) for t in umbrales])   # dilución
fn = np.array([int(((y == 1) & (proba < t)).sum()) for t in umbrales])    # pérdida

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(umbrales, fp, color='#b4312a', lw=2, marker='o', ms=4, label='Estéril a planta (dilución)')
ax.plot(umbrales, fn, color='#0f766e', lw=2, marker='s', ms=4, label='Mineral botado (pérdida)')
ax.axvline(0.5, ls='--', color='#94a3a0'); ax.set_xlabel("Umbral para declarar 'mineral'")
ax.set_ylabel('N° de errores'); ax.set_title('Compensación dilución vs pérdida'); ax.legend()
plt.tight_layout(); plt.show()
```

![Compensación dilución vs pérdida](/assets/articles/mineral-esteril/compensacion_umbral.png)

Las dos curvas se cruzan: bajar el umbral captura más mineral (menos pérdida) a costa de admitir más estéril (más dilución), y subirlo hace lo contrario. No existe un umbral que minimice ambos errores a la vez. La elección depende de cuánto cuesta cada uno.

### 8.2) Elegir el umbral por costo

Si asignamos un costo relativo a cada error, el umbral óptimo es el que **minimiza el costo total** `C = c_dil · FP + c_perd · FN`. Por ejemplo, si perder mineral cuesta el doble que diluir:

```python
c_dil, c_perd = 1.0, 2.0    # perder mineral cuesta el doble que diluir
costo = c_dil * fp + c_perd * fn
i_opt = int(np.argmin(costo))
tabla = pd.DataFrame({'umbral': umbrales.round(2), 'dilucion_FP': fp, 'perdida_FN': fn,
                      'costo_total': costo.round(1)})
print(tabla.to_string(index=False))
print(f'\nUmbral óptimo para c_perdida/c_dilucion = {c_perd/c_dil:.0f}: {umbrales[i_opt]:.2f}')
```

Con `c_perdida = 2 · c_dilucion`, el umbral óptimo es **0.40**, por debajo del 0.5 por defecto. La tabla lo resume:

| Umbral | Dilución (FP) | Pérdida (FN) | Costo total |
|---|---|---|---|
| 0.30 | 30 | 12 | 54.0 |
| 0.35 | 18 | 14 | 46.0 |
| **0.40** | **11** | **17** | **45.0** |
| 0.45 | 10 | 18 | 46.0 |
| 0.50 | 8 | 19 | 46.0 |
| 0.55 | 6 | 24 | 54.0 |

El umbral óptimo se **corre** según la relación de costos: si la dilución es cara, sube; si perder metal duele más, baja. El modelo aporta la evidencia cuantitativa; la elección del punto es una **decisión de ingeniería**.


<a id="conclusiones" class="anchor-clean"></a>
## 9) Conclusiones operativas

### 9.1) Desempeño del clasificador

Una regresión logística con dos variables geoquímicas (**Cu y Mo**) reproduce la clasificación mineral / estéril con **AUC ≈ 0.93** y **~89% de acierto**, un desempeño confirmado por validación cruzada (AUC 5-fold = 0.929 ± 0.043). La cercanía entre el AUC in-sample y el de validación indica que el modelo generaliza sin sobreajuste.

### 9.2) Interpretación geoquímica

El modelo **recupera la dirección del *ground truth***: el coeficiente de Cu (+2.99) domina al de Mo (+0.88), consistente con la geoquímica de un pórfido de Cu-Mo donde el cobre es el metal de interés y el molibdeno actúa como co-indicador. La frontera de decisión aprendida separa correctamente el núcleo mineralizado (alto Cu y Mo) de la periferia estéril.

### 9.3) Los dos errores no cuestan igual

La **matriz de confusión** separa con claridad los dos errores mineros: falsos positivos (8) son dilución, falsos negativos (19) son pérdida. Con el umbral por defecto el modelo es conservador al declarar mineral. El **umbral no es fijo**: se elige minimizando el costo total según la relación dilución / pérdida del proyecto. En el ejemplo, con la pérdida costando el doble que la dilución, el óptimo baja a 0.40. La técnica aporta la evidencia; el criterio minero toma la decisión.

### 9.4) Recomendaciones para implementación en campo

- Calibrar el pXRF contra ensayos de laboratorio antes de confiar en sus lecturas de Cu y Mo, y re-verificar periódicamente.
- Re-entrenar el modelo por zona y campaña: los coeficientes son específicos del sitio y del rango de leyes muestreado.
- Fijar el umbral de decisión a partir de un análisis de costos de dilución y pérdida propio del proyecto, no dejarlo en 0.5 por defecto.
- Mantener al geólogo en el circuito: el modelo acelera y estandariza la clasificación, pero no sustituye la supervisión experta, sobre todo en muestras de borde.

### 9.5) Trabajo futuro

- Incorporar más variables geoquímicas (As, Au, Ag, S) y de alteración para robustecer la clasificación.
- Comparar con modelos no lineales (Random Forest, gradient boosting) que capturen fronteras curvas.
- Calibrar las probabilidades (Platt, isotónica) para que el umbral por costo sea aún más fiable.
- Integrar la clasificación en un flujo de despacho en tiempo real con el pXRF en el frente.


<a id="refs" class="anchor-clean"></a>
## 10) Referencias

<div class="references" markdown="1">

Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). **Applied Logistic Regression** (3rd ed.). Wiley. Texto de referencia sobre regresión logística: formulación, interpretación de odds ratios, diagnóstico y validación del modelo.

Lane, K. F. (1988). **The Economic Definition of Ore: Cut-off Grades in Theory and Practice.** Mining Journal Books. Obra clásica sobre la definición económica de la ley de corte y el balance entre dilución y pérdida en el destino de la roca.

Sinclair, A. J., & Blackwell, G. H. (2002). **Applied Mineral Inventory Estimation.** Cambridge University Press. Fundamentos de estimación de recursos, control de leyes y clasificación de material en minería.

Sillitoe, R. H. (2010). **Porphyry Copper Systems.** Economic Geology, 105(1), 3–41. Síntesis de la geología de los sistemas porfídicos de cobre, incluyendo la zonación de Cu y Mo desde el núcleo mineralizado hacia la periferia.

Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python.** Journal of Machine Learning Research, 12, 2825–2830. Descripción de la librería usada para el ajuste, la validación cruzada y las métricas de clasificación de este tutorial.

Fawcett, T. (2006). **An introduction to ROC analysis.** Pattern Recognition Letters, 27(8), 861–874. Referencia sobre la curva ROC y el AUC como métrica de desempeño de clasificadores binarios.

</div>
