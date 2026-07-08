---
layout: article
title: "Mapa de isolíneas de PPV sobre el plano de mina"
subtitle: "Cartografía de la vibración por voladura y evaluación de cumplimiento normativo, con matplotlib y plotly."
date: 2026-07-03

# Hero + card
cover: "/assets/articles/ppv-isolineas/cover.jpg"
hero: "/assets/articles/ppv-isolineas/hero.jpg"
card: /assets/articles/ppv-isolineas/card.jpg

# Metadatos / filtros
series: "Vibraciones"
series_order: 2
tags: [Python, Blasting, PPV, Matplotlib, Plotly]
reading_time: "22 min"

# Resumen destacado
summary: >
  <p>Segunda parte de la serie de vibraciones. Tomamos la <strong>ley USBM</strong> calibrada en la primera parte (<code>PPV = 1065 · SD<sup>-1.6185</sup></code>) y la llevamos al plano de mina: un <strong>campo de PPV</strong> contorneado en <strong>isolíneas</strong>, con su <strong>frontera de cumplimiento</strong> normativo y las <strong>reglas de carga por retardo</strong>, con Matplotlib y una versión interactiva en Plotly. Cerramos contrastando el modelo homogéneo con un <strong>campo realista</strong> (heterogeneidad, direccionalidad y topografía).</p>
  <p></p>
  <div class="ppv-note">El campo espacial de PPV: <strong>PPV(x,y) = K · W<sup>β/2</sup> · D(x,y)<sup>-β</sup></strong></div>

# Índice de contenidos
contents:
  - { anchor: "#contexto", title: "La vibración como problema espacial" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos: el plano de mina" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#mapa", title: "Mapa de isolíneas (Matplotlib)" }
  - { anchor: "#receptores", title: "Evaluación de receptores" }
  - { anchor: "#realista", title: "El campo realista (anisotropía)" }
  - { anchor: "#interactivo", title: "Mapa interactivo (Plotly)" }
  - { anchor: "#reglas", title: "Reglas de campo" }
  - { anchor: "#conclusiones", title: "Conclusiones operativas" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Entrada", body: "Ley del sitio K = 1065, β = 1.6185 (Predicción de vibraciones) + carga por retardo." }
  - { title: "Salida", body: <div class="ppv-card__value">Isolíneas PPV(x,y)</div> }
  - { title: "Stack", body: "Python · NumPy · pandas · Matplotlib · Plotly" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/ppv-isolineas/notebooks/ppv-isolineas.ipynb" }
  - { type: "data",     label: "Polígono de disparo (CSV)", url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/ppv-isolineas/data/raw/blast_polygon.csv" }
  - { type: "data",     label: "Receptores (CSV)",  url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/ppv-isolineas/data/raw/receptores.csv" }
  - { type: "repo",     label: "Repositorio",       url: "https://github.com/nrgarridoa/talleres-mineria-python" }

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

<a id="contexto" class="anchor-clean"></a>
## 1) La vibración como problema espacial

En la [primera parte de esta serie](/articles/vibraciones/) calibramos la **ley de atenuación USBM** `PPV = K · SD⁻ᵝ` sobre una campaña de monitoreo y estimamos los parámetros del sitio: **K = 1065 mm/s** y **β = 1.6185**, con R² = 0.956 en escala log-log. Ese modelo es **puntual**: entrega la PPV en función de la distancia escalada, sin ubicarla en el espacio.

El cumplimiento de vibraciones, en cambio, es un problema **espacial y direccional**. Antes de disparar, ingeniería de perforación y voladura debe verificar que cada **receptor sensible** quede por debajo de su límite regulatorio, y ese problema tiene tres rasgos que una evaluación puntual no captura:

- **Es multi-receptor y multi-límite.** Un mismo disparo expone simultáneamente a estructuras distintas (una vivienda, una caseta industrial, el portal de una labor, la cresta de un talud instrumentado), cada una con un **límite normativo diferente** (12.5 mm/s residencial, 50.8 mm/s industrial, criterios geotécnicos) y a distancias y azimuts distintos del disparo.
- **Es un campo, no un punto.** La excedencia se **localiza**: hay una región del plano donde la carga supera el límite y otra donde cumple. Cartografiar esa región es más informativo que una tabla de distancias, y liga la vibración con la **carga operante por retardo** en el diseño del disparo.
- **Es incierto y anisótropo.** La atenuación real no es igual en toda dirección (geología, topografía, orientación de la cara libre). El mapa debe reflejar tanto la **dispersión estadística** del ajuste como la **heterogeneidad del sitio**.

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} Lo que resuelve este mapa</div>
  Cartografiar la PPV sobre todo el plano permite responder de un vistazo:
  <ul>
    <li>¿Dónde, exactamente, una carga determinada excede el límite de vibración de cada estructura?</li>
    <li>¿Qué receptor es la <em>restricción vinculante</em> del diseño, y por qué?</li>
    <li>¿Cuál es la carga máxima por retardo que mantiene a todos en cumplimiento, incluso en el escenario conservador?</li>
  </ul>
</div>

La herramienta que responde a esto es un mapa de **isolíneas de PPV** (curvas de igual velocidad de partícula), análogo a un mapa topográfico pero de vibración. Sobre él trazamos la **isolínea del límite normativo** de cada receptor: dentro hay excedencia, fuera hay cumplimiento. Convertimos así una ley empírica en una **herramienta de planificación y cumplimiento** trazable.

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Llevar la ley de atenuación ajustada al plano de mina y traducirla en decisiones de carga por retardo, con visualización estática e interactiva.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Construir el campo espacial de PPV a partir de la ley USBM y la distancia al polígono del disparo.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Trazar isolíneas de PPV y la frontera de cumplimiento normativo con Matplotlib.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Diferenciar el mapa esperado (media) del conservador (P95) usando la dispersión de <em>Predicción de vibraciones con Scikit-learn</em>.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Publicar un mapa interactivo con Plotly y derivar reglas de distancia mínima y carga máxima por retardo.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) El campo espacial de PPV

La ley USBM relaciona la PPV con la **distancia escalada** `SD = D / √W`, donde `D` es la distancia al disparo (m) y `W` la carga máxima por retardo (kg). Sustituyendo en `PPV = K · SD⁻ᵝ`:

<div class="formula">
  <code>PPV(x,y) = K · (D(x,y) / √W)<sup>-β</sup> = K · W<sup>β/2</sup> · D(x,y)<sup>-β</sup></code>
</div>

Fijada la carga `W`, la PPV en cada punto del plano depende únicamente de la **distancia a la voladura** `D(x,y)`. Evaluando esta expresión sobre una malla 2D obtenemos un **campo escalar** de PPV, que luego contorneamos en isolíneas.

### 2.2) Distancia al polígono del disparo, no a un punto

Un disparo de producción no es un punto: es un **polígono de taladros** (el *round*). Medir la distancia al centroide sobreestima la PPV en el campo cercano. Por eso usamos la **distancia al borde más cercano del polígono**: así las isolíneas **abrazan la forma real del round** cerca del disparo y solo tienden a circunferencias a lo lejos. El interior del polígono se enmascara, porque la fórmula diverge cuando `D → 0`.

### 2.3) Incertidumbre: mapa medio y mapa P95

[Predicción de vibraciones con Scikit-learn](/articles/vibraciones/) mostró que los residuos del ajuste son **normales en log₁₀** con desviación σ ≈ 0.110. La PPV real de un disparo se dispersa alrededor de la predicción media, así que para diseño **conservador** no se usa la media sino la **banda superior de predicción**. El percentil 95 unilateral es un factor multiplicativo constante sobre el mapa medio:

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} El factor conservador</div>
  <div class="formula">
    <code>PPV<sub>95</sub>(x,y) = PPV<sub>media</sub>(x,y) · 10<sup>1.645 · σ</sup> ≈ 1.52 · PPV<sub>media</sub>(x,y)</code>
  </div>
  <p>Trabajamos <strong>los dos mapas</strong>: el esperado y el conservador. La diferencia entre ambos es, con frecuencia, lo que decide si un receptor cumple o no.</p>
</div>

### 2.4) Límites normativos y frontera de cumplimiento

| Norma / Referencia | Límite PPV (mm/s) | Aplicación |
|---|---|---|
| USBM RI 8507 (conservador) | 12.5 | Estructuras residenciales |
| USBM RI 8507 (general) | 50.8 | Estructuras comerciales/industriales |
| NTP 350.004 (Perú) | 50 | Vibraciones por voladura |
| DIN 4150-3 (Alemania) | 5 – 50 | Según estructura y frecuencia |

La **isolínea del límite** de cada receptor divide el plano en zona de cumplimiento y zona de excedencia. Por la forma potencial de la ley, esa isolínea (para fuente puntual) es una circunferencia de radio `D_lim = √W · (K / PPV_lim)^(1/β)`.

### 2.5) Reglas de campo en forma cerrada

De la misma ley se despejan dos reglas de uso directo en el frente:

<div class="formula">
  <code>D<sub>min</sub> = √W · (K / PPV<sub>lim</sub>)<sup>1/β</sup></code> &nbsp;&nbsp;·&nbsp;&nbsp; <code>W<sub>max</sub> = (PPV<sub>lim</sub> / K)<sup>2/β</sup> · D²</code>
</div>

La primera da la **distancia mínima segura** para una carga dada; la segunda, la **carga máxima por retardo** admisible para un receptor a distancia `D`.

### 2.6) Limitaciones

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} Alcance del modelo</div>
  <ul>
    <li>El mapa base hereda los supuestos del modelo USBM: carga equivalente por retardo, propagación radial, macizo homogéneo. Por eso sus isolíneas son <strong>suaves y casi concéntricas</strong>: es fiel al modelo, no una simplificación del dibujo.</li>
    <li>Añade el supuesto de <strong>fuente equivalente</strong> distribuida en un polígono plano.</li>
    <li>En un sitio real los contornos son <strong>irregulares y lobulados</strong> por heterogeneidad geológica, direccionalidad de la voladura y topografía. En la <a href="#realista">Sección 7</a> integramos esos tres efectos.</li>
    <li>Es una herramienta de planificación y cumplimiento, no un sustituto del monitoreo instrumental.</li>
  </ul>
</div>


<a id="datos" class="anchor-clean"></a>
## 3) Datos: el plano de mina

### 3.1) Geometría del escenario

Definimos un sector de rajo sintético en una **grilla local en metros**: el contorno del rajo, el **polígono de disparo** con su carga por retardo, y cuatro **receptores** con sus límites normativos. Todo es determinístico y reproducible.

| Elemento | Valor |
|---|---|
| Extensión del plano | 0 – 1050 m (X) × 0 – 780 m (Y) |
| Carga por retardo `W` | 500 kg (disparo de producción) |
| Ley del sitio (Predicción de vibraciones) | K = 1065.44 mm/s · β = 1.6185 · σ<sub>log</sub> = 0.110 |

### 3.2) Receptores y límites

| Receptor | Tipo | Límite (mm/s) |
|---|---|---|
| Poblado | Residencial (USBM conservador) | 12.5 |
| Caseta de operaciones | Industrial (USBM general) | 50.8 |
| Portal / bocamina | Estructura / labor | 50.0 |
| Cresta de talud | Geotécnico (tolerancia alta) | 100.0 |


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

### 4.1) Librerías y parámetros heredados

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
import plotly.graph_objects as go

# Ley del sitio heredada del Parte 1 (ajuste USBM)
K_SITE    = 1065.44     # mm/s
BETA      = 1.6185      # exponente de atenuación
SIGMA_LOG = 0.110       # desviación de residuos en log10 (Parte 1)
F_P95     = 10 ** (1.645 * SIGMA_LOG)   # factor medio -> P95 (~1.52)
W_DELAY   = 500.0       # carga máxima por retardo (kg)
```

### 4.2) Geometría del plano y receptores

```python
# Polígono de disparo (round), coords locales en m
blast = np.array([(240, 310), (370, 300), (380, 400), (255, 405)], dtype=float)

# Receptores: nombre, coords, límite normativo (mm/s)
receptores = pd.DataFrame([
    ('Poblado',               720, 610, 12.5, 'Residencial'),
    ('Caseta de operaciones',  60, 120, 50.8, 'Industrial'),
    ('Portal / bocamina',     640, 150, 50.0, 'Estructura'),
    ('Cresta de talud',       230, 530, 100.0, 'Geotecnico'),
], columns=['receptor', 'x', 'y', 'limite_mm_s', 'tipo'])
```

### 4.3) Distancia al polígono del disparo (vectorizada)

Para cada punto se calcula la distancia mínima a los **segmentos** del polígono, operando sobre toda la malla a la vez.

```python
def dist_a_poligono(px, py, poly):
    """Distancia de cada punto (px, py) al borde del polígono."""
    P = np.column_stack([px.ravel(), py.ravel()])
    best = np.full(P.shape[0], np.inf)
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(P - proj, axis=1))
    return best.reshape(px.shape)
```

### 4.4) Campo de PPV sobre la malla

Evaluamos `PPV(x,y) = K · W^(β/2) · D^(-β)` en cada nodo. Enmascaramos el interior del polígono y ponemos un piso físico de 8 m a la distancia (nadie mide a 0 m del disparo). El campo conservador es el medio por el factor P95.

```python
NX, NY = 420, 320
xx = np.linspace(0, 1050, NX)
yy = np.linspace(0, 780, NY)
XX, YY = np.meshgrid(xx, yy)

D = dist_a_poligono(XX, YY, blast)
dentro = Path(blast).contains_points(np.column_stack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
D = np.where(dentro, np.nan, np.maximum(D, 8.0))

def campo_ppv(D, W=W_DELAY):
    return K_SITE * (W ** (BETA / 2)) * D ** (-BETA)

PPV_media = campo_ppv(D)
PPV_p95   = PPV_media * F_P95
```


<a id="mapa" class="anchor-clean"></a>
## 5) Mapa de isolíneas con Matplotlib

### 5.1) Mapa medio con frontera de cumplimiento

Rellenamos el campo con una escala de color, trazamos isolíneas etiquetadas en mm/s y resaltamos la **isolínea de 12.5 mm/s** (límite residencial) como frontera de cumplimiento. Superponemos el polígono de disparo, el contorno del rajo y los receptores.

![Mapa de isolíneas de PPV media sobre el plano de mina, con la frontera de cumplimiento de 12.5 mm/s en negro y los cuatro receptores](/assets/articles/ppv-isolineas/fig-mapa-medio.png)

La lectura es inmediata: la **línea negra** (12.5 mm/s) encierra la zona donde una estructura residencial excedería el límite. En el mapa medio, todos los receptores caen fuera de ella. Pero la media no es el criterio de diseño.

### 5.2) Mapa medio vs P95: la frontera se expande

El mapa conservador desplaza la frontera de cumplimiento hacia afuera. Comparamos la isolínea de 12.5 mm/s en ambos escenarios.

![Comparación de la frontera de cumplimiento de 12.5 mm/s en el escenario medio y en el P95: el poblado queda entre ambas curvas](/assets/articles/ppv-isolineas/fig-media-p95.png)

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} El punto que decide el diseño</div>
  El <strong>poblado</strong> queda fuera de la frontera media (línea negra) pero <strong>dentro de la frontera P95</strong> (línea roja): en el escenario esperado cumple, en el conservador excede. Diseñar con la media subestima el riesgo.
</div>


<a id="receptores" class="anchor-clean"></a>
## 6) Evaluación de receptores

Para cada receptor calculamos su distancia al disparo y su PPV media y P95, contra el límite propio.

```python
filas = []
for _, r in receptores.iterrows():
    Dr = dist_a_poligono(np.array([[r.x]], float), np.array([[r.y]], float), blast)[0, 0]
    pm = campo_ppv(np.array(Dr)).item()
    pp = pm * F_P95
    filas.append({'receptor': r.receptor, 'D_m': round(Dr), 'PPV_media': round(pm, 1),
                  'PPV_P95': round(pp, 1), 'limite': r.limite_mm_s,
                  'cumple_media': 'OK' if pm < r.limite_mm_s else 'EXCEDE',
                  'cumple_P95':   'OK' if pp < r.limite_mm_s else 'EXCEDE'})
pd.DataFrame(filas)
```

| Receptor | D (m) | PPV media | PPV P95 | Límite | Media | P95 |
|---|---:|---:|---:|---:|:---:|:---:|
| Poblado | 400 | 10.0 | **15.2** | 12.5 | OK | **EXCEDE** |
| Caseta de operaciones | 262 | 19.9 | 30.2 | 50.8 | OK | OK |
| Portal / bocamina | 309 | 15.2 | 23.1 | 50.0 | OK | OK |
| Cresta de talud | 127 | 63.7 | 96.6 | 100.0 | OK | OK |

El receptor de **mayor PPV** es la cresta de talud (a 127 m del disparo, 64 mm/s), pero su tolerancia geotécnica (100 mm/s) la mantiene holgada. El **vinculante** es el poblado: la PPV más baja de la tabla, pero con el límite más estricto, y en el escenario P95 lo supera.

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} La lección operativa</div>
  La restricción de diseño no la fija el punto más cercano ni el de mayor vibración, sino el de <strong>límite más exigente frente a su exposición</strong>. El mapa lo hace evidente.
</div>


<a id="realista" class="anchor-clean"></a>
## 7) El campo realista: heterogeneidad, direccionalidad y topografía

El mapa base supone un macizo homogéneo y una fuente radial, y por eso sus isolíneas salen suaves. Un campo real es **anisótropo**. Modelamos esa realidad con un **factor de sitio** `M(x,y)` que multiplica la PPV media e integra tres efectos, cada uno documentado y reproducible:

<div class="formula">
  <code>PPV<sub>real</sub>(x,y) = PPV<sub>media</sub>(x,y) · M(x,y)</code> &nbsp;&nbsp;,&nbsp;&nbsp; <code>M = M<sub>dir</sub> · M<sub>geo</sub> · M<sub>topo</sub></code>
</div>

- **Direccionalidad** `M_dir`: la cara libre y la secuencia de iniciación enfocan energía en una dirección. Se modela como `1 + A·cos(θ − θ₀)`, con `θ₀` el azimut de la cara libre (aquí, hacia el poblado) y `A = 0.22` (±22 % entre el frente y la espalda del disparo).
- **Heterogeneidad geológica** `M_geo`: K y β varían en el espacio (litologías, fracturamiento, agua). Se representa con un campo suave de baja frecuencia (semilla fija) más una **zona más fracturada** que amplifica hacia el poblado.
- **Topografía** `M_topo`: amplificación en una banda a lo largo de la cresta del rajo (+15 %).

```python
cen = blast.mean(axis=0)                        # centroide del disparo
th0 = np.arctan2(610 - cen[1], 720 - cen[0])    # azimut disparo -> poblado (cara libre)
A_DIR = 0.22

def modificador(X, Y):
    m_dir = 1 + A_DIR * np.cos(np.arctan2(Y - cen[1], X - cen[0]) - th0)   # direccionalidad
    m_geo = np.ones_like(X, dtype=float)                                   # heterogeneidad
    rng = np.random.default_rng(11)
    for _ in range(5):
        ax_, ay_ = rng.uniform(0.4, 1.4, 2) / 1000; ph = rng.uniform(0, 2 * np.pi, 2)
        m_geo += rng.uniform(0.04, 0.08) * np.sin(ax_ * X + ph[0]) * np.cos(ay_ * Y + ph[1])
    m_geo *= 1 + 0.15 * np.exp(-(((X - 720) / 380) ** 2 + ((Y - 600) / 320) ** 2))
    pd_ = np.min(np.sqrt((X[..., None] - pit[:, 0]) ** 2 + (Y[..., None] - pit[:, 1]) ** 2), axis=-1)
    m_topo = 1 + 0.15 * np.exp(-(pd_ / 50) ** 2)                           # topografía
    return m_dir * m_geo * m_topo

PPV_real = PPV_media * modificador(XX, YY)
```

![Comparación lado a lado: a la izquierda el modelo homogéneo con isolíneas suaves y concéntricas; a la derecha el campo realista con isolíneas lobuladas que se abultan hacia el poblado](/assets/articles/ppv-isolineas/fig-campo-real.png)

Las isolíneas del campo realista son **lobuladas**: la frontera de 12.5 mm/s se **abulta hacia el poblado** (dirección de la cara libre y zona más fracturada) y se **contrae hacia la caseta** (a la espalda del disparo). El punto de mayor PPV deja de coincidir con el disparo y la forma abandona la simetría radial.

Reevaluando los receptores bajo el campo realista, el efecto endurece la conclusión anterior:

| Receptor | PPV homogénea | Factor M | PPV realista | Límite | Cumple |
|---|---:|---:|---:|---:|:---:|
| Poblado | 10.0 | 1.67 | **16.7** | 12.5 | **EXCEDE** |
| Caseta de operaciones | 19.9 | 0.82 | 16.4 | 50.8 | OK |
| Portal / bocamina | 15.2 | 1.34 | 20.4 | 50.0 | OK |
| Cresta de talud | 63.7 | 1.22 | 77.6 | 100.0 | OK |

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} La anisotropía cambia el veredicto</div>
  Con la direccionalidad hacia el poblado más la zona fracturada, su PPV sube de 10.0 a <strong>16.7 mm/s</strong> y <strong>excede el límite ya en la media</strong>, sin necesidad del P95. El mapa homogéneo <strong>subestima</strong> el riesgo en la dirección crítica: cuando el sitio es anisótropo, el receptor a favor de la cara libre manda todavía más, y el diseño conservador deja de ser opcional.
</div>

Los parámetros de `M(x,y)` son ilustrativos. En un caso real se calibran con registros de geófonos distribuidos y, típicamente, interpolando (kriging) las mediciones de varias voladuras: el mapa base es para planificación; el campo realista exige datos de sitio.


<a id="interactivo" class="anchor-clean"></a>
## 8) Mapa interactivo con Plotly

La versión interactiva permite pasar el cursor por cualquier punto del plano y leer la PPV, hacer zoom y aislar receptores. Es la misma data del mapa estático, ahora explorable.

<div class="ppv-plot-embed" style="position:relative; width:100%; border:1px solid rgba(148,163,184,.25); border-radius:12px; overflow:hidden; margin:1.2rem 0;">
  <iframe src="/assets/articles/ppv-isolineas/ppv_interactivo.html" title="Mapa interactivo de PPV sobre el plano de mina" loading="lazy" style="width:100%; height:680px; border:0; display:block;"></iframe>
</div>

El código que la genera es una figura de Plotly con una traza `Contour` para el campo de PPV, una segunda `Contour` aislando la isolínea de 12.5 mm/s, y trazas `Scatter` para el polígono de disparo y los receptores. Se exporta como fragmento HTML autocontenido:

```python
fig.write_html('ppv_interactivo.html', include_plotlyjs='cdn', full_html=True)
```


<a id="reglas" class="anchor-clean"></a>
## 9) Reglas de campo

### 9.1) Distancia mínima segura

Dado un límite y una carga por retardo, `D_min = √W · (K / PPV_lim)^(1/β)`. Es el radio de la isolínea del límite en el mapa medio.

| W (kg) | PPV < 50.8 | PPV < 25 | PPV < 12.5 | PPV < 5 |
|---:|---:|---:|---:|---:|
| 100 | 66 | 102 | 156 | 275 |
| 250 | 104 | 161 | 246 | 434 |
| 500 | 147 | 227 | 349 | 614 |
| 750 | 180 | 278 | 427 | 752 |
| 1000 | 207 | 321 | 493 | 868 |

<p style="text-align:center; color:#94a3b8; font-size:.85rem; margin-top:-.4rem;">Distancia mínima segura D<sub>min</sub> (m) por carga por retardo y límite de PPV.</p>

### 9.2) Carga máxima por retardo por receptor

`W_max = (PPV_lim / K)^(2/β) · D²`. Es la regla que usa el supervisor: dada la distancia al receptor y su límite, la carga máxima que puede detonar por retardo. La calculamos en el escenario medio y en el conservador P95.

| Receptor | D (m) | Límite | W_max media (kg) | W_max P95 (kg) |
|---|---:|---:|---:|---:|
| Poblado | 400 | 12.5 | 657 | **393** |
| Caseta de operaciones | 262 | 50.8 | 1594 | 953 |
| Portal / bocamina | 309 | 50.0 | 2177 | 1301 |
| Cresta de talud | 127 | 100.0 | 873 | 522 |

La carga máxima por retardo que satisface a **todos** los receptores en el escenario conservador la fija el vinculante: **393 kg**, el poblado.

### 9.3) Rediseño: un disparo que cumple al P95

El disparo actual (500 kg/retardo) satisface a todos en la media, pero el poblado excede al P95. Aplicamos una carga de diseño **con margen** sobre ese límite (350 kg) y volvemos a mapear el escenario conservador.

![Mapa P95 del disparo rediseñado a 350 kg por retardo: todos los receptores cumplen, con su PPV anotada](/assets/articles/ppv-isolineas/fig-rediseno.png)

Con 350 kg/retardo la frontera de cumplimiento se contrae lo suficiente para que **el poblado quede fuera incluso en el mapa P95** (11.4 mm/s). El costo es operativo (más retardos, disparos más secuenciados), y esa es la compensación que el mapa pone sobre la mesa con evidencia.


<a id="conclusiones" class="anchor-clean"></a>
## 10) Conclusiones operativas

- La ley USBM calibrada en la primera parte (**K = 1065, β = 1.6185**) se lleva al plano como un **campo de PPV** y se contornea en isolíneas: el mapa muestra de un vistazo dónde una carga cumple y dónde excede.
- Usar la **distancia al polígono del disparo** (no al centroide) hace que las isolíneas cercanas sigan la forma real del round; a lo lejos tienden a circunferencias.
- El **mapa P95** (× 1.52 sobre la media, derivado de σ = 0.110 de la primera parte) desplaza la frontera de cumplimiento hacia afuera. Diseñar con la media subestima el riesgo; el criterio conservador usa el P95.
- La restricción de diseño la fija el **receptor vinculante**: en este caso el poblado, con la PPV más baja pero el límite más estricto. No es el más cercano ni el de mayor vibración.
- El **campo realista** (heterogeneidad + direccionalidad + topografía) deforma las isolíneas y, en la dirección de la cara libre, lleva al poblado a exceder ya en la media (16.7 mm/s): el modelo homogéneo subestima el riesgo donde la geometría del disparo enfoca la energía.
- Bajar la carga de **500 a 350 kg/retardo** devuelve el cumplimiento del poblado en el escenario P95. El mapa convierte la ley de atenuación en una **decisión de carga por retardo** auditable.

El flujo es reproducible de principio a fin y se conecta con el monitoreo real: cada campaña recalibra K y β (primera parte), el factor de sitio M(x,y) y actualiza el mapa.


<a id="refs" class="anchor-clean"></a>
## 11) Referencias

<div class="references" markdown="1">

Siskind, D. E., Stagg, M. S., Kopp, J. W., & Dowding, C. H. (1980). **Structure response and damage produced by ground vibration from surface mine blasting.** U.S. Bureau of Mines, Report of Investigations RI 8507. Estableció las relaciones empíricas y los límites de PPV por tipo de estructura que aún se usan como referencia normativa.

Dowding, C. H. (1985). **Blast Vibration Monitoring and Control.** Prentice-Hall. Texto de referencia para instrumentación, criterios de daño y control de vibraciones por voladura.

Agrawal, H., & Mishra, A. K. (2019). **Modified scaled distance regression analysis approach for prediction of blast-induced ground vibration in multi-hole blasting.** Journal of Rock Mechanics and Geotechnical Engineering, 11, 202–207. Discute la superposición de ondas de múltiples taladros y su efecto sobre la PPV respecto de la carga equivalente.

Hustrulid, W. (1999). **Blasting Principles for Open Pit Mining.** Balkema. Fundamentos de diseño de voladura en minería a cielo abierto, incluida la gestión de la carga por retardo.

DIN 4150-3 (2016). **Structural vibration — Effects of vibration on structures.** Norma alemana con límites de PPV diferenciados por tipo de estructura y frecuencia dominante.

</div>
