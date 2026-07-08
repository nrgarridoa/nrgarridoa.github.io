---
layout: article
title: "Signature hole: optimizar los retardos para bajar la vibración"
subtitle: "Superposición de ondas para minimizar el PPV, y por qué solo el detonador electrónico lo consigue."
date: 2026-07-05

# Hero + card
cover: "/assets/articles/signature-hole/cover.jpg"
hero: "/assets/articles/signature-hole/hero.jpg"
card: /assets/articles/signature-hole/card.jpg

# Metadatos / filtros
series: "Vibraciones"
series_order: 4
tags: [Python, Vibraciones, Voladura, Retardos, Detonador electrónico, FFT]
reading_time: "24 min"

# Resumen destacado
summary: >
  <p>Cuarta parte de la serie de vibraciones. Con el mismo explosivo, la misma malla y la misma distancia, el PPV puede variar más de diez veces según cómo se secuencien los retardos. Tomamos la onda de un solo taladro (la <strong>signature hole</strong>), la superponemos para toda la voladura, <strong>optimizamos la secuencia</strong> que minimiza el PPV y mostramos por qué solo el <strong>detonador electrónico</strong> puede realizarla.</p>
  <p></p>
  <div class="ppv-note">La vibración total es la suma de la onda de cada taladro: <strong>v(t) = Σ v<sub>sig</sub>(t − t<sub>i</sub>)</strong></div>

# Índice de contenidos
contents:
  - { anchor: "#contexto", title: "El tiempo es una palanca gratis" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos: signature y voladura" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#superposicion", title: "La superposición en acción" }
  - { anchor: "#barrido", title: "Barrido de retardos" }
  - { anchor: "#scatter", title: "El límite real: el scatter" }
  - { anchor: "#comb", title: "Vista en frecuencia: el filtro peine" }
  - { anchor: "#conclusiones", title: "Conclusiones" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Entrada", body: "Signature hole: la onda de un solo taladro." }
  - { title: "Método", body: <div class="ppv-card__value">Superposición</div> }
  - { title: "Stack", body: "Python · NumPy · pandas · Matplotlib" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/signature-hole/notebooks/signature-hole.ipynb" }
  - { type: "data",     label: "Signature CSV",    url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/signature-hole/data/raw/signature_hole.csv" }
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

<a id="contexto" class="anchor-clean"></a>
## 1) El tiempo es una palanca gratis

En las partes anteriores la vibración dependía de la **carga** y la **distancia** ([Predicción de vibraciones con Scikit-learn](/articles/vibraciones/)), de la **posición** en el plano ([Mapa de isolíneas de PPV](/articles/ppv-isolineas/)) y de la **frecuencia** ([Análisis en frecuencia de una onda de voladura](/articles/fft-voladura/)). Falta la palanca más barata de todas: el **tiempo**.

El diseño de voladura suele pensarse en carga por retardo y distancia. Pero la carga cuesta fragmentación y la distancia no se puede mover: el receptor está donde está. El **tiempo de detonación**, en cambio, no cuesta nada. Cambiar la secuencia de retardos no cambia el explosivo ni la malla, y sin embargo puede partir el PPV a la mitad o multiplicarlo por diez.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Una voladura es una secuencia, no una explosión</div>
  Cada taladro detona con milisegundos de diferencia y lanza su propia onda al terreno. Lo que el geófono registra es la <strong>suma</strong> de todas esas ondas, cada una desfasada según su retardo. Controlar ese desfase es controlar la interferencia, y con ella el pico de vibración.
</div>

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Convertir la secuencia de retardos en una decisión cuantitativa: modelar la voladura por superposición y encontrar el timing que minimiza el PPV.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Superponer la signature hole para construir la onda de la voladura completa.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Barrer el retardo y encontrar la secuencia que minimiza el pico de vibración.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Cuantificar el ahorro de PPV al re-temporizar, sin tocar el explosivo.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Mostrar con Monte Carlo por qué el scatter del detonador decide si la optimización se realiza.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) La signature hole

La **signature hole** es la onda de velocidad de partícula que produce **un solo taladro** medido en el receptor. Se obtiene disparando un taladro aislado, o de un modelo. Es la huella del par sitio-carga: una oscilación amortiguada con una frecuencia dominante propia del macizo.

### 2.2) Superposición lineal

Para amplitudes moderadas, el terreno responde de forma **lineal**: la onda de la voladura completa es la **suma** de la signature repetida, una por taladro, cada una desplazada por su tiempo de detonación:

<div class="formula">
  <code>v<sub>total</sub>(t) = Σ<sub>i=1</sub><sup>N</sup> v<sub>sig</sub>(t − t<sub>i</sub>)</code>
</div>

donde `t_i` es el instante en que detona el taladro `i`. Con retardo uniforme `Δ`, `t_i = (i−1)·Δ`. El **PPV** del conjunto es `max|v_total(t)|`, y depende por completo de los `t_i`.

### 2.3) Interferencia constructiva y destructiva

Si el retardo `Δ` es casi cero (o coincide con el **periodo** de la signature), los picos de cada onda se apilan: interferencia **constructiva**, PPV máximo (hasta `N` veces el de un taladro). Si `Δ` acerca los picos a los valles de las ondas vecinas, se cancelan: interferencia **destructiva**, PPV mínimo. Entre ambos extremos, el PPV cambia con el retardo.

### 2.4) El scatter del detonador

El retardo real nunca es exacto. Cada detonador dispara con una **dispersión** alrededor de su tiempo nominal:

| Detonador | Dispersión típica (σ) | Efecto |
|---|---|---|
| Pirotécnico (mecha lenta) | 1 a 3 ms | Borra las cancelaciones finas |
| Electrónico | 0.1 a 0.3 ms | Mantiene la secuencia diseñada |

Una cancelación destructiva requiere alinear los valles con precisión de fracción de milisegundo. Con scatter grande (pirotécnico), esa precisión se pierde y la optimización se evapora. El detonador electrónico es lo que convierte el diseño en realidad.


<a id="datos" class="anchor-clean"></a>
## 3) Datos: la signature y la voladura

Modelamos la signature de un taladro como una oscilación amortiguada, y una fila de taladros que detonan con un retardo uniforme.

| Parámetro | Valor | Rol |
|---|---|---|
| Frecuencia de muestreo `fs` | 2000 Hz | Resolver retardos finos (0.5 ms) |
| Frecuencia dominante de la signature | 22 Hz | Periodo de 45 ms |
| Amortiguamiento `τ` | 0.11 s | La onda repica unos 3 ciclos |
| Pico de un taladro | 4 mm/s | Amplitud de la signature |
| Número de taladros `N` | 15 | Una fila de la malla |


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

```python
import numpy as np
import pandas as pd

FS = 2000.0
DUR = 0.9
F0, TAU = 22.0, 0.11          # frecuencia dominante (Hz) y amortiguamiento (s)
A_HOLE = 4.0                   # pico de un taladro (mm/s)
N_HOLES = 15
t = np.arange(0, DUR, 1/FS)

def signature(tt):
    """Onda de velocidad de un solo taladro (oscilación amortiguada)."""
    s = np.zeros_like(tt); m = tt >= 0
    s[m] = A_HOLE * np.exp(-tt[m]/TAU) * np.sin(2*np.pi*F0*tt[m])
    return s

def onda_conjunto(delay_ms, jitter_ms=0.0, rng=None, n=N_HOLES):
    d = delay_ms/1000.0
    total = np.zeros_like(t)
    for i in range(n):
        ti = i*d + (0.0 if jitter_ms == 0 else rng.normal(0, jitter_ms/1000.0))
        total += signature(t - ti)          # superposición
    return total

def ppv_conjunto(delay_ms, **kw):
    return np.max(np.abs(onda_conjunto(delay_ms, **kw)))
```


<a id="superposicion" class="anchor-clean"></a>
## 5) La superposición en acción

Comparamos la signature de un taladro con la onda del conjunto a un retardo típico de 8 ms. Los quince taladros se apilan en un pico mucho mayor que el de uno solo: **3.6 mm/s** (un taladro) frente a **6.9 mm/s** (los quince a 8 ms), y hasta **54 mm/s** si detonaran simultáneos.

![Arriba, la onda de un solo taladro (signature); abajo, la superposición de los quince taladros a 8 ms, con un pico mayor](/assets/articles/signature-hole/fig-superposicion.png)


<a id="barrido" class="anchor-clean"></a>
## 6) Barrido de retardos: la curva que decide

Antes de barrer todo el rango, vale ver la onda completa en tres retardos concretos: casi constructivo, típico y óptimo. La forma de cada traza explica por qué el PPV cambia tanto.

![Tres superposiciones de los mismos quince taladros a 2, 8 y 15 ms: a 2 ms los picos se apilan (26.2 mm/s), a 8 ms el pico baja a 6.9 mm/s, y a 15 ms las ondas quedan escalonadas y el pico cae a 3.7 mm/s, cerca del piso de un solo taladro](/assets/articles/signature-hole/fig-interferencia.png)

Recorremos ahora el retardo entre taladros en el rango operativo (1 a 25 ms) y medimos el PPV del conjunto. La curva cae desde la interferencia constructiva (retardos cortos, las ondas se apilan) hasta la destructiva (el óptimo, las ondas se escalonan).

![PPV del conjunto en función del retardo entre taladros: cae desde más de 45 mm/s a retardos cortos hasta el piso de un solo taladro en el óptimo de 15 ms](/assets/articles/signature-hole/fig-barrido.png)

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} El tiempo, gratis, parte el PPV a la mitad</div>
  Con el mismo explosivo y la misma distancia, el PPV va de <strong>3.6 a más de 45 mm/s</strong> solo cambiando el retardo. Re-temporizar de los 8 ms típicos al óptimo de <strong>15 ms</strong> baja el PPV un <strong>47 %</strong>. En el óptimo los taladros quedan tan escalonados que el conjunto apenas supera a un solo taladro: el piso teórico, disparar de a uno.
</div>


<a id="scatter" class="anchor-clean"></a>
## 7) El límite real: el scatter del detonador

El óptimo supone que cada taladro detona exactamente a su tiempo. En la realidad hay dispersión. Simulamos el PPV en el retardo óptimo con el scatter de un detonador **pirotécnico** (σ = 2.5 ms) y uno **electrónico** (σ = 0.2 ms).

![Distribución del PPV realizado en el retardo óptimo: el detonador electrónico se concentra cerca de 3.8 mm/s, el pirotécnico se dispersa hasta más de 7 mm/s](/assets/articles/signature-hole/fig-scatter.png)

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} Sin precisión de tiempo, no hay optimización</div>
  Con detonador <strong>electrónico</strong> el PPV se mantiene cerca del óptimo (P95 ≈ 3.8 mm/s): la reducción del 47 % se <strong>realiza</strong>. Con <strong>pirotécnico</strong>, el scatter de 2.5 ms borra las cancelaciones finas y el PPV se dispersa hasta <strong>7 mm/s (P95)</strong>, peor que el diseño sin optimizar. El detonador electrónico es lo que convierte el diseño en vibración real más baja.
</div>


<a id="comb" class="anchor-clean"></a>
## 8) Vista en frecuencia: por qué el retardo filtra

El barrido de la Sección 6 responde *cuál* retardo minimiza el PPV, pero no *por qué* ese valor y no otro cercano. La [Parte 3](/articles/fft-voladura/) mostró que toda vibración tiene contenido en frecuencia; la superposición de taladros, vista con Fourier, es literalmente un **filtro peine**: refuerza unas frecuencias y cancela otras según el retardo.

```python
def espectro(x):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1/FS)
    return freqs, np.abs(X)

# Barremos el retardo y guardamos el espectro completo de cada conjunto
delays_fino = np.linspace(1, 25, 220)
espectros = [espectro(onda_conjunto(d))[1] for d in delays_fino]
```

Graficando la magnitud del espectro del conjunto para cada retardo obtenemos un mapa retardo-frecuencia: las franjas diagonales son la firma del filtro peine.

![Mapa retardo-frecuencia del espectro del conjunto: franjas diagonales de refuerzo (verde/amarillo) y cancelación (morado oscuro); la línea de 22 Hz —frecuencia dominante del taladro— cruza una banda de refuerzo a 8 ms y una banda de cancelación muy cerca de 15 ms](/assets/articles/signature-hole/fig-comb.png)

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} El óptimo no es arbitrario</div>
  A <strong>8 ms</strong>, la línea de 22 Hz cruza una banda de <strong>refuerzo</strong>: el retardo típico, sin saberlo, amplifica justo la frecuencia dominante del taladro. A <strong>~15 ms</strong>, esa misma línea cae casi exactamente sobre una banda de <strong>cancelación</strong>. El barrido de PPV (Sección 6) y el mapa espectral llegan al mismo número por caminos distintos: minimizar el PPV en el tiempo equivale a buscar el retardo que cancela la frecuencia donde el taladro concentra su energía.
</div>

Esto también explica por qué el óptimo es una banda angosta y no una meseta: los retardos vecinos (12 ms, 18 ms) caen en otras bandas de cancelación del mismo peine, pero con menos margen, porque el amortiguamiento de la signature reparte algo de energía alrededor de los 22 Hz. El dominio del tiempo (PPV) y el de la frecuencia (cumplimiento normativo de la Parte 3) son, en el fondo, la misma decisión mirada desde dos ángulos.


<a id="conclusiones" class="anchor-clean"></a>
## 9) Conclusiones

- La vibración de una voladura es la **superposición** de la onda de cada taladro desfasada por su retardo. El PPV depende del **tiempo**, no solo de la carga y la distancia.
- Con la misma malla y explosivo, el PPV varía **más de diez veces** con la secuencia: de 3.6 a más de 45 mm/s. El tiempo es la palanca más barata del diseño de voladura.
- Re-temporizar del retardo típico (8 ms) al **óptimo (15 ms)** baja el PPV un **47 %**, sin tocar el explosivo. El piso teórico es el de un solo taladro.
- El **scatter del detonador** es el límite real: el pirotécnico (σ 2.5 ms) borra la optimización (P95 7 mm/s, peor que sin optimizar); solo el **electrónico** (σ 0.2 ms) realiza la reducción.
- Visto en **frecuencia**, el retardo típico (8 ms) refuerza la frecuencia dominante del taladro (22 Hz) y el óptimo (15 ms) la cancela — el mismo resultado que entrega el barrido de PPV, confirmado desde el dominio espectral que abrió la Parte 3.
- El modelo de **signature hole** convierte el diseño de secuencia en una decisión cuantitativa, trazable y verificable antes de disparar.

Una signature medida en campo alimenta la superposición, el barrido encuentra el retardo óptimo, su espectro explica el porqué, y el Monte Carlo del scatter dice si el detonador disponible puede realizarlo.


<a id="refs" class="anchor-clean"></a>
## 10) Referencias

<div class="references" markdown="1">

Anderson, D. A. (2008). **Signature hole blast vibration control - twenty years hence and beyond.** Proceedings of the Annual Conference on Explosives and Blasting Technique, ISEE. Formaliza el modelado por superposición de la onda de un taladro para diseñar secuencias de baja vibración.

Hinzen, K.-G. (1988). **Modelling of blast vibrations.** International Journal of Rock Mechanics and Mining Sciences, 25(6), 439-445. Base física de la superposición lineal de ondas de taladros individuales.

Yang, R., & Scovira, D. S. (2010). **A model to predict blast vibration considering delay time and charge weight scatter.** ISEE. Incorpora el scatter de tiempo y de carga al modelo de superposición, clave para el rol del detonador electrónico.

Siskind, D. E., Stagg, M. S., Kopp, J. W., & Dowding, C. H. (1980). **Structure response and damage produced by ground vibration from surface mine blasting.** U.S. Bureau of Mines, RI 8507.

Dowding, C. H. (1985). **Blast Vibration Monitoring and Control.** Prentice-Hall.

</div>
