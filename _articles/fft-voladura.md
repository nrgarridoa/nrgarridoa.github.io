---
layout: article
title: "Análisis en frecuencia de una onda de voladura"
subtitle: "FFT, filtrado y frecuencia dominante: el cumplimiento depende de la frecuencia, no solo del PPV."
date: 2026-07-05

# Hero + card
cover: "/assets/articles/fft-voladura/cover.jpg"
hero: "/assets/articles/fft-voladura/hero.jpg"
card: /assets/articles/fft-voladura/card.jpg

# Metadatos / filtros
series: "Vibraciones"
tags: [Python, Vibraciones, FFT, Fourier, DIN 4150]
reading_time: "22 min"

# Resumen destacado
summary: >
  <p>Tercera parte de la serie de vibraciones. Tras la <strong>magnitud</strong> (Parte 1) y el <strong>mapa</strong> (Parte 2), llega la <strong>frecuencia</strong>. Tomamos la forma de onda de un sismógrafo, la descomponemos con <strong>Fourier/FFT</strong>, la <strong>filtramos</strong>, identificamos su <strong>frecuencia dominante</strong> y decidimos el cumplimiento contra la curva normativa (DIN 4150-3). El cierre: un evento que por magnitud parece cumplir, pero por su baja frecuencia dominante no cumple.</p>
  <p></p>
  <div class="ppv-note">El PPV dice <strong>cuánto</strong> vibra; la frecuencia dice <strong>cuánto daña</strong>.</div>

# Índice de contenidos
contents:
  - { anchor: "#contexto", title: "PPV vs frecuencia" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos: la forma de onda" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#senal", title: "La señal y el filtrado" }
  - { anchor: "#espectro", title: "El espectro y la frecuencia dominante" }
  - { anchor: "#espectrograma", title: "El espectrograma" }
  - { anchor: "#cumplimiento", title: "Cumplimiento frecuencia-dependiente" }
  - { anchor: "#conclusiones", title: "Conclusiones" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Entrada", body: "Forma de onda del sismógrafo (v vs t) a 1000 Hz." }
  - { title: "Método", body: <div class="ppv-card__value">FFT + filtrado</div> }
  - { title: "Stack", body: "Python · NumPy · SciPy · Matplotlib" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/fft-voladura/notebooks/fft-voladura.ipynb" }
  - { type: "data",     label: "Onda CSV",         url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/fft-voladura/data/raw/onda_voladura.csv" }
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
## 1) El PPV dice cuánto, la frecuencia dice cuánto daña

En la [Parte 1](/articles/vibraciones/) predijimos la **magnitud** de la vibración (el PPV) y en la [Parte 2](/articles/ppv-isolineas/) la mapeamos sobre el plano. Falta la otra mitad del problema, la que el Parte 1 dejó anotada como pendiente: la **frecuencia**.

Un geófono no mide un número: registra una **señal en el tiempo**, la velocidad de partícula del suelo durante los milisegundos que dura el evento. El PPV es apenas el **pico** de esa traza. Reducir toda la onda a su pico descarta una variable que la ingeniería de daño considera tan importante como la amplitud: su **contenido en frecuencia**.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Por qué la frecuencia importa</div>
  Una edificación tiene frecuencias naturales de resonancia (una vivienda típica, entre 4 y 12 Hz). Si la energía de la voladura se concentra cerca de esa banda, la estructura amplifica la vibración y el daño potencial crece, aunque el PPV sea moderado. Por eso las normas (USBM RI 8507, DIN 4150-3) <strong>no fijan un límite único</strong>: fijan un límite de PPV que <strong>cae a baja frecuencia</strong>.
</div>

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Pasar de la traza cruda del sismógrafo al veredicto de cumplimiento, aplicando la norma de vibración como está escrita: en función de la frecuencia.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Descomponer la onda con la FFT y controlar la fuga espectral con una ventana de Hann.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Filtrar la deriva de línea base y el ruido con un pasa-banda Butterworth de fase cero.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Identificar la frecuencia dominante y leer la evolución tiempo-frecuencia con un espectrograma.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Decidir el cumplimiento contra la curva DIN 4150-3, no contra un límite plano.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) La señal en el tiempo

La salida del sismógrafo es una serie temporal `v(t)`: velocidad de partícula (mm/s) muestreada a una **frecuencia de muestreo** `fs` (aquí 1000 Hz, un muestreo por milisegundo). El **PPV** es `max|v(t)|`. La señal de una voladura es **transitoria** y **multi-pulso**: cada retardo de la secuencia aporta un paquete de ondas que se superponen.

### 2.2) Fourier y la transformada rápida (FFT)

El **teorema de Fourier** dice que cualquier señal puede escribirse como suma de senoidales de distintas frecuencias, amplitudes y fases. La **Transformada de Fourier** obtiene esas componentes; la **FFT** (Fast Fourier Transform) la calcula de forma eficiente. El resultado es el **espectro**: amplitud en función de la frecuencia. Dos límites gobiernan lo que puede resolver:

<div class="formula">
  <code>Nyquist = fs / 2</code> &nbsp;&nbsp;·&nbsp;&nbsp; <code>Δf = fs / N</code>
</div>

- **Nyquist** `fs/2` es la máxima frecuencia representable (500 Hz aquí). Por encima aparece *aliasing*.
- **Resolución** `Δf = fs/N` es la separación entre líneas del espectro. Más señal (mayor N) da mejor resolución.

### 2.3) Fugas espectrales y ventanas

La FFT asume que la señal es **periódica** en la ventana analizada. Una voladura no lo es: los extremos no empatan y la energía de cada componente se **derrama** hacia frecuencias vecinas (*spectral leakage*). Multiplicar la señal por una **ventana** que va suavemente a cero en los bordes (la de **Hann** es la estándar) reduce esa fuga y afina los picos.

### 2.4) Frecuencia dominante

La **frecuencia dominante** es la del pico del espectro: donde se concentra la energía. La normativa a veces la estima por el conteo de **cruces por cero** alrededor del pico de la señal; ambas suelen coincidir, pero la del espectro es más robusta cuando hay varios modos.

### 2.5) Filtrado

La traza cruda trae contaminación que no es vibración del terreno: **deriva de línea base** (muy baja frecuencia, del sensor) y **ruido** de banda ancha. Un **filtro pasa-banda** (Butterworth, típicamente de 2 a 250 Hz en sismógrafos de voladura) los remueve. Se aplica con **fase cero** (`filtfilt`, hacia adelante y hacia atrás) para no desplazar los picos en el tiempo.

### 2.6) Límites que dependen de la frecuencia

Las guías más usadas fijan el PPV admisible **en función de la frecuencia dominante**:

| Norma | Baja frecuencia | Alta frecuencia |
|---|---|---|
| DIN 4150-3 (residencial, cimentación) | 5 mm/s (1 a 10 Hz) | 15 a 20 mm/s (50 a 100 Hz) |
| USBM RI 8507 (residencial) | 12.7 mm/s (4 a 15 Hz) | 50.8 mm/s (> 40 Hz) |

La lógica es la misma: **a baja frecuencia, el límite baja**, porque ahí la estructura resuena. Un límite plano (usar 12.5 mm/s para todo, como simplificación) ignora esto y puede aprobar un evento que la norma completa rechaza.


<a id="datos" class="anchor-clean"></a>
## 3) Datos: la forma de onda

Generamos una traza sintética físicamente plausible: un **tren de pulsos** (los retardos) que excitan varios **modos amortiguados**, más **deriva de línea base** y **ruido**. El modo dominante se ubica en una frecuencia baja (cerca de la resonancia estructural), el caso interesante para cumplimiento.

| Componente | Valor | Rol |
|---|---|---|
| Frecuencia de muestreo `fs` | 1000 Hz | Un muestreo por ms |
| Duración | 1.0 s (N = 1000) | Ventana del evento |
| Modo dominante | 9 Hz | Cerca de la resonancia residencial |
| Modos secundarios | 28 y 45 Hz | Contenido de mayor frecuencia, más amortiguado |
| Retardos | 7 pulsos (0 a 122 ms) | Secuencia de iniciación |
| Deriva + ruido | 0.6 Hz + banda ancha | Contaminación a filtrar |


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

```python
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.fft import rfft, rfftfreq

FS = 1000.0
rng = np.random.default_rng(2026)

N = 1000
t = np.arange(N) / FS
delays = [0.000, 0.021, 0.038, 0.060, 0.079, 0.101, 0.122]   # secuencia de retardos
amps   = [1.00, 0.85, 0.90, 0.70, 0.60, 0.50, 0.40]
modos  = [(9.0, 0.055, 1.00), (28.0, 0.10, 0.50), (45.0, 0.14, 0.28)]  # (frec, amortiguamiento, peso)

limpia = np.zeros(N)
for t0, a in zip(delays, amps):
    tau = t - t0; m = tau >= 0
    for fr_, z, w in modos:
        limpia[m] += a * w * np.exp(-z * 2*np.pi*fr_ * tau[m]) * np.sin(2*np.pi*fr_ * tau[m])

limpia *= 11.0 / np.max(np.abs(limpia))          # PPV de la señal limpia = 11 mm/s
v = limpia + 1.4*np.sin(2*np.pi*0.6*t + 0.7) + rng.normal(0, 0.55, N)   # + deriva + ruido
```


<a id="senal" class="anchor-clean"></a>
## 5) La señal en el tiempo y el filtrado

El PPV es el pico absoluto de la traza. Pero la traza cruda trae deriva y ruido. Un Butterworth de orden 4 entre 2 y 250 Hz, de fase cero, los remueve:

```python
b, a = sps.butter(4, [2, 250], btype='band', fs=FS)
v_filt = sps.filtfilt(b, a, v)
```

![Onda de voladura: la traza cruda (gris) y la filtrada (verde), con la banda de PPV marcada](/assets/articles/fft-voladura/fig-onda.png)

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} No juzgues sobre la traza cruda</div>
  El PPV cae de <strong>13.2 a 12.0 mm/s</strong> al filtrar: la deriva de línea base y el ruido inflaban el pico. Doce mm/s es el PPV real. Sobre la traza cruda (13.2), un límite de 12.5 mm/s daría un <strong>falso positivo de excedencia</strong>.
</div>

Con un **límite plano** de 12.5 mm/s (la simplificación habitual), el evento de 12.0 mm/s **cumpliría**. Pero el límite no es plano.


<a id="espectro" class="anchor-clean"></a>
## 6) El espectro y la frecuencia dominante

Calculamos el espectro de amplitud con `rfft`. Sin ventana, la fuga espectral ensancha la base de los picos; la ventana de Hann los afina.

```python
def espectro(x, win=None):
    w = np.ones(len(x)) if win is None else win
    amp = np.abs(rfft(x * w)) * 2 / np.sum(w)     # amplitud de un lado, normalizada
    return rfftfreq(len(x), 1/FS), amp

hann = sps.windows.hann(len(v_filt))
f, A = espectro(v_filt, hann)
f_dom = f[f >= 2][np.argmax(A[f >= 2])]           # ignora la componente casi-DC
```

![Comparación del espectro sin ventana (con fuga) y con ventana de Hann (picos afinados)](/assets/articles/fft-voladura/fig-ventana.png)

Con Nyquist en 500 Hz y una resolución `Δf = 1.0 Hz`, el espectro resuelve el evento con holgura. El pico domina claramente:

![Espectro de amplitud con la frecuencia dominante en 9 Hz, dentro de la banda de resonancia residencial de 4 a 12 Hz](/assets/articles/fft-voladura/fig-espectro.png)

La energía se concentra en **9 Hz**, justo dentro de la banda de resonancia residencial (4 a 12 Hz). Los modos de 28 y 45 Hz aportan menos porque están más amortiguados: decaen rápido y dejan poca energía. La dominante baja, en cambio, apenas se amortigua y resuena.


<a id="espectrograma" class="anchor-clean"></a>
## 7) El espectrograma: cómo evoluciona la frecuencia

La FFT da el espectro de toda la ventana, pero una voladura es transitoria. El **espectrograma** (STFT) muestra la energía en el plano tiempo-frecuencia, y los pulsos de los retardos aparecen como columnas.

```python
f_sp, t_sp, Sxx = sps.spectrogram(v_filt, fs=FS, nperseg=128, noverlap=112, window='hann')
```

![Espectrograma de la onda: la banda de 9 Hz persiste durante todo el evento mientras el contenido alto aparece y decae con cada pulso](/assets/articles/fft-voladura/fig-espectrograma.png)

La banda de 9 Hz **persiste** durante todo el evento (el modo que resuena), mientras el contenido alto aparece y decae con cada pulso. Esa persistencia de la baja frecuencia es lo que la hace peligrosa: mantiene a la estructura excitada cerca de su resonancia.


<a id="cumplimiento" class="anchor-clean"></a>
## 8) Cumplimiento que depende de la frecuencia

Unimos magnitud y frecuencia. Ubicamos el evento `(frecuencia dominante, PPV)` sobre la curva de DIN 4150-3 y lo comparamos con el límite plano.

```python
def limite_din(f):
    """DIN 4150-3, edificaciones residenciales, cimentación (mm/s)."""
    if f < 10:  return 5.0
    if f < 50:  return 5.0 + (15.0 - 5.0) * (f - 10) / 40.0
    if f < 100: return 15.0 + (20.0 - 15.0) * (f - 50) / 50.0
    return 20.0
```

![Cumplimiento frecuencia-dependiente: el evento cae bajo el límite plano de 12.5 mm/s pero por encima de la curva DIN 4150-3 en su frecuencia de 9 Hz](/assets/articles/fft-voladura/fig-cumplimiento.png)

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} El veredicto que la magnitud sola no puede dar</div>
  El evento cae <strong>por debajo del límite plano</strong> de 12.5 mm/s (parece cumplir), pero <strong>por encima de la curva DIN</strong> en su frecuencia dominante de 9 Hz, que exige 5 mm/s. <strong>Excede el límite real por 2.4 veces.</strong> La magnitud lo aprobaba; la frecuencia lo reprueba.
</div>


<a id="conclusiones" class="anchor-clean"></a>
## 9) Conclusiones

- La vibración de una voladura es una **señal en el tiempo**, no un número. Reducirla al PPV descarta su **contenido en frecuencia**, que la ingeniería de daño trata como igual de importante.
- La **FFT** entrega el espectro; la **ventana de Hann** controla la fuga y afina los picos. Nyquist (`fs/2`) y la resolución (`Δf = fs/N`) fijan lo que se puede resolver.
- El **filtrado pasa-banda de fase cero** (2 a 250 Hz) remueve deriva y ruido: el PPV cae de 13.2 a 12.0 mm/s. Juzgar sobre la traza cruda habría dado un falso positivo de excedencia.
- La **frecuencia dominante** (9 Hz) cae en la banda de resonancia residencial. Contra el límite plano el evento cumple; contra la **curva DIN 4150-3** excede por 2.4 veces. La frecuencia decide.
- Este taller cierra la trilogía de vibraciones: **magnitud** (Parte 1), **distribución espacial** (Parte 2) y **contenido en frecuencia** (Parte 3). Un límite de voladura sin frecuencia está incompleto.


<a id="refs" class="anchor-clean"></a>
## 10) Referencias

<div class="references" markdown="1">

Siskind, D. E., Stagg, M. S., Kopp, J. W., & Dowding, C. H. (1980). **Structure response and damage produced by ground vibration from surface mine blasting.** U.S. Bureau of Mines, RI 8507. Establece los criterios de daño de vibración por voladura, incluidos los límites de PPV que dependen de la frecuencia.

DIN 4150-3 (2016). **Erschütterungen im Bauwesen – Einwirkungen auf bauliche Anlagen.** Norma alemana con valores guía de velocidad de vibración por tipo de estructura y frecuencia, ampliamente usada en cumplimiento de voladura.

Dowding, C. H. (1985). **Blast Vibration Monitoring and Control.** Prentice-Hall. Texto de referencia de instrumentación, análisis de señales y criterios de daño en vibración por voladura.

Oppenheim, A. V., & Schafer, R. W. (2009). **Discrete-Time Signal Processing** (3rd ed.). Pearson. Fundamento de la transformada discreta de Fourier, muestreo, aliasing y filtrado digital.

Harris, F. J. (1978). **On the use of windows for harmonic analysis with the discrete Fourier transform.** Proceedings of the IEEE, 66(1), 51-83. Referencia clásica sobre ventanas y fuga espectral.

</div>
