---
layout: article
title: "Sobrepresión de aire (airblast): la otra emisión de la voladura"
subtitle: "Nivel en dB, atenuación cúbica y la inversión térmica que lleva el ruido lejos del disparo."
date: 2026-07-07

# Hero + card
cover: "/assets/articles/airblast/cover.jpg"
hero: "/assets/articles/airblast/hero.jpg"
card: /assets/articles/airblast/card.jpg

# Metadatos / filtros
series: "Vibraciones"
series_order: 5
tags: [Python, Vibraciones, Voladura, Airblast, DIN 4150, Viento]
reading_time: "25 min"

# Resumen destacado
summary: >
  <p>Quinta parte de la serie de vibraciones. Toda la serie trató la vibración del terreno; esta cierra con la otra emisión de la voladura, la que viaja por el aire: la <strong>sobrepresión de aire</strong> (airblast), la onda de presión que hace traquetear las ventanas y dispara las quejas. La modelamos en su escala propia (el <strong>decibel</strong>), con su atenuación (distancia escalada <strong>cúbica</strong>) y con el comodín que la vibración del suelo no tiene: la <strong>inversión térmica</strong>, que lleva el airblast lejos del disparo.</p>
  <p></p>
  <div class="ppv-note">El airblast rara vez daña, pero molesta lejos: <strong>L = 20 log₁₀(ΔP / P₀)</strong></div>

# Índice de contenidos
contents:
  - { anchor: "#contexto", title: "El PPV cumple y las ventanas traquetean" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos: receptores y voladura" }
  - { anchor: "#impl", title: "Implementación en Python" }
  - { anchor: "#atenuacion", title: "Atenuación y cumplimiento" }
  - { anchor: "#cumplimiento", title: "El mapa de cumplimiento" }
  - { anchor: "#inversion", title: "La inversión térmica" }
  - { anchor: "#viento", title: "El efecto del viento" }
  - { anchor: "#conclusiones", title: "Conclusiones" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Magnitud", body: <div class="ppv-card__value">Nivel en dB</div> }
  - { title: "Atenuación", body: "Distancia escalada cúbica (D / W^⅓)." }
  - { title: "Stack", body: "Python · NumPy · pandas · Matplotlib" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/airblast/notebooks/airblast.ipynb" }
  - { type: "data",     label: "Receptores CSV",   url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/airblast/data/raw/receptores_airblast.csv" }
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
## 1) El PPV cumple y las ventanas igual traquetean

Toda esta serie trató la vibración del **terreno**: [magnitud](/articles/vibraciones/), [mapa](/articles/ppv-isolineas/), [frecuencia](/articles/fft-voladura/) y [timing](/articles/signature-hole/). Pero una voladura emite por dos vías, y la segunda viaja por el **aire**.

Es una escena común en minería cercana a poblados: el sismógrafo confirma que el PPV quedó muy por debajo del límite, y aun así llegan quejas de casas a un kilómetro. La causa casi siempre es la **otra emisión**, la que no viaja por el suelo sino por el aire.

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} Qué es el airblast</div>
  El <strong>airblast</strong> es la onda de sobrepresión que la detonación lanza a la atmósfera. Rara vez daña estructuras (haría falta una sobrepresión enorme), pero <strong>se percibe y molesta</strong> a niveles mucho más bajos: hace vibrar vidrios y objetos sueltos. Es, ante todo, un problema de <strong>relación con la comunidad</strong>, y conviene predecirlo con el mismo rigor que la vibración del suelo.
</div>

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Modelar el airblast en su escala y atenuación propias, y mostrar cómo la atmósfera lo lleva lejos del disparo.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Entender la escala en decibeles (logarítmica) y la atenuación con distancia escalada cúbica.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Distinguir el umbral de daño (133 dB) del de molestia (~115 dB) y ubicar los receptores.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Modelar el ducto de una inversión térmica y su efecto sobre un receptor lejano.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Concluir por qué el control de airblast agrega una regla propia: no disparar bajo inversión.
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) Sobrepresión y la escala en decibeles

El airblast se mide como **sobrepresión** `ΔP` (la presión por encima de la atmosférica, en pascales), pero se reporta en **decibeles lineales** (dB, sin ponderar), por el enorme rango dinámico:

<div class="formula">
  <code>L = 20 · log₁₀(ΔP / P₀)</code> &nbsp;&nbsp;,&nbsp;&nbsp; <code>P₀ = 2×10⁻⁵ Pa</code>
</div>

La escala es **logarítmica**: cada **+6 dB** duplica la sobrepresión, y **+20 dB** la multiplica por diez. Un airblast de 130 dB no es "un poco más" que uno de 110 dB: es **diez veces** la presión.

### 2.2) Atenuación con distancia escalada cúbica

A diferencia de la vibración del terreno (que usa distancia escalada de **raíz cuadrada**), el airblast atenúa con distancia escalada de **raíz cúbica**, porque la energía se expande en el volumen del aire:

<div class="formula">
  <code>ΔP = K · (D / W<sup>1/3</sup>)<sup>-α</sup></code>
</div>

En dB, la caída es lineal con el logaritmo de la distancia. El airblast **cae rápido**: se disipa en cientos de metros.

### 2.3) Daño contra molestia

| Nivel | dB | Qué ocurre |
|---|---|---|
| Daño de vidrios (USBM RI 8507) | 133 | Umbral de rotura de ventanas |
| Quejas / traqueteo | ~115 a 120 | Ventanas y objetos vibran, la gente reacciona |
| Percepción | ~100 | Se oye y se siente |

El **daño** (133 dB) solo ocurre muy cerca del disparo; las **quejas** aparecen mucho más lejos y con niveles menores. El airblast se gestiona por el umbral de **molestia**, no por el de daño.

### 2.4) El comodín atmosférico: la inversión térmica

La velocidad del sonido crece con la temperatura. En un día normal el aire se enfría con la altura (*lapse*): el sonido se refracta hacia **arriba** y se aleja del suelo, formando una **zona de sombra**. Pero en una **inversión térmica** (aire más caliente arriba, típica al amanecer) el sonido se refracta hacia **abajo**: queda atrapado en un **ducto** cerca del suelo, deja de dispersarse en volumen y **viaja lejos con poca pérdida**. Un receptor distante puede recibir **10 a 20 dB más** bajo inversión. Es la razón por la que se evita disparar bajo inversión.


<a id="datos" class="anchor-clean"></a>
## 3) Datos: los receptores y la voladura

| Parámetro | Valor | Rol |
|---|---|---|
| Carga `W` | 500 kg | Carga de la voladura |
| Nivel de referencia | 130 dB a 100 m | Airblast de un disparo de producción |
| Exponente de atenuación `α` | 1.3 | Caída de la sobrepresión |
| Distancia de ducto `D_skip` | 250 m | Donde el ducto de inversión empieza a atrapar |

Dos receptores: la caseta de operaciones (cerca) y un poblado (lejos).


<a id="impl" class="anchor-clean"></a>
## 4) Implementación en Python

```python
import numpy as np

P0 = 2e-5                       # presión de referencia (Pa)
W = 500.0                       # carga (kg)
ALFA = 1.30                     # exponente de atenuación
L_REF, D_REF = 130.0, 100.0     # 130 dB a 100 m
D_SKIP, DECAY_DUCT = 250.0, 3.0 # ducto de inversión: inicio y caída (dB/decada)

def L_aire_libre(D):
    return L_REF - 20*ALFA*np.log10(np.asarray(D, float)/D_REF)

def L_inversion(D):
    D = np.asarray(D, float)
    return np.where(D > D_SKIP,
                    L_aire_libre(D_SKIP) - DECAY_DUCT*np.log10(D/D_SKIP),
                    L_aire_libre(D))
```

La escala es logarítmica: 100 dB son 2 Pa, 115 dB son 11 Pa, 130 dB son 63 Pa. De 115 a 130 dB (15 dB), la sobrepresión se multiplica por **5.6**.


<a id="atenuacion" class="anchor-clean"></a>
## 5) La curva de atenuación y el cumplimiento

Trazamos el nivel en aire libre contra la distancia, con los umbrales de daño y de quejas y los receptores.

![Atenuación del airblast con la distancia: el daño (133 dB) solo dentro de 77 m, las quejas (115 dB) dentro de 378 m; la caseta a 125 dB y el poblado a 99 dB](/assets/articles/airblast/fig-atenuacion.png)

El airblast **cae rápido**: el **daño** (133 dB) solo es posible dentro de **77 m** del disparo, y las **quejas** (115 dB) dentro de **378 m**. El poblado, a 1500 m, recibe **99 dB** en aire libre: lo oye, pero está muy por debajo del umbral de quejas. En un día normal, no hay problema.


<a id="cumplimiento" class="anchor-clean"></a>
## 6) Del punto a la carga: el mapa de cumplimiento

La curva anterior responde para **una carga fija** (500 kg) y **dos receptores**. La pregunta que de verdad hace el área de perforación y voladura es la inversa: *dado un receptor a cierta distancia, ¿cuánta carga por retardo puedo disparar sin cruzar el umbral?* Para eso hay que traer de vuelta la distancia escalada cúbica (`D / W^⅓`) y generalizar el modelo a **cualquier carga**, no solo a 500 kg:

```python
D_REF, W_REF = 100.0, 500.0   # calibracion: 130 dB a 100 m con 500 kg

def L_aire_libre_W(D, W):
    return L_REF - 20*ALFA*np.log10(D/D_REF) + (20*ALFA/3)*np.log10(W/W_REF)

def L_inversion_W(D, W):
    libre = L_aire_libre_W(D, W)
    skip_level = L_aire_libre_W(D_SKIP, W)
    return np.where(D > D_SKIP, skip_level - DECAY_DUCT*np.log10(D/D_SKIP), libre)
```

Barriendo distancia y carga a la vez (en vez de un solo receptor) obtenemos un mapa de cumplimiento completo, con los umbrales de daño y quejas como curvas de nivel:

![Mapa de cumplimiento del airblast: dos paneles en escala log-log, distancia (50 a 3000 m) contra carga por retardo (10 a 2000 kg), con curvas de nivel en 133 dB (daño) y 115 dB (quejas). En aire libre la curva de quejas es una diagonal simple; bajo inversión térmica la misma curva se quiebra y se desplaza hacia distancias mayores más allá de los 250 m del ducto, reduciendo la carga admisible a igual distancia](/assets/articles/airblast/fig-cumplimiento.png)

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} De un cálculo puntual a una regla de diseño</div>
  El mapa reemplaza "¿cumple este disparo?" por <strong>"¿qué combinaciones de distancia y carga cumplen?"</strong>. En aire libre, la curva de 115 dB es una diagonal recta en escala log-log: el límite de carga crece con el cubo de la distancia. Bajo inversión, la misma curva se <strong>quiebra</strong> en el <code>D_skip</code> (250 m) y se aplana: más allá de ese punto, alejarse ya no compra tanto margen como en aire libre, porque el ducto reduce la pérdida con la distancia. El mismo mapa sirve para cualquier receptor nuevo, sin recalcular caso por caso.
</div>


<a id="inversion" class="anchor-clean"></a>
## 7) La inversión térmica: el airblast viaja lejos

El escenario cambia bajo una inversión. El ducto atrapa el sonido y reduce su pérdida más allá del `D_skip`.

![Aire libre vs inversión térmica: en aire libre el poblado recibe 99 dB, pero bajo la inversión el ducto lo lleva a 117 dB, por encima del umbral de quejas](/assets/articles/airblast/fig-inversion.png)

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} La atmósfera cambia el veredicto</div>
  El poblado que en aire libre recibe <strong>99 dB</strong> (cómodo), bajo inversión recibe <strong>117 dB</strong>, <strong>+18 dB</strong>, y cruza el umbral de quejas. El mismo disparo, con la misma carga y la misma distancia, genera quejas a 1.5 km <strong>solo por la atmósfera</strong>. Por eso el control de airblast incluye una regla que la vibración del suelo no necesita: <strong>no disparar bajo inversión térmica</strong>.
</div>

Puesto lado a lado, el contraste entre los dos receptores es la clave de lectura: la caseta, cerca del disparo (dentro de `D_skip`), no cambia con la inversión — el ducto todavía no actúa a esa distancia. El poblado, lejos, es el que paga el efecto atmosférico.

![Nivel de airblast por receptor en aire libre y bajo inversión térmica: la caseta de operaciones (150 m) se mantiene en 125 dB en ambos escenarios porque está dentro del radio del ducto; el poblado (1500 m) sube de 99 a 117 dB bajo inversión, cruzando el umbral de quejas de 115 dB](/assets/articles/airblast/fig-receptores.png)


<a id="viento" class="anchor-clean"></a>
## 8) El modelo no es isotrópico: el efecto del viento

Los modelos anteriores solo dependen de la distancia: predicen el mismo nivel en cualquier dirección alrededor del disparo. En la práctica, el sonido viaja mejor **a favor del viento** que en contra — el mismo efecto de refracción que la inversión térmica, pero horizontal y de menor magnitud. Un factor direccional simple lo captura:

```python
DL_VIENTO = 5.0  # dB, amplificacion/atenuacion maxima por efecto de viento

def L_direccional(L_base, theta, theta_viento=0.0):
    """theta=theta_viento -> a favor del viento (maximo); theta opuesto -> en contra (minimo)."""
    return L_base + DL_VIENTO * np.cos(theta - theta_viento)
```

Aplicamos el factor sobre los dos escenarios ya calculados (aire libre e inversión) para el poblado, barriendo la dirección 0-360°:

![Nivel de airblast en el poblado según la dirección del viento, en coordenadas polares: en aire libre el círculo completo (94 a 104 dB) queda siempre bajo el umbral de quejas; bajo inversión térmica el nivel varía de 112 dB en contra del viento a 122 dB a favor del viento, cruzando el umbral de 115 dB según la dirección](/assets/articles/airblast/fig-viento.png)

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} La dirección decide, pero solo bajo inversión</div>
  En <strong>aire libre</strong>, el viento mueve el nivel del poblado entre <strong>94 y 104 dB</strong>: nunca se acerca al umbral de 115 dB, la dirección no importa. Bajo <strong>inversión térmica</strong> el rango es <strong>112 a 122 dB</strong> — <strong>cruza el umbral en ambos sentidos</strong>. Disparar con el poblado a favor del viento empeora una condición que ya era de riesgo (122 dB); disparar con el poblado a contraviento la revierte (112 dB, de vuelta en cumplimiento). Bajo inversión, la dirección del viento no es un detalle: es la diferencia entre cumplir y no cumplir.
</div>

Ninguno de los dos efectos atmosféricos importa por separado en un día normal y sin viento a favor. Es la combinación —inversión que atrapa el sonido, viento que lo empuja hacia el receptor— la que produce el peor escenario, y ninguno de los dos aparece en un modelo que solo mira la distancia.


<a id="conclusiones" class="anchor-clean"></a>
## 9) Conclusiones

- El **airblast** es la segunda emisión de la voladura, por el aire. Rara vez daña, pero **molesta** a niveles mucho menores: es sobre todo un problema de relación con la comunidad.
- Se mide en **dB lineales** (escala logarítmica: +6 dB duplica la presión) y atenúa con distancia escalada **cúbica**, no cuadrada como la vibración del suelo. Cae rápido: **daño solo dentro de 77 m**, **quejas dentro de 378 m** en aire libre.
- La **atmósfera** es el comodín: una **inversión térmica** forma un ducto que lleva el airblast lejos. El poblado a 1500 m pasa de **99 dB** (cómodo) a **117 dB** (quejas), **+18 dB**, sin cambiar la carga ni la distancia.
- Por eso el control de airblast agrega una regla propia: **no disparar bajo inversión térmica**, y monitorear el airblast, no solo la vibración del terreno.
- El **viento** por sí solo no decide nada en aire libre (el poblado se mueve entre 94 y 104 dB, lejos de los 115 dB de quejas), pero bajo **inversión térmica** la dirección del viento determina si el poblado queda en 112 dB (sin quejas) o en 122 dB (quejas seguras): la atmósfera y la dirección actúan juntas, no por separado.
- Cierra la serie de vibraciones cubriendo la emisión que suele originar las quejas: magnitud, mapa, frecuencia, timing y, ahora, **aire**.


<a id="refs" class="anchor-clean"></a>
## 10) Referencias

La refracción atmosférica que describe la Sección 7 es un mecanismo estándar en la literatura de monitoreo de airblast (Dowding, 1985; McKenzie, 1990). El siguiente esquema resume la idea física detrás de esas referencias:

![Esquema de refracción atmosférica del sonido: en atmósfera normal la temperatura decrece con la altura y el sonido se refracta hacia arriba, dejando una zona de sombra cerca del suelo; en inversión térmica la temperatura aumenta con la altura cerca del suelo y el sonido se refracta hacia abajo, quedando atrapado en un ducto que lo lleva lejos con poca pérdida](/assets/articles/airblast/fig-esquema-inversion.png)

<div class="references" markdown="1">

Siskind, D. E., Stagg, M. S., Kopp, J. W., & Dowding, C. H. (1980). **Structure response and damage produced by ground vibration from surface mine blasting.** U.S. Bureau of Mines, RI 8507. Incluye el criterio de 133 dB para airblast.

Siskind, D. E., Stachura, V. J., Stagg, M. S., & Kopp, J. W. (1980). **Structure response and damage produced by airblast from surface mining.** U.S. Bureau of Mines, RI 8485. Estudio de referencia sobre airblast, respuesta de estructuras y umbrales de daño y molestia.

McKenzie, C. (1990). **Quarry blast monitoring: technical and environmental perspectives.** Quarry Management. Discute el rol de las condiciones atmosféricas en la propagación del airblast.

Dowding, C. H. (1985). **Blast Vibration Monitoring and Control.** Prentice-Hall.

ISEE (2011). **Field Practice Guidelines for Blasting Seismographs.** International Society of Explosives Engineers.

</div>
