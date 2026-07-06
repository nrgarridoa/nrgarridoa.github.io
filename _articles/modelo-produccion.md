---
layout: article
title: "Tu primer modelo de producción minera en Python"
subtitle: "Tonelaje, ley y recuperación: el balance metalúrgico que sostiene el reporte mensual."
date: 2026-07-05

# Hero + card
cover: "/assets/articles/modelo-produccion/cover.jpg"
hero: "/assets/articles/modelo-produccion/hero.jpg"
card: /assets/articles/modelo-produccion/card.jpg

# Metadatos / filtros
series: "Planeamiento minero"
tags: [Python, Planeamiento, Producción, Cut-off, Monte Carlo]
reading_time: "20 min"

# Resumen destacado
summary: >
  <p>El reporte de producción de cualquier mina descansa en una sola identidad: <strong>metal recuperado = tonelaje × ley × recuperación</strong>. En Python son unas veinte líneas. En este taller la construimos para un pórfido <strong>Cu-Au</strong> y la endurecemos hasta convertirla en una herramienta de decisión: <strong>ley de corte</strong>, <strong>análisis de sensibilidad</strong> y <strong>cuantificación de la incertidumbre</strong> con Monte Carlo.</p>
  <p></p>
  <div class="ppv-note">Metal recuperado = <strong>Σ (Tonelaje × Ley × Recuperación)</strong>, con las unidades bien puestas.</div>

# Índice de contenidos
contents:
  - { anchor: "#contexto", title: "El reporte de producción" }
  - { anchor: "#teoria", title: "Marco teórico" }
  - { anchor: "#datos", title: "Datos: un mes de producción" }
  - { anchor: "#modelo", title: "El modelo en 20 líneas" }
  - { anchor: "#produccion", title: "La producción del mes" }
  - { anchor: "#cutoff", title: "Ley de corte" }
  - { anchor: "#sensibilidad", title: "Análisis de sensibilidad" }
  - { anchor: "#incertidumbre", title: "Incertidumbre (Monte Carlo)" }
  - { anchor: "#reconciliacion", title: "Reconciliación" }
  - { anchor: "#conclusiones", title: "Conclusiones" }
  - { anchor: "#refs", title: "Referencias" }

# Cards técnicas
tech_cards:
  - { title: "Modelo", body: <div class="ppv-card__value">T × ley × rec</div> }
  - { title: "Salida", body: "Metal (Cu, Au), valor (NSR), cut-off, P10/P50/P90." }
  - { title: "Stack", body: "Python · NumPy · pandas · Matplotlib" }

# Recursos (con íconos por tipo)
resources:
  - { type: "notebook", label: "Notebook Jupyter", url: "https://nbviewer.org/github/nrgarridoa/talleres-mineria-python/blob/main/modelo-produccion/notebooks/modelo-produccion.ipynb" }
  - { type: "data",     label: "Dataset CSV",      url: "https://github.com/nrgarridoa/talleres-mineria-python/raw/main/modelo-produccion/data/raw/produccion_mes.csv" }
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
## 1) El reporte de producción y la identidad que lo sostiene

Cada mes, una operación cierra su reporte de producción y responde una pregunta simple de enunciar y difícil de cumplir: **¿cuánto metal pagable produjo, y cuánto vale?** La respuesta no sale de una planta piloto ni de un modelo de caja negra. Sale de una identidad contable que se calcula igual en toda mina: la **contabilidad metalúrgica**.

El aporte de Python no es reemplazar esa identidad, sino **convertirla en un modelo**: parametrizado, reproducible y, sobre todo, interrogable. Un número puntual (5,879 t de Cu) parece una certeza. El modelo permite preguntarle qué lo mueve, dónde está el punto de equilibrio y con qué probabilidad se cumple.

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} De un cálculo a una decisión</div>
  Ese salto es el objetivo del taller. En veinte líneas tendremos el modelo de producción funcionando; el resto es hacerlo hablar:
  <ul>
    <li>¿Cuánto vale la producción, y cuánto aporta cada metal?</li>
    <li>¿Qué material paga su costo (ley de corte), y cuánto lo baja el subproducto?</li>
    <li>¿Qué variable mueve más el resultado, y con qué probabilidad se cumple la meta?</li>
  </ul>
</div>

<!-- Objetivos -->
<section class="objetivos">
<div class="objetivos-header">
  <div class="objetivos-badge">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2v4" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="14" r="6" stroke="currentColor" stroke-width="1.6"/><path d="M12 10v4" stroke="currentColor" stroke-width="1.6"/></svg>
    OBJETIVOS
  </div>
  <p class="objetivos-lead">
    Construir un modelo de producción reproducible y llevarlo del cálculo puntual a la decisión de planeamiento.
  </p>
</div>

<div class="objetivos-card">
  <ul class="objetivos-list">
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Escribir la contabilidad metalúrgica Cu-Au con el manejo correcto de unidades (%, g/t, oz, lb).
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Valorizar la producción con un NSR simplificado y calcular la ley de corte con crédito por subproducto.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Identificar las palancas del resultado con un diagrama de tornado.
    </li>
    <li>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      Cuantificar la probabilidad de cumplir la meta con simulación de Monte Carlo (P10/P50/P90).
    </li>
  </ul>
</div>
</section>

<a id="teoria" class="anchor-clean"></a>
## 2) Marco teórico

### 2.1) La identidad de contabilidad metalúrgica

El metal recuperado en un periodo es la suma, sobre cada lote de mineral procesado, del producto de tres factores: cuánto material se trató, qué tan rico era y qué fracción se recuperó en planta:

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} La identidad</div>
  <div class="formula">
    <code>Metal recuperado = Σ<sub>i</sub> ( T<sub>i</sub> · g<sub>i</sub> · R<sub>i</sub> )</code>
  </div>
  <p>con <code>T</code> el tonelaje del lote (t), <code>g</code> la ley (fracción) y <code>R</code> la recuperación metalúrgica (fracción).</p>
</div>

El detalle que hace fallar a la mitad de las hojas de cálculo es el **manejo de unidades**: la ley de Cu se reporta en **%** (dividir entre 100) y la de Au en **g/t** (pasar de gramos a onzas troy, 31.1035 g/oz). El metal de Cu se expresa en toneladas o libras (2204.62 lb/t) y el de Au en onzas.

### 2.2) De metal a valor: pagables y precio

El metal en concentrado no se cobra completo. La fundición paga una **fracción pagable** (típicamente 96 % del Cu, 95 % del Au) y descuenta cargos de tratamiento y refinación. El ingreso aproximado es:

<div class="formula">
  <code>Ingreso = Metal<sub>Cu</sub> · P<sub>Cu</sub> · f<sub>Cu</sub> + Metal<sub>Au</sub> · P<sub>Au</sub> · f<sub>Au</sub></code>
</div>

con `P` el precio y `f` el factor pagable. Es un **NSR** (Net Smelter Return) simplificado, suficiente para dimensionar el valor de la producción y el peso de cada metal.

### 2.3) Ley de corte (cut-off)

La ley de corte es la ley mínima a la que un lote paga su propio costo de tratamiento. Igualando ingreso y costo por tonelada:

<div class="formula">
  <code>g<sub>corte</sub> = Costo / ( P<sub>Cu</sub> · R<sub>Cu</sub> · f<sub>Cu</sub> )</code>
</div>

El **crédito por subproducto** (el oro) baja la ley de corte del cobre: cada tonelada trae onzas de oro que ayudan a pagar el costo, así que se acepta como mineral material de menor ley de cobre.

### 2.4) Sensibilidad e incertidumbre

El modelo determinístico entrega un número, pero sus entradas (ley, recuperación, tonelaje, precio) son estimaciones con rango. De ahí nacen dos preguntas de planeamiento:

- **Sensibilidad:** ¿qué variable mueve más el resultado? Se responde con un **diagrama de tornado**, perturbando cada entrada dentro de su rango plausible. La intuición de campo (mover más tonelaje) suele perder frente a la ley y el precio.
- **Incertidumbre:** ¿con qué probabilidad se cumple la meta? Se responde con **simulación de Monte Carlo**: se muestrean las entradas según su distribución y se obtiene la distribución del metal producido, con sus percentiles.


<a id="datos" class="anchor-clean"></a>
## 3) Datos: un mes de producción Cu-Au

Generamos un mes de operación de un pórfido Cu-Au, con un registro por día:

| Variable | Símbolo | Unidad | Descripción |
|---|---|---|---|
| Tonelaje | T | t/día | Mineral tratado en planta |
| Ley de cobre | g_Cu | % | Ley de cabeza de Cu |
| Ley de oro | g_Au | g/t | Ley de cabeza de Au |
| Recuperación Cu | R_Cu | fracción | Recuperación metalúrgica de Cu |
| Recuperación Au | R_Au | fracción | Recuperación metalúrgica de Au |

Parámetros comerciales y de costo:

| Parámetro | Valor | Nota |
|---|---|---|
| Precio Cu | 4.00 USD/lb | Precio de referencia |
| Precio Au | 2400 USD/oz | Precio de referencia |
| Pagable Cu / Au | 0.96 / 0.95 | Fracción que paga la fundición |
| Costo | 18 USD/t | Mina + planta + G&A por tonelada tratada |

La ley y la recuperación **co-varían**: un día con mineral más rico suele recuperar algo mejor. El oro acompaña al cobre, como en todo pórfido Cu-Au.


<a id="modelo" class="anchor-clean"></a>
## 4) El modelo, en veinte líneas

Con el dataset cargado, la contabilidad metalúrgica del mes cabe en un bloque. Este es el núcleo del taller; todo lo demás se construye sobre él.

```python
import numpy as np
import pandas as pd

PRECIO_CU, PRECIO_AU = 4.00, 2400.0    # USD/lb Cu, USD/oz Au
PAG_CU, PAG_AU       = 0.96, 0.95      # fracción pagable
LB_POR_T, G_POR_OZ   = 2204.62, 31.1035

df = pd.read_csv("produccion_mes.csv")

# metal recuperado por día (cuidado con las unidades: Cu en %, Au en g/t)
df["cu_t"]  = df.tonelaje_t * df.cu_pct/100      * df.rec_cu     # t de Cu
df["au_oz"] = df.tonelaje_t * df.au_gpt/G_POR_OZ * df.rec_au     # oz de Au

# totales del mes
cu_t, au_oz = df.cu_t.sum(), df.au_oz.sum()
cu_lb = cu_t * LB_POR_T

# valor (NSR simplificado)
ing_cu  = cu_lb * PRECIO_CU * PAG_CU
ing_au  = au_oz * PRECIO_AU * PAG_AU
ingreso = ing_cu + ing_au

print(f"Cu: {cu_t:,.0f} t  |  Au: {au_oz:,.0f} oz  |  Ingreso: US$ {ingreso/1e6:.1f} M")
```

```
Cu: 5,879 t  |  Au: 5,106 oz  |  Ingreso: US$ 61.4 M
```

Ahí está el modelo de producción funcionando: **5,879 t de Cu** (12.96 Mlb), **5,106 oz de Au** y un ingreso de **US$ 61.4 M**. Un solo bloque reproduce el número que ocupa la primera línea del reporte mensual. A partir de aquí lo interrogamos.


<a id="produccion" class="anchor-clean"></a>
## 5) La producción del mes

El total del mes es la acumulación de la producción diaria. Ver el aporte de cada día muestra la variabilidad real de una operación (días ricos y días pobres) que el promedio esconde. La ley media ponderada es **0.543 % Cu** y **0.180 g/t Au** sobre **1,244,665 t** tratadas.

![Producción diaria de Cu (barras) y acumulado del mes (línea), mostrando la variabilidad día a día](/assets/articles/modelo-produccion/fig-produccion-diaria.png)

Del lado del valor, el cobre aporta el **81 %** del ingreso y el oro el **19 %**. El oro es subproducto, pero uno que no se desprecia: paga por sí solo una parte relevante del costo, y eso reaparece en la ley de corte.

![Desglose del ingreso del mes: cobre US$ 49.8 M (81 %) y oro US$ 11.6 M (19 %)](/assets/articles/modelo-produccion/fig-ingreso.png)


<a id="cutoff" class="anchor-clean"></a>
## 6) Ley de corte: qué es mineral

La ley de corte separa mineral de estéril. La calculamos sin y con el crédito del oro, para ver cuánto baja el subproducto la exigencia sobre el cobre.

```python
precio_cu_t = PRECIO_CU * LB_POR_T / 100    # USD por (t · 1% Cu) recuperado y pagable
credito_au  = ley_media_au / G_POR_OZ * df.rec_au.mean() * PRECIO_AU * PAG_AU   # USD/t

cutoff_sin = COSTO / (precio_cu_t * df.rec_cu.mean() * PAG_CU)
cutoff_con = (COSTO - credito_au) / (precio_cu_t * df.rec_cu.mean() * PAG_CU)
```

<div class="callout-info">
  <div class="callout-icon">{% include icons/lightbulb.svg class="h-5 w-5" %} El subproducto cambia qué es mineral</div>
  El crédito del oro (<strong>US$ 9.28 / t</strong>) baja la ley de corte del cobre de <strong>0.246 %</strong> a <strong>0.119 %</strong>. La ley de cabeza (0.543 % Cu) corre a <strong>4.6 veces</strong> la ley de corte con crédito: el mes opera con margen holgado.
</div>

Ese múltiplo (ley de cabeza sobre ley de corte) es el primer indicador de salud económica de la alimentación: cuanto más alto, más resistente es la operación a caídas de precio o subidas de costo.


<a id="sensibilidad" class="anchor-clean"></a>
## 7) Análisis de sensibilidad: ¿qué mueve el resultado?

Perturbamos cada variable dentro de un rango plausible y medimos el efecto sobre el ingreso. El resultado se ordena en un **diagrama de tornado**: la barra más larga es la variable que más manda.

![Diagrama de tornado: el precio del Cu y la ley de Cu producen los mayores cambios en el ingreso, muy por encima del tonelaje](/assets/articles/modelo-produccion/fig-tornado.png)

<div class="callout-warning">
  <div class="callout-icon">{% include icons/alert-triangle.svg class="h-5 w-5" %} La intuición de campo pierde</div>
  El precio del Cu (±20 % → US$ 19.9 M) y la ley de Cu (±15 % → US$ 14.9 M) dominan, muy por encima del tonelaje (±8 % → US$ 9.8 M). La reacción instintiva ante una meta de metal es <em>mover más material</em>, pero donde más se gana es asegurando la <strong>ley alimentada</strong> (control de leyes, dilución) y gestionando la <strong>exposición al precio</strong>, no acelerando la pala.
</div>


<a id="incertidumbre" class="anchor-clean"></a>
## 8) Incertidumbre: ¿con qué probabilidad se cumple la meta?

El modelo determinístico da 5,879 t de Cu como si fuera un hecho. Pero la ley proviene de un modelo de recursos con error, la recuperación varía y el tonelaje depende de la disponibilidad de planta. Propagamos esa **incertidumbre sistemática** (a nivel de mes, no ruido diario) con Monte Carlo.

```python
sim_rng = np.random.default_rng(7)
M = 50_000
f_ley = sim_rng.lognormal(0, 0.09, M)     # incertidumbre del modelo de leyes (~9%)
f_rec = sim_rng.normal(1, 0.025, M)       # recuperación de planta (~2.5%)
f_ton = sim_rng.normal(1, 0.04, M)        # throughput (~4%)
cu_sim = df.cu_t.sum() * f_ley * f_rec * f_ton

p10, p50, p90 = np.percentile(cu_sim, [10, 50, 90])
```

![Histograma de 50,000 escenarios de producción mensual de Cu, con P10, P50, P90 y el plan determinístico cayendo justo en el centro](/assets/articles/modelo-produccion/fig-montecarlo.png)

Este es el resultado que cambia la conversación de planeamiento:

| Escenario | Cu (t) | Probabilidad de cumplir |
|---|---:|---:|
| P10 (pesimista) | 5,151 | 90 % |
| **P50 / Plan determinístico** | **5,872 / 5,879** | **50 %** |
| Compromiso P80 | 5,400 | ~80 % |
| P90 (optimista) | 6,685 | 10 % |

<div class="callout-success">
  <div class="callout-icon">{% include icons/check-circle.svg class="h-5 w-5" %} La lección central</div>
  El plan determinístico (5,879 t) tiene apenas <strong>50 % de probabilidad de cumplirse</strong>, porque es el centro de la distribución. Comprometer ese número es lanzar una moneda. Un compromiso defendible se toma más abajo: <strong>5,400 t</strong> se cumplen con cerca del <strong>80 %</strong> de probabilidad (P80), el estándar habitual para comprometer producción. La diferencia entre el P50 y el P80 es la <strong>reserva de riesgo</strong> que el número puntual no muestra.
</div>


<a id="reconciliacion" class="anchor-clean"></a>
## 9) Reconciliación: el modelo contra la realidad

El cierre del ciclo es la **reconciliación**: comparar lo planeado con lo realmente producido mediante factores. Un factor cercano a 1.0 valida el modelo; una desviación sistemática lo corrige.

```python
cu_real_t = 5610.0                       # t de Cu según balance de planta
factor = cu_real_t / df.cu_t.sum()       # 0.954
```

Un factor de **0.954** cae dentro de la tolerancia habitual de ±5 %: el modelo predice bien y no requiere corrección. Fuera de ese rango, la desviación apunta a dilución no contabilizada, un modelo de recursos sesgado o una recuperación distinta a la supuesta. La reconciliación es lo que mantiene honesto al modelo mes a mes.


<a id="conclusiones" class="anchor-clean"></a>
## 10) Conclusiones

- La **contabilidad metalúrgica** (tonelaje × ley × recuperación, con las unidades bien puestas) es un modelo de producción completo en unas veinte líneas: **5,879 t de Cu, 5,106 oz de Au, US$ 61.4 M**.
- El **oro subproducto** aporta el 19 % del ingreso y baja la ley de corte del cobre de 0.246 % a 0.119 %: un subproducto cambia qué material es mineral.
- La **sensibilidad** ordena las palancas: precio y ley de Cu mandan; el tonelaje pesa menos que la intuición de campo. El foco de valor está en la ley alimentada y la exposición al precio.
- La **incertidumbre** es la lección central: el plan determinístico se cumple solo el 50 % de las veces. Comprometer producción exige bajar al P80.
- La **reconciliación** cierra el ciclo y mantiene calibrado el modelo.

El modelo de veinte líneas no es el final: es el esqueleto sobre el que se montan la ley de corte, la sensibilidad, la incertidumbre y la reconciliación. Ahí está la diferencia entre un cálculo y una herramienta de planeamiento.


<a id="refs" class="anchor-clean"></a>
## 11) Referencias

<div class="references" markdown="1">

Hustrulid, W., Kuchta, M., & Martin, R. (2013). **Open Pit Mine Planning and Design** (3rd ed.). CRC Press. Referencia integral de planeamiento de rajo, incluida la definición económica de mineral y el rol de la ley de corte.

Wills, B. A., & Finch, J. A. (2016). **Wills' Mineral Processing Technology** (8th ed.). Butterworth-Heinemann. Fundamentos de recuperación metalúrgica y balance de planta que sostienen la contabilidad de metal.

Rendu, J.-M. (2014). **An Introduction to Cut-off Grade Estimation** (2nd ed.). SME. Tratamiento moderno de la ley de corte, incluidos los créditos por subproducto.

Lane, K. F. (1988). **The Economic Definition of Ore: Cut-off Grades in Theory and Practice**. Mining Journal Books. Obra clásica que formaliza la ley de corte como una decisión económica.

Morley, C. (2003). **Beyond reconciliation: a proactive approach to using mining data.** Mining Magazine. Discute el uso de factores de reconciliación mina-modelo-planta para mantener calibrados los modelos de producción.

</div>
