---
title: "Python para Geología"
subtitle: "Procesamiento de datos geológicos, sondajes y modelamiento espacial con Python, desde cero y con datos de un proyecto de exploración."
date: 2026-11-25
status: in_development   # published | in_development | draft
cover: /assets/courses/python-geologia/cover.svg
order: 3
category: "Python"   # herramienta/tecnología: genera el filtro en /cursos/
area: "Geología"            # área de aplicación: genera el filtro en /cursos/
level: "Básico - Intermedio"
hours: 70
module_start: 0   # el badge del temario parte en 0 (Módulo 0 · Arranque)
modality: "Asincrónico · A tu ritmo"
access: "Por postulación"
tags: [Python, Geología, Geoestadística, Exploración, Machine Learning]
tools: [Python, Pandas, NumPy, GeoPandas, Matplotlib, Jupyter]

# Contenido real. IMPORTANTE: los ids de sesión son posicionales (m1s1, m1s2...). No reordenar módulos ni sesiones una vez cargados videos, materiales o evaluaciones en el aula.
audience: >
  Geólogos, ingenieros de minas y estudiantes que quieren automatizar el trabajo con sus datos. No necesitas haber programado antes: el curso parte de la instalación y practica desde la primera cápsula con datos sintéticos de un proyecto de exploración tipo pórfido Cu-Au.

prerequisites:
  - Manejo básico de computadora
  - Conocimientos generales de geología o minería (sondajes, leyes, mapas)
  - No se requiere experiencia previa programando
  - Computadora con Windows, Mac o Linux para instalar el entorno

outcomes:
  - Programar en Python y trabajar en Jupyter con soltura, desde la instalación del entorno
  - Limpiar, filtrar y resumir bases de datos de sondajes y ensayes con NumPy y pandas
  - Construir mapas, histogramas y paneles de informe con estilo profesional en matplotlib
  - Analizar datos estructurales y geoquímicos, y producir estereogramas, diagramas de Harker y TAS
  - Trabajar con coordenadas y mapas geoespaciales, e interpolar leyes con variogramas, IDW y kriging
  - Detectar y localizar eventos sísmicos y entrenar modelos de machine learning con validación espacial
  - Entregar un proyecto final reproducible que genera un informe HTML de una campaña de perforación

features:
  - Práctica desde la primera cápsula, con ejercicios de revisión automática
  - Notebooks de clase y notebooks con soluciones
  - Presentación de apoyo y hoja de resumen por módulo
  - Datos sintéticos de un proyecto de exploración tipo pórfido Cu-Au
  - Proyecto final con un script y un informe HTML reproducibles

modules:
  - title: "Módulo 0 · Arranque"
    summary: "Instalación del entorno y primeros pasos en Jupyter, con un primer gráfico de sondaje."
    sessions:
      - title: "Cápsula 0.1: Instala tu entorno"
        duration: "50 min"
        summary: "Explicar qué hacen Python, Jupyter, una librería, pip y un entorno virtual, y por qué se usan juntos"
        status: available
      - title: "Cápsula 0.2: Tu primera celda"
        duration: "60 min"
        summary: "Distinguir celdas de código y de texto (Markdown), ejecutarlas y moverte con los atajos básicos"
        status: available
      - title: "Cápsula 0.3: Tu primer gráfico de sondaje"
        duration: "55 min"
        summary: "Cargar la tabla de ensayes con pandas y reconocer qué representa cada fila y cada columna"
        status: available
  - title: "Módulo 1 · Fundamentos de Python"
    summary: "Variables, texto, listas, diccionarios, condicionales, bucles y funciones, aplicados a muestras y ensayes."
    sessions:
      - title: "Cápsula 1.1: Variables y tipos de datos"
        duration: "50 min"
        summary: "Crear variables, reasignarlas y explicar qué hace el signo = (una etiqueta que apunta a un valor)"
        status: available
      - title: "Cápsula 1.2: Operaciones: tonelaje, ley y metal contenido"
        duration: "55 min"
        summary: "Usar los operadores aritméticos de Python (incluidos // y %) respetando la jerarquía de operaciones"
        status: available
      - title: "Cápsula 1.3: Texto y f-strings: etiquetas de muestras"
        duration: "55 min"
        summary: "Explicar por qué el texto es inmutable y extraer partes de un código con índices y rebanadas"
        status: available
      - title: "Cápsula 1.4: Listas: los ensayes de un sondaje"
        duration: "35 min"
        summary: "Guardar muchos valores en una lista y entender por qué se cuenta desde 0"
        status: available
      - title: "Cápsula 1.5: Diccionarios: la ficha de una muestra"
        duration: "50 min"
        summary: "Representar una muestra con pares clave-valor y consultarla con [] o con get sin que el programa se detenga"
        status: available
      - title: "Cápsula 1.6: Condicionales: ¿mineral o desmonte?"
        duration: "50 min"
        summary: "Escribir comparaciones y combinarlas con and, or y not, incluida la cadena 0.5 <= grade < 1.0"
        status: available
      - title: "Cápsula 1.7: Bucles: recorrer todas las muestras"
        duration: "55 min"
        summary: "Recorrer una lista con for y explicar qué valor toma la variable en cada vuelta"
        status: available
      - title: "Cápsula 1.8: Funciones: tus propias herramientas"
        duration: "55 min"
        summary: "Definir una función con def, parámetros y return, y llamarla con argumentos por posición o por nombre"
        status: available
      - title: "Cápsula 1.9: Leer errores sin miedo"
        duration: "55 min"
        summary: "Leer un traceback e identificar el tipo de error, su causa y la línea donde ocurrió"
        status: available
      - title: "Cápsula 1.10: Mini-proyecto: reporte de un sondaje"
        duration: "60 min"
        summary: "Descomponer un encargo real en funciones pequeñas y escribirlas una por una"
        status: available
  - title: "Módulo 2 · NumPy: cálculo con arrays"
    summary: "Cálculo vectorizado, máscaras y estadística de leyes, hasta una curva de ley de corte."
    sessions:
      - title: "Cápsula 2.1: Arrays: la libreta de miles de ensayes"
        duration: "50 min"
        summary: "Explicar por qué un array de NumPy es más rápido y compacto que una lista (memoria contigua y un solo tipo)"
        status: available
      - title: "Cápsula 2.2: Operaciones vectorizadas: sin bucles"
        duration: "55 min"
        summary: "Aplicar una operación a todo un array y combinar dos arrays elemento a elemento, sin bucles"
        status: available
      - title: "Cápsula 2.3: Máscaras booleanas: filtrar mineral de desmonte"
        duration: "55 min"
        summary: "Consultar y rebanar arrays con índices, rebanadas y posiciones en dos dimensiones"
        status: available
      - title: "Cápsula 2.4: Estadística de leyes con datos reales"
        duration: "60 min"
        summary: "Calcular y comparar media, mediana, desviación estándar (con ddof), coeficiente de variación y percentiles de las leyes reales"
        status: available
      - title: "Cápsula 2.5: Matrices: un modelo de bloques en miniatura"
        duration: "50 min"
        summary: "Describir una matriz con shape, ndim y size, y consultar bloques, filas y columnas con [fila, columna]"
        status: available
      - title: "Cápsula 2.6: Mini-proyecto: curva de ley de corte"
        duration: "60 min"
        summary: "Diseñar un análisis por pasos (pregunta, datos, método, salida) y verificar cada paso con una cifra comprobable"
        status: available
  - title: "Módulo 3 · Pandas: tablas de ensayes"
    summary: "Selección, filtros, limpieza, agrupación y unión de tablas de collars y ensayes."
    sessions:
      - title: "Cápsula 3.1: DataFrames: tu tabla de ensayes"
        duration: "60 min"
        summary: "Explicar qué es un DataFrame (columnas Series con un índice común) y qué dice cada tipo de dato (dtype)"
        status: available
      - title: "Cápsula 3.2: Seleccionar columnas y filas"
        duration: "60 min"
        summary: "Elegir columnas con [] y saber cuándo el resultado es una Series y cuándo un DataFrame"
        status: available
      - title: "Cápsula 3.3: Filtrar y ordenar: ley de corte en pandas"
        duration: "60 min"
        summary: "Filtrar filas con una máscara booleana y combinar condiciones con &, | y ~ (con sus paréntesis)"
        status: available
      - title: "Cápsula 3.4: Columnas nuevas y limpieza de datos"
        duration: "60 min"
        summary: "Crear columnas calculadas (longitud, metal, clase de ley) y evitar el SettingWithCopyWarning con .copy()"
        status: available
      - title: "Cápsula 3.5: Agrupar y resumir con groupby"
        duration: "60 min"
        summary: "Explicar el patrón dividir-aplicar-combinar y resumir una tabla por una o por varias columnas"
        status: available
      - title: "Cápsula 3.6: Unir tablas: collars y ensayes"
        duration: "60 min"
        summary: "Explicar por qué los datos viven en varias tablas y unirlas con merge por una clave común"
        status: available
      - title: "Cápsula 3.7: Mini-proyecto: reporte por sondaje"
        duration: "60 min"
        summary: "Diseñar un reporte a partir de las preguntas que debe responder y traducirlas a columnas calculadas"
        status: available
  - title: "Módulo 4 · Visualización con matplotlib"
    summary: "Mapas de planta, histogramas, subplots y estilo profesional para un panel de campaña."
    sessions:
      - title: "Cápsula 4.1: Figura y ejes: anatomía de un gráfico"
        duration: "55 min"
        summary: "Distinguir Figure, Axes y Artists, y construir un gráfico con el estilo orientado a objetos (fig, ax)"
        status: available
      - title: "Cápsula 4.2: Mapa de planta: dispersión con color y tamaño"
        duration: "55 min"
        summary: "Construir un mapa de planta con scatter que codifique ley (color) y profundidad (tamaño)"
        status: available
      - title: "Cápsula 4.3: Histogramas y boxplots: la distribución de leyes"
        duration: "60 min"
        summary: "Elegir los intervalos (bins) de un histograma comparando varios valores y explicando el efecto"
        status: available
      - title: "Cápsula 4.4: Subplots: una columna por sondaje"
        duration: "60 min"
        summary: "Crear una cuadrícula de ejes con plt.subplots, recorrerla con zip y ravel y saber qué devuelve en cada caso"
        status: available
      - title: "Cápsula 4.5: Estilo profesional: colores y firma"
        duration: "60 min"
        summary: "Aplicar un estilo global con rcParams, y cambiarlo solo por un momento con rc_context"
        status: available
      - title: "Cápsula 4.6: Mini-proyecto: panel de informe de campaña"
        duration: "55 min"
        summary: "Diseñar la lámina antes de programarla: qué pregunta responde cada panel y qué cifras se verificarán"
        status: available
  - title: "Módulo 5 · Geología estructural"
    summary: "Rumbo y buzamiento, estereogramas, densidad de polos y rosas de diaclasas."
    sessions:
      - title: "Cápsula 5.1: Rumbo, buzamiento y dirección de buzamiento"
        duration: "50 min"
        summary: "Convertir entre las notaciones de campo (cuadrante, rumbo/buzamiento, dirección de buzamiento/buzamiento) con la regla de la mano derecha"
        status: available
      - title: "Cápsula 5.2: Datos estructurales con pandas"
        duration: "60 min"
        summary: "Explorar y validar la tabla de mediciones (tamaños, tipos, rangos y duplicados) antes de analizarla"
        status: available
      - title: "Cápsula 5.3: El estereograma: planos y polos"
        duration: "60 min"
        summary: "Explicar cómo se construye un estereograma (hemisferio inferior, círculo primitivo, centro) y por qué la geología estructural usa la proyección de igual área"
        status: available
      - title: "Cápsula 5.4: Densidad de polos y orientación media"
        duration: "60 min"
        summary: "Leer un diagrama de contornos de densidad de polos: qué cuenta, qué significa cada valor y cuánto vale el ruido"
        status: available
      - title: "Cápsula 5.5: La rosa de diaclasas"
        duration: "60 min"
        summary: "Construir una rosa de rumbos bidireccional (conteo, duplicado a +180° y dibujo en un eje polar orientado como brújula)"
        status: available
      - title: "Cápsula 5.6: Mini-proyecto: análisis estructural por dominio"
        duration: "60 min"
        summary: "Diseñar un análisis estructural por dominio: definir sets, medida de orientación y criterio de comparación antes de programar"
        status: available
  - title: "Módulo 6 · Geoquímica"
    summary: "Límites de detección, valores atípicos, diagramas de Harker, TAS y multielementales."
    sessions:
      - title: "Cápsula 6.1: Datos geoquímicos: unidades, límites de detección y calidad"
        duration: "60 min"
        summary: "Convertir entre óxido y elemento, y entre ppm, g/t y %, con factores calculados desde masas atómicas"
        status: available
      - title: "Cápsula 6.2: Estadística y valores atípicos"
        duration: "55 min"
        summary: "Distinguir valor atípico (estadístico), error (dato incorrecto) y anomalía (señal geológica) y aplicar pruebas para decidir"
        status: available
      - title: "Cápsula 6.3: Diagramas de Harker"
        duration: "60 min"
        summary: "Construir un diagrama de Harker y leer su tendencia (qué óxidos suben, bajan o no cambian con la sílice)"
        status: available
      - title: "Cápsula 6.4: Clasificación química: el diagrama TAS"
        duration: "55 min"
        summary: "Clasificar rocas volcánicas por SiO2 con una función y con pd.cut, y explicar qué le falta a esta versión frente al TAS completo"
        status: available
      - title: "Cápsula 6.5: Normalización y diagramas multielementales"
        duration: "55 min"
        summary: "Normalizar elementos traza contra una referencia (manto primitivo) con pandas y explicar por qué se hace"
        status: available
      - title: "Cápsula 6.6: Mini-proyecto: anomalías de cobre"
        duration: "55 min"
        summary: "Definir un umbral de anomalía con un percentil y con un método robusto, y comparar cuántas muestras marca cada uno"
        status: available
  - title: "Módulo 7 · Datos geoespaciales"
    summary: "Sistemas de referencia, GeoDataFrames, mapas, operaciones espaciales y sondajes en XYZ."
    sessions:
      - title: "Cápsula 7.1: Coordenadas y sistemas de referencia"
        duration: "55 min"
        summary: "Explicar qué define un CRS (datum, proyección y unidades) y por qué UTM sirve para medir y los grados no"
        status: available
      - title: "Cápsula 7.2: GeoDataFrames: tablas con geometría"
        duration: "55 min"
        summary: "Construir un GeoDataFrame de puntos a partir de columnas de coordenadas y declarar su CRS"
        status: available
      - title: "Cápsula 7.3: Mapas del proyecto con geopandas"
        duration: "55 min"
        summary: "Verificar las capas del proyecto (tipo, CRS, áreas y longitudes en metros) antes de dibujarlas"
        status: available
      - title: "Cápsula 7.4: Operaciones espaciales: buffer, distancia y spatial join"
        duration: "55 min"
        summary: "Elegir el predicado espacial correcto (within, contains, intersects, touches) para una pregunta de contacto o pertenencia"
        status: available
      - title: "Cápsula 7.5: Sondajes en el espacio: de collar a coordenadas XYZ"
        duration: "55 min"
        summary: "Calcular la posición (x, y, z) de una muestra a partir del collar, el azimut, el dip y la profundidad"
        status: available
      - title: "Cápsula 7.6: Mini-proyecto: mapa de anomalías del proyecto"
        duration: "55 min"
        summary: "Diseñar un análisis espacial paso a paso (pregunta, capas, método, entregable) antes de escribir código"
        status: available
  - title: "Módulo 8 · Geoestadística"
    summary: "Variograma, modelo de variograma, IDW y kriging ordinario hasta un mapa de leyes y tonelaje."
    sessions:
      - title: "Cápsula 8.1: Autocorrelación espacial: lo cercano se parece"
        duration: "55 min"
        summary: "Explicar por qué la ubicación de una muestra es información y qué es la autocorrelación espacial"
        status: available
      - title: "Cápsula 8.2: El variograma experimental"
        duration: "60 min"
        summary: "Calcular la semivarianza de un grupo de pares y construir el variograma experimental por intervalos (lag y tolerancia)"
        status: available
      - title: "Cápsula 8.3: Ajustar un modelo de variograma"
        duration: "60 min"
        summary: "Programar el modelo esférico y explicar qué controla cada parámetro (pepita, meseta, alcance)"
        status: available
      - title: "Cápsula 8.4: Interpolación por inverso de la distancia (IDW)"
        duration: "60 min"
        summary: "Calcular a mano y con código una estimación IDW como promedio ponderado de vecinos"
        status: available
      - title: "Cápsula 8.5: Kriging ordinario"
        duration: "60 min"
        summary: "Armar y resolver el sistema de kriging ordinario con NumPy en un ejemplo de tres puntos, y verificarlo ecuación por ecuación"
        status: available
      - title: "Cápsula 8.6: Mini-proyecto: mapa de leyes y tonelaje"
        duration: "60 min"
        summary: "Construir una malla de bloques, estimar su ley con kriging y mapear la ley y la incertidumbre"
        status: available
  - title: "Módulo 9 · Sismología y geofísica"
    summary: "Catálogo sísmico, Gutenberg-Richter, ondas con ObsPy, STA/LTA, localización y magnetometría."
    sessions:
      - title: "Cápsula 9.1: El catálogo sísmico de una mina"
        duration: "60 min"
        summary: "Leer un catálogo sísmico, convertir sus fechas a datetime y comprobar su calidad antes de analizarlo"
        status: available
      - title: "Cápsula 9.2: Magnitud y la ley de Gutenberg-Richter"
        duration: "60 min"
        summary: "Explicar por qué la magnitud es logarítmica y calcular cuántas veces más energía libera un evento que otro"
        status: available
      - title: "Cápsula 9.3: Ondas sísmicas con ObsPy"
        duration: "60 min"
        summary: "Leer un registro MiniSEED con ObsPy y calcular, a partir de su frecuencia de muestreo, la frecuencia de Nyquist y la duración"
        status: available
      - title: "Cápsula 9.4: Detectar llegadas con STA/LTA"
        duration: "60 min"
        summary: "Calcular la razón STA/LTA, explicar qué mide y por qué está acotada por nlta / nsta"
        status: available
      - title: "Cápsula 9.5: Localizar el sismo con las ondas P y S"
        duration: "60 min"
        summary: "Deducir por qué el tiempo S-P da la distancia y calcularla con Vp y Vs"
        status: available
      - title: "Cápsula 9.6: Geofísica de exploración: un perfil magnético"
        duration: "60 min"
        summary: "Separar la anomalía local del campo regional y entender por qué el ajuste del regional puede sesgarla"
        status: available
      - title: "Cápsula 9.7: Mini-proyecto: monitoreo sísmico de una mina"
        duration: "60 min"
        summary: "Armar un reporte de monitoreo (tasa diaria, valor b por periodos con su error y ubicación) a partir del catálogo"
        status: available
  - title: "Módulo 10 · Machine learning"
    summary: "Regresión, clasificación, árboles, clustering y validación espacial para clasificar litologías."
    sessions:
      - title: "Cápsula 10.1: El flujo de ML y la regresión lineal"
        duration: "55 min"
        summary: "Aplicar el flujo definir X e y, separar, entrenar, predecir y evaluar con una regresión lineal"
        status: available
      - title: "Cápsula 10.2: Clasificar rocas con su geoquímica"
        duration: "55 min"
        summary: "Plantear un problema de clasificación y evaluarlo con exactitud, línea base y matriz de confusión"
        status: available
      - title: "Cápsula 10.3: Árboles y bosques aleatorios"
        duration: "55 min"
        summary: "Leer las reglas de un árbol de decisión y distinguir subajuste de sobreajuste con datos de entrenamiento y prueba"
        status: available
      - title: "Cápsula 10.4: Agrupar muestras sin etiquetas: clustering"
        duration: "60 min"
        summary: "Aplicar K-means a datos geoquímicos preparados (logaritmo y estandarización) y elegir k con inercia y silueta"
        status: available
      - title: "Cápsula 10.5: Validar bien con datos espaciales"
        duration: "60 min"
        summary: "Medir la fuga espacial con la distancia de cada muestra de prueba a su vecina más cercana del entrenamiento"
        status: available
      - title: "Cápsula 10.6: Mini-proyecto: clasificar litologías en sondajes"
        duration: "60 min"
        summary: "Construir un clasificador de litología y validarlo dejando sondajes completos fuera, contra una línea base"
        status: available
  - title: "Módulo 11 · Proyecto final"
    summary: "Un script reproducible que genera un informe HTML de una campaña de perforación."
    sessions:
      - title: "Cápsula 11.1: Planificar un informe automatizado"
        duration: "60 min"
        summary: "Decidir cuándo conviene un script y cuándo un notebook, y qué debe quedar registrado para poder repetir un informe"
        status: available
      - title: "Cápsula 11.2: Funciones reutilizables y pruebas"
        duration: "60 min"
        summary: "Convertir un cálculo suelto de un notebook en una función pura con parámetros, return y docstring"
        status: available
      - title: "Cápsula 11.3: De notebook a script (.py)"
        duration: "60 min"
        summary: "Reconocer las partes de un script ejecutable (docstring, imports, funciones, main y el guardián __main__) y por qué las rutas deben construirse con pathlib"
        status: available
      - title: "Cápsula 11.4: Generar el informe HTML"
        duration: "60 min"
        summary: "Armar un documento HTML completo con f-strings, escapando el texto con html.escape para que ningún dato rompa la página"
        status: available
      - title: "Cápsula 11.5: Validar, registrar y reproducir"
        duration: "60 min"
        summary: "Distinguir qué errores detecta validate() y cuáles no, y añadir comprobaciones de rango y de parámetros (corte nan, archivo inexistente)"
        status: available
      - title: "Cápsula 11.6: Tu proyecto final y cierre del curso"
        duration: "60 min"
        summary: "Probar la línea de comandos del script con una tabla de casos (argumentos, código de salida esperado) y detectar una opción sin texto de ayuda"
        status: available
---

Curso en cápsulas, cada una sobre un tema específico, pensado para quien nunca ha escrito código. Cada cápsula incluye un notebook con ejercicios de revisión automática, su versión con soluciones y una presentación de apoyo. El núcleo de cada una toma entre 35 y 60 minutos; las extensiones marcadas como opcionales suman un poco más para quien quiera profundizar.
