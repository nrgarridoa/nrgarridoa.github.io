---
layout: project
title: "Dashboard de Mantenimiento y Confiabilidad - Flota Minera"
subtitle: "Dashboard interactivo para el análisis de disponibilidad, fallas, costos y confiabilidad de una flota de 21 equipos pesados en minería a cielo abierto. Análisis Weibull, Pareto 80/20 y productividad de taller."
date: 2026-05-06
tags: [Power BI, DAX, Python, Minería, Mantenimiento, Confiabilidad, Weibull, MTBF, Dashboard]
stack: [Power BI, DAX, Python, Star Schema]
cover: /assets/projects/maintenance/social-preview.png
thumbnail: /assets/projects/maintenance/social-preview.png
org: "Proyecto personal"
roles: ["Data Analyst", "Reliability Engineer"]
repo: "https://github.com/nrgarridoa/powerbi-maintenance-reliability"
data_url: "https://github.com/nrgarridoa/powerbi-maintenance-reliability/tree/main/data"
status: published

embed:
  type: powerbi
  url: "https://app.powerbi.com/view?r=eyJrIjoiY2I4ZWI5MWMtZDVlNC00OTc5LTliMTgtOWZkMDVjZjAzYTRmIiwidCI6ImY3YWNmODc2LWU3ZTgtNDQ0Yy05NWFlLWY5NTQ4YWNmZTMyZiIsImMiOjR9"
  aspect_ratio: "16/9"
  poster: "/assets/projects/maintenance/preview.png"

data_overview: >
  74,000+ registros de operación, fallas y órdenes de trabajo de 21 equipos pesados 
  (9 camiones, 6 palas, 6 perforadoras) en una operación minera a cielo abierto en la 
  sierra peruana. Datos sintéticos generados con distribuciones Weibull realistas (seed=42), 
  3 años de ventana temporal (2023-2025). Modelo estrella con 11 tablas (7 dimensiones + 4 hechos). 
  Flota: Caterpillar (797F, 793F, 6060), Komatsu (930E, PC8000), Liebherr (R9800), 
  Epiroc (320XPC P&H, Pit Viper 271), DM45.

results: >
  El dashboard revela que el costo de no disponibilidad (USD 13.4M) supera al costo de 
  mantenimiento directo (USD 12M), evidenciando que cada dólar no invertido en preventivo 
  genera más de un dólar en paradas. El ratio PM/CM de 0.31 (vs 0.80 world-class) confirma 
  una estrategia 70% reactiva. El sistema hidráulico lidera en horas de parada y los componentes 
  con Beta Weibull > 3 (desgaste acelerado) son candidatos claros para mantenimiento basado en condición. 
  Solo el 54% de las OTs cumplen plazo.

metrics:
  disponibilidad: "82.6%"
  mtbf: "154.9 hrs"
  costo_mto: "USD 11.97M"
  ratio_pmcm: "0.31"

gallery:
  - /assets/projects/maintenance/availability.png
  - /assets/projects/maintenance/failures.png
  - /assets/projects/maintenance/reliability.png
  - /assets/projects/maintenance/costs.png
---
