---
layout: project
title: "Dashboard Control de Durezas - Perforación de Voladura"
subtitle: "Dashboard interactivo para el análisis y control de durezas en perforación de voladura en minería a cielo abierto. Visualización geoespacial 3D, rendimiento de equipos y análisis geológico."
date: 2026-05-06
tags: [Power BI, DAX, Power Query, Minería, Voladura, MWD, Geología, Dashboard]
stack: [Power BI, DAX, Power Query, Star Schema]
cover: /assets/projects/durezas/social-preview.png
thumbnail: /assets/projects/durezas/thumb.png
org: "Proyecto personal"
roles: ["Data Analyst", "Mining Engineer"]
repo: "https://github.com/nrgarridoa/powerbi-control-durezas"
data_url: "https://github.com/nrgarridoa/powerbi-control-durezas/tree/main/data"
status: published

embed:
  type: powerbi
  url: "https://app.powerbi.com/view?r=eyJrIjoiMWNhNWMwNmEtNjE4YS00NzgxLTkzOTUtYTliYjMyYTk0NTI3IiwidCI6ImY3YWNmODc2LWU3ZTgtNDQ0Yy05NWFlLWY5NTQ4YWNmZTMyZiIsImMiOjR9"
  aspect_ratio: "16/9"
  poster: "/assets/projects/durezas/preview.jpg"

data_overview: >
  97,988 registros MWD (Measurement While Drilling) de 582 taladros de voladura en un tajo abierto 
  en la sierra peruana (3,400-4,000 msnm), yacimiento tipo pórfido. Modelo estrella con 6 tablas. 
  Flota de 6 perforadoras: Pit Viper 271/351 (Epiroc) y DM-M3 (Caterpillar). 
  Limpieza rigurosa de datos en origen: outliers de sensor MWD, coordenadas invertidas, 
  zonas no operativas y alteraciones completadas con lógica geológica. 98% de calidad final.

results: >
  El dashboard permite identificar zonas de mayor dureza antes de planificar la perforación, 
  comparar rendimiento de equipos según tipo de roca (PenRate varía de 120.7 m/hr en roca blanda 
  a 34.4 m/hr en roca muy dura), analizar el impacto de la alteración hidrotermal en la dureza 
  y optimizar la asignación de equipos por zona. Eficiencia operativa del 95.2%.

metrics:
  taladros: "582"
  registros_mwd: "97,988"
  eficiencia: "95.2%"
  categorias_dureza: "4"

gallery:
  - /assets/projects/durezas/shot1.jpg
  - /assets/projects/durezas/shot2.jpg
  - /assets/projects/durezas/shot3.jpg
  - /assets/projects/durezas/shot4.jpg
---
