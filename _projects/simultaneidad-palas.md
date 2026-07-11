---
layout: project
title: "Dashboard de Simultaneidad de Palas - Monitoreo de Carguío"
subtitle: "Disponibilidad de flota, Gantt día/noche y pérdidas de producción por pala, en tiempo real."
date: 2026-07-06
tags: [Python, Dash, Plotly, Minería, Mantenimiento, Dashboard]
stack: [Python, Dash, Plotly, Pandas, Render]
cover: /assets/projects/simultaneidad-palas/social-preview.png
thumbnail: /assets/projects/simultaneidad-palas/thumb.png
org: "Proyecto personal"
roles: ["Data Analyst", "Mining Engineer"]
repo: "https://github.com/nrgarridoa/simultaneidad_palas"
status: published

embed:
  type: html
  url: "https://simultaneidad-palas.onrender.com"
  aspect_ratio: "16/9"
  poster: "/assets/projects/simultaneidad-palas/social-preview.png"

data_overview: >
  Datos sintéticos (seed=42) de una flota de 15 palas (10 P&H 4100XPC, 5 EX5600-6) en tajo abierto durante el año 2025 completo: cerca de 113,800 eventos operativos individuales (estado, subtipo, turno, duración, producción y toneladas perdidas) y 5,475 resúmenes diarios por pala.

results: >
  Disponibilidad de flota de 85.3%, en el límite del benchmark típico (85%). El 56% de las toneladas perdidas (61.3M ton) son por demoras operativas evitables, no fallas de equipo (44%, 48.5M ton): la mayor oportunidad de mejora está en la gestión operativa, no solo en mantenimiento. El mantenimiento es ~60/40 reactivo/programado, y la flota P&H concentra el 70% de la pérdida total por ser la de mayor tamaño y capacidad.

metrics:
  disponibilidad: "85.3%"
  perdidas_demora: "61.3M ton (56%)"
  perdidas_falla: "48.5M ton (44%)"
  palas: "15"

gallery:
  - /assets/projects/simultaneidad-palas/estados_equipos.png
  - /assets/projects/simultaneidad-palas/kpis.png
  - /assets/projects/simultaneidad-palas/pareto.png
  - /assets/projects/simultaneidad-palas/disponibilidad_pala_semana.png
  - /assets/projects/simultaneidad-palas/disponibilidad_pala_mes.png
---
