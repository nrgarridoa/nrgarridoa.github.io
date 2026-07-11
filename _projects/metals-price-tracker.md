---
layout: project
title: "Metals Price Tracker - Dashboard de Precios de Metales en Tiempo Real"
subtitle: "13 metales, 3 clases de instrumento (futuro, ETN, ETF): el dashboard distingue cada uno con su unidad de cotización real."
date: 2026-07-07
tags: [Python, Streamlit, Plotly, Finanzas, Commodities, Dashboard]
stack: [Python, Streamlit, Plotly, yfinance, Pandas, SQLite]
cover: /assets/projects/metals-price-tracker/social-preview.png
thumbnail: /assets/projects/metals-price-tracker/thumb.png
org: "Proyecto personal"
roles: ["Data Analyst", "Python Developer"]
repo: "https://github.com/nrgarridoa/metals_price_tracker"
status: published

embed:
  type: streamlit
  url: "https://metalspricetracker.streamlit.app/?embed=true"
  aspect_ratio: "16/9"
  poster: "/assets/projects/metals-price-tracker/social-preview.png"

data_overview: >
  Datos en vivo de Yahoo Finance para 13 metales (preciosos, industriales y estratégicos), clasificados en 3 tipos de instrumento: futuro directo, ETN atado a un subíndice y ETF de acciones mineras. Cada instrumento lleva su unidad de cotización real (USD/oz troy, USD/lb, USD/tonelada) y un respaldo local en SQLite con scheduler automático para cuando Yahoo Finance no responde.

results: >
  El dashboard distingue el tipo de cada instrumento en la propia interfaz (badge ETN/ETF y unidad real), normaliza metales de escala de precio muy distinta en un mismo gráfico de comparación (variación % desde el inicio del periodo) y marca como N/D la variación de instrumentos de baja liquidez en vez de mostrar ruido de actualización espaciada como si fuera movimiento de mercado real. El estado (metal, periodo, tipo de gráfico) queda reflejado en la URL, compartible tal cual.

metrics:
  metales: "13"
  clases_instrumento: "3"
  tickers_reemplazados: "2 (ETN redimidos en 2023)"

gallery:
  - /assets/projects/metals-price-tracker/home_dark.png
  - /assets/projects/metals-price-tracker/home_light.png
  - /assets/projects/metals-price-tracker/detalle_etf.png
  - /assets/projects/metals-price-tracker/comparar.png
---
