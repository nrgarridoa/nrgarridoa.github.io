# Jupyter del aula (JupyterLite)

`/jupyter/` es una compilación estática de JupyterLite (Notebook 7 + kernel Python en el navegador con Pyodide).
Se genera con estos dos archivos de configuración:

```
python -m venv venv && venv/Scripts/pip install jupyterlite-core jupyterlite-pyodide-kernel
mkdir src && cp jupyter-lite.json overrides.json src/ && cd src
../venv/Scripts/jupyter-lite build --output-dir ../final
# luego: borrar *.map, rspack.config*.js y package.json, y copiar la carpeta a /jupyter/ del sitio
```

El aula (`assets/js/aula/jupyterseed.js`) escribe el notebook de la sesión y los archivos del curso en el almacenamiento
del navegador (IndexedDB "AulaJupyter") antes de abrir `/jupyter/notebooks/index.html?path=...`.
Pyodide se descarga de jsDelivr (versión fijada en `jupyter-lite.json`, la misma del cuaderno interactivo del aula).
