// Prepara el Jupyter del aula (JupyterLite, en /jupyter/): escribe en su almacenamiento del navegador el notebook de la
// sesion y los archivos del curso (geotools y datos), para que el notebook los encuentre como en una carpeta normal.
// El almacenamiento (IndexedDB "AulaJupyter") es el mismo que usa JupyterLite, por eso aula y /jupyter/ deben compartir origen.
const DB = 'AulaJupyter';
const STORES = ['files', 'counters', 'checkpoints', 'settings'];
const TEXT_EXT = new Set(['py', 'txt', 'csv', 'json', 'geojson', 'md', 'prj', 'cpg', 'html', 'ipynb']);
const MIME = { py: 'text/x-python', csv: 'text/csv', json: 'application/json', geojson: 'application/geo+json', txt: 'text/plain', md: 'text/markdown', html: 'text/html' };

function openDb() {
  const open = (version) => new Promise((resolve, reject) => {
    const req = version ? indexedDB.open(DB, version) : indexedDB.open(DB);
    req.onupgradeneeded = () => { const db = req.result; for (const st of STORES) if (!db.objectStoreNames.contains(st)) db.createObjectStore(st); };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('No se pudo abrir el almacenamiento de Jupyter'));
  });
  // Sin version: respeta la que ya haya creado JupyterLite (suele ser mayor que 1). Si falta algun almacen, se sube la version.
  return open().then(async (db) => {
    if (STORES.every((st) => db.objectStoreNames.contains(st))) return db;
    const v = db.version + 1; db.close(); return open(v);
  });
}
const wrap = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });

function toBase64(u8) {
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
  return btoa(s);
}
function asText(u8) { try { return new TextDecoder('utf-8', { fatal: true }).decode(u8); } catch { return null; } }

/**
 * files: [{ path: 'geotools/revisar.py', bytes: Uint8Array, overwrite: true }, ...]
 * notebook: { path, json } (se escribe solo si todavia no existe: asi no se pierde el trabajo del alumno)
 * Devuelve { notebookExists }.
 */
export async function seedJupyter({ files, notebook }) {
  const db = await openDb();
  const now = new Date().toISOString();
  const base = (path, type, extra) => ({ name: path.split('/').pop(), path, last_modified: now, created: now, writable: true, type, ...extra });
  const dirs = new Set();
  const addDirs = (path) => { const parts = path.split('/'); parts.pop(); let acc = ''; for (const p of parts) { acc = acc ? `${acc}/${p}` : p; dirs.add(acc); } };

  const tx = db.transaction('files', 'readwrite'); const st = tx.objectStore('files');
  const puts = [];
  for (const f of files) {
    addDirs(f.path);
    if (!f.overwrite && await wrap(st.get(f.path)) !== undefined) continue;
    const ext = (f.path.split('.').pop() || '').toLowerCase();
    const text = TEXT_EXT.has(ext) ? asText(f.bytes) : null;
    puts.push([f.path, text !== null
      ? base(f.path, 'file', { format: 'text', mimetype: MIME[ext] || 'text/plain', content: text, size: f.bytes.length })
      : base(f.path, 'file', { format: 'base64', mimetype: 'application/octet-stream', content: toBase64(f.bytes), size: f.bytes.length })]);
  }
  addDirs(notebook.path);
  for (const d of dirs) if (await wrap(st.get(d)) === undefined) puts.push([d, base(d, 'directory', { format: 'json', mimetype: '', content: null, size: 0 })]);
  const exists = await wrap(st.get(notebook.path)) !== undefined;
  if (!exists) puts.push([notebook.path, base(notebook.path, 'notebook', { format: 'json', mimetype: 'application/json', content: notebook.json, size: JSON.stringify(notebook.json).length })]);
  for (const [k, v] of puts) st.put(v, k);
  await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });
  db.close();
  return { notebookExists: exists };
}

// ---------- adaptacion del notebook para Jupyter en el navegador ----------
const COMPAT = `# Compatibilidad con Jupyter en el navegador (Pyodide). Ejecútala una vez, sin cambios.
import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _n, _v in (("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str), ("complex", complex)):
        if not hasattr(np, _n):
            setattr(np, _n, _v)
try:
    import inspect, matplotlib.axes as _ax
    _bp = _ax.Axes.boxplot
    if "tick_labels" not in inspect.signature(_bp).parameters:
        def _boxplot(self, *a, tick_labels=None, **k):
            if tick_labels is not None:
                k["labels"] = tick_labels
            return _bp(self, *a, **k)
        _ax.Axes.boxplot = _boxplot
except Exception:
    pass
`;
const srcOf = (c) => [].concat(c.source || []).join('');
export const notebookUsesObspy = (nb) => (nb.cells || []).some((c) => c.cell_type === 'code' && /^\s*(?:import|from)\s+obspy/m.test(srcOf(c)));

/** Agrega al inicio una celda de compatibilidad (y la instalacion de mplstereonet si el notebook la usa). */
export function prepareNotebook(nb) {
  const code = (nb.cells || []).filter((c) => c.cell_type === 'code').map(srcOf).join('\n');
  const pip = /^\s*(?:import|from)\s+mplstereonet/m.test(code) ? 'import piplite\nawait piplite.install("mplstereonet")\n' : '';
  return {
    ...nb,
    cells: [
      { cell_type: 'markdown', metadata: {}, source: '**Jupyter en el navegador.** La celda siguiente adapta el entorno (Pyodide); ejecútala una vez y sigue con la cápsula. Tu trabajo se guarda en este navegador.' },
      { cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: (pip ? pip + '\n' : '') + COMPAT },
      ...(nb.cells || []),
    ],
  };
}
