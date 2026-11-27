// Kernel de Python del cuaderno interactivo: Pyodide (Python compilado a WebAssembly) dentro de un Worker.
// Se comunica con notebook.js por mensajes. No hay servidor: todo corre en el navegador del alumno.
/* global importScripts, loadPyodide */
const PYODIDE_VERSION = '0.29.5';
const BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// Importaciones del codigo -> paquetes (los que Pyodide no detecta solo)
const PIP = { mplstereonet: 'mplstereonet' };

let py = null;
let current = null;
const post = (o) => self.postMessage(o);

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') await init(m);
    else if (m.type === 'run') await run(m);
  } catch (err) {
    post({ type: 'fatal', error: String((err && err.message) || err) });
  }
};

async function init({ zip, cwd }) {
  post({ type: 'status', msg: 'Descargando Python (solo la primera vez)…' });
  importScripts(BASE + 'pyodide.js');
  py = await loadPyodide({ indexURL: BASE });
  py.setStdout({ batched: (s) => current != null && post({ type: 'out', cell: current, stream: 'stdout', text: s + '\n' }) });
  py.setStderr({ batched: (s) => current != null && post({ type: 'out', cell: current, stream: 'stderr', text: s + '\n' }) });
  if (zip) py.unpackArchive(zip, 'zip', { extractDir: '/course' });
  py.globals.set('CWD', cwd || '/course');
  py.runPython(KERNEL);
  post({ type: 'ready' });
}

// Paquetes que necesita el codigo (incluidos los modulos de geotools que importe).
async function prepare(code) {
  const msg = (t) => post({ type: 'status', msg: t });
  const src = [code];
  for (const m of code.matchAll(/geotools(?:\.|\s+import\s+)(\w+)/g)) {
    try { src.push(py.FS.readFile(`/course/geotools/${m[1]}.py`, { encoding: 'utf8' })); } catch { /* no es un modulo */ }
  }
  if (/informe_campana/.test(code)) { try { src.push(py.FS.readFile('/course/modulo_11_proyecto_final/informe_campana.py', { encoding: 'utf8' })); } catch { /* nada */ } }
  await py.loadPackagesFromImports(src.join('\n'), { messageCallback: msg, errorCallback: msg });
  for (const [imp, pkg] of Object.entries(PIP)) {
    if (new RegExp(`\\b${imp}\\b`).test(code) && !prepare.done[pkg]) {
      msg(`Instalando ${pkg}…`);
      await py.loadPackage('micropip');
      await py.pyimport('micropip').install(pkg);
      prepare.done[pkg] = true;
    }
  }
}
prepare.done = {};

async function run({ cell, code }) {
  current = cell;
  try {
    await prepare(code);
    post({ type: 'status', msg: 'Ejecutando…' });
    py.globals.set('SRC', code);
    const res = py.runPython('__run_cell(SRC)').toJs({ dict_converter: Object.fromEntries, create_proxies: false });
    post({ type: 'result', cell, ...res });
  } catch (err) {
    post({ type: 'result', cell, error: { ename: 'Error', evalue: String((err && err.message) || err), traceback: '' }, display: [], figures: [], checks: [] });
  } finally {
    current = null;
  }
}

const KERNEL = String.raw`
import sys, os, ast, io, base64, types, traceback, warnings
os.environ["MPLBACKEND"] = "Agg"
warnings.filterwarnings("ignore", message=".*non-interactive.*")
os.makedirs(CWD, exist_ok=True)
os.chdir(CWD)
sys.path.insert(0, "/course")

_ns = {"__name__": "__main__"}
_DISPLAY = []
_CHECKS = []

def _emit(o):
    if o is None:
        return
    for meth, mime in (("_repr_html_", "text/html"), ("_repr_markdown_", "text/markdown"), ("_repr_png_", "image/png")):
        f = getattr(o, meth, None)
        if f is not None:
            try:
                v = f()
            except Exception:
                v = None
            if v is not None:
                if mime == "image/png" and isinstance(v, (bytes, bytearray)):
                    v = base64.b64encode(bytes(v)).decode()
                _DISPLAY.append({"mime": mime, "data": v})
                return
    if hasattr(o, "savefig"):
        buf = io.BytesIO()
        o.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        _DISPLAY.append({"mime": "image/png", "data": base64.b64encode(buf.getvalue()).decode()})
        return
    _DISPLAY.append({"mime": "text/plain", "data": repr(o)})

# --- Compatibilidad con IPython/Jupyter: lo que usan los notebooks del curso
class _HTML:
    def __init__(self, data=""):
        self.data = data
    def _repr_html_(self):
        return self.data
class _Markdown:
    def __init__(self, data=""):
        self.data = data
    def _repr_markdown_(self):
        return self.data
def _display(*objs, **kw):
    for o in objs:
        _emit(o)
class _Events:
    def register(self, *a, **k):
        return None
    def unregister(self, *a, **k):
        return None
class _FakeIP:
    config = {"IPKernelApp": {}}
    events = _Events()
    def run_line_magic(self, *a, **k):
        return None
    def run_cell_magic(self, *a, **k):
        return None
_ip = types.ModuleType("IPython")
_ipd = types.ModuleType("IPython.display")
_ipd.HTML, _ipd.Markdown, _ipd.display = _HTML, _Markdown, _display
_ip.display = _ipd
_ip.get_ipython = lambda: None   # matplotlib no debe creer que hay un kernel IPython real
sys.modules["IPython"] = _ip
sys.modules["IPython.display"] = _ipd

def _compat(src=""):
    # Bibliotecas antiguas (mplstereonet) usan alias de numpy que ya no existen
    try:
        import numpy as np
        for name, val in (("float", float), ("int", int), ("bool", bool), ("object", object), ("str", str), ("complex", complex)):
            if not hasattr(np, name):
                setattr(np, name, val)
    except Exception:
        pass

    # matplotlib < 3.9 no conoce tick_labels (nombre nuevo de "labels" en boxplot)
    if "matplotlib.axes" not in sys.modules and ("matplotlib" in src or "plt." in src or "graficos" in src):
        try:
            import matplotlib.axes
        except Exception:
            pass
    ax = sys.modules.get("matplotlib.axes")
    if ax is not None:
        try:
            import inspect
            bp = ax.Axes.boxplot
            if not getattr(bp, "_patched", False) and "tick_labels" not in inspect.signature(bp).parameters:
                def boxplot(self, *a, tick_labels=None, **k):
                    if tick_labels is not None:
                        k["labels"] = tick_labels
                    return bp(self, *a, **k)
                boxplot._patched = True
                ax.Axes.boxplot = boxplot
        except Exception:
            pass

def _install_hooks():
    """Registra cada check() para contar ejercicios y silencia el estilo de Jupyter (el editor ya tiene el suyo)."""
    try:
        import geotools.revisar as rv
    except Exception:
        return
    if getattr(rv, "_hooked", False):
        return
    orig = rv.check
    def check(exercise_id, answer, hint=None):
        ok = None
        try:
            p = rv._answer_path(exercise_id)
            if p.exists():
                ok = rv._hash(answer) == p.read_text(encoding="utf-8").strip()
        except Exception:
            ok = None
        _CHECKS.append({"id": exercise_id, "ok": ok})
        return orig(exercise_id, answer, hint)
    rv.check = check
    rv._in_jupyter = lambda: True   # los mensajes de check() salen en HTML, como en Jupyter
    rv._hooked = True
    try:
        import geotools.estilo as es
        es.aplicar_estilo = lambda: None
    except Exception:
        pass

def _figures():
    out = []
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        for n in plt.get_fignums():
            buf = io.BytesIO()
            plt.figure(n).savefig(buf, format="png", dpi=110, bbox_inches="tight")
            out.append(base64.b64encode(buf.getvalue()).decode())
        plt.close("all")
    return out

def __run_cell(src):
    global _DISPLAY, _CHECKS
    _DISPLAY, _CHECKS = [], []
    _compat(src)
    err = None
    try:
        tree = ast.parse(src, "<celda>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr) and not src.rstrip().endswith(";"):
            last = tree.body.pop()
        _install_hooks()
        exec(compile(tree, "<celda>", "exec"), _ns)
        if last is not None:
            val = eval(compile(ast.Expression(last.value), "<celda>", "eval"), _ns)
            if val is not None:
                _ns["_"] = val
                _emit(val)
    except BaseException as e:
        tb = e.__traceback__.tb_next if (e.__traceback__ is not None and not isinstance(e, SyntaxError)) else None
        err = {"ename": type(e).__name__, "evalue": str(e), "traceback": "".join(traceback.format_exception(type(e), e, tb))}
    try:
        figs = _figures()
        if err is not None:
            figs = []   # como en Jupyter, una celda con error no muestra gráficos a medias
    except Exception:
        figs = []
    return {"error": err, "display": list(_DISPLAY), "figures": figs, "checks": list(_CHECKS)}
`;
