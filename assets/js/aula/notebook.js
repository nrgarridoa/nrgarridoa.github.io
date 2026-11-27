// Cuaderno interactivo del aula: muestra un .ipynb con celdas ejecutables.
// Python corre en el navegador (Pyodide, en un Worker); los ejercicios se corrigen con el mismo check() del curso.
import { createEditor, renderMarkdown, sanitizeOutputHtml } from '../../vendor/nbvendor.mjs';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ICON = {
  play: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 001.5.86l10.5-6.5a1 1 0 000-1.72L9.5 4.64A1 1 0 008 5.5z"/></svg>',
  all: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 5.5v13l8-6.5zM14 5.5v13l8-6.5z"/></svg>',
  restart: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.6L3 8M3 3v5h5"/></svg>',
  stop: '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  check: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 13l4 4L19 7"/></svg>',
  undo: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7v6h6M3 13a9 9 0 109-9 9 9 0 00-6 2.3"/></svg>',
};
const exId = (src) => (src.match(/check\(\s*["']([^"']+)["']/) || [])[1] || null;

export function mountNotebook(host, opts) {
  const nb = opts.nb;
  const cells = nb.cells.map((c, i) => {
    const src = Array.isArray(c.source) ? c.source.join('') : String(c.source || '');
    return { i, type: c.cell_type, src, orig: src, ex: c.cell_type === 'code' ? exId(src) : null, count: null, state: 'idle', out: [], editor: null, el: null };
  });
  const codeCells = cells.filter((c) => c.type === 'code');
  const exercises = [...new Set(codeCells.map((c) => c.ex).filter(Boolean))];
  const usesObspy = codeCells.some((c) => /^\s*(?:import|from)\s+obspy/m.test(c.src));
  const solved = new Set(opts.saved?.ok || []);
  const wrong = new Set();

  // Lo que el alumno escribio queda guardado en su navegador
  let mods = {};
  try { mods = JSON.parse(localStorage.getItem(opts.storeKey) || '{}'); } catch { mods = {}; }
  for (const c of codeCells) if (typeof mods[c.i] === 'string') c.src = mods[c.i];
  let saveT = null;
  const persist = () => { clearTimeout(saveT); saveT = setTimeout(() => {
    const o = {}; for (const c of codeCells) if (c.src !== c.orig) o[c.i] = c.src;
    try { if (Object.keys(o).length) localStorage.setItem(opts.storeKey, JSON.stringify(o)); else localStorage.removeItem(opts.storeKey); } catch { /* sin espacio */ }
  }, 500); };

  // ---------- estructura ----------
  host.innerHTML = `<div class="nb">
    <div class="nb-bar">
      <button type="button" class="au-btn nb-btn" data-nb="runall" title="Ejecutar todas las celdas">${ICON.all}<span>Ejecutar todo</span></button>
      <button type="button" class="au-ghost" data-nb="restart" title="Reinicia Python y borra las variables">${ICON.restart}<span>Reiniciar</span></button>
      <button type="button" class="au-ghost nb-stop" data-nb="stop" hidden title="Detiene la ejecución y reinicia Python">${ICON.stop}<span>Detener</span></button>
      <span class="nb-status" id="nb-status" data-s="idle"><i></i><b>Python sin iniciar</b></span>
      <span class="flex-1"></span>
      ${exercises.length ? `<div class="nb-prog" title="Ejercicios resueltos"><span id="nb-pt"></span><div class="au-bar"><i id="nb-pb" style="width:0%"></i></div></div>` : ''}
      ${opts.onStatic ? `<button type="button" class="nb-link" data-nb="static">Ver versión de lectura</button>` : ''}
    </div>
    ${usesObspy ? `<div class="nb-note">Esta cápsula usa <b>ObsPy</b>, que todavía no funciona en el navegador. Puedes leerla aquí y ejecutarla en tu computadora con el notebook y el paquete del curso (menú Descargas).</div>` : ''}
    <div class="nb-cells" id="nb-cells"></div>
  </div>`;
  const $ = (s) => host.querySelector(s);
  const cellsEl = $('#nb-cells');

  for (const c of cells) {
    const el = document.createElement('div');
    c.el = el;
    if (c.type === 'markdown') {
      el.className = 'nb-cell nb-md';
      el.innerHTML = renderMarkdown(c.src);
    } else if (c.type === 'code') {
      el.className = 'nb-cell nb-code' + (c.ex ? ' is-ex' : '');
      el.innerHTML = `<div class="nb-gutter"><button type="button" class="nb-run" data-nb="run" data-i="${c.i}" title="Ejecutar (Shift+Enter)" aria-label="Ejecutar celda">${ICON.play}</button><span class="nb-count">[ ]</span></div>
        <div class="nb-main">
          ${c.ex ? `<div class="nb-exhead"><span class="nb-exname">Ejercicio ${esc(c.ex)}</span><span class="nb-exchip" data-exchip></span></div>` : ''}
          <div class="nb-editor"></div>
          <div class="nb-tools"><span class="nb-cstatus"></span><button type="button" class="nb-link" data-nb="reset" data-i="${c.i}" hidden>${ICON.undo} Restaurar celda</button></div>
          <div class="nb-out"></div></div>`;
      c.editor = createEditor(el.querySelector('.nb-editor'), c.src, {
        onRun: (mode) => { runCell(c.i, true).then(() => { if (mode === 'next') focusNext(c.i); }); },
        onChange: (v) => { c.src = v; syncCell(c); persist(); },
      });
      syncCell(c);
      paintEx(c);
    } else continue;
    cellsEl.appendChild(el);
  }
  updateProgress();

  // ---------- presentacion de cada celda ----------
  function syncCell(c) {
    const reset = c.el.querySelector('[data-nb=reset]');
    if (reset) reset.hidden = c.src === c.orig;
  }
  function paintEx(c) {
    const chip = c.el.querySelector('[data-exchip]'); if (!chip) return;
    const ok = solved.has(c.ex), bad = wrong.has(c.ex);
    chip.dataset.s = ok ? 'ok' : bad ? 'bad' : 'todo';
    chip.innerHTML = ok ? `${ICON.check} Correcto` : bad ? 'Revisa tu respuesta' : 'Por resolver';
  }
  function updateProgress() {
    if (!exercises.length) return;
    const n = exercises.filter((e) => solved.has(e)).length;
    $('#nb-pt').textContent = `Ejercicios ${n} de ${exercises.length}`;
    $('#nb-pb').style.width = `${Math.round((n / exercises.length) * 100)}%`;
  }
  const outEl = (c) => c.el.querySelector('.nb-out');
  function paintOut(c) {
    const box = outEl(c); box.innerHTML = '';
    for (const o of c.out) {
      const d = document.createElement('div');
      if (o.k === 'stream') { d.className = 'nb-pre ' + (o.name === 'stderr' ? 'is-err' : ''); d.textContent = o.text; }
      else if (o.k === 'html') { d.className = 'nb-html'; d.innerHTML = sanitizeOutputHtml(o.html); }
      else if (o.k === 'md') { d.className = 'nb-html'; d.innerHTML = renderMarkdown(o.text); }
      else if (o.k === 'png') { d.className = 'nb-img'; d.innerHTML = `<img alt="Gráfico generado" src="data:image/png;base64,${o.b64}">`; }
      else if (o.k === 'text') { d.className = 'nb-pre is-res'; d.textContent = o.text; }
      else if (o.k === 'error') { d.className = 'nb-err'; d.innerHTML = errorHtml(o); }
      box.appendChild(d);
    }
  }
  function errorHtml(o) {
    if (o.ename === 'ModuleNotFoundError' && /obspy/.test(o.evalue)) return '<b>ObsPy no está disponible en el navegador.</b> Ejecuta esta cápsula en tu computadora con el notebook y el paquete del curso (menú Descargas).';
    if (/emscripten does not support processes|subprocess/i.test(o.evalue)) return '<b>Esta celda necesita ejecutar un proceso del sistema</b>, algo que el navegador no permite. Pruébala en tu computadora.';
    return `<pre>${esc(o.tb || `${o.ename}: ${o.evalue}`)}</pre>`;
  }
  function setState(c, state, text = '') {
    c.state = state; c.el.dataset.state = state;
    const s = c.el.querySelector('.nb-cstatus'); if (s) s.textContent = text;
    const cnt = c.el.querySelector('.nb-count'); if (cnt) cnt.textContent = state === 'running' ? '[*]' : state === 'queued' ? '[…]' : c.count ? `[${c.count}]` : '[ ]';
  }
  function setStatus(s, text) {
    const el = $('#nb-status'); el.dataset.s = s; el.querySelector('b').textContent = text;
    $('.nb-stop').hidden = s !== 'busy' && s !== 'loading';
  }
  function focusNext(i) {
    const nxt = codeCells.find((c) => c.i > i); if (!nxt) return;
    nxt.el.scrollIntoView({ block: 'center', behavior: 'smooth' }); nxt.editor.focus();
  }

  // ---------- kernel (Worker con Pyodide) ----------
  let worker = null, starting = null, execCount = 0, active = null, initWait = null, zipBuf = null, chain = Promise.resolve(), disposed = false;

  function startKernel() {
    if (starting) return starting;
    setStatus('loading', 'Iniciando Python…');
    starting = (async () => {
      if (!zipBuf) zipBuf = await opts.loadZip();
      worker = new Worker(new URL('./pyworker.js', import.meta.url));
      worker.onmessage = onMsg;
      worker.onerror = (e) => { if (initWait) initWait.rej(new Error(e.message || 'No se pudo iniciar Python')); };
      await new Promise((res, rej) => { initWait = { res, rej }; worker.postMessage({ type: 'init', zip: zipBuf.slice(0), cwd: opts.cwd }); });
      initWait = null;
      setStatus('idle', 'Python listo');
    })().catch((e) => { starting = null; worker?.terminate(); worker = null; setStatus('error', 'No se pudo iniciar Python'); throw e; });
    return starting;
  }
  function onMsg(ev) {
    const m = ev.data;
    if (m.type === 'status') { setStatus(initWait ? 'loading' : 'busy', m.msg); if (active) setState(cells[active.i], 'running', m.msg); }
    else if (m.type === 'ready') initWait?.res();
    else if (m.type === 'fatal') { if (initWait) initWait.rej(new Error(m.error)); else if (active) finish(m.error); }
    else if (m.type === 'out' && active) {
      const c = cells[active.i], last = c.out[c.out.length - 1];
      if (last && last.k === 'stream' && last.name === m.stream) last.text += m.text; else c.out.push({ k: 'stream', name: m.stream, text: m.text });
      paintOut(c);
    } else if (m.type === 'result' && active) finish(null, m);
  }
  function finish(fatal, m) {
    const a = active, c = cells[a.i];
    if (fatal) c.out.push({ k: 'error', ename: 'Error', evalue: fatal, tb: '' });
    else {
      for (const d of m.display || []) {
        if (d.mime === 'text/html') c.out.push({ k: 'html', html: d.data });
        else if (d.mime === 'text/markdown') c.out.push({ k: 'md', text: d.data });
        else if (d.mime === 'image/png') c.out.push({ k: 'png', b64: d.data });
        else c.out.push({ k: 'text', text: d.data });
      }
      for (const b64 of m.figures || []) c.out.push({ k: 'png', b64 });
      if (m.error) c.out.push({ k: 'error', ename: m.error.ename, evalue: m.error.evalue, tb: m.error.traceback });
      let changed = false;
      for (const k of m.checks || []) {
        if (k.ok === true) { if (!solved.has(k.id)) { solved.add(k.id); changed = true; } wrong.delete(k.id); }
        else if (k.ok === false && !solved.has(k.id)) wrong.add(k.id);
      }
      for (const x of codeCells) if (x.ex) paintEx(x);
      updateProgress();
      if (changed && opts.onExercises) opts.onExercises({ ok: exercises.filter((e) => solved.has(e)), total: exercises.length });
    }
    c.count = ++execCount;
    paintOut(c); setState(c, 'idle');
    setStatus('idle', 'Python listo');
    const hadError = !!(fatal || m?.error);
    active = null; a.res(hadError);
  }

  function runCell(i, focusOnly) {
    const c = cells[i];
    if (c.type !== 'code') return Promise.resolve(false);
    setState(c, 'queued');
    const job = chain.then(async () => {
      if (disposed) return false;
      c.out = []; paintOut(c); setState(c, 'running', 'Preparando…');
      try { await startKernel(); } catch (e) {
        c.out.push({ k: 'error', ename: 'Error', evalue: String(e.message || e), tb: '' }); paintOut(c); setState(c, 'idle'); return true;
      }
      setStatus('busy', 'Ejecutando…');
      return new Promise((res) => { active = { i, res }; setState(c, 'running', ''); worker.postMessage({ type: 'run', cell: i, code: c.editor.getValue() }); });
    });
    chain = job.catch(() => false);
    return job;
  }
  async function runAll() {
    for (const c of codeCells) { if (disposed) break; await runCell(c.i); }
  }
  function killKernel(msg) {
    worker?.terminate(); worker = null; starting = null; initWait = null;
    if (active) { const c = cells[active.i]; c.out.push({ k: 'stream', name: 'stderr', text: msg + '\n' }); paintOut(c); setState(c, 'idle'); const r = active.res; active = null; r(true); }
    chain = Promise.resolve();
    for (const c of codeCells) if (c.state !== 'idle') setState(c, 'idle');
    setStatus('idle', 'Python sin iniciar');
  }

  // ---------- eventos ----------
  host.addEventListener('click', (e) => {
    const b = e.target.closest('[data-nb]'); if (!b) return;
    const act = b.dataset.nb;
    if (act === 'run') runCell(Number(b.dataset.i));
    else if (act === 'runall') runAll();
    else if (act === 'restart') { killKernel('Python reiniciado: las variables se borraron.'); for (const c of codeCells) { c.count = null; setState(c, 'idle'); } }
    else if (act === 'stop') killKernel('Ejecución detenida. Python se reiniciará la próxima vez que ejecutes una celda.');
    else if (act === 'reset') { const c = cells[Number(b.dataset.i)]; c.editor.setValue(c.orig); }
    else if (act === 'static') opts.onStatic();
  });

  return {
    dispose() { disposed = true; worker?.terminate(); worker = null; },
    exercises: () => ({ ok: exercises.filter((e) => solved.has(e)), total: exercises.length }),
  };
}
