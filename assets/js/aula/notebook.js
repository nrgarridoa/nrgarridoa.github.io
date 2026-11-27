// Cuaderno interactivo del aula: muestra un .ipynb con celdas ejecutables y una consola libre.
// Python corre en el navegador (Pyodide, en un Worker). Un mismo kernel se comparte entre el cuaderno de la clase
// y la practica libre; los ejercicios se corrigen con el mismo check() del curso.
import { createEditor, renderMarkdown, sanitizeOutputHtml } from '../../vendor/nbvendor.mjs';

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ICON = {
  play: '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 5.5v13a1 1 0 001.5.86l10.5-6.5a1 1 0 000-1.72L9.5 4.64A1 1 0 008 5.5z"/></svg>',
  all: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 5.5v13l8-6.5zM14 5.5v13l8-6.5z"/></svg>',
  restart: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.6L3 8M3 3v5h5"/></svg>',
  stop: '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  check: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 13l4 4L19 7"/></svg>',
  undo: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 7v6h6M3 13a9 9 0 109-9 9 9 0 00-6 2.3"/></svg>',
  plus: '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  swap: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 4L3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/></svg>',
  up: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 15l6-6 6 6"/></svg>',
  down: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
  trash: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3"/></svg>',
};
const exId = (src) => (src.match(/check\(\s*["']([^"']+)["']/) || [])[1] || null;

/* ------------------------------------------------------------------ kernel compartido */
export function createKernel({ loadZip, cwd }) {
  let worker = null, starting = null, active = null, initWait = null, zipBuf = null, chain = Promise.resolve(), dead = false;
  let state = { s: 'idle', text: 'Python sin iniciar' };
  const status = new Set(), restarts = new Set();
  const emit = (s, text) => { state = { s, text }; status.forEach((f) => f(s, text)); };

  function start() {
    if (starting) return starting;
    emit('loading', 'Iniciando Python…');
    starting = (async () => {
      if (!zipBuf) zipBuf = await loadZip();
      worker = new Worker(new URL('./pyworker.js', import.meta.url));
      worker.onmessage = onMsg;
      worker.onerror = (e) => { if (initWait) initWait.rej(new Error(e.message || 'No se pudo iniciar Python')); };
      await new Promise((res, rej) => { initWait = { res, rej }; worker.postMessage({ type: 'init', zip: zipBuf.slice(0), cwd }); });
      initWait = null;
      emit('idle', 'Python listo');
    })().catch((e) => { starting = null; worker?.terminate(); worker = null; emit('error', 'No se pudo iniciar Python'); throw e; });
    return starting;
  }
  function onMsg(ev) {
    const m = ev.data;
    if (m.type === 'status') { emit(initWait ? 'loading' : 'busy', m.msg); active?.h.onStatus?.(m.msg); }
    else if (m.type === 'ready') initWait?.res();
    else if (m.type === 'fatal') { if (initWait) initWait.rej(new Error(m.error)); else if (active) { const a = active; active = null; emit('idle', 'Python listo'); a.res({ fatal: m.error }); } }
    else if (m.type === 'out') active?.h.onOut?.(m.stream, m.text);
    else if (m.type === 'result' && active) { const a = active; active = null; emit('idle', 'Python listo'); a.res(m); }
  }
  function run(code, h = {}) {
    const job = chain.then(async () => {
      if (dead) return { fatal: 'El cuaderno se cerró.' };
      h.onStart?.();
      await start();
      emit('busy', 'Ejecutando…');
      return new Promise((res) => { active = { h, res }; worker.postMessage({ type: 'run', cell: 0, code }); });
    }).catch((e) => ({ fatal: String((e && e.message) || e) }));
    chain = job.then(() => {}, () => {});
    return job;
  }
  function kill(msg) {
    worker?.terminate(); worker = null; starting = null; initWait = null;
    if (active) { const a = active; active = null; a.res({ killed: true, message: msg }); }
    chain = Promise.resolve();
    emit('idle', 'Python sin iniciar');
    restarts.forEach((f) => f());
  }
  return {
    run, kill, start,
    onStatus(cb) { status.add(cb); cb(state.s, state.text); return () => status.delete(cb); },
    onRestart(cb) { restarts.add(cb); return () => restarts.delete(cb); },
    dispose() { dead = true; worker?.terminate(); worker = null; status.clear(); restarts.clear(); },
  };
}

/* ------------------------------------------------------------------ cuaderno */
export function mountNotebook(host, opts) {
  const kernel = opts.kernel, scratch = !!opts.scratch;
  let uidc = 0;
  const mk = (type, src, user = false) => ({ i: uidc++, key: null, type, src, orig: src, user, ex: type === 'code' && !user ? exId(src) : null, count: null, state: 'idle', out: [], editor: null, el: null });

  // ---------- lo que el alumno escribio (guardado en su navegador) ----------
  let store = null;
  try { store = JSON.parse(localStorage.getItem(opts.storeKey) || 'null'); } catch { store = null; }
  const newKey = () => 'u:' + Math.random().toString(36).slice(2, 8);
  let cells = [];
  let pendingAdded = [];
  if (scratch) {
    const items = Array.isArray(store) && store.length ? store : [{ t: 'code', s: '# Escribe aquí tu código y ejecútalo con Shift+Enter\n' }];
    cells = items.map((it) => { const o = typeof it === 'string' ? { t: 'code', s: it } : it; const c = mk(o.t === 'md' ? 'markdown' : 'code', String(o.s ?? ''), true); c.key = o.id ? 'u:' + o.id : newKey(); return c; });
  } else {
    const edits = store && store.v === 2 ? store.edits || {} : (store && !store.v && typeof store === 'object' ? store : {});
    pendingAdded = store && store.v === 2 ? store.added || [] : [];
    cells = opts.nb.cells.map((raw, k) => {
      const c = mk(raw.cell_type, Array.isArray(raw.source) ? raw.source.join('') : String(raw.source || ''));
      c.idx = k; c.key = 'o:' + k; if (c.type === 'code' && typeof edits[k] === 'string') c.src = edits[k];
      return c;
    });
  }
  const codeCells = () => cells.filter((c) => c.type === 'code');
  const exercises = scratch ? [] : [...new Set(codeCells().map((c) => c.ex).filter(Boolean))];
  const usesObspy = !scratch && codeCells().some((c) => /^\s*(?:import|from)\s+obspy/m.test(c.orig));
  const solved = new Set(opts.saved?.ok || []);
  const wrong = new Set();

  let saveT = null;
  const persist = () => { clearTimeout(saveT); saveT = setTimeout(() => {
    try {
      if (scratch) { localStorage.setItem(opts.storeKey, JSON.stringify(cells.map((c) => ({ id: c.key.slice(2), t: c.type === 'code' ? 'code' : 'md', s: c.src })))); return; }
      const edits = {}; for (const c of cells) if (!c.user && c.type === 'code' && c.src !== c.orig) edits[c.idx] = c.src;
      const added = []; cells.forEach((c, k) => { if (c.user) added.push({ id: c.key.slice(2), after: k ? cells[k - 1].key : null, t: c.type === 'code' ? 'code' : 'md', s: c.src }); });
      if (Object.keys(edits).length || added.length) localStorage.setItem(opts.storeKey, JSON.stringify({ v: 2, edits, added })); else localStorage.removeItem(opts.storeKey);
    } catch { /* sin espacio */ }
  }, 400); };

  host.innerHTML = `<div class="nb">
    <div class="nb-bar">
      ${scratch ? `<button type="button" class="au-btn nb-btn" data-nb="addcode" data-i="-2" title="Agregar una celda de código al final">${ICON.plus}<span>Código</span></button>
      <button type="button" class="au-ghost" data-nb="addtext" data-i="-2" title="Agregar una celda de texto al final">${ICON.plus}<span>Texto</span></button>` : ''}
      <button type="button" class="${scratch ? 'au-ghost' : 'au-btn nb-btn'}" data-nb="runall" title="Ejecutar todas las celdas">${ICON.all}<span>Ejecutar todo</span></button>
      <button type="button" class="au-ghost" data-nb="restart" title="Reinicia Python y borra las variables">${ICON.restart}<span>Reiniciar</span></button>
      <button type="button" class="au-ghost nb-stop" data-nb="stop" hidden title="Detiene la ejecución y reinicia Python">${ICON.stop}<span>Detener</span></button>
      <span class="nb-status" data-role="status" data-s="idle"><i></i><b>Python sin iniciar</b></span>
      <span class="flex-1"></span>
      ${exercises.length ? `<div class="nb-prog" title="Ejercicios resueltos"><span data-role="pt"></span><div class="au-bar"><i data-role="pb" style="width:0%"></i></div></div>` : ''}
      ${!scratch && opts.onStatic ? `<button type="button" class="nb-link" data-nb="static">Ver versión de lectura</button>` : ''}
    </div>
    ${scratch ? `<div class="nb-note nb-note-info">Práctica libre: escribe y ejecuta código sin tocar la clase. Pasa el mouse entre celdas para agregar código o texto; usa las flechas y la papelera de cada celda para ordenarlas. Comparte variables con el cuaderno de la clase y se guarda en tu navegador.</div>` : ''}
    ${usesObspy ? `<div class="nb-note">Esta cápsula usa <b>ObsPy</b>, que todavía no funciona en el navegador. Puedes leerla aquí y ejecutarla en tu computadora con el notebook y el paquete del curso (menú Descargas).</div>` : ''}
    <div class="nb-cells" data-role="cells">${scratch ? `<div class="nb-add nb-add-top"><button type="button" data-nb="addcode" data-i="-1">${ICON.plus} Código</button><button type="button" data-nb="addtext" data-i="-1">${ICON.plus} Texto</button></div>` : ''}</div>
  </div>`;
  const $ = (s) => host.querySelector(s);
  const cellsEl = $('[data-role=cells]');
  const byId = (i) => cells.find((c) => c.i === i);
  const addBar = (c) => `<div class="nb-add"><button type="button" data-nb="addcode" data-i="${c.i}">${ICON.plus} Código</button><button type="button" data-nb="addtext" data-i="${c.i}">${ICON.plus} Texto</button></div>`;
  const ctools = (c) => (scratch || c.user) ? `<div class="nb-ctools" role="toolbar" aria-label="Acciones de la celda">
      <button type="button" data-nb="convert" data-i="${c.i}" title="${c.type === 'code' ? 'Convertir en celda de texto' : 'Convertir en celda de código'}" aria-label="Cambiar el tipo de celda">${ICON.swap}</button>
      <button type="button" data-nb="up" data-i="${c.i}" title="Subir celda" aria-label="Subir celda">${ICON.up}</button>
      <button type="button" data-nb="down" data-i="${c.i}" title="Bajar celda" aria-label="Bajar celda">${ICON.down}</button>
      <button type="button" data-nb="del" data-i="${c.i}" title="Eliminar celda" aria-label="Eliminar celda">${ICON.trash}</button></div>` : '';

  function mdRender(c) {
    const v = c.el.querySelector('.nb-mdview');
    v.innerHTML = c.src.trim() ? renderMarkdown(c.src) : '<p class="nb-mdhint">Celda de texto vacía. Haz doble clic para escribir.</p>';
  }
  function mdEdit(c, on) {
    const ta = c.el.querySelector('.nb-mdedit'), v = c.el.querySelector('.nb-mdview');
    if (on) { ta.value = c.src; ta.hidden = false; v.hidden = true; ta.style.height = 'auto'; ta.style.height = Math.max(90, ta.scrollHeight + 4) + 'px'; ta.focus(); }
    else { c.src = ta.value; ta.hidden = true; v.hidden = false; mdRender(c); persist(); }
  }

  function buildCell(c) {
    const el = document.createElement('div');
    c.el = el; el.dataset.i = c.i;
    if (c.type === 'markdown') {
      el.className = 'nb-cell nb-md' + (c.user ? ' is-user' : '');
      if (c.user) {
        el.innerHTML = `${ctools(c)}<div class="nb-mdview"></div><textarea class="nb-mdedit" hidden aria-label="Celda de texto (Markdown)" placeholder="Escribe aquí tu texto (admite Markdown). Shift+Enter para terminar."></textarea>${addBar(c)}`;
        const ta = el.querySelector('.nb-mdedit');
        ta.addEventListener('blur', () => { if (!ta.hidden) mdEdit(c, false); });
        ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.shiftKey || e.ctrlKey)) { e.preventDefault(); mdEdit(c, false); } });
        ta.addEventListener('input', () => { c.src = ta.value; ta.style.height = 'auto'; ta.style.height = Math.max(90, ta.scrollHeight + 4) + 'px'; persist(); });
        el.querySelector('.nb-mdview').addEventListener('dblclick', () => mdEdit(c, true));
        mdRender(c);
      } else { el.innerHTML = `${renderMarkdown(c.src)}${addBar(c)}`; }
    } else {
      el.className = 'nb-cell nb-code' + (c.ex ? ' is-ex' : '') + (c.user ? ' is-user' : '');
      el.innerHTML = `<div class="nb-gutter"><button type="button" class="nb-run" data-nb="run" data-i="${c.i}" title="Ejecutar (Shift+Enter)" aria-label="Ejecutar celda">${ICON.play}</button><span class="nb-count">[ ]</span></div>
        <div class="nb-main">
          ${c.ex ? `<div class="nb-exhead"><span class="nb-exname">Ejercicio ${esc(c.ex)}</span><span class="nb-exchip" data-exchip></span></div>` : ''}
          ${ctools(c)}
          <div class="nb-editor"></div>
          <div class="nb-tools"><span class="nb-cstatus"></span>${c.user ? '' : `<button type="button" class="nb-link" data-nb="reset" data-i="${c.i}" hidden>${ICON.undo} Restaurar celda</button>`}</div>
          <div class="nb-out"></div></div>
        ${addBar(c)}`;
      c.editor = createEditor(el.querySelector('.nb-editor'), c.src, {
        onRun: (mode) => { runCell(c.i).then(() => { if (mode === 'next') focusNext(c.i); else if (mode === 'insert') insertCell('code', c, true); }); },
        onChange: (v) => { c.src = v; syncCell(c); persist(); },
      });
      syncCell(c); paintEx(c);
    }
    return el;
  }
  function insertCell(type, after, focus = true) {
    const c = mk(type === 'code' ? 'code' : 'markdown', '', true);
    c.key = newKey();
    const at = after ? cells.indexOf(after) + 1 : 0;
    cells.splice(at, 0, c);
    const el = buildCell(c);
    if (after) after.el.after(el); else cellsEl.querySelector('.nb-add-top')?.after(el) || cellsEl.prepend(el);
    persist();
    if (focus) { el.scrollIntoView({ block: 'center', behavior: 'smooth' }); if (c.type === 'code') c.editor.focus(); else mdEdit(c, true); }
    return c;
  }
  // Cambia una celda propia de código a texto o al revés, conservando su contenido y su lugar.
  function convertCell(c) {
    const n = mk(c.type === 'code' ? 'markdown' : 'code', c.src, true);
    n.key = c.key;
    cells[cells.indexOf(c)] = n;
    const el = buildCell(n);
    c.el.replaceWith(el);
    persist();
    if (n.type === 'code') n.editor.focus(); else if (n.src.trim()) mdRender(n); else mdEdit(n, true);
  }
  function moveCell(c, dir) {
    const k = cells.indexOf(c), j = k + dir;
    if (j < 0 || j >= cells.length) return;
    [cells[k], cells[j]] = [cells[j], cells[k]];
    if (dir < 0) cells[j + 1].el.before(c.el); else cells[j - 1].el.after(c.el);
    if (c.type === 'code') c.editor.focus();
    persist();
  }
  function removeCell(c) {
    if (scratch && cells.length === 1) { if (c.type === 'code') { c.editor.setValue(''); c.out = []; paintOut(c); } else { c.src = ''; mdRender(c); persist(); } return; }
    cells.splice(cells.indexOf(c), 1); c.el.remove(); persist();
  }

  cells.forEach((c) => cellsEl.appendChild(buildCell(c)));
  // celdas agregadas por el alumno en la clase: se vuelven a colocar despues de la celda que las precedia
  for (const a of pendingAdded) {
    const anchor = a.after ? cells.find((c) => c.key === a.after) : null;
    const c = mk(a.t === 'md' ? 'markdown' : 'code', String(a.s ?? ''), true); c.key = 'u:' + a.id;
    cells.splice(anchor ? cells.indexOf(anchor) + 1 : 0, 0, c);
    const el = buildCell(c); if (anchor) anchor.el.after(el); else cellsEl.prepend(el);
  }
  updateProgress();

  function syncCell(c) { const r = c.el.querySelector('[data-nb=reset]'); if (r) r.hidden = c.src === c.orig; }
  function paintEx(c) {
    const chip = c.el.querySelector('[data-exchip]'); if (!chip) return;
    const ok = solved.has(c.ex), bad = wrong.has(c.ex);
    chip.dataset.s = ok ? 'ok' : bad ? 'bad' : 'todo';
    chip.innerHTML = ok ? `${ICON.check} Correcto` : bad ? 'Revisa tu respuesta' : 'Por resolver';
  }
  function updateProgress() {
    if (!exercises.length) return;
    const n = exercises.filter((e) => solved.has(e)).length;
    $('[data-role=pt]').textContent = `Ejercicios ${n} de ${exercises.length}`;
    $('[data-role=pb]').style.width = `${Math.round((n / exercises.length) * 100)}%`;
  }
  function errorHtml(o) {
    if (o.ename === 'ModuleNotFoundError' && /obspy/.test(o.evalue)) return '<b>ObsPy no está disponible en el navegador.</b> Ejecuta esta cápsula en tu computadora con el notebook y el paquete del curso (menú Descargas).';
    if (/emscripten does not support processes|subprocess/i.test(o.evalue)) return '<b>Esta celda necesita ejecutar un proceso del sistema</b>, algo que el navegador no permite. Pruébala en tu computadora.';
    if (/pyogrio/.test(o.evalue)) return '<b>Esta función necesita <code>pyogrio</code>, que no está disponible en el navegador.</b> Pruébala en tu computadora.';
    return `<pre>${esc(o.tb || `${o.ename}: ${o.evalue}`)}</pre>`;
  }
  function paintOut(c) {
    const box = c.el.querySelector('.nb-out'); box.innerHTML = '';
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
  function setState(c, state, text = '') {
    c.state = state; c.el.dataset.state = state;
    const s = c.el.querySelector('.nb-cstatus'); if (s) s.textContent = text;
    const cnt = c.el.querySelector('.nb-count'); if (cnt) cnt.textContent = state === 'running' ? '[*]' : state === 'queued' ? '[…]' : c.count ? `[${c.count}]` : '[ ]';
  }
  function focusNext(i) {
    const list = codeCells(), k = list.findIndex((c) => c.i === i);
    let nxt = list[k + 1];
    if (!nxt && scratch) nxt = insertCell('code', byId(i), false);
    if (!nxt) return;
    nxt.el.scrollIntoView({ block: 'center', behavior: 'smooth' }); nxt.editor.focus();
  }

  // ---------- ejecucion ----------
  let execCount = 0;
  function finish(c, res) {
    if (res.killed) { c.out.push({ k: 'stream', name: 'stderr', text: res.message + '\n' }); paintOut(c); setState(c, 'idle'); return true; }
    if (res.fatal) { c.out.push({ k: 'error', ename: 'Error', evalue: String(res.fatal), tb: '' }); paintOut(c); setState(c, 'idle'); return true; }
    for (const d of res.display || []) {
      if (d.mime === 'text/html') c.out.push({ k: 'html', html: d.data });
      else if (d.mime === 'text/markdown') c.out.push({ k: 'md', text: d.data });
      else if (d.mime === 'image/png') c.out.push({ k: 'png', b64: d.data });
      else c.out.push({ k: 'text', text: d.data });
    }
    for (const b64 of res.figures || []) c.out.push({ k: 'png', b64 });
    if (res.error) c.out.push({ k: 'error', ename: res.error.ename, evalue: res.error.evalue, tb: res.error.traceback });
    let changed = false;
    for (const k of res.checks || []) {
      if (k.ok === true) { if (!solved.has(k.id)) { solved.add(k.id); changed = true; } wrong.delete(k.id); }
      else if (k.ok === false && !solved.has(k.id)) wrong.add(k.id);
    }
    if (exercises.length) { for (const x of codeCells()) if (x.ex) paintEx(x); updateProgress(); }
    if (changed && opts.onExercises) opts.onExercises({ ok: exercises.filter((e) => solved.has(e)), total: exercises.length });
    c.count = ++execCount; paintOut(c); setState(c, 'idle');
    return !!res.error;
  }
  function runCell(i) {
    const c = byId(i); if (!c || c.type !== 'code') return Promise.resolve(false);
    setState(c, 'queued');
    return kernel.run(c.editor.getValue(), {
      onStart: () => { c.out = []; paintOut(c); setState(c, 'running', 'Preparando…'); },
      onStatus: (t) => setState(c, 'running', t),
      onOut: (name, text) => { const l = c.out[c.out.length - 1]; if (l && l.k === 'stream' && l.name === name) l.text += text; else c.out.push({ k: 'stream', name, text }); paintOut(c); },
    }).then((res) => finish(c, res));
  }
  async function runAll() { for (const c of codeCells()) await runCell(c.i); }

  const offStatus = kernel.onStatus((s, text) => {
    const el = $('[data-role=status]'); el.dataset.s = s; el.querySelector('b').textContent = text;
    $('.nb-stop').hidden = s !== 'busy' && s !== 'loading';
  });
  const offRestart = kernel.onRestart(() => { for (const c of codeCells()) { c.count = null; setState(c, 'idle'); } });

  host.addEventListener('click', (e) => {
    const b = e.target.closest('[data-nb]'); if (!b || !host.contains(b)) return;
    const act = b.dataset.nb, id = Number(b.dataset.i);
    if (act === 'run') runCell(id);
    else if (act === 'runall') runAll();
    else if (act === 'addcode' || act === 'addtext') insertCell(act === 'addcode' ? 'code' : 'md', id === -1 ? null : id === -2 ? cells[cells.length - 1] || null : byId(id));
    else if (act === 'convert') convertCell(byId(id));
    else if (act === 'up') moveCell(byId(id), -1);
    else if (act === 'down') moveCell(byId(id), 1);
    else if (act === 'del') removeCell(byId(id));
    else if (act === 'restart') kernel.kill('Python reiniciado: las variables se borraron.');
    else if (act === 'stop') kernel.kill('Ejecución detenida. Python se reiniciará la próxima vez que ejecutes una celda.');
    else if (act === 'reset') { const c = byId(id); c.editor.setValue(c.orig); }
    else if (act === 'static') opts.onStatic();
  });

  return {
    dispose() { offStatus(); offRestart(); },
    exercises: () => ({ ok: exercises.filter((e) => solved.has(e)), total: exercises.length }),
  };
}
