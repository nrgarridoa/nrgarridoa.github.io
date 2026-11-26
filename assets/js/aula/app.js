// Aula virtual — Nilson Garrido
// SPA sin build: Firebase (Auth con Google + Firestore) cargado como módulo ES.
// Estructura pública de los cursos: /aula/catalog.json (generada por Jekyll).
// Contenido privado (videos, materiales, evaluaciones), alumnos, progreso y
// certificados: Firestore (ver /firestore.rules).
import { firebaseConfig, ADMIN_EMAIL, NOTIFY_URL } from './firebase-config.js';

const FB = 'https://www.gstatic.com/firebasejs/10.14.1/';
const root = document.getElementById('aula-root');
const DEMO = ['localhost', '127.0.0.1'].includes(location.hostname) && new URLSearchParams(location.search).has('demo');
const PASS_DEFAULT = 70;
const NOTIFY = DEMO ? (new URLSearchParams(location.search).get('notify') || '') : NOTIFY_URL;

/* ---------- utilidades ---------- */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const safeUrl = (u) => (/^https:\/\//i.test(u || '') ? u : '#');
const norm = (e) => String(e || '').trim().toLowerCase();
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const fmtDate = (iso) => { try { return new Date(iso).toLocaleDateString('es-PE', { day: 'numeric', month: 'long', year: 'numeric' }); } catch { return ''; } };
const ICON = {
  lock: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 018 0v3"/></svg>',
  chev: '<svg class="chev h-4 w-4 shrink-0 text-white/40 transition-transform" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  clock: '<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  arrowL: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>',
  arrowR: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>',
  ext: '<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 01-1 1H5a1 1 0 01-1-1V7a1 1 0 011-1h5"/></svg>',
  check: '<svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>',
  file: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8z"/><path d="M14 3v5h5"/></svg>',
  quiz: '<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 114 2c-.8.6-1.5 1-1.5 2M12 17h.01"/></svg>',
};

function parseVideo(v) {
  v = String(v || '').trim();
  if (!v) return null;
  let m;
  if ((m = v.match(/drive\.google\.com\/file\/d\/([\w-]+)/))) return { src: `https://drive.google.com/file/d/${m[1]}/preview` };
  if ((m = v.match(/(?:youtu\.be\/|[?&]v=|embed\/)([\w-]{11})/))) return { src: `https://www.youtube-nocookie.com/embed/${m[1]}?rel=0` };
  return null;
}

// Formato de evaluaciones (texto editable en administración):
//   ? Pregunta
//   - Opción incorrecta
//   + Opción correcta
//   = Explicación (opcional)
// Una línea en blanco separa preguntas.
function parseQuiz(text) {
  const questions = [];
  const blocks = String(text || '').split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  blocks.forEach((b, i) => {
    const q = { q: '', options: [], answer: -1, explain: '' };
    b.split('\n').forEach((raw) => {
      const l = raw.trim();
      if (l.startsWith('?')) q.q = l.slice(1).trim();
      else if (l.startsWith('+')) { q.answer = q.options.length; q.options.push(l.slice(1).trim()); }
      else if (l.startsWith('-')) q.options.push(l.slice(1).trim());
      else if (l.startsWith('=')) q.explain = l.slice(1).trim();
    });
    if (!q.q) throw new Error(`Pregunta ${i + 1}: falta el enunciado (línea que empieza con ?).`);
    if (q.options.length < 2) throw new Error(`Pregunta ${i + 1}: necesita al menos 2 opciones.`);
    if (q.answer < 0 || b.split('\n').filter((l) => l.trim().startsWith('+')).length !== 1) throw new Error(`Pregunta ${i + 1}: debe tener exactamente 1 opción correcta (línea que empieza con +).`);
    if (!q.explain) delete q.explain;
    questions.push(q);
  });
  return questions;
}
const quizToText = (questions) => (questions || []).map((q) => [`? ${q.q}`, ...q.options.map((o, i) => `${i === q.answer ? '+' : '-'} ${o}`), ...(q.explain ? [`= ${q.explain}`] : [])].join('\n')).join('\n\n');

function newCode() {
  const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  const b = crypto.getRandomValues(new Uint8Array(12));
  const s = [...b].map((x) => A[x % A.length]).join('');
  return `NG-${s.slice(0, 4)}-${s.slice(4, 8)}-${s.slice(8, 12)}`;
}

/* ---------- capa de datos ---------- */
async function firebaseApi() {
  const [{ initializeApp }, A, F] = await Promise.all([
    import(FB + 'firebase-app.js'),
    import(FB + 'firebase-auth.js'),
    import(FB + 'firebase-firestore.js'),
  ]);
  const app = initializeApp(firebaseConfig);
  const auth = A.getAuth(app);
  const db = F.getFirestore(app);
  const read = async (ref) => { const s = await F.getDoc(ref); return s.exists() ? s.data() : null; };
  const certs = async (q) => (await F.getDocs(q)).docs.map((d) => ({ code: d.id, ...d.data() }));
  return {
    onAuth: (cb) => A.onAuthStateChanged(auth, (u) => cb(u && u.email ? { email: norm(u.email), name: u.displayName || u.email, photo: u.photoURL || '' } : null)),
    async signIn() {
      const p = new A.GoogleAuthProvider();
      p.setCustomParameters({ prompt: 'select_account' });
      try { await A.signInWithPopup(auth, p); }
      catch (e) {
        if (e.code === 'auth/popup-blocked' || e.code === 'auth/cancelled-popup-request') await A.signInWithRedirect(auth, p);
        else if (e.code !== 'auth/popup-closed-by-user') throw e;
      }
    },
    signOut: () => A.signOut(auth),
    idToken: async () => (auth.currentUser ? auth.currentUser.getIdToken() : ''),
    getEnrollment: (e) => read(F.doc(db, 'enrollments', e)),
    getPrivate: (c) => read(F.doc(db, 'courses', c)).catch(() => null),
    getProgress: (e, c) => read(F.doc(db, 'users', e, 'progress', c)).catch(() => null),
    saveProgress: (e, c, data) => F.setDoc(F.doc(db, 'users', e, 'progress', c), { ...data, updatedAt: F.serverTimestamp() }, { merge: true }),
    listEnrollments: async () => (await F.getDocs(F.collection(db, 'enrollments'))).docs.map((d) => ({ email: d.id, ...d.data() })),
    saveEnrollment: (e, data) => F.setDoc(F.doc(db, 'enrollments', e), data, { merge: true }),
    deleteEnrollment: (e) => F.deleteDoc(F.doc(db, 'enrollments', e)),
    savePrivate: (c, data) => F.setDoc(F.doc(db, 'courses', c), data),
    listMyCerts: (e) => certs(F.query(F.collection(db, 'certificates'), F.where('ownerEmail', '==', e))).catch(() => []),
    requestCert: (code, data) => F.setDoc(F.doc(db, 'certificates', code), data),
    listCerts: () => certs(F.collection(db, 'certificates')),
    updateCert: (code, patch) => F.setDoc(F.doc(db, 'certificates', code), patch, { merge: true }),
    deleteCert: (code) => F.deleteDoc(F.doc(db, 'certificates', code)),
    listMyApps: (e) => certs(F.query(F.collection(db, 'applications'), F.where('ownerEmail', '==', e))).catch(() => []),
    listApps: () => certs(F.collection(db, 'applications')),
    updateApp: (id, patch) => F.setDoc(F.doc(db, 'applications', id), patch, { merge: true }),
    deleteApp: (id) => F.deleteDoc(F.doc(db, 'applications', id)),
  };
}

// Modo demostración: solo en localhost con ?demo=1 (datos ficticios en memoria, sin Firebase).
function demoApi() {
  const qs = new URLSearchParams(location.search);
  const asAdmin = qs.has('admin');
  const user = { email: asAdmin ? ADMIN_EMAIL : 'alumno@gmail.com', name: asAdmin ? 'Nilson (demo)' : 'Alumno Demo', photo: '' };
  const enr = {
    'alumno@gmail.com': { name: 'Alumno Demo', courses: ['pbi-mineria', 'python-mineria'], active: true },
    'maria@gmail.com': { name: 'María Pérez', courses: ['pbi-mineria'], active: true },
  };
  const priv = {
    'pbi-mineria': { pass: 70, sessions: {
      m1s1: { video: 'https://www.youtube.com/watch?v=M7lc1UVf-VE', resources: [{ label: 'Guía de la sesión (PDF)', url: 'https://drive.google.com/' }] },
    }, quizzes: { m1: { questions: [
      { q: '¿Qué es una medida en DAX?', options: ['Una columna calculada', 'Una expresión evaluada en contexto', 'Una tabla'], answer: 1, explain: 'Las medidas se calculan al consultar, según el contexto de filtro.' },
      { q: '¿Qué herramienta transforma datos antes de modelar?', options: ['Power Query', 'Power Pivot', 'Power Apps'], answer: 0 },
    ] } } },
  };
  const allDone = {};
  const prog = { 'alumno@gmail.com|pbi-mineria': { done: { m1s1: true }, last: 'm1s1' } };
  if (qs.has('complete')) {
    ['m1', 'm2', 'm3', 'm4', 'm5'].forEach((m) => [1, 2, 3].forEach((n) => { allDone[`${m}s${n}`] = true; }));
    prog['alumno@gmail.com|pbi-mineria'] = { done: allDone, quiz: { m1: { score: 100, passed: true, attempts: 1, best: 100 } } };
  }
  const certs = [];
  if (qs.has('pending')) certs.push({ code: 'NG-DEMO-PEND-0001', ownerEmail: 'maria@gmail.com', name: 'María Pérez', courseId: 'pbi-mineria', courseTitle: 'Power BI para Minería', hours: 24, status: 'pending', requestedAt: new Date().toISOString() });
  const apps = [];
  if (qs.has('apps')) {
    apps.push({ code: 'ana@gmail.com__pbi-mineria', ownerEmail: 'ana@gmail.com', name: 'Ana Torres', courseId: 'pbi-mineria', courseTitle: 'Power BI para Minería', linkedin: 'https://www.linkedin.com/in/ana-torres', profile: 'Analista de planeamiento en una minera de Arequipa, 3 años de experiencia con Excel avanzado.', goal: 'Automatizar los reportes semanales de producción.', status: 'pending', createdAt: new Date().toISOString() });
    apps.push({ code: 'luis@gmail.com__python-mineria', ownerEmail: 'luis@gmail.com', name: 'Luis Rojas', courseId: 'python-mineria', courseTitle: 'Python para Minería', linkedin: 'https://www.linkedin.com/in/luis-rojas', profile: 'Estudiante de Ingeniería de Minas, 9.º ciclo.', goal: 'Aplicar Python en mi tesis de voladura.', status: 'pending', createdAt: new Date().toISOString() });
  }
  return {
    onAuth: (cb) => setTimeout(() => cb(user), 50),
    signIn: async () => {}, signOut: async () => location.reload(), idToken: async () => 'demo',
    getEnrollment: async (e) => enr[e] || null,
    getPrivate: async (c) => priv[c] || null,
    getProgress: async (e, c) => prog[e + '|' + c] || null,
    saveProgress: async (e, c, d) => {
      const k = e + '|' + c; const p = prog[k] || (prog[k] = {});
      for (const [key, val] of Object.entries(d)) p[key] = (val && typeof val === 'object') ? { ...(p[key] || {}), ...val } : val;
    },
    listEnrollments: async () => Object.entries(enr).map(([email, d]) => ({ email, ...d })),
    saveEnrollment: async (e, d) => { enr[e] = { ...(enr[e] || {}), ...d }; },
    deleteEnrollment: async (e) => { delete enr[e]; },
    savePrivate: async (c, d) => { priv[c] = d; },
    listMyCerts: async (e) => certs.filter((c) => c.ownerEmail === e),
    requestCert: async (code, d) => { certs.push({ code, ...d }); },
    listCerts: async () => certs.slice(),
    updateCert: async (code, patch) => { Object.assign(certs.find((c) => c.code === code), patch); },
    deleteCert: async (code) => { certs.splice(certs.findIndex((c) => c.code === code), 1); },
    listMyApps: async (e) => apps.filter((a) => a.ownerEmail === e),
    listApps: async () => apps.slice(),
    updateApp: async (id, patch) => { Object.assign(apps.find((a) => a.code === id), patch); },
    deleteApp: async (id) => { apps.splice(apps.findIndex((a) => a.code === id), 1); },
  };
}

/* ---------- estado y modelo de progreso ---------- */
const S = { api: null, user: null, isAdmin: false, enr: null, catalog: [], priv: {}, prog: {}, certs: [], myApps: [], pending: 0, flash: '', admin: null, error: '', graded: null, quizMsg: '' };
const quizzesOf = (cid) => S.priv[cid]?.quizzes || {};
const hasQuiz = (cid, mid) => (quizzesOf(cid)[mid]?.questions || []).length > 0;
const passOf = (cid) => S.priv[cid]?.pass || PASS_DEFAULT;

// Secuencia del curso: sesiones de cada módulo seguidas de su evaluación (si existe).
function seqOf(c) {
  const out = [];
  for (const m of c.modules) {
    m.sessions.forEach((s) => out.push({ ...s, mid: m.id, mtitle: m.title, type: 'session' }));
    if (hasQuiz(c.id, m.id)) out.push({ id: 'q-' + m.id, type: 'quiz', mid: m.id, mtitle: m.title, title: 'Evaluación del módulo', duration: `${quizzesOf(c.id)[m.id].questions.length} preguntas`, status: 'available' });
  }
  return out;
}
const trackable = (c) => seqOf(c).filter((s) => s.status !== 'soon');
const isDone = (p, s) => (s.type === 'quiz' ? !!p?.quiz?.[s.mid]?.passed : !!p?.done?.[s.id]);
const doneCount = (c, p) => trackable(c).filter((s) => isDone(p, s)).length;
const pct = (c, p) => { const n = trackable(c).length; return n ? Math.round((doneCount(c, p) / n) * 100) : 0; };
const isComplete = (c, p) => seqOf(c).every((s) => s.status !== 'soon') && trackable(c).length > 0 && doneCount(c, p) === trackable(c).length;
const myCourses = () => (S.isAdmin ? S.catalog : S.catalog.filter((c) => (S.enr?.courses || []).includes(c.id)));
const certFor = (cid, email) => S.certs.find((x) => x.courseId === cid && x.ownerEmail === (email || S.user.email));

async function loadUserData() {
  S.enr = await S.api.getEnrollment(S.user.email).catch(() => null);
  S.prog = {}; S.priv = {};
  await Promise.all(myCourses().map(async (c) => {
    S.priv[c.id] = (await S.api.getPrivate(c.id)) || {};
    S.prog[c.id] = (await S.api.getProgress(S.user.email, c.id)) || {};
  }));
  S.certs = await S.api.listMyCerts(S.user.email);
  S.myApps = (!S.isAdmin && !(S.enr?.courses || []).length) ? await S.api.listMyApps(S.user.email) : [];
  if (S.isAdmin) await refreshPending();
}
// Pide al script de Apps Script que envíe un correo. Devuelve true si se envió.
async function notify(payload) {
  if (!NOTIFY) return null;
  try {
    const idToken = await S.api.idToken();
    const r = await fetch(NOTIFY, { method: 'POST', headers: { 'Content-Type': 'text/plain;charset=utf-8' }, body: JSON.stringify({ ...payload, idToken }) });
    return !!(await r.json()).ok;
  } catch { return false; }
}
async function refreshPending() {
  S.pending = (await S.api.listApps().catch(() => [])).filter((a) => a.status === 'pending').length;
}

/* ---------- vistas ---------- */
const btnPrimary = 'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-white';
const gradient = 'background:linear-gradient(135deg, #0068FF, #00D2FF); box-shadow:0 2px 8px rgba(0,194,255,0.25);';
const card = 'rounded-2xl border border-white/10 bg-white/5';
const bar = (n) => `<div class="h-1.5 rounded-full bg-white/10 overflow-hidden"><div class="h-full rounded-full" style="width:${n}%; background:linear-gradient(90deg, #0068FF, #00D2FF);"></div></div>`;

function topbar(active) {
  const link = (href, label, key) => `<a href="${href}" class="px-3 py-1.5 rounded-lg text-sm ${active === key ? 'bg-white/10 text-white' : 'text-white/70 hover:text-white hover:bg-white/5'}">${label}</a>`;
  return `<div class="flex flex-wrap items-center justify-between gap-3 mb-8 pb-4 border-b border-white/10">
    <nav class="flex items-center gap-1">
      <span class="mr-3 hidden text-xs font-semibold uppercase tracking-[0.18em] text-sky-300/80 sm:inline">Aula virtual</span>
      ${link('#/', 'Mis cursos', 'home')}
      ${S.isAdmin ? link('#/admin', 'Administración' + (S.pending ? ` <span class="ml-1 rounded-md bg-amber-400/90 text-slate-900 text-[11px] font-semibold px-1.5 py-0.5">${S.pending}</span>` : ''), 'admin') : ''}
    </nav>
    <div class="flex items-center gap-3 text-sm">
      ${S.user.photo ? `<img src="${esc(S.user.photo)}" referrerpolicy="no-referrer" alt="" class="h-8 w-8 rounded-full ring-1 ring-white/20">` : ''}
      <span class="hidden sm:block text-white/70">${esc(S.user.name)}</span>
      <button data-act="signout" class="rounded-lg border border-white/15 px-3 py-1.5 text-white/70 hover:text-white hover:border-white/30">Salir</button>
    </div>
  </div>`;
}

function loginView() {
  return `<div class="max-w-md mx-auto mt-10 ${card} p-8 text-center">
    <h1 class="text-2xl font-display font-bold">Aula virtual</h1>
    <p class="mt-2 text-sm text-white/70">Acceso solo por invitación. Ingresa con la cuenta de Google con la que te inscribiste.</p>
    ${S.error ? `<p class="mt-4 text-sm text-red-300">${esc(S.error)}</p>` : ''}
    <button data-act="signin" class="${btnPrimary} mt-6 w-full" style="${gradient}">Continuar con Google</button>
    <p class="mt-5 text-xs text-white/50">¿Aún no tienes acceso? Revisa los <a class="underline" href="/cursos/">cursos</a> y escríbeme por correo.</p>
  </div>`;
}

function noAccessView() {
  return `${topbar('home')}<div class="max-w-xl ${card} p-8">
    <h1 class="text-xl font-bold">Tu cuenta aún no tiene cursos asignados</h1>
    <p class="mt-2 text-sm text-white/70">Ingresaste como <strong>${esc(S.user.email)}</strong>.</p>
    ${S.myApps.length ? `<h2 class="mt-5 text-sm font-semibold uppercase tracking-wider text-white/50">Tus postulaciones</h2>
      <ul class="mt-2 space-y-2">${S.myApps.map((a) => `<li class="flex items-center justify-between gap-3 rounded-lg bg-white/5 px-3 py-2 text-sm"><span>${esc(a.courseTitle)}</span><span class="text-xs ${a.status === 'rejected' ? 'text-red-300' : 'text-amber-300'}">${a.status === 'rejected' ? 'No seleccionada' : a.status === 'approved' ? 'Aprobada' : 'En revisión'}</span></li>`).join('')}</ul>`
      : `<p class="mt-3 text-sm text-white/70">Si quieres acceder a un curso, envía tu postulación desde la sección de cursos con esta misma cuenta.</p>`}
    <a href="/cursos/" class="inline-flex mt-5 rounded-lg border border-white/15 px-4 py-2 text-sm text-white/80 hover:text-white">Ver cursos</a>
  </div>`;
}

function certBlock(c, p) {
  const ct = certFor(c.id);
  if (ct?.status === 'issued') return `<a href="/certificado/?c=${encodeURIComponent(ct.code)}" target="_blank" rel="noopener" class="mt-3 flex items-center justify-center rounded-lg border border-emerald-400/50 bg-emerald-500/15 px-4 py-2 text-sm font-semibold text-emerald-200 hover:bg-emerald-500/25">Ver certificado</a>`;
  if (ct) return '<p class="mt-3 text-center text-xs text-amber-300/90">Certificado solicitado: pendiente de aprobación.</p>';
  if (isComplete(c, p)) return `<button data-act="cert-request" data-cid="${esc(c.id)}" class="mt-3 w-full rounded-lg border border-white/20 px-4 py-2 text-sm font-semibold text-white hover:bg-white/10">Solicitar certificado</button>`;
  return '';
}

// Curso con actividad más reciente (para "Continúa donde lo dejaste").
function lastActive(cs) {
  let best = null, bt = -1;
  for (const c of cs) {
    const p = S.prog[c.id] || {};
    if (!p.last && !doneCount(c, p)) continue;
    const t = (p.updatedAt && (p.updatedAt.seconds || p.updatedAt)) || 0;
    if (best === null || t > bt) { best = c; bt = t; }
  }
  return best;
}

function homeView() {
  const cs = myCourses();
  const first = esc((S.user.name || '').split(' ')[0]);
  const issued = S.certs.filter((x) => x.status === 'issued');
  const totalDone = cs.reduce((n, c) => n + doneCount(c, S.prog[c.id] || {}), 0);
  const totalAll = cs.reduce((n, c) => n + trackable(c).length, 0);
  const stat = (label, value, sub) => `<div class="${card} px-4 py-3"><p class="text-[11px] uppercase tracking-wider text-white/40">${label}</p>
    <p class="mt-1 text-2xl font-display font-bold">${value}</p><p class="text-xs text-white/50">${sub}</p></div>`;

  const cur = lastActive(cs);
  let hero = '';
  if (cur) {
    const p = S.prog[cur.id] || {};
    const ls = seqOf(cur).find((s) => s.id === p.last);
    const n = pct(cur, p);
    hero = `<section class="${card} mb-8 grid overflow-hidden md:grid-cols-[300px_1fr]">
      <img src="${esc(cur.cover)}" alt="" class="h-full min-h-[150px] w-full object-cover object-left">
      <div class="flex flex-col justify-center p-6">
        <p class="text-xs font-semibold uppercase tracking-[0.18em] text-sky-300/80">Continúa donde lo dejaste</p>
        <h2 class="mt-1 text-xl font-bold">${esc(cur.title)}</h2>
        ${ls ? `<p class="mt-1 text-sm text-white/60">Última sesión: ${esc(ls.title)}</p>` : ''}
        <div class="mt-4 max-w-md"><div class="mb-1.5 flex justify-between text-xs text-white/60"><span>${doneCount(cur, p)} de ${trackable(cur).length} actividades</span><span>${n}%</span></div>${bar(n)}</div>
        <a href="#/c/${esc(cur.id)}/${esc(p.last || '')}" class="${btnPrimary} mt-5 self-start px-6" style="${gradient}">Continuar</a>
      </div></section>`;
  }

  const cards = cs.map((c) => {
    const p = S.prog[c.id] || {};
    const seq = trackable(c);
    const target = p.last || (seq[0] || seqOf(c)[0])?.id || '';
    const n = pct(c, p);
    const chip = n === 100 ? '<span class="absolute left-3 top-3 rounded-md bg-emerald-500/90 px-2 py-0.5 text-[11px] font-semibold text-white">Completado</span>'
      : c.status === 'in_development' ? '<span class="absolute left-3 top-3 rounded-md bg-amber-400/90 px-2 py-0.5 text-[11px] font-semibold text-slate-900">En desarrollo</span>' : '';
    return `<article class="${card} flex flex-col overflow-hidden">
      <div class="relative"><img src="${esc(c.cover)}" alt="" class="aspect-[1200/630] w-full object-cover" loading="lazy">${chip}</div>
      <div class="flex flex-1 flex-col p-4">
        <h3 class="font-semibold">${esc(c.title)}</h3>
        <p class="mt-1 line-clamp-2 text-sm text-white/70">${esc(c.subtitle)}</p>
        <div class="mt-4">
          <div class="mb-1.5 flex justify-between text-xs text-white/60"><span>${doneCount(c, p)} de ${seq.length} actividades</span><span>${n}%</span></div>
          ${bar(n)}
        </div>
        <a href="#/c/${esc(c.id)}/${esc(target)}" class="${btnPrimary} mt-4" style="${gradient}">${n === 0 ? 'Comenzar' : n === 100 ? 'Repasar' : 'Continuar'}</a>
        ${certBlock(c, p)}
      </div>
    </article>`;
  }).join('');

  const certs = issued.length ? `<section class="mt-10"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Mis certificados</h2>
    <ul class="mt-3 grid gap-3 sm:grid-cols-2">${issued.map((ct) => `<li class="${card} flex items-center justify-between gap-3 px-4 py-3">
      <div class="min-w-0"><p class="truncate text-sm font-medium">${esc(ct.courseTitle)}</p><p class="text-xs text-white/50">Emitido el ${esc(fmtDate(ct.issuedAt))}</p></div>
      <a href="/certificado/?c=${encodeURIComponent(ct.code)}" target="_blank" rel="noopener" class="shrink-0 rounded-lg border border-emerald-400/50 bg-emerald-500/15 px-3 py-1.5 text-xs font-semibold text-emerald-200 hover:bg-emerald-500/25">Ver</a></li>`).join('')}</ul></section>` : '';

  return `${topbar('home')}
    <header class="mb-6"><h1 class="text-2xl font-display font-bold">Hola, ${first}</h1>
    <p class="mt-1 text-sm text-white/60">Este es tu espacio de aprendizaje.</p></header>
    <div class="mb-8 grid grid-cols-3 gap-3">
      ${stat('Cursos', cs.length, 'asignados')}
      ${stat('Actividades', totalDone, `de ${totalAll} completadas`)}
      ${stat('Certificados', issued.length, 'emitidos')}
    </div>
    ${hero}
    <h2 class="mb-4 text-sm font-semibold uppercase tracking-wider text-white/50">Mis cursos</h2>
    <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">${cards}</div>
    ${certs}`;
}

function sidebarHtml(c, cur, p) {
  const seq = seqOf(c);
  const mods = c.modules.map((m, mi) => {
    const items = seq.filter((s) => s.mid === m.id);
    const track = items.filter((s) => s.status !== 'soon');
    const md = track.filter((s) => isDone(p, s)).length;
    const full = track.length > 0 && md === track.length;
    const hasCur = items.some((s) => s.id === cur.id);
    const lis = items.map((s) => {
      const done = isDone(p, s);
      const soon = s.status === 'soon';
      const isCur = s.id === cur.id;
      const q = s.type === 'quiz';
      const state = soon ? `<span class="mt-0.5 text-white/30">${ICON.lock.replace('h-4 w-4', 'h-3.5 w-3.5')}</span>`
        : done ? `<span class="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-white">${ICON.check}</span>`
        : `<span class="mt-0.5 h-4 w-4 shrink-0 rounded-full border ${isCur ? 'border-sky-300' : 'border-white/30'}"></span>`;
      return `<li><a href="#/c/${esc(c.id)}/${esc(s.id)}" ${isCur ? 'aria-current="page"' : ''} class="flex items-start gap-3 rounded-lg px-3 py-2 text-sm transition ${isCur ? 'bg-accent/15 font-medium text-white ring-1 ring-accent/40' : soon ? 'text-white/40' : 'text-white/70 hover:bg-white/5 hover:text-white'}">
        ${state}
        <span class="min-w-0 flex-1"><span class="block leading-snug ${q ? 'text-sky-200' : ''}">${q ? ICON.quiz.replace('class="', 'class="mr-1 inline -mt-0.5 ') : ''}${esc(s.title)}</span>
        <span class="block text-xs text-white/40">${esc(s.duration || '')}${soon ? ' · Próximamente' : ''}</span></span></a></li>`;
    }).join('');
    return `<details class="overflow-hidden rounded-xl border border-white/10 bg-white/5" ${hasCur ? 'open' : ''}>
      <summary class="flex cursor-pointer list-none items-center gap-3 px-3.5 py-3 hover:bg-white/5">
        <span class="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${full ? 'bg-emerald-500/20 text-emerald-300' : 'bg-white/10 text-white/70'}">${full ? ICON.check : mi + 1}</span>
        <span class="min-w-0 flex-1"><span class="block text-sm font-semibold leading-snug">${esc(m.title)}</span>
        <span class="mt-0.5 block text-xs text-white/45">${md} de ${track.length} completadas</span></span>
        ${ICON.chev}
      </summary>
      <ul class="space-y-0.5 px-2 pb-2">${lis}</ul></details>`;
  }).join('');
  return `<div class="space-y-2.5">${mods}</div>`;
}

function quizMain(c, cur, p) {
  const quiz = quizzesOf(c.id)[cur.mid];
  const pass = passOf(c.id);
  const best = p.quiz?.[cur.mid];
  const g = S.graded && S.graded.key === c.id + '/' + cur.id ? S.graded : null;
  const qs = quiz.questions.map((q, i) => {
    const opts = q.options.map((o, j) => {
      let cls = 'border-white/10 hover:bg-white/10';
      if (g) cls = j === q.answer ? 'border-emerald-400/60 bg-emerald-500/15' : (g.answers[i] === j ? 'border-red-400/60 bg-red-500/15' : 'border-white/10 opacity-70');
      return `<label class="flex items-start gap-3 rounded-xl border ${cls} px-4 py-2.5 text-sm cursor-pointer">
        <input type="radio" name="q${i}" value="${j}" class="mt-1" ${g ? 'disabled' : ''} ${g && g.answers[i] === j ? 'checked' : ''}><span>${esc(o)}</span></label>`;
    }).join('');
    const fb = g ? `<p class="mt-2 text-xs ${g.answers[i] === q.answer ? 'text-emerald-300' : 'text-red-300'}">${g.answers[i] === q.answer ? 'Correcto.' : 'Incorrecto.'} ${esc(q.explain || '')}</p>` : '';
    return `<fieldset class="${card} p-4"><legend class="sr-only">Pregunta ${i + 1}</legend>
      <p class="text-sm font-medium"><span class="text-white/40 mr-1">${i + 1}.</span>${esc(q.q)}</p><div class="mt-3 space-y-2">${opts}</div>${fb}</fieldset>`;
  }).join('');
  const result = g ? `<div class="${card} p-4 ${g.passed ? 'border-emerald-400/50' : 'border-amber-400/50'}">
      <p class="font-semibold ${g.passed ? 'text-emerald-300' : 'text-amber-300'}">${g.passed ? 'Aprobado' : 'Aún no alcanzas el puntaje mínimo'}: ${g.score}% (${g.correct} de ${g.total})</p>
      <p class="text-xs text-white/60 mt-1">Puntaje mínimo: ${pass}%.</p></div>` : '';
  return `<p class="text-xs text-white/50">${esc(cur.mtitle)}</p>
    <h1 class="mt-1 text-2xl font-display font-bold">Evaluación del módulo</h1>
    <p class="mt-2 text-white/70 text-sm">${quiz.questions.length} preguntas · puntaje mínimo ${pass}%. Puedes repetirla las veces que necesites.${best ? ` Tu mejor resultado: <strong>${best.best}%</strong> (${best.attempts} ${best.attempts === 1 ? 'intento' : 'intentos'}).` : ''}</p>
    <div class="mt-5 space-y-4">${result}${qs}</div>
    ${S.quizMsg ? `<p class="mt-4 text-sm text-amber-300">${esc(S.quizMsg)}</p>` : ''}
    <div class="mt-5 flex flex-wrap gap-3">
      ${g ? `<button data-act="quiz-retry" class="rounded-lg border border-white/20 px-4 py-2 text-sm text-white hover:bg-white/10">Reintentar</button>`
          : `<button data-act="quiz-submit" class="${btnPrimary}" style="${gradient}">Calificar</button>`}
    </div>`;
}

// Etiqueta corta según el destino de un material.
function matMeta(url) {
  try {
    const u = new URL(url); const h = u.hostname.replace(/^www\./, '');
    if (/(^|\.)(drive|docs)\.google\.com$/.test(h)) return 'Drive';
    if (/colab\.research\.google\.com$/.test(h)) return 'Colab';
    if (/github\.com$/.test(h)) return 'GitHub';
    if (/\.pdf$/i.test(u.pathname)) return 'PDF';
    return h;
  } catch { return 'Enlace'; }
}

function courseView(cid, sid) {
  const c = S.catalog.find((x) => x.id === cid);
  if (!c || !myCourses().includes(c)) return `${topbar('home')}<p class="text-white/70">No tienes acceso a este curso. <a class="underline" href="#/">Volver</a></p>`;
  const all = seqOf(c);
  const p = S.prog[cid] || (S.prog[cid] = {});
  const cur = all.find((s) => s.id === sid) || all.find((s) => s.id === p.last) || trackable(c)[0] || all[0];
  const idx = all.indexOf(cur);
  const prev = all[idx - 1], next = all[idx + 1];
  const prevHref = prev ? `#/c/${cid}/${prev.id}` : '', nextHref = next ? `#/c/${cid}/${next.id}` : '';
  const total = trackable(c).length, done = doneCount(c, p), n = pct(c, p);

  const navBtn = 'inline-flex items-center gap-2 rounded-lg border border-white/15 px-3.5 py-2 text-sm text-white/80 transition hover:border-white/30 hover:text-white';
  const actionBar = (center) => `<div class="sticky bottom-3 z-10 mt-8"><div class="flex items-center gap-2 rounded-2xl border border-white/10 px-2.5 py-2.5 shadow-soft backdrop-blur" style="background:rgba(13,31,58,.92);">
      ${prev ? `<a href="${prevHref}" title="Anterior (←)" class="${navBtn}">${ICON.arrowL}<span class="hidden sm:inline">Anterior</span></a>` : `<span class="${navBtn} pointer-events-none opacity-30">${ICON.arrowL}<span class="hidden sm:inline">Anterior</span></span>`}
      <div class="flex flex-1 justify-center">${center}</div>
      ${next ? `<a href="${nextHref}" title="Siguiente (→)" class="${navBtn}"><span class="hidden sm:inline">Siguiente</span>${ICON.arrowR}</a>` : `<span class="${navBtn} pointer-events-none opacity-30"><span class="hidden sm:inline">Siguiente</span>${ICON.arrowR}</span>`}
    </div></div>`;

  let main;
  if (cur.type === 'quiz') {
    main = `${quizMain(c, cur, p)}${actionBar('')}`;
  } else {
    const ov = S.priv[cid]?.sessions?.[cur.id] || {};
    const video = parseVideo(ov.video);
    const res = (ov.resources || []).filter((r) => r && r.url);
    const soon = cur.status === 'soon';
    const isDone = !!p.done?.[cur.id];

    const media = soon
      ? `<div class="${card} mt-6 flex flex-col items-center px-6 py-14 text-center"><span class="text-white/40">${ICON.lock.replace('h-4 w-4', 'h-7 w-7')}</span>
          <p class="mt-3 font-semibold">Esta sesión se publicará próximamente</p><p class="mt-1 text-sm text-white/60">Podrás verla aquí en cuanto esté disponible.</p></div>`
      : video ? `<div class="mt-6 aspect-video overflow-hidden rounded-2xl border border-white/10 bg-black shadow-soft">
          <iframe src="${esc(video.src)}" title="${esc(cur.title)}" class="h-full w-full" allow="autoplay; encrypted-media; fullscreen" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div>` : '';

    const resHtml = res.length ? `<ul class="mt-3 space-y-2">${res.map((r) => `<li><a href="${esc(safeUrl(r.url))}" target="_blank" rel="noopener noreferrer"
        class="group flex items-center gap-3 ${card} px-4 py-3 text-sm transition hover:border-white/25 hover:bg-white/10">
        <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-sky-400/10 text-sky-300">${ICON.file}</span>
        <span class="min-w-0 flex-1"><span class="block truncate font-medium">${esc(r.label || 'Material')}</span><span class="block text-xs text-white/45">${esc(matMeta(r.url))}</span></span>
        <span class="flex shrink-0 items-center gap-1 text-xs text-white/50 group-hover:text-white/80">Abrir ${ICON.ext}</span></a></li>`).join('')}</ul>`
      : `<p class="mt-3 rounded-xl border border-dashed border-white/10 px-4 py-6 text-center text-sm text-white/50">Aún no hay materiales publicados para esta sesión.</p>`;

    const doneBtn = soon ? '' : isDone
      ? `<button data-act="toggle-done" title="Clic para desmarcar" class="${btnPrimary} border border-emerald-400/50 bg-emerald-500/20 text-emerald-200">✓ Completada</button>`
      : `<button data-act="toggle-done" class="${btnPrimary}" style="${gradient}"><span class="sm:hidden">Completar</span><span class="hidden sm:inline">Marcar como completada</span></button>`;

    main = `<div class="flex flex-wrap items-center gap-2 text-xs text-white/50">
        <span>${esc(cur.mtitle)}</span>${cur.duration ? `<span>·</span><span class="inline-flex items-center gap-1">${ICON.clock}${esc(cur.duration)}</span>` : ''}
        ${isDone ? '<span class="rounded-md bg-emerald-500/20 px-2 py-0.5 font-semibold text-emerald-300">Completada</span>' : ''}
      </div>
      <h1 class="mt-2 text-2xl font-display font-bold leading-tight sm:text-3xl">${esc(cur.title)}</h1>
      ${cur.summary ? `<p class="mt-2 text-white/70">${esc(cur.summary)}</p>` : ''}
      ${media}
      ${soon ? '' : `<div class="mt-8 grid gap-6 lg:grid-cols-5">
        <div class="lg:col-span-3"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Materiales</h2>${resHtml}</div>
        <div class="lg:col-span-2"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Mis notas <span id="note-status" class="ml-2 font-normal normal-case tracking-normal text-white/40"></span></h2>
          <textarea id="note" rows="7" placeholder="Escribe tus apuntes de esta sesión…" class="mt-3 w-full resize-y rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-white placeholder-white/30 focus:border-white/30 focus:outline-none">${esc(p.notes?.[cur.id] || '')}</textarea></div>
      </div>`}
      ${actionBar(doneBtn)}`;
  }

  const cb = isComplete(c, p) ? certBlock(c, p).replace('mt-3 ', '') : '';
  const certBanner = isComplete(c, p) ? `<div class="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-emerald-400/40 bg-emerald-500/10 p-4">
      <div><p class="font-semibold text-emerald-200">¡Completaste el curso!</p><p class="text-sm text-emerald-100/70">Terminaste todas las actividades.</p></div>
      ${cb ? `<div class="w-full sm:w-60">${cb}</div>` : ''}</div>` : '';

  const side = sidebarHtml(c, cur, p);
  return `${topbar('home')}
  <div class="mb-6 flex flex-wrap items-end justify-between gap-4">
    <div class="min-w-0"><a href="#/" class="text-sm text-white/50 hover:text-white/80">← Mis cursos</a>
      <p class="mt-1 truncate font-display text-lg font-bold sm:text-xl">${esc(c.title)}</p></div>
    <div class="w-full sm:w-72"><div class="mb-1.5 flex justify-between text-xs text-white/60"><span>${done} de ${total} actividades</span><span>${n}%</span></div>${bar(n)}</div>
  </div>
  ${certBanner}
  <details class="${card} mb-5 lg:hidden"><summary class="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-semibold"><span>Contenido del curso</span><span class="text-xs font-normal text-white/50">${done}/${total}</span></summary><div class="px-3 pb-3">${side}</div></details>
  <div class="grid items-start gap-8 lg:grid-cols-[320px_1fr]">
    <aside class="hidden pr-1 lg:sticky lg:top-24 lg:block lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto">${side}</aside>
    <section class="min-w-0" data-session="${esc(cur.id)}" data-course="${esc(cid)}" data-type="${cur.type}" data-prev="${esc(prevHref)}" data-next="${esc(nextHref)}">${main}</section>
  </div>`;
}

/* ---------- administración ---------- */
async function loadAdmin() {
  const enrollments = (await S.api.listEnrollments()).sort((a, b) => a.email.localeCompare(b.email));
  await Promise.all(S.catalog.map(async (c) => { if (!(c.id in S.priv)) S.priv[c.id] = (await S.api.getPrivate(c.id)) || {}; }));
  const progress = {};
  await Promise.all(enrollments.flatMap((e) => (e.courses || []).map(async (cid) => {
    (progress[e.email] ||= {})[cid] = (await S.api.getProgress(e.email, cid)) || {};
  })));
  const certs = (await S.api.listCerts()).sort((a, b) => String(b.requestedAt).localeCompare(String(a.requestedAt)));
  const apps = (await S.api.listApps()).sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  S.pending = apps.filter((a) => a.status === 'pending').length;
  S.admin = { enrollments, progress, certs, apps, courseId: S.admin?.courseId || S.catalog[0]?.id };
}

// Filtra la lista de alumnos (búsqueda y curso) sin volver a dibujar la página.
function filterStudents() {
  const q = (document.getElementById('al-q')?.value || '').trim().toLowerCase();
  const cid = document.getElementById('al-course')?.value || '';
  let n = 0;
  root.querySelectorAll('[data-row]').forEach((el) => {
    const ok = (!q || el.dataset.search.includes(q)) && (!cid || el.dataset.courses.includes(',' + cid + ','));
    el.classList.toggle('hidden', !ok);
    if (ok) n++;
  });
  const cnt = document.getElementById('al-count'); if (cnt) cnt.textContent = `${n} alumno${n === 1 ? '' : 's'}`;
  const empty = document.getElementById('al-empty'); if (empty) empty.classList.toggle('hidden', n !== 0);
}

// Descarga la lista de alumnos con su avance en un CSV (abre bien en Excel).
function exportCsv() {
  const A = S.admin;
  const q = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = [['Nombre', 'Correo', 'Activo', 'Curso', 'Avance %', 'Actividades completadas', 'Actividades totales', 'Alta'].map(q).join(',')];
  for (const e of A.enrollments) {
    const cs = (e.courses || []).map((id) => S.catalog.find((c) => c.id === id)).filter(Boolean);
    if (!cs.length) rows.push([e.name, e.email, e.active !== false ? 'Sí' : 'No', '', '', '', '', e.createdAt || ''].map(q).join(','));
    for (const c of cs) {
      const pr = A.progress[e.email]?.[c.id];
      rows.push([e.name, e.email, e.active !== false ? 'Sí' : 'No', c.title, pct(c, pr), doneCount(c, pr), trackable(c).length, e.createdAt || ''].map(q).join(','));
    }
  }
  const blob = new Blob(['\ufeff' + rows.join('\r\n')], { type: 'text/csv;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = `alumnos-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
}

function adminView() {
  const A = S.admin;
  const tab = ['resumen', 'postulaciones', 'alumnos', 'certificados', 'contenido'].includes(parseRoute().tab) ? parseRoute().tab : 'resumen';
  const pendingApps = A.apps.filter((a) => a.status === 'pending');
  const doneApps = A.apps.filter((a) => a.status !== 'pending');
  const pendingCerts = A.certs.filter((x) => x.status !== 'issued');
  const issuedCerts = A.certs.filter((x) => x.status === 'issued');
  const activeN = A.enrollments.filter((e) => e.active !== false).length;

  const flash = S.flash ? `<div class="mb-6 rounded-xl border border-emerald-400/40 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">${esc(S.flash)}</div>` : '';
  S.flash = '';

  const progOf = (email, cid) => { const c = S.catalog.find((x) => x.id === cid); return c ? pct(c, A.progress[email]?.[cid]) : 0; };
  const pairs = A.enrollments.flatMap((e) => (e.courses || []).map((cid) => progOf(e.email, cid)));
  const avgProg = pairs.length ? Math.round(pairs.reduce((a, b) => a + b, 0) / pairs.length) : null;
  const badge = (n, warn) => (n ? `<span class="ml-1.5 rounded-md px-1.5 py-0.5 text-[11px] font-semibold ${warn ? 'bg-amber-400/90 text-slate-900' : 'bg-white/10 text-white/70'}">${n}</span>` : '');
  const ghost = 'rounded-lg border border-white/15 px-3.5 py-2 text-sm text-white/80 transition hover:border-white/30 hover:text-white';
  const input = 'w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm placeholder-white/30 focus:border-white/30 focus:outline-none';

  /* ---- piezas reutilizables ---- */
  const appCard = (a) => `<div class="${card} p-4">
      <div class="flex flex-wrap items-start justify-between gap-3">
        <div class="min-w-0"><div class="text-sm font-medium">${esc(a.name)} <span class="text-xs text-white/50">· ${esc(a.ownerEmail)}</span>${a.linkedin ? ` <a href="${esc(safeUrl(a.linkedin))}" target="_blank" rel="noopener noreferrer" class="ml-1 text-xs text-sky-300 hover:text-sky-200">LinkedIn ↗</a>` : ''}</div>
          <div class="mt-0.5 text-xs text-sky-300/80">${esc(a.courseTitle)} · ${esc(fmtDate(a.createdAt))}</div></div>
        <div class="flex flex-wrap items-center gap-2 text-xs">
          ${a.status === 'pending' ? `<button data-act="app-approve" data-code="${esc(a.code)}" class="rounded-lg bg-emerald-500/20 px-3 py-1.5 font-semibold text-emerald-200 hover:bg-emerald-500/30">Aprobar e inscribir</button><button data-act="app-reject" data-code="${esc(a.code)}" class="rounded-lg border border-white/15 px-3 py-1.5 text-red-300 hover:border-red-300/40">Rechazar</button>`
            : `<span class="${a.status === 'approved' ? 'text-emerald-300' : 'text-white/40'}">${a.status === 'approved' ? 'Aprobada' : 'Rechazada'}</span>`}
          <button data-act="app-delete" data-code="${esc(a.code)}" class="text-white/35 hover:text-white/70">Eliminar</button></div></div>
      <p class="mt-3 text-xs text-white/45">Perfil</p><p class="text-sm text-white/80 whitespace-pre-line">${esc(a.profile)}</p>
      <p class="mt-2 text-xs text-white/45">Qué espera aplicar</p><p class="text-sm text-white/80 whitespace-pre-line">${esc(a.goal)}</p></div>`;

  const certCard = (ct) => {
    const issued = ct.status === 'issued';
    const c = S.catalog.find((x) => x.id === ct.courseId);
    const pr = A.progress[ct.ownerEmail]?.[ct.courseId];
    const n = c && pr ? pct(c, pr) : null;
    return `<div class="${card} flex flex-wrap items-center justify-between gap-3 p-4">
      <div class="min-w-0"><p class="text-sm font-medium">${esc(ct.name)} <span class="text-xs text-white/50">· ${esc(ct.ownerEmail)}</span></p>
        <p class="mt-0.5 text-xs text-sky-300/80">${esc(ct.courseTitle)}${n !== null ? ` · Progreso ${n}%` : ''}</p>
        <p class="mt-0.5 text-xs text-white/35">${esc(ct.code)} · ${issued ? 'Emitido el ' + esc(fmtDate(ct.issuedAt)) : 'Solicitado el ' + esc(fmtDate(ct.requestedAt))}</p></div>
      <div class="flex flex-wrap items-center gap-2 text-xs">
        ${issued ? `<a class="rounded-lg border border-white/15 px-3 py-1.5 text-sky-300 hover:border-sky-300/40" target="_blank" rel="noopener" href="/certificado/?c=${encodeURIComponent(ct.code)}">Ver</a>`
          : `<button data-act="cert-issue" data-code="${esc(ct.code)}" class="rounded-lg bg-emerald-500/20 px-3 py-1.5 font-semibold text-emerald-200 hover:bg-emerald-500/30">Aprobar y emitir</button>`}
        <button data-act="cert-delete" data-code="${esc(ct.code)}" class="text-red-300 hover:text-red-200">${issued ? 'Revocar' : 'Rechazar'}</button></div></div>`;
  };

  const sessCount = (c) => c.modules.reduce((n, m) => n + m.sessions.length, 0);
  const hasContent = (cid, sid) => { const o = S.priv[cid]?.sessions?.[sid]; return !!(o && (o.video || (o.resources || []).length)); };
  const coverage = (c) => c.modules.reduce((n, m) => n + m.sessions.filter((x) => hasContent(c.id, x.id)).length, 0);
  const quizCount = (c) => Object.values(S.priv[c.id]?.quizzes || {}).filter((z) => (z.questions || []).length).length;

  /* ---- pestañas ---- */
  const tabs = [['resumen', 'Resumen', ''], ['postulaciones', 'Postulaciones', badge(pendingApps.length, true)],
    ['alumnos', 'Alumnos', badge(A.enrollments.length, false)], ['certificados', 'Certificados', badge(pendingCerts.length, true)], ['contenido', 'Contenido', '']];
  const nav = `<nav class="mb-7 flex gap-1 overflow-x-auto border-b border-white/10" aria-label="Secciones de administración">${tabs.map(([k, label, b]) =>
    `<a href="#/admin/${k}" ${k === tab ? 'aria-current="page"' : ''} class="-mb-px shrink-0 border-b-2 px-4 py-2.5 text-sm transition ${k === tab ? 'border-sky-400 font-semibold text-white' : 'border-transparent text-white/60 hover:text-white'}">${label}${b}</a>`).join('')}</nav>`;

  let body = '';

  /* ---- RESUMEN ---- */
  if (tab === 'resumen') {
    const stat = (label, value, sub, warn) => `<div class="${card} px-4 py-3 ${warn ? 'border-amber-400/40' : ''}"><p class="text-[11px] uppercase tracking-wider text-white/40">${label}</p>
      <p class="mt-1 font-display text-2xl font-bold ${warn ? 'text-amber-300' : ''}">${value}</p><p class="text-xs text-white/50">${sub}</p></div>`;
    const attention = pendingApps.length || pendingCerts.length
      ? `${pendingApps.length ? `<div><div class="mb-2 flex items-center justify-between"><h3 class="text-sm font-semibold">Postulaciones pendientes</h3><a href="#/admin/postulaciones" class="text-xs text-sky-300 hover:text-sky-200">Ver todas</a></div><div class="space-y-3">${pendingApps.slice(0, 3).map(appCard).join('')}</div></div>` : ''}
         ${pendingCerts.length ? `<div><div class="mb-2 flex items-center justify-between"><h3 class="text-sm font-semibold">Certificados por aprobar</h3><a href="#/admin/certificados" class="text-xs text-sky-300 hover:text-sky-200">Ver todos</a></div><div class="space-y-3">${pendingCerts.slice(0, 3).map(certCard).join('')}</div></div>` : ''}`
      : `<div class="${card} px-5 py-6 text-center"><p class="font-semibold text-emerald-300">Todo al día</p><p class="mt-1 text-sm text-white/60">No tienes postulaciones ni certificados pendientes.</p></div>`;
    const courseRows = S.catalog.map((c) => {
      const enrolled = A.enrollments.filter((e) => (e.courses || []).includes(c.id));
      const ap = enrolled.length ? Math.round(enrolled.reduce((n, e) => n + progOf(e.email, c.id), 0) / enrolled.length) : null;
      return `<tr class="border-t border-white/10"><td class="py-3 pr-4 text-sm font-medium">${esc(c.title)}${c.status === 'in_development' ? ' <span class="ml-1 rounded bg-amber-400/90 px-1.5 py-0.5 text-[10px] font-semibold text-slate-900">En desarrollo</span>' : ''}</td>
        <td class="py-3 pr-4 text-sm text-white/70">${enrolled.length}</td><td class="py-3 pr-4 text-sm text-white/70">${ap === null ? '—' : ap + '%'}</td>
        <td class="py-3 pr-4 text-sm text-white/70">${coverage(c)} de ${sessCount(c)}</td><td class="py-3 text-sm text-white/70">${quizCount(c)}</td></tr>`;
    }).join('');
    body = `<div class="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
        ${stat('Alumnos activos', activeN, `de ${A.enrollments.length} registrados`)}
        ${stat('Postulaciones', pendingApps.length, pendingApps.length === 1 ? 'pendiente' : 'pendientes', pendingApps.length > 0)}
        ${stat('Certificados', pendingCerts.length, pendingCerts.length === 1 ? 'por aprobar' : 'por aprobar', pendingCerts.length > 0)}
        ${stat('Avance promedio', avgProg === null ? '—' : avgProg + '%', 'de los alumnos inscritos')}
      </div>
      <section class="mb-8 space-y-5"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Requiere tu atención</h2>${attention}</section>
      <section><h2 class="mb-3 text-sm font-semibold uppercase tracking-wider text-white/50">Cursos</h2>
        <div class="${card} overflow-x-auto px-5 py-2"><table class="w-full text-left"><thead><tr class="text-[11px] uppercase tracking-wider text-white/40">
          <th class="py-3 pr-4 font-medium">Curso</th><th class="py-3 pr-4 font-medium">Alumnos</th><th class="py-3 pr-4 font-medium">Avance</th><th class="py-3 pr-4 font-medium">Sesiones con contenido</th><th class="py-3 font-medium">Evaluaciones</th></tr></thead>
          <tbody>${courseRows}</tbody></table></div></section>`;
  }

  /* ---- POSTULACIONES ---- */
  if (tab === 'postulaciones') {
    body = `<p class="mb-4 text-sm text-white/60">Llegan desde el formulario de la vitrina. Al aprobar, el postulante queda inscrito con su cuenta de Google y recibe un correo.</p>
      <div class="space-y-3">${pendingApps.map(appCard).join('') || `<div class="${card} px-5 py-8 text-center text-sm text-white/55">Sin postulaciones pendientes.</div>`}</div>
      ${doneApps.length ? `<details class="mt-6"><summary class="cursor-pointer text-sm text-white/60 hover:text-white">Historial (${doneApps.length})</summary><div class="mt-3 space-y-3">${doneApps.map(appCard).join('')}</div></details>` : ''}`;
  }

  /* ---- ALUMNOS ---- */
  if (tab === 'alumnos') {
    const checks = S.catalog.map((c) => `<label class="inline-flex cursor-pointer items-center gap-1.5 text-sm"><input type="checkbox" name="new-course" value="${esc(c.id)}"> ${esc(c.title)}</label>`).join('');
    const rows = A.enrollments.map((e) => {
      const chips = S.catalog.map((c) => `<label class="cursor-pointer"><input type="checkbox" class="peer sr-only" data-act="enr-course" data-email="${esc(e.email)}" data-cid="${esc(c.id)}" ${(e.courses || []).includes(c.id) ? 'checked' : ''}>
        <span class="inline-flex items-center rounded-lg border border-white/10 px-2.5 py-1 text-xs text-white/40 transition peer-checked:border-sky-400/40 peer-checked:bg-sky-400/10 peer-checked:text-sky-200 peer-focus-visible:ring-2 peer-focus-visible:ring-sky-400/60">${esc(c.title)}</span></label>`).join('');
      const bars = (e.courses || []).map((cid) => { const c = S.catalog.find((x) => x.id === cid); if (!c) return ''; const n = progOf(e.email, cid);
        return `<div class="min-w-[9rem] flex-1"><div class="mb-1 flex justify-between text-[11px] text-white/45"><span class="truncate pr-2">${esc(c.title)}</span><span>${n}%</span></div>${bar(n)}</div>`; }).join('');
      return `<div data-row data-search="${esc(((e.name || '') + ' ' + e.email).toLowerCase())}" data-courses=",${esc((e.courses || []).join(','))}," class="${card} p-4">
        <div class="flex flex-wrap items-start justify-between gap-3">
          <div class="min-w-0"><p class="text-sm font-medium">${esc(e.name || '—')}</p><p class="text-xs text-white/50">${esc(e.email)}</p></div>
          <div class="flex items-center gap-4 text-xs">
            <label class="inline-flex cursor-pointer items-center gap-2 text-white/70"><input type="checkbox" data-act="enr-active" data-email="${esc(e.email)}" ${e.active !== false ? 'checked' : ''}> Activo</label>
            <button data-act="enr-delete" data-email="${esc(e.email)}" class="text-red-300 hover:text-red-200">Eliminar</button></div></div>
        <div class="mt-3 flex flex-wrap gap-2">${chips}</div>
        ${bars ? `<div class="mt-3 flex flex-wrap gap-x-5 gap-y-2">${bars}</div>` : ''}</div>`;
    }).join('');
    body = `<div class="mb-5 flex flex-wrap items-center gap-3">
        <input id="al-q" type="search" placeholder="Buscar por nombre o correo…" class="${input} sm:w-72" aria-label="Buscar alumno">
        <select id="al-course" class="rounded-lg border border-white/10 bg-base px-3 py-2 text-sm text-white/90" aria-label="Filtrar por curso"><option value="">Todos los cursos</option>${S.catalog.map((c) => `<option value="${esc(c.id)}">${esc(c.title)}</option>`).join('')}</select>
        <span id="al-count" class="text-sm text-white/50">${A.enrollments.length} alumno${A.enrollments.length === 1 ? '' : 's'}</span>
        <div class="ml-auto flex gap-2"><button data-act="export-csv" class="${ghost}">Exportar CSV</button></div></div>
      <details class="${card} mb-5"><summary class="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-semibold"><span>Agregar alumno manualmente</span>${ICON.chev}</summary>
        <form data-act="enr-add" class="grid items-end gap-3 border-t border-white/10 p-4 md:grid-cols-[1fr_1fr_auto]">
          <div><label class="text-xs text-white/50">Cuenta de Google</label><input name="email" type="email" required placeholder="alumno@gmail.com" class="mt-1 ${input}"></div>
          <div><label class="text-xs text-white/50">Nombre</label><input name="name" type="text" placeholder="Nombre y apellido" class="mt-1 ${input}"></div>
          <button class="${btnPrimary}" style="${gradient}">Agregar</button>
          <div class="flex flex-wrap gap-x-5 gap-y-2 md:col-span-3">${checks}</div></form></details>
      <div class="space-y-3">${rows}</div>
      <p id="al-empty" class="${A.enrollments.length ? 'hidden ' : ''}${card} px-5 py-8 text-center text-sm text-white/55">${A.enrollments.length ? 'Ningún alumno coincide con la búsqueda.' : 'Aún no hay alumnos.'}</p>`;
  }

  /* ---- CERTIFICADOS ---- */
  if (tab === 'certificados') {
    body = `<p class="mb-4 text-sm text-white/60">Los alumnos lo solicitan al completar un curso. Antes de aprobar, revisa su progreso: es un dato que registra el propio alumno.</p>
      <h2 class="mb-3 text-sm font-semibold uppercase tracking-wider text-white/50">Por aprobar${badge(pendingCerts.length, true)}</h2>
      <div class="space-y-3">${pendingCerts.map(certCard).join('') || `<div class="${card} px-5 py-6 text-center text-sm text-white/55">Sin solicitudes pendientes.</div>`}</div>
      <h2 class="mb-3 mt-8 text-sm font-semibold uppercase tracking-wider text-white/50">Emitidos${badge(issuedCerts.length, false)}</h2>
      <div class="space-y-3">${issuedCerts.map(certCard).join('') || `<div class="${card} px-5 py-6 text-center text-sm text-white/55">Aún no has emitido certificados.</div>`}</div>`;
  }

  /* ---- CONTENIDO ---- */
  if (tab === 'contenido') {
    const course = S.catalog.find((c) => c.id === A.courseId) || S.catalog[0];
    const pv = S.priv[course.id] || {};
    const ovs = pv.sessions || {};
    const mods = course.modules.map((m, mi) => {
      const covM = m.sessions.filter((x) => hasContent(course.id, x.id)).length;
      const hasQ = (pv.quizzes?.[m.id]?.questions || []).length > 0;
      const sess = m.sessions.map((x) => {
        const o = ovs[x.id] || {};
        const nRes = (o.resources || []).length;
        const has = !!(o.video || nRes);
        const resText = (o.resources || []).map((r) => `${r.label} | ${r.url}`).join('\n');
        const state = has ? [o.video ? 'Video' : '', nRes ? `${nRes} material${nRes === 1 ? '' : 'es'}` : ''].filter(Boolean).join(' · ') : 'Sin contenido';
        return `<div class="rounded-xl border border-white/10 bg-white/[0.03] p-3" data-sform="${esc(x.id)}">
          <div class="flex items-center justify-between gap-2"><p class="text-sm font-medium">${esc(x.title)} <span class="text-xs text-white/35">(${esc(x.id)})</span></p><span class="shrink-0 text-[11px] ${has ? 'text-emerald-300' : 'text-white/35'}">${esc(state)}</span></div>
          <div class="mt-3 grid gap-3 md:grid-cols-2">
            <div><label class="text-xs text-white/50">Video (URL de Drive o YouTube) — opcional</label><input type="text" data-f="video" value="${esc(o.video || '')}" placeholder="https://drive.google.com/file/d/…/view" class="mt-1 ${input}"></div>
            <div><label class="text-xs text-white/50">Materiales — Etiqueta | https://enlace (uno por línea)</label><textarea data-f="res" rows="2" placeholder="Guía (PDF) | https://drive.google.com/…" class="mt-1 ${input}">${esc(resText)}</textarea></div></div></div>`;
      }).join('');
      return `<details class="overflow-hidden rounded-xl border border-white/10 bg-white/5" ${mi === 0 ? 'open' : ''}>
        <summary class="flex cursor-pointer list-none items-center gap-3 px-4 py-3 hover:bg-white/5">
          <span class="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-white/10 text-xs font-bold text-white/70">${mi + 1}</span>
          <span class="min-w-0 flex-1"><span class="block text-sm font-semibold">${esc(m.title)}</span>
          <span class="block text-xs text-white/45">${covM} de ${m.sessions.length} sesiones con contenido · ${hasQ ? 'con evaluación' : 'sin evaluación'}</span></span>${ICON.chev}</summary>
        <div class="space-y-3 border-t border-white/10 p-4">${sess}
          <div class="rounded-xl border border-white/10 bg-white/[0.03] p-3"><p class="text-sm font-medium">Evaluación del módulo <span class="text-xs text-white/35">(${esc(m.id)})</span></p>
            <textarea data-qform="${esc(m.id)}" rows="6" placeholder="? Pregunta&#10;- Opción incorrecta&#10;+ Opción correcta&#10;= Explicación (opcional)&#10;&#10;? Otra pregunta…" class="mt-2 ${input} font-mono">${esc(quizToText(pv.quizzes?.[m.id]?.questions))}</textarea></div></div></details>`;
    }).join('');
    body = `<div class="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div><p class="text-sm text-white/60">La estructura (módulos y sesiones) sale de la vitrina. Aquí agregas el video, los materiales y las evaluaciones.</p></div>
        <select data-act="admin-course" class="rounded-lg border border-white/10 bg-base px-3 py-2 text-sm" aria-label="Curso">${S.catalog.map((c) => `<option value="${esc(c.id)}" ${c.id === course.id ? 'selected' : ''}>${esc(c.title)}</option>`).join('')}</select></div>
      <details class="${card} mb-5"><summary class="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm"><span>Formato de las evaluaciones</span>${ICON.chev}</summary>
        <div class="border-t border-white/10 px-4 py-3 text-xs leading-relaxed text-white/60"><code>?</code> pregunta · <code>-</code> opción incorrecta · <code>+</code> opción correcta (una sola) · <code>=</code> explicación opcional. Deja una línea en blanco entre preguntas. Un módulo sin texto no tiene evaluación. Las respuestas son visibles para los alumnos inscritos: sirven para autoevaluación.</div></details>
      <div class="space-y-3">${mods}</div>
      <div class="sticky bottom-3 z-10 mt-6"><div class="flex flex-wrap items-center gap-3 rounded-2xl border border-white/10 px-4 py-3 shadow-soft backdrop-blur" style="background:rgba(13,31,58,.92);">
        <label class="text-sm text-white/70">Puntaje mínimo para aprobar (%) <input data-f="pass" type="number" min="1" max="100" value="${esc(pv.pass || PASS_DEFAULT)}" class="ml-2 w-20 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-sm"></label>
        <button data-act="save-content" class="${btnPrimary}" style="${gradient}">Guardar contenido</button><span id="admin-status" class="text-sm text-white/60"></span></div></div>`;
  }

  return `${topbar('admin')}
  <h1 class="mb-5 font-display text-2xl font-bold">Administración</h1>
  ${flash}${nav}${body}`;
}

/* ---------- render y rutas ---------- */
function parseRoute() {
  const p = location.hash.replace(/^#\/?/, '').split('/');
  if (p[0] === 'c' && p[1]) return { v: 'course', cid: p[1], sid: p[2] };
  if (p[0] === 'admin') return { v: 'admin', tab: p[1] || '' };
  return { v: 'home' };
}

async function render() {
  try {
    if (!S.user) { root.innerHTML = loginView(); return; }
    const r = parseRoute();
    const noAccess = !S.isAdmin && (!S.enr || S.enr.active === false || !(S.enr.courses || []).length);
    if (noAccess) { root.innerHTML = noAccessView(); return; }
    if (r.v === 'admin') {
      if (!S.isAdmin) { location.hash = '#/'; return; }
      if (!S.admin) { root.innerHTML = `${topbar('admin')}<p class="text-white/60 text-sm">Cargando…</p>`; await loadAdmin(); }
      root.innerHTML = adminView();
    } else if (r.v === 'course') {
      if (!(r.cid in S.priv)) S.priv[r.cid] = (await S.api.getPrivate(r.cid)) || {};
      root.innerHTML = courseView(r.cid, r.sid);
      const sec = root.querySelector('[data-session]');
      if (sec) {
        const p = S.prog[r.cid] || (S.prog[r.cid] = {});
        if (p.last !== sec.dataset.session) { p.last = sec.dataset.session; S.api.saveProgress(S.user.email, r.cid, { last: p.last }).catch(() => {}); }
      }
    } else {
      root.innerHTML = homeView();
    }
    window.scrollTo({ top: 0 });
  } catch (e) {
    console.error(e);
    root.innerHTML = `${S.user ? topbar('') : ''}<div class="${card} p-6 max-w-xl"><h1 class="font-semibold">No se pudo cargar</h1><p class="mt-2 text-sm text-white/70">${esc(e.message || e)}</p></div>`;
  }
}

/* ---------- eventos ---------- */
const saveNote = debounce(async (cid, sid, text) => {
  const st = document.getElementById('note-status');
  try {
    const p = S.prog[cid] || (S.prog[cid] = {});
    (p.notes ||= {})[sid] = text;
    await S.api.saveProgress(S.user.email, cid, { notes: { [sid]: text } });
    if (st) st.textContent = 'Guardado';
  } catch (e) { if (st) st.textContent = 'No se pudo guardar'; }
}, 800);

root.addEventListener('input', (e) => {
  if (e.target.id === 'al-q') { filterStudents(); return; }
  if (e.target.id === 'note') {
    const sec = root.querySelector('[data-session]');
    const st = document.getElementById('note-status'); if (st) st.textContent = 'Guardando…';
    saveNote(sec.dataset.course, sec.dataset.session, e.target.value);
  }
});

root.addEventListener('change', async (e) => {
  const t = e.target, act = t.dataset.act;
  if (t.id === 'al-course') { filterStudents(); return; }
  try {
    if (act === 'admin-course') {
      S.admin.courseId = t.value;
      if (!(t.value in S.priv)) S.priv[t.value] = (await S.api.getPrivate(t.value)) || {};
      root.innerHTML = adminView();
    }
    if (act === 'enr-course') {
      const en = S.admin.enrollments.find((x) => x.email === t.dataset.email);
      const set = new Set(en.courses || []); t.checked ? set.add(t.dataset.cid) : set.delete(t.dataset.cid);
      en.courses = [...set]; await S.api.saveEnrollment(en.email, { courses: en.courses });
      const row = t.closest('[data-row]'); if (row) row.dataset.courses = ',' + en.courses.join(',') + ',';
    }
    if (act === 'enr-active') {
      const en = S.admin.enrollments.find((x) => x.email === t.dataset.email);
      en.active = t.checked; await S.api.saveEnrollment(en.email, { active: en.active });
    }
  } catch (err) { alert('Error: ' + (err.message || err)); }
});

root.addEventListener('submit', async (e) => {
  if (e.target.dataset.act !== 'enr-add') return;
  e.preventDefault();
  const f = e.target;
  const email = norm(f.email.value);
  const courses = [...f.querySelectorAll('[name=new-course]:checked')].map((i) => i.value);
  try {
    await S.api.saveEnrollment(email, { name: f.name.value.trim(), courses, active: true, createdAt: new Date().toISOString() });
    S.admin = null; await render();
  } catch (err) { alert('Error: ' + (err.message || err)); }
});

function gradeQuiz(sec) {
  const cid = sec.dataset.course, sid = sec.dataset.session;
  const mid = sid.replace(/^q-/, '');
  const qz = quizzesOf(cid)[mid].questions;
  const answers = qz.map((_, i) => { const r = root.querySelector(`input[name=q${i}]:checked`); return r ? Number(r.value) : -1; });
  if (answers.includes(-1)) { S.quizMsg = 'Responde todas las preguntas antes de calificar.'; return null; }
  S.quizMsg = '';
  const correct = answers.filter((a, i) => a === qz[i].answer).length;
  const score = Math.round((correct / qz.length) * 100);
  return { key: cid + '/' + sid, answers, correct, total: qz.length, score, passed: score >= passOf(cid), cid, mid };
}

root.addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-act]');
  if (!b) return;
  const act = b.dataset.act;
  try {
    if (act === 'signin') { S.error = ''; await S.api.signIn(); }
    if (act === 'signout') { await S.api.signOut(); }
    if (act === 'toggle-done') {
      const sec = root.querySelector('[data-session]');
      const cid = sec.dataset.course, sid = sec.dataset.session;
      const p = S.prog[cid] || (S.prog[cid] = {});
      const v = !p.done?.[sid];
      (p.done ||= {})[sid] = v;
      await S.api.saveProgress(S.user.email, cid, { done: { [sid]: v } });
      const note = document.getElementById('note')?.value;
      await render();
      if (note != null) { const n = document.getElementById('note'); if (n) n.value = note; }
    }
    if (act === 'quiz-submit') {
      const sec = root.querySelector('[data-session]');
      const g = gradeQuiz(sec);
      if (!g) { await render(); return; }
      const p = S.prog[g.cid] || (S.prog[g.cid] = {});
      const prev = p.quiz?.[g.mid] || {};
      const res = { score: g.score, best: Math.max(prev.best || 0, g.score), attempts: (prev.attempts || 0) + 1, passed: !!prev.passed || g.passed, at: new Date().toISOString() };
      (p.quiz ||= {})[g.mid] = res;
      await S.api.saveProgress(S.user.email, g.cid, { quiz: { [g.mid]: res } });
      S.graded = g; await render();
    }
    if (act === 'quiz-retry') { S.graded = null; S.quizMsg = ''; await render(); }
    if (act === 'cert-request') {
      const cid = b.dataset.cid; const c = S.catalog.find((x) => x.id === cid);
      if (!isComplete(c, S.prog[cid] || {}) || certFor(cid)) return;
      const data = { ownerEmail: S.user.email, name: S.user.name, courseId: cid, courseTitle: c.title, hours: c.hours, status: 'pending', requestedAt: new Date().toISOString() };
      const code = newCode();
      await S.api.requestCert(code, data);
      S.certs.push({ code, ...data }); await render();
    }
    if (act === 'export-csv') exportCsv();
    if (act === 'enr-delete') {
      if (!confirm('¿Eliminar a ' + b.dataset.email + '? Perderá el acceso.')) return;
      await S.api.deleteEnrollment(b.dataset.email); S.admin = null; await render();
    }
    if (act === 'cert-issue') {
      const ct = S.admin.certs.find((x) => x.code === b.dataset.code);
      const name = prompt('Nombre que aparecerá en el certificado:', ct.name);
      if (!name || !name.trim()) return;
      const patch = { status: 'issued', name: name.trim(), issuedAt: new Date().toISOString() };
      await S.api.updateCert(ct.code, patch); Object.assign(ct, patch); root.innerHTML = adminView();
    }
    if (act === 'app-approve') {
      const ap = S.admin.apps.find((x) => x.code === b.dataset.code);
      const en = S.admin.enrollments.find((e) => e.email === ap.ownerEmail);
      const courses = [...new Set([...(en?.courses || []), ap.courseId])];
      await S.api.saveEnrollment(ap.ownerEmail, { name: en?.name || ap.name, courses, active: true, ...(en ? {} : { createdAt: new Date().toISOString() }) });
      await S.api.updateApp(ap.code, { status: 'approved', reviewedAt: new Date().toISOString() });
      const sent = await notify({ action: 'approved', to: ap.ownerEmail, name: ap.name, course: ap.courseTitle });
      S.flash = sent === null ? `${ap.name} quedó inscrito/a. Los correos automáticos no están activados.`
        : sent ? `${ap.name} quedó inscrito/a y se le envió el correo de aprobación.`
        : `${ap.name} quedó inscrito/a, pero no se pudo enviar el correo de aprobación. Avísale tú.`;
      S.admin = null; await refreshPending(); await render();
    }
    if (act === 'app-reject') {
      await S.api.updateApp(b.dataset.code, { status: 'rejected', reviewedAt: new Date().toISOString() });
      S.admin = null; await refreshPending(); await render();
    }
    if (act === 'app-delete') {
      if (!confirm('¿Eliminar esta postulación?')) return;
      await S.api.deleteApp(b.dataset.code); S.admin = null; await refreshPending(); await render();
    }
    if (act === 'cert-delete') {
      if (!confirm('¿Eliminar este certificado? Dejará de ser verificable.')) return;
      await S.api.deleteCert(b.dataset.code); S.admin.certs = S.admin.certs.filter((x) => x.code !== b.dataset.code); root.innerHTML = adminView();
    }
    if (act === 'save-content') {
      const cid = S.admin.courseId;
      const sessions = {};
      root.querySelectorAll('[data-sform]').forEach((el) => {
        const video = el.querySelector('[data-f=video]').value.trim();
        const resources = el.querySelector('[data-f=res]').value.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => {
          const i = l.indexOf('|');
          return { label: i > -1 ? l.slice(0, i).trim() : 'Material', url: (i > -1 ? l.slice(i + 1) : l).trim() };
        }).filter((r) => /^https:\/\//i.test(r.url));
        if (video || resources.length) sessions[el.dataset.sform] = { ...(video ? { video } : {}), ...(resources.length ? { resources } : {}) };
      });
      const quizzes = {};
      for (const el of root.querySelectorAll('[data-qform]')) {
        let questions;
        try { questions = parseQuiz(el.value); } catch (err) { alert(`${el.dataset.qform.replace('m', 'Módulo ')}: ${err.message}`); return; }
        if (questions.length) quizzes[el.dataset.qform] = { questions };
      }
      const pass = Math.min(100, Math.max(1, Number(root.querySelector('[data-f=pass]').value) || PASS_DEFAULT));
      const doc = { sessions, quizzes, pass };
      await S.api.savePrivate(cid, doc);
      S.priv[cid] = doc;
      const st = document.getElementById('admin-status'); if (st) st.textContent = 'Guardado ✓';
    }
  } catch (err) {
    if (act === 'signin') { S.error = 'No se pudo iniciar sesión: ' + (err.message || err); await render(); }
    else alert('Error: ' + (err.message || err));
  }
});

// Atajos: flechas izquierda/derecha para cambiar de sesión.
window.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select, [contenteditable]')) return;
  const sec = root.querySelector('[data-session]');
  if (!sec) return;
  if (e.key === 'ArrowRight' && sec.dataset.next) location.hash = sec.dataset.next;
  if (e.key === 'ArrowLeft' && sec.dataset.prev) location.hash = sec.dataset.prev;
});

window.addEventListener('hashchange', () => { S.graded = null; S.quizMsg = ''; render(); });

/* ---------- arranque ---------- */
(async function init() {
  try {
    const [catalog, api] = await Promise.all([
      fetch(new URL('catalog.json', location.href.replace(/[#?].*$/, '')).href).then((r) => r.json()),
      DEMO ? demoApi() : firebaseApi(),
    ]);
    S.catalog = catalog.courses;
    S.api = api;
    api.onAuth(async (u) => {
      S.user = u; S.isAdmin = !!u && u.email === norm(ADMIN_EMAIL);
      S.priv = {}; S.admin = null; S.certs = []; S.myApps = []; S.pending = 0;
      if (u) { root.innerHTML = '<p class="text-white/60 text-sm">Cargando tus cursos…</p>'; await loadUserData(); }
      await render();
    });
  } catch (e) {
    console.error(e);
    root.innerHTML = `<div class="${card} p-6 max-w-xl"><h1 class="font-semibold">No se pudo iniciar el aula</h1><p class="mt-2 text-sm text-white/70">${esc(e.message || e)}</p></div>`;
  }
})();
