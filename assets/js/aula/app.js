// Aula virtual — Nilson Garrido
// SPA sin build: Firebase (Auth con Google + Firestore) cargado como módulo ES.
// Estructura pública de los cursos: /aula/catalog.json (generada por Jekyll).
// Contenido privado (videos, materiales, evaluaciones), alumnos, progreso y
// certificados: Firestore (ver /firestore.rules).
import { firebaseConfig, ADMIN_EMAIL, CONTACT_EMAIL } from './firebase-config.js';

const FB = 'https://www.gstatic.com/firebasejs/10.14.1/';
const root = document.getElementById('aula-root');
const DEMO = ['localhost', '127.0.0.1'].includes(location.hostname) && new URLSearchParams(location.search).has('demo');
const PASS_DEFAULT = 70;

/* ---------- utilidades ---------- */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const safeUrl = (u) => (/^https:\/\//i.test(u || '') ? u : '#');
const norm = (e) => String(e || '').trim().toLowerCase();
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const fmtDate = (iso) => { try { return new Date(iso).toLocaleDateString('es-PE', { day: 'numeric', month: 'long', year: 'numeric' }); } catch { return ''; } };
const ICON = {
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
  return {
    onAuth: (cb) => setTimeout(() => cb(user), 50),
    signIn: async () => {}, signOut: async () => location.reload(),
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
  };
}

/* ---------- estado y modelo de progreso ---------- */
const S = { api: null, user: null, isAdmin: false, enr: null, catalog: [], priv: {}, prog: {}, certs: [], admin: null, error: '', graded: null, quizMsg: '' };
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
}

/* ---------- vistas ---------- */
const btnPrimary = 'inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold text-white';
const gradient = 'background:linear-gradient(135deg, var(--accent), var(--accent2)); box-shadow:0 2px 8px rgba(0,194,255,0.25);';
const card = 'rounded-2xl border border-white/10 bg-white/5';
const bar = (n) => `<div class="h-1.5 rounded-full bg-white/10 overflow-hidden"><div class="h-full rounded-full" style="width:${n}%; background:linear-gradient(90deg, var(--accent), var(--accent2));"></div></div>`;

function topbar(active) {
  const link = (href, label, key) => `<a href="${href}" class="px-3 py-1.5 rounded-lg text-sm ${active === key ? 'bg-white/10 text-white' : 'text-white/70 hover:text-white hover:bg-white/5'}">${label}</a>`;
  return `<div class="flex flex-wrap items-center justify-between gap-3 mb-8 pb-4 border-b border-white/10">
    <nav class="flex items-center gap-1">
      ${link('#/', 'Mis cursos', 'home')}
      ${S.isAdmin ? link('#/admin', 'Administración', 'admin') : ''}
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
    <p class="mt-2 text-sm text-white/70">Ingresaste como <strong>${esc(S.user.email)}</strong>. Si crees que debería tener acceso, escribe a
      <a class="underline" href="mailto:${esc(CONTACT_EMAIL)}?subject=${encodeURIComponent('Acceso al aula virtual')}">${esc(CONTACT_EMAIL)}</a> indicando esta cuenta y el curso.</p>
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

function homeView() {
  const cs = myCourses();
  const cards = cs.map((c) => {
    const p = S.prog[c.id] || {};
    const seq = trackable(c);
    const target = p.last || (seq[0] || seqOf(c)[0])?.id || '';
    const n = pct(c, p);
    return `<article class="${card} overflow-hidden flex flex-col">
      <img src="${esc(c.cover)}" alt="" class="w-full h-40 object-cover" loading="lazy">
      <div class="p-4 flex flex-col flex-1">
        <h3 class="font-semibold">${esc(c.title)}</h3>
        <p class="mt-1 text-sm text-white/70 line-clamp-2">${esc(c.subtitle)}</p>
        <div class="mt-4">
          <div class="flex justify-between text-xs text-white/60 mb-1"><span>${doneCount(c, p)} de ${seq.length} actividades</span><span>${n}%</span></div>
          ${bar(n)}
        </div>
        <a href="#/c/${esc(c.id)}/${esc(target)}" class="${btnPrimary} mt-4" style="${gradient}">${n === 0 ? 'Comenzar' : n === 100 ? 'Repasar' : 'Continuar'}</a>
        ${certBlock(c, p)}
      </div>
    </article>`;
  }).join('');
  return `${topbar('home')}
    <header class="mb-6"><h1 class="text-2xl font-display font-bold">Mis cursos</h1>
    <p class="text-sm text-white/60 mt-1">Hola, ${esc((S.user.name || '').split(' ')[0])}. Retoma donde lo dejaste.</p></header>
    <div class="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">${cards}</div>`;
}

function sidebarHtml(c, cur, p) {
  const seq = seqOf(c);
  const mods = c.modules.map((m) => {
    const items = seq.filter((s) => s.mid === m.id);
    const hasCur = items.some((s) => s.id === cur.id);
    const lis = items.map((s) => {
      const done = isDone(p, s);
      const soon = s.status === 'soon';
      const isCur = s.id === cur.id;
      const q = s.type === 'quiz';
      return `<li><a href="#/c/${esc(c.id)}/${esc(s.id)}" class="flex items-start gap-3 rounded-lg px-3 py-2 text-sm ${isCur ? 'bg-white/10 text-white' : 'text-white/70 hover:bg-white/5 hover:text-white'}">
        <span class="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${done ? 'bg-emerald-500 border-emerald-500 text-white' : 'border-white/30'}">${done ? ICON.check : ''}</span>
        <span class="min-w-0"><span class="block ${q ? 'text-sky-200' : ''}">${q ? ICON.quiz.replace('class="', 'class="inline mr-1 -mt-0.5 ') : ''}${esc(s.title)}</span><span class="block text-xs text-white/40">${esc(s.duration || '')}${soon ? ' · Próximamente' : ''}</span></span></a></li>`;
    }).join('');
    const track = items.filter((s) => s.status !== 'soon');
    const md = track.filter((s) => isDone(p, s)).length;
    return `<details class="group rounded-xl border border-white/10 bg-white/5" ${hasCur ? 'open' : ''}>
      <summary class="cursor-pointer list-none px-4 py-3"><span class="block text-xs text-white/50">${esc(m.id.replace('m', 'Módulo '))} · ${md}/${track.length}</span><span class="block font-semibold text-sm">${esc(m.title)}</span></summary>
      <ul class="px-2 pb-2 space-y-1">${lis}</ul></details>`;
  }).join('');
  const n = pct(c, p);
  return `<div class="${card} p-4 mb-3"><div class="flex justify-between text-xs text-white/60 mb-1"><span>Tu progreso</span><span>${n}%</span></div>${bar(n)}</div>
    <div class="space-y-2">${mods}</div>`;
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

function courseView(cid, sid) {
  const c = S.catalog.find((x) => x.id === cid);
  if (!c || !myCourses().includes(c)) return `${topbar('home')}<p class="text-white/70">No tienes acceso a este curso. <a class="underline" href="#/">Volver</a></p>`;
  const all = seqOf(c);
  const p = S.prog[cid] || (S.prog[cid] = {});
  const cur = all.find((s) => s.id === sid) || all.find((s) => s.id === p.last) || trackable(c)[0] || all[0];
  const idx = all.indexOf(cur);
  const prev = all[idx - 1], next = all[idx + 1];
  const nav = `<div class="flex gap-2">
      ${prev ? `<a href="#/c/${esc(cid)}/${esc(prev.id)}" class="rounded-lg border border-white/15 px-4 py-2 text-sm text-white/80 hover:text-white hover:border-white/30">← Anterior</a>` : ''}
      ${next ? `<a href="#/c/${esc(cid)}/${esc(next.id)}" class="rounded-lg border border-white/15 px-4 py-2 text-sm text-white/80 hover:text-white hover:border-white/30">Siguiente →</a>` : ''}</div>`;

  let main;
  if (cur.type === 'quiz') {
    main = `${quizMain(c, cur, p)}<div class="mt-8">${nav}</div>`;
  } else {
    const ov = S.priv[cid]?.sessions?.[cur.id] || {};
    const video = parseVideo(ov.video);
    const res = (ov.resources || []).filter((r) => r && r.url);
    const soon = cur.status === 'soon';
    const done = !!p.done?.[cur.id];
    const videoHtml = video ? `<div class="mt-5 aspect-video overflow-hidden rounded-2xl border border-white/10 bg-black">
        <iframe src="${esc(video.src)}" title="${esc(cur.title)}" class="h-full w-full" allow="autoplay; encrypted-media; fullscreen" allowfullscreen loading="lazy" referrerpolicy="strict-origin-when-cross-origin"></iframe></div>` : '';
    const resHtml = res.length ? `<ul class="mt-3 space-y-2">${res.map((r) => `<li><a href="${esc(safeUrl(r.url))}" target="_blank" rel="noopener noreferrer"
        class="flex items-center gap-3 ${card} px-4 py-3 text-sm hover:bg-white/10"><span class="text-sky-300">${ICON.file}</span><span class="flex-1">${esc(r.label || 'Material')}</span><span class="text-xs text-white/50">Abrir</span></a></li>`).join('')}</ul>`
      : `<p class="mt-3 text-sm text-white/50">${soon ? 'Esta sesión se publicará próximamente.' : 'Aún no hay materiales publicados para esta sesión.'}</p>`;
    main = `<p class="text-xs text-white/50">${esc(cur.mtitle)}${cur.duration ? ' · ' + esc(cur.duration) : ''}</p>
      <h1 class="mt-1 text-2xl font-display font-bold">${esc(cur.title)}</h1>
      ${cur.summary ? `<p class="mt-2 text-white/70 text-sm">${esc(cur.summary)}</p>` : ''}
      ${videoHtml}
      <h2 class="mt-6 text-sm font-semibold uppercase tracking-wider text-white/50">Materiales</h2>
      ${resHtml}
      <h2 class="mt-8 text-sm font-semibold uppercase tracking-wider text-white/50">Mis notas <span id="note-status" class="ml-2 normal-case tracking-normal font-normal text-white/40"></span></h2>
      <textarea id="note" rows="5" placeholder="Escribe tus apuntes de esta sesión…" class="mt-3 w-full rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-white placeholder-white/30 focus:outline-none focus:border-white/30">${esc(p.notes?.[cur.id] || '')}</textarea>
      <div class="mt-6 flex flex-wrap items-center justify-between gap-3">${nav}
        ${soon ? '' : `<button data-act="toggle-done" class="${btnPrimary} ${done ? 'border border-emerald-400/50 bg-emerald-500/20 text-emerald-200' : ''}" ${done ? '' : `style="${gradient}"`}>${done ? '✓ Completada (clic para desmarcar)' : 'Marcar como completada'}</button>`}
      </div>`;
  }

  const certBanner = isComplete(c, p) ? `<div class="${card} p-4 mb-4 flex flex-wrap items-center justify-between gap-3"><p class="text-sm text-emerald-200">Completaste el curso.</p><div class="w-full sm:w-64">${certBlock(c, p).replace('mt-3 ', '')}</div></div>` : '';

  return `${topbar('home')}
  <nav class="text-sm text-white/50 mb-4"><a href="#/" class="hover:text-white/80">Mis cursos</a> <span class="mx-2">/</span> <span class="text-white/70">${esc(c.title)}</span></nav>
  ${certBanner}
  <div class="grid gap-6 lg:grid-cols-[340px_1fr] items-start">
    <aside class="lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto order-2 lg:order-1">${sidebarHtml(c, cur, p)}</aside>
    <section class="min-w-0 order-1 lg:order-2" data-session="${esc(cur.id)}" data-course="${esc(cid)}" data-type="${cur.type}">${main}</section>
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
  S.admin = { enrollments, progress, certs, courseId: S.admin?.courseId || S.catalog[0]?.id };
}

function adminView() {
  const A = S.admin;
  const rows = A.enrollments.map((e) => {
    const boxes = S.catalog.map((c) => `<label class="inline-flex items-center gap-1.5 text-xs text-white/80 mr-3"><input type="checkbox" data-act="enr-course" data-email="${esc(e.email)}" data-cid="${esc(c.id)}" ${(e.courses || []).includes(c.id) ? 'checked' : ''}> ${esc(c.title)}</label>`).join('');
    const prog = (e.courses || []).map((cid) => { const c = S.catalog.find((x) => x.id === cid); return c ? `${esc(c.title)}: ${pct(c, A.progress[e.email]?.[cid])}%` : ''; }).filter(Boolean).join(' · ');
    return `<tr class="border-t border-white/10 align-top">
      <td class="py-3 pr-3"><div class="font-medium text-sm">${esc(e.name || '—')}</div><div class="text-xs text-white/50">${esc(e.email)}</div></td>
      <td class="py-3 pr-3">${boxes}<div class="mt-1 text-xs text-white/40">${prog || 'Sin cursos'}</div></td>
      <td class="py-3 pr-3 whitespace-nowrap"><label class="inline-flex items-center gap-1.5 text-xs"><input type="checkbox" data-act="enr-active" data-email="${esc(e.email)}" ${e.active !== false ? 'checked' : ''}> Activo</label></td>
      <td class="py-3 text-right"><button data-act="enr-delete" data-email="${esc(e.email)}" class="text-xs text-red-300 hover:text-red-200">Eliminar</button></td></tr>`;
  }).join('');
  const checks = S.catalog.map((c) => `<label class="inline-flex items-center gap-1.5 text-sm mr-4"><input type="checkbox" name="new-course" value="${esc(c.id)}"> ${esc(c.title)}</label>`).join('');

  const certRows = A.certs.map((ct) => {
    const c = S.catalog.find((x) => x.id === ct.courseId);
    const pr = A.progress[ct.ownerEmail]?.[ct.courseId];
    const info = c && pr ? `Progreso ${pct(c, pr)}%` : '';
    const issued = ct.status === 'issued';
    return `<tr class="border-t border-white/10 align-top">
      <td class="py-3 pr-3"><div class="text-sm font-medium">${esc(ct.name)}</div><div class="text-xs text-white/50">${esc(ct.ownerEmail)}</div></td>
      <td class="py-3 pr-3 text-sm">${esc(ct.courseTitle)}<div class="text-xs text-white/40">${info}</div></td>
      <td class="py-3 pr-3 text-xs">${issued ? `<span class="text-emerald-300">Emitido</span><div class="text-white/40">${esc(fmtDate(ct.issuedAt))}</div>` : '<span class="text-amber-300">Pendiente</span>'}<div class="text-white/30">${esc(ct.code)}</div></td>
      <td class="py-3 text-right whitespace-nowrap text-xs">
        ${issued ? `<a class="text-sky-300 hover:text-sky-200 mr-3" target="_blank" rel="noopener" href="/certificado/?c=${encodeURIComponent(ct.code)}">Ver</a>` : `<button data-act="cert-issue" data-code="${esc(ct.code)}" class="text-emerald-300 hover:text-emerald-200 mr-3">Aprobar y emitir</button>`}
        <button data-act="cert-delete" data-code="${esc(ct.code)}" class="text-red-300 hover:text-red-200">${issued ? 'Revocar' : 'Rechazar'}</button></td></tr>`;
  }).join('');

  const course = S.catalog.find((c) => c.id === A.courseId);
  const pv = S.priv[A.courseId] || {};
  const ovs = pv.sessions || {};
  const sessionForms = course ? course.modules.flatMap((m) => m.sessions.map((s) => {
    const o = ovs[s.id] || {};
    const resText = (o.resources || []).map((r) => `${r.label} | ${r.url}`).join('\n');
    return `<div class="${card} p-4" data-sform="${esc(s.id)}">
      <div class="text-sm font-medium">${esc(s.title)} <span class="text-xs text-white/40">(${esc(s.id)})</span></div>
      <label class="block mt-3 text-xs text-white/50">Video (URL de Drive o YouTube) — opcional</label>
      <input type="text" data-f="video" value="${esc(o.video || '')}" placeholder="https://drive.google.com/file/d/…/view" class="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm">
      <label class="block mt-3 text-xs text-white/50">Materiales — una línea por archivo: Etiqueta | https://enlace</label>
      <textarea data-f="res" rows="3" placeholder="Guía (PDF) | https://drive.google.com/…" class="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm">${esc(resText)}</textarea></div>`;
  })).join('') : '';
  const quizForms = course ? course.modules.map((m) => `<div class="${card} p-4">
      <div class="text-sm font-medium">${esc(m.title)} <span class="text-xs text-white/40">(${esc(m.id)})</span></div>
      <textarea data-qform="${esc(m.id)}" rows="6" placeholder="? Pregunta&#10;- Opción incorrecta&#10;+ Opción correcta&#10;= Explicación (opcional)&#10;&#10;? Otra pregunta…" class="mt-2 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm font-mono">${esc(quizToText(pv.quizzes?.[m.id]?.questions))}</textarea></div>`).join('') : '';

  return `${topbar('admin')}
  <h1 class="text-2xl font-display font-bold mb-6">Administración</h1>
  <section class="${card} p-5 mb-8">
    <h2 class="font-semibold">Alumnos autorizados</h2>
    <form data-act="enr-add" class="mt-4 grid gap-3 md:grid-cols-[1fr_1fr_auto] items-end">
      <div><label class="text-xs text-white/50">Gmail</label><input name="email" type="email" required placeholder="alumno@gmail.com" class="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm"></div>
      <div><label class="text-xs text-white/50">Nombre</label><input name="name" type="text" placeholder="Nombre y apellido" class="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm"></div>
      <button class="${btnPrimary}" style="${gradient}">Agregar</button>
      <div class="md:col-span-3">${checks}</div>
    </form>
    <div class="mt-5 overflow-x-auto"><table class="w-full text-left">${rows ? `<tbody>${rows}</tbody>` : ''}</table>${rows ? '' : '<p class="text-sm text-white/50">Aún no hay alumnos.</p>'}</div>
  </section>
  <section class="${card} p-5 mb-8">
    <h2 class="font-semibold">Certificados ${A.certs.some((x) => x.status !== 'issued') ? '<span class="ml-2 rounded-md bg-amber-400/90 text-slate-900 text-[11px] font-semibold px-2 py-0.5">Hay pendientes</span>' : ''}</h2>
    <p class="mt-1 text-xs text-white/50">Los alumnos lo solicitan al completar un curso. Antes de aprobar, revisa su progreso: es un dato que el propio alumno registra.</p>
    <div class="mt-4 overflow-x-auto">${certRows ? `<table class="w-full text-left"><tbody>${certRows}</tbody></table>` : '<p class="text-sm text-white/50">Sin solicitudes.</p>'}</div>
  </section>
  <section class="${card} p-5">
    <div class="flex flex-wrap items-center justify-between gap-3"><h2 class="font-semibold">Contenido privado por curso</h2>
      <select data-act="admin-course" class="rounded-lg border border-white/10 bg-base px-3 py-2 text-sm">${S.catalog.map((c) => `<option value="${esc(c.id)}" ${c.id === A.courseId ? 'selected' : ''}>${esc(c.title)}</option>`).join('')}</select></div>
    <p class="mt-2 text-xs text-white/50">La estructura (módulos y sesiones) sale de la vitrina. Aquí agregas el video y los materiales de cada sesión, y las evaluaciones de cada módulo.</p>
    <h3 class="mt-5 text-sm font-semibold uppercase tracking-wider text-white/50">Sesiones</h3>
    <div class="mt-3 space-y-3">${sessionForms}</div>
    <h3 class="mt-8 text-sm font-semibold uppercase tracking-wider text-white/50">Evaluaciones por módulo</h3>
    <p class="mt-2 text-xs text-white/50">Formato: <code>?</code> pregunta, <code>-</code> opción incorrecta, <code>+</code> opción correcta (una sola), <code>=</code> explicación opcional. Deja una línea en blanco entre preguntas. Módulo sin texto = sin evaluación. Las respuestas son visibles para los alumnos inscritos: sirven para autoevaluación.</p>
    <div class="mt-3 space-y-3">${quizForms}</div>
    <div class="mt-5 flex flex-wrap items-center gap-3">
      <label class="text-sm text-white/70">Puntaje mínimo para aprobar (%) <input data-f="pass" type="number" min="1" max="100" value="${esc(pv.pass || PASS_DEFAULT)}" class="ml-2 w-20 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-sm"></label>
      <button data-act="save-content" class="${btnPrimary}" style="${gradient}">Guardar contenido</button><span id="admin-status" class="text-sm text-white/60"></span></div>
  </section>`;
}

/* ---------- render y rutas ---------- */
function parseRoute() {
  const p = location.hash.replace(/^#\/?/, '').split('/');
  if (p[0] === 'c' && p[1]) return { v: 'course', cid: p[1], sid: p[2] };
  if (p[0] === 'admin') return { v: 'admin' };
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
      root.innerHTML = `${topbar('admin')}<p class="text-white/60 text-sm">Cargando…</p>`;
      if (!S.admin) await loadAdmin();
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
  if (e.target.id === 'note') {
    const sec = root.querySelector('[data-session]');
    const st = document.getElementById('note-status'); if (st) st.textContent = 'Guardando…';
    saveNote(sec.dataset.course, sec.dataset.session, e.target.value);
  }
});

root.addEventListener('change', async (e) => {
  const t = e.target, act = t.dataset.act;
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
      S.priv = {}; S.admin = null; S.certs = [];
      if (u) { root.innerHTML = '<p class="text-white/60 text-sm">Cargando tus cursos…</p>'; await loadUserData(); }
      await render();
    });
  } catch (e) {
    console.error(e);
    root.innerHTML = `<div class="${card} p-6 max-w-xl"><h1 class="font-semibold">No se pudo iniciar el aula</h1><p class="mt-2 text-sm text-white/70">${esc(e.message || e)}</p></div>`;
  }
})();
