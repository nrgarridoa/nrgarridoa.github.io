// Certificado: vista, verificación pública, descarga en PDF, QR y botón para LinkedIn.
// Lee /certificates/{codigo} de Firestore: solo los emitidos son públicos.
import { firebaseConfig } from './firebase-config.js';
import qrcode from '../../vendor/qrcode.mjs';

const FB = 'https://www.gstatic.com/firebasejs/10.14.1/';
const wrap = document.getElementById('cert-wrap');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (iso) => new Date(iso).toLocaleDateString('es-PE', { day: 'numeric', month: 'long', year: 'numeric' });
// Vista de prueba: solo en localhost con ?demo=1 (datos ficticios).
const DEMO = ['localhost', '127.0.0.1'].includes(location.hostname) && new URLSearchParams(location.search).has('demo');
const code = (new URLSearchParams(location.search).get('c') || '').trim().toUpperCase();
const W = 1123, H = 794;   // A4 horizontal a 96 ppp: el certificado se dibuja a este tamaño y se escala a la pantalla

const ICON = {
  check: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 13l4 4L19 7"/></svg>',
  down: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m0 0l-4-4m4 4l4-4M5 20h14"/></svg>',
  link: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1"/></svg>',
  li: '<svg class="h-4 w-4" viewBox="0 0 24 24" fill="currentColor"><path d="M4.98 3.5a2.5 2.5 0 110 5 2.5 2.5 0 010-5zM3 9.75h4V21H3zM9.75 9.75h3.8v1.54h.05c.53-1 1.82-2.05 3.75-2.05 4 0 4.75 2.63 4.75 6.05V21h-4v-4.9c0-1.17-.02-2.67-1.63-2.67-1.63 0-1.88 1.27-1.88 2.58V21h-4z"/></svg>',
};

function invalid(msg, title = 'Certificado no encontrado') {
  wrap.innerHTML = `<div class="max-w-lg mx-auto rounded-2xl border border-white/10 bg-white/5 p-8 text-center">
    <h1 class="text-xl font-bold">${esc(title)}</h1>
    <p class="mt-2 text-sm text-white/70">${esc(msg)}</p></div>`;
}

// Curvas de nivel (mapa topográfico) como fondo discreto del certificado.
function contours(cx, cy, rs, seed) {
  return rs.map((r, k) => {
    const pts = [];
    for (let i = 0; i <= 72; i++) {
      const t = (i / 72) * Math.PI * 2;
      const rr = r * (1 + 0.1 * Math.sin(3 * t + seed + k * 0.25) + 0.05 * Math.sin(5 * t + seed * 2) + 0.03 * Math.sin(8 * t + k));
      pts.push(`${(cx + rr * Math.cos(t) * 1.25).toFixed(1)},${(cy + rr * Math.sin(t) * 0.85).toFixed(1)}`);
    }
    return `<polyline points="${pts.join(' ')}"/>`;
  }).join('');
}

function qrSvg(url) {
  const q = qrcode(0, 'M'); q.addData(url); q.make();
  return q.createSvgTag({ cellSize: 4, margin: 0, scalable: true }).replace(/fill="#000000"/g, 'fill="#0A192F"');
}

function certHtml(d, url) {
  const grad = 'linear-gradient(135deg,#0068FF,#00D2FF)';
  const hours = `${esc(d.hours)} ${Number(d.hours) === 1 ? 'hora académica' : 'horas académicas'}`;
  return `<div id="cert" style="position:absolute;left:0;top:0;width:${W}px;height:${H}px;background:#fff;color:#0A192F;font-family:Inter,Arial,sans-serif;overflow:hidden;transform-origin:0 0;">
    <svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="position:absolute;inset:0" fill="none" stroke="#0068FF" stroke-opacity=".09" stroke-width="1.2">
      ${contours(980, 690, [60, 105, 150, 195, 240, 285, 330, 375], 1.1)}${contours(120, 90, [50, 90, 130, 170, 210], 2.3)}</svg>
    <div style="position:absolute;inset:26px;border:1px solid #CBD5E1;"></div>
    <div style="position:absolute;left:26px;right:26px;top:26px;height:8px;background:${grad};"></div>
    <div style="position:absolute;inset:26px;display:flex;flex-direction:column;padding:46px 70px 40px;">
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="display:flex;align-items:center;gap:14px;">
          <div style="width:46px;height:46px;border-radius:11px;background:${grad};display:flex;align-items:center;justify-content:center;color:#fff;font:800 17px Montserrat,Inter,Arial,sans-serif;">NG</div>
          <div><div style="font:700 17px Montserrat,Inter,Arial,sans-serif;">Nilson Garrido</div><div style="font-size:12px;color:#64748B;margin-top:2px;">Mining Engineer · Data Analyst</div></div></div>
        <div style="text-align:right;font-size:12px;color:#64748B;">Certificado N.º<br><strong style="font-size:14px;letter-spacing:.04em;color:#0A192F;">${esc(code)}</strong></div></div>

      <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
        <div style="font:700 14px Inter,Arial,sans-serif;letter-spacing:.42em;color:#0068FF;">CERTIFICADO DE FINALIZACIÓN</div>
        <div style="margin-top:34px;font-size:16px;color:#64748B;">Se otorga a</div>
        <div style="margin-top:12px;font:800 50px/1.1 Montserrat,Inter,Arial,sans-serif;letter-spacing:-.01em;">${esc(d.name)}</div>
        <div style="margin:18px 0 20px;width:240px;height:4px;border-radius:2px;background:${grad};"></div>
        <div style="font-size:16px;color:#64748B;">por completar satisfactoriamente el curso</div>
        <div style="margin-top:12px;font:700 32px/1.2 Montserrat,Inter,Arial,sans-serif;max-width:860px;">${esc(d.courseTitle)}</div>
        <div style="margin-top:16px;font-size:15px;color:#475569;">${hours} · Modalidad asincrónica · Emitido el ${esc(fmt(d.issuedAt))}</div></div>

      <div style="display:flex;align-items:flex-end;justify-content:space-between;">
        <div style="width:300px;"><div style="border-top:1.5px solid #0A192F;padding-top:8px;font:700 15px Inter,Arial,sans-serif;">Nilson Rolando Garrido Asenjo</div>
          <div style="font-size:12px;color:#64748B;margin-top:2px;">Ingeniero de Minas · Instructor</div></div>
        <div style="width:96px;height:96px;border-radius:50%;background:${grad};padding:5px;box-shadow:0 0 0 4px #fff,0 0 0 5px #CBD5E1;">
          <div style="width:100%;height:100%;border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="font:800 22px Montserrat,Inter,Arial,sans-serif;color:#0068FF;line-height:1;">NG</div>
            <div style="font-size:8px;letter-spacing:.2em;color:#64748B;margin-top:4px;">CURSOS</div></div></div>
        <div style="width:300px;display:flex;align-items:center;justify-content:flex-end;gap:14px;">
          <div style="text-align:right;font-size:11px;color:#64748B;line-height:1.5;">Verifica la autenticidad<br>escaneando el código o en<br><strong style="color:#0A192F;">${esc(location.host)}/certificado</strong></div>
          <div style="width:84px;height:84px;">${qrSvg(url)}</div></div></div>
    </div></div>`;
}

async function loadCourse(courseId) {
  try {
    const r = await fetch(new URL('../aula/catalog.json', location.href).href);
    return ((await r.json()).courses || []).find((c) => c.id === courseId) || null;
  } catch { return null; }
}

(async () => {
  if (!/^NG-[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$/.test(code)) return invalid('El código no tiene un formato válido.');
  try {
    let d;
    if (DEMO) {
      d = { name: 'María Fernanda Pérez Quispe', courseId: 'python-geologia', courseTitle: 'Python para Geología', hours: 70, status: 'issued', issuedAt: new Date().toISOString() };
    } else {
      const [{ initializeApp }, { getFirestore, doc, getDoc }] = await Promise.all([import(FB + 'firebase-app.js'), import(FB + 'firebase-firestore.js')]);
      const db = getFirestore(initializeApp(firebaseConfig));
      const snap = await getDoc(doc(db, 'certificates', code));
      if (!snap.exists()) return invalid('No existe un certificado con ese código.');
      d = snap.data();
    }
    if (d.status !== 'issued') return invalid('Este certificado aún no ha sido emitido.');

    const url = `${location.origin}/certificado/?c=${code}`;
    const d0 = new Date(d.issuedAt);
    const li = 'https://www.linkedin.com/profile/add?' + new URLSearchParams({
      startTask: 'CERTIFICATION_NAME', name: d.courseTitle, organizationName: 'Nilson Garrido',
      issueYear: String(d0.getFullYear()), issueMonth: String(d0.getMonth() + 1), certUrl: url, certId: code,
    }).toString();
    const course = await loadCourse(d.courseId);
    const btn = 'inline-flex items-center gap-2 rounded-lg border border-white/20 px-4 py-2 text-sm font-medium text-white transition hover:bg-white/10';

    wrap.innerHTML = `
      <div class="no-print mb-6 flex flex-wrap items-center justify-between gap-3">
        <div class="inline-flex items-center gap-2 rounded-lg border border-emerald-400/40 bg-emerald-500/10 px-3.5 py-2 text-sm font-medium text-emerald-300">${ICON.check} Certificado verificado · emitido por Nilson Garrido</div>
        <div class="flex flex-wrap gap-2">
          <button id="pdf" class="${btn}" style="background:linear-gradient(135deg,#0068FF,#00D2FF);border-color:transparent;">${ICON.down} Descargar PDF</button>
          <a href="${esc(li)}" target="_blank" rel="noopener noreferrer" class="${btn}">${ICON.li} Añadir a LinkedIn</a>
          <button id="copy" class="${btn}">${ICON.link} <span id="copy-l">Copiar enlace</span></button></div></div>
      <div id="stage" style="position:relative;width:100%;height:${H}px;border-radius:12px;overflow:hidden;box-shadow:0 18px 60px rgba(0,0,0,.45);">${certHtml(d, url)}</div>
      <p class="no-print mt-3 text-xs text-white/45">Para guardar el PDF, elige “Guardar como PDF” como destino en la ventana de impresión.</p>
      <div class="no-print mt-8 grid gap-6 md:grid-cols-2">
        <div class="rounded-2xl border border-white/10 bg-white/5 p-5"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Qué acredita</h2>
          <dl class="mt-3 space-y-2 text-sm"><div class="flex justify-between gap-4"><dt class="text-white/50">Titular</dt><dd class="text-right">${esc(d.name)}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-white/50">Curso</dt><dd class="text-right">${esc(d.courseTitle)}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-white/50">Duración</dt><dd>${esc(d.hours)} h</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-white/50">Emitido</dt><dd>${esc(fmt(d.issuedAt))}</dd></div>
            <div class="flex justify-between gap-4"><dt class="text-white/50">Código</dt><dd class="font-mono text-xs">${esc(code)}</dd></div></dl></div>
        ${course ? `<div class="rounded-2xl border border-white/10 bg-white/5 p-5"><h2 class="text-sm font-semibold uppercase tracking-wider text-white/50">Contenido del curso</h2>
          <ul class="mt-3 space-y-1.5 text-sm text-white/80">${course.modules.map((m) => `<li class="flex gap-2"><span class="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-sky-400"></span><span>${esc(m.title)}</span></li>`).join('')}</ul></div>` : ''}
      </div>`;

    const stage = document.getElementById('stage'), cert = document.getElementById('cert');
    const fit = () => { const k = Math.min(1, stage.parentElement.clientWidth / W); cert.style.transform = `scale(${k})`; stage.style.height = `${H * k}px`; };
    fit(); window.addEventListener('resize', fit);
    window.addEventListener('beforeprint', () => { cert.style.transform = 'none'; });
    window.addEventListener('afterprint', fit);
    document.getElementById('pdf').addEventListener('click', () => window.print());
    document.getElementById('copy').addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(url); document.getElementById('copy-l').textContent = 'Enlace copiado'; setTimeout(() => { const l = document.getElementById('copy-l'); if (l) l.textContent = 'Copiar enlace'; }, 2000); } catch { prompt('Copia este enlace:', url); }
    });
  } catch (e) {
    console.error(e);
    invalid('Intenta nuevamente en unos minutos.', 'No se pudo verificar el certificado');
  }
})();
