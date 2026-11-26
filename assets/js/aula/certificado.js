// Verificación y vista imprimible de certificados.
// Lee /certificates/{codigo} de Firestore: solo los emitidos son públicos.
import { firebaseConfig } from './firebase-config.js';

const FB = 'https://www.gstatic.com/firebasejs/10.14.1/';
const wrap = document.getElementById('cert-wrap');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (iso) => new Date(iso).toLocaleDateString('es-PE', { day: 'numeric', month: 'long', year: 'numeric' });
// Vista de prueba: solo en localhost con ?demo=1 (datos ficticios).
const DEMO = ['localhost', '127.0.0.1'].includes(location.hostname) && new URLSearchParams(location.search).has('demo');
const code = (new URLSearchParams(location.search).get('c') || '').trim().toUpperCase();

function invalid(msg, title = 'Certificado no encontrado') {
  wrap.innerHTML = `<div class="max-w-lg mx-auto rounded-2xl border border-white/10 bg-white/5 p-8 text-center">
    <h1 class="text-xl font-bold">${esc(title)}</h1>
    <p class="mt-2 text-sm text-white/70">${esc(msg)}</p></div>`;
}

(async () => {
  if (!/^NG-[A-Z2-9]{4}-[A-Z2-9]{4}-[A-Z2-9]{4}$/.test(code)) return invalid('El código no tiene un formato válido.');
  try {
    let d;
    if (DEMO) {
      d = { name: 'María Fernanda Pérez Quispe', courseTitle: 'Power BI para Minería', hours: 24, status: 'issued', issuedAt: new Date().toISOString() };
    } else {
      const [{ initializeApp }, { getFirestore, doc, getDoc }] = await Promise.all([import(FB + 'firebase-app.js'), import(FB + 'firebase-firestore.js')]);
      const db = getFirestore(initializeApp(firebaseConfig));
      const snap = await getDoc(doc(db, 'certificates', code));
      if (!snap.exists()) return invalid('No existe un certificado con ese código.');
      d = snap.data();
    }
    if (d.status !== 'issued') return invalid('Este certificado aún no ha sido emitido.');

    const url = `${location.origin}/certificado/?c=${code}`;
    wrap.innerHTML = `
      <div class="no-print mb-5 flex flex-wrap items-center justify-between gap-3">
        <p class="text-sm text-emerald-300">✓ Certificado verificado</p>
        <button id="print" class="rounded-lg border border-white/20 px-4 py-2 text-sm text-white hover:bg-white/10">Imprimir / Guardar como PDF</button>
      </div>
      <div id="cert" style="background:#fff;color:#0A192F;aspect-ratio:297/210;width:100%;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.4);padding:2.2%;font-family:Inter,Arial,sans-serif;">
        <div style="height:100%;border:3px solid #0A192F;padding:6px;"><div style="height:100%;border:1px solid #0068FF;display:flex;flex-direction:column;align-items:center;justify-content:space-between;text-align:center;padding:3% 6%;">
          <div>
            <div style="font-size:clamp(10px,1.5vw,16px);letter-spacing:.35em;color:#0068FF;font-weight:700;">NILSON GARRIDO · CURSOS</div>
            <div style="margin-top:1.2%;font-family:Montserrat,Inter,Arial,sans-serif;font-size:clamp(24px,5vw,56px);font-weight:800;letter-spacing:.08em;">CERTIFICADO</div>
            <div style="font-size:clamp(10px,1.4vw,15px);color:#475569;margin-top:.5%;">de finalización</div>
          </div>
          <div>
            <div style="font-size:clamp(10px,1.5vw,16px);color:#475569;">Se otorga a</div>
            <div style="margin:1.2% 0;font-family:Montserrat,Inter,Arial,sans-serif;font-size:clamp(18px,3.2vw,36px);font-weight:700;border-bottom:2px solid #0068FF;display:inline-block;padding:0 4%;">${esc(d.name)}</div>
            <div style="font-size:clamp(10px,1.5vw,16px);color:#475569;">por completar satisfactoriamente el curso</div>
            <div style="margin-top:1%;font-family:Montserrat,Inter,Arial,sans-serif;font-size:clamp(16px,3vw,32px);font-weight:700;color:#0A192F;">${esc(d.courseTitle)}</div>
            <div style="margin-top:1%;font-size:clamp(10px,1.4vw,15px);color:#475569;">${esc(d.hours)} ${Number(d.hours) === 1 ? 'hora académica' : 'horas académicas'} · Modalidad asincrónica</div>
          </div>
          <div style="width:100%;display:flex;align-items:flex-end;justify-content:space-between;gap:3%;">
            <div style="text-align:left;font-size:clamp(8px,1.1vw,12px);color:#475569;line-height:1.5;">
              Emitido el ${esc(fmt(d.issuedAt))}<br>Código: <strong style="color:#0A192F;">${esc(code)}</strong><br>Verificar en: ${esc(url)}</div>
            <div style="text-align:center;min-width:30%;">
              <div style="border-top:1px solid #0A192F;padding-top:4px;font-size:clamp(10px,1.4vw,15px);font-weight:700;">Nilson Rolando Garrido Asenjo</div>
              <div style="font-size:clamp(8px,1.1vw,12px);color:#475569;">Ingeniero de Minas · Instructor</div></div>
          </div>
        </div></div>
      </div>`;
    document.getElementById('print').addEventListener('click', () => window.print());
  } catch (e) {
    console.error(e);
    invalid('Intenta nuevamente en unos minutos.', 'No se pudo verificar el certificado');
  }
})();
