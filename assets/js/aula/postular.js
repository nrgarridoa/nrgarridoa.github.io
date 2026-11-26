// Postulación a cursos desde la vitrina.
// La persona inicia sesión con Google y la postulación se guarda en Firestore
// (colección "applications"), donde se revisa desde el panel de administración del aula.
// Firebase se carga bajo demanda, al abrir el formulario.
import { firebaseConfig, NOTIFY_URL } from './firebase-config.js';

const FB = 'https://www.gstatic.com/firebasejs/10.14.1/';
const norm = (e) => String(e || '').trim().toLowerCase();

let loading;
function load() {
  return (loading ||= (async () => {
    const [{ initializeApp }, A, F] = await Promise.all([
      import(FB + 'firebase-app.js'),
      import(FB + 'firebase-auth.js'),
      import(FB + 'firebase-firestore.js'),
    ]);
    const app = initializeApp(firebaseConfig);
    return { A, F, auth: A.getAuth(app), db: F.getFirestore(app) };
  })());
}

// No se redefine si ya existe (permite sustituirlo en pruebas).
window.aulaApply ||= {
  preload: load,
  async onUser(cb) {
    const { A, auth } = await load();
    return A.onAuthStateChanged(auth, (u) => cb(u && u.email ? { email: norm(u.email), name: u.displayName || '' } : null));
  },
  async signIn() {
    const { A, auth } = await load();
    const p = new A.GoogleAuthProvider();
    p.setCustomParameters({ prompt: 'select_account' });
    await A.signInWithPopup(auth, p);
  },
  async signOut() {
    const { A, auth } = await load();
    await A.signOut(auth);
  },
  // Devuelve { existing: estado } si ya había una postulación a ese curso.
  async submit({ courseId, courseTitle, name, profile, goal }) {
    const { auth, db, F } = await load();
    const email = norm(auth.currentUser && auth.currentUser.email);
    if (!email) throw new Error('Sin sesión');
    const ref = F.doc(db, 'applications', `${email}__${courseId}`);
    const prev = await F.getDoc(ref);
    if (prev.exists()) return { existing: prev.data().status || 'pending' };
    await F.setDoc(ref, {
      ownerEmail: email, name: name.trim(), courseId, courseTitle,
      profile: profile.trim(), goal: goal.trim(), status: 'pending', createdAt: new Date().toISOString(),
    });
    // Aviso por correo al administrador (si está configurado). No bloquea ni afecta la postulación.
    if (NOTIFY_URL) {
      try {
        const idToken = await auth.currentUser.getIdToken();
        fetch(NOTIFY_URL, { method: 'POST', keepalive: true, headers: { 'Content-Type': 'text/plain;charset=utf-8' },
          body: JSON.stringify({ action: 'new_application', name: name.trim(), course: courseTitle, idToken }) }).catch(() => {});
      } catch { /* sin aviso */ }
    }
    return { ok: true };
  },
};

window.dispatchEvent(new Event('aula-apply-ready'));
