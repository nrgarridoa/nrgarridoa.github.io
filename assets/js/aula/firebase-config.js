// Configuración pública del proyecto Firebase (no es un secreto: la seguridad
// la dan las reglas de Firestore, ver /firestore.rules).
export const firebaseConfig = {
  apiKey: "AIzaSyBnUiDujvVRtx-7MCjYFVRZikJpYpW7Ud4",
  authDomain: "aula-nrgarridoa.firebaseapp.com",
  projectId: "aula-nrgarridoa",
  storageBucket: "aula-nrgarridoa.firebasestorage.app",
  messagingSenderId: "1024643055379",
  appId: "1:1024643055379:web:55557a6ea90b8eb6b764e6"
};

// Correo del administrador (debe coincidir con firestore.rules).
export const ADMIN_EMAIL = "ngarridoa18_1@unc.edu.pe";

// Contacto para solicitar acceso.
export const CONTACT_EMAIL = "ngarridoa18_1@unc.edu.pe";

// URL de la aplicación web de Google Apps Script que envía los correos (ver apps-script/LEEME.md).
// Vacía = no se envían correos y el resto del aula funciona igual.
export const NOTIFY_URL = "https://script.google.com/macros/s/AKfycbwCrfKJ9wnMQOVsiOyaORjAq5lHIC0WWfsAj9I6rLtHn6pLLmAov_pfRLVTNQ13fJdMDA/exec";
