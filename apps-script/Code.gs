/**
 * Notificaciones por correo del aula virtual (Google Apps Script).
 *
 * Qué hace:
 *  - action "approved": envía al postulante el correo de aprobación.
 *    Solo lo puede pedir el administrador (se verifica con su sesión de Firebase).
 *  - action "new_application": avisa al dueño de este script de una postulación nueva.
 *    Lo puede pedir cualquier persona con sesión de Google, con límite de frecuencia.
 *
 * Los correos salen desde la cuenta con la que se publica este script.
 * Instrucciones de publicación: apps-script/LEEME.md
 */
const CONFIG = {
  // Clave web de Firebase (pública, la misma de assets/js/aula/firebase-config.js).
  FIREBASE_API_KEY: 'AIzaSyBnUiDujvVRtx-7MCjYFVRZikJpYpW7Ud4',
  // Único correo autorizado a pedir envíos de aprobación.
  ADMIN_EMAIL: 'ngarridoa18_1@unc.edu.pe',
  // Dirección del aula y nombre que verá el destinatario.
  AULA_URL: 'https://nrgarridoa.github.io/aula/',
  SENDER_NAME: 'Nilson Garrido',
  // Dónde llegan las respuestas de los postulantes.
  REPLY_TO: 'ngarridoa18_1@unc.edu.pe',
  // A dónde llega el aviso de postulaciones nuevas (vacío = la cuenta dueña del script).
  NOTIFY_TO: '',
  // Límites por hora para el aviso de postulaciones (evita abusos).
  MAX_NOTICES_PER_USER_PER_HOUR: 3,
  MAX_NOTICES_PER_HOUR: 30,
};

function doPost(e) {
  try {
    const req = JSON.parse((e && e.postData && e.postData.contents) || '{}');
    const user = verifyFirebaseUser_(req.idToken);

    if (req.action === 'approved') {
      if (user.email !== CONFIG.ADMIN_EMAIL.toLowerCase()) throw new Error('No autorizado');
      sendApproval_(req);
    } else if (req.action === 'new_application') {
      checkRate_(user.email);
      sendNotice_(user.email, req);
    } else {
      throw new Error('Acción no válida');
    }
    return json_({ ok: true });
  } catch (err) {
    return json_({ ok: false, error: String(err && err.message || err) });
  }
}

/** Confirma con Firebase que el token es válido y devuelve el correo verificado. */
function verifyFirebaseUser_(idToken) {
  if (!idToken) throw new Error('Sin sesión');
  const res = UrlFetchApp.fetch(
    'https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=' + encodeURIComponent(CONFIG.FIREBASE_API_KEY),
    { method: 'post', contentType: 'application/json', payload: JSON.stringify({ idToken: idToken }), muteHttpExceptions: true }
  );
  if (res.getResponseCode() !== 200) throw new Error('Sesión no válida');
  const u = (JSON.parse(res.getContentText()).users || [])[0];
  if (!u || !u.email || !u.emailVerified) throw new Error('Cuenta no verificada');
  return { email: String(u.email).toLowerCase() };
}

function sendApproval_(req) {
  const to = clean_(req.to, 200);
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)) throw new Error('Destinatario no válido');
  const name = clean_(req.name, 120) || 'Hola';
  const course = clean_(req.course, 200);
  const body =
    'Hola ' + name + ',\n\n' +
    'Tu postulación al curso "' + course + '" fue aprobada.\n\n' +
    'Ya puedes ingresar al aula virtual con tu cuenta de Google (' + to + '):\n' +
    CONFIG.AULA_URL + '\n\n' +
    'En "Mis cursos" encontrarás el contenido. Si tienes alguna dificultad para ingresar, responde a este correo.\n\n' +
    'Saludos,\n' + CONFIG.SENDER_NAME;
  MailApp.sendEmail({
    to: to,
    subject: 'Tu postulación a "' + course + '" fue aprobada',
    body: body,
    name: CONFIG.SENDER_NAME,
    replyTo: CONFIG.REPLY_TO,
  });
}

function sendNotice_(applicantEmail, req) {
  const to = CONFIG.NOTIFY_TO || Session.getEffectiveUser().getEmail();
  const body =
    'Hay una postulación nueva en el aula virtual.\n\n' +
    'Postulante: ' + clean_(req.name, 120) + ' (' + applicantEmail + ')\n' +
    'Curso: ' + clean_(req.course, 200) + '\n\n' +
    'Revísala en: ' + CONFIG.AULA_URL + '#/admin';
  MailApp.sendEmail({ to: to, subject: 'Nueva postulación: ' + clean_(req.course, 120), body: body, name: CONFIG.SENDER_NAME });
}

/** Límite de frecuencia por usuario y global (se reinicia cada hora). */
function checkRate_(email) {
  const cache = CacheService.getScriptCache();
  const bump = function (key, max) {
    const n = Number(cache.get(key) || 0);
    if (n >= max) throw new Error('Demasiadas solicitudes');
    cache.put(key, String(n + 1), 3600);
  };
  bump('u:' + email, CONFIG.MAX_NOTICES_PER_USER_PER_HOUR);
  bump('global', CONFIG.MAX_NOTICES_PER_HOUR);
}

/** Quita saltos de línea y limita la longitud de textos que vienen del navegador. */
function clean_(s, max) {
  return String(s || '').replace(/[\r\n\t]+/g, ' ').trim().slice(0, max);
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

/**
 * Ejecuta esta función UNA vez desde el editor (botón "Ejecutar") para conceder los permisos
 * de correo y de conexión externa antes de publicar.
 */
function autorizar() {
  MailApp.getRemainingDailyQuota();
  UrlFetchApp.fetch('https://www.google.com');
  Logger.log('Permisos concedidos. Cuota diaria de correos restante: ' + MailApp.getRemainingDailyQuota());
}
