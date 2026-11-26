/**
 * Notificaciones por correo del aula virtual (Google Apps Script).
 *
 * Qué hace:
 *  - action "approved": envía al postulante el correo de aprobación.
 *    Solo lo puede pedir el administrador (se verifica con su sesión de Firebase).
 *  - action "new_application": avisa al dueño de este script de una postulación nueva.
 *    Lo puede pedir cualquier persona con sesión de Google, con límite de frecuencia.
 *
 * Los correos salen desde la cuenta con la que se publica este script, en HTML con
 * versión de texto plano de respaldo. Instrucciones de publicación: apps-script/LEEME.md
 */
const CONFIG = {
  // Clave web de Firebase (pública, la misma de assets/js/aula/firebase-config.js).
  FIREBASE_API_KEY: 'AIzaSyBnUiDujvVRtx-7MCjYFVRZikJpYpW7Ud4',
  // Único correo autorizado a pedir envíos de aprobación.
  ADMIN_EMAIL: 'ngarridoa18_1@unc.edu.pe',
  // Dirección del aula y datos que verá el destinatario.
  AULA_URL: 'https://nrgarridoa.github.io/aula/',
  SITE_URL: 'https://nrgarridoa.github.io/',
  SENDER_NAME: 'Nilson Garrido',
  SENDER_ROLE: 'Mining Engineer · Data Analyst',
  // Dónde llegan las respuestas de los postulantes.
  REPLY_TO: 'ngarridoa18_1@unc.edu.pe',
  // A dónde llega el aviso de postulaciones nuevas (vacío = la cuenta dueña del script).
  NOTIFY_TO: '',
  // Límites por hora para el aviso de postulaciones (evita abusos).
  MAX_NOTICES_PER_USER_PER_HOUR: 3,
  MAX_NOTICES_PER_HOUR: 30,
  // Colores de marca.
  // Proyecto de Firebase (para comprobar la inscripción del alumno).
  FIREBASE_PROJECT_ID: 'aula-nrgarridoa',
  // Archivos del aula (clases, presentaciones, descargas), guardados en Drive.
  // Clave = id del curso. folderName = carpeta del curso en "Mi unidad" de la cuenta dueña del script;
  // folderId (opcional) la fija por id y evita buscarla por nombre.
  COURSE_FILES: {
    'python-geologia': { folderName: 'Python aplicado a la geología', folderId: '' },
  },
  NAVY: '#0A192F',
  ACCENT: '#0068FF',
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
    } else if (req.action === 'file') {
      return json_(serveFile_(user, req));
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

/* --------------------- Archivos del aula (Drive) --------------------- */

// Nombre lógico (lo que pide el aula) -> ruta dentro de la carpeta de la cápsula.
const FILE_KINDS = {
  clase: { path: 'aula/clase.html', mime: 'text/html' },
  soluciones: { path: 'aula/soluciones.html', mime: 'text/html' },
  presentacion: { path: 'aula/presentacion.pdf', mime: 'application/pdf' },
  notebook: { path: 'notebook.ipynb', mime: 'application/x-ipynb+json' },
  pptx: { path: 'presentacion.pptx', mime: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' },
};
// Archivo suelto en la raíz del curso (por ejemplo el paquete para ejecutar los notebooks).
const ROOT_KINDS = { paquete: { path: 'paquete_curso.zip', mime: 'application/zip' } };

/** Entrega un archivo del curso solo a quien está inscrito (o es el administrador). */
function serveFile_(user, req) {
  const courseId = String(req.courseId || '');
  const course = CONFIG.COURSE_FILES[courseId];
  if (!course) throw new Error('Curso sin archivos');
  if (user.email !== CONFIG.ADMIN_EMAIL.toLowerCase()) requireEnrollment_(user.email, courseId, req.idToken);

  const kind = String(req.kind || '');
  let spec, folder = String(req.folder || '');
  if (ROOT_KINDS[kind]) { spec = ROOT_KINDS[kind]; folder = ''; }
  else {
    spec = FILE_KINDS[kind];
    if (!spec) throw new Error('Archivo no válido');
    if (!/^modulo_\d{2}_[a-z0-9_]+\/c\d+\.\d+_[a-z0-9_]+$/.test(folder)) throw new Error('Carpeta no válida');
  }
  const rel = (folder ? folder + '/' : '') + spec.path;
  const cache = CacheService.getScriptCache();
  const ckey = 'f:' + courseId + ':' + rel;
  let file = null;
  const cached = cache.get(ckey);
  if (cached) { try { file = DriveApp.getFileById(cached); } catch (e) { file = null; } }
  if (!file) {
    file = findFile_(course, rel);
    if (!file) throw new Error('Archivo no disponible todavía');
    cache.put(ckey, file.getId(), 21600);
  }
  return { ok: true, mime: spec.mime, b64: Utilities.base64Encode(file.getBlob().getBytes()) };
}

/** Comprueba, con la propia sesión del alumno, que su inscripción activa incluye el curso. */
function requireEnrollment_(email, courseId, idToken) {
  const url = 'https://firestore.googleapis.com/v1/projects/' + CONFIG.FIREBASE_PROJECT_ID +
    '/databases/(default)/documents/enrollments/' + encodeURIComponent(email);
  const res = UrlFetchApp.fetch(url, { headers: { Authorization: 'Bearer ' + idToken }, muteHttpExceptions: true });
  if (res.getResponseCode() !== 200) throw new Error('Sin acceso al curso');
  const f = JSON.parse(res.getContentText()).fields || {};
  const active = !f.active || f.active.booleanValue !== false;
  const list = ((f.courses && f.courses.arrayValue && f.courses.arrayValue.values) || []).map(function (v) { return v.stringValue; });
  if (!active || list.indexOf(courseId) < 0) throw new Error('Sin acceso al curso');
}

function findFile_(course, rel) {
  let dir;
  if (course.folderId) dir = DriveApp.getFolderById(course.folderId);
  else {
    const it = DriveApp.getFoldersByName(course.folderName);
    if (!it.hasNext()) return null;
    dir = it.next();
  }
  const parts = rel.split('/');
  for (let i = 0; i < parts.length - 1; i++) {
    const it = dir.getFoldersByName(parts[i]);
    if (!it.hasNext()) return null;
    dir = it.next();
  }
  const fit = dir.getFilesByName(parts[parts.length - 1]);
  return fit.hasNext() ? fit.next() : null;
}

/* ------------------------------ Envíos ------------------------------ */

function sendApproval_(req) {
  const to = clean_(req.to, 200);
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(to)) throw new Error('Destinatario no válido');
  const mail = approvalEmail_(to, clean_(req.name, 120), clean_(req.course, 200));
  MailApp.sendEmail({
    to: to, subject: mail.subject, body: mail.text, htmlBody: mail.html,
    name: CONFIG.SENDER_NAME, replyTo: CONFIG.REPLY_TO,
  });
}

function sendNotice_(applicantEmail, req) {
  const to = CONFIG.NOTIFY_TO || Session.getEffectiveUser().getEmail();
  const mail = noticeEmail_({
    name: clean_(req.name, 120), email: applicantEmail, course: clean_(req.course, 200),
    linkedin: clean_(req.linkedin, 300), profile: text_(req.profile, 800), goal: text_(req.goal, 800),
  });
  MailApp.sendEmail({ to: to, subject: mail.subject, body: mail.text, htmlBody: mail.html, name: CONFIG.SENDER_NAME });
}

/* ------------------------- Contenido de los correos ------------------------- */

function approvalEmail_(to, name, course) {
  const subject = 'Tu postulación a "' + course + '" fue aprobada';
  const html = layout_({
    preheader: 'Ya puedes ingresar al aula virtual con tu cuenta de Google.',
    title: 'Tu postulación fue aprobada',
    body:
      p_('Hola ' + esc_(name || '') + ',') +
      p_('Revisé tu perfil y te doy la bienvenida al curso:') +
      card_('<strong style="font-size:16px;color:' + CONFIG.NAVY + ';">' + esc_(course) + '</strong>') +
      p_('Para comenzar:') +
      steps_([
        'Ingresa al aula virtual con tu cuenta de Google <strong>' + esc_(to) + '</strong>.',
        'En <strong>Mis cursos</strong> encontrarás el contenido, tu progreso y tus notas.',
      ]),
    cta: { label: 'Ingresar al aula', url: CONFIG.AULA_URL },
    after: p_('Si tienes alguna dificultad para ingresar, responde a este correo y te ayudo.', true),
    footer: 'Recibes este correo porque postulaste a un curso en nrgarridoa.github.io.',
  });
  const text =
    'Hola ' + (name || '') + ',\n\n' +
    'Tu postulación al curso "' + course + '" fue aprobada.\n\n' +
    'Ingresa al aula virtual con tu cuenta de Google (' + to + '):\n' + CONFIG.AULA_URL + '\n\n' +
    'En "Mis cursos" encontrarás el contenido. Si tienes alguna dificultad, responde a este correo.\n\n' +
    'Saludos,\n' + CONFIG.SENDER_NAME;
  return { subject: subject, html: html, text: text };
}

function noticeEmail_(a) {
  const link = safeLinkedin_(a.linkedin);
  const linkedinHtml = link
    ? '<a href="' + link + '" style="color:' + CONFIG.ACCENT + ';text-decoration:none;">' + esc_(a.linkedin) + '</a>'
    : (a.linkedin ? esc_(a.linkedin) : '<span style="color:#94a3b8;">No indicado</span>');
  const row = function (k, v) {
    return '<tr><td valign="top" style="padding:8px 12px 8px 0;font-size:13px;color:#64748b;white-space:nowrap;">' + k +
      '</td><td style="padding:8px 0;font-size:14px;color:#0f172a;line-height:1.5;">' + v + '</td></tr>';
  };
  const html = layout_({
    preheader: a.name + ' postuló a ' + a.course,
    title: 'Nueva postulación',
    body:
      p_('Hay una postulación nueva en el aula virtual.') +
      '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-top:1px solid #e2e8f0;margin:6px 0 4px;">' +
        row('Postulante', '<strong>' + esc_(a.name) + '</strong>') +
        row('Correo', esc_(a.email)) +
        row('Curso', esc_(a.course)) +
        row('LinkedIn', linkedinHtml) +
        row('Perfil', nl2br_(a.profile)) +
        row('Espera aplicar', nl2br_(a.goal)) +
      '</table>',
    cta: { label: 'Revisar postulación', url: CONFIG.AULA_URL + '#/admin' },
    footer: 'Aviso automático del aula virtual.',
  });
  const text =
    'Hay una postulación nueva en el aula virtual.\n\n' +
    'Postulante: ' + a.name + ' (' + a.email + ')\n' +
    'Curso: ' + a.course + '\n' +
    'LinkedIn: ' + (a.linkedin || 'No indicado') + '\n\n' +
    'Perfil:\n' + a.profile + '\n\n' +
    'Espera aplicar:\n' + a.goal + '\n\n' +
    'Revísala en: ' + CONFIG.AULA_URL + '#/admin';
  return { subject: 'Nueva postulación: ' + a.course, html: html, text: text };
}

/* ------------------------- Plantilla HTML (apta para correo) ------------------------- */

function layout_(o) {
  const font = "font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;";
  const cta = o.cta
    ? '<table role="presentation" cellpadding="0" cellspacing="0" style="margin:26px 0 6px;"><tr><td bgcolor="' + CONFIG.ACCENT +
      '" style="border-radius:8px;"><a href="' + esc_(o.cta.url) + '" style="display:inline-block;padding:13px 26px;' + font +
      'font-size:15px;font-weight:600;color:#ffffff;text-decoration:none;border-radius:8px;">' + esc_(o.cta.label) + '</a></td></tr></table>'
    : '';
  return '<!DOCTYPE html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<meta name="color-scheme" content="light"><title>' + esc_(o.title) + '</title></head>' +
    '<body style="margin:0;padding:0;background:#eef2f7;">' +
    '<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">' + esc_(o.preheader || '') + '</div>' +
    '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" bgcolor="#eef2f7"><tr><td align="center" style="padding:28px 12px;">' +
      '<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="width:100%;max-width:600px;background:#ffffff;border-radius:14px;overflow:hidden;' + font + '">' +
        // cabecera
        '<tr><td bgcolor="' + CONFIG.NAVY + '" style="padding:22px 28px;">' +
          '<table role="presentation" cellpadding="0" cellspacing="0"><tr>' +
            '<td bgcolor="' + CONFIG.ACCENT + '" width="40" height="40" align="center" style="border-radius:9px;color:#ffffff;font-size:15px;font-weight:700;' + font + '">NG</td>' +
            '<td style="padding-left:12px;' + font + '"><div style="color:#ffffff;font-size:16px;font-weight:700;">' + esc_(CONFIG.SENDER_NAME) + '</div>' +
            '<div style="color:#94a3b8;font-size:12px;margin-top:2px;">' + esc_(CONFIG.SENDER_ROLE) + '</div></td>' +
          '</tr></table></td></tr>' +
        // línea de acento
        '<tr><td height="3" bgcolor="' + CONFIG.ACCENT + '" style="font-size:0;line-height:0;">&nbsp;</td></tr>' +
        // contenido
        '<tr><td style="padding:32px 28px 8px;' + font + 'color:#334155;">' +
          '<h1 style="margin:0 0 18px;font-size:24px;line-height:1.25;color:' + CONFIG.NAVY + ';font-weight:700;">' + esc_(o.title) + '</h1>' +
          o.body + cta + (o.after || '') +
        '</td></tr>' +
        // pie
        '<tr><td style="padding:22px 28px 28px;' + font + '"><div style="border-top:1px solid #e2e8f0;padding-top:16px;font-size:12px;line-height:1.6;color:#94a3b8;">' +
          esc_(o.footer || '') + '<br><a href="' + esc_(CONFIG.SITE_URL) + '" style="color:#64748b;text-decoration:none;">nrgarridoa.github.io</a></div></td></tr>' +
      '</table></td></tr></table></body></html>';
}

function p_(html, small) {
  return '<p style="margin:0 0 14px;font-size:' + (small ? '13px' : '15px') + ';line-height:1.65;color:' + (small ? '#64748b' : '#334155') + ';">' + html + '</p>';
}

function card_(html) {
  return '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:6px 0 18px;"><tr>' +
    '<td width="4" bgcolor="' + CONFIG.ACCENT + '" style="border-radius:4px 0 0 4px;font-size:0;">&nbsp;</td>' +
    '<td bgcolor="#f1f5f9" style="padding:14px 16px;border-radius:0 8px 8px 0;font-size:15px;line-height:1.5;">' + html + '</td></tr></table>';
}

function steps_(items) {
  return '<table role="presentation" cellpadding="0" cellspacing="0" style="margin:0 0 6px;">' + items.map(function (t, i) {
    return '<tr><td valign="top" style="padding:0 12px 10px 0;"><div style="width:24px;height:24px;line-height:24px;text-align:center;border-radius:12px;background:' +
      CONFIG.NAVY + ';color:#ffffff;font-size:12px;font-weight:700;">' + (i + 1) + '</div></td>' +
      '<td style="padding:2px 0 10px;font-size:15px;line-height:1.55;color:#334155;">' + t + '</td></tr>';
  }).join('') + '</table>';
}

/* ------------------------------ Utilidades ------------------------------ */

function esc_(s) {
  return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function nl2br_(s) { return esc_(s).replace(/\n/g, '<br>'); }

/** Texto de una sola línea: sin saltos de línea y con longitud limitada. */
function clean_(s, max) {
  return String(s || '').replace(/[\r\n\t]+/g, ' ').trim().slice(0, max);
}
/** Texto multilínea: conserva saltos de línea (máx. 2 seguidos) y limita la longitud. */
function text_(s, max) {
  return String(s || '').replace(/\r/g, '').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim().slice(0, max);
}
/** Devuelve la URL escapada solo si es un enlace de LinkedIn por https. */
function safeLinkedin_(url) {
  return /^https:\/\/([a-z0-9-]+\.)?linkedin\.com\/[^\s"'<>]+$/i.test(url || '') ? esc_(url) : '';
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
  DriveApp.getRootFolder();   // permiso de lectura de Drive (archivos del aula)
  Logger.log('Permisos concedidos. Cuota diaria de correos restante: ' + MailApp.getRemainingDailyQuota());
}

/**
 * Prueba opcional: envía a la cuenta dueña del script un ejemplo de cada correo.
 * Útil para ver el diseño sin hacer una postulación real.
 */
function enviarPruebas() {
  const me = Session.getEffectiveUser().getEmail();
  const a = approvalEmail_(me, 'Nombre de Prueba', 'Python para Minería');
  MailApp.sendEmail({ to: me, subject: '[Prueba] ' + a.subject, body: a.text, htmlBody: a.html, name: CONFIG.SENDER_NAME });
  const n = noticeEmail_({
    name: 'Nombre de Prueba', email: 'prueba@gmail.com', course: 'Python para Minería',
    linkedin: 'https://www.linkedin.com/in/prueba', profile: 'Analista de planeamiento en una minera.\n3 años de experiencia.', goal: 'Automatizar reportes semanales.',
  });
  MailApp.sendEmail({ to: me, subject: '[Prueba] ' + n.subject, body: n.text, htmlBody: n.html, name: CONFIG.SENDER_NAME });
  Logger.log('Se enviaron 2 correos de prueba a ' + me);
}
