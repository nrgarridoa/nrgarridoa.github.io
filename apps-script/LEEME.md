# Notificaciones por correo (Google Apps Script)

Envía por correo (1) la aprobación al postulante y (2) un aviso al administrador por cada postulación nueva.
No se publica en el sitio: es solo el código que se pega en Google Apps Script.

## Publicación

1. Entra a <https://script.google.com> con la cuenta **desde la que saldrán los correos** y crea un proyecto nuevo.
2. Borra el contenido del editor y pega todo el archivo `Code.gs`. Guarda.
3. En el selector de funciones elige `autorizar` y pulsa **Ejecutar**. Acepta los permisos
   (si aparece "Google no verificó esta aplicación": **Configuración avanzada → Ir a (proyecto)**).
4. **Implementar → Nueva implementación → tipo "Aplicación web"**:
   - Ejecutar como: **Yo**
   - Quién tiene acceso: **Cualquier persona**
5. Pulsa **Implementar** y copia la **URL de la aplicación web** (termina en `/exec`).
6. Pega esa URL en `assets/js/aula/firebase-config.js`, en `NOTIFY_URL`, y publica el sitio.

## Cambios posteriores al script

Si editas `Code.gs`, hay que publicar una versión nueva: **Implementar → Administrar implementaciones →
editar (lápiz) → Versión: Nueva versión → Implementar**. La URL no cambia.

## Notas

- Límite de Gmail: unos 100 correos diarios con una cuenta personal.
- El script verifica la sesión de Firebase de quien lo llama: las aprobaciones solo las puede pedir `ADMIN_EMAIL`.
- Si la clave web de Firebase se restringe por referente HTTP en Google Cloud, hay que permitir también
  las llamadas del servidor, porque el script la usa sin referente.
