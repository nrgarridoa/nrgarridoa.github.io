# Correos y archivos del aula (Google Apps Script)

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

## Archivos del aula (clases, presentaciones, descargas)

El mismo script entrega al aula los archivos del curso guardados en Drive. Antes de enviar nada comprueba la sesión
de Firebase y que el alumno tenga el curso en su inscripción activa (lee su propio documento `enrollments/{correo}`
con su token). Los archivos no necesitan compartirse: solo tiene que poder leerlos la cuenta dueña del script.

- `CONFIG.COURSE_FILES` indica la carpeta del curso en Drive (por nombre, o por id con `folderId`).
- Cada cápsula tiene `aula/clase.html`, `aula/soluciones.html`, `aula/presentacion.pdf` (se generan con
  `export_aula.py` del curso), además de `notebook.ipynb` y `presentacion.pptx`. En la raíz va `paquete_curso.zip`.
- Tras editar `Code.gs`: ejecutar `autorizar` (pide el permiso de Drive) y publicar una versión nueva.

## Liberación por alumno

Los archivos del aula solo se entregan si la sesión está liberada para ese alumno (`released` o `releaseAll` en su
inscripción de Firestore). La carpeta de cada sesión sale del catálogo público del sitio (`aula/catalog.json`), no de lo
que envía el navegador. Al publicar este cambio hay que pegar también `firestore.rules` en la consola de Firebase.
