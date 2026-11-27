// Bundle de librerias para el cuaderno interactivo del aula (CodeMirror 6, marked, DOMPurify, KaTeX).
// Se compila con: npx esbuild entry.js --bundle --format=esm --minify --outfile=assets/vendor/nbvendor.mjs
import { EditorView, keymap, highlightSpecialChars, drawSelection } from '@codemirror/view';
import { EditorState, Prec } from '@codemirror/state';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { python } from '@codemirror/lang-python';
import { syntaxHighlighting, HighlightStyle, indentOnInput, bracketMatching } from '@codemirror/language';
import { closeBrackets, closeBracketsKeymap } from '@codemirror/autocomplete';
import { tags as t } from '@lezer/highlight';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
import { unzipSync, strFromU8 } from 'fflate';
export { unzipSync, strFromU8 };

// Colores clasicos de IDLE, igual que en los notebooks del curso.
const idle = HighlightStyle.define([
  { tag: [t.keyword, t.controlKeyword, t.definitionKeyword, t.moduleKeyword, t.operatorKeyword, t.bool, t.null, t.self], color: '#FF7700' },
  { tag: [t.string, t.special(t.string)], color: '#00AA00' },
  { tag: t.comment, color: '#DD0000', fontStyle: 'italic' },
  { tag: [t.standard(t.variableName), t.standard(t.function(t.variableName))], color: '#900090' },
  { tag: [t.definition(t.variableName), t.definition(t.function(t.variableName)), t.className], color: '#0000FF' },
  { tag: [t.number, t.integer, t.float], color: '#0F172A' },
  { tag: t.invalid, color: '#DD0000' },
]);

const theme = EditorView.theme({
  '&': { backgroundColor: '#FFFFFF', color: '#0F172A', fontSize: '13.5px' },
  '.cm-content': { fontFamily: '"JetBrains Mono", ui-monospace, Menlo, Consolas, monospace', padding: '10px 4px', caretColor: '#059669' },
  '.cm-line': { padding: '0 10px' },
  '&.cm-focused': { outline: 'none' },
  '.cm-cursor': { borderLeftColor: '#059669' },
  '.cm-selectionBackground, &.cm-focused .cm-selectionBackground': { backgroundColor: 'rgba(16,185,129,.22)' },
  '.cm-matchingBracket': { backgroundColor: 'rgba(16,185,129,.25)', outline: 'none' },
  '.cm-scroller': { overflow: 'auto', lineHeight: '1.55' },
});

export function createEditor(parent, doc, { onRun, onChange }) {
  const run = (mode) => () => { onRun && onRun(mode); return true; };
  const state = EditorState.create({
    doc,
    extensions: [
      history(), drawSelection(), highlightSpecialChars(), indentOnInput(), bracketMatching(), closeBrackets(),
      python(), syntaxHighlighting(idle), theme,
      Prec.highest(keymap.of([{ key: 'Shift-Enter', run: run('next') }, { key: 'Mod-Enter', run: run('stay') }, { key: 'Alt-Enter', run: run('insert') }])),
      keymap.of([...closeBracketsKeymap, ...defaultKeymap, ...historyKeymap, indentWithTab]),
      EditorView.updateListener.of((u) => { if (u.docChanged && onChange) onChange(u.state.doc.toString()); }),
    ],
  });
  const view = new EditorView({ state, parent });
  return {
    view,
    getValue: () => view.state.doc.toString(),
    setValue: (v) => view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: v } }),
    focus: () => view.focus(),
  };
}

// Markdown + HTML + LaTeX de las celdas de texto, saneado.
export function renderMarkdown(src) {
  const maths = [];
  const stash = (m, display) => { maths.push([m, display]); return `@@M${maths.length - 1}@@`; };
  let s = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => stash(m, true));
  s = s.replace(/(?<![\\$])\$([^\n$]+?)\$(?!\d)/g, (_, m) => stash(m, false));
  let html = marked.parse(s, { gfm: true });
  html = DOMPurify.sanitize(html, { FORBID_TAGS: ['style', 'script', 'iframe', 'form', 'object', 'embed'], ADD_ATTR: ['target'] });
  return html.replace(/@@M(\d+)@@/g, (_, i) => katex.renderToString(maths[i][0], { displayMode: maths[i][1], throwOnError: false }));
}

// HTML producido por el codigo del alumno (por ejemplo un DataFrame).
export function sanitizeOutputHtml(html) {
  return DOMPurify.sanitize(html, { FORBID_TAGS: ['style', 'script', 'iframe', 'form', 'object', 'embed'] });
}
