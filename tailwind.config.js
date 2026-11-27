/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./_layouts/**/*.html",
    "./_includes/**/*.html",
    "./*.html",
    "./_articles/**/*.md",
    "./_projects/**/*.md",
    "./_courses/**/*.md",
    "./cursos/**/*.html",
    "./pages/**/*.html",
    "./articles/**/*.html",
    "./projects/**/*.html",
    "./assets/js/**/*.js",
  ],
  theme: {
    extend: {
      colors: {
        base: "#0A192F",
        accent: "#0068FF",
        accent2: "#00D2FF",
        // El aula virtual (body.aula) usa tema claro: estas variables permiten reasignar el blanco y los tonos
        // claros de acento a sus equivalentes oscuros sin tocar el resto del sitio (los valores por defecto son los de siempre).
        white: "rgb(var(--c-white, 255 255 255) / <alpha-value>)",
        emerald: {
          100: "rgb(var(--c-em-100, 209 250 229) / <alpha-value>)",
          200: "rgb(var(--c-em-200, 167 243 208) / <alpha-value>)",
          300: "rgb(var(--c-em-300, 110 231 183) / <alpha-value>)",
          400: "rgb(var(--c-em-400, 52 211 153) / <alpha-value>)",
          500: "rgb(var(--c-em-500, 16 185 129) / <alpha-value>)",
        },
        amber: {
          200: "rgb(var(--c-am-200, 253 230 138) / <alpha-value>)",
          300: "rgb(var(--c-am-300, 252 211 77) / <alpha-value>)",
          400: "rgb(var(--c-am-400, 251 191 36) / <alpha-value>)",
        },
        red: {
          200: "rgb(var(--c-rd-200, 254 202 202) / <alpha-value>)",
          300: "rgb(var(--c-rd-300, 252 165 165) / <alpha-value>)",
          400: "rgb(var(--c-rd-400, 248 113 113) / <alpha-value>)",
          500: "rgb(var(--c-rd-500, 239 68 68) / <alpha-value>)",
        },
        cyan: {
          200: "rgb(var(--c-cy-200, 165 243 252) / <alpha-value>)",
          300: "rgb(var(--c-cy-300, 103 232 249) / <alpha-value>)",
          400: "rgb(var(--c-cy-400, 34 211 238) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        display: ["Montserrat", "Inter", "ui-sans-serif"],
      },
      boxShadow: {
        soft: "0 10px 30px -12px rgba(0,0,0,0.35)",
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      animation: {
        float: 'float 6s ease-in-out infinite',
        shimmer: 'shimmer 3s linear infinite',
      },
    },
  },
  safelist: [],
  plugins: [],
}
