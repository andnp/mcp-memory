/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0d1117',
        panel: '#161b22',
        border: '#30363d',
        text: '#c9d1d9',
        muted: '#8b949e',
        accent: '#58a6ff',
        success: '#3fb950',
        danger: '#f85149',
        amber: '#d29922'
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace']
      },
      boxShadow: {
        panel: '0 0 0 1px rgba(48, 54, 61, 0.75)'
      }
    },
  },
  plugins: [],
};