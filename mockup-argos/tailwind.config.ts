import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        argos: {
          ink: '#0C0A08',
          charcoal: '#1B1712',
          bronze: '#5C4626',
          gold: '#C9A658',
          goldDeep: '#8A6E2F',
          goldPale: '#E9D4A2',
          ivory: '#F6EEDD',
          parchment: '#EFE4CC',
          storm: '#1B2436',
          crimson: '#6E1F22',
        },
      },
      fontFamily: {
        display: ['"Cormorant Garamond"', 'Cormorant', 'serif'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        signature: ['"Pinyon Script"', 'cursive'],
      },
      boxShadow: {
        'argos-sm': '0 2px 6px rgba(12,10,8,0.08), 0 8px 20px rgba(12,10,8,0.06)',
        'argos': '0 4px 14px rgba(12,10,8,0.10), 0 24px 48px rgba(201,166,88,0.10), 0 40px 80px rgba(12,10,8,0.08)',
        'argos-lg': '0 8px 24px rgba(12,10,8,0.12), 0 32px 64px rgba(201,166,88,0.18), 0 60px 120px rgba(12,10,8,0.10)',
        'gold-glow': '0 0 24px rgba(201,166,88,0.45), 0 0 48px rgba(201,166,88,0.20)',
      },
      backgroundImage: {
        'gold-gradient':
          'linear-gradient(135deg, #E9D4A2 0%, #C9A658 20%, #8A6E2F 50%, #C9A658 80%, #E9D4A2 100%)',
        'gold-vertical':
          'linear-gradient(180deg, #E9D4A2 0%, #C9A658 40%, #8A6E2F 70%, #C9A658 100%)',
        'parchment':
          'radial-gradient(ellipse at top, #F6EEDD 0%, #EFE4CC 60%, #E7D9BA 100%)',
      },
      animation: {
        shimmer: 'shimmer 3s linear infinite',
        float: 'float 8s ease-in-out infinite',
        'float-slow': 'floatSlow 14s ease-in-out infinite',
        'spin-slow': 'spin 30s linear infinite',
        'gradient-shift': 'gradientShift 20s ease infinite',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 50%' },
          '100%': { backgroundPosition: '200% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translate3d(0,0,0)' },
          '50%': { transform: 'translate3d(0,-12px,0)' },
        },
        floatSlow: {
          '0%, 100%': { transform: 'translate3d(0,0,0) rotate(0deg)' },
          '50%': { transform: 'translate3d(0,-20px,0) rotate(4deg)' },
        },
        gradientShift: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
