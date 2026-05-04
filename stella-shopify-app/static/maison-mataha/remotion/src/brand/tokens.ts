// Maison Mataha brand tokens — extracted from the prototype design system.
// Single source of truth for colors, typography, motion durations.

export const colors = {
  // Core brand
  black: '#0a0604',
  paper: '#dfd0b2',
  cream: '#f5ede0',
  ivory: '#fff8ea',
  // Gold spectrum
  gold: '#c8984e',
  goldLight: '#e8c88a',
  goldDeep: '#8b6914',
  goldHighlight: '#f4d27a',
  // Wood / brown spectrum
  brownDark: '#1a0e06',
  brownWarm: '#3a2412',
  brownMid: '#6b4520',
  // Editorial dark mode
  inkBlack: '#0a0604',
  inkDeep: '#1a1108',
} as const;

export const fonts = {
  display: '"Cormorant Garamond", "Playfair Display", serif',
  body: '"Inter", system-ui, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, monospace',
} as const;

export const type = {
  hookLarge: { size: 110, weight: 300, italic: true, lineHeight: 1, letter: '-0.015em' },
  noteLarge: { size: 130, weight: 300, italic: true, lineHeight: 0.95, letter: '-0.01em' },
  eyebrow:   { size: 22,  weight: 500, italic: false, letter: '0.5em', tt: 'uppercase' as const },
  brandWord: { size: 38,  weight: 400, italic: false, letter: '0.42em', tt: 'uppercase' as const },
  brandTiny: { size: 14,  weight: 600, italic: false, letter: '0.6em',  tt: 'uppercase' as const },
  caption:   { size: 26,  weight: 600, italic: false, letter: '0.04em' },
  cta:       { size: 32,  weight: 600, italic: false, letter: '0.04em' },
} as const;

// Motion — durations in frames at 30 fps unless suffixed
export const fps = 30;
export const seconds = (s: number) => Math.round(s * fps);

export const motion = {
  // Standard ease-out — used for entries
  easeOutCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  easeInCubic:  (t: number) => t * t * t,
  // Used for stamp / overshoot effects
  easeOutBack: (t: number) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
} as const;

// Composition defaults — TikTok 9:16
export const composition = {
  width: 1080,
  height: 1920,
  fps,
  durationSeconds: 14,
} as const;

// Drop shadows used across overlays for legibility on busy footage
export const shadows = {
  textOnDark:  '0 4px 24px rgba(40,20,2,0.85), 0 0 50px rgba(0,0,0,0.4)',
  textOnLight: '0 2px 14px rgba(255,240,200,0.6), 0 0 40px rgba(255,255,255,0.4)',
  cta:         '10px 10px 0 #1a0e06, 0 30px 80px rgba(0,0,0,0.45)',
} as const;
