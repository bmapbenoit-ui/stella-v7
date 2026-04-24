import { motion } from 'framer-motion';

/**
 * Atmospheric background for the hero — multi-layer night sky with drifting
 * clouds, stars, depth vignette, and a slow gold glow behind the bottle.
 * Renders only static + GPU-accelerated animated elements (no JS loop).
 */
export function StormSky() {
  return (
    <div aria-hidden className="absolute inset-0 overflow-hidden">
      {/* Deep sky gradient */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 90% 70% at 50% 40%, #1B2436 0%, #0C1220 45%, #07080E 100%)',
        }}
      />

      {/* Golden aura behind the bottle */}
      <motion.div
        className="absolute left-1/2 top-1/2 w-[62vmin] h-[62vmin] -translate-x-1/2 -translate-y-1/2 rounded-full blur-3xl"
        style={{
          background:
            'radial-gradient(circle, rgba(233,212,162,0.35) 0%, rgba(201,166,88,0.18) 35%, rgba(201,166,88,0.05) 60%, transparent 78%)',
        }}
        animate={{ opacity: [0.65, 1, 0.72, 1, 0.65], scale: [1, 1.06, 0.98, 1.04, 1] }}
        transition={{ duration: 11, repeat: Infinity, ease: 'easeInOut' }}
      />

      {/* Slow drifting clouds (SVG) */}
      <motion.div
        className="absolute inset-0 opacity-60"
        animate={{ x: ['-4%', '4%', '-4%'] }}
        transition={{ duration: 60, repeat: Infinity, ease: 'easeInOut' }}
      >
        <svg viewBox="0 0 1200 800" className="absolute inset-0 w-full h-full">
          <defs>
            <radialGradient id="cloudA" cx="0.5" cy="0.5" r="0.5">
              <stop offset="0%" stopColor="#324A66" stopOpacity="0.55" />
              <stop offset="100%" stopColor="#1A2233" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="cloudB" cx="0.5" cy="0.5" r="0.5">
              <stop offset="0%" stopColor="#3A5578" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#1A2233" stopOpacity="0" />
            </radialGradient>
          </defs>
          <ellipse cx="220" cy="180" rx="380" ry="110" fill="url(#cloudA)" />
          <ellipse cx="900" cy="140" rx="420" ry="130" fill="url(#cloudB)" />
          <ellipse cx="620" cy="640" rx="500" ry="110" fill="url(#cloudA)" opacity="0.7" />
          <ellipse cx="1050" cy="520" rx="300" ry="90" fill="url(#cloudB)" opacity="0.8" />
        </svg>
      </motion.div>

      {/* Subtle stars */}
      <svg className="absolute inset-0 w-full h-full opacity-80" aria-hidden>
        {STAR_POSITIONS.map((s, i) => (
          <circle
            key={i}
            cx={`${s.x}%`}
            cy={`${s.y}%`}
            r={s.r}
            fill="#F3DFA8"
            opacity={s.o}
          >
            <animate
              attributeName="opacity"
              values={`${s.o};${s.o * 0.2};${s.o}`}
              dur={`${s.dur}s`}
              repeatCount="indefinite"
              begin={`${s.delay}s`}
            />
          </circle>
        ))}
      </svg>

      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 45%, rgba(0,0,0,0.55) 100%)',
        }}
      />

      {/* Film grain */}
      <div
        className="absolute inset-0 opacity-[0.06] mix-blend-overlay pointer-events-none"
        style={{
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'><filter id='n'><feTurbulence baseFrequency='0.9' numOctaves='2' seed='3'/></filter><rect width='120' height='120' filter='url(%23n)'/></svg>\")",
          backgroundSize: '240px 240px',
        }}
      />
    </div>
  );
}

const STAR_POSITIONS = [
  { x: 8, y: 12, r: 1.2, o: 0.9, dur: 3.2, delay: 0 },
  { x: 22, y: 24, r: 0.8, o: 0.55, dur: 4.1, delay: 0.6 },
  { x: 38, y: 9, r: 1.6, o: 1, dur: 2.8, delay: 1.2 },
  { x: 54, y: 18, r: 0.9, o: 0.7, dur: 3.7, delay: 0.3 },
  { x: 72, y: 11, r: 1.3, o: 0.95, dur: 3, delay: 1.8 },
  { x: 88, y: 22, r: 0.7, o: 0.5, dur: 4.5, delay: 0.9 },
  { x: 16, y: 74, r: 0.9, o: 0.6, dur: 3.9, delay: 2.1 },
  { x: 42, y: 88, r: 1.1, o: 0.8, dur: 3.3, delay: 0 },
  { x: 68, y: 82, r: 0.8, o: 0.55, dur: 4.2, delay: 1.5 },
  { x: 92, y: 68, r: 1.4, o: 0.95, dur: 2.9, delay: 0.7 },
  { x: 5, y: 52, r: 0.7, o: 0.5, dur: 4, delay: 2.4 },
  { x: 96, y: 48, r: 1, o: 0.7, dur: 3.6, delay: 1.1 },
];
