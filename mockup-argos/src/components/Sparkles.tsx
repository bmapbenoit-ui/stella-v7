import { motion } from 'framer-motion';

const DOTS = [
  { x: '12%', y: '18%', delay: 0, dur: 9 },
  { x: '88%', y: '22%', delay: 1.2, dur: 11 },
  { x: '72%', y: '78%', delay: 2.4, dur: 13 },
  { x: '18%', y: '82%', delay: 3.6, dur: 10 },
  { x: '52%', y: '8%', delay: 4.8, dur: 12 },
];

export function Sparkles() {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {DOTS.map((d, i) => (
        <motion.span
          key={i}
          className="sparkle"
          style={{ left: d.x, top: d.y }}
          animate={{
            y: [0, -14, 0, 10, 0],
            x: [0, 6, -6, 4, 0],
            opacity: [0, 0.9, 0.4, 0.8, 0],
            scale: [0.6, 1.1, 0.8, 1, 0.6],
          }}
          transition={{
            duration: d.dur,
            delay: d.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}
