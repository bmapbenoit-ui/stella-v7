import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/**
 * Random lightning flashes across the hero. Three layers:
 * - Fullscreen white flash (very short)
 * - A lightning bolt SVG drawn across the sky
 * - A golden echo flash that lingers a beat longer
 */
export function LightningFlash() {
  const [boltKey, setBoltKey] = useState(0);
  const [boltVariant, setBoltVariant] = useState(0);

  useEffect(() => {
    let timeout: number;
    const schedule = () => {
      const next = 5500 + Math.random() * 6000;
      timeout = window.setTimeout(() => {
        setBoltVariant((v) => (v + 1) % BOLTS.length);
        setBoltKey((k) => k + 1);
        schedule();
      }, next);
    };
    schedule();
    return () => window.clearTimeout(timeout);
  }, []);

  const bolt = BOLTS[boltVariant];

  return (
    <AnimatePresence>
      <motion.div
        key={boltKey}
        className="absolute inset-0 pointer-events-none"
        aria-hidden
      >
        {/* Fullscreen white flash */}
        <motion.div
          className="absolute inset-0 bg-white"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 0.85, 0.2, 0.6, 0] }}
          transition={{ duration: 0.55, times: [0, 0.08, 0.2, 0.32, 1], ease: 'easeOut' }}
        />
        {/* Gold echo */}
        <motion.div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse at 50% 20%, rgba(233,212,162,0.45) 0%, transparent 55%)',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 0.9, 0.6, 0.35, 0] }}
          transition={{ duration: 1.4, times: [0, 0.1, 0.3, 0.6, 1] }}
        />
        {/* Bolt */}
        <svg
          viewBox="0 0 1000 800"
          preserveAspectRatio="none"
          className="absolute inset-0 w-full h-full"
        >
          <defs>
            <linearGradient id="boltGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#FFFFFF" />
              <stop offset="30%" stopColor="#F3DFA8" />
              <stop offset="100%" stopColor="#C9A658" />
            </linearGradient>
            <filter id="boltGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <motion.path
            d={bolt}
            stroke="url(#boltGrad)"
            strokeWidth={3}
            fill="none"
            filter="url(#boltGlow)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: [0, 1, 1, 1], opacity: [0, 1, 1, 0] }}
            transition={{ duration: 1.4, times: [0, 0.15, 0.6, 1] }}
          />
          {/* branching strokes */}
          <motion.path
            d={bolt}
            stroke="#ffffff"
            strokeWidth={1.5}
            fill="none"
            initial={{ opacity: 0 }}
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 0.45, times: [0, 0.2, 1] }}
          />
        </svg>
      </motion.div>
    </AnimatePresence>
  );
}

// Ranges of plausible lightning paths across the hero viewport.
const BOLTS = [
  'M 180 -20 L 260 160 L 200 220 L 340 380 L 270 460 L 420 640 L 360 720 L 500 820',
  'M 860 -20 L 780 140 L 840 200 L 700 360 L 760 440 L 620 600 L 680 700 L 560 820',
  'M 500 -20 L 450 180 L 520 240 L 400 400 L 480 480 L 380 640 L 460 720 L 380 820',
  'M 320 -20 L 420 200 L 360 260 L 480 420 L 420 500 L 540 680 L 480 760 L 600 820',
];
