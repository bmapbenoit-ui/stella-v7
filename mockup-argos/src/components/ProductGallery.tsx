import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef, useState } from 'react';
import { Bottle } from './Bottle';
import { Sparkles } from './Sparkles';
import { Expand } from 'lucide-react';

const THUMBS = [
  { key: 'pack', label: 'Pack Shot' },
  { key: 'mood', label: 'Étude Mythologique' },
  { key: 'ingredient', label: 'Matière' },
];

export function ProductGallery() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });
  const y = useTransform(scrollYProgress, [0, 1], ['-6%', '6%']);
  const [active, setActive] = useState(0);

  return (
    <div ref={ref} className="relative">
      <div className="relative aspect-[4/5] md:aspect-square rounded-[4px] overflow-hidden gold-border bg-gradient-to-br from-argos-ivory via-argos-parchment to-[#E9DDC4]">
        {/* Parallax mood background */}
        <motion.div
          style={{ y }}
          className="absolute inset-0"
          aria-hidden
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_25%,rgba(255,255,255,0.8),transparent_55%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(201,166,88,0.18),transparent_60%)]" />
          {/* Subtle Greek meander border */}
          <div className="absolute left-6 right-6 top-6 h-3 meander-border opacity-60" />
          <div className="absolute left-6 right-6 bottom-6 h-3 meander-border opacity-60 rotate-180" />
        </motion.div>

        <Sparkles />

        {/* Centered bottle with entrance */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center px-6 md:px-12"
          initial={{ scale: 1.12, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
          style={{ y: useTransform(scrollYProgress, [0, 1], ['0%', '-8%']) }}
        >
          <Bottle />
        </motion.div>

        {/* Artist series badge */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1, duration: 0.7 }}
          className="absolute top-6 left-6 flex items-center gap-2"
        >
          <span className="h-[1px] w-6 bg-argos-gold" />
          <span className="text-[10px] uppercase tracking-[0.3em] text-argos-bronze">
            Artist Series · V
          </span>
        </motion.div>

        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.3 }}
          className="absolute bottom-5 right-5 w-10 h-10 rounded-full bg-argos-ink/70 backdrop-blur text-argos-goldPale flex items-center justify-center hover:bg-argos-ink transition-colors cursor-hover"
          aria-label="Agrandir"
        >
          <Expand size={16} />
        </motion.button>
      </div>

      {/* Thumbnails */}
      <div className="grid grid-cols-3 gap-3 md:gap-4 mt-4">
        {THUMBS.map((t, i) => (
          <motion.button
            key={t.key}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 + i * 0.12, duration: 0.5 }}
            whileHover={{ y: -4 }}
            onClick={() => setActive(i)}
            className={`group relative aspect-square rounded-[3px] overflow-hidden border transition-all cursor-hover ${
              active === i
                ? 'border-argos-gold shadow-[0_6px_20px_rgba(201,166,88,0.35)]'
                : 'border-argos-bronze/20 hover:border-argos-gold/60'
            }`}
          >
            <Thumb kind={t.key as 'pack' | 'mood' | 'ingredient'} />
            <span className="absolute bottom-1.5 left-2 text-[8.5px] tracking-[0.2em] uppercase text-argos-ivory/80 drop-shadow">
              {t.label}
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

function Thumb({ kind }: { kind: 'pack' | 'mood' | 'ingredient' }) {
  if (kind === 'pack') {
    return (
      <div className="absolute inset-0 bg-white flex items-center justify-center">
        <div className="w-[55%] h-[70%] scale-75 origin-center">
          <Bottle interactive={false} />
        </div>
      </div>
    );
  }
  if (kind === 'mood') {
    return (
      <div className="absolute inset-0 bg-argos-ink overflow-hidden">
        <div
          className="absolute inset-0 ken-burns"
          style={{
            background:
              'radial-gradient(ellipse at 60% 40%, #2a2538 0%, #0C0A08 65%), repeating-linear-gradient(45deg, rgba(201,166,88,0.08), rgba(201,166,88,0.08) 2px, transparent 2px, transparent 10px)',
          }}
        />
        {/* Lightning bolt */}
        <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full">
          <path
            d="M 55 10 L 40 50 L 55 50 L 38 92 L 70 42 L 55 42 L 65 10 Z"
            fill="url(#thGold)"
            stroke="#F3DFA8"
            strokeWidth="0.5"
            className="drop-shadow-[0_0_14px_rgba(233,212,162,0.7)]"
          />
          <defs>
            <linearGradient id="thGold" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#F3DFA8" />
              <stop offset="50%" stopColor="#C9A658" />
              <stop offset="100%" stopColor="#8A6E2F" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    );
  }
  return (
    <div className="absolute inset-0 overflow-hidden ken-burns">
      {/* Ingredient mood - cardamom / cedar wood */}
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at 40% 30%, #8B6F3E 0%, #5C4626 40%, #2C2011 90%)',
        }}
      />
      <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full opacity-70">
        {/* Cardamom pods */}
        <ellipse cx="30" cy="55" rx="14" ry="8" fill="#A8C474" opacity="0.7" transform="rotate(-20 30 55)" />
        <ellipse cx="65" cy="42" rx="12" ry="7" fill="#C4D88A" opacity="0.8" transform="rotate(18 65 42)" />
        <ellipse cx="55" cy="70" rx="13" ry="7" fill="#B8CC80" opacity="0.65" transform="rotate(-5 55 70)" />
      </svg>
    </div>
  );
}
