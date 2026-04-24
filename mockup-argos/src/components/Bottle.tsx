import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useRef } from 'react';

/**
 * Argos signature bottle — based on the brand's design language:
 * frosted thick glass, solid metal neck, matte gold cap, hand-painted
 * enamel placard with mythological figure and Swarovski crystal accents.
 */
export function Bottle({ interactive = true }: { interactive?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rx = useSpring(useTransform(my, [-0.5, 0.5], [8, -8]), {
    stiffness: 120,
    damping: 14,
  });
  const ry = useSpring(useTransform(mx, [-0.5, 0.5], [-8, 8]), {
    stiffness: 120,
    damping: 14,
  });

  function handleMove(e: React.MouseEvent<HTMLDivElement>) {
    if (!interactive || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    mx.set((e.clientX - rect.left) / rect.width - 0.5);
    my.set((e.clientY - rect.top) / rect.height - 0.5);
  }

  function handleLeave() {
    mx.set(0);
    my.set(0);
  }

  return (
    <motion.div
      ref={ref}
      className="bottle-wrap relative mx-auto w-full max-w-[420px] aspect-[3/4]"
      style={{
        rotateX: rx,
        rotateY: ry,
        transformPerspective: 1200,
        transformStyle: 'preserve-3d',
      }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {/* Soft shadow beneath bottle */}
      <div
        aria-hidden
        className="absolute left-1/2 -translate-x-1/2 bottom-[2%] w-[65%] h-[32px] rounded-[50%]"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(12,10,8,0.28) 0%, rgba(12,10,8,0.10) 50%, transparent 75%)',
          filter: 'blur(8px)',
        }}
      />

      <svg
        viewBox="0 0 300 400"
        className="relative w-full h-full drop-shadow-[0_30px_60px_rgba(12,10,8,0.25)]"
        style={{ overflow: 'visible' }}
      >
        <defs>
          {/* Frosted glass gradient */}
          <linearGradient id="glass" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F3EADB" stopOpacity="0.95" />
            <stop offset="35%" stopColor="#E9DDC4" stopOpacity="0.8" />
            <stop offset="60%" stopColor="#D9C9A5" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#C2AC7F" stopOpacity="0.85" />
          </linearGradient>
          {/* Glass highlight */}
          <linearGradient id="highlight" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ffffff" stopOpacity="0.7" />
            <stop offset="30%" stopColor="#ffffff" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          {/* Gold cap gradient */}
          <linearGradient id="gold" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F3DFA8" />
            <stop offset="35%" stopColor="#C9A658" />
            <stop offset="65%" stopColor="#8A6E2F" />
            <stop offset="100%" stopColor="#C9A658" />
          </linearGradient>
          <linearGradient id="goldRim" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F3DFA8" />
            <stop offset="50%" stopColor="#C9A658" />
            <stop offset="100%" stopColor="#6E5724" />
          </linearGradient>
          <linearGradient id="plaque" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#1B2436" />
            <stop offset="100%" stopColor="#0C0A08" />
          </linearGradient>
          <radialGradient id="crystal" cx="0.3" cy="0.3" r="0.8">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="35%" stopColor="#E9D4A2" />
            <stop offset="100%" stopColor="#8A6E2F" />
          </radialGradient>
          <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="1.5" />
          </filter>
        </defs>

        {/* Bottle body */}
        <motion.g
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1.1, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
        >
          <rect
            x="50"
            y="120"
            width="200"
            height="240"
            rx="14"
            fill="url(#glass)"
            stroke="#B89E6E"
            strokeWidth="1.2"
          />
          {/* Inner liquid suggestion */}
          <rect
            x="62"
            y="150"
            width="176"
            height="200"
            rx="8"
            fill="#D4B884"
            opacity="0.28"
          />
          {/* Glass highlight streaks */}
          <rect
            x="62"
            y="130"
            width="36"
            height="220"
            rx="6"
            fill="url(#highlight)"
          />
          <rect
            x="220"
            y="148"
            width="18"
            height="180"
            rx="4"
            fill="url(#highlight)"
            opacity="0.6"
          />
        </motion.g>

        {/* Shoulder neck */}
        <motion.g
          initial={{ y: 10, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.15 }}
        >
          <path
            d="M 110 120 L 110 98 Q 110 92 116 92 L 184 92 Q 190 92 190 98 L 190 120 Z"
            fill="url(#goldRim)"
            stroke="#6E5724"
            strokeWidth="0.8"
          />
          <rect x="112" y="96" width="76" height="3" fill="#F3DFA8" opacity="0.7" />
        </motion.g>

        {/* Gold cap */}
        <motion.g
          initial={{ y: -24, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.05 }}
        >
          <rect
            x="102"
            y="40"
            width="96"
            height="58"
            rx="6"
            fill="url(#gold)"
            stroke="#6E5724"
            strokeWidth="1"
          />
          {/* subtle cap top shadow */}
          <rect
            x="102"
            y="40"
            width="96"
            height="14"
            rx="6"
            fill="#F3DFA8"
            opacity="0.7"
          />
          <rect
            x="102"
            y="92"
            width="96"
            height="6"
            fill="#6E5724"
            opacity="0.45"
          />
          {/* engraved logo on cap */}
          <text
            x="150"
            y="76"
            textAnchor="middle"
            fontFamily="Cormorant Garamond, serif"
            fontSize="14"
            fill="#3A2C10"
            letterSpacing="3"
          >
            ARGOS
          </text>
        </motion.g>

        {/* Enamel placard — Jupiter / thunderbolt */}
        <motion.g
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          <rect
            x="90"
            y="170"
            width="120"
            height="160"
            rx="6"
            fill="url(#plaque)"
            stroke="url(#gold)"
            strokeWidth="2"
          />
          {/* Inner gold border */}
          <rect
            x="96"
            y="176"
            width="108"
            height="148"
            rx="3"
            fill="none"
            stroke="#C9A658"
            strokeWidth="0.6"
            opacity="0.7"
          />
          {/* Jupiter figure — silhouette */}
          <g transform="translate(150,240)">
            {/* aura */}
            <circle r="42" fill="#1B2436" opacity="0.5" />
            <circle r="42" fill="url(#crystal)" opacity="0.08" />
            {/* body */}
            <path
              d="M -14 -30 Q -14 -38 0 -38 Q 14 -38 14 -30 L 14 -16 Q 20 -8 20 0 L 20 28 Q 20 36 12 36 L -12 36 Q -20 36 -20 28 L -20 0 Q -20 -8 -14 -16 Z"
              fill="#C9A658"
              opacity="0.85"
            />
            {/* head */}
            <circle cx="0" cy="-44" r="9" fill="#E9D4A2" />
            {/* beard */}
            <path d="M -6 -38 Q 0 -26 6 -38" stroke="#8A6E2F" strokeWidth="1.2" fill="none" />
            {/* thunderbolt */}
            <g className="thunderbolt">
              <path
                d="M 22 -18 L 38 -12 L 30 -6 L 44 0 L 32 6 L 42 12 L 26 8 L 34 18"
                fill="#F3DFA8"
                stroke="#8A6E2F"
                strokeWidth="0.8"
                strokeLinejoin="round"
              />
            </g>
            {/* outstretched arm */}
            <path
              d="M 12 -16 L 24 -16 L 26 -14"
              stroke="#C9A658"
              strokeWidth="3"
              strokeLinecap="round"
              fill="none"
            />
          </g>
          {/* crystal accents */}
          <motion.circle
            cx="105"
            cy="186"
            r="2.5"
            fill="url(#crystal)"
            animate={{ opacity: [0.4, 1, 0.4], scale: [0.9, 1.2, 0.9] }}
            transition={{ duration: 3, repeat: Infinity, delay: 0 }}
          />
          <motion.circle
            cx="195"
            cy="186"
            r="2.5"
            fill="url(#crystal)"
            animate={{ opacity: [0.4, 1, 0.4], scale: [0.9, 1.2, 0.9] }}
            transition={{ duration: 3, repeat: Infinity, delay: 0.8 }}
          />
          <motion.circle
            cx="105"
            cy="314"
            r="2.5"
            fill="url(#crystal)"
            animate={{ opacity: [0.4, 1, 0.4], scale: [0.9, 1.2, 0.9] }}
            transition={{ duration: 3, repeat: Infinity, delay: 1.6 }}
          />
          <motion.circle
            cx="195"
            cy="314"
            r="2.5"
            fill="url(#crystal)"
            animate={{ opacity: [0.4, 1, 0.4], scale: [0.9, 1.2, 0.9] }}
            transition={{ duration: 3, repeat: Infinity, delay: 2.2 }}
          />
          {/* Caption */}
          <text
            x="150"
            y="306"
            textAnchor="middle"
            fontFamily="Cormorant Garamond, serif"
            fontStyle="italic"
            fontSize="10"
            fill="#E9D4A2"
            letterSpacing="1.2"
          >
            IVPITER · TONANS
          </text>
        </motion.g>

        {/* Base etching */}
        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.1, duration: 0.8 }}
        >
          <line x1="70" y1="348" x2="230" y2="348" stroke="#8A6E2F" strokeWidth="0.6" opacity="0.6" />
          <text
            x="150"
            y="358"
            textAnchor="middle"
            fontFamily="Inter, sans-serif"
            fontSize="6"
            fill="#5C4626"
            letterSpacing="3"
          >
            ARTIST SERIES · V · EXTRAIT DE PARFUM · 30 ML
          </text>
        </motion.g>
      </svg>

      {/* Moving reflection sweep on hover */}
      <div className="bottle-reflect rounded-xl" />
    </motion.div>
  );
}
