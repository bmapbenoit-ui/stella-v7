import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useRef } from 'react';

/**
 * Jupiter's Lightning bottle — modeled after the actual product photography
 * (frosted hexagonal glass body, faceted matte-gold cap, engraved gold
 * neck ring reading "ARGOS", enameled Zeus plaque with eagle & thunderbolts
 * on navy sky, ARGOS embossed at the stepped base).
 */
export function Bottle({ interactive = true }: { interactive?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);
  const rx = useSpring(useTransform(my, [-0.5, 0.5], [7, -7]), {
    stiffness: 110,
    damping: 14,
  });
  const ry = useSpring(useTransform(mx, [-0.5, 0.5], [-9, 9]), {
    stiffness: 110,
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
      className="bottle-wrap relative mx-auto w-full max-w-[480px]"
      style={{
        rotateX: interactive ? rx : 0,
        rotateY: interactive ? ry : 0,
        transformPerspective: 1400,
        transformStyle: 'preserve-3d',
      }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {/* Ambient shadow beneath the stepped base */}
      <div
        aria-hidden
        className="absolute left-1/2 -translate-x-1/2 bottom-[-1%] w-[78%] h-[38px] rounded-[50%]"
        style={{
          background:
            'radial-gradient(ellipse at center, rgba(12,10,8,0.42) 0%, rgba(12,10,8,0.14) 55%, transparent 78%)',
          filter: 'blur(10px)',
        }}
      />

      <svg
        viewBox="0 0 320 480"
        className="relative w-full h-auto drop-shadow-[0_40px_80px_rgba(12,10,8,0.35)]"
        style={{ overflow: 'visible' }}
      >
        <defs>
          {/* Frosted glass body */}
          <linearGradient id="glassFront" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#B8C4CE" stopOpacity="0.78" />
            <stop offset="10%" stopColor="#E3E8EC" stopOpacity="0.92" />
            <stop offset="45%" stopColor="#F1F2F1" stopOpacity="0.95" />
            <stop offset="72%" stopColor="#D8DCDD" stopOpacity="0.88" />
            <stop offset="100%" stopColor="#A8B2BB" stopOpacity="0.75" />
          </linearGradient>
          <linearGradient id="glassSide" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#8C97A1" stopOpacity="0.85" />
            <stop offset="60%" stopColor="#BCC3C8" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#707C87" stopOpacity="0.85" />
          </linearGradient>
          <linearGradient id="glassHi" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.9" />
            <stop offset="40%" stopColor="#FFFFFF" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
          </linearGradient>

          {/* Gold gradients */}
          <linearGradient id="goldCap" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F1D999" />
            <stop offset="35%" stopColor="#C9A658" />
            <stop offset="68%" stopColor="#8A6E2F" />
            <stop offset="100%" stopColor="#D9BA74" />
          </linearGradient>
          <linearGradient id="goldCapFace" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#E9D0A0" />
            <stop offset="50%" stopColor="#C9A658" />
            <stop offset="100%" stopColor="#9E7E36" />
          </linearGradient>
          <linearGradient id="goldCapTop" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#9E7E36" />
            <stop offset="50%" stopColor="#F1D999" />
            <stop offset="100%" stopColor="#9E7E36" />
          </linearGradient>
          <linearGradient id="goldNeck" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F1D999" />
            <stop offset="45%" stopColor="#C9A658" />
            <stop offset="100%" stopColor="#7F6429" />
          </linearGradient>
          <linearGradient id="plaqueFrame" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F1D999" />
            <stop offset="50%" stopColor="#C9A658" />
            <stop offset="100%" stopColor="#8A6E2F" />
          </linearGradient>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#2E4169" />
            <stop offset="55%" stopColor="#1C2A4C" />
            <stop offset="100%" stopColor="#0F1730" />
          </linearGradient>

          {/* Enamel flesh & details */}
          <linearGradient id="flesh" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F0C5A5" />
            <stop offset="100%" stopColor="#C78E69" />
          </linearGradient>
          <linearGradient id="toga" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F5ECD9" />
            <stop offset="100%" stopColor="#B9C8D4" />
          </linearGradient>
          <linearGradient id="boltGold" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFF3BD" />
            <stop offset="50%" stopColor="#F1D27A" />
            <stop offset="100%" stopColor="#C98A2A" />
          </linearGradient>
          <radialGradient id="crystal" cx="0.3" cy="0.3" r="0.8">
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="40%" stopColor="#F1D999" />
            <stop offset="100%" stopColor="#8A6E2F" />
          </radialGradient>

          <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="0.35" />
          </filter>
        </defs>

        {/* ---------------- BOTTLE BODY ---------------- */}
        <motion.g
          initial={{ y: 18, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 1.2, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
        >
          {/* Hexagonal side facets (left + right) */}
          <path
            d="M 52 152 L 34 176 L 34 388 L 52 412 Z"
            fill="url(#glassSide)"
            stroke="#6E7C87"
            strokeWidth="0.8"
          />
          <path
            d="M 268 152 L 286 176 L 286 388 L 268 412 Z"
            fill="url(#glassSide)"
            stroke="#6E7C87"
            strokeWidth="0.8"
            transform="translate(320, 0) scale(-1, 1)"
            opacity="0.9"
          />

          {/* Front face (main body) */}
          <path
            d="M 52 152 L 268 152 L 268 412 L 52 412 Z"
            fill="url(#glassFront)"
            stroke="#8A9399"
            strokeWidth="0.9"
          />

          {/* Front-face inner liquid tint */}
          <rect
            x="62"
            y="168"
            width="196"
            height="228"
            rx="2"
            fill="#D8E0E8"
            opacity="0.35"
          />

          {/* Glass highlight streak */}
          <rect
            x="62"
            y="158"
            width="26"
            height="240"
            fill="url(#glassHi)"
            opacity="0.9"
          />
          <rect
            x="240"
            y="180"
            width="14"
            height="200"
            fill="url(#glassHi)"
            opacity="0.55"
          />

          {/* Stepped base with ARGOS embossed */}
          <path
            d="M 46 412 L 274 412 L 280 426 L 40 426 Z"
            fill="url(#glassFront)"
            stroke="#8A9399"
            strokeWidth="0.9"
          />
          <path
            d="M 40 426 L 280 426 L 272 440 L 48 440 Z"
            fill="#C4CCD2"
            opacity="0.95"
            stroke="#8A9399"
            strokeWidth="0.8"
          />
          <text
            x="160"
            y="437"
            textAnchor="middle"
            fontFamily="Cormorant Garamond, serif"
            fontWeight="500"
            fontSize="14"
            fill="#6B7680"
            letterSpacing="6"
            opacity="0.75"
          >
            ARGOS
          </text>
        </motion.g>

        {/* ---------------- GOLD NECK RING ---------------- */}
        <motion.g
          initial={{ y: 6, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.12 }}
        >
          <path
            d="M 112 136 Q 112 128 120 128 L 200 128 Q 208 128 208 136 L 208 152 L 112 152 Z"
            fill="url(#goldNeck)"
            stroke="#6E5724"
            strokeWidth="0.9"
          />
          {/* engraved ARGOS around the ring */}
          <text
            x="160"
            y="145"
            textAnchor="middle"
            fontFamily="Cormorant Garamond, serif"
            fontSize="10"
            letterSpacing="4"
            fill="#4A3A14"
            opacity="0.9"
          >
            ARGOS · ARGOS
          </text>
          {/* ring shadow under */}
          <rect x="112" y="150" width="96" height="3" fill="#6E5724" opacity="0.45" />
        </motion.g>

        {/* ---------------- HEXAGONAL GOLD CAP ---------------- */}
        <motion.g
          initial={{ y: -28, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.05 }}
        >
          {/* Left facet */}
          <path
            d="M 96 46 L 110 30 L 110 118 L 96 128 Z"
            fill="url(#goldCap)"
            stroke="#6E5724"
            strokeWidth="0.8"
            opacity="0.92"
          />
          {/* Right facet */}
          <path
            d="M 224 46 L 210 30 L 210 118 L 224 128 Z"
            fill="url(#goldCap)"
            stroke="#6E5724"
            strokeWidth="0.8"
            opacity="0.92"
            transform="translate(320, 0) scale(-1, 1)"
          />
          {/* Top bevel band */}
          <path
            d="M 110 30 L 210 30 L 224 46 L 96 46 Z"
            fill="url(#goldCapTop)"
            stroke="#6E5724"
            strokeWidth="0.8"
          />
          {/* Front face of cap — flat matte area */}
          <path
            d="M 110 30 L 210 30 L 210 118 L 110 118 Z"
            fill="url(#goldCapFace)"
            stroke="#7F6429"
            strokeWidth="0.9"
          />
          {/* subtle face highlight */}
          <rect x="112" y="30" width="28" height="88" fill="#F5DEA1" opacity="0.4" />
          <rect x="118" y="30" width="4" height="88" fill="#FFFFFF" opacity="0.35" />

          {/* Cap top edge highlight */}
          <rect x="110" y="30" width="100" height="4" fill="#FBE9B7" opacity="0.85" />

          {/* Cap bottom shadow to neck */}
          <rect x="110" y="114" width="100" height="4" fill="#6E5724" opacity="0.45" />
        </motion.g>

        {/* ---------------- ENAMEL PLAQUE ---------------- */}
        <motion.g
          initial={{ scale: 0.86, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
        >
          {/* Outer gold frame */}
          <rect
            x="82"
            y="186"
            width="156"
            height="208"
            rx="3"
            fill="url(#plaqueFrame)"
            stroke="#6E5724"
            strokeWidth="1"
          />
          {/* Inner dark well */}
          <rect
            x="88"
            y="192"
            width="144"
            height="196"
            rx="2"
            fill="url(#sky)"
          />

          {/* Title bar "JUPITER'S LIGHTNING" */}
          <rect x="92" y="196" width="136" height="16" rx="1" fill="#241A0B" opacity="0.35" />
          <text
            x="160"
            y="207"
            textAnchor="middle"
            fontFamily="Cormorant Garamond, serif"
            fontSize="7"
            fontWeight="600"
            fill="#E9D4A2"
            letterSpacing="2.2"
          >
            JUPITER'S LIGHTNING
          </text>
          {/* Crystal corners on title bar */}
          <motion.circle
            cx="98"
            cy="204"
            r="2.2"
            fill="url(#crystal)"
            animate={{ opacity: [0.5, 1, 0.5], scale: [0.9, 1.15, 0.9] }}
            transition={{ duration: 3, repeat: Infinity }}
          />
          <motion.circle
            cx="222"
            cy="204"
            r="2.2"
            fill="url(#crystal)"
            animate={{ opacity: [0.5, 1, 0.5], scale: [0.9, 1.15, 0.9] }}
            transition={{ duration: 3, repeat: Infinity, delay: 1.5 }}
          />

          {/* Cloud silhouette behind Jupiter */}
          <ellipse cx="160" cy="235" rx="64" ry="20" fill="#3A4E77" opacity="0.7" />
          <ellipse cx="195" cy="244" rx="30" ry="12" fill="#5A6F9B" opacity="0.45" />

          {/* ---------- JUPITER FIGURE ---------- */}
          {/* Throne suggestion behind */}
          <path
            d="M 128 250 L 128 340 L 130 350 L 135 350 L 135 252 Z"
            fill="#C98A2A"
            opacity="0.85"
          />
          <path
            d="M 190 250 L 190 340 L 188 350 L 183 350 L 183 252 Z"
            fill="#C98A2A"
            opacity="0.85"
          />
          {/* Red drape accent on throne */}
          <path
            d="M 180 252 L 198 252 L 196 340 L 178 340 Z"
            fill="#8C2220"
            opacity="0.75"
          />

          {/* Toga — draped white cloth */}
          <path
            d="M 135 295 Q 160 310 200 298 L 210 345 Q 170 360 125 355 Z"
            fill="url(#toga)"
            stroke="#8A9CB2"
            strokeWidth="0.4"
          />
          {/* Toga shading folds */}
          <path
            d="M 150 310 Q 160 320 175 312"
            stroke="#8A9CB2"
            strokeWidth="0.6"
            fill="none"
            opacity="0.4"
          />
          <path
            d="M 138 340 Q 150 350 180 348"
            stroke="#8A9CB2"
            strokeWidth="0.6"
            fill="none"
            opacity="0.4"
          />

          {/* Jupiter torso */}
          <path
            d="M 140 245 Q 140 238 148 238 L 178 238 Q 186 238 188 248 L 190 295 Q 170 305 140 300 Z"
            fill="url(#flesh)"
            stroke="#7A4E2E"
            strokeWidth="0.4"
          />
          {/* Chest/pecs shading */}
          <path
            d="M 148 260 Q 160 270 180 260"
            stroke="#9A6340"
            strokeWidth="0.6"
            fill="none"
            opacity="0.6"
          />
          <path
            d="M 148 275 Q 160 282 180 275"
            stroke="#9A6340"
            strokeWidth="0.5"
            fill="none"
            opacity="0.5"
          />

          {/* Head */}
          <circle cx="164" cy="230" r="10" fill="url(#flesh)" stroke="#7A4E2E" strokeWidth="0.4" />
          {/* Beard */}
          <path
            d="M 156 232 Q 164 250 172 232"
            fill="#E5DACB"
            stroke="#8A7A6B"
            strokeWidth="0.4"
          />
          {/* Hair/crown */}
          <path
            d="M 154 222 Q 164 214 174 222 L 174 226 Q 164 220 154 226 Z"
            fill="#E5DACB"
            stroke="#8A7A6B"
            strokeWidth="0.4"
          />

          {/* Right arm raised with thunderbolts */}
          <path
            d="M 185 244 Q 200 240 208 228 L 212 218 Q 208 216 204 220 L 198 232 Q 192 234 187 238 Z"
            fill="url(#flesh)"
            stroke="#7A4E2E"
            strokeWidth="0.3"
          />
          {/* Thunderbolts radiating upward */}
          <g style={{ filter: 'drop-shadow(0 0 4px rgba(241,217,122,0.8))' }}>
            <path
              d="M 207 218 L 204 198 L 208 200 L 206 186 L 212 205 L 208 205 L 213 218 Z"
              fill="url(#boltGold)"
              stroke="#8A6E2F"
              strokeWidth="0.3"
            />
            <path
              d="M 213 220 L 218 200 L 222 206 L 224 190 L 226 212 L 222 212 L 220 222 Z"
              fill="url(#boltGold)"
              stroke="#8A6E2F"
              strokeWidth="0.3"
              opacity="0.9"
            />
            <path
              d="M 200 222 L 196 205 L 200 207 L 198 195 L 202 210 L 199 212 L 202 220 Z"
              fill="url(#boltGold)"
              stroke="#8A6E2F"
              strokeWidth="0.3"
              opacity="0.85"
            />
          </g>

          {/* Left arm resting */}
          <path
            d="M 140 248 Q 128 268 128 290 Q 132 292 136 286 L 140 268 Q 144 256 142 250 Z"
            fill="url(#flesh)"
            stroke="#7A4E2E"
            strokeWidth="0.3"
          />
          {/* Leg peeking through toga */}
          <path
            d="M 162 340 Q 160 360 160 372 L 170 372 Q 172 358 172 340 Z"
            fill="url(#flesh)"
            stroke="#7A4E2E"
            strokeWidth="0.3"
            opacity="0.9"
          />

          {/* ---------- EAGLE ---------- */}
          <g>
            {/* left wing */}
            <path
              d="M 100 350 Q 118 320 140 334 Q 138 346 124 360 Q 110 370 100 362 Z"
              fill="#1A1411"
              stroke="#E9D4A2"
              strokeWidth="0.3"
            />
            {/* right wing */}
            <path
              d="M 184 340 Q 208 326 224 352 Q 218 368 200 368 Q 188 362 180 350 Z"
              fill="#1A1411"
              stroke="#E9D4A2"
              strokeWidth="0.3"
            />
            {/* body */}
            <path
              d="M 138 356 Q 160 348 182 356 Q 180 374 160 380 Q 140 376 138 360 Z"
              fill="#2C1E14"
              stroke="#E9D4A2"
              strokeWidth="0.3"
            />
            {/* head */}
            <circle cx="148" cy="354" r="5" fill="#F3EAD8" stroke="#8A7A6B" strokeWidth="0.3" />
            {/* beak */}
            <path d="M 143 355 L 138 356 L 143 358 Z" fill="#C9A658" />
            {/* eye */}
            <circle cx="148" cy="353" r="0.9" fill="#0C0A08" />
            {/* feather strokes on wings */}
            <path d="M 108 350 L 128 358 M 110 358 L 130 360 M 195 350 L 212 358 M 195 360 L 215 362" stroke="#3A2C18" strokeWidth="0.35" opacity="0.7" />
          </g>

          {/* ---------- WAVES / SEA AT BASE ---------- */}
          <g opacity="0.9">
            <path
              d="M 88 380 Q 108 370 128 378 Q 148 386 168 378 Q 188 370 208 378 Q 228 386 232 382 L 232 388 L 88 388 Z"
              fill="#1A2C53"
            />
            <path
              d="M 88 378 Q 108 368 128 376"
              stroke="#C9A658"
              strokeWidth="0.6"
              fill="none"
              opacity="0.7"
            />
            <path
              d="M 128 376 Q 148 384 168 376"
              stroke="#C9A658"
              strokeWidth="0.6"
              fill="none"
              opacity="0.6"
            />
            <path
              d="M 168 376 Q 188 368 208 376"
              stroke="#C9A658"
              strokeWidth="0.6"
              fill="none"
              opacity="0.6"
            />
          </g>
        </motion.g>
      </svg>

      {/* Reflection sweep on hover */}
      <div className="bottle-reflect rounded-xl" />
    </motion.div>
  );
}
