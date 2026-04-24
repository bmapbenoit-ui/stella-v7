/**
 * Minimal CSS/SVG ingredient illustration (no external images needed).
 * Used for the olfactory pyramid hero visuals.
 */
export function IngredientGlyph({ kind }: { kind: 'cardamome' | 'safran' | 'cedre' }) {
  if (kind === 'cardamome') {
    return (
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <defs>
          <radialGradient id="bg1" cx="0.5" cy="0.5" r="0.6">
            <stop offset="0%" stopColor="#E8F0D0" />
            <stop offset="100%" stopColor="#A8C474" />
          </radialGradient>
          <linearGradient id="pod" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#D8E8A8" />
            <stop offset="100%" stopColor="#6F8A3C" />
          </linearGradient>
        </defs>
        <rect width="100" height="100" fill="url(#bg1)" />
        {/* pods */}
        <g>
          <ellipse cx="30" cy="58" rx="14" ry="8" fill="url(#pod)" transform="rotate(-20 30 58)" />
          <line x1="22" y1="52" x2="42" y2="66" stroke="#3E4F1F" strokeWidth="0.6" strokeDasharray="1.5 2" opacity="0.5" transform="rotate(-20 30 58)" />
          <ellipse cx="65" cy="44" rx="13" ry="7" fill="url(#pod)" transform="rotate(18 65 44)" />
          <ellipse cx="56" cy="72" rx="12" ry="7" fill="url(#pod)" transform="rotate(-8 56 72)" />
        </g>
        <circle cx="80" cy="22" r="10" fill="#fff" opacity="0.35" />
      </svg>
    );
  }
  if (kind === 'safran') {
    return (
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <defs>
          <radialGradient id="bg2" cx="0.5" cy="0.5" r="0.6">
            <stop offset="0%" stopColor="#FCE5B4" />
            <stop offset="100%" stopColor="#D97A2A" />
          </radialGradient>
          <linearGradient id="thread" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#F4A03C" />
            <stop offset="70%" stopColor="#9B2414" />
            <stop offset="100%" stopColor="#5C1209" />
          </linearGradient>
        </defs>
        <rect width="100" height="100" fill="url(#bg2)" />
        {/* saffron threads */}
        {[
          { x1: 18, y1: 40, x2: 72, y2: 60, r: -12 },
          { x1: 22, y1: 62, x2: 78, y2: 48, r: 8 },
          { x1: 30, y1: 30, x2: 68, y2: 70, r: 20 },
          { x1: 14, y1: 52, x2: 82, y2: 40, r: -4 },
        ].map((t, i) => (
          <path
            key={i}
            d={`M ${t.x1} ${t.y1} Q ${(t.x1 + t.x2) / 2} ${(t.y1 + t.y2) / 2 + (i % 2 ? -8 : 8)} ${t.x2} ${t.y2}`}
            stroke="url(#thread)"
            strokeWidth={2.4}
            fill="none"
            strokeLinecap="round"
          />
        ))}
        <circle cx="25" cy="18" r="8" fill="#fff" opacity="0.3" />
      </svg>
    );
  }
  // cedre
  return (
    <svg viewBox="0 0 100 100" className="w-full h-full">
      <defs>
        <radialGradient id="bg3" cx="0.5" cy="0.5" r="0.6">
          <stop offset="0%" stopColor="#A78554" />
          <stop offset="100%" stopColor="#3E2A14" />
        </radialGradient>
        <linearGradient id="wood" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#6F4D24" />
          <stop offset="50%" stopColor="#8F6A38" />
          <stop offset="100%" stopColor="#5C3B18" />
        </linearGradient>
      </defs>
      <rect width="100" height="100" fill="url(#bg3)" />
      {/* wood grain rings */}
      {[6, 12, 20, 28, 38, 48, 60, 76].map((r, i) => (
        <ellipse
          key={i}
          cx="50"
          cy="52"
          rx={r}
          ry={r * 0.72}
          fill="none"
          stroke="url(#wood)"
          strokeWidth="0.8"
          opacity={0.85 - i * 0.08}
        />
      ))}
      <circle cx="50" cy="52" r="3" fill="#2A1D0A" />
      <circle cx="80" cy="22" r="10" fill="#fff" opacity="0.18" />
    </svg>
  );
}
