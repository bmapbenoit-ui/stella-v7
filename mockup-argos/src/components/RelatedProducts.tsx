import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';

const RELATED = [
  {
    brand: 'ARGOS',
    name: "Neptune's Trident",
    price: 195,
    tint: { a: '#1F3A4F', b: '#0C1A26' },
    accent: '#8BA4B8',
  },
  {
    brand: 'ARGOS',
    name: 'Perseus Triumphant',
    price: 210,
    tint: { a: '#2F1B10', b: '#0F0906' },
    accent: '#C9A658',
  },
  {
    brand: 'ARGOS',
    name: 'Vulcan’s Revenge',
    price: 220,
    tint: { a: '#4A1410', b: '#1A0604' },
    accent: '#E5744C',
  },
  {
    brand: 'ARGOS',
    name: 'Danaë · Golden Rain',
    price: 205,
    tint: { a: '#3A2A0E', b: '#140C03' },
    accent: '#E9D4A2',
  },
];

export function RelatedProducts() {
  return (
    <div>
      <div className="flex items-end justify-between mb-6">
        <div>
          <p className="text-[10px] tracking-[0.3em] uppercase text-argos-gold/90 mb-1">
            La Maison Argos
          </p>
          <h2 className="font-display text-3xl md:text-4xl text-argos-ink">
            Vous aimerez peut-être aussi
          </h2>
        </div>
        <a
          href="#"
          className="hidden md:inline-flex items-center gap-2 text-[11px] tracking-[0.25em] uppercase text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover"
        >
          Voir la maison <ArrowRight size={14} />
        </a>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {RELATED.map((r, i) => (
          <motion.a
            key={r.name}
            href="#"
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ delay: i * 0.08, duration: 0.55 }}
            whileHover={{ y: -6 }}
            className="group block cursor-hover"
          >
            <div
              className="relative aspect-[4/5] rounded-[2px] overflow-hidden"
              style={{
                background: `radial-gradient(ellipse at 40% 30%, ${r.tint.a} 0%, ${r.tint.b} 85%)`,
              }}
            >
              <div
                aria-hidden
                className="absolute inset-0 opacity-70"
                style={{
                  background:
                    'repeating-linear-gradient(45deg, rgba(255,255,255,0.03), rgba(255,255,255,0.03) 2px, transparent 2px, transparent 12px)',
                }}
              />
              {/* Bottle silhouette */}
              <div className="absolute inset-0 flex items-center justify-center">
                <svg viewBox="0 0 100 140" className="w-[46%] h-auto drop-shadow-2xl transition-transform duration-500 group-hover:scale-105">
                  <defs>
                    <linearGradient id={`g-${i}`} x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor={r.accent} stopOpacity="0.9" />
                      <stop offset="100%" stopColor={r.accent} stopOpacity="0.4" />
                    </linearGradient>
                  </defs>
                  <rect x="30" y="38" width="40" height="80" rx="4" fill={`url(#g-${i})`} opacity="0.92" />
                  <rect x="36" y="22" width="28" height="16" rx="2" fill={r.accent} />
                  <rect x="38" y="16" width="24" height="6" rx="1" fill={r.accent} opacity="0.8" />
                  <rect x="40" y="58" width="20" height="36" rx="2" fill="#0C0A08" opacity="0.5" />
                </svg>
              </div>
              <div
                className="absolute inset-x-0 bottom-0 h-20"
                style={{
                  background: `linear-gradient(180deg, transparent, ${r.tint.b})`,
                }}
              />
              <div
                aria-hidden
                className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
                style={{
                  background: `radial-gradient(circle at 50% 50%, ${r.accent}33 0%, transparent 60%)`,
                }}
              />
            </div>
            <div className="mt-3 px-1">
              <p className="text-[9.5px] tracking-[0.32em] uppercase text-argos-bronze">
                {r.brand}
              </p>
              <p className="font-display text-lg text-argos-ink leading-tight mt-0.5">
                {r.name}
              </p>
              <p className="text-[12px] text-argos-ink/80 mt-1">
                À partir de {r.price} €
              </p>
            </div>
          </motion.a>
        ))}
      </div>
    </div>
  );
}
