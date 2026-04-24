import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { product } from '../data/product';
import { IngredientGlyph } from './IngredientGlyph';

/**
 * Acte II — La Foudre.
 * Trois chapitres plein écran — Tête, Cœur, Fond. Chaque chapitre est une
 * scène cinématographique à part entière, avec un grand glyph d'ingrédient
 * à gauche, une typographie monumentale à droite, et un parallax léger.
 */
export function ActII() {
  return (
    <section id="acte-ii" className="relative bg-argos-ivory text-argos-ink">
      <Overture />
      <Chapter
        roman="I"
        tier={product.pyramid.top}
        glyph="cardamome"
        tone="fresh"
      />
      <Chapter
        roman="II"
        tier={product.pyramid.heart}
        glyph="safran"
        tone="amber"
        reverse
      />
      <Chapter
        roman="III"
        tier={product.pyramid.base}
        glyph="cedre"
        tone="wood"
      />
    </section>
  );
}

function Overture() {
  return (
    <div className="max-w-[1400px] mx-auto px-5 md:px-12 pt-28 md:pt-40 pb-8">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-100px' }}
        transition={{ duration: 0.9 }}
      >
        <div className="flex items-center gap-4 mb-6">
          <span className="font-display italic text-2xl md:text-3xl text-argos-gold">
            Acte II
          </span>
          <motion.span
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1.4, delay: 0.2 }}
            style={{ transformOrigin: 'left' }}
            className="h-[1px] w-32 md:w-48 bg-argos-gold/40"
          />
        </div>
        <h2
          className="font-display leading-[0.95] tracking-tight text-argos-ink"
          style={{ fontSize: 'clamp(40px, 6vw, 96px)' }}
        >
          La Foudre
        </h2>
        <p className="mt-5 md:mt-7 max-w-xl text-[14px] md:text-[15px] leading-relaxed text-argos-ink/70">
          Trois étages pour une théophanie. La pyramide se lit comme une
          sculpture d'Olympie : on en fait le tour.
        </p>
      </motion.div>
    </div>
  );
}

interface ChapterProps {
  roman: string;
  tier: {
    title: string;
    hero: string;
    description: string;
    notes: string[];
  };
  glyph: 'cardamome' | 'safran' | 'cedre';
  tone: 'fresh' | 'amber' | 'wood';
  reverse?: boolean;
}

function Chapter({ roman, tier, glyph, tone, reverse }: ChapterProps) {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });
  const imgY = useTransform(scrollYProgress, [0, 1], ['8%', '-12%']);
  const titleY = useTransform(scrollYProgress, [0, 1], ['12%', '-8%']);

  const toneBg = {
    fresh: 'bg-[#E8EED3]',
    amber: 'bg-[#F4DFB2]',
    wood: 'bg-[#D8C49E]',
  }[tone];

  return (
    <div
      ref={ref}
      className={`relative py-24 md:py-36 ${reverse ? 'bg-argos-parchment/60' : ''}`}
    >
      <div className="max-w-[1500px] mx-auto px-5 md:px-12">
        <div
          className={`grid md:grid-cols-12 gap-8 md:gap-16 items-center ${
            reverse ? 'md:[direction:rtl]' : ''
          }`}
        >
          {/* Glyph */}
          <motion.div
            style={{ y: imgY }}
            className={`md:col-span-6 relative ${reverse ? 'md:[direction:ltr]' : ''}`}
          >
            <div
              className={`relative aspect-square rounded-[2px] overflow-hidden ${toneBg}`}
            >
              <div className="absolute inset-0">
                <IngredientGlyph kind={glyph} />
              </div>
              <div
                className="absolute inset-0"
                style={{
                  background:
                    'linear-gradient(135deg, transparent 50%, rgba(12,10,8,0.25) 100%)',
                }}
              />
              {/* Frame marks */}
              <span className="absolute top-4 left-4 text-[10px] tracking-[0.4em] uppercase text-argos-ink/70">
                Plate · {roman}
              </span>
              <span className="absolute top-4 right-4 text-[10px] tracking-[0.4em] uppercase text-argos-ink/70">
                {tier.title}
              </span>
              <span className="absolute bottom-4 left-4 right-4 flex items-end justify-between">
                <span className="font-display italic text-argos-ink/80 text-lg">
                  {glyph}
                </span>
                <span className="text-[10px] tracking-[0.3em] uppercase text-argos-ink/60">
                  Jupiter's Lightning · 2026
                </span>
              </span>
            </div>
          </motion.div>

          {/* Text */}
          <div
            className={`md:col-span-6 ${reverse ? 'md:[direction:ltr]' : ''} space-y-6 md:space-y-8`}
          >
            <motion.div
              style={{ y: titleY }}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.9 }}
            >
              <p className="text-[10px] tracking-[0.5em] uppercase text-argos-gold mb-3">
                Chapitre {roman} · {tier.title}
              </p>
              <h3
                className="font-display leading-[0.92] tracking-tight text-argos-ink"
                style={{ fontSize: 'clamp(48px, 7vw, 128px)' }}
              >
                <span className="italic">{tier.hero.split(' ')[0]}</span>
                {tier.hero.includes(' ') && (
                  <span className="block text-argos-ink/75">
                    {tier.hero.split(' ').slice(1).join(' ')}
                  </span>
                )}
              </h3>
            </motion.div>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="text-[16px] md:text-[18px] leading-[1.7] text-argos-ink/80 font-light italic max-w-[52ch]"
            >
              « {tier.description} »
            </motion.p>
            <motion.div
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.8, delay: 0.35 }}
              className="flex flex-wrap gap-2"
            >
              {tier.notes.map((n) => (
                <span
                  key={n}
                  className="px-3.5 py-1.5 rounded-full text-[11px] tracking-[0.18em] uppercase border border-argos-gold/40 text-argos-ink/80 hover:bg-argos-ink hover:text-argos-goldPale hover:border-argos-ink transition-all cursor-hover"
                >
                  {n}
                </span>
              ))}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
