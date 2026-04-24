import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { ActWrapper } from './ActWrapper';
import { product } from '../data/product';
import { useCounter } from '../hooks/useCounter';
import { Eye } from 'lucide-react';

/**
 * Acte I — L'Approche.
 * Le rideau se lève. On pose le produit, le parfumeur, le nom. On raconte
 * le geste, pas le packshot.
 */
export function ActI() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start end', 'end start'],
  });
  const yShift = useTransform(scrollYProgress, [0, 1], ['0%', '-12%']);

  return (
    <div ref={ref}>
      <ActWrapper
        id="acte-i"
        label="I"
        title="L'Approche"
        subtitle="Avant que la foudre ne tombe, il y a l'instant où l'air change. L'odeur précède le bruit. C'est là qu'Argos travaille."
      >
        <div className="grid md:grid-cols-12 gap-10 md:gap-16 items-start">
          {/* Big editorial column left */}
          <motion.div style={{ y: yShift }} className="md:col-span-7 space-y-10">
            <Dropcap>
              La cinquième toile de l'Artist Series d'Argos s'ouvre sur un
              ciel d'orage. Christian Petrovich a voulu y loger une
              théophanie — l'apparition d'un dieu. Jupiter trônant, le
              foudre en main, comme l'avait sculpté Phidias à Olympie.
            </Dropcap>
            <p className="text-[15.5px] md:text-[17px] leading-[1.75] text-argos-ivory/80 font-light max-w-[62ch]">
              Le flacon, coulé à Milan, porte en plaque émaillée la
              silhouette du Maître de l'Olympe. Quatre cristaux Swarovski
              captent la lumière comme quatre éclats de foudre. La capsule
              d'or mat, ciselée à la main, pèse dans la paume le poids
              d'une médaille.
            </p>
            <FigureQuote>
              J'ai voulu capturer l'instant où la foudre fend la nuit —
              la seconde de vertige avant le tonnerre.
              <cite>— Christian Petrovich</cite>
            </FigureQuote>
          </motion.div>

          {/* Right: identification + social proof in editorial style */}
          <aside className="md:col-span-5 space-y-10 md:sticky md:top-32 self-start">
            <MetaRow label="Maison">
              Argos Fragrances · Dallas, Texas
            </MetaRow>
            <MetaRow label="Parfumeur">{product.perfumer}</MetaRow>
            <MetaRow label="Concentration">
              {product.concentration} · 35 %
            </MetaRow>
            <MetaRow label="Famille">{product.family}</MetaRow>
            <MetaRow label="Année">{product.year}</MetaRow>
            <MetaRow label="Série">{product.series}</MetaRow>
            <InterestPulse />
          </aside>
        </div>
      </ActWrapper>
    </div>
  );
}

function Dropcap({ children }: { children: React.ReactNode }) {
  // A single dropcap on the first letter for editorial feel
  const text = String(children);
  const first = text[0];
  const rest = text.slice(1);
  return (
    <p className="text-[15.5px] md:text-[18px] leading-[1.8] text-argos-ivory/90 font-light max-w-[62ch]">
      <span
        className="font-display italic text-transparent bg-clip-text bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep float-left mr-3 md:mr-4"
        style={{ fontSize: 'clamp(68px, 8vw, 120px)', lineHeight: 0.86 }}
      >
        {first}
      </span>
      {rest}
    </p>
  );
}

function FigureQuote({ children }: { children: React.ReactNode }) {
  return (
    <figure className="border-l border-argos-goldPale/40 pl-6 md:pl-10 py-2 max-w-[56ch]">
      <blockquote className="font-display italic text-2xl md:text-3xl leading-[1.25] text-argos-ivory">
        {children}
      </blockquote>
    </figure>
  );
}

function MetaRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.6 }}
      className="border-b border-argos-goldPale/15 pb-3"
    >
      <p className="text-[9.5px] tracking-[0.45em] uppercase text-argos-goldPale/70">
        {label}
      </p>
      <p className="font-display text-[22px] md:text-[26px] text-argos-ivory mt-1.5 leading-tight">
        {children}
      </p>
    </motion.div>
  );
}

function InterestPulse() {
  const count = useCounter(product.interestedCount, 2500);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 1, delay: 0.3 }}
      className="flex items-center gap-3 text-argos-goldPale/80 text-[12px] tracking-[0.18em] uppercase"
    >
      <Eye size={14} className="text-argos-goldPale" />
      <span>
        <b className="text-argos-ivory text-[16px] tracking-normal font-display not-italic tabular-nums">
          {count}
        </b>{' '}
        · personnes en attente
      </span>
    </motion.div>
  );
}
