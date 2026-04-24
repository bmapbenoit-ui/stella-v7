import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { product } from '../data/product';
import { ActWrapper } from './ActWrapper';
import { Flame, Wind, Clock } from 'lucide-react';

/**
 * Acte III — Le Temple.
 * Les chiffres deviennent monument. Intensité, sillage, tenue, prix.
 * Typography à grande échelle, chaque donnée respire.
 */
export function ActIII() {
  return (
    <ActWrapper
      id="acte-iii"
      label="III"
      title="Le Temple"
      subtitle="Quatre colonnes pour un parfum d'autorité. La lecture chiffrée d'une sensation — seule mesure qu'une boutique physique ne saura jamais offrir."
      variant="dark"
    >
      <div className="grid md:grid-cols-2 gap-14 md:gap-20">
        <Column
          icon={<Flame size={18} strokeWidth={1.3} />}
          label="Intensité"
          value="IV"
          caption={`${product.profile.intensity} sur 5 — dense, non envahissant`}
          bar={product.profile.intensity / 5}
        />
        <Column
          icon={<Wind size={18} strokeWidth={1.3} />}
          label="Sillage"
          value={product.profile.sillageLabel}
          caption="2 à 3 mètres de nuage doré derrière vous"
          bar={product.profile.sillage / 5}
        />
        <Column
          icon={<Clock size={18} strokeWidth={1.3} />}
          label="Tenue"
          value={`${product.profile.longevity}+`}
          valueSuffix="heures"
          caption="De l'aube au dîner, sans retouche"
          bar={product.profile.longevity / product.profile.longevityMax}
        />
        <Column
          icon={<PriceRune />}
          label="Extrait · 30 ml"
          value="210"
          valueSuffix="€"
          caption={`Soit ${product.pricePerMl.toFixed(2).replace('.', ',')} €/ml — prix public conseillé, TTC.`}
          bar={1}
          isPrice
        />
      </div>
    </ActWrapper>
  );
}

function PriceRune() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.4">
      <circle cx="12" cy="12" r="9" />
      <path d="M 8 10 h 8 M 8 14 h 8 M 12 6 v 12" />
    </svg>
  );
}

interface ColumnProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueSuffix?: string;
  caption: string;
  bar: number;
  isPrice?: boolean;
}

function Column({ icon, label, value, valueSuffix, caption, bar, isPrice }: ColumnProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      className="relative border-t border-argos-goldPale/20 pt-8 md:pt-10"
    >
      <div className="flex items-center gap-3 text-argos-goldPale/80 text-[10.5px] tracking-[0.4em] uppercase">
        <span className="text-argos-goldPale">{icon}</span>
        {label}
      </div>
      <div className="mt-6 md:mt-10 flex items-end gap-4 md:gap-6">
        <motion.span
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className={`font-display leading-[0.85] tracking-[-0.02em] ${
            isPrice ? 'text-transparent bg-clip-text bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep' : 'text-argos-ivory'
          }`}
          style={{ fontSize: 'clamp(92px, 14vw, 240px)' }}
        >
          {value}
        </motion.span>
        {valueSuffix && (
          <motion.span
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="font-display italic text-argos-goldPale/80 pb-4 md:pb-8"
            style={{ fontSize: 'clamp(28px, 3vw, 46px)' }}
          >
            {valueSuffix}
          </motion.span>
        )}
      </div>
      <motion.div
        className="mt-6 h-[2px] w-full bg-argos-goldPale/10 overflow-hidden"
      >
        <motion.span
          initial={{ scaleX: 0 }}
          animate={inView ? { scaleX: bar } : {}}
          transition={{ duration: 1.4, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
          style={{ transformOrigin: 'left', display: 'block' }}
          className="h-full w-full bg-gradient-to-r from-argos-goldPale via-argos-gold to-argos-goldDeep"
        />
      </motion.div>
      <p className="mt-5 text-[12.5px] leading-relaxed text-argos-ivory/60 max-w-sm">
        {caption}
      </p>
    </motion.div>
  );
}
