import { motion } from 'framer-motion';
import { ChevronRight, Eye } from 'lucide-react';
import { product } from '../data/product';
import { useCounter } from '../hooks/useCounter';

export function ProductIdentification() {
  const count = useCounter(product.interestedCount, 2200);
  return (
    <div className="space-y-4">
      {/* Breadcrumb */}
      <motion.nav
        initial={{ opacity: 0, x: -10 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.1, duration: 0.5 }}
        className="flex items-center gap-1.5 text-[10.5px] tracking-[0.15em] uppercase text-argos-bronze/80"
      >
        <a href="#" className="hover:text-argos-gold cursor-hover">Accueil</a>
        <ChevronRight size={12} />
        <a href="#" className="hover:text-argos-gold cursor-hover">Parfums</a>
        <ChevronRight size={12} />
        <a href="#" className="hover:text-argos-gold cursor-hover">Boisé Aromatique</a>
        <ChevronRight size={12} />
        <span className="text-argos-ink/80">{product.name}</span>
      </motion.nav>

      {/* Brand */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.25, duration: 0.6 }}
        className="flex items-center gap-3"
      >
        <span className="h-[1px] w-8 bg-gold-gradient" />
        <span className="font-display text-[22px] tracking-[0.55em] text-argos-ink">
          {product.brand}
        </span>
        <span className="h-[1px] flex-1 bg-argos-gold/20 max-w-[140px]" />
      </motion.div>
      <p className="text-[11px] tracking-[0.25em] uppercase text-argos-bronze -mt-2">
        {product.brandTagline}
      </p>

      {/* Title */}
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="font-display text-[40px] md:text-[56px] leading-[1] text-argos-ink"
      >
        {product.name}
      </motion.h1>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.5 }}
        className="flex items-center gap-3 flex-wrap"
      >
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-argos-ink text-argos-goldPale text-[10.5px] tracking-[0.2em] uppercase">
          {product.concentration}
        </span>
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-argos-gold text-argos-ink text-[10.5px] tracking-[0.2em] uppercase">
          {product.volume}
        </span>
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gold-gradient text-argos-ink text-[10.5px] tracking-[0.2em] uppercase font-medium">
          {product.series}
        </span>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.85, duration: 0.5 }}
        className="flex items-center gap-2 text-[12px] text-argos-bronze/90"
      >
        <Eye size={14} className="text-argos-gold" />
        <span>
          <b className="text-argos-ink tabular-nums">{count}</b> personnes sont intéressées par
          ce produit
        </span>
      </motion.div>
    </div>
  );
}
