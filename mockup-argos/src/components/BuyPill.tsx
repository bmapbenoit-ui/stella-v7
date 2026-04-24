import { motion, useMotionValueEvent, useScroll } from 'framer-motion';
import { useState } from 'react';
import { ArrowRight, ShoppingBag } from 'lucide-react';
import { product } from '../data/product';

/**
 * Floating commerce pill — emerges after the hero, stays pinned bottom-right.
 * Expands on hover to reveal price-per-ml and installment info. Keeps the
 * page feeling editorial while still providing the e-commerce hook Argos
 * cares about.
 */
export function BuyPill() {
  const { scrollY } = useScroll();
  const [visible, setVisible] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useMotionValueEvent(scrollY, 'change', (v) => setVisible(v > 600));

  return (
    <motion.div
      initial={{ y: 120, opacity: 0 }}
      animate={{ y: visible ? 0 : 120, opacity: visible ? 1 : 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className="fixed bottom-4 md:bottom-6 right-4 md:right-6 z-50"
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      <motion.a
        href="#panier"
        layout
        className="cursor-hover group flex items-center gap-3 md:gap-4 rounded-full pl-4 pr-2 md:pl-6 md:pr-3 py-2 md:py-2.5 bg-argos-ink text-argos-goldPale border border-argos-goldPale/25 shadow-[0_12px_32px_rgba(12,10,8,0.45),0_0_24px_rgba(201,166,88,0.25)] backdrop-blur-md"
        style={{ isolation: 'isolate' }}
      >
        {/* Gold glow behind */}
        <span
          aria-hidden
          className="absolute inset-0 rounded-full pointer-events-none opacity-60"
          style={{
            background:
              'radial-gradient(ellipse at center, rgba(201,166,88,0.18) 0%, transparent 70%)',
          }}
        />

        <ShoppingBag size={15} className="text-argos-goldPale shrink-0" strokeWidth={1.5} />

        <div className="flex flex-col leading-tight">
          <span className="text-[9.5px] tracking-[0.35em] uppercase text-argos-goldPale/60">
            Extrait · 30 ml
          </span>
          <span className="font-display text-[17px] md:text-[18px] text-argos-goldPale tracking-tight">
            {product.price} <span className="text-argos-goldPale/70">€</span>
          </span>
        </div>

        <motion.div
          layout
          initial={false}
          animate={{
            width: expanded ? 'auto' : 0,
            opacity: expanded ? 1 : 0,
            marginLeft: expanded ? 4 : 0,
          }}
          transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
          className="overflow-hidden hidden md:flex items-center"
        >
          <span className="text-[10.5px] tracking-[0.22em] text-argos-goldPale/70 whitespace-nowrap pr-3 border-r border-argos-goldPale/20">
            ou 3× {(product.price / 3).toFixed(2).replace('.', ',')} €
          </span>
          <span className="pl-3 text-[10.5px] tracking-[0.22em] text-argos-goldPale/80 whitespace-nowrap">
            Try Me 2 ml · 18 €
          </span>
        </motion.div>

        <span className="ml-1 md:ml-2 w-9 h-9 md:w-10 md:h-10 rounded-full bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep text-argos-ink flex items-center justify-center shrink-0 group-hover:rotate-[-12deg] transition-transform">
          <ArrowRight size={16} strokeWidth={1.8} />
        </span>
      </motion.a>
    </motion.div>
  );
}
