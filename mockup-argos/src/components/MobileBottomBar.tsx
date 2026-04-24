import { motion, useScroll, useMotionValueEvent } from 'framer-motion';
import { useState } from 'react';
import { product } from '../data/product';

export function MobileBottomBar() {
  const { scrollY } = useScroll();
  const [visible, setVisible] = useState(false);
  useMotionValueEvent(scrollY, 'change', (v) => setVisible(v > 500));

  return (
    <motion.div
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: visible ? 0 : 100, opacity: visible ? 1 : 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="fixed bottom-0 inset-x-0 z-50 md:hidden px-3 pb-3"
    >
      <div className="glass-card rounded-sm flex items-center gap-3 p-2 pr-3">
        <div className="w-11 h-11 shrink-0 rounded-sm bg-argos-ivory flex items-center justify-center border border-argos-bronze/20">
          <svg viewBox="0 0 50 70" className="w-6 h-auto">
            <rect x="10" y="18" width="30" height="42" rx="3" fill="#E9D4A2" />
            <rect x="14" y="6" width="22" height="12" rx="2" fill="#C9A658" />
            <rect x="16" y="30" width="18" height="22" rx="1" fill="#0C0A08" opacity="0.7" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-[10px] tracking-[0.25em] uppercase text-argos-bronze">
            {product.brand}
          </p>
          <p className="text-[12.5px] font-medium text-argos-ink truncate leading-tight">
            {product.name}
          </p>
          <p className="text-[11px] text-argos-gold mt-0.5">{product.price} €</p>
        </div>
        <button className="rounded-[2px] bg-argos-ink text-argos-goldPale px-4 h-10 text-[10.5px] tracking-[0.2em] uppercase btn-shimmer">
          Ajouter
        </button>
      </div>
    </motion.div>
  );
}
