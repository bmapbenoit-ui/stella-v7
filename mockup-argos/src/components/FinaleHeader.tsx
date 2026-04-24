import { motion, useMotionValueEvent, useScroll } from 'framer-motion';
import { useState } from 'react';
import { Menu, ShoppingBag } from 'lucide-react';

/**
 * A minimalist overlay header that only appears after the hero.
 * Dark, editorial, matches the storm aesthetic.
 */
export function FinaleHeader() {
  const { scrollY } = useScroll();
  const [visible, setVisible] = useState(false);
  useMotionValueEvent(scrollY, 'change', (v) => setVisible(v > 500));

  return (
    <motion.header
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: visible ? 0 : -80, opacity: visible ? 1 : 0 }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
      className="fixed top-0 left-0 right-0 z-40 bg-[#0A0D16]/80 backdrop-blur-xl border-b border-argos-goldPale/10"
    >
      <div className="max-w-[1500px] mx-auto px-5 md:px-10 h-14 md:h-16 flex items-center justify-between">
        <div className="flex items-center gap-5 text-argos-ivory">
          <button className="hover:text-argos-goldPale transition-colors cursor-hover">
            <Menu size={20} strokeWidth={1.3} />
          </button>
          <nav className="hidden md:flex items-center gap-7 text-[10.5px] tracking-[0.3em] uppercase">
            <a href="#acte-i" className="hover:text-argos-goldPale cursor-hover">Acte I</a>
            <a href="#acte-ii" className="hover:text-argos-goldPale cursor-hover">Acte II</a>
            <a href="#acte-iii" className="hover:text-argos-goldPale cursor-hover">Acte III</a>
            <a href="#acte-iv" className="hover:text-argos-goldPale cursor-hover">Acte IV</a>
          </nav>
        </div>
        <a href="#top" className="cursor-hover">
          <span className="font-signature text-[22px] md:text-[26px] bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep bg-clip-text text-transparent leading-none">
            Planetebeauty
          </span>
        </a>
        <div className="flex items-center gap-4 text-argos-ivory">
          <span className="hidden md:inline text-[10.5px] tracking-[0.3em] uppercase text-argos-goldPale/70">
            Argos · Artist V
          </span>
          <button className="relative hover:text-argos-goldPale transition-colors cursor-hover">
            <ShoppingBag size={19} strokeWidth={1.3} />
            <span className="absolute -top-1.5 -right-2 w-[16px] h-[16px] rounded-full bg-argos-goldPale text-argos-ink text-[9px] font-medium flex items-center justify-center">
              2
            </span>
          </button>
        </div>
      </div>
    </motion.header>
  );
}
