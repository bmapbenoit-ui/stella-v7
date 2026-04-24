import { Menu, Search, ShoppingBag, User, Heart } from 'lucide-react';
import { motion, useMotionValueEvent, useScroll } from 'framer-motion';
import { useState } from 'react';

export function Header() {
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);
  useMotionValueEvent(scrollY, 'change', (v) => setScrolled(v > 100));

  return (
    <motion.header
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      className={`sticky top-0 z-40 transition-all duration-500 ${
        scrolled
          ? 'bg-argos-ivory/70 backdrop-blur-xl backdrop-saturate-150 shadow-[0_1px_0_rgba(201,166,88,0.25)]'
          : 'bg-argos-ivory/0'
      }`}
    >
      <div className="max-w-[1400px] mx-auto px-4 md:px-8 h-16 md:h-20 flex items-center justify-between">
        <div className="flex items-center gap-5">
          <button className="text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover">
            <Menu size={22} strokeWidth={1.5} />
          </button>
          <button className="hidden md:block text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover">
            <Search size={20} strokeWidth={1.5} />
          </button>
        </div>

        <a
          href="#"
          className="flex flex-col items-center gap-0.5 cursor-hover"
          aria-label="Planetebeauty"
        >
          <span className="font-signature text-3xl md:text-4xl text-gold-foil leading-none">
            Planetebeauty
          </span>
          <span className="text-[9px] tracking-[0.6em] text-argos-bronze uppercase mt-0.5">
            Parfums de Niche
          </span>
        </a>

        <div className="flex items-center gap-4 md:gap-5">
          <button className="hidden md:block text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover">
            <Heart size={20} strokeWidth={1.5} />
          </button>
          <button className="hidden md:block text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover">
            <User size={20} strokeWidth={1.5} />
          </button>
          <button className="relative text-argos-ink/80 hover:text-argos-gold transition-colors cursor-hover">
            <ShoppingBag size={22} strokeWidth={1.5} />
            <span className="absolute -top-1.5 -right-2 w-[18px] h-[18px] rounded-full bg-argos-ink text-argos-goldPale text-[10px] font-medium flex items-center justify-center">
              2
            </span>
          </button>
        </div>
      </div>

      {/* Secondary nav when not scrolled */}
      <motion.nav
        initial={{ opacity: 0 }}
        animate={{ opacity: scrolled ? 0 : 1, height: scrolled ? 0 : 'auto' }}
        transition={{ duration: 0.35 }}
        className="overflow-hidden border-t border-argos-gold/15"
      >
        <ul className="max-w-[1400px] mx-auto px-4 md:px-8 h-10 hidden md:flex items-center justify-center gap-10 text-[11px] uppercase tracking-argos text-argos-ink/70">
          {['Nouveautés', 'Parfums de niche', 'Maisons', 'Découvrir', 'Coffrets', 'Le Journal'].map(
            (l) => (
              <li key={l}>
                <a href="#" className="hover:text-argos-gold transition-colors cursor-hover">
                  {l}
                </a>
              </li>
            )
          )}
        </ul>
      </motion.nav>
    </motion.header>
  );
}
