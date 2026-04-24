import { motion, useScroll } from 'framer-motion';
import { useLenis } from './hooks/useLenis';
import { CustomCursor } from './components/CustomCursor';
import { HeroOverture } from './components/HeroOverture';
import { ActI } from './components/ActI';
import { ActII } from './components/ActII';
import { ActIII } from './components/ActIII';
import { ActIV } from './components/ActIV';
import { Finale } from './components/Finale';
import { FinaleHeader } from './components/FinaleHeader';
import { BuyPill } from './components/BuyPill';

export default function App() {
  useLenis();
  const { scrollYProgress } = useScroll();

  return (
    <div className="relative bg-[#07080E] text-argos-ivory">
      <CustomCursor />

      {/* Gold reading-progress ribbon */}
      <motion.div
        aria-hidden
        className="fixed top-0 left-0 right-0 h-[2px] z-[60] origin-left"
        style={{
          scaleX: scrollYProgress,
          background:
            'linear-gradient(90deg, #F1D999 0%, #C9A658 50%, #8A6E2F 100%)',
        }}
      />

      <FinaleHeader />

      <main id="top">
        <HeroOverture />
        <ActI />
        <ActII />
        <ActIII />
        <ActIV />
        <Finale />
      </main>

      <BuyPill />
    </div>
  );
}
