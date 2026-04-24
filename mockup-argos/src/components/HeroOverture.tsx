import { motion, useScroll, useTransform } from 'framer-motion';
import { useRef } from 'react';
import { Bottle } from './Bottle';
import { StormSky } from './StormSky';
import { LightningFlash } from './LightningFlash';
import { ChevronDown } from 'lucide-react';

export function HeroOverture() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  });
  const bottleY = useTransform(scrollYProgress, [0, 1], ['0%', '-18%']);
  const titleY = useTransform(scrollYProgress, [0, 1], ['0%', '-40%']);
  const skyScale = useTransform(scrollYProgress, [0, 1], [1, 1.15]);
  const fade = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <section
      ref={ref}
      className="relative h-[100svh] min-h-[720px] w-full overflow-hidden bg-[#07080E]"
    >
      <motion.div style={{ scale: skyScale }} className="absolute inset-0">
        <StormSky />
      </motion.div>

      <LightningFlash />

      {/* Wordmark tiny on top */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 0.2 }}
        className="absolute top-6 md:top-10 left-0 right-0 z-30 flex items-center justify-between px-5 md:px-10 text-argos-goldPale"
      >
        <span className="text-[10px] tracking-[0.4em] uppercase opacity-70">
          Planetebeauty
        </span>
        <span className="text-[10px] tracking-[0.5em] uppercase opacity-80">
          A Film for Argos
        </span>
        <span className="text-[10px] tracking-[0.4em] uppercase opacity-70">
          MMXXVI
        </span>
      </motion.div>

      {/* Editorial title LEFT + bottle CENTER */}
      <motion.div
        style={{ y: titleY, opacity: fade }}
        className="absolute inset-0 z-20 flex items-center"
      >
        <div className="relative w-full max-w-[1600px] mx-auto px-5 md:px-12 flex flex-col md:flex-row items-center justify-between gap-10">
          {/* Left: editorial title */}
          <div className="text-argos-ivory w-full md:w-[46%] text-center md:text-left">
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 0.8, y: 0 }}
              transition={{ duration: 1, delay: 0.6 }}
              className="text-[11px] md:text-[12px] tracking-[0.55em] uppercase text-argos-goldPale/85 mb-4 md:mb-6"
            >
              — Argos · Artist Series · V
            </motion.p>
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 0.9, ease: [0.22, 1, 0.36, 1] }}
              className="font-display leading-[0.88] tracking-[-0.015em]"
              style={{ fontSize: 'clamp(56px, 11vw, 168px)' }}
            >
              <span className="block italic text-transparent bg-clip-text bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep">
                Jupiter's
              </span>
              <span className="block text-argos-ivory">Lightning.</span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1.2, delay: 1.6 }}
              className="mt-5 md:mt-8 text-[13px] md:text-[15px] leading-relaxed text-argos-ivory/75 max-w-[460px] mx-auto md:mx-0 font-light"
            >
              Un extrait de parfum né de la seconde de vertige où l'air se
              charge d'ozone et d'or — avant même le tonnerre.
            </motion.p>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 1, delay: 2 }}
              className="mt-6 md:mt-9 flex items-center gap-4 justify-center md:justify-start"
            >
              <span className="h-[1px] w-10 bg-argos-goldPale/60" />
              <span className="text-[10px] tracking-[0.45em] uppercase text-argos-goldPale/80">
                Christian Petrovich · Parfumeur
              </span>
            </motion.div>
          </div>

          {/* Right: floating bottle */}
          <motion.div
            style={{ y: bottleY }}
            className="w-full md:w-[46%] flex items-center justify-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 1.08, y: 30 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ duration: 1.6, delay: 0.4, ease: [0.22, 1, 0.36, 1] }}
              className="w-[72%] md:w-[88%] max-w-[520px]"
            >
              <motion.div
                animate={{ y: [0, -14, 0, 10, 0], rotate: [0, 0.8, 0, -0.8, 0] }}
                transition={{ duration: 9, repeat: Infinity, ease: 'easeInOut' }}
              >
                <Bottle />
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </motion.div>

      {/* Scroll cue */}
      <motion.div
        style={{ opacity: fade }}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, delay: 2.4 }}
        className="absolute left-1/2 -translate-x-1/2 bottom-7 md:bottom-10 z-30 flex flex-col items-center gap-2 text-argos-goldPale/70"
      >
        <span className="text-[9.5px] tracking-[0.45em] uppercase">
          Faire défiler
        </span>
        <motion.div
          animate={{ y: [0, 6, 0] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
        >
          <ChevronDown size={18} strokeWidth={1.2} />
        </motion.div>
      </motion.div>

      {/* Bottom fade to the next act */}
      <div className="absolute left-0 right-0 bottom-0 h-40 bg-gradient-to-b from-transparent to-[#07080E] pointer-events-none" />
    </section>
  );
}
