import { motion } from 'framer-motion';

/**
 * Finale — cinema-style credits. Closes the narrative.
 */
export function Finale() {
  return (
    <section className="relative bg-[#07080E] text-argos-ivory py-28 md:py-40 overflow-hidden">
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse at 50% 80%, rgba(201,166,88,0.15) 0%, transparent 55%)',
        }}
      />
      <div className="relative max-w-[1100px] mx-auto px-5 md:px-12 text-center">
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1.2 }}
          className="text-[10px] tracking-[0.6em] uppercase text-argos-goldPale/70 mb-8 md:mb-12"
        >
          — Fin —
        </motion.p>
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, delay: 0.2 }}
          className="font-display italic leading-[0.95]"
          style={{ fontSize: 'clamp(48px, 7vw, 112px)' }}
        >
          <span className="bg-gradient-to-br from-argos-goldPale via-argos-gold to-argos-goldDeep bg-clip-text text-transparent">
            Tu te souviendras d'Argos.
          </span>
        </motion.h2>
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, delay: 0.6 }}
          className="mt-10 md:mt-14 text-[12.5px] text-argos-ivory/60 max-w-xl mx-auto leading-relaxed"
        >
          Jupiter's Lightning · Artist Series V · Extrait de Parfum · 30 ml ·
          210 € TTC. Composé par Christian Petrovich · Dallas, TX. Distribué en
          avant-première en Europe par Planetebeauty — parfumerie de niche
          confidentielle depuis 2014.
        </motion.p>
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, delay: 1 }}
          className="mt-16 md:mt-20 grid grid-cols-2 md:grid-cols-4 gap-y-6 gap-x-8 text-[10.5px] tracking-[0.3em] uppercase text-argos-goldPale/60"
        >
          <CreditLine role="Maison" name="Argos Fragrances" />
          <CreditLine role="Parfumeur" name="Christian Petrovich" />
          <CreditLine role="Art Director" name="Planetebeauty Studio" />
          <CreditLine role="Distribution" name="Planetebeauty · 2026" />
        </motion.div>
        <div className="mt-16 md:mt-20 border-t border-argos-goldPale/10 pt-6 text-[10.5px] text-argos-ivory/40 tracking-[0.2em]">
          © 2026 Planetebeauty — Conçu pour la maison Argos, Dallas TX.
        </div>
      </div>
    </section>
  );
}

function CreditLine({ role, name }: { role: string; name: string }) {
  return (
    <div>
      <p className="text-argos-goldPale/50">{role}</p>
      <p className="font-display italic text-argos-ivory/90 text-[13px] tracking-[0.05em] normal-case mt-1">
        {name}
      </p>
    </div>
  );
}
