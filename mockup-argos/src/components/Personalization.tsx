import { motion } from 'framer-motion';
import { ArrowRight, UserCircle2 } from 'lucide-react';

export function Personalization() {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      {/* Pour qui ? */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.5 }}
        className="glass-card rounded-sm p-5 flex gap-4"
      >
        <div className="shrink-0 w-12 h-12 rounded-full bg-gold-gradient flex items-center justify-center">
          <UserCircle2 size={22} className="text-argos-ink" />
        </div>
        <div>
          <p className="text-[10px] tracking-[0.3em] uppercase text-argos-gold/90 mb-1">
            Pour qui
          </p>
          <h3 className="font-display text-xl text-argos-ink leading-tight mb-1">
            L'âme cérémonielle
          </h3>
          <p className="text-[13px] text-argos-ink/80 leading-relaxed">
            Pour celle ou celui qui aime habiller un soir d'automne d'une sensation d'autorité
            tranquille. Qui préfère la verticalité d'une colonne dorique à la fébrilité d'une mode.
            Un parfum qui marque la peau et les mémoires.
          </p>
        </div>
      </motion.div>

      {/* Quiz CTA */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: '-80px' }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="relative overflow-hidden rounded-sm bg-argos-ink text-argos-ivory p-6 group cursor-hover"
      >
        <motion.div
          aria-hidden
          className="absolute -top-16 -right-16 w-64 h-64 rounded-full blur-3xl opacity-40"
          style={{
            background:
              'radial-gradient(circle, rgba(201,166,88,0.6) 0%, transparent 70%)',
          }}
          animate={{ x: [0, 10, 0], y: [0, -10, 0] }}
          transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
        />
        <p className="text-[10px] tracking-[0.3em] uppercase text-argos-goldPale/80">
          Le Quiz Planetebeauty
        </p>
        <h3 className="font-display text-2xl md:text-3xl mt-2 leading-tight">
          Vous hésitez ? Trouvez le parfum qui vous correspond.
        </h3>
        <p className="text-[13px] text-argos-ivory/70 mt-2 max-w-md">
          8 questions, 2 minutes. Notre chef parfumeur vous propose 3 fragrances de niche choisies
          pour votre peau, vos saisons et votre humeur.
        </p>
        <button className="mt-5 inline-flex items-center gap-3 rounded-[2px] bg-gold-gradient text-argos-ink px-5 h-11 text-[11px] tracking-[0.25em] uppercase font-medium btn-shimmer cursor-hover">
          Faire le quiz
          <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
        </button>
      </motion.div>
    </div>
  );
}
