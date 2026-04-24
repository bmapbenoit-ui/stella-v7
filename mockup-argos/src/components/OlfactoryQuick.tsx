import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { product } from '../data/product';
import { Flame, Wind, Clock, Sparkles as SparkIcon } from 'lucide-react';

export function OlfactoryQuick() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <div ref={ref} className="space-y-6">
      {/* Accord chips */}
      <div>
        <p className="text-[10px] tracking-[0.3em] uppercase text-argos-bronze/80 mb-3">
          Famille & Accords
        </p>
        <div className="flex flex-wrap gap-2">
          {product.accords.map((a, i) => (
            <motion.span
              key={a}
              initial={{ opacity: 0, y: 10 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: i * 0.06, duration: 0.45 }}
              className="tag-pulse px-3.5 py-1.5 text-[11px] tracking-[0.15em] uppercase rounded-full border border-argos-gold/40 text-argos-ink/80 bg-argos-ivory/40 cursor-hover"
            >
              {a}
            </motion.span>
          ))}
        </div>
      </div>

      {/* Pyramid strip */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { key: 'top', l: 'Tête', v: product.pyramid.top.hero },
          { key: 'heart', l: 'Cœur', v: product.pyramid.heart.hero },
          { key: 'base', l: 'Fond', v: product.pyramid.base.hero },
        ].map((row, i) => (
          <motion.div
            key={row.key}
            initial={{ opacity: 0, y: 12 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.35 + i * 0.1, duration: 0.5 }}
            className="glass-card rounded-sm px-3 py-3"
          >
            <p className="text-[9px] tracking-[0.32em] uppercase text-argos-gold/90">
              {row.l}
            </p>
            <p className="font-display text-lg md:text-xl text-argos-ink mt-0.5 leading-tight">
              {row.v}
            </p>
          </motion.div>
        ))}
      </div>

      {/* Profile card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ delay: 0.65, duration: 0.6 }}
        className="glass-card rounded-sm p-5 md:p-6 space-y-4 relative overflow-hidden"
      >
        <div
          aria-hidden
          className="absolute -top-12 -right-10 w-48 h-48 rounded-full"
          style={{
            background:
              'radial-gradient(circle, rgba(201,166,88,0.18) 0%, transparent 70%)',
            animation: 'float 10s ease-in-out infinite',
          }}
        />

        {/* Intensity */}
        <ProfileRow
          icon={<Flame size={14} />}
          label="Intensité"
          trailing={<span className="text-argos-ink font-medium">{product.profile.intensity}/5</span>}
        >
          <BarFill value={product.profile.intensity / 5} delay={0.2} inView={inView} />
        </ProfileRow>

        {/* Sillage */}
        <ProfileRow
          icon={<Wind size={14} />}
          label="Sillage"
          trailing={
            <span className="text-argos-ink font-medium">{product.profile.sillageLabel}</span>
          }
        >
          <div className="flex items-center gap-1.5">
            {Array.from({ length: 5 }).map((_, i) => (
              <motion.div
                key={i}
                initial={{ scale: 0, opacity: 0 }}
                animate={inView ? { scale: 1, opacity: 1 } : {}}
                transition={{ delay: 0.4 + i * 0.08 }}
                className={`w-3 h-3 rounded-full border transition-all ${
                  i < product.profile.sillage
                    ? 'bg-gold-gradient border-argos-goldDeep shadow-[0_0_8px_rgba(201,166,88,0.7)]'
                    : 'border-argos-bronze/30 bg-transparent'
                }`}
              />
            ))}
          </div>
        </ProfileRow>

        {/* Longevity */}
        <ProfileRow
          icon={<Clock size={14} />}
          label="Tenue"
          trailing={
            <span className="text-argos-ink font-medium">
              {product.profile.longevity}+ heures
            </span>
          }
        >
          <BarFill
            value={product.profile.longevity / product.profile.longevityMax}
            delay={0.45}
            inView={inView}
          />
        </ProfileRow>

        {/* Usage tags */}
        <div className="pt-2 border-t border-argos-gold/20 flex flex-wrap gap-x-5 gap-y-1.5 text-[11px] text-argos-ink/75">
          <span className="flex items-center gap-1.5">
            <SparkIcon size={11} className="text-argos-gold" />
            Saison : <b className="text-argos-ink">{product.usageTags.season}</b>
          </span>
          <span>
            Genre : <b className="text-argos-ink">{product.usageTags.gender}</b>
          </span>
          <span>
            Moment : <b className="text-argos-ink">{product.usageTags.moment}</b>
          </span>
        </div>
      </motion.div>
    </div>
  );
}

function ProfileRow({
  icon,
  label,
  trailing,
  children,
}: {
  icon: React.ReactNode;
  label: string;
  trailing?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-2 text-[11px] tracking-[0.22em] uppercase text-argos-bronze">
          <span className="text-argos-gold">{icon}</span>
          {label}
        </span>
        <span className="text-sm">{trailing}</span>
      </div>
      {children}
    </div>
  );
}

function BarFill({
  value,
  delay,
  inView,
}: {
  value: number;
  delay: number;
  inView: boolean;
}) {
  return (
    <div className="h-[5px] w-full rounded-full bg-argos-bronze/15 overflow-hidden">
      <motion.div
        initial={{ scaleX: 0 }}
        animate={inView ? { scaleX: value } : {}}
        transition={{ duration: 1.2, delay, ease: [0.22, 1, 0.36, 1] }}
        style={{ transformOrigin: 'left center' }}
        className="h-full w-full bg-gold-gradient rounded-full"
      />
    </div>
  );
}
