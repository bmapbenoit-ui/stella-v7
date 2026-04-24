import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface Props {
  label: string; // I, II, III, IV
  title: string;
  subtitle?: string;
  variant?: 'dark' | 'ivory';
  children: ReactNode;
  id?: string;
}

export function ActWrapper({ label, title, subtitle, variant = 'dark', children, id }: Props) {
  const dark = variant === 'dark';
  return (
    <section
      id={id}
      className={`relative w-full py-28 md:py-40 overflow-hidden ${
        dark ? 'bg-[#0A0D16] text-argos-ivory' : 'bg-argos-ivory text-argos-ink'
      }`}
    >
      {/* Soft atmospheric background */}
      <div
        aria-hidden
        className="absolute inset-0 pointer-events-none"
        style={{
          background: dark
            ? 'radial-gradient(ellipse at 20% 20%, rgba(201,166,88,0.08) 0%, transparent 55%), radial-gradient(ellipse at 80% 80%, rgba(27,36,54,0.6) 0%, transparent 65%)'
            : 'radial-gradient(ellipse at 20% 20%, rgba(233,212,162,0.25) 0%, transparent 55%), radial-gradient(ellipse at 80% 80%, rgba(201,166,88,0.15) 0%, transparent 65%)',
        }}
      />

      <div className="relative max-w-[1400px] mx-auto px-5 md:px-12">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-100px' }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="mb-14 md:mb-20"
        >
          <div className="flex items-center gap-4 mb-4 md:mb-6">
            <span
              className={`font-display italic text-2xl md:text-3xl ${
                dark ? 'text-argos-goldPale' : 'text-argos-gold'
              }`}
            >
              Acte {label}
            </span>
            <motion.span
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.4, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              style={{ transformOrigin: 'left' }}
              className={`h-[1px] w-32 md:w-48 ${
                dark ? 'bg-argos-goldPale/40' : 'bg-argos-gold/40'
              }`}
            />
          </div>
          <h2
            className={`font-display leading-[0.95] tracking-tight ${
              dark ? 'text-argos-ivory' : 'text-argos-ink'
            }`}
            style={{ fontSize: 'clamp(40px, 6vw, 96px)' }}
          >
            {title}
          </h2>
          {subtitle && (
            <p
              className={`mt-4 md:mt-6 max-w-xl text-[14px] md:text-[15px] leading-relaxed ${
                dark ? 'text-argos-ivory/65' : 'text-argos-ink/70'
              }`}
            >
              {subtitle}
            </p>
          )}
        </motion.div>
        {children}
      </div>
    </section>
  );
}
