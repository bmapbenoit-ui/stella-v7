import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import {
  Truck,
  Droplets,
  Gift,
  RefreshCcw,
  Headphones,
  ShieldCheck,
  LucideIcon,
} from 'lucide-react';

type Badge = { icon: LucideIcon; title: string; sub: string };

const BADGES: Badge[] = [
  { icon: Truck, title: 'Livraison offerte', sub: 'Dès 99 € en France' },
  { icon: Droplets, title: 'Échantillon offert', sub: 'Par tranche de 50 €' },
  { icon: Gift, title: '2 achetés = 1 offert', sub: '15 ml au choix' },
  { icon: RefreshCcw, title: 'Satisfait ou remboursé', sub: '30 jours' },
  { icon: Headphones, title: 'Service client expert', sub: 'Conseil personnalisé' },
  { icon: ShieldCheck, title: 'Authenticité garantie', sub: '100 % originaux' },
];

export function TrustBadges() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <div
      ref={ref}
      className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4 py-2"
    >
      {BADGES.map((b, i) => (
        <motion.div
          key={b.title}
          initial={{ opacity: 0, y: 14 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: i * 0.06, duration: 0.45 }}
          className="group flex items-start gap-3 p-3.5 rounded-[2px] border border-argos-bronze/20 bg-argos-ivory/40 hover:border-argos-gold/50 hover:bg-argos-ivory/70 transition-all cursor-hover"
        >
          <div className="shrink-0 w-9 h-9 rounded-full border border-argos-gold/40 flex items-center justify-center bg-gradient-to-br from-argos-ivory to-argos-parchment group-hover:shadow-gold-glow transition-shadow">
            <b.icon size={15} className="text-argos-goldDeep" strokeWidth={1.5} />
          </div>
          <div className="min-w-0">
            <div className="text-[11.5px] font-medium text-argos-ink leading-tight">
              {b.title}
            </div>
            <div className="text-[10.5px] text-argos-bronze/90 mt-0.5 leading-tight">
              {b.sub}
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
