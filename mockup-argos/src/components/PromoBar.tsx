import { Truck, Percent, Gift } from 'lucide-react';

const MESSAGES = [
  { icon: Truck, text: 'Livraison offerte dès 99 €' },
  { icon: Percent, text: 'Cashback 5 % sur chaque commande' },
  { icon: Gift, text: 'Code PB580 — 5 % dès 80 €' },
  { icon: Gift, text: 'Code PB10180 — 10 % dès 180 €' },
  { icon: Truck, text: '2 parfums achetés = 1 parfum 15 ml offert' },
];

export function PromoBar() {
  return (
    <div className="bg-argos-ink text-argos-ivory border-b border-argos-gold/20">
      <div className="marquee py-2.5 text-[11px] tracking-argos uppercase">
        {[0, 1].map((k) => (
          <div key={k} className="marquee-track" aria-hidden={k === 1}>
            {MESSAGES.map((m, i) => (
              <span key={`${k}-${i}`} className="flex items-center gap-3">
                <m.icon size={13} className="text-argos-goldPale" />
                <span>{m.text}</span>
                <span className="text-argos-gold/60">·</span>
              </span>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
