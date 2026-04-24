import { useState } from 'react';
import { motion } from 'framer-motion';
import { Minus, Plus, CheckCircle2, Sparkles, ShieldCheck } from 'lucide-react';
import { product } from '../data/product';

export function CommerceBlock() {
  const [size, setSize] = useState<'30' | 'try'>('30');
  const [qty, setQty] = useState(1);
  const [installments, setInstallments] = useState<2 | 3 | 4>(3);
  const displayPrice = size === '30' ? product.price : 18;

  return (
    <div className="space-y-5">
      {/* Price */}
      <div className="flex items-end gap-4">
        <div className="font-display text-[44px] md:text-[56px] leading-none text-argos-ink tracking-tight">
          {displayPrice.toFixed(2).replace('.', ',')}&nbsp;
          <span className="text-argos-gold">€</span>
        </div>
        <div className="pb-2 text-[11px] text-argos-bronze/90 space-y-0.5">
          <div>
            {size === '30' ? '30 ml' : '2 ml'} ·{' '}
            {size === '30' ? product.pricePerMl.toFixed(2).replace('.', ',') : '9,00'} €/ml
          </div>
          <div className="uppercase tracking-[0.25em] text-[9px]">TTC</div>
        </div>
      </div>

      {/* Promo strip */}
      <motion.div
        animate={{ backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'linear' }}
        className="rounded-[3px] p-[1px] bg-gold-gradient"
        style={{ backgroundSize: '200% 200%' }}
      >
        <div className="rounded-[2px] bg-argos-ink px-4 py-2.5 text-argos-ivory flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-[12px]">
            <Sparkles size={14} className="text-argos-goldPale" />
            <span>
              Avec <b className="text-argos-goldPale">{product.promoCode}</b>, payez{' '}
              <b className="text-argos-goldPale">
                {product.priceWithPromo.toFixed(2).replace('.', ',')} €
              </b>
            </span>
          </div>
          <span className="px-2.5 py-1 rounded-full bg-gold-gradient text-argos-ink text-[10px] font-semibold tracking-[0.15em]">
            −10 %
          </span>
        </div>
      </motion.div>

      {/* Size selector */}
      <div className="space-y-2">
        <p className="text-[10px] tracking-[0.3em] uppercase text-argos-bronze/80">
          Contenance
        </p>
        <div className="grid grid-cols-2 gap-2.5">
          <SizeChip
            active={size === '30'}
            onClick={() => setSize('30')}
            title="30 ML"
            sub="Extrait de Parfum"
            price={`${product.price} €`}
          />
          <SizeChip
            active={size === 'try'}
            onClick={() => setSize('try')}
            title="TRY ME — 2 ML"
            sub="Atomiseur voyage"
            price="18 €"
          />
        </div>
      </div>

      {/* Stock status */}
      <div className="flex items-center gap-2 text-[12px] text-argos-ink/80">
        <span className="relative flex w-2 h-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500" />
        </span>
        {product.stockStatus}
      </div>

      {/* Try Me reimbursement */}
      <div className="glass-card rounded-sm p-4 flex gap-3">
        <div className="shrink-0 w-9 h-9 rounded-full bg-gold-gradient flex items-center justify-center">
          <ShieldCheck size={18} className="text-argos-ink" />
        </div>
        <div className="text-[12px] leading-snug">
          <p className="font-medium text-argos-ink">Vous hésitez ? Votre Try Me remboursé.</p>
          <p className="text-argos-bronze/90 mt-0.5">
            Commandez l'atomiseur 2 ml. Le prix est déduit si vous passez ensuite au flacon 30 ml.{' '}
            <a href="#" className="underline decoration-argos-gold underline-offset-2 hover:text-argos-gold cursor-hover">
              Conditions
            </a>
          </p>
        </div>
      </div>

      {/* Qty + CTA */}
      <div className="flex items-stretch gap-3">
        <div className="flex items-center border border-argos-ink/80 rounded-[2px] overflow-hidden">
          <button
            onClick={() => setQty((q) => Math.max(1, q - 1))}
            className="w-11 h-full flex items-center justify-center hover:bg-argos-ink/5 transition-colors cursor-hover"
          >
            <Minus size={14} />
          </button>
          <span className="w-10 text-center text-sm font-medium">{qty}</span>
          <button
            onClick={() => setQty((q) => q + 1)}
            className="w-11 h-full flex items-center justify-center hover:bg-argos-ink/5 transition-colors cursor-hover"
          >
            <Plus size={14} />
          </button>
        </div>
        <motion.button
          whileTap={{ scale: 0.98 }}
          className="btn-shimmer flex-1 rounded-[2px] bg-argos-ink text-argos-ivory px-6 h-12 flex items-center justify-center gap-3 text-[12px] tracking-[0.22em] uppercase hover:bg-argos-charcoal transition-colors cursor-hover"
        >
          <span>Ajouter au panier</span>
          <span className="text-argos-goldPale">
            · {(displayPrice * qty).toFixed(2).replace('.', ',')} €
          </span>
        </motion.button>
      </div>

      {/* Loyalty */}
      <div className="flex items-start gap-2.5 text-[12px] text-argos-bronze/95 bg-argos-parchment/60 rounded-sm p-3">
        <CheckCircle2 size={15} className="text-argos-gold mt-0.5 shrink-0" />
        <span>
          En achetant ce produit, gagnez{' '}
          <b className="text-argos-ink">{product.fidelityEuros.toFixed(2).replace('.', ',')} €</b>{' '}
          de fidélité pour votre prochain achat.
        </span>
      </div>

      {/* Installments */}
      <div className="space-y-2">
        <p className="text-[10px] tracking-[0.3em] uppercase text-argos-bronze/80">
          Paiement en plusieurs fois, sans frais
        </p>
        <div className="flex items-center gap-2">
          {[2, 3, 4].map((n) => (
            <button
              key={n}
              onClick={() => setInstallments(n as 2 | 3 | 4)}
              className={`px-3 py-2 text-[11px] tracking-[0.15em] uppercase border transition-all cursor-hover ${
                installments === n
                  ? 'border-argos-ink bg-argos-ink text-argos-goldPale'
                  : 'border-argos-bronze/30 text-argos-ink/70 hover:border-argos-gold'
              }`}
            >
              {n}×
            </button>
          ))}
          <span className="text-[12px] text-argos-bronze ml-2">
            {installments} × {(displayPrice / installments).toFixed(2).replace('.', ',')} €
          </span>
        </div>
        <div className="flex items-center gap-3 pt-1 text-[10px] text-argos-bronze/80">
          <Logo name="Klarna" />
          <Logo name="Scalapay" />
          <Logo name="PayPal" />
          <span className="ml-auto">Sans frais · Sans dossier</span>
        </div>
      </div>
    </div>
  );
}

function SizeChip({
  active,
  onClick,
  title,
  sub,
  price,
}: {
  active: boolean;
  onClick: () => void;
  title: string;
  sub: string;
  price: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`text-left px-4 py-3 rounded-[2px] border transition-all cursor-hover ${
        active
          ? 'bg-argos-ink text-argos-goldPale border-argos-ink shadow-[0_6px_18px_rgba(12,10,8,0.25)]'
          : 'bg-argos-ivory/40 text-argos-ink border-argos-bronze/25 hover:border-argos-gold'
      }`}
    >
      <div className="text-[12px] font-semibold tracking-[0.18em]">{title}</div>
      <div className={`text-[10.5px] mt-0.5 ${active ? 'text-argos-goldPale/80' : 'text-argos-bronze/90'}`}>
        {sub}
      </div>
      <div className={`mt-1.5 text-[12px] ${active ? 'text-argos-goldPale' : 'text-argos-ink/80'}`}>
        {price}
      </div>
    </button>
  );
}

function Logo({ name }: { name: string }) {
  return (
    <span className="px-2 py-0.5 rounded-sm border border-argos-bronze/30 font-medium tracking-wider">
      {name}
    </span>
  );
}
