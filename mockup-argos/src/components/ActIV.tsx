import { motion } from 'framer-motion';
import { product } from '../data/product';
import { ActWrapper } from './ActWrapper';
import { Typewriter } from './Typewriter';
import { RelatedProducts } from './RelatedProducts';
import { Gift, Truck, ShieldCheck, RefreshCcw, Droplets, Headphones } from 'lucide-react';

/**
 * Acte IV — La Trace.
 * Commerce et rassurance tissés dans la narration — la preuve que l'e-shop
 * peut offrir ce qu'une boutique physique ne peut pas : essai remboursé,
 * paiement en 3 fois, avis vérifiés, authenticité garantie.
 */
export function ActIV() {
  return (
    <ActWrapper
      id="acte-iv"
      label="IV"
      title="La Trace"
      subtitle="Le parfum quitte le ciel d'orage pour marquer la peau et les mémoires. C'est ici que Planetebeauty relaie la maison — service, essai, fidélité."
      variant="ivory"
    >
      {/* Pull quotes row */}
      <div className="grid md:grid-cols-3 gap-8 md:gap-10 mb-20 md:mb-28">
        {REVIEWS.map((r, i) => (
          <motion.figure
            key={i}
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.7, delay: i * 0.12 }}
            className="border-t border-argos-gold/30 pt-6"
          >
            <blockquote className="font-display italic text-[22px] md:text-[26px] leading-[1.25] text-argos-ink">
              « {r.quote} »
            </blockquote>
            <figcaption className="mt-4 text-[10.5px] tracking-[0.35em] uppercase text-argos-bronze/80">
              {r.author} · {r.location}
            </figcaption>
          </motion.figure>
        ))}
      </div>

      {/* Description narrative */}
      <div className="grid md:grid-cols-12 gap-10 md:gap-14 mb-20 md:mb-28">
        <div className="md:col-span-7">
          <p className="text-[10px] tracking-[0.45em] uppercase text-argos-gold mb-5">
            Parole du Parfumeur
          </p>
          <blockquote className="border-l-2 border-argos-gold pl-6 md:pl-8 py-1 mb-10">
            <p className="font-display italic text-[28px] md:text-[36px] leading-[1.2] text-argos-ink">
              <Typewriter text={`« ${product.description.quote} »`} speed={22} />
            </p>
            <footer className="mt-4 text-[11px] tracking-[0.3em] uppercase text-argos-bronze">
              — {product.description.quoteAuthor}
            </footer>
          </blockquote>
          <div className="text-[15.5px] md:text-[16.5px] leading-[1.85] text-argos-ink/82 whitespace-pre-line font-light max-w-[62ch]">
            {product.description.paragraph}
          </div>
        </div>
        <aside className="md:col-span-5 space-y-8 md:sticky md:top-32 self-start">
          <InfoBlock title="Ingrédients — INCI">
            {product.ingredientsRaw}
          </InfoBlock>
          <InfoBlock title="Allergènes">{product.allergens}</InfoBlock>
          <InfoBlock title="Conseil de port">
            Vaporisez 2 à 3 pulvérisations sur les points de pulsation — poignets,
            creux du cou, derrière les oreilles. L'Extrait se révèle sur peau
            tiède, après 5 à 10 minutes.
          </InfoBlock>
        </aside>
      </div>

      {/* Trust strip */}
      <div className="border-y border-argos-gold/25 py-10 md:py-14 mb-20 md:mb-28">
        <p className="text-[10px] tracking-[0.45em] uppercase text-argos-gold mb-6 md:mb-8">
          Ce que Planetebeauty apporte à la maison Argos
        </p>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-y-6 gap-x-8 md:gap-x-14">
          {TRUST.map((t, i) => (
            <motion.div
              key={t.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-40px' }}
              transition={{ delay: i * 0.06, duration: 0.55 }}
              className="flex items-start gap-4"
            >
              <span className="shrink-0 w-10 h-10 rounded-full border border-argos-gold/50 flex items-center justify-center text-argos-goldDeep">
                <t.icon size={15} strokeWidth={1.3} />
              </span>
              <div>
                <p className="font-display text-[20px] md:text-[22px] leading-tight text-argos-ink">
                  {t.title}
                </p>
                <p className="text-[12.5px] text-argos-bronze/90 mt-1 leading-snug">
                  {t.sub}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Related as gallery */}
      <RelatedProducts />
    </ActWrapper>
  );
}

function InfoBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.6 }}
      className="border-t border-argos-bronze/25 pt-4"
    >
      <p className="text-[10px] tracking-[0.4em] uppercase text-argos-bronze/90 mb-2">
        {title}
      </p>
      <p className="text-[13px] leading-[1.7] text-argos-ink/80 font-light">
        {children}
      </p>
    </motion.div>
  );
}

const REVIEWS = [
  {
    quote:
      "La cardamome est foudroyante. Puis le safran prend la main, comme un dieu qui s'avance.",
    author: 'Élodie M.',
    location: 'Paris',
  },
  {
    quote:
      "Le flacon est déjà une œuvre. Le parfum tient plus de dix heures et change sur la peau comme un lever du jour.",
    author: 'Julien R.',
    location: 'Lyon',
  },
  {
    quote:
      "Offert à mon compagnon pour notre anniversaire. Il ne le pose plus. Argos est devenu notre découverte de l'année.",
    author: 'Clara B.',
    location: 'Bordeaux',
  },
];

const TRUST = [
  { icon: Droplets, title: 'Échantillon offert', sub: 'Par tranche de 50 € d’achat' },
  { icon: RefreshCcw, title: 'Try Me remboursé', sub: 'Hésitez ? Le 2 ml vous est déduit du 30 ml' },
  { icon: Gift, title: '2 achetés = 1 offert', sub: '15 ml à choisir dans la maison' },
  { icon: Truck, title: 'Livraison offerte', sub: 'Dès 99 € partout en France' },
  { icon: ShieldCheck, title: 'Authenticité garantie', sub: '100 % originaux · traçabilité lot par lot' },
  { icon: Headphones, title: 'Service client parfumeur', sub: 'Conseil olfactif personnalisé, 6/7 j' },
];
