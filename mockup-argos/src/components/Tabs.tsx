import { AnimatePresence, motion } from 'framer-motion';
import { useLayoutEffect, useRef, useState } from 'react';
import { product } from '../data/product';
import { IngredientGlyph } from './IngredientGlyph';
import { Typewriter } from './Typewriter';
import { ChevronDown, Star } from 'lucide-react';

const TABS = [
  { key: 'notes', label: 'Notes Olfactives' },
  { key: 'profile', label: 'Profil Olfactif' },
  { key: 'desc', label: 'Description' },
  { key: 'reviews', label: 'Avis' },
] as const;

type Tab = (typeof TABS)[number]['key'];

export function Tabs() {
  const [active, setActive] = useState<Tab>('notes');
  const listRef = useRef<HTMLDivElement>(null);
  const [underline, setUnderline] = useState({ left: 0, width: 0 });

  useLayoutEffect(() => {
    const el = listRef.current?.querySelector<HTMLButtonElement>(`[data-tab="${active}"]`);
    if (!el) return;
    const parentRect = listRef.current!.getBoundingClientRect();
    const rect = el.getBoundingClientRect();
    setUnderline({ left: rect.left - parentRect.left, width: rect.width });
  }, [active]);

  return (
    <div className="space-y-8">
      <div
        ref={listRef}
        className="relative flex flex-wrap gap-6 md:gap-10 border-b border-argos-bronze/20"
      >
        {TABS.map((t) => (
          <button
            key={t.key}
            data-tab={t.key}
            onClick={() => setActive(t.key)}
            className={`py-4 text-[11px] md:text-[12px] tracking-[0.28em] uppercase transition-colors cursor-hover ${
              active === t.key ? 'text-argos-ink' : 'text-argos-bronze/70 hover:text-argos-ink'
            }`}
          >
            {t.label}
          </button>
        ))}
        <span
          className="tab-underline"
          style={{ left: underline.left, width: underline.width }}
        />
      </div>

      <div className="min-h-[420px] relative">
        <AnimatePresence mode="wait">
          {active === 'notes' && (
            <TabPane key="notes">
              <NotesTab />
            </TabPane>
          )}
          {active === 'profile' && (
            <TabPane key="profile">
              <ProfileTab />
            </TabPane>
          )}
          {active === 'desc' && (
            <TabPane key="desc">
              <DescriptionTab />
            </TabPane>
          )}
          {active === 'reviews' && (
            <TabPane key="reviews">
              <ReviewsTab />
            </TabPane>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function TabPane({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -30 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  );
}

// ----- Notes tab ----------------------------------------------------------
function NotesTab() {
  const tiers = [
    { ...product.pyramid.top, glyph: 'cardamome' as const },
    { ...product.pyramid.heart, glyph: 'safran' as const },
    { ...product.pyramid.base, glyph: 'cedre' as const },
  ];
  return (
    <div className="grid md:grid-cols-3 gap-5">
      {tiers.map((tier, i) => (
        <motion.div
          key={tier.title}
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 + i * 0.12, duration: 0.55 }}
          className="group glass-card rounded-sm p-5 relative overflow-hidden"
        >
          <div className="absolute top-3 right-3 text-[9px] tracking-[0.4em] uppercase text-argos-gold/90">
            {['I', 'II', 'III'][i]}
          </div>
          <div className="aspect-[4/3] rounded-[2px] overflow-hidden mb-4 relative">
            <div className="absolute inset-0 transition-transform duration-700 group-hover:scale-110">
              <IngredientGlyph kind={tier.glyph} />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-argos-ink/50 via-transparent to-transparent" />
            <div className="absolute left-3 bottom-3">
              <p className="text-[9.5px] tracking-[0.3em] uppercase text-argos-goldPale/80">
                {tier.title}
              </p>
              <p className="font-display text-2xl text-white leading-tight">
                {tier.hero}
              </p>
            </div>
          </div>
          <p className="text-[12.5px] text-argos-ink/80 leading-relaxed italic">
            “{tier.description}”
          </p>
          <div className="flex flex-wrap gap-1.5 mt-4">
            {tier.notes.map((n) => (
              <span
                key={n}
                className="px-2.5 py-1 rounded-full text-[10.5px] border border-argos-bronze/30 text-argos-ink/80 hover:border-argos-gold hover:text-argos-ink transition-colors cursor-hover"
              >
                {n}
              </span>
            ))}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

// ----- Profile tab --------------------------------------------------------
function ProfileTab() {
  const p = product.profile;
  const details = [
    { l: 'Famille', v: product.family },
    { l: 'Concentration', v: `${product.concentration} · ${product.concentrationPct}%` },
    { l: 'Parfumeur', v: product.perfumer },
    { l: 'Année', v: String(product.year) },
    { l: 'Occasion', v: product.usageTags.moment },
    { l: 'Saison', v: product.usageTags.season },
    { l: 'Made in', v: product.madeIn },
    { l: 'Maison', v: product.maison },
  ];
  return (
    <div className="grid md:grid-cols-5 gap-8">
      <div className="md:col-span-2 space-y-6">
        <BigMeter
          label="Sillage"
          value={p.sillage}
          max={5}
          trailing={p.sillageLabel}
        />
        <BigMeter
          label="Tenue"
          value={p.longevity}
          max={p.longevityMax}
          trailing={`${p.longevity}+ heures`}
        />
        <BigMeter
          label="Intensité"
          value={p.intensity}
          max={5}
          trailing={`${p.intensity} / 5`}
        />
      </div>
      <div className="md:col-span-3 grid grid-cols-2 gap-x-8 gap-y-5">
        {details.map((d, i) => (
          <motion.div
            key={d.l}
            initial={{ opacity: 0, y: 10 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: i * 0.05, duration: 0.4 }}
            className="border-b border-argos-bronze/15 pb-3"
          >
            <p className="text-[10px] tracking-[0.3em] uppercase text-argos-bronze/80">
              {d.l}
            </p>
            <p className="font-display text-lg text-argos-ink mt-1">{d.v}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function BigMeter({
  label,
  value,
  max,
  trailing,
}: {
  label: string;
  value: number;
  max: number;
  trailing: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-[10.5px] tracking-[0.3em] uppercase text-argos-bronze">
          {label}
        </span>
        <span className="font-display text-2xl text-argos-ink">{trailing}</span>
      </div>
      <div className="h-[10px] rounded-full bg-argos-bronze/15 overflow-hidden relative">
        <motion.div
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: value / max }}
          viewport={{ once: true, margin: '-40px' }}
          transition={{ duration: 1.3, ease: [0.22, 1, 0.36, 1] }}
          style={{ transformOrigin: 'left' }}
          className="h-full w-full bg-gold-gradient"
        />
      </div>
    </div>
  );
}

// ----- Description tab ---------------------------------------------------
function DescriptionTab() {
  return (
    <div className="grid md:grid-cols-12 gap-10">
      <div className="md:col-span-7 space-y-6">
        <blockquote className="border-l-2 border-argos-gold pl-5 py-1">
          <p className="font-display italic text-xl md:text-2xl text-argos-ink leading-snug">
            <Typewriter
              text={`« ${product.description.quote} »`}
              speed={24}
            />
          </p>
          <footer className="mt-3 text-[11px] tracking-[0.25em] uppercase text-argos-bronze">
            — {product.description.quoteAuthor}
          </footer>
        </blockquote>
        <div className="text-[14px] leading-[1.8] text-argos-ink/85 space-y-4 whitespace-pre-line">
          {product.description.paragraph}
        </div>
      </div>
      <div className="md:col-span-5 space-y-3">
        <Accordion title="Ingrédients — INCI" defaultOpen>
          <p className="text-[12px] leading-relaxed text-argos-ink/80">
            {product.ingredientsRaw}
          </p>
        </Accordion>
        <Accordion title="Allergènes">
          <p className="text-[12px] leading-relaxed text-argos-ink/80">
            {product.allergens}
          </p>
        </Accordion>
        <Accordion title="Conseil de port">
          <p className="text-[12px] leading-relaxed text-argos-ink/80">
            Vaporisez 2 à 3 pulvérisations sur les points de pulsation — intérieur des poignets,
            creux du cou, derrière les oreilles. Évitez les frictions qui altèrent la pyramide.
            L'Extrait se révèle sur peau tiède, après 5 à 10 minutes.
          </p>
        </Accordion>
      </div>
    </div>
  );
}

function Accordion({
  title,
  defaultOpen = false,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-argos-bronze/20">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between py-3.5 text-left cursor-hover"
      >
        <span className="text-[11px] tracking-[0.28em] uppercase text-argos-ink">
          {title}
        </span>
        <ChevronDown
          size={16}
          className={`transition-transform duration-400 text-argos-gold ${
            open ? 'rotate-180' : ''
          }`}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35 }}
            className="overflow-hidden"
          >
            <div className="pb-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ----- Reviews tab -------------------------------------------------------
function ReviewsTab() {
  return (
    <div className="grid md:grid-cols-3 gap-8">
      <div className="md:col-span-1 glass-card rounded-sm p-6 text-center">
        <p className="font-display text-6xl text-argos-ink leading-none">4,9</p>
        <div className="flex items-center justify-center gap-0.5 mt-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Star
              key={i}
              size={16}
              className="fill-argos-gold text-argos-gold"
            />
          ))}
        </div>
        <p className="text-[11px] text-argos-bronze mt-2 tracking-[0.15em] uppercase">
          Sur 42 avis vérifiés
        </p>
        <div className="mt-5 space-y-1.5">
          {[5, 4, 3, 2, 1].map((star) => (
            <div key={star} className="flex items-center gap-2 text-[11px]">
              <span className="w-3 text-argos-ink/80">{star}</span>
              <div className="flex-1 h-1.5 rounded-full bg-argos-bronze/15">
                <div
                  className="h-full rounded-full bg-gold-gradient"
                  style={{ width: `${[84, 14, 2, 0, 0][5 - star]}%` }}
                />
              </div>
              <span className="w-7 text-argos-bronze text-right">
                {[35, 6, 1, 0, 0][5 - star]}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="md:col-span-2 space-y-4">
        {REVIEWS.map((r, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ delay: i * 0.08 }}
            className="p-5 rounded-sm border border-argos-bronze/20 bg-argos-ivory/40"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="w-8 h-8 rounded-full bg-gold-gradient flex items-center justify-center text-argos-ink font-semibold text-sm">
                  {r.initial}
                </span>
                <div>
                  <p className="text-[12.5px] font-medium text-argos-ink">{r.name}</p>
                  <p className="text-[10px] text-argos-bronze tracking-[0.2em] uppercase">
                    Achat vérifié · {r.date}
                  </p>
                </div>
              </div>
              <div className="flex">
                {Array.from({ length: 5 }).map((_, j) => (
                  <Star
                    key={j}
                    size={12}
                    className={j < r.rating ? 'fill-argos-gold text-argos-gold' : 'text-argos-bronze/30'}
                  />
                ))}
              </div>
            </div>
            <p className="font-display text-lg italic text-argos-ink mb-1">
              « {r.title} »
            </p>
            <p className="text-[13px] text-argos-ink/85 leading-relaxed">
              {r.text}
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

const REVIEWS = [
  {
    initial: 'E',
    name: 'Élodie M.',
    date: 'Mars 2026',
    rating: 5,
    title: "Un parfum qui marche au tonnerre",
    text:
      "La cardamome d'ouverture est foudroyante, puis le safran prend la main avec une majesté rare. Sur ma peau, le cèdre tient plus de 10 heures. Je n'ai jamais reçu autant de compliments.",
  },
  {
    initial: 'J',
    name: 'Julien R.',
    date: 'Février 2026',
    rating: 5,
    title: "L'Olympe dans un flacon",
    text:
      "Le flacon est déjà une œuvre — la plaque émaillée mériterait une vitrine. Et le parfum tient ses promesses : cérémonieux sans être pesant, unique sans être agressif.",
  },
  {
    initial: 'C',
    name: 'Clara B.',
    date: 'Février 2026',
    rating: 5,
    title: 'Le plus beau cadeau',
    text:
      "Offert à mon compagnon pour notre anniversaire. Il ne le pose plus. Le sillage est de 2 à 3 mètres sans être envahissant. Argos est une découverte pour nous.",
  },
];
