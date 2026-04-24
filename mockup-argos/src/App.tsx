import { useLenis } from './hooks/useLenis';
import { PromoBar } from './components/PromoBar';
import { Header } from './components/Header';
import { ProductGallery } from './components/ProductGallery';
import { ProductIdentification } from './components/ProductIdentification';
import { OlfactoryQuick } from './components/OlfactoryQuick';
import { CommerceBlock } from './components/CommerceBlock';
import { TrustBadges } from './components/TrustBadges';
import { Personalization } from './components/Personalization';
import { Tabs } from './components/Tabs';
import { RelatedProducts } from './components/RelatedProducts';
import { MobileBottomBar } from './components/MobileBottomBar';
import { Footer } from './components/Footer';
import { CustomCursor } from './components/CustomCursor';
import { motion, useScroll } from 'framer-motion';

export default function App() {
  useLenis();
  const { scrollYProgress } = useScroll();

  return (
    <div className="relative min-h-screen">
      <CustomCursor />

      {/* Progress bar */}
      <motion.div
        aria-hidden
        className="fixed top-0 left-0 right-0 h-[2px] bg-gold-gradient origin-left z-[60]"
        style={{ scaleX: scrollYProgress }}
      />

      <PromoBar />
      <Header />

      <main className="relative z-10">
        {/* Hero product block ------------------------------------------ */}
        <section className="max-w-[1400px] mx-auto px-4 md:px-8 pt-6 md:pt-10 pb-14">
          <div className="grid md:grid-cols-12 gap-8 md:gap-14">
            <div className="md:col-span-7 md:sticky md:top-28 self-start">
              <ProductGallery />
            </div>
            <div className="md:col-span-5 space-y-10">
              <ProductIdentification />
              <SectionDivider />
              <OlfactoryQuick />
              <SectionDivider />
              <CommerceBlock />
              <SectionDivider />
              <TrustBadges />
              <SectionDivider />
              <Personalization />
            </div>
          </div>
        </section>

        {/* Detail tabs --------------------------------------------------- */}
        <section className="max-w-[1400px] mx-auto px-4 md:px-8 py-14 md:py-20">
          <div className="mb-8 md:mb-10">
            <p className="text-[10px] tracking-[0.3em] uppercase text-argos-gold/90 mb-1">
              La fiche complète
            </p>
            <h2 className="font-display text-3xl md:text-5xl text-argos-ink">
              Au plus près du parfum
            </h2>
          </div>
          <Tabs />
        </section>

        {/* Related ------------------------------------------------------- */}
        <section className="max-w-[1400px] mx-auto px-4 md:px-8 py-14 md:py-20">
          <RelatedProducts />
        </section>
      </main>

      <Footer />
      <MobileBottomBar />
    </div>
  );
}

function SectionDivider() {
  return (
    <motion.div
      initial={{ scaleX: 0, opacity: 0 }}
      whileInView={{ scaleX: 1, opacity: 1 }}
      viewport={{ once: true, margin: '-40px' }}
      transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
      style={{ transformOrigin: 'left center' }}
      className="divider-draw"
      aria-hidden
    />
  );
}
