export function Footer() {
  return (
    <footer className="mt-24 border-t border-argos-gold/20 bg-argos-ink text-argos-ivory">
      <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-14 grid md:grid-cols-4 gap-10">
        <div>
          <div className="font-signature text-3xl text-gold-foil leading-none">
            Planetebeauty
          </div>
          <p className="text-[11px] tracking-[0.3em] uppercase text-argos-goldPale/70 mt-2">
            Parfums de Niche
          </p>
          <p className="text-[12px] text-argos-ivory/70 mt-4 leading-relaxed">
            Parfumerie confidentielle, sélection de maisons rares, conseil olfactif personnalisé.
            Depuis 2014.
          </p>
        </div>
        {[
          {
            t: 'La Maison',
            items: ['À propos', 'Le Journal', 'Nos parfumeurs', 'Affiliation'],
          },
          {
            t: 'Service Client',
            items: ['Livraison', 'Retours', 'Authenticité', 'Nous contacter'],
          },
          {
            t: 'Découvrir',
            items: ['Le Quiz', 'Coffrets', 'Carte cadeau', 'Programme fidélité'],
          },
        ].map((col) => (
          <div key={col.t}>
            <p className="text-[10px] tracking-[0.3em] uppercase text-argos-goldPale/70 mb-3">
              {col.t}
            </p>
            <ul className="space-y-2 text-[13px]">
              {col.items.map((i) => (
                <li key={i}>
                  <a
                    href="#"
                    className="text-argos-ivory/80 hover:text-argos-goldPale transition-colors cursor-hover"
                  >
                    {i}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="border-t border-argos-goldPale/10">
        <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-5 flex flex-col md:flex-row justify-between items-center gap-3 text-[11px] text-argos-ivory/60">
          <span>© 2026 Planetebeauty · Parfumerie de niche en ligne</span>
          <span>
            Site conçu pour la marque <b className="text-argos-goldPale">ARGOS FRAGRANCES</b> —
            Dallas, TX
          </span>
        </div>
      </div>
    </footer>
  );
}
