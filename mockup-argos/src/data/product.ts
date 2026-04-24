export const product = {
  brand: 'ARGOS',
  brandTagline: 'Fragrances of the Artist Series',
  name: "Jupiter's Lightning",
  subtitle: 'Extrait de Parfum',
  concentration: 'Extrait de Parfum',
  volume: '30 ml',
  series: 'Artist Series V',
  perfumer: 'Christian Petrovich',
  year: 2026,
  madeIn: 'United States',
  maison: 'Argos Fragrances, Dallas TX',
  family: 'Boisé Aromatique',
  concentrationPct: 35,
  price: 210,
  pricePerMl: 7.0,
  vatIncluded: true,

  // Planetebeauty context ------------------------------------------------
  priceWithPromo: 189,
  promoCode: 'PB10180',
  fidelityEuros: 10.5,
  stockStatus: 'Sur commande — Expédition 3 à 6 jours ouvrés',
  interestedCount: 127,

  // Marketing tags -------------------------------------------------------
  accords: ['Boisé', 'Aromatique', 'Épicé', 'Herbacé', 'Cuivré', 'Divin'],
  usageTags: {
    season: 'Automne · Hiver',
    gender: 'Mixte',
    moment: 'Soirée · Cérémonial',
  },

  // Olfactory pyramid ----------------------------------------------------
  pyramid: {
    top: {
      title: 'Tête',
      hero: 'Cardamome',
      heroImage: 'cardamome',
      description:
        "Un éclair de cardamome verte, fendu d'un zeste de bergamote de Calabre. Le genévrier crépite, le citron électrise.",
      notes: ['Cardamome verte', 'Bergamote de Calabre', 'Baies de genévrier', 'Citron'],
    },
    heart: {
      title: 'Cœur',
      hero: 'Safran',
      heroImage: 'safran',
      description:
        "La sauge sclarée enveloppe le safran comme une toge dorée. Un thé noir fumé trouve dans l'iris une rondeur minérale.",
      notes: ['Sauge sclarée', 'Thé noir', 'Safran', 'Iris · Orris'],
    },
    base: {
      title: 'Fond',
      hero: 'Cèdre de l’Atlas',
      heroImage: 'cedre',
      description:
        "Le cèdre grave son nom sur la peau. Le cyprès, sentinelle méditerranéenne, répond au musc comme la foudre répond au tonnerre.",
      notes: ['Cèdre de l’Atlas', 'Cyprès', 'Musc blanc'],
    },
  },

  // Intensity profile ----------------------------------------------------
  profile: {
    intensity: 4,
    sillage: 4,
    sillageLabel: 'Puissant',
    longevity: 9,
    longevityMax: 12,
  },

  // Description narrative -----------------------------------------------
  description: {
    quote:
      "J’ai voulu capturer l’instant où la foudre de Zeus fend la nuit — la seconde de vertige avant le tonnerre, quand l’air se charge d’ozone et d’or.",
    quoteAuthor: 'Christian Petrovich, parfumeur',
    paragraph: `Cinquième opus de l’Artist Series d’Argos, Jupiter’s Lightning s’inspire du Zeus trônant de Phidias et de la statuaire hellène où le père des dieux brandit son foudre. Le flacon — verre givré sculpté à Milan, col de métal plein, capuchon d’or mat — porte en plaque émaillée la silhouette du Maître de l’Olympe, ornée de cristaux Swarovski qui captent chaque inclinaison de lumière comme autant d’éclairs.

Le parfum s’ouvre sur une cardamome tranchante, comme la première goutte de pluie avant l’orage. En cœur, safran et sauge sclarée bâtissent un temple d’encens profane ; un thé noir fumé y dépose sa patine d’or vieilli. En fond, le cèdre de l’Atlas se fait colonne dorique, soutenu par le cyprès méditerranéen et un musc blanc qui laisse sur la peau la trace cuivrée du bronze divin.

Ce n’est pas un parfum. C’est une théophanie.`,
  },

  // Ingredient / allergen accordions -------------------------------------
  ingredientsRaw:
    'Alcohol Denat., Parfum (Fragrance), Aqua (Water), Benzyl Salicylate, Linalool, Limonene, Geraniol, Coumarin, Citral, Eugenol, Farnesol, Benzyl Alcohol, Tocopherol.',
  allergens:
    'Contient : Linalool, Limonene, Geraniol, Coumarin, Citral, Eugenol, Farnesol, Benzyl Salicylate, Benzyl Alcohol.',
};

export type Product = typeof product;
