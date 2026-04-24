# Mockup Argos × Planetebeauty

Maquette d'une fiche produit Planetebeauty.com habillée aux couleurs et à l'ADN
de la maison **Argos Fragrances** (Dallas, TX). Produit mis en avant :
**Jupiter's Lightning — Extrait de Parfum, Artist Series V**.

L'objectif unique de cette page est de produire **un screenshot** à envoyer
à l'équipe marketing d'Argos pour ouvrir une négociation de distribution.
Elle n'a pas vocation à être mise en production.

## Lancement

```bash
pnpm install
pnpm dev
```

Ouvrir http://localhost:5173. Préférer Chrome desktop pour le screenshot,
car plusieurs effets (backdrop-filter, drop-shadow composés, blend modes,
curseur custom) rendent mieux sur moteurs WebKit/Blink.

### Build de production (vérification)

```bash
pnpm build
pnpm preview
```

## Stack

| Techno        | Rôle                                          |
| ------------- | --------------------------------------------- |
| Vite + React 18 + TypeScript | Rendu et types                   |
| Tailwind CSS  | Palette Argos + design tokens                 |
| Framer Motion | Séquencement, reveals au scroll, tilt 3D      |
| Lenis         | Smooth scroll global                           |
| Lucide React  | Icônes fines et homogènes                     |
| Cormorant Garamond / Inter / Pinyon Script | Typographies (Google Fonts) |

Aucune image externe : **flacon, plaque mythologique, ingrédients, produits
croisés** sont tous fabriqués en SVG pour rester fidèles à la maquette,
indépendants de la disponibilité des assets Argos, et légers.

## Fichiers clés

- `src/App.tsx` — orchestration des 26 blocs
- `src/data/product.ts` — toutes les données du parfum
- `src/components/Bottle.tsx` — flacon signature Argos (SVG, plaque Jupiter,
  cristaux Swarovski, reflet dynamique)
- `src/components/Tabs.tsx` — onglets Notes / Profil / Description / Avis
- `BRAND_ANALYSIS.md` — note stratégique sur l'identité Argos
- `SCREENSHOTS.md` — plan de captures recommandé pour l'email commercial

## Recommandations pour le screenshot

Voir **[SCREENSHOTS.md](./SCREENSHOTS.md)** pour la liste détaillée (sections,
formats, ordre, astuces pour capter les animations).
