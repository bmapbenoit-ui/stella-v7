# Plan de captures d'écran pour l'email Argos

Cette maquette est volontairement dense : l'envoyer brute en 1 image tuerait
le rendu. Le plan ci-dessous permet un email de **5 à 6 visuels maximum**,
dans cet ordre.

## Préparation

- Navigateur : **Chrome ou Arc desktop**, largeur **1440 px**, zoom 100 %.
- Mode clair forcé, mode incognito (aucune extension, aucune barre de favoris).
- Désactiver les barres de défilement custom (dans un mode incognito, la
  scrollbar Planetebeauty apparaît ; si ça gêne, utiliser un aperçu Figma /
  CleanShot avec masquage scrollbar).
- Attendre **~2 s après chargement** pour que toutes les entrées séquencées
  soient posées (notamment le compteur *X personnes intéressées* et la barre
  d'intensité qui se remplit).

## Visuels recommandés

### 1. Hero complet — « Le choc visuel »
> **Le seul visuel indispensable.** À placer en tête d'email.

- Cadre : depuis le top du header jusqu'au bas du bloc commerce inclus
  (CTA *Ajouter au panier*).
- Format : **1440 × ~1800 px**, image PNG.
- À capter en particulier : flacon parfaitement centré, plaque Jupiter
  nette, bandeau promo doré dynamique visible, prix en grand.
- Astuce : capture `cmd + shift + 4` puis espace sur l'écran, puis
  `CleanShot` en mode scroll capture pour intégrer le hero en une seule
  image sans couper.

### 2. Galerie avec reflet en survol — « La matière »
- Cadre : zoom ~800 × 900 px sur la galerie produit avec **le curseur sur
  le flacon** (activer le tilt 3D + reflet doré qui glisse).
- Accepter de légèrement flouter le reflet en mouvement — ça prouve
  visuellement que la page vit.
- Alternative : capture vidéo de 3 s en MP4 (très puissant en email).

### 3. Pyramide olfactive & profil — « La science du parfum »
- Cadre : depuis *Famille & Accords* jusqu'à la fin du bloc Profil
  Olfactif, ~680 × 1100 px.
- Les barres d'intensité, sillage et tenue sont déjà remplies à ce stade :
  la promesse chiffrée est un levier majeur vs la boutique physique.

### 4. Onglet *Notes Olfactives* — « La cascade de matières »
- Scroller jusqu'aux tabs, cliquer sur *Notes Olfactives* (actif par
  défaut), attendre les 3 cartes (tête / cœur / fond) entrées en cascade.
- Cadre : **1400 × 700 px**. Visuel très éditorial, textures d'ingrédients
  propres.

### 5. Onglet *Description* — « La voix du parfumeur »
- Cliquer sur *Description*, **laisser démarrer l'effet machine à écrire**
  et capturer **à mi-parcours** (caret clignotant visible, moitié du texte
  déjà posé). Preuve vivante, très cinématographique.
- Cadre : 1400 × 650 px.

### 6. Carrousel *Vous aimerez aussi* — « L'upsell maison »
- Cadre : 1400 × 650 px.
- Montre à Argos que la fiche contribue à **cross-sell l'ensemble de la
  gamme**. Neptune, Perseus, Vulcan, Danaë sont tous visibles.

## Capture alternative : onepager vertical

Si Argos préfère une version condensée, utiliser un navigateur avec
*Full page screenshot* (DevTools → `cmd+shift+P` → *Capture full size
screenshot*) pour obtenir un visuel **1440 × ~7500 px** de toute la page.
Réduire ensuite à **1440 × 4500 px** dans Figma en conservant seulement
le hero, la pyramide, les tabs *Notes* et *Description*, et le carrousel.

## Email d'accompagnement (proposition)

> Objet : *Une fiche produit digne de Jupiter's Lightning*
>
> Cher(e) [nom],
>
> Merci pour votre retour. Avant de revenir vers vous avec une proposition
> commerciale classique, j'ai préféré vous montrer ce que serait une fiche
> produit Argos sur Planetebeauty. La maquette ci-dessous ne représente
> qu'une partie de l'expérience : toutes les interactions (tilt du flacon,
> reveal des matières, typewriter sur la citation de Christian, pyramide
> animée) ne se capturent qu'en mouvement. Je me tiens à disposition pour
> une démo live de 10 minutes.
>
> Cordialement,
> Benoit Hodiesne — Planetebeauty.com

## Liste des effets visibles sur la maquette

1. Fade-in séquencé (header → image → identification → bloc olfactif)
2. Image produit avec drop-shadow qui se construit + légère décompression
3. Compteur *interestedCount* de 0 à 127 en 2,2 s
4. Barre d'intensité qui se remplit
5. Cercles de sillage qui pop en cascade
6. Pyramide condensée en glassmorphism, entrée en relais
7. Smooth scroll global (Lenis)
8. Header passant en glassmorphism après 100 px
9. Reveal au scroll sur chaque bloc (IntersectionObserver via Framer)
10. Parallax léger sur l'image produit principale
11. Effet Ken Burns sur 2 thumbnails (mood Jupiter + ingrédient)
12. Tilt 3D sur le flacon qui suit le curseur
13. Reflet doré qui glisse en diagonale au survol du flacon
14. Cristaux Swarovski qui pulsent sur la plaque
15. Sparkles dorés flottants autour du flacon (5, sur 9-13 s)
16. Bandeau promo avec gradient qui oscille
17. Boutons CTA avec shimmer doré au survol
18. Tags famille qui pulsent au survol
19. Thumbnails qui s'élèvent + ombre dorée au hover
20. Transitions crossfade + slide entre onglets
21. Underline doré glissant entre les onglets
22. Typewriter sur la citation du parfumeur (tab *Description*)
23. Accordion animé sur INCI / Allergènes / Conseil
24. Carrousel de produits croisés avec hover halo coloré
25. Sticky bottom bar mobile qui apparaît après 500 px
26. Curseur custom : point doré + anneau suiveur + agrandissement interactif
27. Marquee infini sur le bandeau promo en haut
28. Dividers dorés qui se dessinent à l'apparition
29. Gradient shifting dans le bandeau code promo
30. Progress bar dorée de lecture en haut de page
