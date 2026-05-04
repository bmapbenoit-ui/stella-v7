# Conversation — Maison Mataha TikTok / PlanèteBeauty Content Engine

**Date** : 2026-05-04
**Branche Git** : `claude/implement-maison-mataha-tiktok-1TXKG`
**Sujet** : Construction d'un système de production vidéo TikTok automatisé pour PlanèteBeauty, focalisé sur la promotion de marques de parfum de niche peu connues.

---

## TL;DR (lecture en 60 secondes)

1. **Demande initiale** : implémenter un design TikTok pour Maison Mataha *Escapade Gourmande* (3 variations 9:16 / 14s) — fait, commité, MP4 livrés sous `stella-shopify-app/static/maison-mataha/renders/`.
2. **Pivot** : passer d'un one-shot à un **système de production 1 vidéo/jour** entièrement automatisé.
3. **Différenciateur découvert** : PlanèteBeauty a comme **value prop** la curation de **marques de niche peu connues** — angle stratégique unique sur PerfumeTok 2026 (terrain peu compétitif, marges meilleures, audience qualifiée).
4. **Décision finale** : Phase 2 directe, automation totale, stack `HeyGen Avatar V Team + Veo 3 Fast (GCP) + Pexels + Remotion`.
5. **Crédits Vertex AI** : 300 € disponibles (~5-6 mois de runway gratuit sur Veo 3 Fast).
6. **Marques à développer** : 10 marques, AB Signature en flagship.

---

## Contexte initial

L'utilisateur a fait analyser une vidéo de parfum via Claude Design (claude.ai/design). Le designer Claude a livré un bundle HTML/CSS/JS avec 3 variations TikTok ASMR pour Maison Mataha *Escapade Gourmande*.

Brief originel :
- 3 variations TikTok 9:16 / 14s
- ASMR sensoriel (gros plans, son du spray, particules dorées)
- Ambiance : doré chaud, golden hour, papier froissé
- Notes : vanille, fève tonka, canne à sucre, benjoin
- CTA : `planetebeauty.com`

---

## Phase 1 — Implémentation initiale (livrée)

### Décision : déployer la prototype HTML/CSS/JS comme micro-site statique

Localisation : `stella-shopify-app/static/maison-mataha/`

Structure :
- `index.html` — showcase 3 phones côte-à-côte
- `a.html` / `b.html` / `c.html` — chaque variation en plein écran (TikTok format)
- `animations.jsx` — moteur Stage + Sprite + easings (timeline + scrub)
- `scenes.jsx` — composants visuels (HeroSprayShot, VolumetricMist, FineSpray, AmbientMotes, NoteOverlay, HookOverlay, CTASticker, MahatahSound Web Audio engine)
- `videos.jsx` — VideoA / VideoB / VideoC avec timelines audio
- `assets/` — bottle-cutout.png, hero-spray.png

### Décision : générer les vrais MP4 via pipeline Playwright + ffmpeg

Pour passer du prototype interactif à des MP4 uploadables sur TikTok :

- `render.html` — page de capture, désactive RAF de Stage via `window.__renderMode`
- `render-audio.html` — rejoue MahatahSound dans `OfflineAudioContext` pour générer le WAV (audio identique au prototype, sans réimplémenter le DSP)
- `render-videos.mjs` — Node 22 + Playwright Chromium headless, step 420 frames @ 30 fps via `flushSync` + screenshots JPEG q=92, ffmpeg encode H.264 + AAC
- `vendor/` + `node_modules/` auto-installés au premier run, gitignorés

**Optimisations clés trouvées en route** :
- JPEG q=92 vs PNG → ~10× plus rapide à l'écriture disque
- `ReactDOM.flushSync(() => setTick(x => x+1))` dans `__forceTick` = re-render synchrone, élimine les RAF round-trips → 0.18 s/frame au lieu de 3 s
- Tunnel HTTPS sandbox → les CDN unpkg avec `crossorigin` étaient bloqués → vendoré React/Babel local

**Résultat** : 3 MP4 commités (1080×1920, H.264 + AAC mono 48 kHz, 14 s) :
- `tiktok-A.mp4` — 6.4 MB, ASMR Hero
- `tiktok-B.mp4` — 5.5 MB, Reveal flacon → spray
- `tiktok-C.mp4` — 5.8 MB, Editorial dark

Commits : `fd133d9` (prototype), `667a14f` (renders + pipeline).

---

## Phase 2 — Pivot stratégique

### Question budget : "45-60 € pour une seule vidéo Veo 3 Standard ?"

Analyse honnête : non, ça ne tient pas à 600-800 vues moyennes. Veo 3 Standard à $0.75/s n'a de sens que pour des hero films / paid ads. Pour de l'organique, 3 alternatives :

| Niveau | Stack | Coût/vidéo |
|---|---|---|
| Cheap | Pexels + iPhone + ElevenLabs + Remotion | ~0.50 € |
| Hybrid | Veo 3 Fast (1 plan) + Pexels + Remotion | ~5-8 € |
| Premium | Veo 3 Standard plein gaz | 45-60 € |

### Recherche TikTok virality pour parfum (mai 2026)

- **PerfumeTok** : 2.3 milliards de vues sur le hashtag, 1.1 milliard sur #FragranceTikTok
- **TikTok drive 45%** des achats fragrance social-driven aux US
- **3-second rule** : 71% de la rétention se joue dans les 3 premières secondes
- Algorithme 2026 : watch time = **74% du signal FYP**
- **UGC bat le brandé de 93%** sur la rétention
- Hooks dominants en niche : **Compliment Monster** (80% des viraux), **Dupe**, **POV sillage**
- Cas d'école 2026 : Bianco Latte de Giardini di Toscana — 85% des acheteurs ont reçu un compliment non-sollicité
- **Algo deprioritize l'AI-generated content** quand c'est trop évident

### 3 scénarios calibrés proposés

- **A — Compliment Monster** (UGC-style, le plus viral-prone)
- **B — POV : tu es son sillage** (cinématique, niche-coherent)
- **C — Test sur 10 inconnus dans le métro** (UGC documentaire, format Bianco Latte)

Scénario C bootstrappé en Remotion (commit `18bc2bf`) avec :
- 9 prompts Veo 3 shot-par-shot dans `varC.config.ts`
- 3 lignes voix off ElevenLabs Charlotte FR
- Composition React `EscapadeGourmandeVarC.tsx`

---

## Phase 3 — Architecture finale

### Différenciateur business confirmé

> **« On vend la connaissance de petites maisons de parfum de niche peu connues. Moins de concurrence, tarif d'achat meilleur. »**

Cette positioning change tout :
- Terrain TikTok quasi vide
- Légitimité d'expert par défaut
- Petits perfumeurs paieront pour exister (revenu B2B en plus du retail)
- Format "discovery" nativement viral (curiosity gap)
- Audience qualifiée → conversion 3-5× supérieure

### Stack production retenue (Phase 2 directe)

```
█ PRODUCTION (~89 €/mo, +30 € variable Veo 3 absorbé par crédits GCP)
├─ HeyGen Avatar V Team (89 €/mo)    → 60% des vidéos (talking-head)
│   - Avatar V = le moins détecté en 2026 sur TikTok
│   - Voix FR natives + musique royalty-free + sous-titres auto INCLUS
│   - 90 min de vidéo/mois → suffit pour 30 vidéos/mois
│   - PAS de ElevenLabs, PAS de Suno : tout est dans HeyGen Team
├─ Veo 3 Fast via Vertex AI (~30 €/mo)   → hero shots cinéma uniquement
│   - 300 € de crédits dispos = 5-6 mois gratuit
├─ Pexels API (gratuit)              → B-roll 4K Italie/France/lifestyle
└─ Remotion (gratuit)                → Mardi (Marque) + Dimanche (Histoire)
```

### Architecture automatisation (le critère bloquant)

```
DIMANCHE 18h00 (cron Railway sur n8n existant)
  └─ Stella réveille content-engine (nouveau service)
       │
       ├─ 1. Pick : "quelle marque, quel angle, quelle vidéo cette semaine ?"
       ├─ 2. Script Gen via Claude API : 7 scripts adaptés à 7 formats
       ├─ 3. En parallèle :
       │     ├─ HeyGen API → render avatar (voix FR + musique inclus dans Team)
       │     ├─ Vertex AI Veo 3 API → 1-2 hero shots
       │     └─ Pexels API → B-rolls
       ├─ 4. Remotion render (Railway container) pour vidéos premium
       ├─ 5. Upload Google Drive : "PB / Week 19 / 7 MP4"
       └─ 6. Telegram notif

DIMANCHE 18h30 — Utilisateur sur iPhone
  └─ Reçoit notif → ouvre Drive → télécharge

LUN→DIM — Utilisateur sur iPhone (5 min/jour)
  └─ TikTok app → import → coller caption → publier
```

### Calendrier éditorial : 7 formats × discovery angle

| Jour | Format | Hook type |
|---|---|---|
| Lun | 🔍 Maison oubliée du jour | Insider knowledge / FOMO |
| Mar | 📜 L'histoire qu'aucun mag ne raconte | Anti-establishment narrative (premium Remotion) |
| Mer | ⚔️ Niche connu vs niche secret | Dupe inversé (rare = supérieur) |
| Jeu | 🤫 Le parfum confidentiel à Paris | Exclusivité géographique |
| Ven | 👃 Test à l'aveugle (UGC iPhone) | Réaction réelle, social proof |
| Sam | 🏛️ Lieu / atelier (Veo 3 cinematic) | Immersif, "POV: tu rentres dans..." |
| Dim | 🌍 Top 5 marques inconnues (premium Remotion) | Listicle bingeable |

### 10 marques à développer

| Marque | Vidéos/mois | Angle narratif |
|---|---|---|
| 🌟 **AB Signature** (3 nouvelles refs, marque maison) | 8 | Storytelling fondateur + lancement + 3 sous-séries |
| **Maison Mataha** | 3 | Escapade Gourmande (recherche déjà faite) |
| **Les Eaux Primordiales** (Arnaud Poulain) | 3 | Anti-mainstream, ex-Hermès, indépendance |
| **Jousset** | 3 | Parfumeur français indé, savoir-faire |
| **Eminence i Matti** | 2 | Italian craft, héritage parfumerie italienne |
| **Badar** | 2 | Niche français peu distribué |
| **New Notes** | 2 | Marco Maggi, parfumerie italienne contemporaine |
| **Fomowa Paris** | 2 | Matières rares, Paris contemporain |
| **Plume Impression** | 2 | Aquarelle olfactive, niche français récent |
| **Matca** | 2 | Niche tchèque (rareté géographique = scoop FR) |

**TOTAL : 30 vidéos / mois avec 0 répétition d'angle**

### Plan d'exécution 10 jours

- **J1** : recherche approfondie 10 marques + bootstrap content-engine Railway service
- **J2-3** : connecteur Shopify + templates 7 formats + HeyGen API wrapper
- **J4-5** : Pexels API + Veo 3 wrapper + Remotion compositions × 2 (mardi/dimanche)
- **J6** : n8n workflow cron + test end-to-end avec 1 vidéo
- **J7** : génération Semaine 1 complète (7 vidéos)
- **J8-10** : utilisateur poste, mesure perf, ajustements pour Semaine 2

### Inputs requis de l'utilisateur

1. `HEYGEN_API_KEY` (Team plan — inclut voix FR + musique, donc PAS d'ElevenLabs ni Suno)
2. Vertex AI service-account JSON (drag-drop dans repo, gitignored)
3. Google Drive folder partagé avec service account
4. Username Telegram pour notifs
5. Détails 3 nouvelles refs **AB Signature** (nom, notes, prix, photo, histoire)
6. (Optionnel) Validation des fiches marques après recherche web

---

## État actuel de la branche

### Commits

- `fd133d9` — Maison Mataha TikTok prototype (HTML/CSS/JS, showcase + standalone variants)
- `667a14f` — Pipeline rendu MP4 Playwright + ffmpeg, 3 MP4 livrés
- `18bc2bf` — Bootstrap Remotion + Veo 3 + ElevenLabs (TypeScript)

### Structure repo

```
stella-shopify-app/static/maison-mataha/
├── index.html, a.html, b.html, c.html      # Showcase + standalone
├── animations.jsx, scenes.jsx, videos.jsx  # Prototype interactif
├── render.html, render-audio.html          # Pages de capture
├── render-videos.mjs                       # Pipeline rendu MP4
├── vendor/                                 # React + Babel (gitignored)
├── renders/
│   ├── tiktok-A.mp4 (6.4 MB)
│   ├── tiktok-B.mp4 (5.5 MB)
│   └── tiktok-C.mp4 (5.8 MB)
└── remotion/                               # Phase 2 bootstrap
    ├── package.json (258 packages)
    ├── src/
    │   ├── brand/tokens.ts
    │   ├── compositions/EscapadeGourmandeVarC.tsx
    │   ├── compositions/varC.config.ts
    │   └── lib/{veo,elevenlabs,env}.ts
    ├── scripts/{generate-veo,generate-voiceover,fetch-stock}.ts
    └── README.md
```

### Reste à faire

- [ ] Recherche fiches marques (10 marques)
- [ ] Bootstrap `content-engine/` service Railway
- [ ] Connecteur Shopify catalogue
- [ ] HeyGen API wrapper (Team plan)
- [ ] Templates 7 formats × 7 jours
- [ ] n8n cron workflow + Drive + Telegram notif
- [ ] Premier livrable : Semaine 1 (7 vidéos)

---

## Décisions clés (référence rapide)

1. **Pas de Veo 3 Standard pour de l'organique** — trop cher, $0.75/s. Veo 3 Fast à $0.15/s suffit largement.
2. **HeyGen Avatar V Team plutôt que Argil** — plus grande lib d'avatars, Avatar V passe inaperçu dans 50-60% des cas vs ~30% pour Argil. Voix + musique inclus → pas besoin de Suno/ElevenLabs en plus.
3. **Avatar inventé custom** plutôt que Digital Twin de la personne réelle — préserve la vie privée, persona "curateur" stable indépendamment du staff.
4. **Discovery > Ad** — chaque vidéo ouvre un curiosity gap sur l'inconnu, pas une présentation produit classique.
5. **Automation totale obligatoire** — l'utilisateur abandonnera si manuel. n8n cron + content-engine + Drive output.
6. **Rotation 30 vidéos/mois** sur 10 marques avec AB Signature en flagship (8/mois).
7. **TOI sur iPhone 1-2×/semaine** restera nécessaire pour ne pas tomber dans la bulle "100% AI" de l'algo TikTok.

---

## Métadonnées

- **Projet RAG** : `MAISON_MATAHA_TIKTOK`
- **Collection** : `decisions` + `knowledge`
- **Source** : `conversation-2026-05-04`
- **Tags** : `tiktok`, `content-engine`, `heygen`, `veo3`, `remotion`, `niche-perfume`, `automation`, `phase-2`
