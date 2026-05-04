# CLAUDE.md — Stella V7 + PlanèteBeauty

> Contexte projet pour les futures sessions Claude Code. Lu automatiquement au démarrage.

## Vue d'ensemble

**Stella V7** = cerveau permanent de PlanèteBeauty (parfumerie de niche, e-commerce Shopify).

Architecture micro-services sur Railway :
- `embedding-service/` — BGE-M3 + reranker (Qdrant vectoriel)
- `context-engine/` — chef d'orchestre mémoire 3 niveaux (Redis short / Postgres medium / Qdrant long)
- `stella-shopify-app/` — app Shopify embedded + frontend chat
- `n8n` — orchestrateur de workflows
- Externe : OpenAI (images), Telegram, Mistral via vLLM RunPod, Google Drive

Public URLs :
- Context engine : `https://context-engine-production-e525.up.railway.app`
- Stella Shopify : `https://stella-shopify-app-production.up.railway.app`

---

## Projet actif : Content Engine TikTok

**Branche** : `claude/implement-maison-mataha-tiktok-1TXKG`
**Dernière session** : 2026-05-04
**Détails complets** : `docs/CONVERSATION-MAISON-MATAHA-TIKTOK-2026-05-04.md`

### Mission

Construire un système de production vidéo TikTok **automatisé à 100%**, qui génère **1 vidéo/jour** mettant en avant des **petites maisons de parfum de niche peu connues** distribuées par PlanèteBeauty.

**Différenciateur business** : pas de compétition sur les hits niche connus (Maison Margiela, Kayali) → curation de marques rares = terrain vide + marges meilleures + audience qualifiée.

### Stack retenu (Phase 2 — décision finale)

```
Production (~89 €/mo + Veo 3 absorbé par crédits GCP 5-6 mois)
├─ HeyGen Avatar V Team (89 €)        → 60% talking-head
│   - Voix FR + musique royalty-free + sous-titres TOUS INCLUS
│   - Donc PAS ElevenLabs, PAS Suno (économie ~15 €/mo)
├─ Vertex AI Veo 3 Fast (~30 €)       → hero shots cinéma
│   - 300 € de crédits utilisés       → ~5-6 mois gratuit
├─ Pexels API (gratuit)               → B-roll
└─ Remotion (gratuit)                 → vidéos premium mardi/dimanche
```

**Service à construire** : `content-engine/` (nouveau) sur Railway, déclenché chaque dimanche par n8n.

### 10 marques en rotation (30 vidéos/mois, 0 répétition)

| Marque | Vidéos/mois | Angle |
|---|---|---|
| 🌟 AB Signature (3 nouvelles refs) | 8 | Marque maison + lancements |
| Maison Mataha | 3 | Escapade Gourmande |
| Les Eaux Primordiales | 3 | Anti-mainstream, ex-Hermès |
| Jousset | 3 | Parfumeur français indé |
| Eminence i Matti | 2 | Italian craft |
| Badar | 2 | Niche français peu distribué |
| New Notes | 2 | Marco Maggi |
| Fomowa Paris | 2 | Matières rares |
| Plume Impression | 2 | Aquarelle olfactive |
| Matca | 2 | Niche tchèque (scoop FR) |

### 7 formats hebdo

- **Lun** Maison oubliée du jour (HeyGen)
- **Mar** Histoire qu'aucun mag ne raconte (Remotion premium)
- **Mer** Niche connu vs niche secret (HeyGen)
- **Jeu** Le parfum confidentiel à Paris (HeyGen)
- **Ven** Test à l'aveugle UGC iPhone
- **Sam** Lieu/atelier (Veo 3 cinematic + HeyGen)
- **Dim** Top 5 marques inconnues (Remotion premium)

---

## État de la branche

### Livré
- ✅ Prototype Maison Mataha 3 variations (HTML/CSS/JS) sous `stella-shopify-app/static/maison-mataha/`
- ✅ Pipeline render MP4 (Playwright + ffmpeg) → 3 MP4 dans `renders/`
- ✅ Bootstrap Remotion + Veo 3 + ElevenLabs sous `stella-shopify-app/static/maison-mataha/remotion/`

### À construire (sprint 10 jours)
- [ ] Recherche fiches 10 marques
- [ ] Service Railway `content-engine/`
- [ ] HeyGen API wrapper (Team plan)
- [ ] Connecteur Shopify catalogue
- [ ] n8n cron dimanche → Drive + Telegram notif
- [ ] Premier livrable : Semaine 1 (7 vidéos)

### Inputs en attente
- HEYGEN_API_KEY (Team plan)
- Vertex AI service-account JSON
- Google Drive folder + service account
- Username Telegram pour notifs
- Détails 3 nouvelles refs AB Signature

---

## Conventions importantes

### Git workflow
- Branche dédiée par feature/projet : `claude/<feature-slug>`
- Commits descriptifs en français OK
- Pas de `--amend` sur commits déjà poussés
- Pas de `--no-verify`
- `git push -u origin <branch>` toujours

### Stack
- Python 3.11+ pour services Railway (FastAPI)
- TypeScript strict pour Remotion + scripts
- Node 22 pour outils JS
- Polices : Cormorant Garamond (display) + Inter (body)
- Couleurs Maison Mataha : #c8984e (gold), #1a0e06 (brown dark), #f5ede0 (cream), #f4d27a (gold highlight)

### RAG / Mémoire
- Tous les snapshots, décisions, et conversations importantes doivent être sync'd vers le context-engine via `/learn` endpoint
- Script de sync : `scripts/sync-to-rag.sh`
- Project tag pour ce sujet : `MAISON_MATAHA_TIKTOK`

### Tests visuels
- Pour les outputs vidéo, extraire 1-2 frames JPEG via ffmpeg + lecture image
- Toujours vérifier dimensions (1080×1920), durée (14s), codecs (H.264 + AAC), fps (30)

---

## Pour les futures sessions Claude Code

Si on continue le content-engine, regarder en priorité :
1. `docs/CONVERSATION-MAISON-MATAHA-TIKTOK-2026-05-04.md` — historique des décisions
2. `stella-shopify-app/static/maison-mataha/remotion/README.md` — pipeline Remotion en place
3. `stella-shopify-app/static/maison-mataha/remotion/src/compositions/varC.config.ts` — exemple de format prompts Veo 3 + voix off
4. `context-engine/main.py` — endpoints `/learn`, `/snapshot`, `/chat` pour intégration RAG

Si on touche à Stella V7 core (pas le TikTok project) :
1. `README.md` racine — vue d'ensemble services
2. `context-engine/main.py` — schéma DB + endpoints
3. `embedding-service/main.py` — collections Qdrant + recherche

---

*Dernière mise à jour : 2026-05-04 (session content-engine TikTok)*
