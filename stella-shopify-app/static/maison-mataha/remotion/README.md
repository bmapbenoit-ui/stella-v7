# Maison Mataha · Remotion Pipeline

Generate cinematic 1080×1920 TikTok videos for Maison Mataha *Escapade Gourmande* using **Remotion** for composition + **Veo 3.1** (Vertex AI) for shot generation + **ElevenLabs** for French voice-over + **Whisper** for word-level captions.

---

## Stack

| Layer | Technology | Why |
|---|---|---|
| Composition | [Remotion 4](https://www.remotion.dev) | React-based, frame-precise, production-grade |
| Avatar + voice + music | [HeyGen Team Avatar V](https://www.heygen.com) | All-in-one: avatar V (most undetected on TikTok 2026) + 200+ voices + royalty-free music + auto captions |
| Shot generation | Veo 3.1 (Vertex AI) | Photorealistic 8s clips, character consistency via reference images, synced audio |
| Captions | [`@remotion/captions`](https://www.remotion.dev/docs/captions) + Whisper.cpp | Native TikTok-style word-by-word reveal (used only when assembling outside HeyGen) |
| Stock fallback | [Pexels](https://www.pexels.com/api/) | Free 4K HD library for B-roll + reference images |
| Optional fallback | [ElevenLabs](https://elevenlabs.io) `eleven_multilingual_v2` | Backup voice provider if HeyGen voices don't fit a specific brief — not part of the standard pipeline |

---

## One-time setup

1. **Install** Node 22+ deps:
   ```bash
   cd stella-shopify-app/static/maison-mataha/remotion
   npm install
   ```

2. **Add credentials** — copy `.env.example` to `.env` and fill in:
   ```bash
   cp .env.example .env
   $EDITOR .env
   ```
   - `ELEVENLABS_API_KEY` — yours
   - `GOOGLE_CLOUD_PROJECT` — your GCP project ID with Vertex AI enabled
   - `GOOGLE_APPLICATION_CREDENTIALS` — path to your service account JSON (gitignored)
   - `PEXELS_API_KEY` — free at <https://www.pexels.com/api/>

3. **(Recommended) Install Remotion Superpowers** — Claude Code plugin with 5 MCP servers (music, captions, stock, voice, AI review):
   ```
   /plugin marketplace add DojoCodingLabs/remotion-superpowers
   /plugin install remotion-superpowers@remotion-superpowers
   /setup
   ```

---

## Generate a video (Scenario C — "Test sur 10 inconnus dans le métro")

```bash
# 1. Fetch reference images (Pexels) — once per project
npm run stock:fetch

# 2. Generate the 9 Veo 3 shots (~10-15 min, ~$45-60 in Vertex AI credits)
npm run veo:generate

# 3. Render the 3 ElevenLabs voice lines (~3 sec, ~$0.15)
npm run voice:generate

# 4. Render the final MP4
npm run render:varC
# → out/escapade-gourmande-varC.mp4
```

Or all in one shot:
```bash
npm run all
```

---

## Live preview studio

```bash
npm run studio
# opens http://localhost:3000 with hot-reload composition preview
```

---

## Project layout

```
remotion/
├── src/
│   ├── brand/tokens.ts           ← colors, fonts, motion easings
│   ├── compositions/
│   │   ├── varC.config.ts        ← Veo prompts + VO lines + timeline
│   │   └── EscapadeGourmandeVarC.tsx  ← the React composition
│   ├── lib/
│   │   ├── env.ts                ← type-safe env loading (zod)
│   │   ├── veo.ts                ← Vertex AI Veo 3 client
│   │   └── elevenlabs.ts         ← ElevenLabs TTS wrapper
│   ├── Root.tsx                  ← Remotion composition registry
│   └── index.ts                  ← entry point
├── scripts/
│   ├── generate-veo.ts           ← batch Veo 3 generation (3 shots in parallel)
│   ├── generate-voiceover.ts     ← batch ElevenLabs TTS
│   └── fetch-stock.ts            ← Pexels reference image fetcher
├── assets/
│   ├── brand/                    ← bottle-cutout.png, hero-spray.png (symlinks)
│   ├── stock/                    ← Pexels references (gitignored, regen with stock:fetch)
│   ├── veo/                      ← Veo 3 outputs (gitignored, regen with veo:generate)
│   ├── voiceover/                ← ElevenLabs outputs (gitignored)
│   └── music/                    ← background music bed
├── public/                       ← symlinks consumed by Remotion's staticFile()
├── .env.example
└── package.json
```

---

## Adding a new variant

1. Duplicate `varC.config.ts` → `varD.config.ts`. Edit shot prompts + voice lines + timeline.
2. Duplicate `EscapadeGourmandeVarC.tsx` → `EscapadeGourmandeVarD.tsx`. Adjust scene composition.
3. Register in `Root.tsx`:
   ```tsx
   <Composition id="EscapadeGourmande_VarD" component={EscapadeGourmandeVarD} ... />
   ```
4. Add a render script in `package.json`:
   ```json
   "render:varD": "remotion render src/index.ts EscapadeGourmande_VarD out/escapade-gourmande-varD.mp4"
   ```

---

## Costs (May 2026)

| Item | Per video | Per 5-variant batch |
|---|---|---|
| Veo 3.1 — 9 shots × ~6s @ $0.75/s | ~$40 | ~$200 |
| ElevenLabs — ~7s of speech | ~$0.15 | ~$0.75 |
| Whisper.cpp captions | $0 (local) | $0 |
| Pexels references | $0 | $0 |
| Remotion render | $0 (local CPU) | $0 |
| **Total** | **~$40** | **~$200** |

Veo 3 *Fast* mode (3-4× cheaper) is also wired — set `VEO_MODEL=veo-3.1-fast-generate-preview` in `.env`.

---

## Why these choices

- **Veo 3 direct via Vertex AI** instead of through KIE/Replicate: 30-40% cheaper, zero middleman, tightest quota control. The Remotion Superpowers plugin uses KIE by default, which is fine for prototyping but wasteful at production scale.
- **`@google/genai` SDK** is the unified Google AI SDK (replaces older `@google-cloud/aiplatform` for Veo).
- **Static configs in `varC.config.ts`** so Veo regen + Remotion render share one source of truth — change a prompt, both pipelines pick it up.
- **Symlink `public/` → `assets/`** so Remotion's `staticFile()` resolves seamlessly without copying gigabytes of generated MP4s.
- **TypeScript strict mode** + `zod` env validation — fail at startup, not 30 minutes into a Veo generation.

---

## Reference

- Scenario rationale + virality research: see git log entry referencing PerfumeTok 2026 trends, the "Compliment Monster" hook, and TikTok's 70% / 3-second retention rule.
- [Veo 3.1 prompting guide](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1)
- [Remotion TikTok template](https://github.com/remotion-dev/template-tiktok)
- [ElevenLabs Multilingual V2 voice library](https://elevenlabs.io/voice-library)
