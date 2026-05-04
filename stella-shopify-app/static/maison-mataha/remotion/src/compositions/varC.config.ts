// Scenario C — "Test sur 10 inconnus dans le métro"
// Single source of truth for: Veo 3 shot prompts + voice-over lines + scene
// timings. Both `scripts/generate-veo.ts` and the Remotion composition
// import this so a tweak here propagates everywhere.

import type { VeoShot } from '../lib/veo.js';
import type { VoiceLine } from '../lib/elevenlabs.js';

// ─── Character anchors (locked across all shots) ────────────────────────────
// Re-stated verbatim in every prompt so Veo doesn't drift off-character.
const CAMILLE = `Camille, 30 ans, cheveux brun acajou ondulés mi-longs avec une raie au milieu, peau claire avec quelques taches de rousseur discrètes, yeux noisette, lèvres rosées sans rouge à lèvres voyant, manteau en laine beige sable ouvert sur un pull noir col roulé, jeans noir slim, bottines en cuir camel, foulard en soie marron clair noué autour du cou, sac à main en cuir noir minimaliste`;

const PALETTE = `palette chaude dorée et brun chaud, lumière golden hour de fin d'après-midi parisien filtrant à travers les grilles du métro, contraste cinématographique, grain pellicule 35mm subtil, profondeur de champ shallow`;

const CAMERA_BASE = `tournée en 35mm anamorphique, 24fps, balance des blancs 4500K`;

// ─── Veo 3 shots ───────────────────────────────────────────────────────────
// 9 clips. Each is generated as 8s by Veo 3 then trimmed to its on-screen
// window in the Remotion timeline. Reference images are layered:
//   • portrait-camille.jpg → identity lock for protagonist shots
//   • bottle-cutout.png    → the Maison Mataha flacon
//   • metro-paris.jpg      → station style/lighting reference
export const shots: VeoShot[] = [
  {
    id: '01-wrist-spritz',
    prompt: `Macro close-up extrême sur un poignet de femme, ${CAMILLE.split(',').slice(0, 4).join(',')}. Une main applique un flacon de parfum carré en verre transparent ambré avec capuchon doré sculpté en pétales (Maison Mataha Escapade Gourmande). Le spray s'active en slow-motion: brouillard doré atomisé qui jaillit et flotte. Reflet du flacon doré sur la peau. ${PALETTE}. ${CAMERA_BASE}. Audio: spray ASMR feutré avec léger sifflement haute-fréquence, sous-couche de bourdon de métro lointain.`,
    referenceImages: ['assets/brand/bottle-cutout.png', 'assets/stock/parisienne-portrait.jpg'],
    negativePrompt: 'cartoon, animation, low quality, blurry, distorted hands, deformed fingers',
    durationSeconds: 6,
  },
  {
    id: '02-metro-entry',
    prompt: `Plan large suivant Camille de dos qui descend les escaliers d'une station de métro parisienne (style station Saint-Michel ou Cluny-La Sorbonne). ${CAMILLE}. Elle tient discrètement le flacon Maison Mataha dans sa main droite. La lumière chaude de fin d'après-midi vient de derrière elle, projetant une silhouette dorée. Les passants flous en avant-plan. ${PALETTE}. ${CAMERA_BASE}. Audio: bruit de pas sur carrelage, écho de la station, annonce SNCF lointaine "votre attention s'il vous plaît".`,
    referenceImages: ['assets/stock/parisienne-portrait.jpg', 'assets/stock/metro-paris.jpg'],
    durationSeconds: 5,
  },
  {
    id: '03-stranger-react-1',
    prompt: `Plan poitrine d'une femme française élégante d'une trentaine d'années aux cheveux blonds platine attachés en chignon haut, manteau crème, dans une rame de métro parisien. Elle tourne lentement la tête, lève les sourcils en signe de surprise agréable, semble humer l'air. Réaction subtile, naturelle, pas exagérée. Profondeur de champ shallow, fond flou avec barres de métal du métro. ${PALETTE}. ${CAMERA_BASE}. Audio: bruit de roulement de rame de métro, conversations étouffées en arrière-plan.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '04-stranger-react-2',
    prompt: `Plan rapproché poitrine d'un homme français de 35-40 ans en costume sombre bien coupé, barbe courte taillée, lunettes en écaille, debout dans un wagon de métro. Il regarde son téléphone, puis lève soudainement les yeux vers la gauche du cadre, pupilles qui se dilatent légèrement, expression intriguée et touchée par un parfum invisible. Il prend une discrète inspiration. ${PALETTE}. ${CAMERA_BASE}. Audio: rame qui roule, signal sonore de fermeture des portes au loin.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '05-stranger-react-3',
    prompt: `Plan moyen d'une jeune femme étudiante de 22 ans aux cheveux noirs courts coupe pixie, doudoune oversize verte sapin, écouteurs sans fil, debout adossée contre une porte de métro. Elle retire un écouteur, sourit largement, tape doucement sur l'épaule de quelqu'un hors-cadre comme pour dire "c'est quoi ce parfum ?". Mouvement spontané, geste authentique. ${PALETTE}. ${CAMERA_BASE}. Audio: léger tintement de boucles d'oreilles, rame en mouvement.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '06-stranger-react-4',
    prompt: `Plan rapproché d'une femme sénior française élégante de 65 ans, cheveux gris coupe carré court impeccable, manteau Chanel-style en tweed beige et noir, perles aux oreilles, assise sur un siège de métro. Elle ferme les yeux pendant deux secondes, sourire nostalgique apparaît au coin des lèvres, comme si un parfum lui rappelait un souvenir précieux. Très subtil et émouvant. ${PALETTE}. ${CAMERA_BASE}. Audio: bruit de roulement de rame étouffé, ambiance feutrée.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '07-stranger-react-5',
    prompt: `Plan rapproché d'un homme français de 28 ans, cheveux bruns ondulés mi-longs, manteau en cuir noir, écharpe bordeaux, dans un wagon de métro. Il sourit timidement vers la gauche du cadre, semble vouloir parler, hésite, finit par mouvoir les lèvres en silence "wow". Capté en slow-motion partiel sur le sourire qui se forme. ${PALETTE}. ${CAMERA_BASE}. Audio: rame qui ralentit en approche de station, freins.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '08-stranger-react-6',
    prompt: `Plan moyen rapproché d'une femme de 40 ans look "rich mom" parisienne, cheveux brun chocolat coupe carré long avec wavy blowout, trench beige Burberry-style, foulard Hermès rouge et or, sac matelassé noir. Elle tourne franchement la tête, fronce les sourcils en signe d'admiration intense, pince les lèvres dans une moue impressionnée. Mouvement décidé, énergique, reconnaissable. ${PALETTE}. ${CAMERA_BASE}. Audio: bruit de roulement constant de la rame.`,
    referenceImages: ['assets/stock/metro-paris.jpg'],
    durationSeconds: 4,
  },
  {
    id: '09-final-bottle',
    prompt: `Plan macro lent (slow zoom out) sur le flacon Maison Mataha Escapade Gourmande tenu dans la main de Camille, en haut d'un escalier qui débouche sur un toit parisien au coucher de soleil. Vue floue derrière elle de toits zinc parisiens et de la Tour Eiffel à l'horizon. Le verre du flacon attrape la lumière dorée orange du soleil couchant et projette des reflets dans la caméra. La main de Camille se serre doucement autour du flacon. ${PALETTE} mais saturé de chaleur orangée. ${CAMERA_BASE}. Audio: dernières chimes cristallines décroissantes, vent doux, fin paisible.`,
    referenceImages: ['assets/brand/bottle-cutout.png', 'assets/stock/parisienne-portrait.jpg', 'assets/stock/paris-rooftop-sunset.jpg'],
    durationSeconds: 6,
  },
];

// ─── Voice-over lines ──────────────────────────────────────────────────────
// Charlotte FR voice. Each line is rendered separately so we control the
// silence + timing in Remotion (no auto-stitch artifacts).
export const voiceLines: VoiceLine[] = [
  {
    id: 'vo-01-hook',
    text: `<break time="0.2s"/>J'ai testé Escapade Gourmande sur dix inconnus dans le métro à Paris.<break time="0.3s"/>`,
  },
  {
    id: 'vo-02-result',
    text: `<break time="0.1s"/>Sept sur dix m'ont demandé.<break time="0.4s"/>`,
  },
  {
    id: 'vo-03-cta',
    text: `Maison Mataha. Escapade Gourmande.<break time="0.25s"/>Sur planète beauté point com.`,
  },
];

// ─── Timeline (frame-precise scene boundaries at 30 fps) ───────────────────
export const timeline = {
  hook:           { start: 0,   durSec: 3 },     // VO #1 + black-bg text
  wristSpritz:    { start: 3,   durSec: 2 },     // shot 01 trimmed
  metroEntry:     { start: 4.5, durSec: 1 },     // shot 02 brief
  reactions:      { start: 5,   durSec: 6 },     // 6 micro-cuts × 1s
  resultStat:     { start: 11,  durSec: 2 },     // VO #2 + big stat text
  brandCTA:       { start: 13,  durSec: 1 },     // shot 09 + VO #3 + URL
} as const;
