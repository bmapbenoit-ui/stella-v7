// ═══════════════════════════════════════════════════════════════════════════
// Scenario C — "Test sur 10 inconnus dans le métro"
// 14s · 1080x1920 · 30fps. The hooks + scene structure come from the
// research-backed PerfumeTok virality playbook (see ../../README.md).
// ═══════════════════════════════════════════════════════════════════════════

import { AbsoluteFill, Audio, Sequence, Video, Img, staticFile, useCurrentFrame, useVideoConfig, interpolate, Easing } from 'remotion';
import { TransitionSeries, linearTiming } from '@remotion/transitions';
import { fade } from '@remotion/transitions/fade';
import { colors, fonts, type, shadows, motion } from '../brand/tokens.js';
import { timeline } from './varC.config.js';

const sec = (s: number, fps: number) => Math.round(s * fps);

// ─── Scene 1: Hook (0–3s) ──────────────────────────────────────────────────
function HookScene() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = interpolate(frame, [0, sec(0.4, fps)], [0, 1], { extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic) });
  const exit  = interpolate(frame, [sec(2.6, fps), sec(3, fps)], [0, 1], { extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.in(Easing.cubic) });
  const opacity = enter * (1 - exit);
  const ty = (1 - enter) * 30;

  return (
    <AbsoluteFill style={{ background: colors.black, justifyContent: 'center', alignItems: 'center', padding: 80 }}>
      <div style={{
        fontFamily: fonts.display,
        fontSize: 86,
        fontStyle: 'italic',
        fontWeight: 300,
        color: colors.cream,
        textAlign: 'center',
        lineHeight: 1.1,
        letterSpacing: '-0.01em',
        opacity,
        transform: `translateY(${ty}px)`,
        textShadow: shadows.textOnDark,
      }}>
        J'ai testé <span style={{ color: colors.goldHighlight }}>Escapade Gourmande</span><br/>
        sur 10 inconnus<br/>
        dans le métro à Paris.
      </div>
    </AbsoluteFill>
  );
}

// ─── Scene 2: Wrist spritz (3–5s) ──────────────────────────────────────────
function WristSpritzScene({ src }: { src: string }) {
  return (
    <AbsoluteFill style={{ background: colors.black }}>
      <Video src={src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted />
      {/* Subtle vignette to focus the eye */}
      <AbsoluteFill style={{
        background: 'radial-gradient(ellipse 80% 70% at 50% 50%, transparent 50%, rgba(0,0,0,0.45) 100%)',
        pointerEvents: 'none',
      }}/>
    </AbsoluteFill>
  );
}

// ─── Scene 3: Metro entry (4.5–5.5s) — overlap blends with reactions ──────
function MetroEntryScene({ src }: { src: string }) {
  return (
    <AbsoluteFill>
      <Video src={src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted />
    </AbsoluteFill>
  );
}

// ─── Scene 4: Stranger reactions (5–11s) ───────────────────────────────────
const REACTIONS = [
  { src: 'veo/03-stranger-react-1.mp4', label: 'Personne #2 — souriante', accent: '« mhmm... »' },
  { src: 'veo/04-stranger-react-2.mp4', label: 'Personne #4 — costume', accent: '« c\'est quoi ? »' },
  { src: 'veo/05-stranger-react-3.mp4', label: 'Personne #5 — étudiante', accent: '« hé... »' },
  { src: 'veo/06-stranger-react-4.mp4', label: 'Personne #7 — élégante', accent: '« mmm... »' },
  { src: 'veo/07-stranger-react-5.mp4', label: 'Personne #8 — sourire', accent: '« wow »' },
  { src: 'veo/08-stranger-react-6.mp4', label: 'Personne #9 — admiration', accent: '« il sent quoi ? »' },
];

function ReactionCut({ src, label, accent }: { src: string; label: string; accent: string }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = interpolate(frame, [0, 5], [0, 1], { extrapolateRight: 'clamp' });
  return (
    <AbsoluteFill style={{ background: colors.black }}>
      <Video src={staticFile(src)} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted />
      {/* Top label chip */}
      <div style={{
        position: 'absolute', top: 140, left: '50%', transform: `translateX(-50%) translateY(${(1 - enter) * -10}px)`,
        opacity: enter,
        background: 'rgba(255, 248, 234, 0.92)',
        backdropFilter: 'blur(8px)',
        color: colors.brownDark,
        padding: '14px 26px',
        borderRadius: 100,
        fontFamily: fonts.body,
        fontSize: 26,
        fontWeight: 600,
        letterSpacing: '0.04em',
        boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
      }}>{label}</div>
      {/* Bottom accent quote */}
      <div style={{
        position: 'absolute', bottom: 220, left: 60, right: 60,
        textAlign: 'center',
        fontFamily: fonts.display,
        fontStyle: 'italic',
        fontSize: 64,
        fontWeight: 300,
        color: colors.ivory,
        textShadow: shadows.textOnDark,
        opacity: enter,
      }}>{accent}</div>
    </AbsoluteFill>
  );
}

// ─── Scene 5: Result stat (11–13s) ─────────────────────────────────────────
function ResultStatScene() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const stamp = motion.easeOutBack(Math.min(1, frame / sec(0.6, fps)));
  const numberScale = 0.6 + 0.4 * stamp;
  return (
    <AbsoluteFill style={{ background: colors.black, justifyContent: 'center', alignItems: 'center' }}>
      {/* Big "7 / 10" */}
      <div style={{
        fontFamily: fonts.display,
        fontSize: 380,
        fontWeight: 300,
        fontStyle: 'italic',
        color: colors.goldHighlight,
        lineHeight: 1,
        transform: `scale(${numberScale})`,
        textShadow: shadows.textOnDark,
      }}>7<span style={{ color: colors.cream, fontSize: 200, opacity: 0.5 }}>/10</span></div>
      <div style={{
        fontFamily: fonts.body,
        fontSize: 38,
        fontWeight: 500,
        color: colors.cream,
        letterSpacing: '0.32em',
        textIndent: '0.32em',
        textTransform: 'uppercase',
        marginTop: 30,
        opacity: Math.min(1, (frame - sec(0.4, fps)) / sec(0.4, fps)),
      }}>m'ont demandé</div>
    </AbsoluteFill>
  );
}

// ─── Scene 6: Brand CTA (13–14s) ───────────────────────────────────────────
function BrandCTAScene({ src }: { src: string }) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const enter = interpolate(frame, [0, sec(0.5, fps)], [0, 1], { extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic) });
  return (
    <AbsoluteFill style={{ background: colors.black }}>
      <Video src={src} style={{ width: '100%', height: '100%', objectFit: 'cover' }} muted />
      <AbsoluteFill style={{ background: 'linear-gradient(180deg, transparent 50%, rgba(10,5,2,0.85) 100%)', pointerEvents: 'none' }}/>
      {/* Wordmark */}
      <div style={{
        position: 'absolute', bottom: 280, left: 0, right: 0,
        textAlign: 'center',
        opacity: enter,
        transform: `translateY(${(1 - enter) * 20}px)`,
      }}>
        <div style={{
          fontFamily: fonts.body,
          fontSize: 18, fontWeight: 600,
          letterSpacing: '0.6em', textIndent: '0.6em',
          color: colors.goldHighlight,
          textTransform: 'uppercase',
          marginBottom: 14,
        }}>Maison</div>
        <div style={{
          fontFamily: fonts.display,
          fontSize: 92, fontWeight: 400,
          letterSpacing: '0.42em', textIndent: '0.42em',
          color: colors.ivory,
          textTransform: 'uppercase',
          textShadow: shadows.textOnDark,
        }}>MATAHA</div>
        <div style={{
          fontFamily: fonts.display,
          fontStyle: 'italic',
          fontSize: 42,
          color: colors.cream,
          marginTop: 18,
          opacity: 0.85,
        }}>Escapade Gourmande</div>
      </div>
      {/* URL strap */}
      <div style={{
        position: 'absolute', bottom: 140, left: 0, right: 0, textAlign: 'center',
        opacity: enter,
      }}>
        <div style={{
          display: 'inline-block',
          background: 'rgba(255, 248, 234, 0.95)',
          color: colors.brownDark,
          padding: '20px 44px',
          borderRadius: 100,
          fontFamily: fonts.body,
          fontSize: 32,
          fontWeight: 700,
          letterSpacing: '0.04em',
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        }}>✦ planetebeauty.com</div>
      </div>
    </AbsoluteFill>
  );
}

// ─── Composition root ──────────────────────────────────────────────────────
export const EscapadeGourmandeVarC: React.FC = () => {
  const { fps } = useVideoConfig();
  const t = (s: keyof typeof timeline) => sec(timeline[s].start, fps);
  const d = (s: keyof typeof timeline) => sec(timeline[s].durSec, fps);

  return (
    <AbsoluteFill style={{ background: colors.black }}>
      {/* Scene 1: Hook — 0-3s */}
      <Sequence from={0} durationInFrames={d('hook')}>
        <HookScene />
      </Sequence>

      {/* Scene 2: Wrist spritz — 3-5s */}
      <Sequence from={t('wristSpritz')} durationInFrames={d('wristSpritz')}>
        <WristSpritzScene src={staticFile('veo/01-wrist-spritz.mp4')} />
      </Sequence>

      {/* Scene 3: Metro entry overlap — 4.5-5.5s */}
      <Sequence from={t('metroEntry')} durationInFrames={d('metroEntry')}>
        <MetroEntryScene src={staticFile('veo/02-metro-entry.mp4')} />
      </Sequence>

      {/* Scene 4: 6 stranger reactions — 5-11s, 1s each */}
      <Sequence from={t('reactions')} durationInFrames={d('reactions')}>
        <TransitionSeries>
          {REACTIONS.map((r, i) => (
            <TransitionSeries.Sequence key={r.src} durationInFrames={sec(1, fps)}>
              <ReactionCut src={r.src} label={r.label} accent={r.accent} />
            </TransitionSeries.Sequence>
          )).flatMap((seq, i, arr) => i < arr.length - 1
            ? [seq, <TransitionSeries.Transition key={`t-${i}`} presentation={fade()} timing={linearTiming({ durationInFrames: sec(0.15, fps) })}/>]
            : [seq])}
        </TransitionSeries>
      </Sequence>

      {/* Scene 5: 7/10 stat — 11-13s */}
      <Sequence from={t('resultStat')} durationInFrames={d('resultStat')}>
        <ResultStatScene />
      </Sequence>

      {/* Scene 6: CTA — 13-14s */}
      <Sequence from={t('brandCTA')} durationInFrames={d('brandCTA')}>
        <BrandCTAScene src={staticFile('veo/09-final-bottle.mp4')} />
      </Sequence>

      {/* ─── Audio layer ─── */}
      <Sequence from={0}><Audio src={staticFile('voiceover/vo-01-hook.mp3')} volume={1} /></Sequence>
      <Sequence from={t('resultStat')}><Audio src={staticFile('voiceover/vo-02-result.mp3')} volume={1} /></Sequence>
      <Sequence from={t('brandCTA') - sec(0.3, fps)}><Audio src={staticFile('voiceover/vo-03-cta.mp3')} volume={1} /></Sequence>
      {/* Background music ducked under the voice */}
      <Audio src={staticFile('music/escapade-bed.mp3')} volume={0.18} />
    </AbsoluteFill>
  );
};
