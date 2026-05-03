// scenes.jsx — Maison Mataha "Escapade Gourmande" TikTok ASMR scenes
// Photo-real spray approach: SVG turbulence mist + many fine particles + key art

// ── Sound Engine (Web Audio) ────────────────────────────────────────────────
class MahatahSound {
  constructor() {
    this.ctx = null;
    this.master = null;
    this.scheduled = [];
    this.unlocked = false;
    this.muted = false;
    this.startedAt = 0; // audio time when "t=0" of the timeline started
  }
  ensure() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(this.ctx.destination);
  }
  unlock() {
    this.ensure();
    if (!this.ctx) return;
    if (this.ctx.state === 'suspended') this.ctx.resume();
    this.unlocked = true;
  }
  setMuted(m) {
    this.muted = m;
    if (this.master) this.master.gain.linearRampToValueAtTime(m ? 0 : 0.5, this.ctx.currentTime + 0.05);
  }

  // Warm pad
  pad(at, dur, freq = 130) {
    const ctx = this.ctx;
    const t0 = ctx.currentTime + at;
    const g = ctx.createGain();
    g.gain.value = 0;
    g.gain.linearRampToValueAtTime(0.16, t0 + 1.2);
    g.gain.setValueAtTime(0.16, t0 + dur - 1.5);
    g.gain.linearRampToValueAtTime(0, t0 + dur);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(500, t0);
    lp.frequency.linearRampToValueAtTime(2200, t0 + dur * 0.5);
    lp.Q.value = 1;
    [freq, freq * 1.5, freq * 2.005, freq * 3].forEach((f, i) => {
      const o = ctx.createOscillator();
      o.type = i === 0 ? 'sawtooth' : 'sine';
      o.frequency.value = f;
      o.detune.value = (i - 1) * 5;
      const og = ctx.createGain();
      og.gain.value = i === 0 ? 0.35 : 0.18;
      o.connect(og).connect(lp);
      o.start(t0); o.stop(t0 + dur + 0.1);
    });
    lp.connect(g).connect(this.master);
  }

  // Spray ASMR psssht
  spray(at) {
    const ctx = this.ctx;
    const t0 = ctx.currentTime + at;
    const dur = 0.65;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      data[i] = (Math.random() * 2 - 1);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const hp = ctx.createBiquadFilter();
    hp.type = 'highpass';
    hp.frequency.setValueAtTime(2500, t0);
    hp.frequency.linearRampToValueAtTime(1200, t0 + dur);
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 5000;
    bp.Q.value = 0.6;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.7, t0 + 0.015);
    g.gain.exponentialRampToValueAtTime(0.2, t0 + 0.18);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    src.connect(hp).connect(bp).connect(g).connect(this.master);
    src.start(t0); src.stop(t0 + dur + 0.05);
  }

  chime(at, freq = 1320) {
    const ctx = this.ctx;
    const t0 = ctx.currentTime + at;
    const dur = 1.8;
    const c1 = ctx.createOscillator();
    c1.type = 'sine'; c1.frequency.value = freq;
    const m = ctx.createOscillator();
    m.type = 'sine'; m.frequency.value = freq * 3.01;
    const mg = ctx.createGain();
    mg.gain.value = freq * 1.4;
    mg.gain.exponentialRampToValueAtTime(0.01, t0 + dur);
    m.connect(mg).connect(c1.frequency);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.18, t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    c1.connect(g).connect(this.master);
    c1.start(t0); m.start(t0);
    c1.stop(t0 + dur + 0.1); m.stop(t0 + dur + 0.1);
  }

  thump(at) {
    const ctx = this.ctx;
    const t0 = ctx.currentTime + at;
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.setValueAtTime(110, t0);
    o.frequency.exponentialRampToValueAtTime(35, t0 + 0.3);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.5, t0 + 0.005);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.4);
    o.connect(g).connect(this.master);
    o.start(t0); o.stop(t0 + 0.45);
  }

  whoosh(at, dur = 0.8) {
    const ctx = this.ctx;
    const t0 = ctx.currentTime + at;
    const buf = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1);
    const src = ctx.createBufferSource(); src.buffer = buf;
    const f = ctx.createBiquadFilter();
    f.type = 'bandpass'; f.Q.value = 1.5;
    f.frequency.setValueAtTime(400, t0);
    f.frequency.exponentialRampToValueAtTime(2400, t0 + dur);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(0.25, t0 + dur * 0.5);
    g.gain.linearRampToValueAtTime(0, t0 + dur);
    src.connect(f).connect(g).connect(this.master);
    src.start(t0); src.stop(t0 + dur + 0.05);
  }

  scheduleTimeline(events) {
    if (!this.ctx) return;
    this.startedAt = this.ctx.currentTime;
    events.forEach(({ at, type, freq, dur }) => {
      if (type === 'pad') this.pad(at, dur || 14, freq || 130);
      else if (type === 'spray') this.spray(at);
      else if (type === 'chime') this.chime(at, freq || 1320);
      else if (type === 'thump') this.thump(at);
      else if (type === 'whoosh') this.whoosh(at, dur || 0.8);
    });
  }
}
const MahatahSoundInstance = new MahatahSound();

// ── Volumetric mist using SVG turbulence ────────────────────────────────────
// Renders a soft warm cloud that animates by shifting baseFrequency seed
// and translating across the screen.
function VolumetricMist({ x, y, w, h, t0, dur, intensity = 1, hue = '#f4d27a' }) {
  const t = useTime();
  const elapsed = t - t0;
  if (elapsed < 0 || elapsed > dur) return null;

  // Mist lifecycle: explosive burst, then drift, then dissipate
  const burst = Math.min(1, elapsed / 0.35);
  const linger = elapsed > 0.35 ? Math.min(1, (elapsed - 0.35) / 1.5) : 0;
  const fade = elapsed > dur - 2.5 ? (elapsed - (dur - 2.5)) / 2.5 : 0;
  const opacity = burst * (1 - fade) * intensity;

  // Drift upward over time
  const driftY = -elapsed * 12;
  const driftX = Math.sin(elapsed * 0.4) * 8;
  const scale = 0.4 + burst * 1.4 + linger * 0.5;

  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      width: w, height: h,
      transform: `translate(${driftX}px, ${driftY}px) scale(${scale})`,
      transformOrigin: 'right center',
      opacity,
      mixBlendMode: 'screen',
      pointerEvents: 'none',
      filter: 'blur(2px)',
    }}>
      <svg viewBox="0 0 600 600" width="100%" height="100%" preserveAspectRatio="none">
        <defs>
          <filter id={`mist-${t0}`} x="-20%" y="-20%" width="140%" height="140%">
            <feTurbulence
              type="fractalNoise"
              baseFrequency={`${0.008 + Math.sin(elapsed) * 0.002} ${0.012}`}
              numOctaves="4"
              seed={Math.floor(elapsed * 8) % 100}
            />
            <feDisplacementMap in="SourceGraphic" scale="80"/>
            <feGaussianBlur stdDeviation="6"/>
          </filter>
          <radialGradient id={`mistGrad-${t0}`} cx="50%" cy="55%" r="50%">
            <stop offset="0%" stopColor={hue} stopOpacity="0.95"/>
            <stop offset="35%" stopColor={hue} stopOpacity="0.55"/>
            <stop offset="70%" stopColor={hue} stopOpacity="0.18"/>
            <stop offset="100%" stopColor={hue} stopOpacity="0"/>
          </radialGradient>
        </defs>
        <ellipse cx="300" cy="300" rx="260" ry="200" fill={`url(#mistGrad-${t0})`} filter={`url(#mist-${t0})`}/>
      </svg>
    </div>
  );
}

// ── Fine spray particles — many small dots simulating atomized droplets ────
function FineSpray({ originX, originY, t0, dur, count = 200, direction = 'left' }) {
  const t = useTime();
  const elapsed = t - t0;
  if (elapsed < 0 || elapsed > dur) return null;

  // Generate deterministic particles per t0
  const particles = React.useMemo(() => {
    const arr = [];
    const seed = Math.floor(t0 * 1000);
    let s = seed;
    const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    for (let i = 0; i < count; i++) {
      const baseAngle = direction === 'left' ? Math.PI : direction === 'up' ? -Math.PI / 2 : 0;
      const spread = (rand() - 0.5) * 1.0; // ±~30deg cone
      const angle = baseAngle + spread;
      const burstDelay = rand() * 0.25; // staggered launch within burst
      const speed = 250 + rand() * 700;
      arr.push({
        i, angle,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - rand() * 80, // bias upward
        size: 0.8 + rand() * 2.6,
        delay: burstDelay,
        life: 2.5 + rand() * 2.5,
        hue: rand() > 0.7 ? '#fff5d8' : (rand() > 0.4 ? '#f4d27a' : '#e8b94c'),
        wob: rand() * Math.PI * 2,
        wobAmp: 6 + rand() * 18,
        gravity: 30 + rand() * 40, // particles slow + slightly fall over time
      });
    }
    return arr;
  }, [t0, count, direction]);

  return (
    <div style={{
      position: 'absolute',
      left: originX, top: originY,
      width: 1, height: 1,
      pointerEvents: 'none',
      mixBlendMode: 'screen',
    }}>
      {particles.map(p => {
        const local = elapsed - p.delay;
        if (local < 0) return null;
        const lifeT = Math.min(1, local / p.life);
        // Damped motion: ease-out
        const damp = 1 - Math.exp(-local * 1.4);
        const x = p.vx * damp + Math.sin(local * 2 + p.wob) * p.wobAmp * lifeT;
        const y = p.vy * damp + p.gravity * Math.pow(local * 0.5, 2) + Math.cos(local * 1.5 + p.wob) * p.wobAmp * 0.5 * lifeT;
        const opacity = lifeT < 0.15 ? lifeT / 0.15 : Math.pow(1 - lifeT, 1.3);
        const scale = 0.4 + lifeT * 0.8;
        return (
          <div key={p.i} style={{
            position: 'absolute',
            left: 0, top: 0,
            width: p.size * 2, height: p.size * 2,
            marginLeft: -p.size, marginTop: -p.size,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${p.hue} 0%, ${p.hue}cc 30%, ${p.hue}00 70%)`,
            opacity: opacity * 0.9,
            transform: `translate(${x}px, ${y}px) scale(${scale})`,
          }} />
        );
      })}
    </div>
  );
}

// ── Ambient floating dust motes (always-on, low density) ───────────────────
function AmbientMotes({ count = 25, hue = '#f4d27a' }) {
  const t = useTime();
  const motes = React.useMemo(() => {
    const arr = [];
    let s = 1234;
    const r = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
    for (let i = 0; i < count; i++) {
      arr.push({
        i,
        x: r() * 1080,
        yBase: r() * 1920,
        size: 0.8 + r() * 2.2,
        speed: 12 + r() * 25,
        drift: 20 + r() * 50,
        period: 4 + r() * 6,
        delay: r() * 8,
        hue: r() > 0.7 ? '#fff5d8' : hue,
        opa: 0.3 + r() * 0.5,
      });
    }
    return arr;
  }, [count, hue]);

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', mixBlendMode: 'screen' }}>
      {motes.map(m => {
        const tt = t + m.delay;
        const y = ((m.yBase - tt * m.speed) % 2100 + 2100) % 2100 - 100;
        const x = m.x + Math.sin(tt / m.period * Math.PI * 2) * m.drift;
        return (
          <div key={m.i} style={{
            position: 'absolute',
            left: x, top: y,
            width: m.size * 2, height: m.size * 2,
            marginLeft: -m.size, marginTop: -m.size,
            borderRadius: '50%',
            background: `radial-gradient(circle, ${m.hue}, ${m.hue}00 70%)`,
            opacity: m.opa,
            filter: 'blur(0.3px)',
          }} />
        );
      })}
    </div>
  );
}

// ── Hero key art — the main spray photo, animated ──────────────────────────
// We use the photo as a base, then overlay extra mist + particles AT the
// sprayer location to amplify the effect during the animation.
function HeroSprayShot({ kenBurns = true, scale0 = 1.04, scale1 = 1.18, panX = -30, panY = -10 }) {
  const t = useTime();
  const { duration = 14 } = useTimeline();
  const progress = Math.min(1, t / duration);

  const sc = scale0 + (scale1 - scale0) * progress;
  const px = panX * progress;
  const py = panY * progress;

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
      <img
        src="assets/hero-spray.png"
        style={{
          position: 'absolute',
          left: '50%', top: '50%',
          width: '108%', height: '108%',
          objectFit: 'cover',
          transform: `translate(-50%, -50%) translate(${px}px, ${py}px) scale(${sc})`,
          transformOrigin: 'center',
          filter: 'contrast(1.06) saturate(1.08)',
        }}
      />
      {/* Soft golden light leak top-right */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 50% 40% at 75% 15%, rgba(255, 220, 130, 0.25), transparent 60%)',
        mixBlendMode: 'screen',
        pointerEvents: 'none',
      }} />
      {/* Bottom warm fade */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'linear-gradient(180deg, transparent 60%, rgba(60, 30, 8, 0.25) 100%)',
        pointerEvents: 'none',
      }} />
    </div>
  );
}

// ── Bottle clean shot (cutout on paper) ────────────────────────────────────
function CleanBottleShot({ tone = 'paper' }) {
  const t = useTime();
  const breathe = 1 + Math.sin(t * 0.5) * 0.008;
  const sway = Math.sin(t * 0.3) * 0.4;

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
      <PaperBackdrop tone={tone}/>
      <div style={{
        position: 'absolute',
        left: '50%', top: '50%',
        width: 760, height: 760,
        transform: `translate(-50%, -50%) scale(${breathe}) rotate(${sway}deg)`,
      }}>
        {/* Soft glow behind bottle */}
        <div style={{
          position: 'absolute', inset: -60,
          background: 'radial-gradient(ellipse 50% 60% at 50% 55%, rgba(255, 200, 110, 0.45), transparent 65%)',
          filter: 'blur(30px)',
        }}/>
        <img
          src="assets/bottle-cutout.png"
          style={{
            position: 'absolute', inset: 0,
            width: '100%', height: '100%',
            objectFit: 'contain',
            filter: 'drop-shadow(0 30px 40px rgba(60,30,5,0.35)) contrast(1.04) saturate(1.1)',
          }}
        />
      </div>
    </div>
  );
}

// ── Backdrop: warm crinkled paper ──────────────────────────────────────────
function PaperBackdrop({ tone = 'paper', animateLight = true }) {
  const t = useTime();
  const lightX = animateLight ? 30 + Math.sin(t * 0.25) * 12 : 35;

  const palette = tone === 'dark'
    ? { base: '#1a1108', light: '#5a3d1a', accent: '#d4a544' }
    : tone === 'silk'
      ? { base: '#e8dcc6', light: '#fff5e0', accent: '#c9a05c' }
      : { base: '#dfd0b2', light: '#f9eed5', accent: '#c9a05c' };

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
      <div style={{ position: 'absolute', inset: 0, background: palette.base }}/>
      {/* Crinkles via SVG turbulence */}
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.3, mixBlendMode: 'multiply' }} viewBox="0 0 1080 1920" preserveAspectRatio="none">
        <defs>
          <filter id="crinklePB">
            <feTurbulence type="fractalNoise" baseFrequency="0.008 0.015" numOctaves="3" seed="3"/>
            <feColorMatrix values="0 0 0 0 0.15  0 0 0 0 0.08  0 0 0 0 0.02  0 0 0 0.6 0"/>
          </filter>
        </defs>
        <rect width="1080" height="1920" filter="url(#crinklePB)"/>
      </svg>
      {/* Crease lines */}
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.18, mixBlendMode: 'multiply' }} viewBox="0 0 1080 1920" preserveAspectRatio="none">
        <defs>
          <filter id="creasePB">
            <feTurbulence type="fractalNoise" baseFrequency="0.012" numOctaves="2" seed="7"/>
            <feDisplacementMap in="SourceGraphic" scale="50"/>
          </filter>
        </defs>
        <g filter="url(#creasePB)" stroke="#000" strokeWidth="1" fill="none">
          {Array.from({length: 16}).map((_, i) => (
            <line key={i} x1={-100} y1={i * 130} x2={1200} y2={i * 130 + 200}/>
          ))}
        </g>
      </svg>
      {/* Raking light */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse 70% 100% at ${lightX}% 30%, ${palette.light}, transparent 60%)`,
        mixBlendMode: 'screen',
        opacity: 0.85,
      }}/>
      {/* Vignette */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'radial-gradient(ellipse 90% 70% at 50% 50%, transparent 40%, rgba(20,10,2,0.4) 100%)',
      }}/>
    </div>
  );
}

// ── Ingredient note overlay (fades over hero / clean shot) ─────────────────
function NoteOverlay({ note, family }) {
  const { progress, localTime, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(1, localTime / 0.45));
  const exitT = Math.max(0, (localTime - (duration - 0.45)) / 0.45);
  const opacity = enter * (1 - exitT);
  const scale = 0.95 + 0.05 * enter;

  return (
    <div style={{
      position: 'absolute',
      left: 60, right: 60, top: 1180,
      opacity,
      transform: `translateY(${(1 - enter) * 20}px) scale(${scale})`,
      textAlign: 'center',
    }}>
      <div style={{
        fontFamily: '"Inter", sans-serif',
        fontSize: 22,
        fontWeight: 500,
        color: '#fff5d8',
        letterSpacing: '0.5em',
        textTransform: 'uppercase',
        marginBottom: 24,
        textIndent: '0.5em',
        textShadow: '0 2px 12px rgba(0,0,0,0.7)',
      }}>
        {family}
      </div>
      <div style={{
        fontFamily: '"Cormorant Garamond", "Playfair Display", serif',
        fontSize: 130,
        fontWeight: 300,
        fontStyle: 'italic',
        color: '#fff8ea',
        letterSpacing: '-0.01em',
        lineHeight: 0.95,
        textShadow: '0 6px 30px rgba(40,20,2,0.8)',
      }}>
        {note}
      </div>
    </div>
  );
}

// ── Hook overlay (top text) ────────────────────────────────────────────────
function HookOverlay({ text, sub, position = 'top', dark = false }) {
  const { progress, localTime, duration } = useSprite();
  const enter = Easing.easeOutCubic(Math.min(1, localTime / 0.5));
  const exitT = Math.max(0, (localTime - (duration - 0.4)) / 0.4);
  const opacity = enter * (1 - exitT);
  const ty = (1 - enter) * 28;

  const color = dark ? '#1a0e06' : '#fff8ea';
  const shadow = dark
    ? '0 2px 14px rgba(255,240,200,0.6), 0 0 40px rgba(255,255,255,0.4)'
    : '0 4px 24px rgba(40,20,2,0.85), 0 0 50px rgba(0,0,0,0.4)';

  const top = position === 'top' ? 240 : position === 'middle' ? 750 : 1380;

  return (
    <div style={{
      position: 'absolute',
      left: 50, right: 50, top,
      textAlign: 'center',
      opacity,
      transform: `translateY(${ty}px)`,
    }}>
      {sub && (
        <div style={{
          fontFamily: '"Inter", sans-serif',
          fontSize: 22,
          fontWeight: 500,
          color,
          letterSpacing: '0.5em',
          textTransform: 'uppercase',
          marginBottom: 24,
          textIndent: '0.5em',
          opacity: 0.9,
          textShadow: shadow,
        }}>
          {sub}
        </div>
      )}
      <div style={{
        fontFamily: '"Cormorant Garamond", "Playfair Display", serif',
        fontSize: 110,
        fontWeight: 300,
        fontStyle: 'italic',
        color,
        letterSpacing: '-0.015em',
        lineHeight: 1,
        textShadow: shadow,
      }}>
        {text}
      </div>
    </div>
  );
}

// ── Brand Mark (top corner) ────────────────────────────────────────────────
function BrandMark({ visible = true, dark = false }) {
  const t = useTime();
  return (
    <div style={{
      position: 'absolute',
      left: 0, right: 0, top: 90,
      textAlign: 'center',
      opacity: visible ? 0.95 : 0,
      transition: 'opacity 0.5s',
      pointerEvents: 'none',
    }}>
      <div style={{
        fontFamily: '"Inter", sans-serif',
        fontSize: 14,
        fontWeight: 600,
        letterSpacing: '0.6em',
        color: dark ? '#1a0e06' : '#fff8ea',
        textTransform: 'uppercase',
        textIndent: '0.6em',
        textShadow: dark ? 'none' : '0 2px 10px rgba(0,0,0,0.6)',
        opacity: 0.8,
        marginBottom: 8,
      }}>
        Maison
      </div>
      <div style={{
        fontFamily: '"Cormorant Garamond", serif',
        fontSize: 38,
        fontWeight: 400,
        letterSpacing: '0.42em',
        color: dark ? '#1a0e06' : '#fff8ea',
        textTransform: 'uppercase',
        textIndent: '0.42em',
        textShadow: dark ? 'none' : '0 2px 14px rgba(0,0,0,0.6)',
      }}>
        MATAHA
      </div>
    </div>
  );
}

// ── CTA Sticker — animated end card ────────────────────────────────────────
function CTASticker({ url = 'planetebeauty.com', subline = 'Maison Mataha · Escapade Gourmande' }) {
  const { localTime, progress } = useSprite();
  const stampT = Easing.easeOutBack(Math.min(1, localTime / 0.55));
  const wobble = Math.sin(localTime * 9) * Math.max(0, 1 - localTime / 0.9) * 1.4;
  const breathe = 1 + Math.sin(localTime * 1.5) * 0.012;

  return (
    <div style={{
      position: 'absolute',
      left: '50%', top: '50%',
      transform: `translate(-50%, -50%) scale(${(0.55 + stampT * 0.45) * breathe}) rotate(${-2 + wobble}deg)`,
      opacity: Math.min(1, localTime / 0.25),
      transformOrigin: 'center',
    }}>
      <div style={{
        background: 'linear-gradient(135deg, #fff8ea, #f5ede0 60%, #ead9b8)',
        border: '5px solid #1a0e06',
        borderRadius: 22,
        padding: '64px 72px 56px',
        boxShadow: '10px 10px 0 #1a0e06, 0 30px 80px rgba(0,0,0,0.45)',
        textAlign: 'center',
        minWidth: 820,
      }}>
        <div style={{
          fontFamily: '"Inter", sans-serif',
          fontSize: 22,
          fontWeight: 700,
          color: '#3a2412',
          letterSpacing: '0.5em',
          textTransform: 'uppercase',
          textIndent: '0.5em',
          marginBottom: 30,
        }}>
          ✦ Découvrir ✦
        </div>
        <div style={{
          fontFamily: '"Cormorant Garamond", serif',
          fontSize: 72,
          fontWeight: 400,
          fontStyle: 'italic',
          color: '#1a0e06',
          lineHeight: 1,
        }}>
          Maison Mataha
        </div>
        <div style={{
          fontFamily: '"Inter", sans-serif',
          fontSize: 18,
          fontWeight: 500,
          color: '#6b4520',
          letterSpacing: '0.32em',
          textIndent: '0.32em',
          textTransform: 'uppercase',
          marginTop: 14,
        }}>
          Escapade Gourmande
        </div>
        <div style={{
          fontFamily: '"Inter", monospace',
          fontSize: 32,
          fontWeight: 600,
          color: '#1a0e06',
          letterSpacing: '0.04em',
          marginTop: 30,
          paddingTop: 28,
          borderTop: '2px dashed rgba(26,14,6,0.35)',
        }}>
          {url}
        </div>
      </div>
      <Sparkle x={-40} y={-40} delay={0.5} size={56}/>
      <Sparkle x={830} y={-30} delay={0.7} size={44}/>
      <Sparkle x={780} y={350} delay={0.85} size={50}/>
      <Sparkle x={-30} y={330} delay={1.0} size={40}/>
    </div>
  );
}

function Sparkle({ x, y, delay, size = 48 }) {
  const { localTime } = useSprite();
  const t = Math.max(0, localTime - delay);
  const sc = t < 0.4 ? Easing.easeOutBack(t / 0.4) : 1;
  const op = t < 0.4 ? t / 0.4 : 0.6 + Math.sin(t * 4) * 0.4;
  const rot = t * 60;
  return (
    <div style={{
      position: 'absolute',
      left: x, top: y,
      transform: `scale(${sc}) rotate(${rot}deg)`,
      opacity: op,
      width: size, height: size,
      marginLeft: -size / 2, marginTop: -size / 2,
    }}>
      <svg width={size} height={size} viewBox="0 0 48 48">
        <path d="M24 4 L26 22 L44 24 L26 26 L24 44 L22 26 L4 24 L22 22 Z" fill="#c9a05c"/>
      </svg>
    </div>
  );
}

// ── Sticky URL strap ───────────────────────────────────────────────────────
function URLStrap({ url = 'www.planetebeauty.com', visible = true, dark = false }) {
  return (
    <div style={{
      position: 'absolute',
      left: 0, right: 0, bottom: 80,
      textAlign: 'center',
      opacity: visible ? 1 : 0,
      transition: 'opacity 0.4s',
      pointerEvents: 'none',
    }}>
      <div style={{
        display: 'inline-block',
        background: dark ? 'rgba(255, 248, 234, 0.92)' : 'rgba(26, 14, 6, 0.85)',
        backdropFilter: 'blur(10px)',
        color: dark ? '#1a0e06' : '#fff8ea',
        padding: '18px 38px',
        borderRadius: 100,
        fontFamily: '"Inter", sans-serif',
        fontSize: 28,
        fontWeight: 600,
        letterSpacing: '0.06em',
      }}>
        ✦ {url}
      </div>
    </div>
  );
}

// ── Sound toggle button ────────────────────────────────────────────────────
function SoundToggle({ muted, onToggle }) {
  return (
    <button
      onClick={onToggle}
      style={{
        position: 'absolute',
        top: 40, right: 40,
        width: 84, height: 84,
        borderRadius: '50%',
        background: 'rgba(26, 14, 6, 0.55)',
        backdropFilter: 'blur(10px)',
        border: '2px solid rgba(255, 248, 234, 0.3)',
        color: '#fff8ea',
        cursor: 'pointer',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 100,
        padding: 0,
      }}
      title={muted ? 'Activer le son' : 'Couper le son'}
    >
      {muted ? (
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="currentColor"/>
          <line x1="22" y1="9" x2="16" y2="15"/>
          <line x1="16" y1="9" x2="22" y2="15"/>
        </svg>
      ) : (
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" fill="currentColor"/>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        </svg>
      )}
    </button>
  );
}

// ── Cinematic letterbox bars (optional dramatic effect) ────────────────────
function Letterbox({ size = 120, opacity = 1 }) {
  return (
    <>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 0, height: size, background: '#000', opacity, pointerEvents: 'none' }}/>
      <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, height: size, background: '#000', opacity, pointerEvents: 'none' }}/>
    </>
  );
}

// ── Light flare sweep — dramatic cinematic cue ─────────────────────────────
function LightFlare({ t0, dur = 0.6 }) {
  const t = useTime();
  const elapsed = t - t0;
  if (elapsed < 0 || elapsed > dur) return null;
  const p = elapsed / dur;
  const x = -200 + p * 1500;
  const opacity = Math.sin(p * Math.PI);
  return (
    <div style={{
      position: 'absolute', inset: 0,
      pointerEvents: 'none',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute',
        left: x, top: -200,
        width: 600, height: 2400,
        background: 'linear-gradient(105deg, transparent 35%, rgba(255, 240, 180, 0.55) 50%, transparent 65%)',
        transform: 'rotate(15deg)',
        filter: 'blur(40px)',
        opacity: opacity * 0.6,
        mixBlendMode: 'screen',
      }}/>
    </div>
  );
}

// ── Caption chip (small bottom badge for ASMR labels) ──────────────────────
function CaptionChip({ text, icon }) {
  const { localTime, duration } = useSprite();
  const enter = Easing.easeOutBack(Math.min(1, localTime / 0.4));
  const exitT = Math.max(0, (localTime - (duration - 0.3)) / 0.3);
  const opacity = enter * (1 - exitT);

  return (
    <div style={{
      position: 'absolute',
      left: 0, right: 0, top: 1500,
      textAlign: 'center',
      opacity,
      transform: `scale(${0.7 + enter * 0.3})`,
    }}>
      <div style={{
        display: 'inline-flex', alignItems: 'center', gap: 12,
        background: 'rgba(255, 248, 234, 0.95)',
        color: '#1a0e06',
        padding: '14px 28px',
        borderRadius: 100,
        fontFamily: '"Inter", sans-serif',
        fontSize: 26,
        fontWeight: 600,
        letterSpacing: '0.04em',
        boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
      }}>
        {icon && <span style={{ fontSize: 28 }}>{icon}</span>}
        <span>{text}</span>
      </div>
    </div>
  );
}

Object.assign(window, {
  MahatahSoundInstance,
  VolumetricMist, FineSpray, AmbientMotes,
  HeroSprayShot, CleanBottleShot, PaperBackdrop,
  NoteOverlay, HookOverlay, BrandMark, CTASticker, Sparkle,
  URLStrap, SoundToggle, Letterbox, LightFlare, CaptionChip,
});
