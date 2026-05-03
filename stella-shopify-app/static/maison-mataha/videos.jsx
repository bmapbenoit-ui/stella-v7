// videos.jsx — three TikTok video variations for Maison Mataha

// ═══════════════════════════════════════════════════════════════════════════
// VARIATION A — "ASMR Hero" — uses the photo-real spray hero, animated
// 14 seconds. Static-like keyframe with deep zoom + extra mist + ingredient
// note overlays + CTA stamp. Best for "stop the scroll" pure aesthetic.
// ═══════════════════════════════════════════════════════════════════════════
function VideoA({ muted, onToggleMute }) {
  const [t0, setT0] = React.useState(0);

  const events = React.useMemo(() => [
    { at: 0,    type: 'pad', dur: 14 },
    { at: 0.2,  type: 'thump' },
    { at: 0.6,  type: 'spray' },
    { at: 1.4,  type: 'chime', freq: 1320 },
    { at: 4.0,  type: 'chime', freq: 1760 },
    { at: 6.5,  type: 'chime', freq: 1568 },
    { at: 9.0,  type: 'chime', freq: 1976 },
    { at: 11.5, type: 'whoosh', dur: 0.7 },
    { at: 11.8, type: 'chime', freq: 2349 },
    { at: 12.2, type: 'chime', freq: 1760 },
  ], []);

  return (
    <Stage width={1080} height={1920} duration={14} background="#1a1108" persistKey="varA"
      events={events} muted={muted}>
      {/* Hero spray photo with slow zoom */}
      <HeroSprayShot scale0={1.05} scale1={1.22} panX={-30} panY={-15}/>

      {/* Extra mist amplifier at sprayer location (right side, mid-upper) */}
      <VolumetricMist x={120} y={180} w={900} h={700} t0={0.55} dur={9} intensity={0.9}/>
      <VolumetricMist x={20}  y={300} w={780} h={620} t0={0.85} dur={8} intensity={0.65} hue="#e8b94c"/>

      {/* Fine droplets blasting from sprayer */}
      <FineSpray originX={780} originY={520} t0={0.6} dur={10} count={150} direction="left"/>

      {/* Ambient dust always */}
      <AmbientMotes count={22}/>

      {/* Light flare on spray hit */}
      <LightFlare t0={0.5} dur={0.7}/>

      {/* Hook + ingredient notes choreography */}
      <Sprite start={0.2} end={2.4}>
        <HookOverlay text="Prêt·e à vous" sub="Maison Mataha" position="bottom"/>
      </Sprite>
      <Sprite start={2.5} end={4.8}>
        <HookOverlay text="échapper ?" sub="Escapade Gourmande" position="bottom"/>
      </Sprite>

      {/* Ingredient call-outs — quick rhythmic */}
      <Sprite start={5.0} end={6.7}>
        <NoteOverlay note="vanille" family="NOTE DE CŒUR"/>
      </Sprite>
      <Sprite start={6.8} end={8.3}>
        <NoteOverlay note="fève tonka" family="NOTE DE CŒUR"/>
      </Sprite>
      <Sprite start={8.4} end={9.9}>
        <NoteOverlay note="canne à sucre" family="NOTE DE TÊTE"/>
      </Sprite>
      <Sprite start={10.0} end={11.4}>
        <NoteOverlay note="benjoin" family="NOTE DE FOND"/>
      </Sprite>

      {/* Caption chip — ASMR feel */}
      <Sprite start={1.5} end={4.5}>
        <CaptionChip text="son ASMR" icon="🎧"/>
      </Sprite>

      {/* Final CTA */}
      <Sprite start={11.5} end={14}>
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(20, 12, 4, 0.55)', backdropFilter: 'blur(6px)' }}/>
        <CTASticker url="planetebeauty.com"/>
      </Sprite>

      <SoundToggle muted={muted} onToggle={onToggleMute}/>
    </Stage>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// VARIATION B — "Reveal" — starts on clean bottle on paper, big spray reveals
// the ingredients, then settles. More restrained / luxurious.
// ═══════════════════════════════════════════════════════════════════════════
function VideoB({ muted, onToggleMute }) {
  const events = React.useMemo(() => [
    { at: 0,    type: 'pad', dur: 14, freq: 110 },
    { at: 1.5,  type: 'chime', freq: 1568 },
    { at: 3.4,  type: 'thump' },
    { at: 3.5,  type: 'spray' },
    { at: 4.3,  type: 'chime', freq: 1976 },
    { at: 6.0,  type: 'chime', freq: 1320 },
    { at: 8.0,  type: 'chime', freq: 1760 },
    { at: 11.5, type: 'whoosh', dur: 0.6 },
    { at: 11.9, type: 'chime', freq: 2349 },
  ], []);

  return (
    <Stage width={1080} height={1920} duration={14} background="#1a1108" persistKey="varB"
      events={events} muted={muted}>

      {/* Phase 1 (0-3.5s): Clean bottle on paper, slow zoom */}
      <Sprite start={0} end={3.6} keepMounted>
        <CleanBottleShot tone="paper"/>
        <AmbientMotes count={20}/>
      </Sprite>

      <Sprite start={0.4} end={3.2}>
        <HookOverlay text="Une escapade" sub="✦ Maison Mataha ✦" position="top" dark/>
      </Sprite>

      {/* Phase 2 (3.5-12s): Hero spray photo emerges */}
      <Sprite start={3.4} end={14}>
        <HeroSprayShot scale0={1.0} scale1={1.18} panX={-20} panY={-8}/>
      </Sprite>
      <Sprite start={3.5} end={12}>
        <VolumetricMist x={120} y={180} w={900} h={700} t0={3.5} dur={8} intensity={1}/>
        <VolumetricMist x={20}  y={300} w={780} h={620} t0={3.8} dur={7} intensity={0.7} hue="#e8b94c"/>
        <FineSpray originX={780} originY={520} t0={3.5} dur={8} count={180} direction="left"/>
        <LightFlare t0={3.45} dur={0.75}/>
        <AmbientMotes count={15}/>
      </Sprite>

      {/* Hook fade to title */}
      <Sprite start={4.5} end={6.5}>
        <HookOverlay text="gourmande" sub="EXTRAIT DE PARFUM" position="bottom"/>
      </Sprite>

      {/* Notes pyramid */}
      <Sprite start={6.7} end={8.4}>
        <NoteOverlay note="canne à sucre" family="NOTE DE TÊTE"/>
      </Sprite>
      <Sprite start={8.5} end={10.0}>
        <NoteOverlay note="vanille · fève tonka" family="NOTES DE CŒUR"/>
      </Sprite>
      <Sprite start={10.1} end={11.5}>
        <NoteOverlay note="benjoin · musc" family="NOTES DE FOND"/>
      </Sprite>

      {/* CTA */}
      <Sprite start={11.5} end={14}>
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(20, 12, 4, 0.5)', backdropFilter: 'blur(5px)' }}/>
        <CTASticker url="planetebeauty.com"/>
      </Sprite>

      <SoundToggle muted={muted} onToggle={onToggleMute}/>
    </Stage>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// VARIATION C — "POV Atelier" — dark cinematic, repeated spray pulses
// like an editorial perfume ad. Uses the hero shot with extra fades and
// dramatic lighting beats. Best for nighttime / luxe feed.
// ═══════════════════════════════════════════════════════════════════════════
function VideoC({ muted, onToggleMute }) {
  const events = React.useMemo(() => [
    { at: 0,    type: 'pad', dur: 14, freq: 98 },
    { at: 0.3,  type: 'thump' },
    { at: 1.0,  type: 'spray' },
    { at: 1.6,  type: 'chime', freq: 1480 },
    { at: 4.5,  type: 'spray' },
    { at: 5.1,  type: 'chime', freq: 1760 },
    { at: 8.0,  type: 'spray' },
    { at: 8.6,  type: 'chime', freq: 1976 },
    { at: 11.5, type: 'whoosh', dur: 0.7 },
    { at: 12.0, type: 'chime', freq: 2349 },
  ], []);

  return (
    <Stage width={1080} height={1920} duration={14} background="#0a0604" persistKey="varC"
      events={events} muted={muted}>

      {/* Dark hero shot — heavily darkened version of the spray photo */}
      <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
        <HeroSprayShot scale0={1.08} scale1={1.25} panX={-40} panY={-20}/>
        {/* Dark overlay for cinematic feel */}
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(180deg, rgba(10,5,2,0.5) 0%, rgba(10,5,2,0.2) 40%, rgba(10,5,2,0.7) 100%)',
          mixBlendMode: 'multiply',
          pointerEvents: 'none',
        }}/>
      </div>

      {/* Three spray pulses */}
      <VolumetricMist x={120} y={180} w={900} h={700} t0={1.0}  dur={4.5} intensity={1}/>
      <FineSpray  originX={780} originY={520} t0={1.0} dur={5}  count={130} direction="left"/>
      <LightFlare t0={0.95} dur={0.7}/>

      <VolumetricMist x={120} y={180} w={900} h={700} t0={4.5}  dur={4.5} intensity={1}/>
      <FineSpray  originX={780} originY={520} t0={4.5} dur={5}  count={130} direction="left"/>
      <LightFlare t0={4.45} dur={0.7}/>

      <VolumetricMist x={120} y={180} w={900} h={700} t0={8.0}  dur={4} intensity={1}/>
      <FineSpray  originX={780} originY={520} t0={8.0} dur={4.5} count={150} direction="left"/>
      <LightFlare t0={7.95} dur={0.7}/>

      <AmbientMotes count={20}/>

      {/* Editorial single-word beats */}
      <Sprite start={0.3} end={1.6}>
        <HookOverlay text="osez" sub="✦" position="top"/>
      </Sprite>
      <Sprite start={1.8} end={4.2}>
        <HookOverlay text="vanille" sub="NOTE DE CŒUR" position="middle"/>
      </Sprite>
      <Sprite start={4.6} end={7.7}>
        <HookOverlay text="fève tonka" sub="NOTE DE CŒUR" position="middle"/>
      </Sprite>
      <Sprite start={8.2} end={11.3}>
        <HookOverlay text="benjoin" sub="NOTE DE FOND" position="middle"/>
      </Sprite>

      <BrandMark visible={true}/>

      {/* CTA */}
      <Sprite start={11.5} end={14}>
        <div style={{ position: 'absolute', inset: 0, background: 'rgba(10, 5, 2, 0.65)', backdropFilter: 'blur(8px)' }}/>
        <CTASticker url="planetebeauty.com"/>
      </Sprite>

      <SoundToggle muted={muted} onToggle={onToggleMute}/>
    </Stage>
  );
}

Object.assign(window, { VideoA, VideoB, VideoC });
