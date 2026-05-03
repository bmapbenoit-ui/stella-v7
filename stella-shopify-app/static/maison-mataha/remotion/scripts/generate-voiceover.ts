// Renders all ElevenLabs voice-over MP3s for Scenario C.
//
// Usage:
//   npm run voice:generate
//
// Output: assets/voiceover/<line-id>.mp3

import { renderLine } from '../src/lib/elevenlabs.js';
import { voiceLines } from '../src/compositions/varC.config.js';
import { existsSync } from 'node:fs';
import path from 'node:path';

async function main() {
  console.log(`ElevenLabs — rendering ${voiceLines.length} voice lines (Charlotte FR)\n`);

  for (const line of voiceLines) {
    const target = path.resolve(`assets/voiceover/${line.id}.mp3`);
    if (existsSync(target)) {
      console.log(`  ⊙ ${line.id} — already rendered, skipping`);
      continue;
    }
    const t0 = Date.now();
    console.log(`  → ${line.id} — "${line.text.replace(/<[^>]+>/g, '').slice(0, 60)}…"`);
    const out = await renderLine(line);
    console.log(`  ✔ ${line.id} — ${((Date.now() - t0) / 1000).toFixed(1)}s → ${path.relative(process.cwd(), out)}`);
  }
  console.log('\nDone. All voice lines in assets/voiceover/');
}

main().catch((e) => { console.error(e); process.exit(1); });
