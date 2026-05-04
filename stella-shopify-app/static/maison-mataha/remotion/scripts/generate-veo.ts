// Generates all Veo 3 clips for Scenario C in parallel batches of 3
// (to stay under Vertex AI per-region concurrency limits).
//
// Usage:
//   npm run veo:generate
//
// Output: assets/veo/<shot-id>.mp4

import { generateShot } from '../src/lib/veo.js';
import { shots } from '../src/compositions/varC.config.js';
import { existsSync } from 'node:fs';
import path from 'node:path';

const BATCH_SIZE = 3;

async function main() {
  console.log(`Veo 3.1 — generating ${shots.length} shots in batches of ${BATCH_SIZE}\n`);

  for (let i = 0; i < shots.length; i += BATCH_SIZE) {
    const batch = shots.slice(i, i + BATCH_SIZE);
    await Promise.all(
      batch.map(async (shot) => {
        const target = path.resolve(`assets/veo/${shot.id}.mp4`);
        if (existsSync(target)) {
          console.log(`  ⊙ ${shot.id} — already rendered, skipping (delete to regenerate)`);
          return;
        }
        const t0 = Date.now();
        console.log(`  → ${shot.id} — generating ${shot.durationSeconds ?? 8}s…`);
        const out = await generateShot(shot);
        console.log(`  ✔ ${shot.id} — ${((Date.now() - t0) / 1000).toFixed(0)}s → ${path.relative(process.cwd(), out)}`);
      })
    );
  }
  console.log('\nDone. All shots in assets/veo/');
}

main().catch((e) => { console.error(e); process.exit(1); });
