// Fetches reference images from Pexels for Veo 3 character/scene conditioning.
// Run once at project setup; outputs land in assets/stock/.
// Falls back to direct curl from public CDN if PEXELS_API_KEY is missing.
//
// Usage:
//   npm run stock:fetch

import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { env } from '../src/lib/env.js';

type StockTarget = {
  /** Output filename under assets/stock/ */
  out: string;
  /** Pexels search query */
  query: string;
  /** Index in the search result list to pick (0-based) */
  pick?: number;
  /** Pexels orientation filter */
  orientation?: 'landscape' | 'portrait' | 'square';
};

const TARGETS: StockTarget[] = [
  { out: 'parisienne-portrait.jpg', query: 'parisian woman 30 brunette portrait elegant', orientation: 'portrait' },
  { out: 'metro-paris.jpg',         query: 'paris metro station interior',                orientation: 'portrait' },
  { out: 'paris-rooftop-sunset.jpg', query: 'paris rooftops golden hour sunset eiffel',  orientation: 'portrait' },
];

async function fetchPexels(query: string, orientation: string, pick: number): Promise<string> {
  if (!env.PEXELS_API_KEY) {
    throw new Error('PEXELS_API_KEY not set — sign up at https://www.pexels.com/api/ (free) then put it in .env');
  }
  const url = new URL('https://api.pexels.com/v1/search');
  url.searchParams.set('query', query);
  url.searchParams.set('orientation', orientation);
  url.searchParams.set('per_page', '15');
  url.searchParams.set('size', 'large');
  const res = await fetch(url, { headers: { Authorization: env.PEXELS_API_KEY } });
  if (!res.ok) throw new Error(`Pexels API ${res.status}: ${await res.text()}`);
  const json = await res.json() as { photos?: Array<{ src: { large2x: string; original: string } }> };
  const photo = json.photos?.[pick];
  if (!photo) throw new Error(`Pexels returned no photo for "${query}"`);
  return photo.src.large2x ?? photo.src.original;
}

async function downloadTo(url: string, outPath: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download ${res.status}: ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(outPath, buf);
}

async function main() {
  await mkdir('assets/stock', { recursive: true });
  for (const t of TARGETS) {
    const out = path.resolve('assets/stock', t.out);
    console.log(`  → ${t.out} — query "${t.query}"`);
    const url = await fetchPexels(t.query, t.orientation ?? 'portrait', t.pick ?? 0);
    await downloadTo(url, out);
    console.log(`  ✔ ${t.out}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
