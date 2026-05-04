#!/usr/bin/env node
// Render the 3 Maison Mataha TikTok variations to MP4.
//
// Pipeline:
//   1. Headless Chromium opens render.html?v=<A|B|C> at exactly 1080x1920
//   2. Step 0..14s at 30 fps (420 frames), set window.__renderTime + force a
//      React re-render, screenshot each frame as PNG
//   3. Open render-audio.html?v=<A|B|C> which reuses the prototype's
//      MahatahSound synth in an OfflineAudioContext to render a 14s mono
//      WAV that exactly matches the timeline
//   4. ffmpeg encodes PNG sequence + WAV into a single yuv420p H.264 MP4
//
// Usage:
//   PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
//   node stella-shopify-app/static/maison-mataha/render-videos.mjs
//
// Outputs land in stella-shopify-app/static/maison-mataha/renders/
//   tiktok-A.mp4, tiktok-B.mp4, tiktok-C.mp4

import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync, rmSync, existsSync, readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const W = 1080, H = 1920, FPS = 30, DURATION = 14;
const FRAMES = FPS * DURATION;
const SAMPLE_RATE = 48000;

const VARIANTS = [
  { id: 'A', label: 'ASMR Hero' },
  { id: 'B', label: 'Reveal' },
  { id: 'C', label: 'Editorial dark' },
];

// Auto-vendor React + Babel into ./vendor so render.html / render-audio.html
// don't need network access at render time. Idempotent.
async function ensureVendor() {
  const vendorDir = path.join(__dirname, 'vendor');
  const files = ['react.development.js', 'react-dom.development.js', 'babel.min.js'];
  if (files.every(f => existsSync(path.join(vendorDir, f)))) return;
  mkdirSync(vendorDir, { recursive: true });
  console.log('Vendoring React + Babel into ./vendor (one-time, ~4 MB)…');
  await new Promise((resolve, reject) => {
    const p = spawn('npm', [
      'install', '--prefix', '/tmp/maison-mataha-vendor', '--no-audit', '--no-fund', '--silent',
      'react@18.3.1', 'react-dom@18.3.1', '@babel/standalone@7.29.0',
    ], { stdio: 'inherit' });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error(`npm exit ${code}`)));
  });
  const src = '/tmp/maison-mataha-vendor/node_modules';
  writeFileSync(path.join(vendorDir, 'react.development.js'),     readFileSync(path.join(src, 'react/umd/react.development.js')));
  writeFileSync(path.join(vendorDir, 'react-dom.development.js'), readFileSync(path.join(src, 'react-dom/umd/react-dom.development.js')));
  writeFileSync(path.join(vendorDir, 'babel.min.js'),             readFileSync(path.join(src, '@babel/standalone/babel.min.js')));
}

// Tiny static file server so the page is served from http:// (avoids CORS
// errors that affect crossorigin <script> tags loaded from file://).
function startStaticServer(rootDir) {
  const mime = {
    '.html': 'text/html', '.jsx': 'application/javascript',
    '.js':   'application/javascript', '.css': 'text/css',
    '.png':  'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.webp': 'image/webp', '.svg': 'image/svg+xml',
  };
  const server = createServer((req, res) => {
    const u = new URL(req.url, 'http://localhost');
    let p = path.normalize(path.join(rootDir, decodeURIComponent(u.pathname)));
    if (!p.startsWith(rootDir)) { res.writeHead(403).end('forbidden'); return; }
    try {
      if (statSync(p).isDirectory()) p = path.join(p, 'index.html');
    } catch { res.writeHead(404).end('not found'); return; }
    try {
      const buf = readFileSync(p);
      const ext = path.extname(p).toLowerCase();
      res.writeHead(200, { 'content-type': mime[ext] || 'application/octet-stream', 'cache-control': 'no-store' });
      res.end(buf);
    } catch { res.writeHead(404).end('not found'); }
  });
  return new Promise(resolve => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, base: `http://127.0.0.1:${port}` });
    });
  });
}

function encodeWav(float32, sampleRate) {
  // 16-bit PCM mono
  const numSamples = float32.length;
  const blockAlign = 2;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  let off = 0;
  buf.write('RIFF', off); off += 4;
  buf.writeUInt32LE(36 + dataSize, off); off += 4;
  buf.write('WAVE', off); off += 4;
  buf.write('fmt ', off); off += 4;
  buf.writeUInt32LE(16, off); off += 4;
  buf.writeUInt16LE(1, off); off += 2;
  buf.writeUInt16LE(1, off); off += 2;
  buf.writeUInt32LE(sampleRate, off); off += 4;
  buf.writeUInt32LE(byteRate, off); off += 4;
  buf.writeUInt16LE(blockAlign, off); off += 2;
  buf.writeUInt16LE(16, off); off += 2;
  buf.write('data', off); off += 4;
  buf.writeUInt32LE(dataSize, off); off += 4;
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    buf.writeInt16LE(s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff), off); off += 2;
  }
  return buf;
}

function runFfmpeg(args) {
  return new Promise((resolve, reject) => {
    const p = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let err = '';
    p.stderr.on('data', d => { err += d.toString(); });
    p.on('exit', code => {
      if (code === 0) resolve();
      else reject(new Error(`ffmpeg exit ${code}\n${err.split('\n').slice(-20).join('\n')}`));
    });
  });
}

async function captureFrames(browser, base, variant, framesDir) {
  const ctx = await browser.newContext({
    viewport: { width: W, height: H },
    deviceScaleFactor: 1,
    ignoreHTTPSErrors: true,
  });
  const page = await ctx.newPage();
  const url = `${base}/render.html?v=${variant.id}`;
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__renderReady === true, null, { timeout: 60000 });

  for (let i = 0; i < FRAMES; i++) {
    const t = i / FPS;
    // flushSync inside __forceTick commits React synchronously; one RAF is
    // enough to let the browser paint before we screenshot.
    await page.evaluate((tt) => {
      window.__renderTime = tt;
      if (window.__forceTick) window.__forceTick();
      return new Promise(r => requestAnimationFrame(r));
    }, t);
    const file = path.join(framesDir, String(i).padStart(4, '0') + '.jpg');
    await page.screenshot({
      path: file,
      type: 'jpeg',
      quality: 92,
      clip: { x: 0, y: 0, width: W, height: H },
      omitBackground: false,
    });
    if (i % 30 === 0 || i === FRAMES - 1) {
      process.stdout.write(`\r    frames ${i + 1}/${FRAMES}`);
    }
  }
  process.stdout.write('\n');
  await ctx.close();
}

async function captureAudio(browser, base, variant) {
  const ctx = await browser.newContext({ viewport: { width: 800, height: 600 }, ignoreHTTPSErrors: true });
  const page = await ctx.newPage();
  const url = `${base}/render-audio.html?v=${variant.id}`;
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__audioReady === true || window.__audioError != null, null, { timeout: 90000 });
  const err = await page.evaluate(() => window.__audioError);
  if (err) { await ctx.close(); throw new Error('audio render: ' + err); }
  const floats = await page.evaluate(() => Array.from(window.__audioFloats));
  await ctx.close();
  return Float32Array.from(floats);
}

async function renderVariant(browser, base, variant, outDir) {
  console.log(`\n● Variation ${variant.id} — ${variant.label}`);
  const framesDir = path.join(outDir, `frames-${variant.id}`);
  rmSync(framesDir, { recursive: true, force: true });
  mkdirSync(framesDir, { recursive: true });

  console.log('  ▸ capturing frames…');
  await captureFrames(browser, base, variant, framesDir);

  console.log('  ▸ rendering audio…');
  const audio = await captureAudio(browser, base, variant);
  const wavPath = path.join(outDir, `${variant.id}.wav`);
  writeFileSync(wavPath, encodeWav(audio, SAMPLE_RATE));

  console.log('  ▸ encoding mp4…');
  const mp4Path = path.join(outDir, `tiktok-${variant.id}.mp4`);
  await runFfmpeg([
    '-y', '-loglevel', 'error',
    '-framerate', String(FPS),
    '-i', path.join(framesDir, '%04d.jpg'),
    '-i', wavPath,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-preset', 'medium',
    '-r', String(FPS),
    '-c:a', 'aac', '-b:a', '192k',
    '-movflags', '+faststart',
    '-shortest',
    mp4Path,
  ]);
  // Cleanup intermediate frames + wav to keep the renders/ folder lean
  rmSync(framesDir, { recursive: true, force: true });
  rmSync(wavPath, { force: true });
  console.log(`  ✔ ${path.relative(__dirname, mp4Path)}`);
  return mp4Path;
}

(async () => {
  if (!process.env.PLAYWRIGHT_BROWSERS_PATH && existsSync('/opt/pw-browsers')) {
    process.env.PLAYWRIGHT_BROWSERS_PATH = '/opt/pw-browsers';
  }
  await ensureVendor();
  const outDir = path.join(__dirname, 'renders');
  mkdirSync(outDir, { recursive: true });

  const { server, base } = await startStaticServer(__dirname);
  console.log('static server:', base);

  const browser = await chromium.launch({
    headless: true,
    ignoreHTTPSErrors: true,
    args: [
      '--disable-web-security',
      '--no-sandbox',
      '--allow-file-access-from-files',
      '--autoplay-policy=no-user-gesture-required',
      '--ignore-certificate-errors',
      '--ignore-certificate-errors-spki-list',
    ],
  });
  try {
    for (const v of VARIANTS) {
      await renderVariant(browser, base, v, outDir);
    }
  } finally {
    await browser.close();
    server.close();
  }
  console.log('\nDone. Outputs in:', outDir);
})().catch(e => {
  console.error(e);
  process.exit(1);
});
