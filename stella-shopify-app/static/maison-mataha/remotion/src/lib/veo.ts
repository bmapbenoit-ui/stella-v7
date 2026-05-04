// Veo 3.1 client — direct Vertex AI access (no KIE middleman so we control
// cost and quotas). Generates one clip per shot with up to 3 reference
// images for character/scene consistency across shots.
//
// Veo 3.1 capabilities used:
//   • text-to-video with synchronized audio
//   • referenceImages (up to 3 per shot) — locks identity & style
//   • aspectRatio "9:16" — TikTok native
//   • durationSeconds — up to 8s per clip
//
// Pricing (May 2026): ~$0.75/s generated, billed per request.
// Scenario C uses 9 shots × ~1.5s usable = ~13.5s output, so total cost is
// closer to $40-50 (we still ask Veo for 8s clips and trim — gives us
// freedom on cut points).

import { GoogleGenAI } from '@google/genai';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { env, requireEnv } from './env.js';

export type VeoShot = {
  /** Stable file name (without extension) */
  id: string;
  /** Cinematographer-style prompt — see prompt-guide.md */
  prompt: string;
  /** Optional negative prompt to suppress unwanted artifacts */
  negativePrompt?: string;
  /** Up to 3 paths to reference images (subject / object / scene) */
  referenceImages?: string[];
  /** 4–8 seconds — Veo 3 native range */
  durationSeconds?: number;
};

const VEO_OUTPUT_DIR = path.resolve('assets/veo');

export async function generateShot(shot: VeoShot): Promise<string> {
  const project = requireEnv('GOOGLE_CLOUD_PROJECT', 'set GOOGLE_CLOUD_PROJECT in .env to your GCP project ID');
  requireEnv('GOOGLE_APPLICATION_CREDENTIALS', 'set GOOGLE_APPLICATION_CREDENTIALS to the path of your service account JSON');

  const ai = new GoogleGenAI({
    vertexai: true,
    project,
    location: env.GOOGLE_CLOUD_LOCATION,
  });

  const referenceImages = await loadReferenceImages(shot.referenceImages ?? []);

  const op = await ai.models.generateVideos({
    model: env.VEO_MODEL,
    prompt: shot.prompt,
    config: {
      aspectRatio: '9:16',
      durationSeconds: shot.durationSeconds ?? 8,
      personGeneration: 'allow_all',
      negativePrompt: shot.negativePrompt,
      referenceImages: referenceImages.length ? referenceImages : undefined,
    },
  });

  // Veo is async — poll the long-running operation
  let done = op;
  while (!done.done) {
    await new Promise((r) => setTimeout(r, 10_000));
    done = await ai.operations.getVideosOperation({ operation: done });
    process.stdout.write('.');
  }
  process.stdout.write('\n');

  const video = done.response?.generatedVideos?.[0]?.video;
  if (!video) throw new Error(`Veo returned no video for shot "${shot.id}"`);

  await mkdir(VEO_OUTPUT_DIR, { recursive: true });
  const outPath = path.join(VEO_OUTPUT_DIR, `${shot.id}.mp4`);
  if (video.uri) {
    // Vertex AI hands back a GCS URI — download into outPath
    await ai.files.download({ file: video, downloadPath: outPath });
  } else if (video.videoBytes) {
    await writeFile(outPath, Buffer.from(video.videoBytes, 'base64'));
  } else {
    throw new Error(`Veo response had no uri or videoBytes for "${shot.id}"`);
  }
  return outPath;
}

async function loadReferenceImages(paths: string[]) {
  if (paths.length === 0) return [];
  if (paths.length > 3) throw new Error('Veo 3 accepts at most 3 reference images per shot');
  return Promise.all(
    paths.map(async (p) => {
      const data = await readFile(path.resolve(p));
      const ext = path.extname(p).slice(1).toLowerCase() || 'png';
      return {
        image: {
          imageBytes: data.toString('base64'),
          mimeType: ext === 'jpg' ? 'image/jpeg' : `image/${ext}`,
        },
      };
    })
  );
}
