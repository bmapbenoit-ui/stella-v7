// ElevenLabs voice-over wrapper. We render each line of the script as a
// separate MP3 so the Remotion timeline can position each utterance
// precisely against the visual beats. Cheaper than full pre-mix and lets
// us A/B voice IDs without re-rendering everything.

import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { env, requireEnv } from './env.js';

export type VoiceLine = {
  /** Stable file name (without extension) */
  id: string;
  /** French text — supports SSML break tags via <break time="0.4s"/> */
  text: string;
  /** Override the default voice for this line */
  voiceId?: string;
};

const VO_OUTPUT_DIR = path.resolve('assets/voiceover');

export async function renderLine(line: VoiceLine): Promise<string> {
  const apiKey = requireEnv('ELEVENLABS_API_KEY', 'set ELEVENLABS_API_KEY in .env');
  const client = new ElevenLabsClient({ apiKey });

  const stream = await client.textToSpeech.convert(line.voiceId ?? env.ELEVENLABS_VOICE_ID, {
    text: line.text,
    modelId: env.ELEVENLABS_MODEL,
    voiceSettings: {
      // Tuned for an intimate / breathy whisper that fits Maison Mataha's mood
      stability: 0.45,
      similarityBoost: 0.75,
      style: 0.35,
      useSpeakerBoost: true,
    },
    outputFormat: 'mp3_44100_192',
  });

  await mkdir(VO_OUTPUT_DIR, { recursive: true });
  const outPath = path.join(VO_OUTPUT_DIR, `${line.id}.mp3`);
  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  await writeFile(outPath, Buffer.concat(chunks));
  return outPath;
}
