// Type-safe env loading. Fail fast and fail loud when keys are missing,
// because nothing is more frustrating than a 90-minute Veo render that
// silently uses the wrong credentials.

import { z } from 'zod';
import { config } from 'dotenv';

config(); // populate process.env from .env

const Schema = z.object({
  ELEVENLABS_API_KEY: z.string().min(20).optional(),
  ELEVENLABS_VOICE_ID: z.string().default('XB0fDUnXU5powFXDhCwa'), // Charlotte
  ELEVENLABS_MODEL: z.string().default('eleven_multilingual_v2'),

  GOOGLE_CLOUD_PROJECT: z.string().optional(),
  GOOGLE_CLOUD_LOCATION: z.string().default('us-central1'),
  GOOGLE_APPLICATION_CREDENTIALS: z.string().optional(),
  VEO_MODEL: z.string().default('veo-3.1-generate-preview'),

  PEXELS_API_KEY: z.string().optional(),
  KIE_API_KEY: z.string().optional(),
});

export const env = Schema.parse(process.env);

export function requireEnv<K extends keyof typeof env>(key: K, hint: string): NonNullable<(typeof env)[K]> {
  const v = env[key];
  if (!v) {
    throw new Error(`Missing required env var ${key}.\n  → ${hint}`);
  }
  return v as NonNullable<(typeof env)[K]>;
}
