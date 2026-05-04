#!/usr/bin/env bash
# sync-to-rag.sh — pousse une note Markdown vers le RAG long-memory de Stella
# (context-engine /learn endpoint, qui chunk + embed dans Qdrant via embedding-service).
#
# Usage:
#   ./scripts/sync-to-rag.sh <markdown_file> [project_tag]
#
# Exemples:
#   ./scripts/sync-to-rag.sh docs/CONVERSATION-MAISON-MATAHA-TIKTOK-2026-05-04.md
#   ./scripts/sync-to-rag.sh CLAUDE.md GENERAL
#
# Defaults:
#   project_tag = MAISON_MATAHA_TIKTOK
#   collection = knowledge

set -euo pipefail

CONTEXT_ENGINE_URL="${CONTEXT_ENGINE_URL:-https://context-engine-production-e525.up.railway.app}"
DEFAULT_PROJECT="MAISON_MATAHA_TIKTOK"

if [ $# -lt 1 ]; then
  echo "usage: $0 <markdown_file> [project_tag]" >&2
  exit 1
fi

FILE="$1"
PROJECT="${2:-$DEFAULT_PROJECT}"

if [ ! -f "$FILE" ]; then
  echo "error: file not found: $FILE" >&2
  exit 1
fi

# /learn payload schema:
#   { "text": "...", "project": "...", "collection": "knowledge", "source": "..." }
SOURCE="$(basename "$FILE")"

# Use jq to safely build the JSON payload (escapes newlines, quotes, etc.)
if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (apt install jq / brew install jq)" >&2
  exit 1
fi

PAYLOAD=$(jq -Rs --arg project "$PROJECT" --arg source "$SOURCE" \
  '{text: ., project: $project, collection: "knowledge", source: $source}' \
  < "$FILE")

echo "→ Syncing $FILE"
echo "  project: $PROJECT"
echo "  endpoint: $CONTEXT_ENGINE_URL/learn"
echo "  size: $(wc -c < "$FILE") bytes"

RESPONSE=$(curl -sS -X POST "$CONTEXT_ENGINE_URL/learn" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "  response: $RESPONSE"

# Sanity check
if echo "$RESPONSE" | jq -e '.indexed == true' >/dev/null 2>&1; then
  echo "✔ Indexed in long-memory (Qdrant collection: knowledge, project: $PROJECT)"
else
  echo "⚠ Unexpected response — verify the context-engine is up." >&2
  exit 2
fi
