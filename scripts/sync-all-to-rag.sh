#!/usr/bin/env bash
# sync-all-to-rag.sh — pousse tous les documents de référence vers le RAG.
# À exécuter quand un nouveau snapshot de la conversation ou un nouveau
# document de décision est ajouté à docs/.
#
# Usage:
#   ./scripts/sync-all-to-rag.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNC="$DIR/scripts/sync-to-rag.sh"

cd "$DIR"

echo "═══ Syncing all reference docs to context-engine RAG ═══"
echo

# 1. Top-level CLAUDE.md (project context for future sessions)
"$SYNC" CLAUDE.md GENERAL
echo

# 2. All conversation snapshots in docs/
for f in docs/CONVERSATION-*.md; do
  [ -f "$f" ] || continue
  "$SYNC" "$f" MAISON_MATAHA_TIKTOK
  echo
done

# 3. Project READMEs that document architecture
[ -f "stella-shopify-app/static/maison-mataha/remotion/README.md" ] && \
  "$SYNC" stella-shopify-app/static/maison-mataha/remotion/README.md MAISON_MATAHA_TIKTOK

echo "═══ Sync complete ═══"
