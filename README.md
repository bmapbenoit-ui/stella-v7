# STELLA V7 — Cerveau Permanent PlanèteBeauty

## Services

| Service | Dossier | Description |
|---------|---------|-------------|
| Embedding Service | `/embedding-service` | BGE-M3 + Reranker — embeddings et recherche vectorielle |
| Context Engine | `/context-engine` | Chef d orchestre mémoire — fusionne 3 niveaux de mémoire |

## Infrastructure

- **Railway** : n8n, PostgreSQL, Redis, Qdrant, Embedding, Context Engine
- **RunPod** : Mistral Small 3.1 (24B) self-hosted via vLLM
- **Externe** : Shopify GraphQL, OpenAI (images), Telegram, Google Drive
