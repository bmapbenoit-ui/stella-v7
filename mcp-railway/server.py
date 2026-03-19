"""
MCP Server — STELLA Railway Memory Access
Donne à Claude Code un accès direct aux 3 couches mémoire de STELLA sur Railway.
"""
import os
import json
import httpx
from mcp.server.fastmcp import FastMCP

# ══════════════════════ CONFIG ══════════════════════
STELLA_APP_URL = os.getenv(
    "STELLA_APP_URL",
    "https://stella-shopify-app-production.up.railway.app"
)
CONTEXT_ENGINE_URL = os.getenv(
    "CONTEXT_ENGINE_URL",
    "https://context-engine-production-e525.up.railway.app"
)
EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "https://embedding-service-production-7f52.up.railway.app"
)
CLAUDE_MEM_KEY = os.getenv("CLAUDE_MEM_KEY", "stella-mem-2026-planetebeauty")

TIMEOUT = 30

mcp = FastMCP("stella-railway")


# ══════════════════════ HELPERS ══════════════════════
async def _get(url, headers=None):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(url, headers=headers or {})
        return r.json()


async def _post(url, data, headers=None):
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(url, json=data, headers=headers or {})
        return r.json()


def _mem_headers():
    return {"X-API-Key": CLAUDE_MEM_KEY}


# ══════════════════════ TOOLS ══════════════════════

@mcp.tool()
async def health_check() -> str:
    """Vérifie l'état de tous les services Railway (Redis, PostgreSQL, Qdrant, LLM)."""
    try:
        app_health = await _get(f"{STELLA_APP_URL}/health")
        ce_health = await _get(f"{CONTEXT_ENGINE_URL}/health")
        return json.dumps({"stella_app": app_health, "context_engine": ce_health}, indent=2, default=str)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def memory_stats() -> str:
    """Récupère les statistiques mémoire : nombre de messages, conversations, documents dans les 3 couches."""
    try:
        stats = await _get(f"{STELLA_APP_URL}/api/memory/stats")
        return json.dumps(stats, indent=2, default=str)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def list_conversations(limit: int = 20) -> str:
    """Liste les conversations récentes avec leur titre, projet et nombre de messages."""
    try:
        data = await _get(f"{STELLA_APP_URL}/api/conversations")
        convs = data.get("conversations", [])[:limit]
        return json.dumps(convs, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def get_conversation_messages(conversation_id: str) -> str:
    """Récupère tous les messages d'une conversation par son ID."""
    try:
        data = await _get(f"{STELLA_APP_URL}/api/conversations/{conversation_id}/messages")
        return json.dumps(data.get("messages", []), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def context_summary() -> str:
    """Récupère le résumé complet du contexte : mémoires importantes, dernier snapshot, stats."""
    try:
        data = await _get(f"{STELLA_APP_URL}/context/summary", headers=_mem_headers())
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def list_memories() -> str:
    """Liste les 50 dernières mémoires Claude (milestones, décisions, apprentissages)."""
    try:
        data = await _get(f"{STELLA_APP_URL}/memory", headers=_mem_headers())
        return json.dumps(data.get("memories", []), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def search_memories(query: str, category: str = "", limit: int = 10) -> str:
    """Recherche dans les mémoires Claude par mot-clé et catégorie optionnelle."""
    try:
        data = await _post(
            f"{STELLA_APP_URL}/memory/search",
            {"query": query, "category": category, "limit": limit},
            headers=_mem_headers()
        )
        return json.dumps(data.get("results", []), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def latest_snapshot() -> str:
    """Récupère le dernier snapshot de session (état, décisions clés, prochaines actions)."""
    try:
        data = await _get(f"{STELLA_APP_URL}/session/snapshot/latest", headers=_mem_headers())
        return json.dumps(data.get("snapshot"), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def rag_search(query: str, project: str = "GENERAL") -> str:
    """Recherche vectorielle RAG dans Qdrant (connaissances, produits, décisions, leçons)."""
    try:
        data = await _post(
            f"{CONTEXT_ENGINE_URL}/learn",  # This is index, need search
            {"text": query, "project": project, "collection": "knowledge", "source": "mcp_search"}
        )
        # Use the embedding service search_all endpoint
        data = await _post(
            f"{EMBEDDING_URL}/search_all",
            {
                "query": query,
                "collections": ["knowledge", "products", "decisions", "lessons"],
                "project": project,
                "top_k_per_collection": 5,
                "rerank_top": 10
            }
        )
        return json.dumps(data.get("results", []), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def shopify_debug() -> str:
    """Récupère les données Shopify brutes (commandes, remboursements) pour debug."""
    try:
        data = await _get(f"{STELLA_APP_URL}/debug/shopify")
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def queue_stats() -> str:
    """Statistiques de la file d'attente produits (enrichissement par marque)."""
    try:
        data = await _get(f"{CONTEXT_ENGINE_URL}/admin/queue-stats")
        return json.dumps(data, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def brands_list() -> str:
    """Liste toutes les marques configurées avec leurs paramètres."""
    try:
        data = await _get(f"{CONTEXT_ENGINE_URL}/brands")
        return json.dumps(data.get("brands", []), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        return f"Erreur: {e}"


@mcp.tool()
async def dump_all_context() -> str:
    """Dump complet : health + stats + dernières conversations + mémoires + snapshot. Utile pour avoir tout le contexte d'un coup."""
    result = {}
    try:
        result["health"] = await _get(f"{STELLA_APP_URL}/health")
    except Exception as e:
        result["health"] = {"error": str(e)}
    try:
        result["memory_stats"] = await _get(f"{STELLA_APP_URL}/api/memory/stats")
    except Exception as e:
        result["memory_stats"] = {"error": str(e)}
    try:
        data = await _get(f"{STELLA_APP_URL}/api/conversations")
        result["recent_conversations"] = data.get("conversations", [])[:10]
    except Exception as e:
        result["recent_conversations"] = {"error": str(e)}
    try:
        result["context_summary"] = await _get(f"{STELLA_APP_URL}/context/summary", headers=_mem_headers())
    except Exception as e:
        result["context_summary"] = {"error": str(e)}
    try:
        result["latest_snapshot"] = await _get(f"{STELLA_APP_URL}/session/snapshot/latest", headers=_mem_headers())
    except Exception as e:
        result["latest_snapshot"] = {"error": str(e)}
    return json.dumps(result, indent=2, default=str, ensure_ascii=False)


# ══════════════════════ RUN ══════════════════════
if __name__ == "__main__":
    mcp.run(transport="stdio")
