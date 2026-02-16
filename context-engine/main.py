import os, json, time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import redis
import psycopg2
import psycopg2.extras
import httpx
from openai import OpenAI

app = FastAPI(title="STELLA Context Engine")

REDIS_URL = os.getenv("REDIS_URL", "redis://redis.railway.internal:6379")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding-service.railway.internal:8080")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

r = redis.from_url(REDIS_URL, decode_responses=True)

def get_db():
    return psycopg2.connect(DATABASE_URL)

def get_llm_client():
    if RUNPOD_ENDPOINT and RUNPOD_API_KEY:
        return OpenAI(
            base_url=f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/openai/v1",
            api_key=RUNPOD_API_KEY
        ), "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    elif MISTRAL_API_KEY:
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=MISTRAL_API_KEY
        ), "mistral-small-latest"
    else:
        return None, None

class ChatRequest(BaseModel):
    message: str
    project: str = "GENERAL"
    task_context: str = ""
    system_override: str = ""

class SnapshotRequest(BaseModel):
    project: str
    snapshot_type: str = "auto_5min"
    summary: str = ""
    decisions: List[str] = []
    next_actions: List[str] = []

class LearnRequest(BaseModel):
    text: str
    project: str = "GENERAL"
    collection: str = "knowledge"
    source: str = "manual"

def read_short_memory(project: str) -> dict:
    session = r.get(f"stella:session:{project}")
    if session:
        return json.loads(session)
    return {"project": project, "conversation_buffer": [], "current_task": None}

def read_medium_memory(project: str, limit: int = 5) -> list:
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT ai_summary, key_decisions, next_actions, created_at
            FROM snapshots WHERE project = %s
            ORDER BY created_at DESC LIMIT %s
        """, (project, limit))
        rows = cur.fetchall()
        cur.close()
        db.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def read_long_memory(query: str, project: str) -> list:
    try:
        resp = httpx.post(f"{EMBEDDING_URL}/search_all", json={
            "query": query,
            "collections": ["knowledge", "products", "decisions", "lessons"],
            "project": project,
            "top_k_per_collection": 3,
            "rerank_top": 5
        }, timeout=30)
        return resp.json().get("results", [])
    except Exception as e:
        return [{"text": f"[RAG indisponible: {e}]", "source": "error"}]

def save_short_memory(project: str, data: dict):
    r.setex(f"stella:session:{project}", 86400, json.dumps(data, default=str))

def save_snapshot(project, state, summary, decisions, next_actions, snap_type="auto_5min"):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            INSERT INTO snapshots (project, snapshot_type, state_data, ai_summary, key_decisions, next_actions)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (project, snap_type, json.dumps(state, default=str), summary,
              json.dumps(decisions), json.dumps(next_actions)))
        db.commit()
        cur.close()
        db.close()
    except Exception:
        pass

def save_to_long_memory(text, project, collection, source):
    try:
        httpx.post(f"{EMBEDDING_URL}/index", json={
            "text": text, "collection": collection,
            "project": project, "source": source
        }, timeout=30)
    except Exception:
        pass

def build_context(message: str, project: str, task_context: str = "") -> str:
    short = read_short_memory(project)
    short_text = ""
    if short.get("current_task"):
        short_text = f"TACHE EN COURS: {short['current_task']}\n"
    if short.get("conversation_buffer"):
        last_3 = short["conversation_buffer"][-3:]
        short_text += "DERNIERS ECHANGES:\n"
        for ex in last_3:
            short_text += f"  - {ex.get('role','?')}: {ex.get('content','')[:200]}\n"

    medium = read_medium_memory(project, limit=3)
    medium_text = ""
    if medium and not medium[0].get("error"):
        medium_text = "CONTEXTE RECENT:\n"
        for snap in medium[:3]:
            medium_text += f"  [{snap.get('created_at','?')}] {snap.get('ai_summary','N/A')}\n"
            decisions = snap.get('key_decisions', [])
            if isinstance(decisions, str):
                try: decisions = json.loads(decisions)
                except: decisions = []
            for d in (decisions or [])[:2]:
                medium_text += f"    -> Decision: {d}\n"

    long_results = read_long_memory(message, project)
    long_text = ""
    if long_results and not long_results[0].get("source") == "error":
        long_text = "CONNAISSANCES PERTINENTES:\n"
        for lr in long_results[:5]:
            long_text += f"  [{lr.get('source','?')}] {lr.get('text','')[:300]}\n"

    return f"""PROJET ACTIF: {project}
{task_context}

--- MEMOIRE COURTE ---
{short_text or "Pas de tache en cours."}

--- MEMOIRE MOYENNE ---
{medium_text or "Pas d historique recent."}

--- MEMOIRE LONGUE ---
{long_text or "Pas de connaissances pertinentes."}
"""

@app.get("/health")
async def health():
    redis_ok = False
    try:
        redis_ok = r.ping()
    except: pass
    db_ok = False
    try:
        db = get_db()
        db.close()
        db_ok = True
    except: pass
    return {
        "status": "ok" if redis_ok and db_ok else "degraded",
        "redis": redis_ok, "database": db_ok,
        "llm": "runpod" if RUNPOD_ENDPOINT else ("mistral_api" if MISTRAL_API_KEY else "none")
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    context = build_context(req.message, req.project, req.task_context)
    system = req.system_override or f"""Tu es STELLA, l intelligence artificielle de PlaneteBeauty.
Expert en parfumerie de niche, e-commerce Shopify, marketing digital, gestion d entreprise.

REGLES:
1. Ne JAMAIS inventer de donnees
2. Utiliser UNIQUEMENT les donnees du contexte
3. Repondre en francais
4. Etre precis, concis, oriente action
5. Signaler tout conflit ou incoherence

{context}"""

    client, model = get_llm_client()
    if not client:
        return {"error": "No LLM configured", "answer": None}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": req.message}
            ],
            temperature=0.3, max_tokens=4096
        )
        answer = response.choices[0].message.content
    except Exception as e:
        if MISTRAL_API_KEY and RUNPOD_ENDPOINT:
            try:
                fallback = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
                response = fallback.chat.completions.create(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": req.message}
                    ],
                    temperature=0.3, max_tokens=4096
                )
                answer = response.choices[0].message.content
            except Exception as e2:
                return {"error": f"Both LLMs failed: RunPod={e}, Mistral={e2}", "answer": None}
        else:
            return {"error": str(e), "answer": None}

    short = read_short_memory(req.project)
    buf = short.get("conversation_buffer", [])
    buf.append({"role": "user", "content": req.message[:500]})
    buf.append({"role": "assistant", "content": answer[:500]})
    short["conversation_buffer"] = buf[-20:]
    save_short_memory(req.project, short)

    return {"answer": answer, "project": req.project, "llm_used": "runpod" if RUNPOD_ENDPOINT else "mistral_api"}

@app.post("/snapshot")
async def snapshot(req: SnapshotRequest):
    short = read_short_memory(req.project)
    save_snapshot(req.project, short, req.summary or f"Auto-snapshot {req.project}",
                  req.decisions, req.next_actions, req.snapshot_type)
    return {"saved": True, "project": req.project, "type": req.snapshot_type}

@app.post("/learn")
async def learn(req: LearnRequest):
    save_to_long_memory(req.text, req.project, req.collection, req.source)
    return {"indexed": True}

@app.get("/memory/{project}")
async def get_memory(project: str):
    return {
        "short_term": read_short_memory(project),
        "medium_term": read_medium_memory(project, limit=5),
    }
