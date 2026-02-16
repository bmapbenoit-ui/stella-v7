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
    """Mistral API = principal, RunPod = fallback"""
    if MISTRAL_API_KEY:
        return OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=MISTRAL_API_KEY
        ), "mistral-small-latest"
    elif RUNPOD_ENDPOINT and RUNPOD_API_KEY:
        return OpenAI(
            base_url=f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/openai/v1",
            api_key=RUNPOD_API_KEY
        ), "casperhansen/mistral-small-24b-instruct-2501-awq"
    else:
        return None, None

# === AUTO MIGRATION ===
MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id SERIAL PRIMARY KEY,
    project VARCHAR(50) NOT NULL,
    snapshot_type VARCHAR(20) NOT NULL,
    state_data JSONB NOT NULL,
    ai_summary TEXT,
    key_decisions JSONB DEFAULT '[]',
    next_actions JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_snapshots_project ON snapshots(project, created_at DESC);

CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    project VARCHAR(50) NOT NULL,
    decision TEXT NOT NULL,
    reasoning TEXT,
    alternatives_considered JSONB DEFAULT '[]',
    outcome TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS task_queue (
    id SERIAL PRIMARY KEY,
    project VARCHAR(50) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    task_data JSONB NOT NULL,
    priority INTEGER DEFAULT 50,
    status VARCHAR(20) DEFAULT 'PENDING',
    assigned_to VARCHAR(50),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON task_queue(project, status, priority);

CREATE TABLE IF NOT EXISTS operation_logs (
    id SERIAL PRIMARY KEY,
    project VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    details JSONB,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_logs_project ON operation_logs(project, created_at DESC);

CREATE TABLE IF NOT EXISTS lessons_learned (
    id SERIAL PRIMARY KEY,
    project VARCHAR(50),
    error_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    root_cause TEXT,
    solution TEXT NOT NULL,
    prevention TEXT,
    times_encountered INTEGER DEFAULT 1,
    last_encountered TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS product_queue (
    id SERIAL PRIMARY KEY,
    shopify_id BIGINT UNIQUE NOT NULL,
    handle VARCHAR(255) NOT NULL,
    brand VARCHAR(100) NOT NULL,
    title VARCHAR(255) NOT NULL,
    priority INTEGER DEFAULT 50,
    status VARCHAR(20) DEFAULT 'PENDING',
    current_step VARCHAR(100),
    data_json JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_image_cache (
    id SERIAL PRIMARY KEY,
    note_name VARCHAR(255) NOT NULL,
    note_type VARCHAR(10) NOT NULL,
    shopify_file_id VARCHAR(255),
    image_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(note_name, note_type)
);

CREATE TABLE IF NOT EXISTS brand_config (
    id SERIAL PRIMARY KEY,
    brand_name VARCHAR(100) UNIQUE NOT NULL,
    site_url TEXT,
    product_url_pattern TEXT,
    fragrantica_brand_name VARCHAR(100),
    style_bg VARCHAR(7) DEFAULT '#0A0A0A',
    style_border VARCHAR(7) DEFAULT '#C8984E',
    style_font VARCHAR(50) DEFAULT 'serif',
    notes TEXT
);

INSERT INTO brand_config (brand_name, site_url, fragrantica_brand_name, style_bg, style_border) VALUES
    ('BADAR', 'https://badar-paris.com', 'Badar', '#0A0A0A', '#C8984E'),
    ('BDK PARFUMS', 'https://bdkparfums.com', 'BDK Parfums', '#1A1A2E', '#DAA520'),
    ('L''ARTISAN PARFUMEUR', 'https://www.artisanparfumeur.com', 'L''Artisan Parfumeur', '#F5F0E8', '#8B4513'),
    ('GOLDFIELD & BANKS', 'https://goldfieldandbanks.com', 'Goldfield & Banks', '#1C1C1C', '#D4AF37'),
    ('VILHELM PARFUMERIE', 'https://vilhelmparfumerie.com', 'Vilhelm Parfumerie', '#FFFFFF', '#000000'),
    ('LIQUIDES IMAGINAIRES', 'https://www.liquides-imaginaires.com', 'Liquides Imaginaires', '#0D0D0D', '#B8860B'),
    ('FRANCESCA BIANCHI', 'https://francescabianchi.com', 'Francesca Bianchi', '#2C1810', '#D4A574'),
    ('ORTO PARISI', 'https://www.ortoparisi.com', 'Orto Parisi', '#1A1A1A', '#808080'),
    ('NASOMATTO', 'https://www.nasomatto.com', 'Nasomatto', '#000000', '#FFFFFF'),
    ('ROOM 1015', 'https://room1015.com', 'Room 1015', '#1A1A1A', '#FF4500'),
    ('HISTOIRES DE PARFUMS', 'https://www.histoiresdeparfums.com', 'Histoires de Parfums', '#2B1B17', '#C9B037'),
    ('PARLE MOI DE PARFUM', 'https://parle-moi-de-parfum.com', 'Parle Moi de Parfum', '#FAFAFA', '#4A4A4A'),
    ('MAISON TAHITE', 'https://www.maisontahite.com', 'Maison Tahite', '#F8F4E8', '#6B4226'),
    ('JAMES HEELEY', 'https://www.jamesheeley.com', 'James Heeley', '#FFFFFF', '#1A1A1A'),
    ('ESSENTIAL PARFUMS', 'https://essentialparfums.com', 'Essential Parfums', '#FFFFFF', '#000000'),
    ('AMOUAGE', 'https://www.amouage.com', 'Amouage', '#0C0C0C', '#D4AF37'),
    ('FLORAIKU', 'https://www.floraiku.com', 'Floraiku', '#F5F0E8', '#C41E3A'),
    ('PARFUMS DE MARLY', 'https://www.parfums-de-marly.com', 'Parfums de Marly', '#1A1A2E', '#DAA520')
ON CONFLICT (brand_name) DO NOTHING;
"""

@app.on_event("startup")
async def startup():
    if DATABASE_URL:
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute(MIGRATION_SQL)
            db.commit()
            cur.close()
            db.close()
            print("DB migration OK")
        except Exception as e:
            print(f"DB migration error: {e}")

# === MODELS ===
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

# === MEMORY ===
def read_short_memory(project):
    session = r.get(f"stella:session:{project}")
    return json.loads(session) if session else {"project": project, "conversation_buffer": [], "current_task": None}

def read_medium_memory(project, limit=5):
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT ai_summary, key_decisions, next_actions, created_at FROM snapshots WHERE project = %s ORDER BY created_at DESC LIMIT %s", (project, limit))
        rows = cur.fetchall()
        cur.close(); db.close()
        return [dict(row) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def read_long_memory(query, project):
    try:
        resp = httpx.post(f"{EMBEDDING_URL}/search_all", json={"query": query, "collections": ["knowledge","products","decisions","lessons"], "project": project, "top_k_per_collection": 3, "rerank_top": 5}, timeout=30)
        return resp.json().get("results", [])
    except Exception as e:
        return [{"text": f"[RAG indisponible: {e}]", "source": "error"}]

def save_short_memory(project, data):
    r.setex(f"stella:session:{project}", 86400, json.dumps(data, default=str))

def save_snapshot(project, state, summary, decisions, next_actions, snap_type="auto_5min"):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("INSERT INTO snapshots (project, snapshot_type, state_data, ai_summary, key_decisions, next_actions) VALUES (%s,%s,%s,%s,%s,%s)",
                     (project, snap_type, json.dumps(state, default=str), summary, json.dumps(decisions), json.dumps(next_actions)))
        db.commit(); cur.close(); db.close()
    except: pass

def save_to_long_memory(text, project, collection, source):
    try:
        httpx.post(f"{EMBEDDING_URL}/index", json={"text": text, "collection": collection, "project": project, "source": source}, timeout=30)
    except: pass

def build_context(message, project, task_context=""):
    short = read_short_memory(project)
    short_text = ""
    if short.get("current_task"):
        short_text = f"TACHE EN COURS: {short['current_task']}\n"
    if short.get("conversation_buffer"):
        short_text += "DERNIERS ECHANGES:\n"
        for ex in short["conversation_buffer"][-3:]:
            short_text += f"  - {ex.get('role','?')}: {ex.get('content','')[:200]}\n"

    medium = read_medium_memory(project, limit=3)
    medium_text = ""
    if medium and not medium[0].get("error"):
        medium_text = "CONTEXTE RECENT:\n"
        for snap in medium[:3]:
            medium_text += f"  [{snap.get('created_at','?')}] {snap.get('ai_summary','N/A')}\n"
            decs = snap.get('key_decisions', [])
            if isinstance(decs, str):
                try: decs = json.loads(decs)
                except: decs = []
            for d in (decs or [])[:2]:
                medium_text += f"    -> Decision: {d}\n"

    long_results = read_long_memory(message, project)
    long_text = ""
    if long_results and long_results[0].get("source") != "error":
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
{long_text or "Pas de connaissances pertinentes."}"""

# === ENDPOINTS ===
@app.get("/health")
async def health():
    redis_ok = False
    try: redis_ok = r.ping()
    except: pass
    db_ok = False
    tables = []
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        tables = [row[0] for row in cur.fetchall()]
        cur.close(); db.close()
        db_ok = True
    except: pass
    return {"status": "ok" if redis_ok and db_ok else "degraded", "redis": redis_ok, "database": db_ok, "tables": tables, "llm": "mistral_api" if MISTRAL_API_KEY else ("runpod" if RUNPOD_ENDPOINT else "none")}

@app.post("/chat")
async def chat(req: ChatRequest):
    context = build_context(req.message, req.project, req.task_context)
    system = req.system_override or f"""Tu es STELLA, l intelligence artificielle de PlaneteBeauty.
Expert en parfumerie de niche, e-commerce Shopify, marketing digital, gestion d entreprise.
REGLES: 1. Ne JAMAIS inventer de donnees 2. Utiliser UNIQUEMENT les donnees du contexte 3. Repondre en francais 4. Etre precis et oriente action 5. Signaler tout conflit
{context}"""
    client, model = get_llm_client()
    if not client:
        return {"error": "No LLM configured. Set RUNPOD_ENDPOINT_ID+RUNPOD_API_KEY or MISTRAL_API_KEY", "answer": None}
    try:
        response = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":req.message}], temperature=0.3, max_tokens=4096)
        answer = response.choices[0].message.content
    except Exception as e:
        if MISTRAL_API_KEY and RUNPOD_ENDPOINT:
            try:
                fallback = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
                response = fallback.chat.completions.create(model="mistral-small-latest", messages=[{"role":"system","content":system},{"role":"user","content":req.message}], temperature=0.3, max_tokens=4096)
                answer = response.choices[0].message.content
            except Exception as e2:
                return {"error": f"Both LLMs failed: {e} / {e2}", "answer": None}
        else:
            return {"error": str(e), "answer": None}
    short = read_short_memory(req.project)
    buf = short.get("conversation_buffer", [])
    buf.append({"role":"user","content":req.message[:500]})
    buf.append({"role":"assistant","content":answer[:500]})
    short["conversation_buffer"] = buf[-20:]
    save_short_memory(req.project, short)
    return {"answer": answer, "project": req.project, "llm_used": "mistral_api" if MISTRAL_API_KEY else "runpod"}

@app.post("/snapshot")
async def snapshot(req: SnapshotRequest):
    short = read_short_memory(req.project)
    save_snapshot(req.project, short, req.summary or f"Auto-snapshot {req.project}", req.decisions, req.next_actions, req.snapshot_type)
    return {"saved": True, "project": req.project, "type": req.snapshot_type}

@app.post("/learn")
async def learn(req: LearnRequest):
    save_to_long_memory(req.text, req.project, req.collection, req.source)
    return {"indexed": True}

@app.get("/memory/{project}")
async def get_memory(project: str):
    return {"short_term": read_short_memory(project), "medium_term": read_medium_memory(project, limit=5)}

@app.get("/brands")
async def list_brands():
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM brand_config ORDER BY brand_name")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"brands": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e)}

# === ADMIN ENDPOINTS ===

class ProductBatch(BaseModel):
    products: list  # [{shopify_id, handle, brand, title, priority, data_json}]

@app.post("/admin/load-queue")
async def load_product_queue(batch: ProductBatch):
    """Batch load products into product_queue"""
    db = get_db()
    cur = db.cursor()
    inserted = skipped = 0
    for p in batch.products:
        try:
            cur.execute("""
                INSERT INTO product_queue (shopify_id, handle, brand, title, priority, status, data_json)
                VALUES (%s, %s, %s, %s, %s, 'PENDING', %s)
                ON CONFLICT (shopify_id) DO NOTHING
            """, (int(p['shopify_id']), p['handle'], p['brand'], p['title'], 
                  p.get('priority', 50), json.dumps(p.get('data_json', {}), ensure_ascii=False)))
            inserted += 1
        except Exception as e:
            skipped += 1
    db.commit()
    cur.close(); db.close()
    return {"inserted": inserted, "skipped": skipped}

@app.get("/admin/queue-stats")
async def queue_stats():
    """Get product queue statistics"""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT brand, COUNT(*) as total,
               SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) as pending,
               SUM(CASE WHEN status='COMPLETED' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN status='ERROR' THEN 1 ELSE 0 END) as errors,
               MAX(priority) as priority
        FROM product_queue GROUP BY brand ORDER BY MAX(priority) DESC, COUNT(*) DESC
    """)
    brands = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT COUNT(*) as total FROM product_queue")
    total = cur.fetchone()['total']
    cur.close(); db.close()
    return {"total": total, "brands": brands}
