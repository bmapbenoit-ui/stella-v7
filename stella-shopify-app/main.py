"""
STELLA V7 — Shopify Embedded App
Full AI assistant with permanent memory, document upload, Shopify data
"""
import os, time, hmac, hashlib, base64, json, logging, uuid, re
from datetime import datetime, timezone
from typing import Optional, List
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import redis
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stella")

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", SHOPIFY_SHOP)
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://embedding-service.railway.internal:8080")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis.railway.internal:6379")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))

SHOPIFY_API_VERSION = "2026-01"
SHOPIFY_GRAPHQL_URL = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

app = FastAPI(title="STELLA V7")

# DB + Redis connections
r_client = None
def get_redis():
    global r_client
    if r_client is None:
        try:
            r_client = redis.from_url(REDIS_URL, decode_responses=True)
            r_client.ping()
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            r_client = None
    return r_client

def get_db():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)

# ══════════════════════════════════════════════
# AUTO MIGRATION — chat tables
# ══════════════════════════════════════════════
CHAT_MIGRATION = """
CREATE TABLE IF NOT EXISTS stella_conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) DEFAULT 'Nouvelle conversation',
    project VARCHAR(50) DEFAULT 'GENERAL',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    message_count INTEGER DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON stella_conversations(updated_at DESC);

CREATE TABLE IF NOT EXISTS stella_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES stella_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    attachments JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON stella_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS stella_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    file_size INTEGER,
    content_text TEXT,
    content_summary TEXT,
    indexed_in_qdrant BOOLEAN DEFAULT FALSE,
    conversation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_docs_created ON stella_documents(created_at DESC);

CREATE TABLE IF NOT EXISTS stella_memory_log (
    id SERIAL PRIMARY KEY,
    message_id UUID,
    layer VARCHAR(20) NOT NULL,
    action VARCHAR(50) NOT NULL,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

@app.on_event("startup")
async def startup():
    try:
        db = get_db()
        if db:
            cur = db.cursor()
            cur.execute(CHAT_MIGRATION)
            db.commit()
            cur.close()
            db.close()
            logger.info("Chat tables migration OK")
    except Exception as e:
        logger.error(f"Migration error: {e}")

# ══════════════════════════════════════════════
# MEMORY MANAGER — 3 layers
# ══════════════════════════════════════════════
class MemoryManager:
    """Persist every message to all 3 memory layers."""

    @staticmethod
    def save_to_redis(conversation_id: str, role: str, content: str):
        """Layer 1: Short-term (Redis) — session buffer, last 50 messages."""
        try:
            rc = get_redis()
            if not rc:
                return
            key = f"stella:chat:{conversation_id}"
            msg = json.dumps({"role": role, "content": content[:2000], "ts": datetime.now(timezone.utc).isoformat()})
            rc.rpush(key, msg)
            rc.ltrim(key, -50, -1)  # Keep last 50
            rc.expire(key, 86400 * 30)  # 30 days
            # Also update session context
            session_key = f"stella:session:GENERAL"
            session = rc.get(session_key)
            session_data = json.loads(session) if session else {"project": "GENERAL", "conversation_buffer": []}
            buf = session_data.get("conversation_buffer", [])
            buf.append({"role": role, "content": content[:500]})
            session_data["conversation_buffer"] = buf[-20:]
            rc.setex(session_key, 86400, json.dumps(session_data, default=str))
        except Exception as e:
            logger.error(f"Redis save error: {e}")

    @staticmethod
    def save_to_postgres(conversation_id: str, role: str, content: str, metadata: dict = None, attachments: list = None) -> Optional[str]:
        """Layer 2: Medium-term (PostgreSQL) — permanent storage."""
        try:
            db = get_db()
            if not db:
                return None
            cur = db.cursor()
            msg_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO stella_messages (id, conversation_id, role, content, metadata, attachments) VALUES (%s, %s, %s, %s, %s, %s)",
                (msg_id, conversation_id, role, content, json.dumps(metadata or {}), json.dumps(attachments or []))
            )
            cur.execute(
                "UPDATE stella_conversations SET updated_at = NOW(), message_count = message_count + 1 WHERE id = %s",
                (conversation_id,)
            )
            db.commit()
            cur.close()
            db.close()
            return msg_id
        except Exception as e:
            logger.error(f"PostgreSQL save error: {e}")
            return None

    @staticmethod
    async def save_to_qdrant(content: str, source: str = "chat", project: str = "GENERAL"):
        """Layer 3: Long-term (Qdrant via Embedding Service) — semantic search."""
        try:
            # Only index substantial messages (>50 chars)
            if len(content) < 50:
                return
            async with httpx.AsyncClient(timeout=30) as c:
                await c.post(f"{EMBEDDING_URL}/index", json={
                    "text": content[:3000],
                    "collection": "knowledge",
                    "project": project,
                    "source": source
                })
        except Exception as e:
            logger.error(f"Qdrant index error: {e}")

    @staticmethod
    def log_memory_action(msg_id: str, layer: str, action: str, details: dict = None):
        """Log what was saved where for audit."""
        try:
            db = get_db()
            if not db:
                return
            cur = db.cursor()
            cur.execute(
                "INSERT INTO stella_memory_log (message_id, layer, action, details) VALUES (%s, %s, %s, %s)",
                (msg_id, layer, action, json.dumps(details or {}))
            )
            db.commit()
            cur.close()
            db.close()
        except Exception as e:
            logger.error(f"Memory log error: {e}")

memory = MemoryManager()


# ══════════════════════════════════════════════
# CONVERSATION MANAGEMENT
# ══════════════════════════════════════════════
def create_conversation(title: str = "Nouvelle conversation", project: str = "GENERAL") -> Optional[str]:
    try:
        db = get_db()
        if not db:
            return str(uuid.uuid4())  # Fallback UUID
        cur = db.cursor()
        conv_id = str(uuid.uuid4())
        cur.execute(
            "INSERT INTO stella_conversations (id, title, project) VALUES (%s, %s, %s)",
            (conv_id, title, project)
        )
        db.commit()
        cur.close()
        db.close()
        return conv_id
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        return str(uuid.uuid4())

def get_conversations(limit: int = 30) -> list:
    try:
        db = get_db()
        if not db:
            return []
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, title, project, message_count, updated_at FROM stella_conversations WHERE NOT is_archived ORDER BY updated_at DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        db.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Get conversations error: {e}")
        return []

def get_messages(conversation_id: str, limit: int = 100) -> list:
    try:
        db = get_db()
        if not db:
            return []
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT id, role, content, metadata, attachments, created_at FROM stella_messages WHERE conversation_id = %s ORDER BY created_at ASC LIMIT %s",
            (conversation_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        db.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Get messages error: {e}")
        return []

def update_conversation_title(conversation_id: str, title: str):
    try:
        db = get_db()
        if not db:
            return
        cur = db.cursor()
        cur.execute("UPDATE stella_conversations SET title = %s WHERE id = %s", (title[:255], conversation_id))
        db.commit()
        cur.close()
        db.close()
    except Exception as e:
        logger.error(f"Update title error: {e}")

def auto_title_from_message(message: str) -> str:
    """Generate a short title from the first message."""
    clean = message.strip()[:80]
    if len(clean) > 60:
        clean = clean[:57] + "..."
    return clean or "Nouvelle conversation"


# ══════════════════════════════════════════════
# SHOPIFY GRAPHQL
# ══════════════════════════════════════════════
async def shopify_graphql(query: str, variables: dict = None) -> dict:
    if not SHOPIFY_ACCESS_TOKEN:
        return {"error": "SHOPIFY_ACCESS_TOKEN not configured"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            resp = await c.post(SHOPIFY_GRAPHQL_URL, json=payload, headers={
                "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
                "Content-Type": "application/json"
            })
            return resp.json()
        except Exception as e:
            return {"error": str(e)}


async def fetch_products_summary() -> str:
    data = await shopify_graphql("""{ all: products(first: 250) { edges { node { title status vendor totalInventory } } } }""")
    products = [e["node"] for e in data.get("data", {}).get("all", {}).get("edges", [])]
    active = sum(1 for p in products if p["status"] == "ACTIVE")
    draft = sum(1 for p in products if p["status"] == "DRAFT")
    brands = {}
    for p in products:
        b = p.get("vendor", "?")
        brands[b] = brands.get(b, 0) + 1
    top = sorted(brands.items(), key=lambda x: -x[1])[:10]
    low = [p for p in products if p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]
    oos = [p for p in products if p.get("totalInventory") is not None and p["totalInventory"] <= 0]
    lines = [
        f"SHOPIFY TEMPS REEL ({SHOPIFY_STORE_DOMAIN}):",
        f"Produits: {len(products)} total, {active} actifs, {draft} brouillons",
        f"Marques ({len(brands)}): {', '.join(f'{b}({c})' for b,c in top)}",
    ]
    if oos:
        lines.append(f"RUPTURES ({len(oos)}): {', '.join(p['title'][:35] for p in oos[:5])}")
    if low:
        lines.append(f"STOCK BAS ({len(low)}): {', '.join(p['title'][:35] for p in low[:5])}")
    return "\n".join(lines)


async def fetch_orders_summary() -> str:
    data = await shopify_graphql("""{ orders(first: 50, sortKey: CREATED_AT, reverse: true) { edges { node { name createdAt displayFinancialStatus displayFulfillmentStatus totalPriceSet { shopMoney { amount currencyCode } } lineItems(first: 3) { edges { node { title quantity } } } } } } }""")
    orders = [e["node"] for e in data.get("data", {}).get("orders", {}).get("edges", [])]
    if not orders:
        return "Aucune commande recente."
    total_rev = sum(float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)) for o in orders)
    cur = orders[0].get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode", "EUR")
    paid = sum(1 for o in orders if o.get("displayFinancialStatus") == "PAID")
    lines = [f"COMMANDES (50 dernieres): {len(orders)} commandes, CA: {total_rev:.2f} {cur}, {paid} payees"]
    for o in orders[:5]:
        amt = o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", "?")
        items = [f"{i['node']['title'][:25]}x{i['node']['quantity']}" for i in o.get("lineItems", {}).get("edges", [])[:2]]
        lines.append(f"  {o['name']}|{amt}{cur}|{o.get('displayFinancialStatus','?')}|{','.join(items)}")
    return "\n".join(lines)


async def fetch_inventory_alerts() -> str:
    data = await shopify_graphql("""{ products(first: 250) { edges { node { title vendor status totalInventory } } } }""")
    products = [e["node"] for e in data.get("data", {}).get("products", {}).get("edges", [])]
    critical = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and p["totalInventory"] <= 0]
    low = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]
    lines = [f"ALERTES STOCK: {len(critical)} ruptures, {len(low)} stock bas"]
    for p in critical[:8]:
        lines.append(f"  RUPTURE: {p['title'][:45]} ({p.get('vendor','?')})")
    for p in low[:8]:
        lines.append(f"  BAS({p['totalInventory']}): {p['title'][:45]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# SMART SHOPIFY DETECTION
# ══════════════════════════════════════════════
PRODUCT_KW = ["produit", "product", "catalogue", "marque", "brand", "vendor", "combien de produit", "fiche", "actif"]
ORDER_KW = ["commande", "order", "vente", "revenue", "chiffre", "ca ", "panier", "expedition", "rembours"]
STOCK_KW = ["stock", "inventaire", "rupture", "alerte", "reapprovision"]

def detect_shopify_intent(msg: str) -> List[str]:
    m = msg.lower()
    intents = []
    if any(k in m for k in PRODUCT_KW): intents.append("products")
    if any(k in m for k in ORDER_KW): intents.append("orders")
    if any(k in m for k in STOCK_KW): intents.append("stock")
    if "shopify" in m or ("donn" in m and "acc" in m):
        intents = list(set(intents + ["products", "orders"]))
    return intents

async def build_shopify_context(intents: List[str]) -> str:
    if not SHOPIFY_ACCESS_TOKEN or not intents:
        return ""
    parts = []
    try:
        if "products" in intents: parts.append(await fetch_products_summary())
        if "orders" in intents: parts.append(await fetch_orders_summary())
        if "stock" in intents: parts.append(await fetch_inventory_alerts())
    except Exception as e:
        parts.append(f"[Erreur Shopify: {e}]")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════
# SHOPIFY AUTH
# ══════════════════════════════════════════════
def decode_session_token(token: str) -> dict:
    if not SHOPIFY_API_SECRET:
        raise HTTPException(500, "SHOPIFY_API_SECRET not configured")
    try:
        parts = token.split(".")
        if len(parts) != 3: raise ValueError("Invalid JWT")
        h, p, s = parts
        signing_input = f"{h}.{p}".encode()
        expected = base64.urlsafe_b64encode(
            hmac.new(SHOPIFY_API_SECRET.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(expected, s.rstrip("=")): raise ValueError("Bad signature")
        pad = 4 - len(p) % 4
        if pad != 4: p += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(p))
        now = time.time()
        if payload.get("exp", 0) < now - 10: raise ValueError("Expired")
        if payload.get("nbf", 0) > now + 10: raise ValueError("Not yet valid")
        if payload.get("aud") != SHOPIFY_API_KEY: raise ValueError(f"Wrong audience")
        return payload
    except ValueError as e:
        raise HTTPException(401, f"Invalid token: {e}")

async def verify_request(request: Request) -> dict:
    if DEV_MODE:
        return {"sub": "dev", "dest": SHOPIFY_SHOP}
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return decode_session_token(auth[7:])
    token = request.query_params.get("id_token")
    if token:
        return decode_session_token(token)
    raise HTTPException(401, "No session token")


# ══════════════════════════════════════════════
# API — CHAT (with full memory persistence)
# ══════════════════════════════════════════════
class ChatReq(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    project: str = "GENERAL"

@app.post("/api/chat")
async def chat(req: ChatReq, session: dict = Depends(verify_request)):
    """Main chat endpoint — saves to all 3 memory layers."""
    # 1. Create or get conversation
    conv_id = req.conversation_id
    is_new = False
    if not conv_id:
        conv_id = create_conversation(auto_title_from_message(req.message), req.project)
        is_new = True

    # 2. Save user message to all layers
    user_msg_id = memory.save_to_postgres(conv_id, "user", req.message, {"project": req.project})
    memory.save_to_redis(conv_id, "user", req.message)
    # Index substantial user messages to Qdrant
    if len(req.message) > 80:
        await memory.save_to_qdrant(f"[Benoit] {req.message}", source="chat_user", project=req.project)
    if user_msg_id:
        memory.log_memory_action(user_msg_id, "all", "user_message_saved", {"conv_id": conv_id})

    # 3. Detect Shopify intent & fetch live data
    intents = detect_shopify_intent(req.message)
    shopify_ctx = await build_shopify_context(intents) if intents else ""

    # 4. Build context for Context Engine
    task_ctx = ""
    if shopify_ctx:
        task_ctx = f"\n--- DONNEES SHOPIFY TEMPS REEL ---\n{shopify_ctx}\n--- FIN SHOPIFY ---\nUtilise ces donnees REELLES. Ne dis jamais que tu n'as pas acces."

    # 5. Load recent conversation history for context
    recent_msgs = get_messages(conv_id, limit=20)
    if recent_msgs:
        history_text = "\n--- HISTORIQUE CONVERSATION ---\n"
        for m in recent_msgs[-10:]:
            role_label = "Benoit" if m["role"] == "user" else "STELLA"
            history_text += f"{role_label}: {m['content'][:300]}\n"
        task_ctx += history_text

    # 6. Call Context Engine
    async with httpx.AsyncClient(timeout=120.0) as c:
        try:
            resp = await c.post(f"{CONTEXT_ENGINE_URL}/chat", json={
                "message": req.message,
                "project": req.project,
                "task_context": task_ctx,
            })
            result = resp.json()
        except Exception as e:
            result = {"answer": f"Erreur Context Engine: {e}", "error": str(e)}

    answer = result.get("answer", "Je n'ai pas pu répondre.")

    # 7. Save STELLA response to all layers
    stella_msg_id = memory.save_to_postgres(conv_id, "assistant", answer, {
        "llm": result.get("llm_used", "?"),
        "shopify_data": bool(intents),
        "project": req.project
    })
    memory.save_to_redis(conv_id, "assistant", answer)
    # Index important STELLA responses
    if len(answer) > 100:
        await memory.save_to_qdrant(f"[STELLA] {answer[:1500]}", source="chat_stella", project=req.project)
    if stella_msg_id:
        memory.log_memory_action(stella_msg_id, "all", "stella_response_saved", {"conv_id": conv_id})

    # 8. Auto-title on first message
    if is_new and len(req.message) > 5:
        update_conversation_title(conv_id, auto_title_from_message(req.message))

    return {
        "answer": answer,
        "conversation_id": conv_id,
        "is_new_conversation": is_new,
        "llm_used": result.get("llm_used", "?"),
        "shopify_data": bool(intents),
        "memory_saved": True
    }


# ══════════════════════════════════════════════
# API — CONVERSATIONS
# ══════════════════════════════════════════════
@app.get("/api/conversations")
async def list_conversations(session: dict = Depends(verify_request)):
    return {"conversations": get_conversations(30)}

@app.get("/api/conversations/{conv_id}/messages")
async def conversation_messages(conv_id: str, session: dict = Depends(verify_request)):
    return {"messages": get_messages(conv_id, 200)}

@app.post("/api/conversations")
async def new_conversation(session: dict = Depends(verify_request)):
    conv_id = create_conversation()
    return {"conversation_id": conv_id}

@app.delete("/api/conversations/{conv_id}")
async def archive_conversation(conv_id: str, session: dict = Depends(verify_request)):
    try:
        db = get_db()
        if db:
            cur = db.cursor()
            cur.execute("UPDATE stella_conversations SET is_archived = TRUE WHERE id = %s", (conv_id,))
            db.commit()
            cur.close()
            db.close()
        return {"archived": True}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════
# API — DOCUMENT UPLOAD
# ══════════════════════════════════════════════
@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(None),
    project: str = Form("GENERAL"),
    session: dict = Depends(verify_request)
):
    """Upload and process a document — saved to PostgreSQL + indexed in Qdrant."""
    content = await file.read()
    file_size = len(content)
    filename = file.filename or "document"
    file_type = file.content_type or "application/octet-stream"

    # Extract text based on type
    extracted_text = ""
    if file_type in ("text/plain", "text/csv", "text/markdown", "application/json"):
        try:
            extracted_text = content.decode("utf-8", errors="ignore")[:50000]
        except:
            extracted_text = "[Impossible de lire le fichier texte]"
    elif file_type == "application/pdf":
        extracted_text = f"[Document PDF: {filename}, {file_size} octets — extraction PDF à implémenter]"
    elif file_type.startswith("image/"):
        extracted_text = f"[Image: {filename}, {file_size} octets, type: {file_type}]"
    else:
        extracted_text = f"[Document: {filename}, {file_size} octets, type: {file_type}]"

    # Save to PostgreSQL
    doc_id = None
    try:
        db = get_db()
        if db:
            cur = db.cursor()
            doc_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO stella_documents (id, filename, file_type, file_size, content_text, conversation_id) VALUES (%s,%s,%s,%s,%s,%s)",
                (doc_id, filename, file_type, file_size, extracted_text[:100000], conversation_id)
            )
            db.commit()
            cur.close()
            db.close()
    except Exception as e:
        logger.error(f"Doc save error: {e}")

    # Index in Qdrant if we have text
    if extracted_text and len(extracted_text) > 20:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                await c.post(f"{EMBEDDING_URL}/index", json={
                    "text": f"Document '{filename}': {extracted_text[:3000]}",
                    "collection": "knowledge",
                    "project": project,
                    "source": f"upload:{filename}"
                })
                if doc_id:
                    db2 = get_db()
                    if db2:
                        cur2 = db2.cursor()
                        cur2.execute("UPDATE stella_documents SET indexed_in_qdrant = TRUE WHERE id = %s", (doc_id,))
                        db2.commit()
                        cur2.close()
                        db2.close()
        except Exception as e:
            logger.error(f"Doc index error: {e}")

    # If in a conversation, add a system message about the upload
    if conversation_id:
        memory.save_to_postgres(conversation_id, "system", f"📎 Document uploadé: {filename} ({file_size} octets)", {"type": "upload", "doc_id": doc_id})

    return {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "text_extracted": len(extracted_text) > 20,
        "indexed": True
    }


# ══════════════════════════════════════════════
# API — MEMORY STATS
# ══════════════════════════════════════════════
@app.get("/api/memory/stats")
async def memory_stats(session: dict = Depends(verify_request)):
    stats = {"redis": {}, "postgres": {}, "qdrant": {}}
    try:
        rc = get_redis()
        if rc:
            keys = rc.keys("stella:chat:*")
            stats["redis"]["chat_sessions"] = len(keys)
            stats["redis"]["status"] = "connected"
    except:
        stats["redis"]["status"] = "unavailable"

    try:
        db = get_db()
        if db:
            cur = db.cursor()
            cur.execute("SELECT COUNT(*) FROM stella_conversations WHERE NOT is_archived")
            stats["postgres"]["conversations"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM stella_messages")
            stats["postgres"]["messages"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM stella_documents")
            stats["postgres"]["documents"] = cur.fetchone()[0]
            stats["postgres"]["status"] = "connected"
            cur.close()
            db.close()
    except:
        stats["postgres"]["status"] = "unavailable"

    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(f"{EMBEDDING_URL}/health")
            d = resp.json()
            stats["qdrant"] = d.get("collections", {})
            stats["qdrant"]["status"] = "connected"
    except:
        stats["qdrant"]["status"] = "unavailable"

    return stats


# ══════════════════════════════════════════════
# API — SHOPIFY
# ══════════════════════════════════════════════
@app.get("/api/shopify/status")
async def shopify_status():
    if not SHOPIFY_ACCESS_TOKEN:
        return {"connected": False, "error": "no_token"}
    try:
        data = await shopify_graphql("{ shop { name } }")
        return {"connected": True, "shop": data.get("data", {}).get("shop", {}).get("name")}
    except Exception as e:
        return {"connected": False, "error": str(e)}

@app.post("/api/action")
async def action(req: dict, session: dict = Depends(verify_request)):
    act = req.get("action", "")
    if act == "shopify_products": return {"data": await fetch_products_summary()}
    if act == "shopify_orders": return {"data": await fetch_orders_summary()}
    if act == "shopify_stock": return {"data": await fetch_inventory_alerts()}
    endpoints = {"queue_stats": "/admin/queue-stats", "health": "/health", "next_product": "/admin/next-product"}
    ep = endpoints.get(act)
    if not ep: raise HTTPException(400, f"Unknown: {act}")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{CONTEXT_ENGINE_URL}{ep}")
        return r.json()


# ══════════════════════════════════════════════
# SERVE FRONTEND
# ══════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f:
        html = f.read()
    html = html.replace("__SHOPIFY_API_KEY__", SHOPIFY_API_KEY)
    html = html.replace("__SHOPIFY_SHOP__", SHOPIFY_SHOP)
    html = html.replace("__SHOPIFY_HOST__", request.query_params.get("host", ""))
    return HTMLResponse(html)

@app.get("/health")
async def health():
    db_ok = bool(DATABASE_URL)
    redis_ok = False
    try:
        rc = get_redis()
        if rc: redis_ok = rc.ping()
    except: pass
    return {
        "status": "ok",
        "app": "stella-v7",
        "dev_mode": DEV_MODE,
        "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token",
        "database": db_ok,
        "redis": redis_ok
    }

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
