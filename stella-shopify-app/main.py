"""
STELLA V7 — Shopify Embedded App
FULL VERSION: Chat persistence, file upload, Shopify API, 3-layer memory
"""
import os, time, hmac, hashlib, base64, json, logging, uuid, re
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stella-shopify")

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", SHOPIFY_SHOP)
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
DATABASE_URL = os.getenv("DATABASE_URL", "")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "https://embedding-service-production-7f52.up.railway.app")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))
MAX_UPLOAD_MB = 10

SHOPIFY_API_VERSION = "2026-01"
SHOPIFY_GRAPHQL_URL = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

app = FastAPI(title="STELLA Shopify App")

# ══════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════
def get_db():
    """Get a database connection. Try direct DATABASE_URL first, fallback to Context Engine's DB."""
    db_url = DATABASE_URL
    if not db_url:
        # Use the same PostgreSQL as Context Engine
        db_url = "postgresql://stella:35f212b369c668b7101c578f2e28aeed@postgresql.railway.internal/stella_v7"
    return psycopg2.connect(db_url)

MIGRATION_SQL = """
-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(200) NOT NULL DEFAULT 'Nouvelle conversation',
    project VARCHAR(50) NOT NULL DEFAULT 'GENERAL',
    user_id VARCHAR(100) DEFAULT 'benoit',
    message_count INTEGER DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_active ON conversations(is_archived, updated_at DESC);

-- Chat messages table - PERMANENT STORAGE
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    shopify_data BOOLEAN DEFAULT FALSE,
    file_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON chat_messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_search ON chat_messages USING gin(to_tsvector('french', content));

-- Uploaded files table
CREATE TABLE IF NOT EXISTS uploaded_files (
    id SERIAL PRIMARY KEY,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    filename VARCHAR(500) NOT NULL,
    original_name VARCHAR(500) NOT NULL,
    mime_type VARCHAR(100),
    file_size INTEGER,
    file_data BYTEA,
    text_content TEXT,
    project VARCHAR(50) DEFAULT 'GENERAL',
    user_id VARCHAR(100) DEFAULT 'benoit',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_files_conv ON uploaded_files(conversation_id);

-- Memory index tracking (what's been indexed to Qdrant)
CREATE TABLE IF NOT EXISTS memory_index_log (
    id SERIAL PRIMARY KEY,
    conversation_id UUID,
    message_id INTEGER,
    collection VARCHAR(50) DEFAULT 'stella_conversations',
    indexed_at TIMESTAMP DEFAULT NOW()
);
"""

@app.on_event("startup")
async def startup():
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(MIGRATION_SQL)
        db.commit()
        cur.close()
        db.close()
        logger.info("Database migration OK — conversations, chat_messages, uploaded_files ready")
    except Exception as e:
        logger.error(f"Database migration error: {e}")
        logger.info("Will retry DB connection on first request")


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
            r = await c.post(SHOPIFY_GRAPHQL_URL, json=payload, headers={
                "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
                "Content-Type": "application/json"
            })
            return r.json()
        except Exception as e:
            return {"error": str(e)}


async def fetch_products_summary() -> str:
    data = await shopify_graphql("""
    { all: products(first: 250) { edges { node { title status vendor productType totalInventory } } } }
    """)
    products = [e["node"] for e in data.get("data", {}).get("all", {}).get("edges", [])]
    active = sum(1 for p in products if p["status"] == "ACTIVE")
    draft = sum(1 for p in products if p["status"] == "DRAFT")
    archived = sum(1 for p in products if p["status"] == "ARCHIVED")
    brands = {}
    for p in products:
        b = p.get("vendor", "?")
        brands[b] = brands.get(b, 0) + 1
    top = sorted(brands.items(), key=lambda x: -x[1])[:10]
    low = [p for p in products if p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]
    oos = [p for p in products if p.get("totalInventory") is not None and p["totalInventory"] <= 0]
    lines = [
        f"SHOPIFY TEMPS REEL ({SHOPIFY_STORE_DOMAIN}):",
        f"Produits: {len(products)} total (Actifs: {active}, Brouillons: {draft}, Archives: {archived})",
        f"Marques ({len(brands)}): {', '.join(f'{b}({c})' for b,c in top)}",
    ]
    if oos:
        oos_names = ", ".join(p["title"][:40] for p in oos[:5])
        lines.append(f"RUPTURES ({len(oos)}): {oos_names}")
    if low:
        low_items = []
        for p in low[:5]:
            inv = p.get("totalInventory", "?")
            low_items.append(f"{p['title'][:35]}({inv})")
        lines.append(f"STOCK BAS ({len(low)}): {', '.join(low_items)}")
    return "\n".join(lines)


async def fetch_orders_summary() -> str:
    data = await shopify_graphql("""
    { orders(first: 50, sortKey: CREATED_AT, reverse: true) {
        edges { node { name createdAt displayFinancialStatus displayFulfillmentStatus
            totalPriceSet { shopMoney { amount currencyCode } }
            lineItems(first: 5) { edges { node { title quantity } } }
        } }
    } }
    """)
    orders = [e["node"] for e in data.get("data", {}).get("orders", {}).get("edges", [])]
    if not orders:
        return "COMMANDES: Aucune commande recente."
    revenue = sum(float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)) for o in orders)
    cur = orders[0].get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode", "EUR")
    paid = sum(1 for o in orders if o.get("displayFinancialStatus") == "PAID")
    lines = [f"COMMANDES (50 dern.): CA={revenue:.2f}{cur}, {paid} payees/{len(orders)} total"]
    for o in orders[:5]:
        amt = o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", "?")
        items = [f"{i['node']['title'][:25]}x{i['node']['quantity']}" for i in o.get("lineItems", {}).get("edges", [])[:3]]
        lines.append(f"  {o['name']}|{amt}{cur}|{o.get('displayFinancialStatus','?')}|{','.join(items)}")
    return "\n".join(lines)


async def fetch_stock_alerts() -> str:
    data = await shopify_graphql("""
    { products(first: 250) { edges { node { title vendor status totalInventory } } } }
    """)
    products = [e["node"] for e in data.get("data", {}).get("products", {}).get("edges", [])]
    critical = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and p["totalInventory"] <= 0]
    low = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]
    lines = [f"ALERTES STOCK: {len(critical)} ruptures, {len(low)} bas"]
    for p in critical[:8]: lines.append(f"  RUPTURE: {p['title'][:50]} ({p.get('vendor','?')})")
    for p in low[:8]:
        inv = p.get("totalInventory", "?")
        lines.append(f"  BAS({inv}): {p['title'][:50]}")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# SMART CONTEXT DETECTION
# ══════════════════════════════════════════════
PRODUCT_KW = ["produit", "product", "catalogue", "marque", "brand", "vendor", "combien de produit", "fiche", "actif"]
ORDER_KW = ["commande", "order", "vente", "revenue", "chiffre", "ca ", "c.a.", "panier", "expedition", "rembours"]
STOCK_KW = ["stock", "inventaire", "rupture", "out of stock", "alerte", "reapprovision", "bas"]

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
        if "stock" in intents: parts.append(await fetch_stock_alerts())
    except Exception as e:
        parts.append(f"[Erreur Shopify: {e}]")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════
# CONVERSATION HISTORY (MEMORY LAYER 1: PostgreSQL)
# ══════════════════════════════════════════════
def get_or_create_conversation(conversation_id: str = None, project: str = "GENERAL", title: str = None) -> tuple:
    """Returns (conversation_id, is_new)"""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    if conversation_id:
        cur.execute("SELECT id FROM conversations WHERE id = %s AND is_archived = FALSE", (conversation_id,))
        if cur.fetchone():
            cur.close(); db.close()
            return conversation_id, False
    
    # Create new conversation
    new_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO conversations (id, title, project) VALUES (%s, %s, %s) RETURNING id",
        (new_id, title or "Nouvelle conversation", project)
    )
    db.commit()
    cur.close(); db.close()
    return new_id, True


def save_message(conversation_id: str, role: str, content: str, metadata: dict = None, shopify_data: bool = False, file_id: int = None) -> int:
    """Save message to PostgreSQL. Returns message ID."""
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """INSERT INTO chat_messages (conversation_id, role, content, metadata, shopify_data, file_id)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (conversation_id, role, content, json.dumps(metadata or {}), shopify_data, file_id)
    )
    msg_id = cur.fetchone()[0]
    
    # Update conversation
    title_sql = ""
    if role == "user":
        # Auto-title from first user message
        cur.execute("SELECT message_count FROM conversations WHERE id = %s", (conversation_id,))
        row = cur.fetchone()
        if row and row[0] == 0:
            title = content[:80].replace('\n', ' ').strip()
            cur.execute("UPDATE conversations SET title = %s WHERE id = %s", (title, conversation_id))
    
    cur.execute(
        "UPDATE conversations SET message_count = message_count + 1, updated_at = NOW() WHERE id = %s",
        (conversation_id,)
    )
    db.commit()
    cur.close(); db.close()
    return msg_id


def get_conversation_messages(conversation_id: str, limit: int = 200) -> List[dict]:
    """Get all messages for a conversation."""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, role, content, metadata, shopify_data, file_id, created_at FROM chat_messages WHERE conversation_id = %s ORDER BY created_at LIMIT %s",
        (conversation_id, limit)
    )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close(); db.close()
    return rows


def get_recent_context(conversation_id: str, n: int = 6) -> str:
    """Get last N messages as context for the LLM."""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT role, content FROM chat_messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s",
        (conversation_id, n)
    )
    msgs = list(reversed([dict(r) for r in cur.fetchall()]))
    cur.close(); db.close()
    if not msgs:
        return ""
    ctx = "HISTORIQUE CONVERSATION:\n"
    for m in msgs:
        prefix = "Benoit" if m["role"] == "user" else "STELLA"
        ctx += f"  {prefix}: {m['content'][:300]}\n"
    return ctx


def list_conversations(user_id: str = "benoit", limit: int = 50) -> List[dict]:
    """List all active conversations."""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT id, title, project, message_count, created_at, updated_at FROM conversations WHERE user_id = %s AND is_archived = FALSE ORDER BY updated_at DESC LIMIT %s",
        (user_id, limit)
    )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close(); db.close()
    return rows


# ══════════════════════════════════════════════
# LONG-TERM MEMORY (Qdrant via Embedding Service)
# ══════════════════════════════════════════════
async def index_to_long_memory(text: str, source: str, project: str = "GENERAL"):
    """Index important content to Qdrant for permanent recall."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            await c.post(f"{EMBEDDING_URL}/index", json={
                "text": text,
                "collection": "stella_conversations",
                "project": project,
                "source": source
            })
    except Exception as e:
        logger.warning(f"Qdrant index error: {e}")


async def search_long_memory(query: str, project: str = "GENERAL") -> str:
    """Search Qdrant for relevant past conversations."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(f"{EMBEDDING_URL}/search_all", json={
                "query": query,
                "collections": ["knowledge", "products", "decisions", "lessons", "stella_conversations"],
                "project": project,
                "top_k_per_collection": 2,
                "rerank_top": 5
            }, timeout=15)
            results = r.json().get("results", [])
            if results:
                text = "MEMOIRE LONG TERME:\n"
                for lr in results[:5]:
                    text += f"  [{lr.get('source','?')}] {lr.get('text','')[:250]}\n"
                return text
    except Exception as e:
        logger.warning(f"Qdrant search error: {e}")
    return ""


# ══════════════════════════════════════════════
# FILE PROCESSING
# ══════════════════════════════════════════════
def extract_text_from_file(content: bytes, mime: str, filename: str) -> str:
    """Extract text content from uploaded files."""
    text = ""
    try:
        if mime and mime.startswith("text/"):
            text = content.decode("utf-8", errors="replace")
        elif filename.endswith(".csv"):
            text = content.decode("utf-8", errors="replace")
        elif filename.endswith(".json"):
            text = content.decode("utf-8", errors="replace")
        elif filename.endswith(".md"):
            text = content.decode("utf-8", errors="replace")
        elif mime and "pdf" in mime:
            text = f"[PDF uploadé: {filename} — {len(content)} octets. Extraction PDF nécessite PyPDF2.]"
        elif mime and ("excel" in mime or "spreadsheet" in mime) or filename.endswith((".xlsx", ".xls")):
            text = f"[Excel uploadé: {filename} — {len(content)} octets]"
        elif mime and "image" in mime:
            text = f"[Image uploadée: {filename} — {len(content)} octets]"
        else:
            try:
                text = content.decode("utf-8", errors="replace")
            except:
                text = f"[Fichier binaire: {filename} — {len(content)} octets]"
    except Exception as e:
        text = f"[Erreur extraction: {e}]"
    return text[:50000]  # Limit to 50k chars


# ══════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════
def decode_session_token(token: str) -> dict:
    if not SHOPIFY_API_SECRET:
        raise HTTPException(500, "SHOPIFY_API_SECRET not configured")
    try:
        parts = token.split(".")
        if len(parts) != 3: raise ValueError("Invalid JWT")
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = base64.urlsafe_b64encode(
            hmac.new(SHOPIFY_API_SECRET.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(expected, sig_b64.rstrip("=")): raise ValueError("Bad signature")
        pad = 4 - len(payload_b64) % 4
        if pad != 4: payload_b64 += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        now = time.time()
        if payload.get("exp", 0) < now - 10: raise ValueError("Expired")
        if payload.get("nbf", 0) > now + 10: raise ValueError("Not yet valid")
        if payload.get("aud") != SHOPIFY_API_KEY: raise ValueError(f"Wrong audience")
        return payload
    except ValueError as e: raise HTTPException(401, f"Invalid token: {e}")
    except Exception as e: raise HTTPException(401, f"Token error: {e}")

async def verify_request(request: Request) -> dict:
    if DEV_MODE: return {"sub": "dev", "dest": SHOPIFY_SHOP}
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "): return decode_session_token(auth[7:])
    token = request.query_params.get("id_token")
    if token: return decode_session_token(token)
    raise HTTPException(401, "No session token")


# ══════════════════════════════════════════════
# API ROUTES — CHAT
# ══════════════════════════════════════════════
class ChatReq(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    project: str = "GENERAL"


@app.post("/api/chat")
async def chat(req: ChatReq, session: dict = Depends(verify_request)):
    """Chat with STELLA — full persistence + Shopify data + memory."""
    
    # 1. Get or create conversation
    conv_id, is_new = get_or_create_conversation(req.conversation_id, req.project, req.message[:80])
    
    # 2. Save user message to PostgreSQL
    save_message(conv_id, "user", req.message)
    
    # 3. Detect Shopify intents & fetch data
    intents = detect_shopify_intent(req.message)
    shopify_context = await build_shopify_context(intents) if intents else ""
    
    # 4. Get conversation history context
    conv_context = get_recent_context(conv_id, n=8)
    
    # 5. Search long-term memory
    long_memory = await search_long_memory(req.message, req.project)
    
    # 6. Build enriched context for Context Engine
    task_context_parts = []
    if conv_context:
        task_context_parts.append(conv_context)
    if shopify_context:
        task_context_parts.append(f"--- DONNEES SHOPIFY TEMPS REEL ---\n{shopify_context}")
    if long_memory:
        task_context_parts.append(long_memory)
    
    task_context = "\n\n".join(task_context_parts)
    if shopify_context:
        task_context += "\nINSTRUCTION: Utilise ces donnees Shopify REELLES. Ne dis JAMAIS que tu n'as pas acces."
    
    # 7. Call Context Engine
    async with httpx.AsyncClient(timeout=120.0) as c:
        try:
            r = await c.post(f"{CONTEXT_ENGINE_URL}/chat", json={
                "message": req.message,
                "project": req.project,
                "task_context": task_context,
            })
            result = r.json()
        except Exception as e:
            result = {"answer": f"Erreur Context Engine: {e}", "error": str(e)}
    
    answer = result.get("answer", "Pas de réponse")
    
    # 8. Save STELLA's response to PostgreSQL
    save_message(conv_id, "assistant", answer, metadata={
        "shopify_intents": intents,
        "llm": result.get("llm_used"),
        "had_shopify_data": bool(shopify_context),
        "had_long_memory": bool(long_memory),
    }, shopify_data=bool(shopify_context))
    
    # 9. Index important exchanges to Qdrant (async, non-blocking)
    if len(req.message) > 50 or intents:
        summary = f"Q: {req.message[:200]}\nA: {answer[:300]}"
        await index_to_long_memory(summary, source=f"conversation:{conv_id}", project=req.project)
    
    return {
        "answer": answer,
        "conversation_id": conv_id,
        "is_new_conversation": is_new,
        "project": req.project,
        "llm_used": result.get("llm_used"),
        "shopify_data": bool(shopify_context),
        "shopify_intents": intents if intents else None,
    }


# ══════════════════════════════════════════════
# API ROUTES — CONVERSATIONS
# ══════════════════════════════════════════════
@app.get("/api/conversations")
async def get_conversations(session: dict = Depends(verify_request)):
    """List all conversations."""
    convs = list_conversations()
    return {"conversations": convs}


@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, session: dict = Depends(verify_request)):
    """Get all messages for a conversation."""
    msgs = get_conversation_messages(conv_id)
    return {"messages": msgs, "conversation_id": conv_id}


@app.delete("/api/conversations/{conv_id}")
async def archive_conversation(conv_id: str, session: dict = Depends(verify_request)):
    """Archive (soft-delete) a conversation. Data stays in DB for memory."""
    db = get_db()
    cur = db.cursor()
    cur.execute("UPDATE conversations SET is_archived = TRUE WHERE id = %s", (conv_id,))
    db.commit()
    cur.close(); db.close()
    return {"archived": True}


# ══════════════════════════════════════════════
# API ROUTES — FILE UPLOAD
# ══════════════════════════════════════════════
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: str = Form(None),
    project: str = Form("GENERAL"),
    session: dict = Depends(verify_request)
):
    """Upload a file, extract text, save to DB, index to Qdrant."""
    
    # Read file
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max {MAX_UPLOAD_MB}MB")
    
    # Extract text
    text_content = extract_text_from_file(content, file.content_type, file.filename)
    
    # Get or create conversation
    if conversation_id:
        conv_id, is_new = get_or_create_conversation(conversation_id, project)
    else:
        conv_id, is_new = get_or_create_conversation(None, project, f"📎 {file.filename}")
    
    # Save to DB
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """INSERT INTO uploaded_files (conversation_id, filename, original_name, mime_type, file_size, file_data, text_content, project)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (conv_id, f"{uuid.uuid4()}_{file.filename}", file.filename, file.content_type, len(content),
         psycopg2.Binary(content), text_content, project)
    )
    file_id = cur.fetchone()[0]
    db.commit()
    cur.close(); db.close()
    
    # Save system message about the upload
    save_message(conv_id, "system", f"📎 Fichier uploadé: {file.filename} ({len(content)//1024}KB, {file.content_type})", file_id=file_id)
    
    # Index text to Qdrant for long-term memory
    if text_content and len(text_content) > 20:
        await index_to_long_memory(
            f"Document: {file.filename}\n{text_content[:2000]}",
            source=f"upload:{file.filename}",
            project=project
        )
    
    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "text_extracted": len(text_content) if text_content else 0,
        "conversation_id": conv_id,
        "is_new_conversation": is_new,
    }


# ══════════════════════════════════════════════
# API ROUTES — ACTIONS & STATUS
# ══════════════════════════════════════════════
class ActionReq(BaseModel):
    action: str

@app.post("/api/action")
async def action(req: ActionReq, session: dict = Depends(verify_request)):
    if req.action == "shopify_products":
        return {"data": await fetch_products_summary()}
    if req.action == "shopify_orders":
        return {"data": await fetch_orders_summary()}
    if req.action == "shopify_stock":
        return {"data": await fetch_stock_alerts()}
    endpoints = {"queue_stats": "/admin/queue-stats", "health": "/health", "next_product": "/admin/next-product"}
    ep = endpoints.get(req.action)
    if not ep: raise HTTPException(400, f"Unknown action: {req.action}")
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{CONTEXT_ENGINE_URL}{ep}")
        return r.json()


@app.get("/api/shopify/status")
async def shopify_status():
    if not SHOPIFY_ACCESS_TOKEN:
        return {"connected": False, "error": "No token"}
    try:
        data = await shopify_graphql("{ shop { name } }")
        return {"connected": True, "shop": data.get("data", {}).get("shop", {}).get("name")}
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/api/stats")
async def stats():
    """Dashboard stats."""
    try:
        db = get_db()
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) as total_conversations FROM conversations WHERE is_archived = FALSE")
        convs = cur.fetchone()["total_conversations"]
        cur.execute("SELECT COUNT(*) as total_messages FROM chat_messages")
        msgs = cur.fetchone()["total_messages"]
        cur.execute("SELECT COUNT(*) as total_files FROM uploaded_files")
        files = cur.fetchone()["total_files"]
        cur.close(); db.close()
        return {"conversations": convs, "messages": msgs, "files": files, "shopify": bool(SHOPIFY_ACCESS_TOKEN)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/search")
async def search_messages(q: str, session: dict = Depends(verify_request)):
    """Full-text search across all conversations."""
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT cm.id, cm.role, cm.content, cm.created_at, c.title as conversation_title, c.id as conversation_id
           FROM chat_messages cm JOIN conversations c ON cm.conversation_id = c.id
           WHERE to_tsvector('french', cm.content) @@ plainto_tsquery('french', %s)
           ORDER BY cm.created_at DESC LIMIT 20""",
        (q,)
    )
    results = [dict(r) for r in cur.fetchall()]
    cur.close(); db.close()
    return {"results": results, "query": q}


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
    db_ok = False
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT 1")
        cur.close(); db.close()
        db_ok = True
    except: pass
    return {
        "status": "ok" if db_ok else "degraded",
        "app": "stella-shopify-v2",
        "dev_mode": DEV_MODE,
        "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token",
        "database": "connected" if db_ok else "disconnected",
        "features": ["chat_persistence", "file_upload", "voice_input", "shopify_data", "long_term_memory"]
    }


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
