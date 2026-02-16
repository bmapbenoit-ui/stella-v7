"""
STELLA V7 — Shopify Embedded App (Full Version)
- Persistent chat memory (PostgreSQL) across ALL layers
- File upload & document analysis  
- Conversation management
- Real-time Shopify data
- Auto-vectorization to Qdrant long-term memory
"""
import os, time, hmac, hashlib, base64, json, logging, uuid, re
from datetime import datetime
from typing import Optional, List
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import psycopg2, psycopg2.extras
import redis as redis_lib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stella-shopify")

# ══════════════════════ CONFIG ══════════════════════
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", SHOPIFY_SHOP)
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))
REDIS_URL = os.getenv("REDIS_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SHOPIFY_API_VERSION = "2026-01"
SHOPIFY_GRAPHQL_URL = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(title="STELLA Shopify App")

# ══════════════════════ DB & REDIS ══════════════════════
r_client = None
def get_redis():
    global r_client
    if r_client is None and REDIS_URL:
        try:
            r_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
            r_client.ping()
        except Exception as e:
            logger.warning(f"Redis: {e}")
            r_client = None
    return r_client

def get_db():
    if not DATABASE_URL: return None
    try: return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        logger.error(f"DB: {e}")
        return None

# ══════════════════════ MIGRATION ══════════════════════
CHAT_MIGRATION = """
CREATE TABLE IF NOT EXISTS chat_conversations (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) DEFAULT 'Nouvelle conversation',
    project VARCHAR(50) DEFAULT 'GENERAL',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    message_count INTEGER DEFAULT 0,
    is_archived BOOLEAN DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON chat_conversations(updated_at DESC);
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON chat_messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS chat_documents (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(36) REFERENCES chat_conversations(id) ON DELETE SET NULL,
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size INTEGER,
    text_content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

@app.on_event("startup")
async def startup():
    db = get_db()
    if db:
        try:
            cur = db.cursor(); cur.execute(CHAT_MIGRATION); db.commit(); cur.close(); db.close()
            logger.info("Chat tables ready")
        except Exception as e:
            logger.error(f"Migration: {e}")
            try: db.close()
            except: pass
    get_redis()

# ══════════════════════ 3-LAYER MEMORY ══════════════════════
def save_to_redis(conv_id, role, content):
    rc = get_redis()
    if not rc: return
    try:
        key = f"stella:chat:{conv_id}"
        rc.rpush(key, json.dumps({"role": role, "content": content[:1000], "ts": datetime.now().isoformat()}))
        rc.ltrim(key, -50, -1)
        rc.expire(key, 86400 * 365)  # 1 year TTL
    except Exception as e: logger.warning(f"Redis save: {e}")

def save_to_postgres(conv_id, role, content, metadata=None):
    db = get_db()
    if not db: return
    try:
        cur = db.cursor()
        cur.execute("INSERT INTO chat_messages (conversation_id,role,content,metadata) VALUES (%s,%s,%s,%s)", (conv_id, role, content, json.dumps(metadata or {})))
        cur.execute("UPDATE chat_conversations SET updated_at=NOW(), message_count=message_count+1 WHERE id=%s", (conv_id,))
        db.commit(); cur.close(); db.close()
    except Exception as e:
        logger.error(f"PG save: {e}")
        try: db.close()
        except: pass

async def save_to_qdrant(content, source="chat"):
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            await c.post(f"{CONTEXT_ENGINE_URL}/learn", json={"text": content, "project": "CHAT_HISTORY", "collection": "knowledge", "source": source})
    except Exception as e: logger.warning(f"Qdrant save: {e}")

def should_vectorize(msg, answer):
    combined = (msg + " " + answer).lower()
    kws = ["décid","stratégi","budget","prix","stock","fournisseur","objectif","plan","lancement","problème","solution","important","urgent","commande","client","campagne","google ads","revenue","chiffre","investir","priorit","roadmap","contrat","partenaire","concurrent","résultat","mettre en place","créer","développer","modifier","supprimer"]
    return any(k in combined for k in kws) or len(msg) > 200 or len(answer) > 500

async def save_all_layers(conv_id, role, content, metadata=None):
    save_to_redis(conv_id, role, content)
    save_to_postgres(conv_id, role, content, metadata)

# ══════════════════════ CONVERSATIONS ══════════════════════
def create_conversation(title="Nouvelle conversation", project="GENERAL"):
    conv_id = str(uuid.uuid4())
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("INSERT INTO chat_conversations (id,title,project) VALUES (%s,%s,%s)", (conv_id, title[:255], project))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            logger.error(f"Create conv: {e}")
            try: db.close()
            except: pass
    return conv_id

# ══════════════════════ SHOPIFY API ══════════════════════
async def shopify_graphql(query, variables=None):
    if not SHOPIFY_ACCESS_TOKEN: return {"error": "No token"}
    payload = {"query": query}
    if variables: payload["variables"] = variables
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json=payload, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            return r.json()
        except Exception as e: return {"error": str(e)}

async def fetch_products_summary():
    data = await shopify_graphql("{ all: products(first:250) { edges { node { title status vendor totalInventory } } } }")
    products = [e["node"] for e in data.get("data",{}).get("all",{}).get("edges",[])]
    active = sum(1 for p in products if p["status"]=="ACTIVE")
    brands = {}
    for p in products:
        b = p.get("vendor","?"); brands[b] = brands.get(b,0)+1
    top = sorted(brands.items(), key=lambda x:-x[1])[:10]
    low = [p for p in products if p.get("totalInventory") is not None and 0<p["totalInventory"]<=5]
    out = [p for p in products if p.get("totalInventory") is not None and p["totalInventory"]<=0]
    lines = [f"SHOPIFY TEMPS REEL: {len(products)} produits (Actifs:{active})", f"Marques({len(brands)}): {', '.join(f'{b}({c})' for b,c in top)}"]
    if out: lines.append(f"RUPTURE({len(out)}): {', '.join(p['title'][:35] for p in out[:5])}")
    if low: lines.append(f"STOCK BAS({len(low)}): {', '.join(p['title'][:35] for p in low[:5])}")
    return "\n".join(lines)

async def fetch_orders_summary():
    data = await shopify_graphql("""{ orders(first:50,sortKey:CREATED_AT,reverse:true) { edges { node { name createdAt displayFinancialStatus totalPriceSet { shopMoney { amount currencyCode } } lineItems(first:3) { edges { node { title quantity } } } } } } }""")
    orders = [e["node"] for e in data.get("data",{}).get("orders",{}).get("edges",[])]
    if not orders: return "Aucune commande recente."
    rev = sum(float(o.get("totalPriceSet",{}).get("shopMoney",{}).get("amount",0)) for o in orders)
    cur = orders[0].get("totalPriceSet",{}).get("shopMoney",{}).get("currencyCode","EUR")
    paid = sum(1 for o in orders if o.get("displayFinancialStatus")=="PAID")
    lines = [f"COMMANDES(50 dern): {len(orders)} cmd, CA:{rev:.2f}{cur}, Payees:{paid}"]
    for o in orders[:5]:
        a = o.get("totalPriceSet",{}).get("shopMoney",{}).get("amount","?")
        items = [f"{i['node']['title'][:25]}x{i['node']['quantity']}" for i in o.get("lineItems",{}).get("edges",[])[:3]]
        lines.append(f"  {o['name']}|{a}{cur}|{o.get('displayFinancialStatus','?')}|{','.join(items)}")
    return "\n".join(lines)

async def fetch_stock_alerts():
    data = await shopify_graphql("{ products(first:250) { edges { node { title vendor status totalInventory } } } }")
    products = [e["node"] for e in data.get("data",{}).get("products",{}).get("edges",[])]
    crit = [p for p in products if p.get("status")=="ACTIVE" and p.get("totalInventory") is not None and p["totalInventory"]<=0]
    low = [p for p in products if p.get("status")=="ACTIVE" and p.get("totalInventory") is not None and 0<p["totalInventory"]<=5]
    lines = [f"ALERTES STOCK: {len(crit)} ruptures, {len(low)} bas"]
    for p in crit[:10]: lines.append(f"  RUPTURE: {p['title'][:50]} ({p.get('vendor','?')})")
    for p in low[:10]: lines.append(f"  BAS({p.get('totalInventory')}): {p['title'][:50]}")
    return "\n".join(lines)

async def fetch_refunds_summary():
    today = datetime.now().strftime("%Y-%m-%d")
    # Search last 250 orders to find ANY refund created today (not just refunded-status orders)
    data = await shopify_graphql("""{ orders(first: 250, sortKey: CREATED_AT, reverse: true) {
        edges { node { name createdAt displayFinancialStatus
            refunds { createdAt totalRefundedSet { shopMoney { amount currencyCode } }
                refundLineItems(first:10) { edges { node { quantity lineItem { title } } } } note }
        } }
    } }""")
    orders = [e["node"] for e in data.get("data",{}).get("orders",{}).get("edges",[])]
    
    today_refunds = []
    recent_refunds = []
    for o in orders:
        for ref in o.get("refunds", []):
            ref_date = ref.get("createdAt","")[:10]
            amt = ref.get("totalRefundedSet",{}).get("shopMoney",{}).get("amount","0")
            cur = ref.get("totalRefundedSet",{}).get("shopMoney",{}).get("currencyCode","EUR")
            items = [f"{i['node']['lineItem']['title'][:35]} x{i['node']['quantity']}" for i in ref.get("refundLineItems",{}).get("edges",[])[:5]]
            note = ref.get("note","") or ""
            entry = f"  {o['name']} | {amt} {cur} | {ref_date} | {', '.join(items) if items else 'N/A'}" + (f" | Note: {note}" if note else "")
            if ref_date == today:
                today_refunds.append(entry)
            recent_refunds.append(entry)
    
    lines = [f"REMBOURSEMENTS SHOPIFY:"]
    today_total = sum(float(ref.get("totalRefundedSet",{}).get("shopMoney",{}).get("amount",0)) for o in orders for ref in o.get("refunds",[]) if ref.get("createdAt","")[:10] == today)
    
    if today_refunds:
        lines.append(f"\nAUJOURD'HUI ({today}) - {len(today_refunds)} remboursement(s) - TOTAL: {today_total:.2f} EUR:")
        lines.extend(today_refunds)
    else:
        lines.append(f"\nAUJOURD'HUI ({today}): Aucun remboursement")
    
    all_total = sum(float(ref.get("totalRefundedSet",{}).get("shopMoney",{}).get("amount",0)) for o in orders for ref in o.get("refunds",[]))
    lines.append(f"\nDERNIERS REMBOURSEMENTS (total {len(recent_refunds)}, {all_total:.2f} EUR):")
    lines.extend(recent_refunds[:10])
    return "\n".join(lines)

# Shopify intent detection
PROD_KW = ["produit","product","catalogue","inventaire","marque","brand","vendor","combien de produit","fiche","actif"]
ORD_KW = ["commande","order","vente","revenue","chiffre","ca ","panier","expedition"]
STK_KW = ["stock","rupture","out of stock","alerte","reapprovision"]
REF_KW = ["rembours","refund","refunded","annul","retour client","avoir","credit"]

def detect_shopify_intent(msg):
    m = msg.lower(); intents = []
    if any(k in m for k in PROD_KW): intents.append("products")
    if any(k in m for k in ORD_KW): intents.append("orders")
    if any(k in m for k in STK_KW): intents.append("stock")
    if any(k in m for k in REF_KW): intents.append("refunds")
    if "shopify" in m or ("donn" in m and "acc" in m): intents = list(set(intents+["products","orders"]))
    return intents

async def build_shopify_context(intents):
    if not SHOPIFY_ACCESS_TOKEN or not intents: return ""
    parts = []
    try:
        if "products" in intents: parts.append(await fetch_products_summary())
        if "orders" in intents: parts.append(await fetch_orders_summary())
        if "stock" in intents: parts.append(await fetch_stock_alerts())
        if "refunds" in intents: parts.append(await fetch_refunds_summary())
    except Exception as e: parts.append(f"[Erreur Shopify: {e}]")
    return "\n\n".join(parts)

# ══════════════════════ FILE PROCESSING ══════════════════════
def extract_text(content, filename):
    ext = filename.rsplit(".",1)[-1].lower() if "." in filename else ""
    if ext in ("txt","md","csv","json","html","xml","log"):
        try: return content.decode("utf-8")
        except: return content.decode("latin-1", errors="replace")
    if ext == "pdf":
        return f"[Document PDF: {filename}, {len(content)//1024}KB. Décrivez ce que contient le document pour que je puisse vous aider.]"
    if ext in ("png","jpg","jpeg","webp","gif"):
        return f"[Image: {filename}, {len(content)//1024}KB. Décrivez ce que vous souhaitez que j'en fasse.]"
    if ext in ("xlsx","xls","docx","doc","pptx"):
        return f"[Document {ext.upper()}: {filename}, {len(content)//1024}KB. Posez des questions spécifiques sur son contenu.]"
    return f"[Fichier: {filename}, {len(content)//1024}KB]"

# ══════════════════════ AUTH ══════════════════════
def decode_session_token(token):
    if not SHOPIFY_API_SECRET: raise HTTPException(500, "No secret")
    try:
        parts = token.split(".")
        if len(parts) != 3: raise ValueError("Invalid JWT")
        h, p, s = parts
        exp = base64.urlsafe_b64encode(hmac.new(SHOPIFY_API_SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
        if not hmac.compare_digest(exp, s.rstrip("=")): raise ValueError("Bad sig")
        pad = 4 - len(p) % 4
        if pad != 4: p += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(p))
        now = time.time()
        if payload.get("exp",0) < now-10: raise ValueError("Expired")
        if payload.get("nbf",0) > now+10: raise ValueError("Not yet valid")
        if payload.get("aud") != SHOPIFY_API_KEY: raise ValueError("Wrong aud")
        return payload
    except ValueError as e: raise HTTPException(401, str(e))
    except Exception as e: raise HTTPException(401, str(e))

async def verify_request(request: Request):
    if DEV_MODE: return {"sub": "dev", "dest": SHOPIFY_SHOP}
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "): return decode_session_token(auth[7:])
    token = request.query_params.get("id_token")
    if token: return decode_session_token(token)
    raise HTTPException(401, "No session token")

# ══════════════════════ API: CHAT ══════════════════════
class ChatReq(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    project: str = "GENERAL"

@app.post("/api/chat")
async def chat(req: ChatReq, session: dict = Depends(verify_request)):
    is_new = False
    conv_id = req.conversation_id
    if not conv_id:
        title = req.message.strip()[:80] + ("..." if len(req.message)>80 else "")
        conv_id = create_conversation(title, req.project)
        is_new = True

    # Save user message to ALL layers
    await save_all_layers(conv_id, "user", req.message, {"project": req.project, "ts": datetime.now().isoformat()})

    # Shopify real-time data
    intents = detect_shopify_intent(req.message)
    shopify_ctx = await build_shopify_context(intents) if intents else ""

    # Recent history from Redis
    recent = ""
    rc = get_redis()
    if rc:
        try:
            msgs = rc.lrange(f"stella:chat:{conv_id}", -10, -1)
            if msgs:
                lines = []
                for m in msgs:
                    p = json.loads(m)
                    lines.append(f"{'Benoit' if p['role']=='user' else 'STELLA'}: {p['content'][:300]}")
                recent = "\n--- HISTORIQUE ---\n" + "\n".join(lines) + "\n--- FIN ---\n"
        except: pass

    task_ctx = ""
    # Behavioral instructions FIRST (will be in context that CE sends to LLM)
    task_ctx += """=== REGLES STELLA ===
Tu es STELLA, l'assistante IA personnelle de Benoit (PlaneteBeauty). Tu reponds a TOUT: business, perso, culture generale, code, etc.
REGLE 1: REPONDS DIRECTEMENT avec les donnees ci-dessous. Ne dis JAMAIS "je vais verifier/acceder/proceder".
REGLE 2: Tu as DEJA les donnees. Cite les chiffres IMMEDIATEMENT. Pas de listes d'etapes.
REGLE 3: Si les donnees montrent 0 resultat, dis-le. Si tu n'as pas l'info, dis "je n'ai pas cette donnee" et ARRETE.
REGLE 4: Ne repete JAMAIS la meme reponse. Si Benoit dit que c'est faux, admets-le.
REGLE 5: Tutoie Benoit. Sois concis comme un collegue competent.
=== FIN REGLES ===
"""
    if recent:
        task_ctx += recent

    if shopify_ctx:
        task_ctx += f"\n=== DONNEES SHOPIFY TEMPS REEL (lues maintenant depuis l'API) ===\n{shopify_ctx}\n=== FIN DONNEES SHOPIFY ===\n"

    # Call Context Engine (NO system_override - let CE build context with our task_context + RAG + memory)
    async with httpx.AsyncClient(timeout=120) as c:
        try:
            r = await c.post(f"{CONTEXT_ENGINE_URL}/chat", json={"message": req.message, "project": req.project, "task_context": task_ctx})
            result = r.json()
        except Exception as e:
            result = {"answer": f"Erreur connexion: {e}"}

    answer = result.get("answer", "Pas de réponse")

    # Save STELLA response to ALL layers
    await save_all_layers(conv_id, "assistant", answer, {"llm": result.get("llm_used"), "shopify": intents or None, "ts": datetime.now().isoformat()})

    # Auto-vectorize important exchanges to Qdrant
    if should_vectorize(req.message, answer):
        summary = f"[{datetime.now().strftime('%Y-%m-%d')}] Q: {req.message[:200]} | R: {answer[:400]}"
        await save_to_qdrant(summary, source=f"chat_{conv_id[:8]}")

    return {"answer": answer, "conversation_id": conv_id, "is_new_conversation": is_new, "llm_used": result.get("llm_used"), "shopify_data": bool(intents)}

# ══════════════════════ API: CONVERSATIONS ══════════════════════
@app.get("/api/conversations")
async def list_conversations(session: dict = Depends(verify_request)):
    db = get_db()
    if not db: return {"conversations": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id,title,project,created_at,updated_at,message_count FROM chat_conversations WHERE is_archived=FALSE ORDER BY updated_at DESC LIMIT 100")
        rows = cur.fetchall(); cur.close(); db.close()
        return {"conversations": [{"id":r["id"],"title":r["title"],"project":r["project"],"created_at":r["created_at"].isoformat() if r["created_at"] else None,"updated_at":r["updated_at"].isoformat() if r["updated_at"] else None,"message_count":r["message_count"]} for r in rows]}
    except Exception as e:
        try: db.close()
        except: pass
        return {"conversations": [], "error": str(e)}

@app.get("/api/conversations/{conv_id}/messages")
async def get_messages(conv_id: str, session: dict = Depends(verify_request)):
    db = get_db()
    if not db: return {"messages": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT role,content,metadata,created_at FROM chat_messages WHERE conversation_id=%s ORDER BY created_at", (conv_id,))
        rows = cur.fetchall(); cur.close(); db.close()
        return {"messages": [{"role":r["role"],"content":r["content"],"metadata":r["metadata"] if isinstance(r["metadata"],dict) else {},"created_at":r["created_at"].isoformat() if r["created_at"] else None} for r in rows]}
    except Exception as e:
        try: db.close()
        except: pass
        return {"messages": [], "error": str(e)}

@app.delete("/api/conversations/{conv_id}")
async def delete_conv(conv_id: str, session: dict = Depends(verify_request)):
    db = get_db()
    if not db: return {"deleted": False}
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM chat_messages WHERE conversation_id=%s", (conv_id,))
        cur.execute("DELETE FROM chat_conversations WHERE id=%s", (conv_id,))
        db.commit(); cur.close(); db.close()
        rc = get_redis()
        if rc:
            try: rc.delete(f"stella:chat:{conv_id}")
            except: pass
        return {"deleted": True}
    except Exception as e:
        try: db.close()
        except: pass
        return {"deleted": False, "error": str(e)}

# ══════════════════════ API: UPLOAD ══════════════════════
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), conversation_id: str = Form(None), project: str = Form("GENERAL"), session: dict = Depends(verify_request)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE: raise HTTPException(413, "File too large (10MB max)")
    text = extract_text(content, file.filename)
    
    # Save to PostgreSQL
    db = get_db()
    doc_id = None
    if db:
        try:
            cur = db.cursor()
            cur.execute("INSERT INTO chat_documents (conversation_id,filename,file_type,file_size,text_content) VALUES (%s,%s,%s,%s,%s) RETURNING id", (conversation_id, file.filename, file.content_type, len(content), text[:50000]))
            doc_id = cur.fetchone()[0]; db.commit(); cur.close(); db.close()
        except Exception as e:
            logger.error(f"Doc save: {e}")
            try: db.close()
            except: pass
    
    # Vectorize to Qdrant
    if text and len(text) > 50:
        await save_to_qdrant(f"[Doc {datetime.now().strftime('%Y-%m-%d')}] {file.filename}: {text[:2000]}", source=f"upload_{file.filename}")
    
    return {"success": True, "filename": file.filename, "file_size": len(content), "text_content": text[:5000], "doc_id": doc_id}

# ══════════════════════ API: MEMORY STATS ══════════════════════
@app.get("/api/memory/stats")
async def memory_stats(session: dict = Depends(verify_request)):
    stats = {"redis": {}, "postgres": {}, "qdrant": {}}
    rc = get_redis()
    if rc:
        try: stats["redis"] = {"conversations": len(rc.keys("stella:chat:*")), "connected": True}
        except: stats["redis"] = {"connected": False}
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("SELECT COUNT(*) FROM chat_messages"); msgs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chat_conversations WHERE is_archived=FALSE"); convs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chat_documents"); docs = cur.fetchone()[0]
            cur.close(); db.close()
            stats["postgres"] = {"messages": msgs, "conversations": convs, "documents": docs, "connected": True}
        except Exception as e:
            try: db.close()
            except: pass
            stats["postgres"] = {"connected": False}
    return stats

# ══════════════════════ HEALTH ══════════════════════
@app.get("/health")
async def health():
    redis_ok = False
    rc = get_redis()
    if rc:
        try: redis_ok = rc.ping()
        except: pass
    db_ok = False
    db = get_db()
    if db:
        try:
            cur = db.cursor(); cur.execute("SELECT 1"); cur.close(); db.close(); db_ok = True
        except:
            try: db.close()
            except: pass
    qdrant_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{CONTEXT_ENGINE_URL}/health")
            qdrant_ok = r.json().get("status") == "ok"
    except: pass
    return {"status": "ok" if (redis_ok or db_ok) else "degraded", "redis": redis_ok, "database": db_ok, "qdrant": qdrant_ok, "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token", "dev_mode": DEV_MODE, "llm": "mistral_api" if qdrant_ok else "unknown"}

# ══════════════════════ FRONTEND ══════════════════════
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f: html = f.read()
    html = html.replace("__SHOPIFY_API_KEY__", SHOPIFY_API_KEY)
    html = html.replace("__SHOPIFY_SHOP__", SHOPIFY_SHOP)
    html = html.replace("__SHOPIFY_HOST__", request.query_params.get("host", ""))
    return HTMLResponse(html)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
