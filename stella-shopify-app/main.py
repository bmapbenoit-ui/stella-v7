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
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import aiosmtplib
from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
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
SHOPIFY_V8_CLIENT_ID = os.getenv("SHOPIFY_V8_CLIENT_ID", "")
SHOPIFY_V8_CLIENT_SECRET = os.getenv("SHOPIFY_V8_CLIENT_SECRET", "")
HOST_URL = os.getenv("HOST_URL", "")
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))
REDIS_URL = os.getenv("REDIS_URL", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
SHOPIFY_API_VERSION = "2026-01"
SHOPIFY_GRAPHQL_URL = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
MAX_FILE_SIZE = 10 * 1024 * 1024

# SMTP config for BIS notification emails
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "PlanèteBeauty")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "")

app = FastAPI(title="STELLA Shopify App")

# CORS for BIS (Back In Stock) endpoints — storefront needs to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://planetebeauty.com", "https://www.planetebeauty.com", "https://planetemode.myshopify.com"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

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
CREATE TABLE IF NOT EXISTS claude_memories (
    id SERIAL PRIMARY KEY,
    memory_type VARCHAR(50) DEFAULT 'milestone',
    category VARCHAR(100) DEFAULT 'general',
    importance INTEGER DEFAULT 5,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_claude_mem ON claude_memories(category, importance DESC);
CREATE TABLE IF NOT EXISTS shopify_tokens (
    id SERIAL PRIMARY KEY,
    shop VARCHAR(255) NOT NULL UNIQUE,
    access_token TEXT NOT NULL,
    scope TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS claude_snapshots (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100),
    snapshot_type VARCHAR(50) DEFAULT 'milestone',
    state_data JSONB DEFAULT '{}',
    ai_summary TEXT,
    key_decisions JSONB DEFAULT '[]',
    next_actions JSONB DEFAULT '[]',
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS bis_subscriptions (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    product_id BIGINT NOT NULL,
    variant_id BIGINT NOT NULL,
    product_title VARCHAR(500),
    product_handle VARCHAR(255),
    subscribed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notified_at TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    status VARCHAR(20) DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_bis_product ON bis_subscriptions(product_id, status);
CREATE INDEX IF NOT EXISTS idx_bis_email ON bis_subscriptions(email, status);
"""

def load_shopify_token_from_db():
    """Load V8 OAuth token from DB if available (overrides env var)."""
    global SHOPIFY_ACCESS_TOKEN
    db = get_db()
    if not db: return
    try:
        cur = db.cursor()
        cur.execute("SELECT access_token FROM shopify_tokens WHERE shop=%s ORDER BY updated_at DESC LIMIT 1", (SHOPIFY_STORE_DOMAIN,))
        row = cur.fetchone()
        if row and row[0]:
            SHOPIFY_ACCESS_TOKEN = row[0]
            logger.info(f"Loaded Shopify V8 token from DB: {SHOPIFY_ACCESS_TOKEN[:12]}...")
        cur.close(); db.close()
    except Exception as e:
        logger.warning(f"Load token from DB: {e}")
        try: db.close()
        except: pass

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
    load_shopify_token_from_db()

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
    today = datetime.now().strftime("%Y-%m-%d")
    today_start = f"{today}T00:00:00Z"
    # Query: today's orders ONLY
    q_today = f"""{{ orders(first:50, sortKey:CREATED_AT, reverse:true, query:"created_at:>={today_start}") {{
        edges {{ node {{ name createdAt displayFinancialStatus
            totalPriceSet {{ shopMoney {{ amount currencyCode }} }}
            currentTotalPriceSet {{ shopMoney {{ amount currencyCode }} }}
            refunds {{ createdAt totalRefundedSet {{ shopMoney {{ amount currencyCode }} }} }}
            lineItems(first:3) {{ edges {{ node {{ title quantity }} }} }}
        }} }}
    }} }}"""
    # Query: last 5 orders only (just for "dernières commandes" context)
    q_recent = """{ orders(first:5, sortKey:CREATED_AT, reverse:true) {
        edges { node { name createdAt displayFinancialStatus
            totalPriceSet { shopMoney { amount currencyCode } }
            currentTotalPriceSet { shopMoney { amount currencyCode } }
            lineItems(first:3) { edges { node { title quantity } } }
        } }
    } }"""

    data_today = await shopify_graphql(q_today)
    data_recent = await shopify_graphql(q_recent)

    orders_today = [e["node"] for e in data_today.get("data",{}).get("orders",{}).get("edges",[])]
    orders_recent = [e["node"] for e in data_recent.get("data",{}).get("orders",{}).get("edges",[])]

    def fmt_order(o):
        # Use currentTotalPriceSet (net after refunds) when available
        price_set = o.get("currentTotalPriceSet") or o.get("totalPriceSet", {})
        a = price_set.get("shopMoney",{}).get("amount","?")
        cur = price_set.get("shopMoney",{}).get("currencyCode","EUR")
        items = [f"{i['node']['title'][:30]} x{i['node']['quantity']}" for i in o.get("lineItems",{}).get("edges",[])[:3]]
        return f"  {o['name']} | {a} {cur} | {o.get('displayFinancialStatus','?')} | {', '.join(items)}"

    lines = []

    # TODAY'S DATA (this is what matters most)
    if orders_today:
        gross = sum(float(o.get("totalPriceSet",{}).get("shopMoney",{}).get("amount",0)) for o in orders_today)
        # Net = use currentTotalPriceSet which accounts for refunds
        net = sum(float((o.get("currentTotalPriceSet") or o.get("totalPriceSet",{})).get("shopMoney",{}).get("amount",0)) for o in orders_today)
        refunds = sum(float(ref.get("totalRefundedSet",{}).get("shopMoney",{}).get("amount",0)) for o in orders_today for ref in o.get("refunds",[]))
        paid = sum(1 for o in orders_today if o.get("displayFinancialStatus") in ("PAID","PARTIALLY_REFUNDED"))
        refunded = sum(1 for o in orders_today if o.get("displayFinancialStatus") == "REFUNDED")
        cur = orders_today[0].get("totalPriceSet",{}).get("shopMoney",{}).get("currencyCode","EUR")
        lines.append(f"=== COMMANDES AUJOURD'HUI ({today}) ===")
        lines.append(f"Nombre: {len(orders_today)} commandes ({paid} payees, {refunded} remboursees)")
        lines.append(f"CA brut: {gross:.2f} {cur}")
        if refunds > 0:
            lines.append(f"Remboursements: -{refunds:.2f} {cur}")
            lines.append(f"CA net (apres remboursements): {net:.2f} {cur}")
        else:
            lines.append(f"CA net: {gross:.2f} {cur} (aucun remboursement)")
        lines.append(f"Detail des commandes:")
        for o in orders_today:
            lines.append(fmt_order(o))
    else:
        lines.append(f"=== COMMANDES AUJOURD'HUI ({today}) ===")
        lines.append("Aucune commande aujourd'hui.")

    # LAST 5 ORDERS (for "dernières commandes" questions)
    if orders_recent:
        lines.append(f"\n=== 5 DERNIERES COMMANDES (toutes dates) ===")
        for o in orders_recent[:5]:
            date = o.get("createdAt","")[:10]
            lines.append(f"  [{date}] " + fmt_order(o).strip())

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
    logger.info(f"REFUND: Searching refunds, today={today}")
    # Sort by UPDATED_AT to catch refunds on old orders (refund updates the order)
    data = await shopify_graphql("""{ orders(first: 100, sortKey: UPDATED_AT, reverse: true) {
        edges { node { name createdAt updatedAt displayFinancialStatus
            refunds { createdAt totalRefundedSet { shopMoney { amount currencyCode } }
                refundLineItems(first:10) { edges { node { quantity lineItem { title } } } } note }
        } }
    } }""")
    orders = [e["node"] for e in data.get("data",{}).get("orders",{}).get("edges",[])]
    logger.info(f"REFUND: Got {len(orders)} orders from Shopify")
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
    logger.info(f"REFUND: Found {len(today_refunds)} today, {len(recent_refunds)} total")
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

# ══════════════════════ OAUTH V8 ══════════════════════
import secrets
from fastapi.responses import RedirectResponse

OAUTH_SCOPES = ",".join([
    "read_products", "write_products", "read_orders", "read_customers",
    "read_analytics", "read_inventory", "read_content", "read_themes",
    "write_script_tags", "read_script_tags", "read_metaobjects",
    "read_metaobject_definitions", "read_locations", "read_shipping",
    "read_product_listings", "write_product_listings", "read_files",
    "read_publications", "read_discounts", "read_price_rules",
    "read_fulfillments", "read_draft_orders", "read_markets",
    "read_translations", "read_online_store_pages",
    "read_online_store_navigation", "read_locales",
    "write_custom_pixels", "read_custom_pixels",
    "read_customer_events", "read_reports",
])

async def exchange_session_for_offline_token(session_token: str, shop: str) -> dict:
    """Exchange a Shopify session token (JWT) for an offline access token via Token Exchange API."""
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"https://{shop}/admin/oauth/access_token",
            data={
                "client_id": SHOPIFY_V8_CLIENT_ID,
                "client_secret": SHOPIFY_V8_CLIENT_SECRET,
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "subject_token": session_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:id-token",
                "requested_token_type": "urn:shopify:params:oauth:token-type:offline-access-token",
            },
        )
        if r.status_code != 200:
            logger.error(f"Token exchange failed: {r.status_code} {r.text[:300]}")
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        return r.json()

def store_shopify_token(shop: str, access_token: str, scope: str = ""):
    """Store Shopify token in PostgreSQL and update global."""
    global SHOPIFY_ACCESS_TOKEN
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute(
                """INSERT INTO shopify_tokens (shop, access_token, scope, updated_at)
                   VALUES (%s, %s, %s, NOW())
                   ON CONFLICT (shop) DO UPDATE
                   SET access_token=EXCLUDED.access_token, scope=EXCLUDED.scope, updated_at=NOW()""",
                (shop, access_token, scope),
            )
            db.commit(); cur.close(); db.close()
            logger.info(f"Stored V8 token for {shop}")
        except Exception as e:
            logger.error(f"Store token: {e}")
            try: db.close()
            except: pass
    SHOPIFY_ACCESS_TOKEN = access_token
    logger.info(f"Global token updated: {access_token[:12]}...")

@app.post("/auth/token-exchange")
async def auth_token_exchange(request: Request):
    """Exchange a session token for an offline access token (Token Exchange API).
    Called from embedded app frontend when loaded in Shopify admin."""
    if not SHOPIFY_V8_CLIENT_ID or not SHOPIFY_V8_CLIENT_SECRET:
        raise HTTPException(500, "V8 credentials not configured")
    # Get session token from Authorization header
    auth = request.headers.get("authorization", "")
    session_token = None
    if auth.startswith("Bearer "):
        session_token = auth[7:]
    if not session_token:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        session_token = body.get("session_token")
    if not session_token:
        raise HTTPException(400, "No session token provided")
    # Decode to get shop
    try:
        parts = session_token.split(".")
        pad = 4 - len(parts[1]) % 4
        if pad != 4: parts[1] += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(parts[1]))
        dest = payload.get("dest", "")
        shop = dest.replace("https://", "").replace("http://", "")
        if not shop: shop = SHOPIFY_STORE_DOMAIN
    except: shop = SHOPIFY_STORE_DOMAIN
    # Exchange
    result = await exchange_session_for_offline_token(session_token, shop)
    if "error" in result:
        raise HTTPException(500, result["error"])
    access_token = result.get("access_token")
    scope = result.get("scope", "")
    if not access_token:
        raise HTTPException(500, "No access_token in response")
    store_shopify_token(shop, access_token, scope)
    return {"success": True, "shop": shop, "scope": scope, "token_preview": f"{access_token[:10]}...{access_token[-4:]}"}

@app.get("/auth/install")
async def auth_install(request: Request):
    """Start OAuth authorization code flow (fallback for Token Exchange).
    Visit this URL in your browser to authorize the app and get an offline token."""
    shop = request.query_params.get("shop", SHOPIFY_STORE_DOMAIN)
    client_id = SHOPIFY_V8_CLIENT_ID or SHOPIFY_API_KEY
    if not client_id:
        return HTMLResponse("<h1>Error</h1><p>No client_id configured</p>", status_code=500)
    redirect_uri = f"{HOST_URL}/auth/callback" if HOST_URL else f"https://stella-shopify-app-production.up.railway.app/auth/callback"
    scopes = "read_products,write_products,read_orders,read_customers,write_customers,read_content,write_content,read_themes,write_themes,read_inventory,write_inventory"
    nonce = str(uuid.uuid4())
    # Store nonce in Redis for CSRF verification
    rc = get_redis()
    if rc:
        try: rc.setex(f"oauth_nonce:{nonce}", 600, shop)
        except: pass
    auth_url = f"https://{shop}/admin/oauth/authorize?client_id={client_id}&scope={scopes}&redirect_uri={redirect_uri}&state={nonce}"
    logger.info(f"OAuth install: redirecting to {auth_url}")
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    """OAuth callback — exchange authorization code for offline access token."""
    code = request.query_params.get("code")
    shop = request.query_params.get("shop", SHOPIFY_STORE_DOMAIN)
    state = request.query_params.get("state")
    if not code:
        return HTMLResponse("<h1>Erreur</h1><p>Pas de code d'autorisation reçu</p>", status_code=400)
    # Verify nonce if stored
    rc = get_redis()
    if rc and state:
        try:
            stored = rc.get(f"oauth_nonce:{state}")
            if stored:
                rc.delete(f"oauth_nonce:{state}")
        except: pass
    # Exchange code for access token
    client_id = SHOPIFY_V8_CLIENT_ID or SHOPIFY_API_KEY
    client_secret = SHOPIFY_V8_CLIENT_SECRET or SHOPIFY_API_SECRET
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"https://{shop}/admin/oauth/access_token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                },
            )
            if r.status_code != 200:
                logger.error(f"OAuth callback failed: {r.status_code} {r.text[:500]}")
                return HTMLResponse(f"<h1>Erreur</h1><p>Échange token échoué: HTTP {r.status_code}</p><pre>{r.text[:500]}</pre>", status_code=500)
            result = r.json()
    except Exception as e:
        logger.error(f"OAuth callback exception: {e}")
        return HTMLResponse(f"<h1>Erreur</h1><p>Exception: {e}</p>", status_code=500)
    access_token = result.get("access_token")
    scope = result.get("scope", "")
    if not access_token:
        return HTMLResponse(f"<h1>Erreur</h1><p>Pas de token dans la réponse</p><pre>{json.dumps(result, indent=2)}</pre>", status_code=500)
    # Store token in DB + update global
    store_shopify_token(shop, access_token, scope)
    preview = f"{access_token[:12]}...{access_token[-4:]}"
    logger.info(f"OAuth callback SUCCESS: token {preview} for {shop}, scopes: {scope}")
    return HTMLResponse(f"""<html><body style="font-family:system-ui;max-width:600px;margin:60px auto;text-align:center">
<h1 style="color:#16A34A">Token obtenu avec succes !</h1>
<p><strong>Shop:</strong> {shop}</p>
<p><strong>Token:</strong> <code>{preview}</code></p>
<p><strong>Scopes:</strong> {scope}</p>
<hr>
<p>Le token est stocke en base de donnees et actif. STELLA peut maintenant acceder a Shopify.</p>
<p><a href="/health">Verifier le health check</a></p>
</body></html>""")

@app.get("/auth/token-status")
async def token_status(request: Request):
    """Check current token status."""
    verify_claude_key(request)
    token_preview = f"{SHOPIFY_ACCESS_TOKEN[:10]}...{SHOPIFY_ACCESS_TOKEN[-4:]}" if len(SHOPIFY_ACCESS_TOKEN) > 14 else "none"
    # Test token
    working = False
    if SHOPIFY_ACCESS_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(
                    SHOPIFY_GRAPHQL_URL,
                    json={"query": "{ shop { name } }"},
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                )
                working = r.status_code == 200 and "errors" not in r.json()
        except: pass
    # Check DB token
    db_token = None
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("SELECT access_token, scope, updated_at FROM shopify_tokens WHERE shop=%s", (SHOPIFY_STORE_DOMAIN,))
            row = cur.fetchone()
            if row:
                db_token = {"preview": f"{row[0][:10]}...{row[0][-4:]}", "scope_count": len(row[1].split(",")) if row[1] else 0, "updated_at": row[2].isoformat() if row[2] else None}
            cur.close(); db.close()
        except:
            try: db.close()
            except: pass
    return {"token_preview": token_preview, "working": working, "db_token": db_token, "v8_client_id": SHOPIFY_V8_CLIENT_ID[:8] + "..." if SHOPIFY_V8_CLIENT_ID else None}

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

# ══════════════════════ API: CLAUDE MEMORY (replaces stella-memory-v2) ══════════════════════
CLAUDE_MEM_KEY = "stella-mem-2026-planetebeauty"

def verify_claude_key(request):
    key = request.headers.get("X-API-Key", "")
    if key != CLAUDE_MEM_KEY:
        raise HTTPException(401, "Invalid API key")
    return True

class MemoryCreate(BaseModel):
    memory_type: str = "milestone"
    category: str = "general"
    importance: int = 5
    title: str
    content: str
    tags: list = []

class MemorySearch(BaseModel):
    query: str
    category: str = ""
    limit: int = 10

class SnapshotCreate(BaseModel):
    session_id: str = ""
    snapshot_type: str = "milestone"
    state_data: dict = {}
    ai_summary: str = ""
    key_decisions: list = []
    next_actions: list = []

@app.get("/context/summary")
async def context_summary(request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error", "message": "DB unavailable"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Last 20 memories by importance
        cur.execute("SELECT id,memory_type,category,importance,title,content,tags,created_at FROM claude_memories ORDER BY importance DESC, created_at DESC LIMIT 20")
        memories = cur.fetchall()
        # Latest snapshot
        cur.execute("SELECT * FROM claude_snapshots ORDER BY created_at DESC LIMIT 1")
        snap = cur.fetchone()
        # Chat stats
        cur.execute("SELECT COUNT(*) as cnt FROM chat_messages")
        msg_count = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) as cnt FROM chat_conversations WHERE is_archived=FALSE")
        conv_count = cur.fetchone()["cnt"]
        cur.close(); db.close()
        return {
            "status": "ok",
            "memories": [{**m, "created_at": m["created_at"].isoformat() if m["created_at"] else None} for m in memories],
            "latest_snapshot": {**snap, "created_at": snap["created_at"].isoformat() if snap["created_at"] else None} if snap else None,
            "stats": {"messages": msg_count, "conversations": conv_count, "memories": len(memories)}
        }
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

@app.get("/memory")
async def list_memories(request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"memories": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM claude_memories ORDER BY created_at DESC LIMIT 50")
        rows = cur.fetchall(); cur.close(); db.close()
        return {"memories": [{**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]}
    except Exception as e:
        try: db.close()
        except: pass
        return {"memories": [], "error": str(e)}

@app.post("/memory")
async def create_memory(mem: MemoryCreate, request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error", "message": "DB unavailable"}
    try:
        cur = db.cursor()
        cur.execute("INSERT INTO claude_memories (memory_type,category,importance,title,content,tags) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (mem.memory_type, mem.category, mem.importance, mem.title, mem.content, json.dumps(mem.tags)))
        mid = cur.fetchone()[0]
        db.commit(); cur.close(); db.close()
        # Also vectorize important memories to Qdrant
        if mem.importance >= 7:
            await save_to_qdrant(f"[{mem.category}] {mem.title}: {mem.content}", source=f"claude_memory_{mid}")
        return {"status": "ok", "id": mid}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

@app.post("/memory/search")
async def search_memories(search: MemorySearch, request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"results": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        q = f"%{search.query}%"
        if search.category:
            cur.execute("SELECT * FROM claude_memories WHERE category=%s AND (title ILIKE %s OR content ILIKE %s) ORDER BY importance DESC LIMIT %s",
                (search.category, q, q, search.limit))
        else:
            cur.execute("SELECT * FROM claude_memories WHERE title ILIKE %s OR content ILIKE %s ORDER BY importance DESC LIMIT %s",
                (q, q, search.limit))
        rows = cur.fetchall(); cur.close(); db.close()
        return {"results": [{**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows]}
    except Exception as e:
        try: db.close()
        except: pass
        return {"results": [], "error": str(e)}

@app.post("/session/snapshot")
async def create_snapshot(snap: SnapshotCreate, request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error"}
    try:
        cur = db.cursor()
        cur.execute("INSERT INTO claude_snapshots (session_id,snapshot_type,state_data,ai_summary,key_decisions,next_actions) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
            (snap.session_id, snap.snapshot_type, json.dumps(snap.state_data), snap.ai_summary, json.dumps(snap.key_decisions), json.dumps(snap.next_actions)))
        sid = cur.fetchone()[0]
        db.commit(); cur.close(); db.close()
        return {"status": "ok", "id": sid}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

@app.get("/session/snapshot/latest")
async def latest_snapshot(request: Request):
    verify_claude_key(request)
    db = get_db()
    if not db: return {"snapshot": None}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM claude_snapshots ORDER BY created_at DESC LIMIT 1")
        snap = cur.fetchone(); cur.close(); db.close()
        if snap: snap["created_at"] = snap["created_at"].isoformat() if snap["created_at"] else None
        return {"snapshot": snap}
    except Exception as e:
        try: db.close()
        except: pass
        return {"snapshot": None, "error": str(e)}

# ══════════════════════ HEALTH ══════════════════════
APP_VERSION = "2.8.0-oauth-fallback"

@app.get("/debug/shopify")
async def debug_shopify():
    """Debug endpoint to see raw Shopify data"""
    try:
        orders = await fetch_orders_summary()
        refunds = await fetch_refunds_summary()
        return {"version": APP_VERSION, "orders_raw": orders[:1000], "refunds_raw": refunds[:1000]}
    except Exception as e:
        return {"version": APP_VERSION, "error": str(e)}

# ══════════════════════ BACK IN STOCK (BIS) ══════════════════════

class BISSubscribeRequest(BaseModel):
    email: str
    product_id: int
    variant_id: int
    product_title: str = ""
    product_handle: str = ""

# ══════════════════════ PRODUCT VIEW TRACKING ══════════════════════

@app.post("/api/views/{product_id}")
async def track_view(product_id: str, request: Request):
    """Track a product page view. Returns current count (rolling 24h window)."""
    rc = get_redis()
    if not rc:
        return {"count": 0, "error": "redis_unavailable"}
    key = f"pv:{product_id}"
    now = int(time.time())
    pipe = rc.pipeline()
    # Add this view with timestamp as score
    pipe.zadd(key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
    # Remove entries older than 24h
    pipe.zremrangebyscore(key, 0, now - 86400)
    # Count remaining
    pipe.zcard(key)
    # Set TTL to auto-cleanup
    pipe.expire(key, 90000)
    results = pipe.execute()
    count = results[2]
    return {"count": count, "product_id": product_id}

@app.get("/api/views/{product_id}")
async def get_views(product_id: str):
    """Get current view count for a product (rolling 24h)."""
    rc = get_redis()
    if not rc:
        return {"count": 0}
    key = f"pv:{product_id}"
    now = int(time.time())
    # Clean old entries
    rc.zremrangebyscore(key, 0, now - 86400)
    count = rc.zcard(key)
    return {"count": count, "product_id": product_id}

# ══════════════════════ BACK IN STOCK ══════════════════════

@app.post("/api/bis/subscribe")
async def bis_subscribe(req: BISSubscribeRequest):
    """Public endpoint: subscribe email for back-in-stock notification."""
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Email invalide")

    # 1. Save to PostgreSQL
    db = get_db()
    if not db:
        raise HTTPException(500, "Database unavailable")
    try:
        cur = db.cursor()
        # Upsert: if same email+variant already active, skip
        cur.execute("""
            INSERT INTO bis_subscriptions (email, product_id, variant_id, product_title, product_handle)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING id
        """, (email, req.product_id, req.variant_id, req.product_title, req.product_handle))
        db.commit()
        new_id = cur.fetchone()
        cur.close()
        db.close()
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"BIS subscribe DB: {e}")
        # Likely duplicate, still return success
        return {"success": True, "message": "Vous serez averti(e) dès que ce produit sera disponible."}

    # 2. Tag customer in Shopify (native approach)
    if SHOPIFY_ACCESS_TOKEN:
        try:
            tag = f"bis-{req.product_handle}"
            async with httpx.AsyncClient(timeout=10) as client:
                # Search for existing customer
                search_q = f'{{"query": "query {{ customers(first:1, query:\\"email:{email}\\") {{ edges {{ node {{ id tags }} }} }} }}"}}'
                sr = await client.post(
                    SHOPIFY_GRAPHQL_URL,
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    content=search_q
                )
                sr_data = sr.json()
                edges = sr_data.get("data", {}).get("customers", {}).get("edges", [])

                if edges:
                    # Customer exists: add tag
                    cust_id = edges[0]["node"]["id"]
                    existing_tags = edges[0]["node"].get("tags", [])
                    if tag not in existing_tags:
                        existing_tags.append(tag)
                        if "bis-notify" not in existing_tags:
                            existing_tags.append("bis-notify")
                        tags_str = json.dumps(existing_tags)
                        mut = f'{{"query": "mutation {{ customerUpdate(input: {{id: \\"{cust_id}\\", tags: {tags_str}}}) {{ customer {{ id }} userErrors {{ field message }} }} }}"}}'
                        await client.post(
                            SHOPIFY_GRAPHQL_URL,
                            headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                            content=mut
                        )
                else:
                    # Create new customer with tag
                    mut_create = json.dumps({"query": """
                        mutation($input: CustomerInput!) {
                            customerCreate(input: $input) {
                                customer { id }
                                userErrors { field message }
                            }
                        }
                    """, "variables": {"input": {
                        "email": email,
                        "tags": [tag, "bis-notify"],
                        "emailMarketingConsent": {
                            "marketingState": "SUBSCRIBED",
                            "marketingOptInLevel": "SINGLE_OPT_IN"
                        }
                    }}})
                    await client.post(
                        SHOPIFY_GRAPHQL_URL,
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                        content=mut_create
                    )
            logger.info(f"BIS: {email} subscribed for {req.product_handle}, tagged in Shopify")
        except Exception as e:
            logger.warning(f"BIS Shopify tag: {e}")
            # Non-blocking: subscription saved in DB even if Shopify tag fails

    return {"success": True, "message": "Vous serez averti(e) dès que ce produit sera disponible."}


@app.get("/api/bis/dashboard")
async def bis_dashboard(request: Request):
    """Dashboard: subscription counts per product."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    db = get_db()
    if not db:
        return {"total_active": 0, "products": [], "recent": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Total active
        cur.execute("SELECT COUNT(*) as total FROM bis_subscriptions WHERE status='active'")
        total = cur.fetchone()["total"]

        # Per product
        cur.execute("""
            SELECT product_id, product_title, product_handle, COUNT(*) as sub_count,
                   MAX(subscribed_at) as last_sub
            FROM bis_subscriptions WHERE status='active'
            GROUP BY product_id, product_title, product_handle
            ORDER BY sub_count DESC LIMIT 20
        """)
        products = cur.fetchall()

        # Recent 10
        cur.execute("""
            SELECT email, product_title, product_handle, subscribed_at
            FROM bis_subscriptions WHERE status='active'
            ORDER BY subscribed_at DESC LIMIT 10
        """)
        recent = cur.fetchall()

        cur.close()
        db.close()

        # Serialize datetimes
        for p in products:
            if p.get("last_sub"):
                p["last_sub"] = p["last_sub"].isoformat()
        for r in recent:
            if r.get("subscribed_at"):
                r["subscribed_at"] = r["subscribed_at"].isoformat()

        return {"total_active": total, "products": products, "recent": recent}
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"BIS dashboard: {e}")
        return {"total_active": 0, "products": [], "recent": [], "error": str(e)}


# ══════════════════════ BIS DASHBOARD WEB ══════════════════════

BIS_DASHBOARD_TOKEN = os.getenv("BIS_DASHBOARD_TOKEN", "pb-bis-2026")

@app.get("/bis", response_class=HTMLResponse)
async def bis_dashboard_web(request: Request, token: str = ""):
    """Visual BIS dashboard — accessible via /bis?token=pb-bis-2026"""
    if token != BIS_DASHBOARD_TOKEN:
        return HTMLResponse("<h2 style='font-family:sans-serif;padding:40px'>Accès refusé — ajoutez ?token=pb-bis-2026</h2>", status_code=403)

    db = get_db()
    total = 0
    products = []
    recent = []
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COUNT(*) as total FROM bis_subscriptions WHERE status='active'")
            total = cur.fetchone()["total"]
            cur.execute("""
                SELECT product_id, product_title, product_handle, COUNT(*) as sub_count,
                       MAX(subscribed_at) as last_sub
                FROM bis_subscriptions WHERE status='active'
                GROUP BY product_id, product_title, product_handle
                ORDER BY sub_count DESC
            """)
            products = cur.fetchall()
            cur.execute("""
                SELECT email, product_title, product_handle, subscribed_at, status
                FROM bis_subscriptions
                ORDER BY subscribed_at DESC LIMIT 50
            """)
            recent = cur.fetchall()
            cur.close()
            db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.error(f"BIS dashboard web: {e}")

    # Count AB Signature specifically
    ab_products = [p for p in products if "AB Signature" in (p.get("product_title") or "")]
    ab_total = sum(p["sub_count"] for p in ab_products)

    # Build product rows
    product_rows = ""
    for p in products:
        is_ab = "AB Signature" in (p.get("product_title") or "")
        badge = ' <span style="background:#D4AF37;color:#fff;font-size:11px;padding:2px 8px;border-radius:4px;margin-left:6px">AB Signature</span>' if is_ab else ""
        last = str(p.get("last_sub", ""))[:16].replace("T", " ")
        product_rows += f"""<tr>
            <td style="padding:12px 16px;border-bottom:1px solid #f0ede8">{p['product_title']}{badge}</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f0ede8;text-align:center;font-weight:700;font-size:20px;color:#D4AF37">{p['sub_count']}</td>
            <td style="padding:12px 16px;border-bottom:1px solid #f0ede8;color:#888;font-size:13px">{last}</td>
        </tr>"""

    # Build recent rows
    recent_rows = ""
    for r in recent:
        dt = str(r.get("subscribed_at", ""))[:16].replace("T", " ")
        status = r.get("status", "active")
        st_color = "#4CAF50" if status == "active" else "#FF9800" if status == "notified" else "#999"
        st_label = "En attente" if status == "active" else "Notifié" if status == "notified" else status
        recent_rows += f"""<tr>
            <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-size:14px">{r.get('email','')}</td>
            <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-size:14px">{r.get('product_title','')}</td>
            <td style="padding:10px 16px;border-bottom:1px solid #f0ede8;font-size:13px;color:#888">{dt}</td>
            <td style="padding:10px 16px;border-bottom:1px solid #f0ede8"><span style="background:{st_color};color:#fff;font-size:11px;padding:3px 10px;border-radius:12px">{st_label}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PlanèteBeauty — Liste d'attente</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;background:#F9F7F4;color:#1A1A1A}}
  .header{{background:#1A1A1A;padding:24px 40px;display:flex;align-items:center;justify-content:space-between}}
  .header h1{{color:#D4AF37;font-size:18px;letter-spacing:2px;font-weight:700}}
  .header span{{color:#888;font-size:13px}}
  .container{{max-width:1000px;margin:0 auto;padding:30px 20px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:30px}}
  .card{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 2px 12px rgba(0,0,0,.04);text-align:center}}
  .card .num{{font-size:36px;font-weight:800;color:#D4AF37;line-height:1}}
  .card .label{{font-size:13px;color:#888;margin-top:6px}}
  .card.ab{{border:2px solid #D4AF37}}
  .section-title{{font-size:16px;font-weight:700;margin:28px 0 12px;padding-left:4px}}
  table{{width:100%;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.04);border-collapse:collapse}}
  th{{background:#F9F7F4;padding:12px 16px;text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#888;font-weight:600}}
  th:nth-child(2){{text-align:center}}
  .refresh{{display:inline-block;margin-top:20px;color:#D4AF37;font-size:13px;text-decoration:none}}
  .refresh:hover{{text-decoration:underline}}
</style>
</head>
<body>
<div class="header">
  <h1>PLANÈTE BEAUTY — LISTE D'ATTENTE</h1>
  <span>Mis à jour : {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC</span>
</div>
<div class="container">
  <div class="cards">
    <div class="card">
      <div class="num">{total}</div>
      <div class="label">Inscriptions actives</div>
    </div>
    <div class="card ab">
      <div class="num">{ab_total}</div>
      <div class="label">AB Signature Paris</div>
    </div>
    <div class="card">
      <div class="num">{len(products)}</div>
      <div class="label">Produits en attente</div>
    </div>
  </div>

  <div class="section-title">Par produit</div>
  <table>
    <tr><th>Produit</th><th>Inscrits</th><th>Dernière inscription</th></tr>
    {product_rows if product_rows else '<tr><td colspan="3" style="padding:20px;text-align:center;color:#888">Aucune inscription pour le moment</td></tr>'}
  </table>

  <div class="section-title">Inscriptions récentes</div>
  <table>
    <tr><th>Email</th><th>Produit</th><th>Date</th><th>Statut</th></tr>
    {recent_rows if recent_rows else '<tr><td colspan="4" style="padding:20px;text-align:center;color:#888">Aucune inscription</td></tr>'}
  </table>

  <a href="/bis?token={BIS_DASHBOARD_TOKEN}" class="refresh">↻ Rafraîchir</a>
</div>
</body>
</html>"""
    return HTMLResponse(html)


# ══════════════════════ BIS EMAIL ══════════════════════

def bis_email_html(product_title: str, product_handle: str) -> str:
    """Generate branded HTML email for back-in-stock notification."""
    product_url = f"https://planetebeauty.com/products/{product_handle}"
    return f"""<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#F9F7F4;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F9F7F4;padding:30px 0">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:12px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.06)">
  <tr><td style="background:#1A1A1A;padding:28px 40px;text-align:center">
    <span style="font-size:22px;font-weight:700;letter-spacing:3px;color:#D4AF37">PLANÈTE BEAUTY</span>
  </td></tr>
  <tr><td style="padding:40px">
    <h1 style="margin:0 0 8px;font-size:20px;color:#1A1A1A;font-weight:600">Bonne nouvelle !</h1>
    <p style="margin:0 0 24px;font-size:15px;color:#555;line-height:1.6">
      Le produit que vous attendiez est de nouveau disponible :
    </p>
    <div style="background:#F9F7F4;border-radius:10px;padding:20px 24px;margin-bottom:28px;border-left:4px solid #D4AF37">
      <p style="margin:0;font-size:17px;font-weight:600;color:#1A1A1A">{product_title}</p>
    </div>
    <table width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
      <a href="{product_url}" style="display:inline-block;background:#D4AF37;color:#FFFFFF;text-decoration:none;padding:14px 40px;border-radius:8px;font-size:15px;font-weight:600;letter-spacing:.5px">
        Découvrir maintenant
      </a>
    </td></tr></table>
    <p style="margin:28px 0 0;font-size:13px;color:#999;line-height:1.5;text-align:center">
      Les stocks sont limités — ne tardez pas !<br>
      Livraison offerte dès 99€ · Try&amp;Buy · Cashback 5%
    </p>
  </td></tr>
  <tr><td style="background:#F5F3EF;padding:20px 40px;text-align:center">
    <p style="margin:0;font-size:12px;color:#999">
      Vous recevez cet email car vous avez demandé à être averti(e) de la disponibilité de ce produit sur
      <a href="https://planetebeauty.com" style="color:#D4AF37;text-decoration:none">planetebeauty.com</a>.
    </p>
  </td></tr>
</table>
</td></tr></table>
</body></html>"""


async def send_bis_email(to_email: str, product_title: str, product_handle: str) -> bool:
    """Send a back-in-stock notification email via SMTP."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_FROM_EMAIL:
        logger.warning("BIS email: SMTP not configured, skipping email send")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"{product_title} — De nouveau disponible !"

    text_body = f"""Bonne nouvelle !

Le produit que vous attendiez est de nouveau disponible :
{product_title}

Découvrez-le sur https://planetebeauty.com/products/{product_handle}

Les stocks sont limités — ne tardez pas !
Livraison offerte dès 99€ · Try&Buy · Cashback 5%

---
PlanèteBeauty — planetebeauty.com
"""
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(bis_email_html(product_title, product_handle), "html", "utf-8"))

    try:
        await aiosmtplib.send(
            msg,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USER,
            password=SMTP_PASS,
            use_tls=False,
            start_tls=True,
        )
        logger.info(f"BIS email sent to {to_email} for {product_title}")
        return True
    except Exception as e:
        logger.error(f"BIS email failed for {to_email}: {e}")
        return False


@app.get("/api/bis/smtp-status")
async def bis_smtp_status(request: Request):
    """Check SMTP configuration status (authenticated)."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")
    return {
        "configured": bool(SMTP_HOST and SMTP_USER and SMTP_FROM_EMAIL),
        "host": SMTP_HOST or "(not set)",
        "port": SMTP_PORT,
        "from": f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>" if SMTP_FROM_EMAIL else "(not set)",
    }


@app.post("/api/bis/webhook/inventory")
async def bis_webhook_inventory(request: Request):
    """Shopify webhook: inventory_levels/update — notify subscribers when back in stock."""
    body = await request.body()
    try:
        payload = json.loads(body)
    except:
        raise HTTPException(400, "Invalid JSON")

    available = payload.get("available")
    inventory_item_id = payload.get("inventory_item_id")

    logger.info(f"BIS webhook: inventory_item_id={inventory_item_id}, available={available}")

    # Only process if item is now available (stock > 0)
    if not available or available <= 0:
        return {"ok": True, "action": "skipped", "reason": "still_out_of_stock"}

    if not SHOPIFY_ACCESS_TOKEN:
        return {"ok": False, "reason": "no_shopify_token"}

    try:
        # 1. Find variant by inventory_item_id
        async with httpx.AsyncClient(timeout=15) as client:
            query = json.dumps({"query": """
                query($id: ID!) {
                    inventoryItem(id: $id) {
                        variant {
                            id
                            legacyResourceId
                            product {
                                id
                                legacyResourceId
                                title
                                handle
                            }
                        }
                    }
                }
            """, "variables": {"id": f"gid://shopify/InventoryItem/{inventory_item_id}"}})

            r = await client.post(
                SHOPIFY_GRAPHQL_URL,
                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                content=query
            )
            gql_data = r.json()

        inv_item = gql_data.get("data", {}).get("inventoryItem")
        if not inv_item or not inv_item.get("variant"):
            return {"ok": True, "action": "skipped", "reason": "no_variant_found"}

        variant = inv_item["variant"]
        product = variant.get("product", {})
        variant_id = int(variant.get("legacyResourceId", 0))
        product_id = int(product.get("legacyResourceId", 0))
        product_handle = product.get("handle", "")
        product_title = product.get("title", "")

        logger.info(f"BIS webhook: product={product_title} (handle={product_handle}), variant_id={variant_id}, available={available}")

        # 2. Find active subscribers for this product
        db = get_db()
        if not db:
            return {"ok": False, "reason": "db_unavailable"}

        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, email FROM bis_subscriptions
            WHERE product_id = %s AND status = 'active'
        """, (product_id,))
        subscribers = cur.fetchall()

        if not subscribers:
            cur.close(); db.close()
            return {"ok": True, "action": "skipped", "reason": "no_subscribers", "product": product_title}

        # 3. Send notification emails + mark as notified
        sub_ids = [s["id"] for s in subscribers]
        notified_emails = [s["email"] for s in subscribers]
        emails_sent = 0
        for s in subscribers:
            if await send_bis_email(s["email"], product_title, product_handle):
                emails_sent += 1

        cur.execute("""
            UPDATE bis_subscriptions SET status = 'notified', notified_at = NOW()
            WHERE id = ANY(%s)
        """, (sub_ids,))
        db.commit()
        cur.close(); db.close()

        logger.info(f"BIS webhook: {emails_sent}/{len(subscribers)} emails sent for {product_title}")

        # 4. Remove bis-{handle} tag from Shopify customers
        tag_to_remove = f"bis-{product_handle}"
        async with httpx.AsyncClient(timeout=15) as client:
            for email in notified_emails:
                try:
                    search_q = json.dumps({"query": f'{{ customers(first:1, query:"email:{email}") {{ edges {{ node {{ id tags }} }} }} }}'})
                    sr = await client.post(
                        SHOPIFY_GRAPHQL_URL,
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                        content=search_q
                    )
                    edges = sr.json().get("data", {}).get("customers", {}).get("edges", [])
                    if edges:
                        cust_id = edges[0]["node"]["id"]
                        tags = [t for t in edges[0]["node"].get("tags", []) if t != tag_to_remove]
                        if "bis-notify" in tags and not any(t.startswith("bis-") and t != "bis-notify" for t in tags):
                            tags = [t for t in tags if t != "bis-notify"]
                        tags_json = json.dumps(tags)
                        mut = json.dumps({"query": f'mutation {{ customerUpdate(input: {{id: "{cust_id}", tags: {tags_json}}}) {{ customer {{ id }} }} }}'})
                        await client.post(
                            SHOPIFY_GRAPHQL_URL,
                            headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                            content=mut
                        )
                except Exception as e:
                    logger.warning(f"BIS remove tag for {email}: {e}")

        logger.info(f"BIS webhook: notified {len(subscribers)} subscribers for {product_title}, {emails_sent} emails sent, tags removed")

        return {
            "ok": True,
            "action": "notified",
            "product": product_title,
            "subscribers_notified": len(subscribers),
            "emails_sent": emails_sent,
            "emails": notified_emails
        }

    except Exception as e:
        logger.error(f"BIS webhook error: {e}")
        return {"ok": False, "error": str(e)}


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
    return {"status": "ok" if (redis_ok or db_ok) else "degraded", "version": APP_VERSION, "redis": redis_ok, "database": db_ok, "qdrant": qdrant_ok, "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token", "dev_mode": DEV_MODE, "llm": "mistral_api" if qdrant_ok else "unknown"}

# ══════════════════════ FRONTEND ══════════════════════
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """App home — BIS Dashboard embedded in Shopify Admin via App Bridge."""
    api_key = SHOPIFY_V8_CLIENT_ID or SHOPIFY_API_KEY
    host_param = request.query_params.get("host", "")

    # Fetch BIS data
    total = 0
    products = []
    recent = []
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COUNT(*) as total FROM bis_subscriptions WHERE status='active'")
            total = cur.fetchone()["total"]
            cur.execute("""
                SELECT product_id, product_title, product_handle, COUNT(*) as sub_count,
                       MAX(subscribed_at) as last_sub
                FROM bis_subscriptions WHERE status='active'
                GROUP BY product_id, product_title, product_handle
                ORDER BY sub_count DESC
            """)
            products = cur.fetchall()
            cur.execute("""
                SELECT email, product_title, product_handle, subscribed_at, status
                FROM bis_subscriptions
                ORDER BY subscribed_at DESC LIMIT 50
            """)
            recent = cur.fetchall()
            cur.close()
            db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.error(f"App home BIS: {e}")

    ab_products = [p for p in products if "AB Signature" in (p.get("product_title") or "")]
    ab_total = sum(p["sub_count"] for p in ab_products)

    product_rows = ""
    for p in products:
        is_ab = "AB Signature" in (p.get("product_title") or "")
        badge = ' <span class="badge-ab">AB Signature</span>' if is_ab else ""
        last = str(p.get("last_sub", ""))[:16].replace("T", " ")
        handle = p.get("product_handle", "")
        product_rows += f"""<tr>
            <td class="td-product"><a href="https://planetebeauty.com/products/{handle}" target="_top">{p['product_title']}</a>{badge}</td>
            <td class="td-count">{p['sub_count']}</td>
            <td class="td-date">{last}</td>
        </tr>"""

    recent_rows = ""
    for r in recent:
        dt = str(r.get("subscribed_at", ""))[:16].replace("T", " ")
        status = r.get("status", "active")
        st_color = "#4CAF50" if status == "active" else "#FF9800" if status == "notified" else "#999"
        st_label = "En attente" if status == "active" else "Notifi\u00e9" if status == "notified" else status
        recent_rows += f"""<tr>
            <td class="td-email">{r.get('email','')}</td>
            <td class="td-product">{r.get('product_title','')}</td>
            <td class="td-date">{dt}</td>
            <td><span class="status-badge" style="background:{st_color}">{st_label}</span></td>
        </tr>"""

    now_str = datetime.utcnow().strftime('%d/%m/%Y %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>STELLA V8 — PlanèteBeauty</title>
<script src="https://cdn.shopify.com/shopifycloud/app-bridge.js?apiKey={api_key}"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#F9F7F4;color:#1A1A1A;padding:0}}
  .app-header{{background:#1A1A1A;padding:20px 32px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}}
  .app-header h1{{color:#D4AF37;font-size:20px;letter-spacing:2px;font-weight:800}}
  .app-header .subtitle{{color:#888;font-size:13px}}
  .app-header .refresh-btn{{background:#D4AF37;color:#fff;border:none;padding:8px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:.2s}}
  .app-header .refresh-btn:hover{{background:#b8862e}}
  .container{{max-width:1100px;margin:0 auto;padding:24px 20px}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
  .card{{background:#fff;border-radius:14px;padding:24px 20px;box-shadow:0 2px 12px rgba(0,0,0,.04);text-align:center;transition:.2s}}
  .card:hover{{box-shadow:0 4px 20px rgba(0,0,0,.08)}}
  .card .num{{font-size:40px;font-weight:800;color:#D4AF37;line-height:1}}
  .card .label{{font-size:13px;color:#888;margin-top:8px;font-weight:500}}
  .card.ab{{border:2px solid #D4AF37;background:linear-gradient(135deg,#FFFDF7,#FFF8E7)}}
  .section-title{{font-size:15px;font-weight:700;margin:24px 0 10px;padding-left:4px;display:flex;align-items:center;gap:8px}}
  .section-title .icon{{font-size:18px}}
  table{{width:100%;background:#fff;border-radius:14px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.04);border-collapse:collapse}}
  th{{background:#F9F7F4;padding:12px 16px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#888;font-weight:700}}
  td{{padding:12px 16px;border-bottom:1px solid #f5f2ed;font-size:14px}}
  .td-count{{text-align:center;font-weight:800;font-size:22px;color:#D4AF37}}
  .td-date{{color:#888;font-size:13px}}
  .td-email{{font-size:13px;color:#555}}
  .td-product a{{color:#1A1A1A;text-decoration:none;font-weight:500}}
  .td-product a:hover{{color:#D4AF37}}
  .badge-ab{{background:linear-gradient(135deg,#D4AF37,#B8862E);color:#fff;font-size:10px;padding:3px 8px;border-radius:6px;margin-left:8px;font-weight:700;letter-spacing:.5px;vertical-align:middle}}
  .status-badge{{color:#fff;font-size:11px;padding:4px 12px;border-radius:12px;font-weight:600}}
  .empty{{padding:24px;text-align:center;color:#aaa;font-size:14px}}
  .footer{{text-align:center;padding:20px;color:#bbb;font-size:12px}}
  @media(max-width:600px){{
    .cards{{grid-template-columns:1fr 1fr}}
    .card .num{{font-size:32px}}
    td,th{{padding:8px 10px;font-size:12px}}
    .td-count{{font-size:18px}}
  }}
</style>
</head>
<body>
<div class="app-header">
  <div>
    <h1>STELLA V8</h1>
    <span class="subtitle">Liste d'attente &amp; notifications — {now_str} UTC</span>
  </div>
  <button class="refresh-btn" onclick="location.reload()">Rafra&icirc;chir</button>
</div>
<div class="container">
  <div class="cards">
    <div class="card">
      <div class="num">{total}</div>
      <div class="label">Inscriptions actives</div>
    </div>
    <div class="card ab">
      <div class="num">{ab_total}</div>
      <div class="label">AB Signature Paris</div>
    </div>
    <div class="card">
      <div class="num">{len(products)}</div>
      <div class="label">Produits en attente</div>
    </div>
    <div class="card">
      <div class="num">{len(recent)}</div>
      <div class="label">Total inscriptions</div>
    </div>
  </div>

  <div class="section-title"><span class="icon">&#128230;</span> Par produit</div>
  <table>
    <tr><th>Produit</th><th style="text-align:center">Inscrits</th><th>Derni&egrave;re inscription</th></tr>
    {product_rows if product_rows else '<tr><td colspan="3" class="empty">Aucune inscription pour le moment</td></tr>'}
  </table>

  <div class="section-title"><span class="icon">&#128236;</span> Inscriptions r&eacute;centes</div>
  <table>
    <tr><th>Email</th><th>Produit</th><th>Date</th><th>Statut</th></tr>
    {recent_rows if recent_rows else '<tr><td colspan="4" class="empty">Aucune inscription</td></tr>'}
  </table>
</div>
<div class="footer">STELLA V8 &middot; Plan&egrave;teBeauty &middot; Powered by Railway</div>
</body>
</html>"""
    return HTMLResponse(html)

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Legacy chat interface — moved from / to /chat."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path) as f: html = f.read()
    html = html.replace("__SHOPIFY_API_KEY__", SHOPIFY_V8_CLIENT_ID or SHOPIFY_API_KEY)
    html = html.replace("__SHOPIFY_SHOP__", SHOPIFY_SHOP)
    html = html.replace("__SHOPIFY_HOST__", request.query_params.get("host", ""))
    return HTMLResponse(html)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
