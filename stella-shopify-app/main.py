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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service.railway.internal:8080")
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

CREATE TABLE IF NOT EXISTS tryme_purchases (
    id SERIAL PRIMARY KEY,
    customer_email VARCHAR(255) NOT NULL,
    customer_id VARCHAR(255),
    product_id VARCHAR(255) NOT NULL,
    product_handle VARCHAR(255) NOT NULL,
    product_title VARCHAR(500),
    variant_id VARCHAR(255) NOT NULL,
    order_id VARCHAR(255) NOT NULL,
    order_name VARCHAR(50),
    tryme_price NUMERIC(10,2) DEFAULT 9.00,
    discount_code VARCHAR(50),
    discount_code_gid VARCHAR(255),
    discount_used BOOLEAN DEFAULT FALSE,
    full_size_order_id VARCHAR(255),
    purchased_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    discount_expires_at TIMESTAMP WITH TIME ZONE,
    discount_used_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_tryme_customer ON tryme_purchases(customer_email, product_id);
CREATE INDEX IF NOT EXISTS idx_tryme_code ON tryme_purchases(discount_code);
CREATE INDEX IF NOT EXISTS idx_tryme_status ON tryme_purchases(status, discount_expires_at);

CREATE TABLE IF NOT EXISTS cashback_settings (
    id SERIAL PRIMARY KEY,
    shop VARCHAR(255) DEFAULT 'planetemode.myshopify.com',
    cashback_rate DECIMAL(5,4) DEFAULT 0.05,
    expiry_days INTEGER DEFAULT 60,
    min_order_use DECIMAL(10,2) DEFAULT 70.00,
    exclude_shipping BOOLEAN DEFAULT TRUE,
    exclude_taxes BOOLEAN DEFAULT TRUE,
    exclude_discounts BOOLEAN DEFAULT TRUE,
    excluded_tags TEXT DEFAULT 'tryme,no-cashback',
    excluded_product_ids TEXT DEFAULT '',
    min_cashback_amount DECIMAL(10,2) DEFAULT 0.50,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS activity_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    type VARCHAR(50) NOT NULL,
    action VARCHAR(500) NOT NULL,
    details JSONB DEFAULT '{}',
    source VARCHAR(50) DEFAULT 'system',
    status VARCHAR(20) DEFAULT 'success',
    customer_email VARCHAR(200),
    order_name VARCHAR(50),
    product_title VARCHAR(500)
);
CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_log(type, timestamp DESC);

CREATE TABLE IF NOT EXISTS trustpilot_credits (
    id SERIAL PRIMARY KEY,
    order_number TEXT UNIQUE,
    review_id TEXT,
    reviewer_name TEXT,
    customer_id TEXT,
    customer_email TEXT,
    amount NUMERIC(10,2),
    credited_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS product_reviews (
    id SERIAL PRIMARY KEY,
    title TEXT,
    body TEXT,
    rating INTEGER DEFAULT 5,
    review_date TIMESTAMP DEFAULT NOW(),
    source VARCHAR(50) DEFAULT 'manual',
    curated VARCHAR(10) DEFAULT 'ok',
    reviewer_name TEXT,
    reviewer_email TEXT,
    product_id TEXT,
    product_handle TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reviews_handle ON product_reviews(product_handle);

CREATE TABLE IF NOT EXISTS cron_results (
    id SERIAL PRIMARY KEY,
    cron_name TEXT,
    result_data JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);
"""

def log_activity(type: str, action: str, details: dict = None, source: str = "system",
                  status: str = "success", customer_email: str = None,
                  order_name: str = None, product_title: str = None):
    """Log une action dans activity_log — tour de contrôle centralisée."""
    db = get_db()
    if not db: return
    try:
        cur = db.cursor()
        cur.execute("""INSERT INTO activity_log (type, action, details, source, status, customer_email, order_name, product_title)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (type, action, json.dumps(details or {}), source, status, customer_email, order_name, product_title))
        db.commit(); cur.close(); db.close()
    except Exception as e:
        logger.error(f"log_activity error: {e}")
        try: db.close()
        except: pass

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

scheduler = AsyncIOScheduler()

async def _run_cron(name, endpoint):
    """Internal: call our own cron endpoint."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"http://localhost:{os.getenv('PORT', '8000')}{endpoint}",
                                  headers={"X-Cron-Key": os.getenv("RAILWAY_CRON_KEY", "stella-internal")})
            logger.info(f"Scheduler {name}: {r.status_code}")
            status = "success" if r.status_code == 200 else "error"
            log_activity("cron_run", f"Cron {name} exécuté ({r.status_code})",
                         {"cron_name": name, "endpoint": endpoint, "status_code": r.status_code},
                         source="cron", status=status)
    except Exception as e:
        logger.error(f"Scheduler {name} error: {e}")
        log_activity("cron_run", f"Cron {name} ERREUR: {str(e)[:200]}",
                     {"cron_name": name, "endpoint": endpoint, "error": str(e)[:500]},
                     source="cron", status="error")

@app.on_event("startup")
async def startup():
    db = get_db()
    if db:
        try:
            cur = db.cursor(); cur.execute(CHAT_MIGRATION)
            cur.execute("""INSERT INTO cashback_settings (shop) SELECT 'planetemode.myshopify.com'
                           WHERE NOT EXISTS (SELECT 1 FROM cashback_settings WHERE shop = 'planetemode.myshopify.com')""")
            db.commit(); cur.close(); db.close()
            logger.info("Chat tables ready (cashback_settings seeded)")
        except Exception as e:
            logger.error(f"Migration: {e}")
            try: db.close()
            except: pass
    get_redis()
    load_shopify_token_from_db()

    # ══════ SCHEDULER 24/7 (tourne meme si le Mac est eteint) ══════
    scheduler.add_job(_run_cron, 'interval', hours=1, id='stock_check',
                      args=["stock-check", "/api/cron/stock-check"])
    scheduler.add_job(_run_cron, 'cron', hour=3, minute=15, id='sync_tags',
                      args=["sync-tags", "/api/cron/sync-tags"])
    scheduler.add_job(_run_cron, 'cron', hour=3, minute=45, id='nouveautes_expire',
                      args=["nouveautes-expire", "/api/cron/nouveautes-expire"])
    scheduler.add_job(_run_cron, 'cron', day_of_week='mon', hour=7, minute=0, id='audit_qualite',
                      args=["audit-qualite", "/api/cron/audit-qualite"])
    scheduler.add_job(_run_cron, 'cron', hour=4, minute=30, id='tryme_expire',
                      args=["tryme-expire", "/api/cron/tryme-expire"])
    scheduler.add_job(_run_cron, 'interval', hours=6, id='trustpilot_scan',
                      args=["trustpilot-scan", "/api/cron/trustpilot-scan"])
    scheduler.start()
    logger.info("Scheduler started: stock(1h), tags(3h15), nouveautes(3h45), audit(lun 7h), tryme(4h30), trustpilot(6h)")

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
    """Save to Qdrant via Embedding-Service directly (Context-Engine removed)."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            await c.post(f"{EMBEDDING_SERVICE_URL}/learn", json={"text": content, "project": "CHAT_HISTORY", "collection": "knowledge", "source": source})
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

    # Chat endpoint deprecated — use Claude Code instead
    result = {"answer": "Le chat web STELLA est désactivé. Utilisez Claude Code pour interagir avec STELLA."}

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
        # Always vectorize to Qdrant for semantic search
        await save_to_qdrant(f"[{mem.category}] {mem.title}: {mem.content}", source=f"claude_memory_{mid}")
        return {"status": "ok", "id": mid}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

@app.post("/memory/search")
async def search_memories(search: MemorySearch, request: Request):
    verify_claude_key(request)
    # Hybrid search: Qdrant semantic + PostgreSQL fallback
    results = []
    # 1) Qdrant semantic search via Embedding Service
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.post(f"{EMBEDDING_SERVICE_URL}/search", json={
                "query": search.query,
                "collection": "knowledge",
                "top_k": search.limit * 2,
                "rerank_top": search.limit
            })
            if resp.status_code == 200:
                qdrant_results = resp.json().get("results", [])
                for r in qdrant_results:
                    results.append({
                        "id": None,
                        "title": r.get("source", ""),
                        "content": r.get("text", ""),
                        "category": "qdrant",
                        "importance": round(r.get("score", 0) * 10),
                        "score": r.get("score", 0),
                        "created_at": None
                    })
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
    # 2) PostgreSQL text search as complement
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Split query into individual words for OR matching
            words = [w.strip() for w in search.query.split() if len(w.strip()) > 2]
            if words:
                conditions = " OR ".join(["(title ILIKE %s OR content ILIKE %s)"] * len(words))
                params = []
                for w in words:
                    params.extend([f"%{w}%", f"%{w}%"])
                if search.category:
                    sql = f"SELECT * FROM claude_memories WHERE category=%s AND ({conditions}) ORDER BY importance DESC LIMIT %s"
                    params = [search.category] + params + [search.limit]
                else:
                    sql = f"SELECT * FROM claude_memories WHERE ({conditions}) ORDER BY importance DESC LIMIT %s"
                    params.append(search.limit)
                cur.execute(sql, params)
            else:
                q = f"%{search.query}%"
                cur.execute("SELECT * FROM claude_memories WHERE title ILIKE %s OR content ILIKE %s ORDER BY importance DESC LIMIT %s",
                    (q, q, search.limit))
            pg_rows = cur.fetchall(); cur.close(); db.close()
            # Merge PG results (avoid duplicates by content)
            existing_texts = {r["content"][:100] for r in results}
            for r in pg_rows:
                snippet = (r.get("content") or "")[:100]
                if snippet not in existing_texts:
                    results.append({**r, "created_at": r["created_at"].isoformat() if r.get("created_at") else None})
                    existing_texts.add(snippet)
        except Exception as e:
            try: db.close()
            except: pass
            logger.warning(f"PG search failed: {e}")
    # Sort by importance/score desc, limit
    results.sort(key=lambda x: x.get("importance", 0) or 0, reverse=True)
    return {"results": results[:search.limit]}

@app.delete("/memory/cleanup")
async def cleanup_memories(request: Request):
    """Delete spam 'Session terminee' entries from PostgreSQL."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error", "message": "DB unavailable"}
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM claude_memories WHERE title LIKE 'Session terminee%'")
        deleted = cur.rowcount
        db.commit(); cur.close(); db.close()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int, request: Request):
    """Delete a specific memory by ID."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error", "message": "DB unavailable"}
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM claude_memories WHERE id = %s", (memory_id,))
        deleted = cur.rowcount
        db.commit(); cur.close(); db.close()
        return {"status": "ok", "deleted": deleted}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}

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

# ══════════════════════ TRY ME — TRY & BUY ══════════════════════

TRYME_PRICE = 9.00
TRYME_EXPIRY_DAYS = 30
TRYME_SKU_PREFIX = "TRYME-"
TRYME_VARIANT_TITLE = "2 ml"

def is_tryme_item(line_item):
    """Check if a line item is a Try Me product."""
    sku = (line_item.get("sku") or "").upper()
    title = (line_item.get("variant_title") or "").lower()
    name = (line_item.get("name") or "").lower()
    if sku.startswith(TRYME_SKU_PREFIX.upper()):
        return True
    if title == "2 ml" or "try me" in name or "try me" in title:
        return True
    return False

@app.post("/api/webhook/tryme-order")
async def webhook_tryme_order(request: Request):
    """Webhook ORDERS_PAID: detect Try Me purchases, create discount codes, send emails."""
    try:
        body = await request.body()
        payload = json.loads(body)
    except Exception:
        raise HTTPException(400, "Invalid payload")

    order_id = str(payload.get("id", ""))
    order_name = payload.get("name", "")
    customer = payload.get("customer", {})
    customer_email = (customer.get("email") or payload.get("email", "")).strip().lower()
    customer_id = str(customer.get("id", ""))
    line_items = payload.get("line_items", [])

    if not customer_email:
        return {"ok": True, "skipped": True, "reason": "no_email"}

    tryme_items = [li for li in line_items if is_tryme_item(li)]
    if not tryme_items:
        return {"ok": True, "skipped": True, "reason": "no_tryme_items"}

    results = []
    for item in tryme_items:
        product_id = str(item.get("product_id", ""))
        product_handle = item.get("handle", "") or ""
        product_title = item.get("title", "")
        variant_id = str(item.get("variant_id", ""))
        item_price = float(item.get("price", TRYME_PRICE))

        # Check if customer already has an active code for this product
        db_check = get_db()
        already_has_code = False
        if db_check:
            try:
                cur_c = db_check.cursor()
                cur_c.execute("""SELECT COUNT(*) FROM tryme_purchases
                    WHERE customer_email=%s AND product_id=%s AND status='pending'""",
                    (customer_email, product_id))
                already_has_code = cur_c.fetchone()[0] > 0
                cur_c.close(); db_check.close()
            except:
                try: db_check.close()
                except: pass

        if already_has_code:
            logger.info(f"Try Me: {customer_email} already has active code for product {product_id} — skipping")
            results.append({"product": product_title, "skipped": True, "reason": "duplicate"})
            continue

        # Generate unique discount code
        code = f"TRYME-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.utcnow()
        expires = now + __import__('datetime').timedelta(days=TRYME_EXPIRY_DAYS)

        # Create discount code via Shopify Admin API
        discount_gid = None
        try:
            gql_mutation = """mutation discountCodeBasicCreate($basicCodeDiscount: DiscountCodeBasicInput!) {
              discountCodeBasicCreate(basicCodeDiscount: $basicCodeDiscount) {
                codeDiscountNode { id codeDiscount { ... on DiscountCodeBasic { codes(first:1) { edges { node { code } } } } } }
                userErrors { field message }
              }
            }"""
            variables = {
                "basicCodeDiscount": {
                    "title": f"Try Me -{item_price:.0f}€ — {product_title}",
                    "code": code,
                    "startsAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "endsAt": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "usageLimit": 1,
                    "appliesOncePerCustomer": True,
                    "customerSelection": {
                        "customers": {
                            "add": [f"gid://shopify/Customer/{customer_id}"]
                        }
                    } if customer_id else {"all": True},
                    "customerGets": {
                        "items": {
                            "products": {
                                "productsToAdd": [f"gid://shopify/Product/{product_id}"]
                            }
                        },
                        "value": {
                            "discountAmount": {
                                "amount": str(item_price),
                                "appliesOnEachItem": False
                            }
                        }
                    },
                    "combinesWith": {
                        "shippingDiscounts": True,
                        "productDiscounts": False,
                        "orderDiscounts": True
                    }
                }
            }

            headers_gql = {
                "Content-Type": "application/json",
                "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(SHOPIFY_GRAPHQL_URL, json={"query": gql_mutation, "variables": variables}, headers=headers_gql)
                gql_data = resp.json()

            errors = gql_data.get("data", {}).get("discountCodeBasicCreate", {}).get("userErrors", [])
            if errors:
                logger.error(f"Try Me discount creation error: {errors}")
            else:
                discount_gid = gql_data.get("data", {}).get("discountCodeBasicCreate", {}).get("codeDiscountNode", {}).get("id", "")
                logger.info(f"Try Me discount created: {code} for {customer_email} on {product_title}")

        except Exception as e:
            logger.error(f"Try Me discount API error: {e}")

        # Save to PostgreSQL
        db = get_db()
        if db:
            try:
                cur = db.cursor()
                cur.execute("""INSERT INTO tryme_purchases
                    (customer_email, customer_id, product_id, product_handle, product_title, variant_id,
                     order_id, order_name, tryme_price, discount_code, discount_code_gid, discount_expires_at, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')""",
                    (customer_email, customer_id, product_id, product_handle, product_title, variant_id,
                     order_id, order_name, item_price, code, discount_gid,
                     expires.strftime("%Y-%m-%d %H:%M:%S")))
                db.commit()
                cur.close(); db.close()
            except Exception as e:
                logger.error(f"Try Me DB save: {e}")
                try: db.close()
                except: pass

        # Send email with discount code
        if SMTP_HOST and customer_email:
            try:
                html_body = f"""<div style="font-family:Georgia,serif;max-width:600px;margin:0 auto;padding:40px 20px">
                <h2 style="color:#1a1a1a;font-weight:300">Merci pour votre essai !</h2>
                <p style="color:#666;line-height:1.6">Vous avez choisi de découvrir <strong>{product_title}</strong> avec notre format Try Me.</p>
                <p style="color:#666;line-height:1.6">Si ce parfum vous séduit, utilisez le code ci-dessous pour déduire {item_price:.0f}€ de votre achat du format standard :</p>
                <div style="text-align:center;margin:30px 0;padding:20px;background:#FDFBF7;border-radius:12px;border:1px solid #E8E0D6">
                    <div style="font-size:28px;font-weight:700;letter-spacing:3px;color:#C4956A">{code}</div>
                    <div style="font-size:13px;color:#888;margin-top:8px">Valable 30 jours · Usage unique · Sur {product_title}</div>
                </div>
                <a href="https://planetebeauty.com/products/{product_handle}" style="display:block;text-align:center;padding:14px 28px;background:#1a1a1a;color:#fff;text-decoration:none;border-radius:8px;font-size:15px;margin:20px auto;max-width:280px">Découvrir le format complet</a>
                <p style="color:#aaa;font-size:12px;text-align:center;margin-top:30px">PlanèteBeauty · Parfumerie de niche</p>
                </div>"""

                msg = MIMEMultipart("alternative")
                msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
                msg["To"] = customer_email
                msg["Subject"] = f"Votre code Try Me -{item_price:.0f}€ — {product_title}"
                msg.attach(MIMEText(html_body, "html"))

                await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                                      username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
                logger.info(f"Try Me email sent to {customer_email}: {code}")
            except Exception as e:
                logger.error(f"Try Me email error: {e}")

        results.append({"product": product_title, "code": code, "email": customer_email, "discount_gid": discount_gid})

    return {"ok": True, "tryme_count": len(results), "results": results}


@app.post("/api/tryme/create-variants")
async def tryme_create_variants(request: Request):
    """Batch: add Try Me 2ml variant to all active perfume products."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    headers_gql = {"Content-Type": "application/json", "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
    created = []
    skipped = []
    errors_list = []
    cursor = None

    for page in range(20):
        after = f', after: "{cursor}"' if cursor else ''
        query = f"""{{ products(first: 50{after}, query: "status:ACTIVE AND (product_type:'Extrait de Parfum' OR product_type:'Eau de Parfum' OR product_type:'Eau de Parfum Intense' OR product_type:'Eau de Toilette')") {{
            edges {{ node {{ id handle title variants(first: 10) {{ edges {{ node {{ title sku }} }} }} }} }}
            pageInfo {{ hasNextPage endCursor }}
        }} }}"""

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(SHOPIFY_GRAPHQL_URL, json={"query": query}, headers=headers_gql)
            data = resp.json()

        products = data.get("data", {}).get("products", {}).get("edges", [])
        pi = data.get("data", {}).get("products", {}).get("pageInfo", {})

        for edge in products:
            p = edge["node"]
            existing_variants = [v["node"]["title"] for v in p.get("variants", {}).get("edges", [])]
            existing_skus = [v["node"].get("sku", "") for v in p.get("variants", {}).get("edges", [])]

            # Skip if already has Try Me variant
            if TRYME_VARIANT_TITLE in existing_variants or any(s and s.startswith(TRYME_SKU_PREFIX) for s in existing_skus):
                skipped.append(p["handle"])
                continue

            # Create Try Me variant
            sku = f"{TRYME_SKU_PREFIX}{p['handle'][:30]}"
            mutation = """mutation productVariantsBulkCreate($productId: ID!, $variants: [ProductVariantsBulkInput!]!) {
              productVariantsBulkCreate(productId: $productId, variants: $variants) {
                productVariants { id title }
                userErrors { field message }
              }
            }"""
            variables = {
                "productId": p["id"],
                "variants": [{
                    "optionValues": [{"optionName": "Taille", "name": TRYME_VARIANT_TITLE}],
                    "price": str(TRYME_PRICE),
                    "sku": sku,
                    "inventoryQuantities": [{"availableQuantity": 10, "locationId": "gid://shopify/Location/56855527557"}],
                    "taxable": True
                }]
            }

            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": variables}, headers=headers_gql)
                    result = resp.json()
                errs = result.get("data", {}).get("productVariantsBulkCreate", {}).get("userErrors", [])
                if errs:
                    errors_list.append({"handle": p["handle"], "errors": errs})
                else:
                    created.append(p["handle"])
            except Exception as e:
                errors_list.append({"handle": p["handle"], "error": str(e)})

        if not pi.get("hasNextPage"):
            break
        cursor = pi.get("endCursor")

    logger.info(f"Try Me variants: {len(created)} created, {len(skipped)} skipped, {len(errors_list)} errors")
    return {"created": len(created), "skipped": len(skipped), "errors": len(errors_list),
            "created_handles": created[:20], "error_details": errors_list[:10]}


@app.post("/api/tryme/remove-variants")
async def tryme_remove_variants(request: Request):
    """Rollback: remove all Try Me 2ml variants."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    headers_gql = {"Content-Type": "application/json", "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
    removed = 0
    cursor = None

    for page in range(20):
        after = f', after: "{cursor}"' if cursor else ''
        query = f"""{{ productVariants(first: 50{after}, query: "sku:{TRYME_SKU_PREFIX}*") {{
            edges {{ node {{ id title sku product {{ id handle }} }} }}
            pageInfo {{ hasNextPage endCursor }}
        }} }}"""

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(SHOPIFY_GRAPHQL_URL, json={"query": query}, headers=headers_gql)
            data = resp.json()

        variants = data.get("data", {}).get("productVariants", {}).get("edges", [])
        pi = data.get("data", {}).get("productVariants", {}).get("pageInfo", {})

        for edge in variants:
            v = edge["node"]
            product_id = v.get("product", {}).get("id", "")
            mutation = """mutation productVariantsBulkDelete($productId: ID!, $variantsIds: [ID!]!) {
              productVariantsBulkDelete(productId: $productId, variantsIds: $variantsIds) { userErrors { message } }
            }"""
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    await client.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": {"productId": product_id, "variantsIds": [v["id"]]}}, headers=headers_gql)
                removed += 1
            except:
                pass

        if not pi.get("hasNextPage"):
            break
        cursor = pi.get("endCursor")

    return {"removed": removed}


@app.post("/api/cron/tryme-expire")
async def cron_tryme_expire():
    """Clean up expired Try Me discount codes."""
    db = get_db()
    if not db:
        return {"expired": 0}

    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""SELECT id, discount_code, discount_code_gid FROM tryme_purchases
                       WHERE status='pending' AND discount_expires_at < NOW()""")
        expired = cur.fetchall()

        headers_gql = {"Content-Type": "application/json", "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}

        for row in expired:
            # Delete discount from Shopify
            if row.get("discount_code_gid"):
                try:
                    mutation = """mutation discountCodeDelete($id: ID!) {
                      discountCodeDelete(id: $id) { userErrors { message } }
                    }"""
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": {"id": row["discount_code_gid"]}}, headers=headers_gql)
                except:
                    pass

            cur.execute("UPDATE tryme_purchases SET status='expired' WHERE id=%s", (row["id"],))

        db.commit()
        cur.close(); db.close()
        logger.info(f"Try Me expire: {len(expired)} codes expired")
        return {"expired": len(expired)}
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"Try Me expire error: {e}")
        return {"error": str(e)}


@app.get("/api/tryme/dashboard")
async def tryme_dashboard(request: Request):
    """Dashboard data for Try Me tab."""
    db = get_db()
    if not db:
        return {"total": 0, "pending": 0, "used": 0, "expired": 0, "recent": []}

    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) as c FROM tryme_purchases")
        total = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='pending'")
        pending = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE discount_used=true")
        used = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='expired'")
        expired = cur.fetchone()["c"]

        cur.execute("""SELECT customer_email, product_title, discount_code, status,
                       purchased_at, discount_expires_at, discount_used
                       FROM tryme_purchases ORDER BY purchased_at DESC LIMIT 20""")
        recent = cur.fetchall()
        for r in recent:
            if r.get("purchased_at"): r["purchased_at"] = r["purchased_at"].isoformat()
            if r.get("discount_expires_at"): r["discount_expires_at"] = r["discount_expires_at"].isoformat()

        conv_rate = round(used / total * 100, 1) if total > 0 else 0

        cur.close(); db.close()
        return {"total": total, "pending": pending, "used": used, "expired": expired,
                "conversion_rate": conv_rate, "recent": recent}
    except Exception as e:
        try: db.close()
        except: pass
        return {"total": 0, "error": str(e)}


# ══════════════════════ BACK IN STOCK (BIS) ══════════════════════

class BISSubscribeRequest(BaseModel):
    email: str
    product_id: int
    variant_id: int
    product_title: str = ""
    product_handle: str = ""

# ══════════════════════ CRON: AUTO-TAG SYNC ══════════════════════

FAMILLE_MAP = {
    "oriental": "Oriental", "ambré": "Oriental", "oriental ambré": "Oriental",
    "oriental boisé": "Oriental", "oriental floral": "Oriental", "oriental fougère": "Oriental",
    "oriental fruité": "Oriental", "oriental gourmand": "Oriental", "oriental vanillé": "Oriental",
    "oriental épicé": "Oriental", "orientale ambrée": "Oriental",
    "floral": "Floral", "floral ambré": "Floral", "floral fruité": "Floral",
    "floral fruité gourmand": "Floral", "floral vert": "Floral", "florale ambrée": "Floral",
    "florale": "Floral", "florale fruitée": "Floral",
    "boisé": "Boisé", "boisé aromatique": "Boisé", "boisé aquatique": "Boisé",
    "boisé cuiré": "Boisé", "boisé épicé": "Boisé", "boisée": "Boisé", "boisée chyprée": "Boisé",
    "gourmand": "Gourmand", "gourmande": "Gourmand", "gourmande boisée": "Gourmand", "gourmande fruitée": "Gourmand",
    "fruité": "Fruité", "fruité aromatique": "Fruité", "fruité gourmand": "Fruité",
    "fruité aquatique": "Fruité", "fruitée gourmande": "Fruité",
    "hespéridé": "Hespéridé", "hespéridé aromatique": "Hespéridé",
    "hespéridée gourmande": "Hespéridé", "hespéridé fruité": "Hespéridé", "hespéridé épicé": "Hespéridé",
    "cuir": "Cuiré", "cuiré": "Cuiré",
    "chypré": "Chypré", "chyprée": "Chypré", "chyprée florale ambrée": "Chypré",
    "aquatique": "Aquatique", "marin": "Aquatique", "marin ambré épicé": "Aquatique",
    "aromatique": "Aromatique", "aromatique fruité": "Aromatique", "aromatique épicée": "Aromatique",
    "fougère": "Aromatique", "aldéhydé": "Aldéhydé", "aldéhydée": "Aldéhydé",
    "musqué": "Musqué", "poudré": "Gourmand",
}

TOP_OCCASIONS = {"Quotidien", "Soirée", "Bureau", "Rendez-vous", "Occasion spéciale",
                  "Vacances", "Plein air", "Cocooning", "Soirée élégante", "Casual"}

def compute_auto_tags(product_data, cutoff_iso):
    """Compute all auto-tags from metafields + creation date."""
    new_tags = set()

    # Famille
    fam = product_data.get("famille")
    if fam and fam.get("value"):
        val = fam["value"]
        if val.startswith("["):
            try: val = json.loads(val)[0]
            except: pass
        mapped = FAMILLE_MAP.get(val.strip().lower())
        if not mapped:
            for k, v in FAMILLE_MAP.items():
                if k in val.strip().lower(): mapped = v; break
        if mapped: new_tags.add(f"Famille:{mapped}")

    # Saison
    sai = product_data.get("saison")
    if sai and sai.get("value"):
        val = sai["value"]
        items = json.loads(val) if val.startswith("[") else [val]
        for s in items:
            for part in s.split("/"):
                part = part.strip()
                if part in ("Printemps", "Été", "Automne", "Hiver"):
                    new_tags.add(f"Saison:{part}")

    # Genre
    gen = product_data.get("genre")
    if gen and gen.get("value"):
        val = gen["value"].strip()
        if val in ("Mixte", "Unisexe"): new_tags.add("Genre:Mixte")
        elif val == "Féminin": new_tags.add("Genre:Féminin")
        elif val == "Masculin": new_tags.add("Genre:Masculin")

    # Concentration
    conc = product_data.get("concentration")
    if conc and conc.get("value"):
        val = conc["value"].strip().lower()
        if "extrait" in val: new_tags.add("Concentration:Extrait")
        elif "intense" in val: new_tags.add("Concentration:EDP Intense")
        elif "parfum" in val and "eau" in val: new_tags.add("Concentration:EDP")
        elif "toilette" in val: new_tags.add("Concentration:EDT")

    # Occasions
    occ = product_data.get("occasions")
    if occ and occ.get("value"):
        val = occ["value"]
        items = json.loads(val) if val.startswith("[") else [x.strip() for x in val.split(",")]
        for o in items:
            if o.strip() in TOP_OCCASIONS: new_tags.add(f"Occasion:{o.strip()}")

    # Nouveauté (< 30 days)
    if product_data.get("createdAt", "") > cutoff_iso:
        new_tags.add("Nouveauté")

    return new_tags


@app.post("/api/cron/sync-tags")
async def cron_sync_tags(request: Request):
    """Daily cron: sync ALL auto-tags (famille, saison, genre, occasion, concentration, nouveauté)."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    from datetime import timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

    # Auto-tag prefixes we manage
    AUTO_PREFIXES = ("Famille:", "Saison:", "Genre:", "Concentration:", "Occasion:", "Nouveauté")

    updated, skipped = 0, 0
    cursor = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            after = f', after: "{cursor}"' if cursor else ''
            r = await client.post(gql_url, json={"query": f"""{{ products(first: 50{after}) {{ edges {{ cursor node {{
                id tags createdAt
                famille: metafield(namespace: "parfum", key: "famille_olfactive") {{ value }}
                saison: metafield(namespace: "parfum", key: "saison") {{ value }}
                genre: metafield(namespace: "parfum", key: "genre") {{ value }}
                concentration: metafield(namespace: "parfum", key: "concentration") {{ value }}
                occasions: metafield(namespace: "parfum", key: "occasions") {{ value }}
            }} }} pageInfo {{ hasNextPage }} }} }}"""}, headers=headers)
            data = r.json().get("data", {}).get("products", {})
            for edge in data.get("edges", []):
                p = edge["node"]
                cursor = edge["cursor"]
                existing = set(p.get("tags", []))

                # Remove old auto-tags
                manual_tags = {t for t in existing if not any(t.startswith(px) or t == px for px in AUTO_PREFIXES)}
                # Compute new auto-tags
                auto_tags = compute_auto_tags(p, cutoff)
                # Merge
                final_tags = manual_tags | auto_tags

                if final_tags != existing:
                    await client.post(gql_url, json={"query": f'mutation {{ productUpdate(input: {{id: "{p["id"]}", tags: {json.dumps(sorted(list(final_tags)))}}}) {{ product {{ id }} userErrors {{ message }} }} }}'}, headers=headers)
                    updated += 1
                else:
                    skipped += 1

            if not data.get("pageInfo", {}).get("hasNextPage"):
                break

    logger.info(f"Cron sync-tags: {updated} updated, {skipped} unchanged")
    return {"updated": updated, "skipped": skipped, "cutoff": cutoff}


# Keep old endpoint as alias
@app.post("/api/cron/nouveautes")
async def cron_nouveautes(request: Request):
    """Alias for sync-tags."""
    return await cron_sync_tags(request)


# ══════════════════════ CRON: AUDIT QUALITÉ ══════════════════════

@app.post("/api/cron/audit-qualite")
async def cron_audit_qualite(request: Request):
    """Weekly audit: find products with missing SEO or key metafields."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

    issues = []
    cursor = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            after = f', after: "{cursor}"' if cursor else ''
            r = await client.post(gql_url, json={"query": f"""{{ products(first: 50{after}) {{ edges {{ cursor node {{
                id title handle vendor status
                seo {{ title description }}
                parfumeur: metafield(namespace: "parfum", key: "parfumeur") {{ value }}
                famille: metafield(namespace: "parfum", key: "famille_olfactive") {{ value }}
                accord: metafield(namespace: "parfum", key: "accord_principal") {{ value }}
            }} }} pageInfo {{ hasNextPage }} }} }}"""}, headers=headers)
            data = r.json().get("data", {}).get("products", {})
            for edge in data.get("edges", []):
                p = edge["node"]
                cursor = edge["cursor"]
                if p.get("status") != "ACTIVE":
                    continue
                product_issues = []
                seo = p.get("seo") or {}
                if not seo.get("title"):
                    product_issues.append("SEO title manquant")
                if not seo.get("description"):
                    product_issues.append("SEO description manquante")
                if not p.get("parfumeur", {}) or not (p.get("parfumeur") or {}).get("value"):
                    product_issues.append("Parfumeur vide")
                if not p.get("famille", {}) or not (p.get("famille") or {}).get("value"):
                    product_issues.append("Famille olfactive vide")
                if not p.get("accord", {}) or not (p.get("accord") or {}).get("value"):
                    product_issues.append("Accord principal vide")
                if product_issues:
                    issues.append({
                        "title": p.get("title", ""),
                        "handle": p.get("handle", ""),
                        "vendor": p.get("vendor", ""),
                        "issues": product_issues
                    })
            if not data.get("pageInfo", {}).get("hasNextPage"):
                break

    # Save result to DB for dashboard display
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS cron_results (
                id SERIAL PRIMARY KEY, cron_name TEXT, result_data JSONB,
                executed_at TIMESTAMP DEFAULT NOW())""")
            cur.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s, %s)",
                ("audit-qualite", json.dumps({"issues": issues, "total_issues": len(issues)})))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.error(f"Audit save: {e}")

    # Send email report if issues found
    if issues and SMTP_HOST:
        rows_html = ""
        for p in issues[:50]:
            rows_html += f"<tr><td>{p['vendor']}</td><td>{p['title']}</td><td>{'<br>'.join(p['issues'])}</td></tr>"
        body = f"""<h2>Audit Qualité PlanèteBeauty</h2>
        <p><strong>{len(issues)} produit(s)</strong> avec des données manquantes.</p>
        <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;font-family:sans-serif;font-size:13px'>
        <tr style='background:#f5f2ed'><th>Marque</th><th>Produit</th><th>Problèmes</th></tr>
        {rows_html}</table>"""
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
            msg["To"] = "info@planetebeauty.com"
            msg["Subject"] = f"[STELLA] Audit Qualité — {len(issues)} produits à corriger"
            msg.attach(MIMEText(body, "html"))
            await aiosmtplib.send(msg, hostname=SMTP_HOST, port=int(SMTP_PORT or 587),
                username=SMTP_USER, password=SMTP_PASS, use_tls=True)
        except Exception as e:
            logger.error(f"Audit email: {e}")

    logger.info(f"Audit qualité: {len(issues)} issues found")
    return {"status": "ok", "issues_count": len(issues), "issues": issues[:20]}


# ══════════════════════ CRON: NOUVEAUTÉS AUTO-EXPIRE ══════════════════════

@app.post("/api/cron/nouveautes-expire")
async def cron_nouveautes_expire(request: Request):
    """Daily: remove 'Nouveauté' tag from products created more than 30 days ago."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    from datetime import timezone, timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

    expired = []
    still_new = []
    cursor = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            after = f', after: "{cursor}"' if cursor else ''
            r = await client.post(gql_url, json={"query": f"""{{ products(first: 50, query: "tag:Nouveauté"{after}) {{ edges {{ cursor node {{
                id title handle createdAt tags
            }} }} pageInfo {{ hasNextPage }} }} }}"""}, headers=headers)
            data = r.json().get("data", {}).get("products", {})
            for edge in data.get("edges", []):
                p = edge["node"]
                cursor = edge["cursor"]
                if p.get("createdAt", "") < cutoff:
                    # Remove Nouveauté tag
                    new_tags = [t for t in p.get("tags", []) if t != "Nouveauté"]
                    await client.post(gql_url, json={"query": f'mutation {{ productUpdate(input: {{id: "{p["id"]}", tags: {json.dumps(new_tags)}}}) {{ product {{ id }} userErrors {{ message }} }} }}'}, headers=headers)
                    expired.append({"title": p["title"], "created": p["createdAt"][:10]})
                else:
                    days_left = 30 - (datetime.now(timezone.utc) - datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))).days
                    still_new.append({"title": p["title"], "created": p["createdAt"][:10], "days_left": max(0, days_left)})
            if not data.get("pageInfo", {}).get("hasNextPage"):
                break

    # Save result
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS cron_results (
                id SERIAL PRIMARY KEY, cron_name TEXT, result_data JSONB,
                executed_at TIMESTAMP DEFAULT NOW())""")
            cur.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s, %s)",
                ("nouveautes-expire", json.dumps({"expired": expired, "still_new": still_new})))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass

    logger.info(f"Nouveautés expire: {len(expired)} expired, {len(still_new)} still new")
    return {"status": "ok", "expired": len(expired), "still_new": len(still_new), "details": {"expired": expired, "still_new": still_new}}


# ══════════════════════ CRON: TRUSTPILOT SCAN ══════════════════════

@app.post("/api/cron/trustpilot-scan")
async def cron_trustpilot_scan(request: Request):
    """Scan Trustpilot reviews for order numbers and credit 5€ store credit."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    import re
    TRUSTPILOT_BIZ_ID = None  # Will be fetched dynamically
    results = {"scanned": 0, "credited": 0, "already_credited": 0, "no_match": 0, "errors": []}

    try:
        # 1. Fetch recent Trustpilot reviews via public API
        async with httpx.AsyncClient(timeout=15) as client:
            # First find business unit ID
            search_r = await client.get("https://www.trustpilot.com/api/categoriespages/planetebeauty.com")
            if search_r.status_code != 200:
                # Try alternative endpoint
                search_r = await client.get(f"https://api.trustpilot.com/v1/business-units/find?name=planetebeauty.com")

            # Use the public consumer API to get reviews
            reviews_r = await client.get(
                "https://www.trustpilot.com/_next/data/consumersitefront-consumersite-2/fr/review/planetebeauty.com.json",
                headers={"User-Agent": "Mozilla/5.0"}
            )

            reviews = []
            if reviews_r.status_code == 200:
                try:
                    data = reviews_r.json()
                    page_data = data.get("pageProps", {})
                    review_list = page_data.get("reviews", [])
                    for rv in review_list:
                        text = rv.get("text", "")
                        title = rv.get("title", "")
                        consumer = rv.get("consumer", {})
                        reviews.append({
                            "text": f"{title} {text}",
                            "name": consumer.get("displayName", ""),
                            "date": rv.get("createdAt", ""),
                            "id": rv.get("id", "")
                        })
                except:
                    pass

            if not reviews:
                # Fallback: scrape the HTML page
                page_r = await client.get("https://fr.trustpilot.com/review/planetebeauty.com",
                    headers={"User-Agent": "Mozilla/5.0"})
                if page_r.status_code == 200:
                    html = page_r.text
                    # Extract review texts
                    texts = re.findall(r'data-review-content[^>]*>(.*?)</p>', html, re.DOTALL)
                    titles = re.findall(r'data-review-title[^>]*>(.*?)</h2>', html, re.DOTALL)
                    for i, t in enumerate(texts):
                        title = titles[i] if i < len(titles) else ""
                        reviews.append({"text": f"{title} {t}", "name": "", "date": "", "id": str(i)})

            results["scanned"] = len(reviews)

            # 2. For each review, look for order number pattern #XXXXX
            db = get_db()
            if not db:
                return {"error": "No DB", **results}

            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""CREATE TABLE IF NOT EXISTS trustpilot_credits (
                id SERIAL PRIMARY KEY,
                order_number TEXT UNIQUE,
                review_id TEXT,
                reviewer_name TEXT,
                customer_id TEXT,
                amount NUMERIC(10,2),
                credited_at TIMESTAMP DEFAULT NOW()
            )""")

            gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
            headers_gql = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

            for rv in reviews:
                # Find order numbers like #20562 (# followed by 5 digits)
                order_nums = re.findall(r'#(\d{5})', rv["text"])
                if not order_nums:
                    results["no_match"] += 1
                    continue

                for order_num in order_nums:
                    order_name = f"#{order_num}"

                    # Check if already credited
                    cur.execute("SELECT id FROM trustpilot_credits WHERE order_number = %s", (order_name,))
                    if cur.fetchone():
                        results["already_credited"] += 1
                        continue

                    # Find order in Shopify
                    order_q = await client.post(gql_url, json={"query": f'{{ orders(first:1, query:"name:{order_name}") {{ edges {{ node {{ id name customer {{ id email }} }} }} }} }}'}, headers=headers_gql)
                    order_data = order_q.json().get("data", {}).get("orders", {}).get("edges", [])

                    if not order_data:
                        results["no_match"] += 1
                        continue

                    order = order_data[0]["node"]
                    customer = order.get("customer")
                    if not customer or not customer.get("id"):
                        results["no_match"] += 1
                        continue

                    customer_id = customer["id"]

                    # Credit 5€ store credit
                    credit_mutation = """mutation($id: ID!, $credit: StoreCreditAccountCreditInput!) {
                        storeCreditAccountCredit(id: $id, creditInput: $credit) {
                            storeCreditAccountTransaction { id }
                            userErrors { field message }
                        }
                    }"""
                    credit_r = await client.post(gql_url, json={
                        "query": credit_mutation,
                        "variables": {
                            "id": customer_id,
                            "credit": {"creditAmount": {"amount": "5.00", "currencyCode": "EUR"}}
                        }
                    }, headers=headers_gql)

                    credit_data = credit_r.json()
                    errors = credit_data.get("data", {}).get("storeCreditAccountCredit", {}).get("userErrors", [])

                    if not errors:
                        cur.execute("INSERT INTO trustpilot_credits (order_number, review_id, reviewer_name, customer_id, amount) VALUES (%s,%s,%s,%s,%s)",
                            (order_name, rv.get("id",""), rv.get("name",""), customer_id, 5.00))
                        results["credited"] += 1
                        log_activity("trustpilot_credit", f"Trustpilot: 5€ credite pour avis {order_name}",
                                     {"order_name": order_name, "reviewer": rv.get("name",""), "amount": 5.00},
                                     source="cron", customer_email=customer.get("email",""), order_name=order_name)

                        # Also add review to product_reviews for each product in the order
                        items_q = await client.post(gql_url, json={"query": f'{{ order(id: "{order["id"]}") {{ lineItems(first:10) {{ edges {{ node {{ title product {{ handle }} }} }} }} }} }}'}, headers=headers_gql)
                        items_data = items_q.json().get("data", {}).get("order", {}).get("lineItems", {}).get("edges", [])
                        for item in items_data:
                            p_handle = item.get("node", {}).get("product", {}).get("handle", "")
                            if p_handle:
                                try:
                                    review_title = rv.get("text", "")[:80] if rv.get("text") else "Avis Trustpilot"
                                    review_body = rv.get("text", "")
                                    cur.execute("""INSERT INTO product_reviews
                                        (title, body, rating, review_date, source, curated, reviewer_name, reviewer_email, product_id, product_handle)
                                        VALUES (%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s)""",
                                        (review_title, review_body, 5, "trustpilot", "ok", rv.get("name","Client Trustpilot"),
                                         customer.get("email",""), "", p_handle))
                                except Exception as e:
                                    logger.warning(f"Review insert for {p_handle}: {e}")
                    else:
                        results["errors"].append(f"{order_name}: {errors[0].get('message','')}")

            db.commit()
            cur.close()
            db.close()

    except Exception as e:
        logger.error(f"Trustpilot scan error: {e}")
        results["errors"].append(str(e))

    # Save result
    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor()
            cur2.execute("""CREATE TABLE IF NOT EXISTS cron_results (
                id SERIAL PRIMARY KEY, cron_name TEXT, result_data JSONB,
                executed_at TIMESTAMP DEFAULT NOW())""")
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s, %s)",
                ("trustpilot-scan", json.dumps(results)))
            db2.commit(); cur2.close(); db2.close()
        except:
            try: db2.close()
            except: pass

    return {"status": "ok", **results}


# ══════════════════════ CRON: STOCK CHECK ══════════════════════

@app.post("/api/cron/stock-check")
async def cron_stock_check(request: Request):
    """Hourly: check for out-of-stock products, email alert for new ones."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")

    gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

    oos_products = []
    cursor = None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            after = f', after: "{cursor}"' if cursor else ''
            r = await client.post(gql_url, json={"query": f"""{{ products(first: 50, query: "status:active"{after}) {{ edges {{ cursor node {{
                id title handle vendor totalInventory
                variants(first: 5) {{ edges {{ node {{ inventoryQuantity title }} }} }}
            }} }} pageInfo {{ hasNextPage }} }} }}"""}, headers=headers)
            data = r.json().get("data", {}).get("products", {})
            for edge in data.get("edges", []):
                p = edge["node"]
                cursor = edge["cursor"]
                total_inv = p.get("totalInventory", 0)
                if total_inv is not None and total_inv <= 0:
                    # Check BIS subscribers count
                    bis_count = 0
                    db = get_db()
                    if db:
                        try:
                            cur = db.cursor()
                            cur.execute("SELECT COUNT(*) FROM bis_subscriptions WHERE product_handle=%s AND status='active'",
                                (p.get("handle", ""),))
                            bis_count = cur.fetchone()[0]
                            cur.close(); db.close()
                        except:
                            try: db.close()
                            except: pass
                    oos_products.append({
                        "title": p["title"],
                        "handle": p.get("handle", ""),
                        "vendor": p.get("vendor", ""),
                        "total_inventory": total_inv or 0,
                        "bis_subscribers": bis_count
                    })
            if not data.get("pageInfo", {}).get("hasNextPage"):
                break

    # Get previous OOS list to detect NEW ruptures
    db = get_db()
    prev_oos_handles = set()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""CREATE TABLE IF NOT EXISTS cron_results (
                id SERIAL PRIMARY KEY, cron_name TEXT, result_data JSONB,
                executed_at TIMESTAMP DEFAULT NOW())""")
            cur.execute("SELECT result_data FROM cron_results WHERE cron_name='stock-check' ORDER BY executed_at DESC LIMIT 1")
            prev = cur.fetchone()
            if prev and prev.get("result_data"):
                prev_data = prev["result_data"] if isinstance(prev["result_data"], dict) else json.loads(prev["result_data"])
                prev_oos_handles = {p["handle"] for p in prev_data.get("oos_products", [])}
            cur.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s, %s)",
                ("stock-check", json.dumps({"oos_products": oos_products})))
            db.commit(); cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.error(f"Stock check DB: {e}")

    # Detect NEW out-of-stock (not in previous run)
    current_handles = {p["handle"] for p in oos_products}
    new_oos = [p for p in oos_products if p["handle"] not in prev_oos_handles]

    # Email alert for new OOS
    if new_oos and SMTP_HOST:
        rows = ""
        for p in new_oos:
            bis_badge = f' ({p["bis_subscribers"]} en attente)' if p["bis_subscribers"] > 0 else ""
            rows += f"<tr><td>{p['vendor']}</td><td>{p['title']}</td><td style='color:red;font-weight:bold'>0{bis_badge}</td></tr>"
        body = f"""<h2>🚨 Alerte Stock — PlanèteBeauty</h2>
        <p><strong>{len(new_oos)} nouveau(x) produit(s) en rupture !</strong></p>
        <table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;font-family:sans-serif;font-size:13px'>
        <tr style='background:#f5f2ed'><th>Marque</th><th>Produit</th><th>Stock</th></tr>
        {rows}</table>
        <p style='color:#888;font-size:12px;margin-top:20px'>Total produits en rupture : {len(oos_products)}</p>"""
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
            msg["To"] = "info@planetebeauty.com"
            msg["Subject"] = f"[STELLA] 🚨 {len(new_oos)} nouveau(x) produit(s) en rupture"
            msg.attach(MIMEText(body, "html"))
            await aiosmtplib.send(msg, hostname=SMTP_HOST, port=int(SMTP_PORT or 587),
                username=SMTP_USER, password=SMTP_PASS, use_tls=True)
        except Exception as e:
            logger.error(f"Stock alert email: {e}")

    logger.info(f"Stock check: {len(oos_products)} OOS, {len(new_oos)} new")
    return {"status": "ok", "total_oos": len(oos_products), "new_oos": len(new_oos), "oos_products": oos_products[:30]}


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

# ══════════════════════ QUIZ OLFACTIF ANALYTICS ══════════════════════

@app.post("/api/quiz/track")
async def quiz_track(request: Request):
    """Track quiz events (view, start, complete, atc). Stored in Redis."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    event = body.get("event", "unknown")
    rc = get_redis()
    if not rc:
        return {"ok": True, "stored": False}
    now = int(time.time())
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Increment daily counter per event type
    key = f"quiz:{event}:{today}"
    pipe = rc.pipeline()
    pipe.incr(key)
    pipe.expire(key, 90 * 86400)  # Keep 90 days
    # Also increment total counter
    pipe.incr(f"quiz:{event}:total")
    results = pipe.execute()
    return {"ok": True, "event": event, "count_today": results[0]}

@app.get("/api/quiz/stats")
async def quiz_stats():
    """Get quiz analytics: views, completions, conversion rate."""
    rc = get_redis()
    if not rc:
        return {"error": "redis_unavailable"}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Get today's stats
    views_today = int(rc.get(f"quiz:quiz_view:{today}") or 0)
    starts_today = int(rc.get(f"quiz:quiz_start:{today}") or 0)
    completes_today = int(rc.get(f"quiz:quiz_complete:{today}") or 0)
    atc_today = int(rc.get(f"quiz:quiz_atc:{today}") or 0)
    # Get totals
    views_total = int(rc.get("quiz:quiz_view:total") or 0)
    starts_total = int(rc.get("quiz:quiz_start:total") or 0)
    completes_total = int(rc.get("quiz:quiz_complete:total") or 0)
    atc_total = int(rc.get("quiz:quiz_atc:total") or 0)
    # Conversion rates
    conv_rate = round(completes_total / views_total * 100, 1) if views_total > 0 else 0
    atc_rate = round(atc_total / completes_total * 100, 1) if completes_total > 0 else 0
    # Last 7 days
    daily = []
    for i in range(7):
        from datetime import timedelta
        d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        daily.append({
            "date": d,
            "views": int(rc.get(f"quiz:quiz_view:{d}") or 0),
            "completes": int(rc.get(f"quiz:quiz_complete:{d}") or 0),
            "atc": int(rc.get(f"quiz:quiz_atc:{d}") or 0)
        })
    return {
        "today": {"views": views_today, "starts": starts_today, "completes": completes_today, "atc": atc_today},
        "total": {"views": views_total, "starts": starts_total, "completes": completes_total, "atc": atc_total},
        "conversion_rate": conv_rate,
        "atc_rate": atc_rate,
        "daily": daily
    }

@app.post("/api/quiz/regenerate")
async def quiz_regenerate():
    """Regenerate quiz-data.json from Shopify products and upload to theme."""
    if not SHOPIFY_ACCESS_TOKEN:
        return {"success": False, "error": "No Shopify token"}
    try:
        all_products = []
        cursor = None
        headers_gql = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN
        }
        for page in range(10):
            after = f', after: "{cursor}"' if cursor else ''
            query = '''{
              products(first: 50%s, query: "product_type:'Extrait de Parfum' OR product_type:'Eau de Parfum' OR product_type:'Eau de Parfum Intense' OR product_type:'Eau de Toilette'") {
                edges { node {
                  handle title vendor productType
                  priceRangeV2 { minVariantPrice { amount } }
                  metafields(first: 20, keys: ["parfum.genre","parfum.famille_olfactive","parfum.accord_principal","parfum.accords_secondaires","parfum.saison","parfum.occasions","parfum.moment","parfum.intensite","parfum.sillage_level","parfum.note_tete_principale","parfum.note_coeur_principale","parfum.note_fond_principale","parfum.notes_cles","parfum.concentration","parfum.contenance_ml"]) {
                    edges { node { key value } }
                  }
                  featuredImage { url }
                  variants(first: 3) { edges { node { id title price availableForSale } } }
                } }
                pageInfo { hasNextPage endCursor }
              }
            }''' % after
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(SHOPIFY_GRAPHQL_URL, json={"query": query}, headers=headers_gql)
                data = resp.json()
            products = data.get("data", {}).get("products", {})
            edges = products.get("edges", [])
            for edge in edges:
                p = edge["node"]
                mf = {}
                for mf_edge in p.get("metafields", {}).get("edges", []):
                    node = mf_edge["node"]
                    key = node["key"].replace("parfum.", "")
                    mf[key] = node["value"]
                price = float(p["priceRangeV2"]["minVariantPrice"]["amount"])
                img = p.get("featuredImage", {})
                img_url = (img.get("url", "").split("?")[0] + "?width=400") if img.get("url") else ""
                def parse_list(val):
                    if not val: return []
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, list): return parsed
                    except: pass
                    return [val]
                variants = []
                for v in p.get("variants", {}).get("edges", []):
                    vn = v["node"]
                    variants.append({"id": vn["id"].split("/")[-1], "ml": vn["title"], "pr": float(vn["price"]), "av": vn["availableForSale"]})
                all_products.append({
                    "h": p["handle"], "t": p["title"], "v": p["vendor"], "p": price, "img": img_url,
                    "g": mf.get("genre", ""), "fam": parse_list(mf.get("famille_olfactive", "")),
                    "acc": mf.get("accord_principal", ""), "acc2": parse_list(mf.get("accords_secondaires", "")),
                    "sai": parse_list(mf.get("saison", "")), "occ": parse_list(mf.get("occasions", "")),
                    "int": int(mf.get("intensite", "3")), "sil": int(mf.get("sillage_level", "2")),
                    "nt": mf.get("note_tete_principale", ""), "nc": mf.get("note_coeur_principale", ""),
                    "nf": mf.get("note_fond_principale", ""), "nk": parse_list(mf.get("notes_cles", "")),
                    "conc": mf.get("concentration", p["productType"]), "var": variants
                })
            pi = products.get("pageInfo", {})
            if not pi.get("hasNextPage"): break
            cursor = pi.get("endCursor")

        # Upload to theme
        quiz_json = json.dumps(all_products, ensure_ascii=False, separators=(",", ":"))
        # Get active theme
        async with httpx.AsyncClient(timeout=30) as client:
            themes_resp = await client.get(
                f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/themes.json",
                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
            )
            themes = themes_resp.json().get("themes", [])
            main_theme = next((t for t in themes if t["role"] == "main"), None)
            if not main_theme:
                return {"success": False, "error": "No main theme found"}
            # Upload asset
            asset_resp = await client.put(
                f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/themes/{main_theme['id']}/assets.json",
                json={"asset": {"key": "assets/quiz-data.json", "value": quiz_json}},
                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
            )
            if asset_resp.status_code != 200:
                return {"success": False, "error": f"Upload failed: {asset_resp.status_code}"}

        logger.info(f"Quiz data regenerated: {len(all_products)} products")
        return {"success": True, "count": len(all_products), "size_kb": round(len(quiz_json) / 1024, 1)}
    except Exception as e:
        logger.error(f"Quiz regenerate error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/active-discounts")
async def active_discounts():
    """Return active discount codes for cart suggestions. Cached 5min in Redis."""
    rc = get_redis()
    cache_key = "stella:active_discounts"
    if rc:
        cached = rc.get(cache_key)
        if cached:
            return json.loads(cached)
    gql_url = f"https://{SHOPIFY_SHOP}/admin/api/{API_VERSION}/graphql.json"
    query = """{ codeDiscountNodes(first: 10, query: "status:active") { edges { node { codeDiscount {
      ... on DiscountCodeBasic { title codes(first:1){edges{node{code}}} summary endsAt }
      ... on DiscountCodeFreeShipping { title codes(first:1){edges{node{code}}} summary endsAt }
    } } } } }"""
    discounts = []
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={"query": query},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            data = r.json().get("data", {})
            for edge in data.get("codeDiscountNodes", {}).get("edges", []):
                cd = edge["node"]["codeDiscount"]
                codes = cd.get("codes", {}).get("edges", [])
                code = codes[0]["node"]["code"] if codes else ""
                if not code: continue
                discounts.append({"code": code, "summary": cd.get("summary", ""), "endsAt": cd.get("endsAt"), "isPromo": cd.get("endsAt") is not None})
    except Exception as e:
        logger.error(f"Active discounts: {e}")
    result = {"discounts": discounts}
    if rc:
        try: rc.setex(cache_key, 300, json.dumps(result))
        except: pass
    return result

@app.get("/api/validate-discount/{code}")
async def validate_discount(code: str):
    """Validate a discount code and return its rules for client-side calculation."""
    rc = get_redis()
    cache_key = f"stella:discount:{code.upper()}"
    if rc:
        cached = rc.get(cache_key)
        if cached:
            return json.loads(cached)
    gql_url = f"https://{SHOPIFY_SHOP}/admin/api/{API_VERSION}/graphql.json"
    query = """{ codeDiscountNodes(first: 5, query: "code:%s") { edges { node { codeDiscount {
      ... on DiscountCodeBasic {
        title status
        codes(first:1){edges{node{code}}}
        customerGets { value { ... on DiscountPercentage { percentage } ... on DiscountAmount { amount { amount } } } }
        minimumRequirement { ... on DiscountMinimumSubtotal { greaterThanOrEqualToSubtotal { amount } } }
      }
    } } } } }""" % code.upper()
    result = {"valid": False, "code": code.upper()}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={"query": query},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            data = r.json().get("data", {})
            for edge in data.get("codeDiscountNodes", {}).get("edges", []):
                cd = edge["node"]["codeDiscount"]
                if cd.get("status") != "ACTIVE": continue
                codes = cd.get("codes", {}).get("edges", [])
                if not codes: continue
                if codes[0]["node"]["code"].upper() != code.upper(): continue
                value = cd.get("customerGets", {}).get("value", {})
                pct = value.get("percentage")
                fixed = value.get("amount", {}).get("amount") if "amount" in value else None
                min_req = cd.get("minimumRequirement", {})
                min_amount = None
                if "greaterThanOrEqualToSubtotal" in min_req:
                    min_amount = float(min_req["greaterThanOrEqualToSubtotal"]["amount"])
                result = {
                    "valid": True,
                    "code": code.upper(),
                    "type": "percentage" if pct else "fixed",
                    "percentage": pct * 100 if pct else None,
                    "fixedAmount": float(fixed) if fixed else None,
                    "minimumAmount": min_amount
                }
                break
    except Exception as e:
        logger.error(f"Validate discount: {e}")
    if rc and result.get("valid"):
        try: rc.setex(cache_key, 300, json.dumps(result))
        except: pass
    return result

@app.get("/api/promo-codes")
async def get_promo_codes():
    """Read promo codes config from the automatic discount metafield."""
    gql_url = SHOPIFY_GRAPHQL_URL
    query = """{
      discountNodes(first: 10, query: "title:'PB Codes Promo'") {
        edges { node { id metafield(namespace: "planete-beaute", key: "discount-codes-config") { value } } }
      }
    }"""
    codes = []
    discount_id = None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={"query": query},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            data = r.json().get("data", {})
            for edge in data.get("discountNodes", {}).get("edges", []):
                node = edge["node"]
                discount_id = node["id"]
                mf = node.get("metafield")
                if mf and mf.get("value"):
                    config = json.loads(mf["value"])
                    raw_codes = config.get("codes", [])
                    if isinstance(raw_codes, dict):
                        for k, v in raw_codes.items():
                            codes.append({"code": k, "percentage": v.get("percent", v.get("percentage", 0)),
                                         "minimumAmount": v.get("minSubtotal", v.get("minimumAmount", 0)),
                                         "expiresAt": v.get("expiresAt"), "message": v.get("message", "")})
                    elif isinstance(raw_codes, list):
                        codes = raw_codes
                break
    except Exception as e:
        logger.error(f"Get promo codes: {e}")
    return {"codes": codes, "discountId": discount_id}


@app.post("/api/promo-codes")
async def save_promo_codes(request: Request):
    """Save promo codes config to the automatic discount metafield."""
    body = await request.json()
    codes = body.get("codes", [])
    discount_id = body.get("discountId")
    if not discount_id:
        return {"success": False, "error": "discountId manquant"}
    gql_url = SHOPIFY_GRAPHQL_URL
    config_json = json.dumps({"codes": codes, "strategy": "MAXIMUM"})
    mutation = """mutation($ownerId: ID!, $mf: [MetafieldsSetInput!]!) {
      metafieldsSet(metafields: $mf) { metafields { id } userErrors { field message } }
    }"""
    variables = {
        "ownerId": discount_id,
        "mf": [{"ownerId": discount_id, "namespace": "planete-beaute", "key": "discount-codes-config",
                "type": "json", "value": config_json}]
    }
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={"query": mutation, "variables": variables},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("metafieldsSet", {}).get("userErrors", [])
            if errors:
                return {"success": False, "error": errors[0].get("message", "Erreur")}
    except Exception as e:
        logger.error(f"Save promo codes: {e}")
        return {"success": False, "error": str(e)}
    return {"success": True, "count": len(codes)}


@app.post("/api/reviews/import")
async def import_reviews(request: Request):
    """Import reviews from Judge.me CSV export into PostgreSQL."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")
    body = await request.json()
    reviews = body.get("reviews", [])
    if not reviews:
        return {"imported": 0}
    db = get_db()
    if not db:
        return {"imported": 0, "error": "No DB"}
    try:
        cur = db.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS product_reviews (
            id SERIAL PRIMARY KEY,
            title TEXT,
            body TEXT,
            rating INTEGER,
            review_date TIMESTAMP,
            source TEXT,
            curated TEXT,
            reviewer_name TEXT,
            reviewer_email TEXT,
            product_id TEXT,
            product_handle TEXT,
            reply TEXT,
            reply_date TIMESTAMP,
            picture_urls TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )""")
        imported = 0
        for r in reviews:
            try:
                rd = r.get("date") or None
                rpd = r.get("reply_date") or None
                cur.execute("""INSERT INTO product_reviews
                    (title, body, rating, review_date, source, curated, reviewer_name, reviewer_email, product_id, product_handle, reply, reply_date, picture_urls)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (r.get("title",""), r.get("body",""), r.get("rating",0), rd, r.get("source",""), r.get("curated",""),
                     r.get("name",""), r.get("email",""), r.get("product_id",""), r.get("product_handle",""),
                     r.get("reply",""), rpd if rpd else None, r.get("picture_urls","")))
                imported += 1
            except Exception as e:
                logger.warning(f"Review import row error: {e}")
        db.commit()
        cur.close()
        db.close()
        return {"imported": imported}
    except Exception as e:
        try: db.close()
        except: pass
        return {"imported": 0, "error": str(e)}


@app.get("/api/reviews/{product_handle}")
async def get_product_reviews(product_handle: str):
    """Get reviews for a specific product."""
    db = get_db()
    if not db:
        return {"reviews": [], "count": 0, "average": 0}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""SELECT title, body, rating, review_date, reviewer_name, picture_urls
            FROM product_reviews WHERE product_handle = %s AND curated = 'ok'
            ORDER BY review_date DESC""", (product_handle,))
        reviews = cur.fetchall()
        for r in reviews:
            if r.get("review_date"):
                r["review_date"] = r["review_date"].isoformat()
        avg = sum(r["rating"] for r in reviews) / len(reviews) if reviews else 0
        cur.close()
        db.close()
        return {"reviews": reviews, "count": len(reviews), "average": round(avg, 2)}
    except Exception as e:
        try: db.close()
        except: pass
        return {"reviews": [], "count": 0, "average": 0, "error": str(e)}


@app.get("/api/nouveautes")
async def get_nouveautes():
    """Real-time: fetch products with tag Nouveauté from Shopify."""
    from datetime import timezone, timedelta
    gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}
    products = []
    cursor = None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                after = f', after: "{cursor}"' if cursor else ''
                r = await client.post(gql_url, json={"query": f'{{ products(first: 50, query: "tag:Nouveauté", sortKey: CREATED_AT, reverse: true{after}) {{ edges {{ cursor node {{ id title handle createdAt status vendor featuredImage {{ url }} }} }} pageInfo {{ hasNextPage }} }} }}'}, headers=headers)
                data = r.json().get("data", {}).get("products", {})
                for edge in data.get("edges", []):
                    p = edge["node"]
                    cursor = edge["cursor"]
                    days_since = (datetime.now(timezone.utc) - datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))).days
                    days_left = max(0, 30 - days_since)
                    products.append({
                        "title": p["title"],
                        "vendor": p.get("vendor", ""),
                        "handle": p.get("handle", ""),
                        "created": p["createdAt"][:10],
                        "days_left": days_left,
                        "status": p.get("status", ""),
                        "image": p.get("featuredImage", {}).get("url", "") if p.get("featuredImage") else ""
                    })
                if not data.get("pageInfo", {}).get("hasNextPage"):
                    break
    except Exception as e:
        logger.error(f"Get nouveautes: {e}")
        return {"products": [], "error": str(e)}
    return {"products": products, "count": len(products)}


@app.post("/api/webhook/product-change")
async def webhook_product_change(request: Request):
    """Shopify webhook: product created/updated/deleted → regenerate quiz data.
    Debounced: only regenerates if no other call in last 60 seconds."""
    rc = get_redis()
    if rc:
        last = rc.get("quiz:regen:last")
        now = int(time.time())
        if last and now - int(last) < 60:
            return {"ok": True, "skipped": True, "reason": "debounced"}
        rc.set("quiz:regen:last", now, ex=120)
    # Trigger regeneration in background
    import asyncio
    body_raw = await request.body()
    try:
        pdata = json.loads(body_raw)
        ptitle = pdata.get("title", "Unknown")
    except: ptitle = "Unknown"
    asyncio.create_task(quiz_regenerate())
    logger.info("Quiz data regeneration triggered by product webhook")
    log_activity("product_change", f"Produit modifié : {ptitle}", {"product_title": ptitle}, source="webhook", product_title=ptitle)
    return {"ok": True, "regenerating": True}

# ══════════════════════ CASHBACK FIDÉLITÉ (Crédit Magasin) ══════════════════════

_CASHBACK_DEFAULTS = {
    "cashback_rate": 0.05, "expiry_days": 60, "min_order_use": 70.0,
    "exclude_shipping": True, "exclude_taxes": True, "exclude_discounts": True,
    "excluded_tags": "tryme,no-cashback", "excluded_product_ids": "",
    "min_cashback_amount": 0.50, "is_active": True
}

def get_cashback_settings() -> dict:
    """Load cashback settings: Redis cache (5min) → PostgreSQL → defaults."""
    rc = get_redis()
    if rc:
        cached = rc.get("cashback:settings")
        if cached:
            try: return json.loads(cached)
            except: pass
    db = get_db()
    if not db: return dict(_CASHBACK_DEFAULTS)
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM cashback_settings WHERE shop = %s LIMIT 1", ("planetemode.myshopify.com",))
        row = cur.fetchone()
        cur.close(); db.close()
        if row:
            result = {
                "cashback_rate": float(row["cashback_rate"]),
                "expiry_days": int(row["expiry_days"]),
                "min_order_use": float(row["min_order_use"]),
                "exclude_shipping": bool(row["exclude_shipping"]),
                "exclude_taxes": bool(row["exclude_taxes"]),
                "exclude_discounts": bool(row["exclude_discounts"]),
                "excluded_tags": row["excluded_tags"] or "",
                "excluded_product_ids": row["excluded_product_ids"] or "",
                "min_cashback_amount": float(row["min_cashback_amount"]),
                "is_active": bool(row["is_active"]),
            }
            if rc:
                rc.setex("cashback:settings", 300, json.dumps(result))
            return result
    except Exception as e:
        logger.warning(f"Load cashback settings: {e}")
        try: db.close()
        except: pass
    return dict(_CASHBACK_DEFAULTS)

@app.post("/api/webhook/order-paid")
async def webhook_order_paid(request: Request):
    """Shopify webhook: order paid → calculate cashback → add store credit to customer.
    Settings loaded dynamically from cashback_settings table."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}

    # Load dynamic settings
    settings = get_cashback_settings()
    if not settings["is_active"]:
        return {"ok": True, "skipped": True, "reason": "cashback disabled"}

    order_id = body.get("id")
    order_name = body.get("name", "")
    customer = body.get("customer")
    tags = body.get("tags", "")

    if not customer or not customer.get("id"):
        return {"ok": True, "skipped": True, "reason": "no customer"}

    # Skip orders matching excluded tags
    excluded_tags = [t.strip().lower() for t in settings["excluded_tags"].split(",") if t.strip()]
    order_tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if any(et in order_tags for et in excluded_tags):
        return {"ok": True, "skipped": True, "reason": f"excluded tag match"}

    # Calculate cashback base:
    # subtotal_price = produits après remises (hors port, hors taxes)
    # On soustrait le crédit magasin utilisé (store credit used in payment)
    subtotal = float(body.get("subtotal_price", "0"))  # Produits après remises, hors port
    total_discounts = float(body.get("total_discounts", "0"))  # Remises déjà soustraites du subtotal

    # Detect store credit used in this order (payment gateway = "gift_card" or "store_credit")
    store_credit_used = 0.0
    for txn in body.get("payment_gateway_names", []):
        if "gift_card" in txn.lower() or "store_credit" in txn.lower():
            # Look in transactions for the store credit amount
            break
    # More precise: check refunds/transactions for store credit
    for line in body.get("payment_terms", {}).get("payment_schedules", []) or []:
        pass  # Fallback
    # Safest method: check total_price vs subtotal + shipping - discounts
    shipping_total = sum(float(s.get("price", "0")) for s in body.get("shipping_lines", []))
    # If customer paid with partial store credit, the difference shows in transactions
    # For now, use the order's current_subtotal_price which is after ALL adjustments
    current_subtotal = float(body.get("current_subtotal_price", subtotal))

    # Base for cashback = subtotal after discounts, minus any store credit portion
    # Shopify webhook includes total_outstanding (what customer actually paid out of pocket)
    # total_outstanding = 0 means fully paid. We need the non-credit portion.
    # Best approach: fetch order details via Admin API for precise credit detection
    cashback_base = current_subtotal  # Subtotal after discounts, before shipping, before tax
    customer_id = customer["id"]
    customer_email = customer.get("email", "")
    customer_gid = f"gid://shopify/Customer/{customer_id}"
    gql_url = SHOPIFY_GRAPHQL_URL

    # Fetch precise store credit usage from Admin API
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={
                "query": """{order(id: "gid://shopify/Order/%s") {
                    currentSubtotalPriceSet { shopMoney { amount } }
                    totalReceivedSet { shopMoney { amount } }
                    transactions(first: 20) { edges { node { gateway amountSet { shopMoney { amount } } kind status } } }
                }}""" % order_id
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            order_data = r.json().get("data", {}).get("order", {})
            if order_data:
                cashback_base = float(order_data.get("currentSubtotalPriceSet", {}).get("shopMoney", {}).get("amount", cashback_base))
                # Check transactions for store credit / gift card usage
                for edge in order_data.get("transactions", {}).get("edges", []):
                    txn = edge["node"]
                    if txn.get("kind") == "SALE" and txn.get("status") == "SUCCESS":
                        gw = (txn.get("gateway") or "").lower()
                        if "gift_card" in gw or "store_credit" in gw or "credit" in gw:
                            store_credit_used += float(txn.get("amountSet", {}).get("shopMoney", {}).get("amount", "0"))
    except Exception as e:
        logger.warning(f"Cashback order fetch: {e}")

    # Cashback = 5% of (subtotal after discounts MINUS store credit used)
    cashback_base = max(0, cashback_base - store_credit_used)
    cashback_amount = round(cashback_base * settings["cashback_rate"], 2)

    if cashback_amount < settings["min_cashback_amount"]:
        return {"ok": True, "skipped": True, "reason": f"cashback {cashback_amount} < {settings['min_cashback_amount']}"}

    from datetime import timedelta
    expires_at = (datetime.utcnow() + timedelta(days=settings["expiry_days"])).strftime("%Y-%m-%dT23:59:59Z")

    # 1. Get or detect store credit account for customer
    credit_account_id = None
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(gql_url, json={
                "query": """{customer(id: "%s") { storeCreditAccounts(first:1) { edges { node { id balance { amount } } } } }}""" % customer_gid
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            edges = r.json().get("data", {}).get("customer", {}).get("storeCreditAccounts", {}).get("edges", [])
            if edges:
                credit_account_id = edges[0]["node"]["id"]
    except Exception as e:
        logger.warning(f"Cashback get credit account: {e}")

    # 2. Add store credit to customer
    credit_added = False
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            mutation = """mutation($id: ID!, $creditInput: StoreCreditAccountCreditInput!) {
              storeCreditAccountCredit(id: $id, creditInput: $creditInput) {
                storeCreditAccountTransaction { id amount { amount currencyCode } }
                userErrors { field message }
              }
            }"""
            # If no credit account exists, use customer GID and Shopify creates one
            target_id = credit_account_id or customer_gid
            r = await c.post(gql_url, json={
                "query": mutation,
                "variables": {
                    "id": target_id,
                    "creditInput": {
                        "creditAmount": {"amount": str(cashback_amount), "currencyCode": "EUR"},
                        "expiresAt": expires_at
                    }
                }
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("storeCreditAccountCredit", {}).get("userErrors", [])
            if not errors and resp.get("data", {}).get("storeCreditAccountCredit", {}).get("storeCreditAccountTransaction"):
                credit_added = True
            else:
                logger.error(f"Cashback store credit error: {errors} / {resp}")
    except Exception as e:
        logger.error(f"Cashback store credit: {e}")

    if not credit_added:
        return {"ok": False, "error": "Failed to add store credit"}

    # 3. Write metafield on order for invoice (Order Printer)
    try:
        order_gid = f"gid://shopify/Order/{order_id}"
        formatted_amount = f"{cashback_amount:.2f}".replace(".", ",") + " €"
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(gql_url, json={
                "query": """mutation($mf: [MetafieldsSetInput!]!) { metafieldsSet(metafields: $mf) { metafields { id } userErrors { field message } } }""",
                "variables": {"mf": [
                    {"ownerId": order_gid, "namespace": "planete-beaute", "key": "cashback-amount", "type": "single_line_text_field", "value": formatted_amount},
                    {"ownerId": order_gid, "namespace": "planete-beaute", "key": "cashback-amount-raw", "type": "number_decimal", "value": str(cashback_amount)}
                ]}
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
        logger.info(f"Cashback metafield written on order {order_name}: {formatted_amount}")
    except Exception as e:
        logger.warning(f"Cashback metafield write: {e}")

    # 4. Log to PostgreSQL
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS cashback_rewards (
                id SERIAL PRIMARY KEY, customer_id BIGINT, customer_email TEXT,
                order_id BIGINT, order_name TEXT, order_subtotal DECIMAL(10,2),
                store_credit_used DECIMAL(10,2), cashback_base DECIMAL(10,2),
                cashback_amount DECIMAL(10,2), credit_type TEXT DEFAULT 'store_credit',
                expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT NOW(), status TEXT DEFAULT 'active')""")
            cur.execute("""INSERT INTO cashback_rewards (customer_id, customer_email, order_id, order_name,
                order_subtotal, store_credit_used, cashback_base, cashback_amount, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (customer_id, customer.get("email", ""), order_id, order_name,
                 current_subtotal, store_credit_used, cashback_base, cashback_amount, expires_at))
            db.commit()
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.error(f"Cashback DB insert: {e}")

    logger.info(f"Cashback {cashback_amount}EUR store credit for customer {customer_id} order {order_name} (base:{cashback_base}, credit_used:{store_credit_used})")
    log_activity("cashback_credit", f"Cashback {cashback_amount}€ crédité",
                 {"cashback_amount": cashback_amount, "base": cashback_base, "store_credit_used": store_credit_used, "customer_id": customer_id},
                 source="webhook", customer_email=customer_email, order_name=order_name)
    return {"ok": True, "cashback": cashback_amount, "base": cashback_base, "storeCreditUsed": store_credit_used}


@app.post("/api/webhook/refund-created")
async def webhook_refund_created(request: Request):
    """When an order is refunded, debit the cashback store credit from the customer."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}

    order_id = body.get("order_id")
    if not order_id:
        return {"ok": True, "skipped": True, "reason": "no order_id"}

    # Find the cashback reward for this order
    db = get_db()
    if not db:
        return {"ok": False, "error": "no db"}
    try:
        cur = db.cursor()
        cur.execute("SELECT customer_id, cashback_amount, status FROM cashback_rewards WHERE order_id = %s ORDER BY id DESC LIMIT 1", (order_id,))
        row = cur.fetchone()
        cur.close(); db.close()
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"Refund cashback lookup: {e}")
        return {"ok": False, "error": str(e)}

    if not row:
        return {"ok": True, "skipped": True, "reason": "no cashback for this order"}
    if row[2] == "revoked":
        return {"ok": True, "skipped": True, "reason": "already revoked"}

    customer_id, cashback_amount, _ = row
    customer_gid = f"gid://shopify/Customer/{customer_id}"

    # Debit the store credit
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={
                "query": """mutation($id: ID!, $debitInput: StoreCreditAccountDebitInput!) {
                  storeCreditAccountDebit(id: $id, debitInput: $debitInput) {
                    storeCreditAccountTransaction { id }
                    userErrors { field message }
                  }
                }""",
                "variables": {
                    "id": customer_gid,
                    "debitInput": {
                        "debitAmount": {"amount": str(cashback_amount), "currencyCode": "EUR"}
                    }
                }
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("storeCreditAccountDebit", {}).get("userErrors", [])
            if not errors:
                # Mark as revoked in DB
                db2 = get_db()
                if db2:
                    cur2 = db2.cursor()
                    cur2.execute("UPDATE cashback_rewards SET status = 'revoked' WHERE order_id = %s", (order_id,))
                    db2.commit(); cur2.close(); db2.close()
                log_activity("cashback_revoked", f"Cashback {cashback_amount}€ révoqué (remboursement)",
                             {"cashback_amount": cashback_amount, "customer_id": customer_id, "order_id": order_id},
                             source="webhook", order_name=f"Order {order_id}")
                logger.info(f"Cashback {cashback_amount}EUR revoked for customer {customer_id} (refund order {order_id})")
                return {"ok": True, "revoked": cashback_amount}
            else:
                logger.error(f"Refund debit error: {errors}")
                return {"ok": False, "error": str(errors)}
    except Exception as e:
        logger.error(f"Refund cashback debit: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/cashback/dashboard")
async def cashback_dashboard():
    """Dashboard data for cashback tab in STELLA V8. Uses store credit (not discount codes)."""
    db = get_db()
    cb_settings = get_cashback_settings()
    stats = {"total_rewarded": 0, "total_amount": 0, "total_used": 0, "min_use": cb_settings["min_order_use"], "recent": []}
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COUNT(*) as c, COALESCE(SUM(cashback_amount),0) as total FROM cashback_rewards")
            row = cur.fetchone()
            stats["total_rewarded"] = row["c"]
            stats["total_amount"] = float(row["total"])
            cur.execute("SELECT COUNT(*) as c FROM cashback_rewards WHERE status='used'")
            stats["total_used"] = cur.fetchone()["c"]
            cur.execute("""SELECT customer_email, order_name, cashback_amount, discount_code, status, created_at
                          FROM cashback_rewards ORDER BY created_at DESC LIMIT 20""")
            stats["recent"] = [dict(r) for r in cur.fetchall()]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.warning(f"Cashback dashboard: {e}")
    return stats


@app.get("/api/cashback/settings")
async def get_cashback_settings_api():
    """Get cashback configuration for admin panel."""
    return get_cashback_settings()

async def _sync_cashback_metafields(settings: dict) -> bool:
    """Sync cashback settings to Shopify shop metafields for Payment Function."""
    try:
        # Get shop ID first
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": "{ shop { id } }"},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            shop_id = r.json().get("data", {}).get("shop", {}).get("id", "")
        if not shop_id:
            logger.error("Cashback metafield sync: could not get shop ID")
            return False
        config_json = json.dumps({
            "rate": settings.get("cashback_rate", 0.05),
            "expiry_days": settings.get("expiry_days", 60),
            "min_order_use": settings.get("min_order_use", 70.0),
            "exclude_shipping": settings.get("exclude_shipping", True),
            "exclude_taxes": settings.get("exclude_taxes", True),
            "exclude_discounts": settings.get("exclude_discounts", True),
            "excluded_tags": settings.get("excluded_tags", ""),
            "min_cashback_amount": settings.get("min_cashback_amount", 0.50),
            "is_active": settings.get("is_active", True),
        })
        mutation = """mutation($metafields: [MetafieldsSetInput!]!) {
          metafieldsSet(metafields: $metafields) {
            metafields { id namespace key }
            userErrors { field message }
          }
        }"""
        variables = {"metafields": [{
            "ownerId": shop_id,
            "namespace": "stella_cashback",
            "key": "config",
            "type": "json",
            "value": config_json,
        }]}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": variables},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("metafieldsSet", {}).get("userErrors", [])
            if errors:
                logger.error(f"Cashback metafield sync errors: {errors}")
                return False
            return True
    except Exception as e:
        logger.error(f"Cashback metafield sync: {e}")
        return False

@app.post("/api/cashback/settings")
async def save_cashback_settings_api(request: Request):
    """Save cashback configuration + sync to Shopify metafields."""
    body = await request.json()
    db = get_db()
    if not db:
        return {"success": False, "error": "Database unavailable"}
    try:
        cur = db.cursor()
        cur.execute("""UPDATE cashback_settings SET
            cashback_rate = %s, expiry_days = %s, min_order_use = %s,
            exclude_shipping = %s, exclude_taxes = %s, exclude_discounts = %s,
            excluded_tags = %s, excluded_product_ids = %s,
            min_cashback_amount = %s, is_active = %s, updated_at = NOW()
            WHERE shop = %s""",
            (body.get("cashback_rate", 0.05), body.get("expiry_days", 60), body.get("min_order_use", 70.0),
             body.get("exclude_shipping", True), body.get("exclude_taxes", True), body.get("exclude_discounts", True),
             body.get("excluded_tags", "tryme,no-cashback"), body.get("excluded_product_ids", ""),
             body.get("min_cashback_amount", 0.50), body.get("is_active", True),
             "planetemode.myshopify.com"))
        db.commit(); cur.close(); db.close()
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"Save cashback settings: {e}")
        return {"success": False, "error": str(e)}
    # Invalidate Redis cache
    rc = get_redis()
    if rc:
        rc.delete("cashback:settings")
    # Sync to Shopify metafields
    sync_ok = await _sync_cashback_metafields(body)
    logger.info(f"Cashback settings saved (metafield sync: {sync_ok})")
    return {"success": True, "metafield_sync": sync_ok}


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
        log_activity("bis_notification", f"BIS: {emails_sent} emails envoyés pour {product_title}",
                     {"product_title": product_title, "subscribers": len(subscribers), "emails_sent": emails_sent, "emails": notified_emails},
                     source="webhook", product_title=product_title)

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


# ══════════════════════ DASHBOARD V8 API ══════════════════════

@app.get("/api/activity/log")
async def get_activity_log(limit: int = 50, type: str = None):
    """Journal centralisé — tour de contrôle."""
    db = get_db()
    if not db: return {"activities": [], "total": 0}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if type and type != "all":
            cur.execute("SELECT * FROM activity_log WHERE type=%s ORDER BY timestamp DESC LIMIT %s", (type, limit))
        else:
            cur.execute("SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) as total FROM activity_log")
        total = cur.fetchone()["total"]
        cur.close(); db.close()
        for r in rows:
            r["timestamp"] = r["timestamp"].isoformat() if r.get("timestamp") else None
        return {"activities": rows, "total": total}
    except Exception as e:
        try: db.close()
        except: pass
        return {"activities": [], "total": 0, "error": str(e)}

@app.get("/api/kpis/summary")
async def kpis_summary():
    """KPIs temps réel — CA, commandes, panier moyen, cashback, BIS, quiz."""
    result = {"revenue_today": 0, "orders_today": 0, "avg_order_value": 0,
              "cashback_generated_today": 0, "bis_active": 0, "quiz_views_today": 0}
    try:
        # Revenue + orders from Shopify
        from datetime import timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        query = f'''{{ orders(first: 250, query: "created_at:>='{today}'") {{ edges {{ node {{ name totalPriceSet {{ shopMoney {{ amount }} }} displayFinancialStatus }} }} }} }}'''
        data = await shopify_graphql(query)
        orders = data.get("data", {}).get("orders", {}).get("edges", [])
        paid = [o for o in orders if o["node"].get("displayFinancialStatus") in ["PAID", "PARTIALLY_PAID", "PARTIALLY_REFUNDED"]]
        result["orders_today"] = len(paid)
        result["revenue_today"] = round(sum(float(o["node"]["totalPriceSet"]["shopMoney"]["amount"]) for o in paid), 2)
        result["avg_order_value"] = round(result["revenue_today"] / max(result["orders_today"], 1), 2)
    except Exception as e:
        logger.warning(f"KPI shopify error: {e}")
    # Cashback today
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COALESCE(SUM((details->>'cashback_amount')::numeric), 0) as total FROM activity_log WHERE type='cashback_credit' AND timestamp >= CURRENT_DATE")
            result["cashback_generated_today"] = float(cur.fetchone()["total"])
            cur.execute("SELECT COUNT(*) as c FROM bis_subscriptions WHERE status='active'")
            result["bis_active"] = cur.fetchone()["c"]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
    # Quiz views
    rc = get_redis()
    if rc:
        try:
            today_key = datetime.now().strftime("%Y-%m-%d")
            result["quiz_views_today"] = int(rc.get(f"quiz:quiz_view:{today_key}") or 0)
        except: pass
    return result

@app.get("/api/orders/dashboard")
async def orders_dashboard():
    """Commandes du jour depuis Shopify."""
    rc = get_redis()
    cached = None
    if rc:
        try: cached = rc.get("dashboard:orders:today")
        except: pass
    if cached:
        return json.loads(cached)
    try:
        from datetime import timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z")
        query = f'''{{ orders(first: 50, query: "created_at:>='{today}'", sortKey: CREATED_AT, reverse: true) {{ edges {{ node {{ id name createdAt totalPriceSet {{ shopMoney {{ amount currencyCode }} }} displayFinancialStatus displayFulfillmentStatus customer {{ email displayName }} tags }} }} }} }}'''
        data = await shopify_graphql(query)
        orders = []
        for edge in data.get("data", {}).get("orders", {}).get("edges", []):
            n = edge["node"]
            orders.append({
                "name": n["name"], "created_at": n["createdAt"],
                "total": float(n["totalPriceSet"]["shopMoney"]["amount"]),
                "financial_status": n.get("displayFinancialStatus", ""),
                "fulfillment_status": n.get("displayFulfillmentStatus", "UNFULFILLED"),
                "customer_email": n.get("customer", {}).get("email", ""),
                "customer_name": n.get("customer", {}).get("displayName", ""),
                "tags": n.get("tags", [])
            })
        revenue = round(sum(o["total"] for o in orders if o["financial_status"] in ["PAID", "PARTIALLY_PAID", "PARTIALLY_REFUNDED"]), 2)
        result = {"orders": orders, "count": len(orders), "revenue_today": revenue}
        if rc:
            try: rc.setex("dashboard:orders:today", 60, json.dumps(result))
            except: pass
        return result
    except Exception as e:
        return {"orders": [], "count": 0, "revenue_today": 0, "error": str(e)}

@app.get("/api/trustpilot/dashboard")
async def trustpilot_dashboard_api():
    """Stats Trustpilot — crédits, scans, récents."""
    db = get_db()
    if not db: return {"total_credits": 0, "total_amount": 0, "recent": [], "last_scan": None}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) as c, COALESCE(SUM(amount), 0) as total FROM trustpilot_credits")
        stats = cur.fetchone()
        cur.execute("SELECT * FROM trustpilot_credits ORDER BY credited_at DESC LIMIT 20")
        recent = cur.fetchall()
        for r in recent:
            if r.get("credited_at"): r["credited_at"] = r["credited_at"].isoformat()
        cur.execute("SELECT result_data as result, executed_at FROM cron_results WHERE cron_name='trustpilot-scan' ORDER BY executed_at DESC LIMIT 1")
        scan = cur.fetchone()
        if scan and scan.get("executed_at"): scan["executed_at"] = scan["executed_at"].isoformat()
        cur.close(); db.close()
        return {"total_credits": stats["c"], "total_amount": float(stats["total"]), "recent": recent, "last_scan": scan}
    except Exception as e:
        try: db.close()
        except: pass
        return {"total_credits": 0, "total_amount": 0, "recent": [], "last_scan": None, "error": str(e)}

@app.get("/api/catalogue/dashboard")
async def catalogue_dashboard():
    """Catalogue — audit qualité, completude, nouveautés."""
    result = {"total_products": 0, "products_with_issues": 0, "issues": [], "last_audit": None}
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT result_data as result, executed_at FROM cron_results WHERE cron_name='audit-qualite' ORDER BY executed_at DESC LIMIT 1")
            audit = cur.fetchone()
            if audit:
                result["last_audit"] = audit["executed_at"].isoformat() if audit.get("executed_at") else None
                audit_data = audit.get("result", {})
                if isinstance(audit_data, str):
                    try: audit_data = json.loads(audit_data)
                    except: audit_data = {}
                result["products_with_issues"] = audit_data.get("issues_count", 0)
                result["issues"] = audit_data.get("issues", [])[:50]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
    try:
        query = '{ productsCount { count } }'
        data = await shopify_graphql(query)
        result["total_products"] = data.get("data", {}).get("productsCount", {}).get("count", 0)
    except: pass
    return result

@app.get("/api/system/status")
async def system_status():
    """Système — santé services, webhooks, crons, erreurs."""
    result = {"services": {}, "crons": [], "webhooks": [], "errors_24h": 0, "version": "8.0.0"}
    # Services health
    rc = get_redis()
    result["services"]["redis"] = "online" if rc and rc.ping() else "offline"
    db = get_db()
    if db:
        try:
            cur = db.cursor(); cur.execute("SELECT 1"); cur.close()
            result["services"]["postgresql"] = "online"
        except: result["services"]["postgresql"] = "offline"
        finally:
            try: db.close()
            except: pass
    else:
        result["services"]["postgresql"] = "offline"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/shop.json",
                                 headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN})
            result["services"]["shopify"] = "online" if r.status_code == 200 else "error"
    except: result["services"]["shopify"] = "offline"
    result["services"]["smtp"] = "online" if SMTP_HOST and SMTP_USER else "not_configured"
    # Crons — derniers résultats
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""SELECT DISTINCT ON (cron_name) cron_name, result_data as result, executed_at,
                           CASE WHEN result_data::text LIKE '%error%' OR result_data::text LIKE '%Error%' THEN 'error' ELSE 'success' END as status
                           FROM cron_results ORDER BY cron_name, executed_at DESC""")
            crons = cur.fetchall()
            for c in crons:
                if c.get("executed_at"): c["executed_at"] = c["executed_at"].isoformat()
            result["crons"] = crons
            # Errors 24h
            cur.execute("SELECT COUNT(*) as c FROM activity_log WHERE status='error' AND timestamp >= NOW() - INTERVAL '24 hours'")
            result["errors_24h"] = cur.fetchone()["c"]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
    # Webhooks
    try:
        query = '{ webhookSubscriptions(first: 20) { nodes { id topic endpoint { ... on WebhookHttpEndpoint { callbackUrl } } } } }'
        data = await shopify_graphql(query)
        wh = data.get("data", {}).get("webhookSubscriptions", {}).get("nodes", [])
        result["webhooks"] = [{"topic": w["topic"], "url": w.get("endpoint", {}).get("callbackUrl", "")} for w in wh]
    except: pass
    return result

@app.get("/api/reviews/dashboard")
async def reviews_dashboard():
    """Avis — stats globales, récents, par source."""
    db = get_db()
    if not db: return {"total": 0, "avg_rating": 0, "recent": [], "by_source": {}}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT COUNT(*) as c, COALESCE(AVG(rating), 0) as avg FROM product_reviews")
        stats = cur.fetchone()
        cur.execute("SELECT * FROM product_reviews ORDER BY created_at DESC LIMIT 20")
        recent = cur.fetchall()
        for r in recent:
            if r.get("created_at"): r["created_at"] = r["created_at"].isoformat()
        cur.execute("SELECT source, COUNT(*) as c FROM product_reviews GROUP BY source")
        by_source = {r["source"]: r["c"] for r in cur.fetchall()}
        cur.close(); db.close()
        return {"total": stats["c"], "avg_rating": round(float(stats["avg"]), 2), "recent": recent, "by_source": by_source}
    except Exception as e:
        try: db.close()
        except: pass
        return {"total": 0, "avg_rating": 0, "recent": [], "by_source": {}, "error": str(e)}


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
            r = await c.get(f"{EMBEDDING_SERVICE_URL}/health")
            qdrant_ok = r.status_code == 200
    except: pass
    return {"status": "ok" if (redis_ok or db_ok) else "degraded", "version": APP_VERSION, "redis": redis_ok, "database": db_ok, "qdrant": qdrant_ok, "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token", "dev_mode": DEV_MODE}

# ══════════════════════ FRONTEND ══════════════════════
@app.get("/dashboard-legacy", response_class=HTMLResponse)
async def index_legacy(request: Request):
    """Legacy dashboard — kept as fallback."""
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

    # Fetch cron results for dashboard tabs
    audit_data = {"issues": []}
    nouveautes_data = {"expired": [], "still_new": []}
    stock_data = {"oos_products": []}
    audit_last = nouveautes_last = stock_last = "Jamais"
    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur2.execute("""CREATE TABLE IF NOT EXISTS cron_results (
                id SERIAL PRIMARY KEY, cron_name TEXT, result_data JSONB,
                executed_at TIMESTAMP DEFAULT NOW())""")
            cur2.execute("SELECT result_data, executed_at FROM cron_results WHERE cron_name='audit-qualite' ORDER BY executed_at DESC LIMIT 1")
            row = cur2.fetchone()
            if row:
                audit_data = row["result_data"] if isinstance(row["result_data"], dict) else json.loads(row["result_data"])
                audit_last = row["executed_at"].strftime('%d/%m %H:%M') if row["executed_at"] else "?"
            cur2.execute("SELECT result_data, executed_at FROM cron_results WHERE cron_name='nouveautes-expire' ORDER BY executed_at DESC LIMIT 1")
            row = cur2.fetchone()
            if row:
                nouveautes_data = row["result_data"] if isinstance(row["result_data"], dict) else json.loads(row["result_data"])
                nouveautes_last = row["executed_at"].strftime('%d/%m %H:%M') if row["executed_at"] else "?"
            cur2.execute("SELECT result_data, executed_at FROM cron_results WHERE cron_name='stock-check' ORDER BY executed_at DESC LIMIT 1")
            row = cur2.fetchone()
            if row:
                stock_data = row["result_data"] if isinstance(row["result_data"], dict) else json.loads(row["result_data"])
                stock_last = row["executed_at"].strftime('%d/%m %H:%M') if row["executed_at"] else "?"
            cur2.close(); db2.close()
        except Exception as e:
            try: db2.close()
            except: pass
            logger.warning(f"Cron results fetch: {e}")

    audit_issues = audit_data.get("issues", [])
    still_new = nouveautes_data.get("still_new", [])
    oos_products_list = stock_data.get("oos_products", [])

    # Pre-build HTML rows for cron tabs (f-strings can't have backslashes)
    audit_rows_html = ""
    for ai in audit_issues[:30]:
        issues_text = "<br>".join(ai.get("issues", []))
        audit_rows_html += f'<tr><td>{ai.get("vendor","")}</td><td>{ai.get("title","")}</td><td style="color:#e53935;font-size:12px">{issues_text}</td></tr>'
    if not audit_rows_html:
        audit_rows_html = '<tr><td colspan="3" class="empty">Aucun probl&egrave;me d&eacute;tect&eacute; &#127881;</td></tr>'

    nouveautes_rows_html = ""
    for sn in still_new:
        dl = sn.get("days_left", 0)
        color = "#e53935" if dl <= 5 else "#4CAF50"
        nouveautes_rows_html += f'<tr><td>{sn.get("title","")}</td><td>{sn.get("created","")}</td><td style="color:{color};font-weight:bold">{dl}j</td></tr>'
    if not nouveautes_rows_html:
        nouveautes_rows_html = '<tr><td colspan="3" class="empty">Aucune nouveaut&eacute; en cours</td></tr>'

    stock_rows_html = ""
    for sp in oos_products_list[:30]:
        bis_n = sp.get("bis_subscribers", 0)
        stock_rows_html += f'<tr><td>{sp.get("vendor","")}</td><td>{sp.get("title","")}</td><td style="color:#e53935;font-weight:bold">0</td><td>{bis_n} en attente</td></tr>'
    if not stock_rows_html:
        stock_rows_html = '<tr><td colspan="4" class="empty">Tout est en stock &#127881;</td></tr>'

    oos_border_color = "#e53935" if len(oos_products_list) > 0 else "#4CAF50"
    oos_badge_bg = "#e53935" if len(oos_products_list) > 0 else "#4CAF50"

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

    # Try Me data from PostgreSQL
    tryme_total = 0
    tryme_pending = 0
    tryme_used = 0
    tryme_conv_rate = 0
    tryme_rows_html = ""
    db_tryme = get_db()
    if db_tryme:
        try:
            cur_t = db_tryme.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur_t.execute("SELECT COUNT(*) as c FROM tryme_purchases")
            tryme_total = cur_t.fetchone()["c"]
            cur_t.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='pending'")
            tryme_pending = cur_t.fetchone()["c"]
            cur_t.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE discount_used=true")
            tryme_used = cur_t.fetchone()["c"]
            tryme_conv_rate = round(tryme_used / tryme_total * 100, 1) if tryme_total > 0 else 0
            cur_t.execute("""SELECT customer_email, product_title, discount_code, status, purchased_at
                            FROM tryme_purchases ORDER BY purchased_at DESC LIMIT 15""")
            for r in cur_t.fetchall():
                dt = str(r.get("purchased_at", ""))[:16].replace("T", " ")
                st = r.get("status", "pending")
                st_color = "#4CAF50" if st == "pending" else "#C4956A" if st == "used" else "#999"
                st_label = "En attente" if st == "pending" else "Utilis\u00e9" if st == "used" else "Expir\u00e9"
                tryme_rows_html += f'<tr><td>{r.get("customer_email","")}</td><td>{r.get("product_title","")[:40]}</td><td><code>{r.get("discount_code","")}</code></td><td><span class="status-badge" style="background:{st_color}">{st_label}</span></td></tr>'
            if not tryme_rows_html:
                tryme_rows_html = '<tr><td colspan="4" class="empty">Aucun Try Me encore — lancement pr&eacute;vu dans 2 semaines</td></tr>'
            cur_t.close(); db_tryme.close()
        except Exception as e:
            try: db_tryme.close()
            except: pass
            logger.warning(f"Try Me dashboard: {e}")
            if not tryme_rows_html:
                tryme_rows_html = '<tr><td colspan="4" class="empty">Aucun Try Me encore</td></tr>'

    # Quiz analytics from Redis
    quiz_views_24h = 0
    quiz_completes_24h = 0
    quiz_atc_24h = 0
    quiz_conv_rate = 0
    quiz_product_count = 0
    quiz_last_regen = "Jamais"
    rc = get_redis()
    if rc:
        try:
            now_ts = int(time.time())
            rc.zremrangebyscore("pv:quiz-views", 0, now_ts - 86400)
            rc.zremrangebyscore("pv:quiz-completes", 0, now_ts - 86400)
            rc.zremrangebyscore("pv:quiz-atc", 0, now_ts - 86400)
            quiz_views_24h = rc.zcard("pv:quiz-views") or 0
            quiz_completes_24h = rc.zcard("pv:quiz-completes") or 0
            quiz_atc_24h = rc.zcard("pv:quiz-atc") or 0
            quiz_conv_rate = round(quiz_completes_24h / quiz_views_24h * 100) if quiz_views_24h > 0 else 0
            last_regen = rc.get("quiz:regen:last")
            if last_regen:
                quiz_last_regen = datetime.utcfromtimestamp(int(last_regen)).strftime('%d/%m %H:%M')
        except Exception as e:
            logger.warning(f"Quiz stats: {e}")

    # Count products in quiz index (check theme asset)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/themes.json",
                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
            )
            themes = resp.json().get("themes", [])
            main_theme = next((t for t in themes if t["role"] == "main"), None)
            if main_theme:
                asset_resp = await client.get(
                    f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/themes/{main_theme['id']}/assets.json?asset[key]=assets/quiz-data.json&fields=size",
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN}
                )
                quiz_product_count = 230  # Default, updated after regen
    except:
        pass

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>STELLA V8 — PlanèteBeauty</title>
<script src="https://cdn.shopify.com/shopifycloud/app-bridge.js?apiKey={api_key}"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#F9F7F4;color:#1A1A1A;padding:0}}

  /* ═══ HEADER ═══ */
  .app-header{{background:#1A1A1A;padding:18px 32px 0;display:flex;flex-direction:column}}
  .app-header__top{{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}}
  .app-header h1{{color:#D4AF37;font-size:20px;letter-spacing:2px;font-weight:800}}
  .app-header .subtitle{{color:#888;font-size:12px;margin-top:2px}}
  .app-header .refresh-btn{{background:#D4AF37;color:#fff;border:none;padding:8px 20px;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:.2s}}
  .app-header .refresh-btn:hover{{background:#b8862e}}

  /* ═══ TABS ═══ */
  .tabs{{display:flex;gap:0;overflow-x:auto;-webkit-overflow-scrolling:touch}}
  .tabs::-webkit-scrollbar{{display:none}}
  .tab{{padding:10px 20px;color:#888;font-size:13px;font-weight:600;cursor:pointer;border-bottom:3px solid transparent;transition:.2s;white-space:nowrap;display:flex;align-items:center;gap:6px}}
  .tab:hover{{color:#ccc}}
  .tab.active{{color:#D4AF37;border-bottom-color:#D4AF37}}
  .tab .tab-icon{{font-size:15px}}
  .tab .tab-badge{{background:#D4AF37;color:#fff;font-size:10px;padding:2px 7px;border-radius:10px;font-weight:700;min-width:18px;text-align:center}}
  .tab.coming .tab-badge{{background:#555;color:#888}}
  .tab.coming{{color:#555;cursor:default}}

  /* ═══ TAB CONTENT ═══ */
  .tab-content{{display:none}}
  .tab-content.active{{display:block}}
  .container{{max-width:1100px;margin:0 auto;padding:24px 20px}}

  /* ═══ CARDS ═══ */
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
  .card{{background:#fff;border-radius:14px;padding:24px 20px;box-shadow:0 2px 12px rgba(0,0,0,.04);text-align:center;transition:.2s}}
  .card:hover{{box-shadow:0 4px 20px rgba(0,0,0,.08)}}
  .card .num{{font-size:40px;font-weight:800;color:#D4AF37;line-height:1}}
  .card .label{{font-size:13px;color:#888;margin-top:8px;font-weight:500}}
  .card.ab{{border:2px solid #D4AF37;background:linear-gradient(135deg,#FFFDF7,#FFF8E7)}}

  /* ═══ TABLES ═══ */
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

  /* ═══ COMING SOON ═══ */
  .coming-soon{{text-align:center;padding:80px 20px}}
  .coming-soon .cs-icon{{font-size:48px;margin-bottom:16px}}
  .coming-soon h2{{font-size:20px;font-weight:700;color:#1A1A1A;margin-bottom:8px}}
  .coming-soon p{{font-size:14px;color:#888;max-width:400px;margin:0 auto}}

  .footer{{text-align:center;padding:20px;color:#bbb;font-size:12px}}

  @media(max-width:600px){{
    .app-header{{padding:14px 16px 0}}
    .tab{{padding:8px 14px;font-size:12px}}
    .cards{{grid-template-columns:1fr 1fr}}
    .card .num{{font-size:32px}}
    td,th{{padding:8px 10px;font-size:12px}}
    .td-count{{font-size:18px}}
  }}
</style>
</head>
<body>

<!-- ═══ HEADER + TABS ═══ -->
<div class="app-header">
  <div class="app-header__top">
    <div>
      <h1>STELLA V8</h1>
      <span class="subtitle">{now_str} UTC</span>
    </div>
    <button class="refresh-btn" onclick="location.reload()">Rafra&icirc;chir</button>
  </div>
  <nav class="tabs">
    <div class="tab active" data-tab="bis" onclick="switchTab(this)">
      <span class="tab-icon">&#128276;</span> Liste d'attente
      <span class="tab-badge">{total}</span>
    </div>
    <div class="tab" data-tab="audit" onclick="switchTab(this)">
      <span class="tab-icon">&#9989;</span> Audit Qualit&eacute;
      <span class="tab-badge">{len(audit_issues)}</span>
    </div>
    <div class="tab" data-tab="nouveautes" onclick="switchTab(this);loadNouveautes()">
      <span class="tab-icon">&#127381;</span> Nouveaut&eacute;s
      <span class="tab-badge" id="nouveautesBadge">...</span>
    </div>
    <div class="tab" data-tab="stock" onclick="switchTab(this)">
      <span class="tab-icon">&#128230;</span> Stock
      <span class="tab-badge" style="background:{oos_badge_bg}">{len(oos_products_list)}</span>
    </div>
    <div class="tab" data-tab="quiz" onclick="switchTab(this)">
      <span class="tab-icon">&#127919;</span> Quiz
      <span class="tab-badge" style="background:#C4956A">{quiz_views_24h}</span>
    </div>
    <div class="tab" data-tab="tryme" onclick="switchTab(this)">
      <span class="tab-icon">&#129514;</span> Try Me
      <span class="tab-badge" style="background:#C4956A">{tryme_total}</span>
    </div>
    <div class="tab" data-tab="promos" onclick="switchTab(this)">
      <span class="tab-icon">&#127915;</span> Codes Promo
      <span class="tab-badge" style="background:#C4956A" id="promosBadge">...</span>
    </div>
    <div class="tab" data-tab="cashback" onclick="switchTab(this)">
      <span class="tab-icon">&#128176;</span> Cashback
      <span class="tab-badge" style="background:#4CAF50" id="cashbackBadge">...</span>
    </div>
    <div class="tab coming" data-tab="images">
      <span class="tab-icon">&#127912;</span> Images
      <span class="tab-badge">Bient&ocirc;t</span>
    </div>
    <div class="tab coming" data-tab="settings">
      <span class="tab-icon">&#9881;&#65039;</span> R&eacute;glages
      <span class="tab-badge">Bient&ocirc;t</span>
    </div>
  </nav>
</div>

<!-- ═══ TAB: LISTE D'ATTENTE ═══ -->
<div class="tab-content active" id="tab-bis">
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
</div>

<!-- ═══ TAB: AUDIT QUALITÉ ═══ -->
<div class="tab-content" id="tab-audit">
<div class="container">
  <div class="cards">
    <div class="card"><div class="num">{len(audit_issues)}</div><div class="label">Produits avec probl&egrave;mes</div></div>
    <div class="card"><div class="num" style="font-size:14px;color:#888">{audit_last}</div><div class="label">Dernier audit</div></div>
  </div>
  <div class="section-title"><span class="icon">&#9989;</span> Produits &agrave; corriger</div>
  <table>
    <tr><th>Marque</th><th>Produit</th><th>Probl&egrave;mes</th></tr>
    {audit_rows_html}
  </table>
  <p style="text-align:center;margin-top:16px;color:#888;font-size:12px">Audit automatique chaque lundi &agrave; 8h</p>
</div>
</div>

<!-- ═══ TAB: NOUVEAUTÉS ═══ -->
<div class="tab-content" id="tab-nouveautes">
<div class="container">
  <div class="cards">
    <div class="card"><div class="num" id="nouveautesCount">...</div><div class="label">Nouveaut&eacute;s actives</div></div>
    <div class="card"><div class="num" style="font-size:14px;color:#4CAF50">Temps r&eacute;el</div><div class="label">Source: Shopify API</div></div>
  </div>
  <div class="section-title"><span class="icon">&#127381;</span> Produits en collection Nouveaut&eacute;s</div>
  <div id="nouveautes-table">
    <p style="text-align:center;color:#888">Chargement...</p>
  </div>
  <p style="text-align:center;margin-top:16px;color:#888;font-size:12px">Les produits sont retir&eacute;s automatiquement apr&egrave;s 30 jours</p>
</div>
</div>

<!-- ═══ TAB: STOCK ═══ -->
<div class="tab-content" id="tab-stock">
<div class="container">
  <div class="cards">
    <div class="card" style="border:2px solid {oos_border_color}"><div class="num" style="color:{oos_border_color}">{len(oos_products_list)}</div><div class="label">Produits en rupture</div></div>
    <div class="card"><div class="num" style="font-size:14px;color:#888">{stock_last}</div><div class="label">Derni&egrave;re v&eacute;rification</div></div>
  </div>
  <div class="section-title"><span class="icon">&#128230;</span> Produits en rupture de stock</div>
  <table>
    <tr><th>Marque</th><th>Produit</th><th>Stock</th><th>BIS</th></tr>
    {stock_rows_html}
  </table>
  <p style="text-align:center;margin-top:16px;color:#888;font-size:12px">V&eacute;rification automatique toutes les heures &middot; Email si nouveau produit en rupture</p>
</div>
</div>

<!-- ═══ TAB: TRY ME ═══ -->
<div class="tab-content" id="tab-tryme">
<div class="container">
  <div class="cards">
    <div class="card"><div class="num" style="color:#C4956A">{tryme_total}</div><div class="label">Total Try Me</div></div>
    <div class="card" style="border-color:#C4956A"><div class="num" style="color:#4CAF50">{tryme_pending}</div><div class="label">Codes en attente</div></div>
    <div class="card"><div class="num" style="color:#C4956A">{tryme_used}</div><div class="label">Codes utilis&eacute;s</div></div>
    <div class="card"><div class="num">{tryme_conv_rate}%</div><div class="label">Taux conversion</div></div>
  </div>
  <div class="section-title">&#128203; Codes r&eacute;cents</div>
  <table class="data-table">
    <tr><th>Email</th><th>Produit</th><th>Code</th><th>Statut</th></tr>
    {tryme_rows_html}
  </table>
  <div style="margin-top:16px;display:flex;gap:12px">
    <button onclick="fetch('/api/cron/tryme-expire',{{method:'POST'}}).then(function(){{location.reload()}})" style="padding:8px 16px;background:#1a1a1a;color:#fff;border:none;border-radius:8px;cursor:pointer;font-size:13px">&#128465; Nettoyer codes expir&eacute;s</button>
  </div>
</div>
</div>

<!-- ═══ TAB: IMAGES ═══ -->
<div class="tab-content" id="tab-images">
<div class="container">
  <div class="coming-soon">
    <div class="cs-icon">&#127912;</div>
    <h2>Images Produits</h2>
    <p>G&eacute;n&eacute;ration d'images AB Signature 2D/3D, pipeline WebP — bient&ocirc;t disponible.</p>
  </div>
</div>
</div>

<!-- ═══ TAB: QUIZ OLFACTIF ═══ -->
<div class="tab-content" id="tab-quiz">
<div class="container">
  <div class="cards">
    <div class="card"><div class="num" style="color:#C4956A">{quiz_views_24h}</div><div class="label">Vues quiz (24h)</div></div>
    <div class="card" style="border-color:#C4956A"><div class="num" style="color:#C4956A">{quiz_completes_24h}</div><div class="label">Quiz termin&eacute;s (24h)</div></div>
    <div class="card"><div class="num">{quiz_atc_24h}</div><div class="label">Ajouts panier (24h)</div></div>
    <div class="card"><div class="num" style="color:#C4956A">{quiz_conv_rate}%</div><div class="label">Taux de conversion</div></div>
  </div>

  <div class="section-title">&#128200; Actions</div>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px">
    <a href="https://planetebeauty.com/pages/trouvez-votre-parfum" target="_top" style="padding:10px 20px;background:#C4956A;color:#fff;border-radius:8px;text-decoration:none;font-size:14px;font-weight:600">&#128279; Voir le quiz live</a>
    <button onclick="regenQuiz()" id="regenBtn2" style="padding:10px 20px;background:#1a1a1a;color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer">&#128260; R&eacute;g&eacute;n&eacute;rer quiz-data.json</button>
  </div>

  <div class="section-title">&#128202; Infos</div>
  <table class="data-table">
    <tr><th>M&eacute;trique</th><th>Valeur</th></tr>
    <tr><td>Produits index&eacute;s</td><td><strong>{quiz_product_count}</strong></td></tr>
    <tr><td>Derni&egrave;re r&eacute;g&eacute;n&eacute;ration</td><td>{quiz_last_regen}</td></tr>
    <tr><td>Page URL</td><td><a href="https://planetebeauty.com/pages/trouvez-votre-parfum" target="_top">/pages/trouvez-votre-parfum</a></td></tr>
    <tr><td>Webhook auto-update</td><td><span class="status-badge" style="background:#4CAF50">Actif</span></td></tr>
  </table>
</div>
</div>

<!-- ═══ TAB: CODES PROMO ═══ -->
<div class="tab-content" id="tab-promos">
<div class="container">
  <div class="section-title">&#127915; Gestion des Codes Promo</div>
  <p style="color:#999;font-size:13px;margin-bottom:16px">Les codes sont appliqu&eacute;s via la Shopify Function <strong>order-discount-codes</strong>. Les modifications sont instantan&eacute;es.</p>

  <div id="promos-list" style="margin-bottom:24px">
    <div style="text-align:center;padding:24px;color:#666">Chargement...</div>
  </div>

  <div style="background:#1E1E1E;border:1px solid #333;border-radius:12px;padding:20px;margin-bottom:16px">
    <div class="section-title" style="margin-bottom:12px">&#10133; Ajouter / Modifier un code</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Code</label>
        <input id="promo-code" type="text" placeholder="PB580" style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">R&eacute;duction (%)</label>
        <input id="promo-pct" type="number" placeholder="5" min="1" max="100" style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Minimum (&euro;)</label>
        <input id="promo-min" type="number" placeholder="80" min="0" style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Expiration (vide = permanent)</label>
        <input id="promo-expires" type="date" style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
    </div>
    <div style="margin-top:16px;display:flex;gap:12px">
      <button onclick="savePromoCode()" style="padding:10px 24px;background:#C4956A;color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer">&#128190; Sauvegarder</button>
      <button onclick="clearPromoForm()" style="padding:10px 24px;background:#333;color:#fff;border:none;border-radius:8px;font-size:14px;cursor:pointer">Annuler</button>
    </div>
    <div id="promo-msg" style="margin-top:8px;font-size:13px"></div>
  </div>
</div>
</div>

<!-- ═══ TAB: CASHBACK ═══ -->
<div class="tab-content" id="tab-cashback">
<div class="container">
  <div class="section-title">&#128176; Cashback Fid&eacute;lit&eacute;</div>
  <p style="color:#999;font-size:13px;margin-bottom:16px">Chaque commande g&eacute;n&egrave;re du <strong>cr&eacute;dit magasin</strong> automatiquement. Configurez le taux, les exclusions et les conditions.</p>

  <!-- Settings Form -->
  <div style="background:#1E1E1E;border:1px solid #333;border-radius:12px;padding:20px;margin-bottom:24px">
    <div class="section-title" style="color:#D4AF37;margin-bottom:16px">&#9881; Configuration</div>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Taux cashback (%)</label>
        <input id="cb-rate" type="number" min="1" max="50" step="0.5" value="5"
               style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Expiration (jours)</label>
        <input id="cb-expiry" type="number" min="1" max="365" value="60"
               style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
      <div>
        <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Minimum utilisation (&euro;)</label>
        <input id="cb-min-use" type="number" min="0" step="5" value="70"
               style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
      <label style="display:flex;align-items:center;gap:8px;color:#ccc;font-size:13px;cursor:pointer">
        <input id="cb-excl-shipping" type="checkbox" checked> Exclure frais de port
      </label>
      <label style="display:flex;align-items:center;gap:8px;color:#ccc;font-size:13px;cursor:pointer">
        <input id="cb-excl-taxes" type="checkbox" checked> Exclure taxes
      </label>
      <label style="display:flex;align-items:center;gap:8px;color:#ccc;font-size:13px;cursor:pointer">
        <input id="cb-excl-discounts" type="checkbox" checked> Exclure r&eacute;ductions
      </label>
    </div>
    <div style="margin-bottom:16px">
      <label style="color:#999;font-size:12px;display:block;margin-bottom:4px">Tags exclus (s&eacute;par&eacute;s par virgule)</label>
      <input id="cb-excl-tags" type="text" value="tryme,no-cashback" placeholder="tryme,no-cashback,gift"
             style="width:100%;padding:10px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;box-sizing:border-box">
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-top:16px">
      <label style="display:flex;align-items:center;gap:8px;color:#ccc;font-size:14px;font-weight:600;cursor:pointer">
        <input id="cb-active" type="checkbox" checked> Cashback actif
      </label>
      <div style="display:flex;gap:12px;align-items:center">
        <span id="cb-save-msg" style="font-size:13px"></span>
        <button onclick="saveCashbackSettings()"
                style="padding:10px 24px;background:#4CAF50;color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer">
          &#128190; Sauvegarder
        </button>
      </div>
    </div>
  </div>

  <!-- Dashboard Stats -->
  <div id="cashback-stats" style="margin-bottom:20px">
    <div class="cards">
      <div class="card"><div class="num" id="cb-total-rewarded">-</div><div class="label">Cr&eacute;dits g&eacute;n&eacute;r&eacute;s</div></div>
      <div class="card" style="border-color:#4CAF50"><div class="num" style="color:#4CAF50" id="cb-total-amount">-</div><div class="label">Total cashback (&euro;)</div></div>
      <div class="card"><div class="num" id="cb-total-used">-</div><div class="label">Cr&eacute;dits utilis&eacute;s</div></div>
      <div class="card"><div class="num" style="color:#C4956A" id="cb-conv-rate">-</div><div class="label">Taux utilisation</div></div>
    </div>
  </div>

  <div class="section-title">&#128203; Derniers cashbacks</div>
  <table class="data-table">
    <tr><th>Client</th><th>Commande</th><th>Cashback</th><th>Statut</th><th>Date</th></tr>
    <tbody id="cashback-rows">
      <tr><td colspan="5" class="empty">Chargement...</td></tr>
    </tbody>
  </table>
</div>
</div>

<!-- ═══ TAB: SETTINGS ═══ -->
<div class="tab-content" id="tab-settings">
<div class="container">
  <div class="coming-soon">
    <div class="cs-icon">&#9881;&#65039;</div>
    <h2>R&eacute;glages</h2>
    <p>Configuration SMTP, webhooks, tokens API et pr&eacute;f&eacute;rences — bient&ocirc;t disponible.</p>
  </div>
</div>
</div>

<div class="footer">STELLA V8 &middot; Plan&egrave;teBeauty &middot; Powered by Railway</div>

<script>
function switchTab(tab) {{
  if (tab.classList.contains('coming')) return;
  document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
  document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
  tab.classList.add('active');
  var target = document.getElementById('tab-' + tab.getAttribute('data-tab'));
  if (target) target.classList.add('active');
}}

  // ═══ NOUVEAUTÉS REAL-TIME ═══
  var nouveautesLoaded = false;
  window.loadNouveautes = function() {{
    if (nouveautesLoaded) return;
    fetch('/api/nouveautes').then(function(r){{ return r.json(); }}).then(function(d){{
      nouveautesLoaded = true;
      var products = d.products || [];
      var badge = document.getElementById('nouveautesBadge');
      var count = document.getElementById('nouveautesCount');
      if (badge) badge.textContent = products.length;
      if (count) count.textContent = products.length;
      var el = document.getElementById('nouveautes-table');
      if (!products.length) {{ el.innerHTML = '<p class="empty">Aucune nouveaut&eacute; en cours</p>'; return; }}
      var html = '<table><tr><th>Marque</th><th>Produit</th><th>Date</th><th>Jours restants</th><th>Statut</th></tr>';
      products.forEach(function(p){{
        var color = p.days_left <= 5 ? '#e53935' : '#4CAF50';
        var statusColor = p.status === 'ACTIVE' ? '#4CAF50' : '#E67E22';
        var statusText = p.status === 'ACTIVE' ? 'Actif' : 'Brouillon';
        html += '<tr><td>' + p.vendor + '</td><td><a href="https://planetebeauty.com/products/' + p.handle + '" target="_blank" style="color:#C4956A">' + p.title + '</a></td>';
        html += '<td>' + p.created + '</td><td style="color:' + color + ';font-weight:bold">' + p.days_left + 'j</td>';
        html += '<td style="color:' + statusColor + '">' + statusText + '</td></tr>';
      }});
      html += '</table>';
      el.innerHTML = html;
    }}).catch(function(){{ document.getElementById('nouveautes-table').innerHTML = '<p style="color:#e53935">Erreur de chargement</p>'; }});
  }};

  // ═══ PROMO CODES MANAGEMENT ═══
  var promoState = {{ codes: [], discountId: null }};

  function loadPromoCodes() {{
    fetch('/api/promo-codes').then(function(r){{ return r.json(); }}).then(function(d){{
      promoState.codes = d.codes || [];
      promoState.discountId = d.discountId || null;
      var badge = document.getElementById('promosBadge');
      if (badge) badge.textContent = promoState.codes.length;
      renderPromoCodes();
    }}).catch(function(){{ document.getElementById('promos-list').innerHTML = '<p style="color:#e53935">Erreur de chargement</p>'; }});
  }}

  function renderPromoCodes() {{
    var el = document.getElementById('promos-list');
    if (!promoState.codes.length) {{ el.innerHTML = '<p style="color:#666;text-align:center;padding:16px">Aucun code configur&eacute;</p>'; return; }}
    var html = '<table class="data-table"><tr><th>Code</th><th>R&eacute;duction</th><th>Minimum</th><th>Expiration</th><th>Actions</th></tr>';
    promoState.codes.forEach(function(c, i){{
      var exp = c.expiresAt ? c.expiresAt.split('T')[0] : '<span style="color:#4CAF50">Permanent</span>';
      var isExpired = c.expiresAt && new Date(c.expiresAt) < new Date();
      var expStyle = isExpired ? 'color:#e53935;font-weight:bold' : '';
      html += '<tr><td><strong>' + c.code + '</strong></td><td>' + c.percentage + '%</td><td>' + c.minimumAmount + '&euro;</td>';
      html += '<td style="' + expStyle + '">' + exp + (isExpired ? ' (expir&eacute;)' : '') + '</td>';
      html += '<td><button onclick="editPromo(' + i + ')" style="background:#333;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer;margin-right:4px">&#9998;</button>';
      html += '<button onclick="deletePromo(' + i + ')" style="background:#e53935;color:#fff;border:none;padding:4px 10px;border-radius:4px;cursor:pointer">&#128465;</button></td></tr>';
    }});
    html += '</table>';
    el.innerHTML = html;
  }}

  window.editPromo = function(idx) {{
    var c = promoState.codes[idx];
    document.getElementById('promo-code').value = c.code;
    document.getElementById('promo-pct').value = c.percentage;
    document.getElementById('promo-min').value = c.minimumAmount;
    document.getElementById('promo-expires').value = c.expiresAt ? c.expiresAt.split('T')[0] : '';
    document.getElementById('promo-code').dataset.editIndex = idx;
  }};

  window.deletePromo = function(idx) {{
    var code = promoState.codes[idx].code;
    var btn = event.target;
    if (btn.dataset.confirmed) {{
      promoState.codes.splice(idx, 1);
      pushPromoCodes('Code ' + code + ' supprim&eacute;');
      return;
    }}
    btn.textContent = 'Confirmer ?';
    btn.style.background = '#b71c1c';
    btn.dataset.confirmed = '1';
    setTimeout(function(){{ btn.textContent = '\\ud83d\\uddd1'; btn.style.background = '#e53935'; delete btn.dataset.confirmed; }}, 3000);
  }};

  window.clearPromoForm = function() {{
    document.getElementById('promo-code').value = '';
    document.getElementById('promo-pct').value = '';
    document.getElementById('promo-min').value = '';
    document.getElementById('promo-expires').value = '';
    delete document.getElementById('promo-code').dataset.editIndex;
    document.getElementById('promo-msg').innerHTML = '';
  }};

  window.savePromoCode = function() {{
    var code = document.getElementById('promo-code').value.trim().toUpperCase();
    var pct = parseFloat(document.getElementById('promo-pct').value);
    var min = parseFloat(document.getElementById('promo-min').value);
    var exp = document.getElementById('promo-expires').value;
    if (!code || !pct || isNaN(min)) {{ document.getElementById('promo-msg').innerHTML = '<span style="color:#e53935">Remplir code, % et minimum</span>'; return; }}
    var entry = {{ code: code, percentage: pct, minimumAmount: min }};
    if (exp) entry.expiresAt = exp + 'T23:59:59Z';
    var editIdx = document.getElementById('promo-code').dataset.editIndex;
    if (editIdx !== undefined) {{
      promoState.codes[parseInt(editIdx)] = entry;
    }} else {{
      var exists = promoState.codes.findIndex(function(c){{ return c.code === code; }});
      if (exists >= 0) promoState.codes[exists] = entry;
      else promoState.codes.push(entry);
    }}
    pushPromoCodes('Code ' + code + ' sauvegard&eacute;');
    clearPromoForm();
  }};

  function pushPromoCodes(successMsg) {{
    var msg = document.getElementById('promo-msg');
    msg.innerHTML = '<span style="color:#C4956A">Sauvegarde...</span>';
    fetch('/api/promo-codes', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ codes: promoState.codes, discountId: promoState.discountId }})
    }}).then(function(r){{ return r.json(); }}).then(function(d){{
      if (d.success) {{
        msg.innerHTML = '<span style="color:#4CAF50">&#10003; ' + successMsg + '</span>';
        renderPromoCodes();
        var badge = document.getElementById('promosBadge');
        if (badge) badge.textContent = promoState.codes.length;
      }} else {{
        msg.innerHTML = '<span style="color:#e53935">&#10007; ' + (d.error || 'Erreur') + '</span>';
      }}
      setTimeout(function(){{ msg.innerHTML = ''; }}, 4000);
    }}).catch(function(){{ msg.innerHTML = '<span style="color:#e53935">Erreur r&eacute;seau</span>'; }});
  }}

  // Load promo codes on tab switch
  var origSwitchTab = switchTab;
  switchTab = function(tab) {{
    if (tab.classList.contains('coming')) return;
    document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
    document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});
    tab.classList.add('active');
    var target = document.getElementById('tab-' + tab.getAttribute('data-tab'));
    if (target) target.classList.add('active');
    if (tab.getAttribute('data-tab') === 'promos' && !promoState.discountId) loadPromoCodes();
    if (tab.getAttribute('data-tab') === 'cashback') loadCashbackTab();
  }};

  var cbSettingsLoaded = false;
  function loadCashbackTab() {{
    if (!cbSettingsLoaded) {{
      fetch('/api/cashback/settings').then(function(r){{ return r.json(); }}).then(function(s){{
        cbSettingsLoaded = true;
        document.getElementById('cb-rate').value = (s.cashback_rate * 100).toFixed(1);
        document.getElementById('cb-expiry').value = s.expiry_days;
        document.getElementById('cb-min-use').value = s.min_order_use;
        document.getElementById('cb-excl-shipping').checked = s.exclude_shipping;
        document.getElementById('cb-excl-taxes').checked = s.exclude_taxes;
        document.getElementById('cb-excl-discounts').checked = s.exclude_discounts;
        document.getElementById('cb-excl-tags').value = s.excluded_tags || '';
        document.getElementById('cb-active').checked = s.is_active;
      }}).catch(function(e){{ console.error('Load cashback settings:', e); }});
    }}
    loadCashbackDashboard();
  }}

  var cashbackLoaded = false;
  function loadCashbackDashboard() {{
    if (cashbackLoaded) return;
    fetch('/api/cashback/dashboard').then(function(r){{ return r.json(); }}).then(function(d){{
      cashbackLoaded = true;
      document.getElementById('cb-total-rewarded').textContent = d.total_rewarded || 0;
      document.getElementById('cb-total-amount').textContent = (d.total_amount || 0).toFixed(2);
      document.getElementById('cb-total-used').textContent = d.total_used || 0;
      var rate = d.total_rewarded > 0 ? Math.round(d.total_used / d.total_rewarded * 100) : 0;
      document.getElementById('cb-conv-rate').textContent = rate + '%';
      var badge = document.getElementById('cashbackBadge');
      if (badge) badge.textContent = d.total_rewarded || 0;
      var rows = '';
      (d.recent || []).forEach(function(r){{
        var dt = (r.created_at || '').substring(0, 16).replace('T', ' ');
        var stColor = r.status === 'used' ? '#C4956A' : '#4CAF50';
        var stLabel = r.status === 'used' ? 'Utilis&eacute;' : 'Actif';
        rows += '<tr><td>' + (r.customer_email || '') + '</td><td>' + (r.order_name || '') + '</td>';
        rows += '<td style="font-weight:bold">' + (r.cashback_amount || 0) + '&euro;</td>';
        rows += '<td><span class="status-badge" style="background:' + stColor + '">' + stLabel + '</span></td>';
        rows += '<td style="color:#888;font-size:13px">' + dt + '</td></tr>';
      }});
      if (!rows) rows = '<tr><td colspan="5" class="empty">Aucun cashback encore</td></tr>';
      document.getElementById('cashback-rows').innerHTML = rows;
    }}).catch(function(){{ document.getElementById('cashback-rows').innerHTML = '<tr><td colspan="5" class="empty" style="color:#e53935">Erreur de chargement</td></tr>'; }})
  }}

  window.saveCashbackSettings = function() {{
    var msg = document.getElementById('cb-save-msg');
    msg.innerHTML = '<span style="color:#C4956A">Sauvegarde...</span>';
    var payload = {{
      cashback_rate: parseFloat(document.getElementById('cb-rate').value) / 100,
      expiry_days: parseInt(document.getElementById('cb-expiry').value),
      min_order_use: parseFloat(document.getElementById('cb-min-use').value),
      exclude_shipping: document.getElementById('cb-excl-shipping').checked,
      exclude_taxes: document.getElementById('cb-excl-taxes').checked,
      exclude_discounts: document.getElementById('cb-excl-discounts').checked,
      excluded_tags: document.getElementById('cb-excl-tags').value.trim(),
      excluded_product_ids: '',
      min_cashback_amount: 0.50,
      is_active: document.getElementById('cb-active').checked
    }};
    fetch('/api/cashback/settings', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }}).then(function(r){{ return r.json(); }}).then(function(d){{
      if (d.success) {{
        msg.innerHTML = '<span style="color:#4CAF50">&#10003; Sauvegard&eacute; + metafields sync</span>';
      }} else {{
        msg.innerHTML = '<span style="color:#e53935">&#10007; ' + (d.error || 'Erreur') + '</span>';
      }}
      setTimeout(function(){{ msg.innerHTML = ''; }}, 4000);
    }}).catch(function(){{ msg.innerHTML = '<span style="color:#e53935">Erreur r&eacute;seau</span>'; }});
  }};

  // Quiz regeneration
  window.regenQuiz = function() {{
    var btn = document.getElementById('regenBtn2');
    btn.disabled = true;
    btn.textContent = '⏳ Régénération...';
    fetch('/api/quiz/regenerate', {{ method: 'POST' }})
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        if (d.success) {{
          btn.textContent = '✅ ' + d.count + ' produits indexés !';
          setTimeout(function() {{ location.reload(); }}, 2000);
        }} else {{
          btn.textContent = '❌ ' + (d.error || 'Erreur');
          setTimeout(function() {{ btn.textContent = '🔄 Régénérer'; btn.disabled = false; }}, 3000);
        }}
      }})
      .catch(function() {{
        btn.textContent = '❌ Erreur réseau';
        setTimeout(function() {{ btn.textContent = '🔄 Régénérer'; btn.disabled = false; }}, 3000);
      }});
  }};
</script>
</body>
</html>"""
    return HTMLResponse(html)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """STELLA V8 — Dashboard Tour de Contrôle."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "dashboard_v8.html")
    try:
        with open(html_path, "r") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>Dashboard V8 not found</h1>", status_code=500)


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
