"""
STELLA V7 — Shopify Embedded App (Full Version)
- Persistent chat memory (PostgreSQL) across ALL layers
- File upload & document analysis  
- Conversation management
- Real-time Shopify data
- Auto-vectorization to Qdrant long-term memory
"""
import os, time, hmac, hashlib, base64, json, logging, uuid, re
from datetime import datetime, timedelta
from typing import Optional, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import httpx
import aiosmtplib
from tryme_cards import pregenerate_card_assets, generate_order_pdf, get_card_paths, generate_single_card_pdf, stamp_code_on_verso, CARDS_DIR
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

# Shared secret for Shopify Flow → Railway webhook calls (header X-Flow-Token)
FLOW_WEBHOOK_TOKEN = os.getenv("FLOW_WEBHOOK_TOKEN", "")
GOOGLE_REVIEW_URL = os.getenv("GOOGLE_REVIEW_URL", "https://g.page/r/CTmV1GGiJ1rvEBM/review")

# Google Ads API config
GADS_DEVELOPER_TOKEN = os.getenv("GADS_DEVELOPER_TOKEN", "")
GADS_CLIENT_ID = os.getenv("GADS_CLIENT_ID", "")
GADS_CLIENT_SECRET = os.getenv("GADS_CLIENT_SECRET", "")
GADS_REFRESH_TOKEN = os.getenv("GADS_REFRESH_TOKEN", "")
GADS_CUSTOMER_ID = os.getenv("GADS_CUSTOMER_ID", "")  # sans tirets: 9024900792

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
ALTER TABLE product_reviews ADD COLUMN IF NOT EXISTS order_number TEXT;

CREATE TABLE IF NOT EXISTS cron_results (
    id SERIAL PRIMARY KEY,
    cron_name TEXT,
    result_data JSONB,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- ══════ STELLA OS — Operations Management (07/04/2026) ══════

CREATE TABLE IF NOT EXISTS entity_registry (
    id SERIAL PRIMARY KEY,
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    country VARCHAR(3),
    currency VARCHAR(3) DEFAULT 'EUR',
    timezone VARCHAR(50) DEFAULT 'Europe/Paris',
    shopify_domain VARCHAR(200),
    platform VARCHAR(20) DEFAULT 'shopify',
    active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS operations (
    id SERIAL PRIMARY KEY,
    entity_code VARCHAR(20) DEFAULT 'bhtc_fr',
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_by VARCHAR(50) DEFAULT 'stella',
    description TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS operations_files (
    id SERIAL PRIMARY KEY,
    operation_id INT REFERENCES operations(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    state_before TEXT,
    state_after TEXT,
    rollback_action TEXT,
    rolled_back BOOLEAN DEFAULT FALSE,
    rolled_back_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS scheduled_actions (
    id SERIAL PRIMARY KEY,
    operation_id INT,
    entity_code VARCHAR(20) DEFAULT 'bhtc_fr',
    action_type VARCHAR(50) NOT NULL,
    scheduled_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    executed_at TIMESTAMPTZ,
    retry_count INT DEFAULT 0,
    max_retries INT DEFAULT 3,
    action_payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS site_state (
    id SERIAL PRIMARY KEY,
    entity_code VARCHAR(20) DEFAULT 'bhtc_fr',
    category VARCHAR(50) NOT NULL,
    key VARCHAR(200) NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    UNIQUE(entity_code, category, key)
);

CREATE TABLE IF NOT EXISTS ops_audit_log (
    id SERIAL PRIMARY KEY,
    entity_code VARCHAR(20) DEFAULT 'bhtc_fr',
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    actor VARCHAR(50) NOT NULL,
    action VARCHAR(100) NOT NULL,
    operation_id INT,
    severity VARCHAR(20) DEFAULT 'info',
    details JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS process_playbook (
    id SERIAL PRIMARY KEY,
    process_name VARCHAR(100) UNIQUE NOT NULL,
    process_type VARCHAR(50) NOT NULL,
    description TEXT,
    trigger_conditions TEXT,
    steps JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_operations_status ON operations(status);
CREATE INDEX IF NOT EXISTS idx_operations_entity ON operations(entity_code);
CREATE INDEX IF NOT EXISTS idx_scheduled_actions_status ON scheduled_actions(status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_site_state_entity ON site_state(entity_code, category);
CREATE INDEX IF NOT EXISTS idx_ops_audit_timestamp ON ops_audit_log(timestamp);
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
                                  headers={"X-API-Key": "stella-mem-2026-planetebeauty"})
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
            # Reviews migration: add order_number column
            for col_sql in [
                "ALTER TABLE product_reviews ADD COLUMN IF NOT EXISTS order_number TEXT",
                "ALTER TABLE cashback_rewards ADD COLUMN IF NOT EXISTS email_sent BOOLEAN DEFAULT FALSE",
                "ALTER TABLE cashback_rewards ADD COLUMN IF NOT EXISTS reminder_sent_at TIMESTAMP",
                "ALTER TABLE tryme_purchases ADD COLUMN IF NOT EXISTS tryme_type VARCHAR(20) DEFAULT 'classic'",
            ]:
                try: cur.execute(col_sql)
                except Exception: pass
            # Seed entity_registry
            cur.execute("""INSERT INTO entity_registry (code, name, country, currency, timezone, shopify_domain, platform)
                VALUES ('bhtc_fr', 'BHTC SARL', 'FR', 'EUR', 'Europe/Paris', 'planetemode.myshopify.com', 'shopify')
                ON CONFLICT (code) DO NOTHING""")
            # Seed process_playbook
            playbook_seed = [
                ("creation_produit", "catalogue", "Pipeline complet creation/enrichissement produit", "Benoit demande d'ajouter ou enrichir un produit",
                 json.dumps([
                    {"step": 1, "action": "Scraper site officiel marque (notes, parfumeur, annee, INCI)"},
                    {"step": 2, "action": "Scraper Fragrantica via Chrome MCP (note communaute X,XX/5)"},
                    {"step": 3, "action": "Remplir 32 metafields namespace parfum"},
                    {"step": 4, "action": "Description 1 paragraphe narratif (accroche 2-3 phrases, citation via metafield)"},
                    {"step": 5, "action": "SEO title max 70c + meta description max 155c"},
                    {"step": 6, "action": "Vendor MAJUSCULES + productType = concentration exacte"},
                    {"step": 7, "action": "Tags auto (Famille:X, Saison:X, Genre:X, Concentration:X, Occasion:X, Accord:X)"},
                    {"step": 8, "action": "Creer variante Try Me si applicable (5% arrondi sup)"},
                    {"step": 9, "action": "Verifier images (3 images carre 2000x2000 WebP alt SEO)"},
                    {"step": 10, "action": "Verifier affichage page produit"}
                 ])),
                ("creation_marque", "catalogue", "Pipeline ajout nouvelle marque au catalogue", "Benoit demande d'ajouter une marque",
                 json.dumps([
                    {"step": 1, "action": "Creer la collection marque"},
                    {"step": 2, "action": "Enrichir chaque produit (pipeline creation_produit 10 etapes)"},
                    {"step": 3, "action": "Creer un article de blog pour la marque"},
                    {"step": 4, "action": "Ajouter dans liste collections homepage"},
                    {"step": 5, "action": "Ajouter dans le menu navigation"},
                    {"step": 6, "action": "Mettre a jour le compteur Nos XX Marques homepage"},
                    {"step": 7, "action": "Mettre a jour la page collection (description, image)"}
                 ])),
                ("operation_promo", "marketing", "Lancement operation promotionnelle temporaire", "Benoit demande une promo/event temporaire",
                 json.dumps([
                    {"step": 1, "action": "POST /api/ops/create avec tous les fichiers a modifier (state_before)"},
                    {"step": 2, "action": "Executer les modifications (code promo, banniere, sections, metafield)"},
                    {"step": 3, "action": "POST /api/ops/schedule rollback a la date d'expiration"},
                    {"step": 4, "action": "POST /api/ops/state/update avec la promo active"},
                    {"step": 5, "action": "Verifier visuellement (homepage, page produit, panier)"}
                 ])),
                ("modification_theme", "technique", "Modification fichier theme Liquid/JSON/CSS", "Toute modification du theme",
                 json.dumps([
                    {"step": 1, "action": "Lire la carte des dependances (registre §5)"},
                    {"step": 2, "action": "Identifier les systemes impactes"},
                    {"step": 3, "action": "POST /api/ops/create avec state_before du fichier"},
                    {"step": 4, "action": "Faire la modification"},
                    {"step": 5, "action": "Tester ATC sur 3 produits (formValid=true, bisEmailInForm=false)"},
                    {"step": 6, "action": "Tester panier + checkout"},
                    {"step": 7, "action": "REVERT immediat si echec"}
                 ])),
                ("deploy_railway", "technique", "Deploiement code Railway", "Push sur branch main",
                 json.dumps([
                    {"step": 1, "action": "python3 -c 'import py_compile; py_compile.compile(\"main.py\", doraise=True)'"},
                    {"step": 2, "action": "git add + git commit + git push origin main"},
                    {"step": 3, "action": "Attendre deploy (45s) + verifier /api/ops/health"},
                    {"step": 4, "action": "POST /api/ops/state/update deploy commit"},
                    {"step": 5, "action": "Mettre a jour registre technique"}
                 ])),
                ("deploy_shopify_function", "technique", "Deploiement Shopify Function Rust/WASM", "Modification extension Rust",
                 json.dumps([
                    {"step": 1, "action": "cargo build --target wasm32-wasip1 (verifier compilation)"},
                    {"step": 2, "action": "shopify app deploy --force --no-release"},
                    {"step": 3, "action": "shopify app release --version=stella-v8-XX --allow-updates"},
                    {"step": 4, "action": "Verifier que la Function est ACTIVE dans Shopify Admin"},
                    {"step": 5, "action": "Tester le parcours client (ATC, panier, checkout, codes promo)"},
                    {"step": 6, "action": "Mettre a jour registre technique (version, IDs)"}
                 ])),
            ]
            for name, ptype, desc, trigger, steps in playbook_seed:
                cur.execute("""INSERT INTO process_playbook (process_name, process_type, description, trigger_conditions, steps)
                    VALUES (%s,%s,%s,%s,%s) ON CONFLICT (process_name) DO UPDATE SET steps=%s, updated_at=NOW()""",
                    (name, ptype, desc, trigger, steps, steps))
            db.commit(); cur.close(); db.close()
            logger.info("Chat tables ready + STELLA OS tables + entity_registry seeded")
        except Exception as e:
            logger.error(f"Migration: {e}")
            try: db.close()
            except: pass
    get_redis()
    load_shopify_token_from_db()

    # ══════ SCHEDULER 24/7 (tourne meme si le Mac est eteint) ══════
    scheduler.add_job(_run_cron, 'interval', hours=1, id='stock_check',
                      args=["stock-check", "/api/cron/stock-check"])
    # sync_tags + nouveautes_expire DISABLED 09/04/2026 — gestion manuelle des nouveautés par Benoit
    # scheduler.add_job(_run_cron, 'cron', hour=3, minute=15, id='sync_tags',
    #                   args=["sync-tags", "/api/cron/sync-tags"])
    # scheduler.add_job(_run_cron, 'cron', hour=3, minute=45, id='nouveautes_expire',
    #                   args=["nouveautes-expire", "/api/cron/nouveautes-expire"])
    scheduler.add_job(_run_cron, 'cron', day_of_week='mon', hour=7, minute=0, id='audit_qualite',
                      args=["audit-qualite", "/api/cron/audit-qualite"])
    scheduler.add_job(_run_cron, 'cron', hour=4, minute=30, id='tryme_expire',
                      args=["tryme-expire", "/api/cron/tryme-expire"])
    # trustpilot_scan DISABLED 06/04/2026 — migrated to Google Avis clients
    # scheduler.add_job(_run_cron, 'interval', hours=6, id='trustpilot_scan',
    #                   args=["trustpilot-scan", "/api/cron/trustpilot-scan"])
    scheduler.add_job(_run_cron, 'cron', hour=9, minute=15, id='cashback_reminder',
                      args=["cashback-reminder", "/api/cron/cashback-reminder"])
    scheduler.add_job(_run_cron, 'interval', minutes=30, id='google_reviews_poll',
                      args=["google-reviews-poll", "/api/cron/google-reviews-poll"])
    # ══════ STELLA OS — Operations Management crons ══════
    scheduler.add_job(_run_cron, 'interval', minutes=5, id='action_executor',
                      args=["action-executor", "/api/cron/action-executor"])
    scheduler.add_job(_run_cron, 'cron', hour=5, minute=0, id='drift_detector',
                      args=["drift-detector", "/api/cron/drift-detector"])
    scheduler.add_job(_run_cron, 'cron', hour=6, minute=0, id='daily_briefing',
                      args=["daily-briefing", "/api/cron/daily-briefing"])
    scheduler.add_job(_run_cron, 'cron', hour='*/6', minute=17, id='coherence_check',
                      args=["coherence-check", "/api/cron/coherence-check"])
    scheduler.add_job(_run_cron, 'cron', day_of_week='mon', hour=7, minute=33, id='memory_audit',
                      args=["memory-audit", "/api/cron/memory-audit"])
    scheduler.add_job(_run_cron, 'interval', hours=2, id='memory_consolidate',
                      args=["memory-consolidate", "/api/cron/memory-consolidate"])
    scheduler.add_job(_run_cron, 'cron', hour=8, minute=0, id='gads_budget_guard',
                      args=["gads-budget-guard", "/api/cron/gads-budget-guard"])
    scheduler.start()
    logger.info("Scheduler started: stock(1h), audit(lun 7h), tryme(4h30), cashback(9h15), action-executor(5min), drift(5h), briefing(6h), coherence(6h), memory-audit(lun), gads-guard(8h)")

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
    "read_products", "write_products",
    "read_orders", "write_orders",
    "read_customers", "write_customers",
    "read_inventory", "write_inventory",
    "read_content", "write_content",
    "read_themes", "write_themes",
    "read_discounts", "write_discounts",
    "read_price_rules",
    "read_analytics", "read_reports",
    "read_shipping", "read_locations",
    "read_fulfillments", "read_draft_orders",
    "read_files", "write_files",
    "read_publications",
    "read_metaobjects", "read_metaobject_definitions",
    "read_product_listings", "write_product_listings",
    "read_translations",
    "read_online_store_pages", "write_online_store_pages",
    "read_online_store_navigation", "read_locales",
    "read_markets",
    "read_script_tags", "write_script_tags",
    "read_custom_pixels", "write_custom_pixels",
    "read_customer_events",
    "read_legal_policies", "write_legal_policies",
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
    scopes = OAUTH_SCOPES
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
    limit = int(request.query_params.get("limit", "50"))
    offset = int(request.query_params.get("offset", "0"))
    category = request.query_params.get("category", "")
    if limit > 1000: limit = 1000
    db = get_db()
    if not db: return {"memories": [], "total": 0}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if category:
            cur.execute("SELECT COUNT(*) as c FROM claude_memories WHERE category=%s", (category,))
            total = cur.fetchone()["c"]
            cur.execute("SELECT * FROM claude_memories WHERE category=%s ORDER BY importance DESC, created_at DESC LIMIT %s OFFSET %s", (category, limit, offset))
        else:
            cur.execute("SELECT COUNT(*) as c FROM claude_memories")
            total = cur.fetchone()["c"]
            cur.execute("SELECT * FROM claude_memories ORDER BY importance DESC, created_at DESC LIMIT %s OFFSET %s", (limit, offset))
        rows = cur.fetchall(); cur.close(); db.close()
        return {"memories": [{**r, "created_at": r["created_at"].isoformat() if r["created_at"] else None} for r in rows], "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        try: db.close()
        except: pass
        return {"memories": [], "total": 0, "error": str(e)}

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

# ═══════════════════════════════════════════════════════════════════════════
# STELLA-MEM — Rapports IA métier (reçus depuis le daemon local)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/stella-mem/report")
async def receive_stella_mem_report(request: Request):
    """Receive a report from STELLA-MEM daemon (local → Railway)."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"status": "error", "message": "DB unavailable"}
    try:
        body = await request.json()
        cur = db.cursor()
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stella_mem_reports (
                id SERIAL PRIMARY KEY,
                report_type TEXT NOT NULL DEFAULT 'daily',
                report_date TEXT,
                score INTEGER DEFAULT 0,
                sessions_count INTEGER DEFAULT 0,
                total_actions INTEGER DEFAULT 0,
                understood JSONB DEFAULT '[]',
                not_understood JSONB DEFAULT '[]',
                problems JSONB DEFAULT '{}',
                suggestions JSONB DEFAULT '[]',
                lessons JSONB DEFAULT '[]',
                full_report JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            INSERT INTO stella_mem_reports
            (report_type, report_date, score, sessions_count, total_actions,
             understood, not_understood, problems, suggestions, full_report)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            body.get("type", "daily"),
            body.get("date", ""),
            body.get("score", 0),
            body.get("sessions_count", 0),
            body.get("total_actions", 0),
            json.dumps(body.get("understood", [])),
            json.dumps(body.get("not_understood", [])),
            json.dumps(body.get("problem_summary", {})),
            json.dumps(body.get("suggestions", [])),
            json.dumps(body),
        ))
        report_id = cur.fetchone()[0]
        db.commit(); cur.close(); db.close()
        log_activity("stella_mem", "report_received", f"Report #{report_id} type={body.get('type','?')}")
        return {"status": "ok", "report_id": report_id}
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "message": str(e)}


@app.get("/api/stella-mem/reports")
async def get_stella_mem_reports(request: Request, limit: int = 20):
    """Get STELLA-MEM reports for the dashboard."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"reports": []}
    try:
        cur = db.cursor()
        cur.execute("""
            SELECT id, report_type, report_date, score, sessions_count,
                   total_actions, understood, not_understood, problems,
                   suggestions, created_at
            FROM stella_mem_reports
            ORDER BY created_at DESC LIMIT %s
        """, (limit,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        # Serialize datetime
        for r in rows:
            if r.get("created_at"):
                r["created_at"] = r["created_at"].isoformat()
        cur.close(); db.close()
        return {"reports": rows}
    except Exception as e:
        try: db.close()
        except: pass
        return {"reports": [], "error": str(e)}


@app.get("/api/stella-mem/latest")
async def get_stella_mem_latest(request: Request):
    """Get the most recent STELLA-MEM report."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"report": None}
    try:
        cur = db.cursor()
        cur.execute("""
            SELECT id, report_type, report_date, score, sessions_count,
                   total_actions, understood, not_understood, problems,
                   suggestions, lessons, full_report, created_at
            FROM stella_mem_reports
            ORDER BY created_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            cur.close(); db.close()
            return {"report": None}
        cols = [d[0] for d in cur.description]
        report = dict(zip(cols, row))
        if report.get("created_at"):
            report["created_at"] = report["created_at"].isoformat()
        cur.close(); db.close()
        return {"report": report}
    except Exception as e:
        try: db.close()
        except: pass
        return {"report": None, "error": str(e)}


@app.get("/api/stella-mem/lessons")
async def get_stella_mem_lessons(request: Request):
    """Get latest lessons learned from STELLA-MEM."""
    verify_claude_key(request)
    db = get_db()
    if not db: return {"lessons": []}
    try:
        cur = db.cursor()
        cur.execute("""
            SELECT suggestions, not_understood, problems, score, report_date
            FROM stella_mem_reports
            ORDER BY created_at DESC LIMIT 5
        """)
        rows = cur.fetchall()
        all_suggestions = []
        all_problems = {}
        for row in rows:
            sugs = row[0] if isinstance(row[0], list) else []
            all_suggestions.extend(sugs)
            probs = row[2] if isinstance(row[2], dict) else {}
            for k, v in probs.items():
                if k not in all_problems:
                    all_problems[k] = v
        cur.close(); db.close()
        return {
            "suggestions": list(dict.fromkeys(all_suggestions)),
            "problems": all_problems,
            "reports_analyzed": len(rows),
        }
    except Exception as e:
        try: db.close()
        except: pass
        return {"lessons": [], "error": str(e)}


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
LOGO_PB_URL = "https://cdn.shopify.com/s/files/1/0491/5177/0773/files/Logo_PB_200725.png?v=1753038818"
TRYME_VARIANT_TITLE = "2 ml"

def is_tryme_item(line_item):
    """Check if a line item is a Try Me product (variant or legacy product)."""
    sku = (line_item.get("sku") or "").upper()
    title = (line_item.get("variant_title") or "").lower()
    name = (line_item.get("name") or "").lower()
    if sku.startswith(TRYME_SKU_PREFIX.upper()):
        return True
    if "try me" in title or "try le" in title or "try me" in name:
        return True
    return False


def tryme_short_name(product_title: str) -> str:
    """Extract short readable product name for Try Me code (ex: VANILLE, BOURBON)."""
    import unicodedata as _ud
    t = re.sub(r'(?i)\b(eau de parfum|extrait de parfum|le parfum|parfum|edp|edt)\b', '', product_title).strip()
    for brand in ["Les Mignardises by Jousset", "Jousset Parfums", "Plume Impression", "Silona Paris"]:
        t = re.sub(r'(?i)^' + re.escape(brand) + r'\s*', '', t).strip()
    words = [w for w in re.findall(r'[A-Za-zÀ-ÿ]+', t) if len(w) > 2]
    name = words[0][:8].upper() if words else "TRYME"
    name = ''.join(c for c in _ud.normalize('NFD', name) if _ud.category(c) != 'Mn')
    return name

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

    def _is_upsell_item(li):
        return any(p.get("name") == "_tryme_upsell" and p.get("value") == "true"
                   for p in (li.get("properties") or []))

    all_tryme_items = [li for li in line_items if is_tryme_item(li)]
    if not all_tryme_items:
        return {"ok": True, "skipped": True, "reason": "no_tryme_items"}

    # Séparer Try Me classiques (code TM- remise fixe sur produit) des upsells (code TU- -5% prochaine commande)
    tryme_items = [li for li in all_tryme_items if not _is_upsell_item(li)]

    # Redis lock to prevent duplicate webhook processing
    rc = get_redis()
    lock_key = f"tryme_lock:{order_id}"
    if rc:
        try:
            if not rc.set(lock_key, "1", nx=True, ex=120):
                logger.info(f"Try Me webhook already processing for order {order_id} — skipping duplicate")
                return {"ok": True, "skipped": True, "reason": "duplicate_webhook"}
        except Exception:
            pass

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

        # Generate unique discount code: TM-[PRODUIT]-[HEX4]
        short = tryme_short_name(product_title)
        code = f"TM-{short}-{uuid.uuid4().hex[:4].upper()}"
        now = datetime.utcnow()
        expires = now + timedelta(days=TRYME_EXPIRY_DAYS)

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
                        "productDiscounts": True,
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

        results.append({"product": product_title, "code": code, "email": customer_email,
                        "discount_gid": discount_gid, "product_id": product_id, "price": item_price})

    # Write Try Me codes as metafield on order (for Order Printer invoice)
    codes_for_invoice = [r for r in results if r.get("code")]
    if codes_for_invoice:
        try:
            # Read existing metafield to accumulate (don't overwrite)
            existing_text = ""
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r_mf = await c.post(SHOPIFY_GRAPHQL_URL,
                        json={"query": f'{{ order(id: "gid://shopify/Order/{order_id}") {{ metafield(namespace: "planete-beaute", key: "tryme-codes") {{ value }} }} }}'},
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
                    mf_data = r_mf.json().get("data", {}).get("order", {}).get("metafield")
                    if mf_data and mf_data.get("value"):
                        existing_text = mf_data["value"].strip()
            except Exception:
                pass
            # Format: "CODE | Produit\nCODE2 | Produit2"
            new_lines = [f"{r['code']} | -{float(r['price']):.0f}€ sur {r['product']}" for r in codes_for_invoice]
            # Merge: existing + new, deduplicate by code
            existing_codes = set()
            if existing_text:
                for line in existing_text.split("\n"):
                    code_part = line.split("|")[0].strip()
                    if code_part:
                        existing_codes.add(code_part)
            final_lines = existing_text.split("\n") if existing_text else []
            for line in new_lines:
                code_part = line.split("|")[0].strip()
                if code_part not in existing_codes:
                    final_lines.append(line)
                    existing_codes.add(code_part)
            tryme_text = "\n".join(final_lines)
            mf_mutation = """mutation($metafields: [MetafieldsSetInput!]!) {
              metafieldsSet(metafields: $metafields) {
                metafields { id } userErrors { field message }
              }
            }"""
            mf_vars = {"metafields": [{
                "ownerId": f"gid://shopify/Order/{order_id}",
                "namespace": "planete-beaute",
                "key": "tryme-codes",
                "type": "multi_line_text_field",
                "value": tryme_text,
            }]}
            async with httpx.AsyncClient(timeout=10) as c:
                await c.post(SHOPIFY_GRAPHQL_URL, json={"query": mf_mutation, "variables": mf_vars},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            logger.info(f"Try Me metafield written on order {order_id}: {tryme_text}")
        except Exception as e:
            logger.error(f"Try Me metafield write error: {e}")

    # Try Me Upsell : si un item a la propriété _tryme_upsell, générer un code -5% prochaine commande
    upsell_items = [li for li in all_tryme_items if _is_upsell_item(li)]
    if upsell_items and customer_email:
        try:
            upsell_code = f"TU-{uuid.uuid4().hex[:6].upper()}"
            now_u = datetime.utcnow()
            expires_u = now_u + timedelta(days=30)
            # Créer un code -5% valable sur TOUS les produits (format standard)
            # Exclusions : Try Me, cadeaux, etc. gérées par la Function Rust via variant title
            gql_upsell = """mutation discountCodeBasicCreate($basicCodeDiscount: DiscountCodeBasicInput!) {
              discountCodeBasicCreate(basicCodeDiscount: $basicCodeDiscount) {
                codeDiscountNode { id }
                userErrors { field message }
              }
            }"""
            upsell_vars = {
                "basicCodeDiscount": {
                    "title": f"Try Me Upsell -5% — {order_name}",
                    "code": upsell_code,
                    "startsAt": now_u.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "endsAt": expires_u.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "usageLimit": 1,
                    "appliesOncePerCustomer": True,
                    "customerSelection": {"customers": {"add": [f"gid://shopify/Customer/{customer_id}"]}} if customer_id else {"all": True},
                    "customerGets": {
                        "items": {"all": True},
                        "value": {"percentage": 0.05}
                    },
                    "combinesWith": {"orderDiscounts": True, "productDiscounts": True, "shippingDiscounts": True}
                }
            }
            async with httpx.AsyncClient(timeout=15) as c:
                r_u = await c.post(SHOPIFY_GRAPHQL_URL,
                    json={"query": gql_upsell, "variables": upsell_vars},
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
                u_data = r_u.json()
                u_errors = u_data.get("data", {}).get("discountCodeBasicCreate", {}).get("userErrors", [])
                if u_errors:
                    logger.error(f"Try Me Upsell code error: {u_errors}")
                else:
                    logger.info(f"Try Me Upsell code created: {upsell_code} for {customer_email}")
                    log_activity("tryme_upsell", f"Code upsell -5% {upsell_code} créé pour {customer_email} (commande {order_name})",
                                 {"code": upsell_code, "order": order_name, "customer": customer_email}, source="webhook")
                    # Stocker en base pour le dashboard
                    db_u = get_db()
                    if db_u:
                        try:
                            cur_u = db_u.cursor()
                            upsell_product = upsell_items[0]
                            cur_u.execute("""INSERT INTO tryme_purchases
                                (customer_email, customer_id, product_id, product_handle, product_title, variant_id,
                                 order_id, order_name, tryme_price, discount_code, discount_expires_at, status, tryme_type)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending','upsell')""",
                                (customer_email, customer_id,
                                 str(upsell_product.get("product_id", "")),
                                 upsell_product.get("handle", ""),
                                 upsell_product.get("title", ""),
                                 str(upsell_product.get("variant_id", "")),
                                 order_id, order_name, 0, upsell_code,
                                 expires_u.strftime("%Y-%m-%dT%H:%M:%SZ")))
                            db_u.commit(); cur_u.close(); db_u.close()
                        except Exception as e:
                            logger.error(f"Try Me Upsell DB insert error: {e}")
                            try: db_u.close()
                            except: pass
                    # Écrire le code upsell dans le metafield tryme-codes (pour Order Printer facture)
                    try:
                        upsell_line = f"{upsell_code} | -5% sur votre prochaine commande (Try Me Découverte)"
                        # Lire existant et accumuler
                        existing_mf = ""
                        async with httpx.AsyncClient(timeout=10) as c_mf:
                            r_mf2 = await c_mf.post(SHOPIFY_GRAPHQL_URL,
                                json={"query": f'{{ order(id: "gid://shopify/Order/{order_id}") {{ metafield(namespace: "planete-beaute", key: "tryme-codes") {{ value }} }} }}'},
                                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
                            mf2 = r_mf2.json().get("data", {}).get("order", {}).get("metafield")
                            if mf2 and mf2.get("value"):
                                existing_mf = mf2["value"].strip()
                        final_mf = (existing_mf + "\n" + upsell_line).strip() if existing_mf else upsell_line
                        mf_mut = """mutation($mf: [MetafieldsSetInput!]!) { metafieldsSet(metafields: $mf) { metafields { id } userErrors { message } } }"""
                        await httpx.AsyncClient(timeout=10).post(SHOPIFY_GRAPHQL_URL,
                            json={"query": mf_mut, "variables": {"mf": [{"ownerId": f"gid://shopify/Order/{order_id}", "namespace": "planete-beaute", "key": "tryme-codes", "type": "multi_line_text_field", "value": final_mf}]}},
                            headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
                        logger.info(f"Try Me Upsell code written to order metafield: {upsell_code}")
                    except Exception as e:
                        logger.error(f"Try Me Upsell metafield write error: {e}")
                    # Email au client
                    try:
                        html_upsell = f"""<div style="font-family:Georgia,serif;max-width:520px;margin:0 auto;padding:32px 20px;background:#FAF7F2">
                        <h2 style="text-align:center;color:#1a1a1a;font-weight:300;font-style:italic">Merci pour votre Try Me !</h2>
                        <p style="color:#666;text-align:center">En remerciement, voici votre code de r&eacute;duction de 5% valable sur votre prochaine commande :</p>
                        <div style="text-align:center;margin:24px 0">
                          <span style="display:inline-block;border:2px dashed #C4956A;padding:12px 32px;font-size:20px;font-weight:700;letter-spacing:3px;color:#1a1a1a">{upsell_code}</span>
                        </div>
                        <p style="color:#888;font-size:13px;text-align:center">Valable 30 jours sur tous les parfums format standard<br>Cumulable avec vos autres codes promo</p>
                        <p style="color:#aaa;font-size:11px;text-align:center;margin-top:24px">Planètebeauty · Parfumerie de niche</p>
                        </div>"""
                        from email.mime.text import MIMEText
                        from email.mime.multipart import MIMEMultipart
                        msg = MIMEMultipart("alternative")
                        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
                        msg["To"] = customer_email
                        msg["Subject"] = f"Votre code -5% Try Me — {upsell_code}"
                        msg.attach(MIMEText(html_upsell, "html"))
                        await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                                              username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
                        logger.info(f"Try Me Upsell email sent to {customer_email}: {upsell_code}")
                    except Exception as e:
                        logger.error(f"Try Me Upsell email error: {e}")
        except Exception as e:
            logger.error(f"Try Me Upsell error: {e}")

    # Generate PDF card for printing
    if results:
        try:
            cards_data = [{"product_id": r["product_id"], "code": r["code"], "order_name": order_name}
                          for r in results if r.get("code")]
            if cards_data:
                CARDS_DIR.mkdir(parents=True, exist_ok=True)
                pdf_path = str(CARDS_DIR / f"order_{order_id}.pdf")
                ok = generate_order_pdf(cards_data, pdf_path)
                if ok:
                    logger.info(f"Try Me card PDF generated: {pdf_path}")
                    log_activity("tryme_card", f"PDF carte généré pour {order_name} ({len(cards_data)} cartes)",
                                 {"order_id": order_id, "cards": len(cards_data)}, source="webhook")
        except Exception as e:
            logger.error(f"Try Me card PDF error: {e}")

    return {"ok": True, "tryme_count": len(results), "results": results}


# ── Try Me Card Generation Helpers ──

async def _pregenerate_tryme_card(product_id: str, product_title: str, product_handle: str):
    """Fetch metafields and pre-generate card assets for a product with Try Me variant."""
    try:
        query = f"""{{ product(id: "gid://shopify/Product/{product_id}") {{
            metafield_tete: metafield(namespace: "parfum", key: "note_tete_principale") {{ value }}
            metafield_coeur: metafield(namespace: "parfum", key: "note_coeur_principale") {{ value }}
            metafield_fond: metafield(namespace: "parfum", key: "note_fond_principale") {{ value }}
            metafield_tete_sec: metafield(namespace: "parfum", key: "notes_tete_secondaires") {{ value }}
            metafield_coeur_sec: metafield(namespace: "parfum", key: "notes_coeur_secondaires") {{ value }}
            metafield_fond_sec: metafield(namespace: "parfum", key: "notes_fond_secondaires") {{ value }}
            img_tete: metafield(namespace: "parfum", key: "image_note_tete") {{ reference {{ ... on MediaImage {{ image {{ url }} }} }} }}
            img_coeur: metafield(namespace: "parfum", key: "image_note_coeur") {{ reference {{ ... on MediaImage {{ image {{ url }} }} }} }}
            img_fond: metafield(namespace: "parfum", key: "image_note_fond") {{ reference {{ ... on MediaImage {{ image {{ url }} }} }} }}
            mf_famille: metafield(namespace: "parfum", key: "famille_olfactive") {{ value }}
            mf_accord: metafield(namespace: "parfum", key: "accord_principal") {{ value }}
            mf_accords_sec: metafield(namespace: "parfum", key: "accords_secondaires") {{ value }}
            mf_intensite: metafield(namespace: "parfum", key: "intensite") {{ value }}
            mf_sillage: metafield(namespace: "parfum", key: "sillage") {{ value }}
            mf_sillage_level: metafield(namespace: "parfum", key: "sillage_level") {{ value }}
            mf_tenacite: metafield(namespace: "parfum", key: "tenacite") {{ value }}
            mf_duree: metafield(namespace: "parfum", key: "duree_tenue_heures") {{ value }}
            mf_saison: metafield(namespace: "parfum", key: "saison") {{ value }}
            mf_genre: metafield(namespace: "parfum", key: "genre") {{ value }}
            mf_occasions: metafield(namespace: "parfum", key: "occasions") {{ value }}
            featuredImage {{ url }}
        }} }}"""
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": query},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            data = r.json().get("data", {}).get("product", {})

        if not data:
            logger.warning(f"No product data for card generation: {product_id}")
            return

        def _parse_list(val):
            if not val: return []
            try: return json.loads(val)
            except: return []

        notes = {
            "tete": (data.get("metafield_tete") or {}).get("value", "—"),
            "coeur": (data.get("metafield_coeur") or {}).get("value", "—"),
            "fond": (data.get("metafield_fond") or {}).get("value", "—"),
            "tete_sec": _parse_list((data.get("metafield_tete_sec") or {}).get("value")),
            "coeur_sec": _parse_list((data.get("metafield_coeur_sec") or {}).get("value")),
            "fond_sec": _parse_list((data.get("metafield_fond_sec") or {}).get("value")),
        }
        note_image_urls = {
            "tete": ((data.get("img_tete") or {}).get("reference") or {}).get("image", {}).get("url"),
            "coeur": ((data.get("img_coeur") or {}).get("reference") or {}).get("image", {}).get("url"),
            "fond": ((data.get("img_fond") or {}).get("reference") or {}).get("image", {}).get("url"),
        }

        product_image_url = (data.get("featuredImage") or {}).get("url")

        # Extra metadata for card design
        def _val(key):
            return (data.get(key) or {}).get("value", "")
        metadata = {
            "famille": _parse_list(_val("mf_famille")),
            "accord": _val("mf_accord"),
            "accords_sec": _parse_list(_val("mf_accords_sec")),
            "intensite": int(_val("mf_intensite") or "3"),
            "sillage": _val("mf_sillage"),
            "sillage_level": int(_val("mf_sillage_level") or "2"),
            "tenacite": _val("mf_tenacite"),
            "duree_tenue": int(_val("mf_duree") or "0"),
            "saison": _parse_list(_val("mf_saison")),
            "genre": _val("mf_genre"),
            "occasions": _parse_list(_val("mf_occasions")),
        }

        result = await pregenerate_card_assets(product_id, product_title, product_handle,
                                                notes, note_image_urls, LOGO_PB_URL,
                                                product_image_url=product_image_url,
                                                metadata=metadata)
        logger.info(f"Card assets generated for {product_title}: {result}")
        log_activity("tryme_card_gen", f"Visuels carte pré-générés : {product_title}",
                     {"product_id": product_id}, source="auto")
    except Exception as e:
        logger.error(f"Card pregeneration error for {product_id}: {e}")


@app.post("/api/tryme/pregenerate-all")
async def tryme_pregenerate_all(request: Request):
    """Pre-generate card assets for ALL products with Try Me variants."""
    import asyncio
    query = """{ products(first: 100, query: "variant_title:*Try*") {
        edges { node { id title handle variants(first: 10) {
            edges { node { title } }
        } } }
    } }"""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": query},
                       headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
        products = r.json().get("data", {}).get("products", {}).get("edges", [])

    count = 0
    for edge in products:
        p = edge["node"]
        pid = p["id"].split("/")[-1]
        has_tryme = any("try me" in (v["node"]["title"] or "").lower() for v in p["variants"]["edges"])
        if has_tryme:
            await _pregenerate_tryme_card(pid, p["title"], p["handle"])
            count += 1

    return {"ok": True, "generated": count}


@app.get("/api/tryme/card-pdf/{order_id}")
async def tryme_card_pdf(order_id: str):
    """Download card PDF for an order."""
    from starlette.responses import FileResponse
    pdf_path = CARDS_DIR / f"order_{order_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found for this order")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                       filename=f"tryme-cards-{order_id}.pdf")


@app.get("/api/tryme/card-preview/{product_id}")
async def tryme_card_preview(product_id: str, side: str = "recto"):
    """Preview recto or verso card for a product."""
    from starlette.responses import FileResponse
    if side == "verso":
        path = CARDS_DIR / f"verso_{product_id}.png"
    else:
        path = CARDS_DIR / f"recto_{product_id}.png"
    if not path.exists():
        raise HTTPException(404, f"Card {side} not pre-generated for product {product_id}")
    return FileResponse(str(path), media_type="image/png")


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
async def tryme_dashboard(request: Request = None):
    """Dashboard data for Try Me tab — codes + KPIs + pre-generated cards list."""
    result = {"total_codes": 0, "active_codes": 0, "used_codes": 0, "expired_codes": 0,
              "recent_codes": [], "pregenerated_cards": []}

    # DB data
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COUNT(*) as c FROM tryme_purchases")
            result["total_codes"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='pending'")
            result["active_codes"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='used' OR discount_used=true")
            result["used_codes"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='expired'")
            result["expired_codes"] = cur.fetchone()["c"]

            cur.execute("""SELECT customer_email, product_title, product_id, discount_code,
                           tryme_price, status, order_id, order_name,
                           purchased_at, discount_expires_at,
                           COALESCE(tryme_type, 'classic') as tryme_type
                           FROM tryme_purchases ORDER BY purchased_at DESC LIMIT 30""")
            recent = cur.fetchall()
            for r in recent:
                if r.get("purchased_at"): r["purchased_at"] = r["purchased_at"].isoformat()
                if r.get("discount_expires_at"): r["discount_expires_at"] = r["discount_expires_at"].isoformat()
            result["recent_codes"] = recent
            cur.close(); db.close()
        except Exception as e:
            logger.error(f"Try Me dashboard DB error: {e}")
            try: db.close()
            except: pass

    # Pre-generated cards list
    try:
        CARDS_DIR.mkdir(parents=True, exist_ok=True)
        recto_files = sorted(CARDS_DIR.glob("recto_*.png"))
        cards_list = []
        for f in recto_files:
            pid = f.stem.replace("recto_", "")
            cards_list.append({"product_id": pid, "title": pid})
        # Enrich with product titles from Shopify
        if cards_list:
            ids = [f'"gid://shopify/Product/{c["product_id"]}"' for c in cards_list[:20]]
            query = f"""{{ nodes(ids: [{','.join(ids)}]) {{ ... on Product {{ id title }} }} }}"""
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": query},
                                   headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
                    nodes = r.json().get("data", {}).get("nodes", [])
                    title_map = {n["id"].split("/")[-1]: n.get("title", "") for n in nodes if n}
                    for card in cards_list:
                        card["title"] = title_map.get(card["product_id"], card["product_id"])
            except Exception:
                pass
        result["pregenerated_cards"] = cards_list
    except Exception as e:
        logger.error(f"Try Me cards list error: {e}")

    return result


@app.delete("/api/tryme/{discount_code}")
async def tryme_delete_entry(discount_code: str, request: Request = None):
    """Admin: delete a Try Me entry from PostgreSQL by discount_code."""
    if request:
        api_key = request.headers.get("X-API-Key", "")
        if api_key != "stella-mem-2026-planetebeauty":
            raise HTTPException(401, "Unauthorized")
    db = get_db()
    if not db:
        raise HTTPException(500, "DB unavailable")
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM tryme_purchases WHERE discount_code=%s RETURNING id", (discount_code,))
        deleted = cur.fetchone()
        db.commit()
        cur.close(); db.close()
        if deleted:
            log_activity("tryme_admin", f"Entrée Try Me {discount_code} supprimée manuellement", {"code": discount_code}, source="admin")
            return {"ok": True, "deleted": discount_code}
        return {"ok": False, "reason": "not_found"}
    except Exception as e:
        try: db.close()
        except: pass
        raise HTTPException(500, str(e))


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
    "épicé": "Épicé", "épicée": "Épicé",
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
        seasons_found = set()
        for s in items:
            s_clean = s.strip()
            if s_clean.lower() in ("toutes saisons", "toutes"):
                new_tags.add("Saison:Toutes saisons")
                seasons_found = {"Printemps", "Été", "Automne", "Hiver"}
            else:
                for part in s_clean.split("/"):
                    part = part.strip()
                    if part in ("Printemps", "Été", "Automne", "Hiver"):
                        new_tags.add(f"Saison:{part}")
                        seasons_found.add(part)
        # Si les 4 saisons sont presentes, ajouter "Toutes saisons"
        if len(seasons_found) >= 4:
            new_tags.add("Saison:Toutes saisons")

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

    # Accord principal + secondaires
    acc = product_data.get("accord")
    if acc and acc.get("value"):
        val = acc["value"].strip()
        if val: new_tags.add(f"Accord:{val}")
    acc2 = product_data.get("accords_sec")
    if acc2 and acc2.get("value"):
        val2 = acc2["value"]
        items2 = json.loads(val2) if val2.startswith("[") else [x.strip() for x in val2.split(",")]
        for a in items2:
            a = a.strip()
            if a: new_tags.add(f"Accord:{a}")

    # Nouveauté — DISABLED 09/04/2026: gestion manuelle par Benoit
    # if product_data.get("createdAt", "") > cutoff_iso:
    #     new_tags.add("Nouveauté")

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
    AUTO_PREFIXES = ("Famille:", "Saison:", "Genre:", "Concentration:", "Occasion:", "Accord:")

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
                accord: metafield(namespace: "parfum", key: "accord_principal") {{ value }}
                accords_sec: metafield(namespace: "parfum", key: "accords_secondaires") {{ value }}
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
                # Exclure les produits non-parfum de l'audit parfumerie
                title_lower = (p.get("title") or "").lower()
                skip_keywords = ("coffret", "cadeau", "mystère", "mystere", "échantillon", "echantillon", "gift", "card")
                if any(kw in title_lower for kw in skip_keywords):
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


# ══════════════════════ CRON: CASHBACK REMINDER ══════════════════════

@app.post("/api/cron/cashback-reminder")
async def cron_cashback_reminder(request: Request):
    """Daily 9h15: send reminder email 2 days before cashback expiration."""
    db = get_db()
    if not db:
        return {"ok": False, "error": "no db"}
    sent = 0
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Find active cashback expiring within 2 days, not yet reminded
        cur.execute("""SELECT id, customer_email, order_name, cashback_amount, expires_at
                      FROM cashback_rewards
                      WHERE status = 'active'
                      AND reminder_sent_at IS NULL
                      AND expires_at > NOW()
                      AND expires_at < NOW() + INTERVAL '2 days'""")
        rows = cur.fetchall()
        cur.close(); db.close()
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"Cashback reminder query: {e}")
        return {"ok": False, "error": str(e)}

    for row in rows:
        email = row["customer_email"]
        if not email:
            continue
        ok = await _send_cashback_reminder_email(
            email, float(row["cashback_amount"]),
            str(row["expires_at"])[:10], row["order_name"]
        )
        if ok:
            db2 = get_db()
            if db2:
                try:
                    c2 = db2.cursor()
                    c2.execute("UPDATE cashback_rewards SET reminder_sent_at = NOW() WHERE id = %s", (row["id"],))
                    db2.commit(); c2.close(); db2.close()
                    sent += 1
                except Exception:
                    try: db2.close()
                    except: pass

    log_activity("cashback_reminder", f"Relance cashback: {sent} emails envoyes sur {len(rows)} eligibles",
                 {"sent": sent, "total": len(rows)}, source="cron")
    return {"ok": True, "sent": sent, "eligible": len(rows)}


async def _send_cashback_reminder_email(to_email: str, amount: float, expires_date: str, order_name: str):
    """Send reminder email 2 days before cashback store credit expires."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_FROM_EMAIL:
        return False

    formatted = f"{amount:.2f}".replace(".", ",")
    try:
        from datetime import datetime as dt
        exp_display = dt.strptime(expires_date[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        exp_display = expires_date[:10]

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"Votre credit de {formatted} € expire bientot !"

    text_body = f"""Rappel : votre credit magasin de {formatted} EUR (commande {order_name}) expire le {exp_display}.

Utilisez-le sur votre prochaine commande des 70 EUR sur planetebeauty.com.

PlaneteBeauty — planetebeauty.com
"""

    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f8f6f3;">
<div style="max-width:560px;margin:20px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
  <div style="background:#1a1a2e;padding:28px 32px;text-align:center;">
    <h1 style="color:#d4af37;margin:0;font-size:22px;letter-spacing:1px;">PLAN&Egrave;TEBEAUTY</h1>
  </div>
  <div style="padding:32px;">
    <h2 style="color:#c0392b;margin:0 0 16px;font-size:20px;">Votre cr&eacute;dit expire bient&ocirc;t !</h2>
    <p style="color:#444;line-height:1.6;margin:0 0 20px;">
      Le cr&eacute;dit magasin de votre commande <strong>{order_name}</strong> arrive bient&ocirc;t &agrave; expiration.
    </p>
    <div style="background:linear-gradient(135deg,#8e1c1c,#c0392b);border-radius:10px;padding:24px;text-align:center;margin:0 0 20px;">
      <div style="color:#fff;font-size:36px;font-weight:700;margin:0 0 4px;">{formatted} &euro;</div>
      <div style="color:#ffcdd2;font-size:13px;">expire le {exp_display}</div>
    </div>
    <div style="background:#fff5f5;border:1px solid #ffcdd2;border-radius:8px;padding:16px;margin:0 0 20px;">
      <p style="margin:0;color:#c0392b;font-size:14px;font-weight:600;">Ne perdez pas votre cr&eacute;dit !</p>
      <p style="margin:8px 0 0;color:#666;font-size:13px;">Passez commande d'au moins 70 &euro; avant le {exp_display} pour en profiter.</p>
    </div>
    <div style="text-align:center;margin:24px 0;">
      <a href="https://planetebeauty.com" style="display:inline-block;background:#d4af37;color:#1a1a2e;text-decoration:none;padding:12px 32px;border-radius:6px;font-weight:600;font-size:14px;">Utiliser mon cr&eacute;dit</a>
    </div>
  </div>
  <div style="background:#f8f6f3;padding:16px 32px;text-align:center;border-top:1px solid #eee;">
    <p style="margin:0;color:#999;font-size:11px;">
      Plan&egrave;teBeauty &mdash; Parfumerie de niche<br>
      Livraison 24h &middot; Try&amp;Buy &middot; Cashback 5%
    </p>
  </div>
</div>
</body></html>"""

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                              username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
        logger.info(f"Cashback reminder sent to {to_email}: {formatted}€ expires {exp_display}")
        return True
    except Exception as e:
        logger.error(f"Cashback reminder email failed for {to_email}: {e}")
        return False


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
    # Regen info
    last_regen = rc.get("quiz:regen:last")
    regen_count = rc.get("quiz:regen:count")
    last_regen_str = datetime.utcfromtimestamp(int(last_regen)).strftime('%d/%m/%Y %H:%M') if last_regen else "Jamais"

    return {
        "today": {"views": views_today, "starts": starts_today, "completes": completes_today, "atc": atc_today},
        "total": {"views": views_total, "starts": starts_total, "completes": completes_total, "atc": atc_total},
        "conversion_rate": conv_rate,
        "atc_rate": atc_rate,
        "daily": daily,
        "last_regenerated": last_regen_str,
        "products_count": int(regen_count) if regen_count else 0
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
                  handle title vendor productType createdAt
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
                    "conc": mf.get("concentration", p["productType"]), "var": variants,
                    "ca": p.get("createdAt", "")[:10]
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
        # Store regen info in Redis for dashboard
        rc = get_redis()
        if rc:
            rc.set("quiz:regen:last", int(time.time()), ex=86400*90)
            rc.set("quiz:regen:count", len(all_products), ex=86400*90)
        log_activity("quiz_regen", f"Quiz-data.json régénéré: {len(all_products)} produits",
                     {"count": len(all_products), "size_kb": round(len(quiz_json) / 1024, 1)}, source="api")
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
        # Add unique index if not exists (product_handle + reviewer_name + rating + title)
        cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_unique
            ON product_reviews(product_handle, reviewer_name, rating, title)""")
        imported = 0
        skipped = 0
        for r in reviews:
            try:
                rd = r.get("date") or None
                rpd = r.get("reply_date") or None
                cur.execute("""INSERT INTO product_reviews
                    (title, body, rating, review_date, source, curated, reviewer_name, reviewer_email, product_id, product_handle, reply, reply_date, picture_urls)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (product_handle, reviewer_name, rating, title) DO NOTHING""",
                    (r.get("title",""), r.get("body",""), r.get("rating",0), rd, r.get("source",""), r.get("curated",""),
                     r.get("name",""), r.get("email",""), r.get("product_id",""), r.get("product_handle",""),
                     r.get("reply",""), rpd if rpd else None, r.get("picture_urls","")))
                if cur.rowcount > 0:
                    imported += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(f"Review import row error: {e}")
        db.commit()
        cur.close()
        db.close()
        return {"imported": imported, "skipped": skipped}
    except Exception as e:
        try: db.close()
        except: pass
        return {"imported": 0, "skipped": 0, "error": str(e)}


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
            if r.get("review_date"): r["review_date"] = r["review_date"].isoformat()
        cur.execute("SELECT source, COUNT(*) as c FROM product_reviews GROUP BY source")
        by_source = {r["source"]: r["c"] for r in cur.fetchall()}
        cur.close(); db.close()
        return {"total": stats["c"], "avg_rating": round(float(stats["avg"]), 2), "recent": recent, "by_source": by_source}
    except Exception as e:
        try: db.close()
        except: pass
        return {"total": 0, "avg_rating": 0, "recent": [], "by_source": {}, "error": str(e)}


@app.post("/api/reviews/deduplicate")
async def deduplicate_reviews(request: Request):
    """Remove duplicate reviews, keeping the oldest entry per (product_handle, reviewer_email, rating, title)."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(403, "Unauthorized")
    db = get_db()
    if not db: return {"error": "No DB"}
    try:
        cur = db.cursor()
        cur.execute("SELECT COUNT(*) FROM product_reviews")
        before = cur.fetchone()[0]
        cur.execute("""DELETE FROM product_reviews
            WHERE id NOT IN (
                SELECT MIN(id) FROM product_reviews
                GROUP BY product_handle, reviewer_name, rating, title
            )""")
        deleted = cur.rowcount
        cur.execute("SELECT COUNT(*) FROM product_reviews")
        after = cur.fetchone()[0]
        db.commit()
        cur.close(); db.close()
        log_activity("cron_run", f"Reviews deduplicated: {before} → {after} ({deleted} supprimés)", {"before": before, "after": after, "deleted": deleted})
        return {"before": before, "after": after, "deleted": deleted}
    except Exception as e:
        try: db.close()
        except: pass
        return {"error": str(e)}


@app.get("/api/reviews/pending")
async def get_pending_reviews(request: Request):
    """Dashboard: list all pending reviews for moderation."""
    api_key = request.headers.get("x-api-key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        return {"error": "Unauthorized"}
    try:
        db = get_db()
        if not db:
            return {"reviews": [], "count": 0}
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Ensure order_number column exists
        try:
            cur.execute("ALTER TABLE product_reviews ADD COLUMN IF NOT EXISTS order_number TEXT")
            db.commit()
        except Exception:
            try: db.rollback()
            except: pass
        cur.execute("""SELECT id, title, body, rating, reviewer_name, reviewer_email, product_handle,
                        order_number, source, review_date, created_at
                    FROM product_reviews WHERE curated = 'pending'
                    ORDER BY created_at DESC LIMIT 100""")
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("review_date"): d["review_date"] = str(d["review_date"])
            if d.get("created_at"): d["created_at"] = str(d["created_at"])
            result.append(d)
        cur.close(); db.close()
        return {"reviews": result, "count": len(result)}
    except Exception as e:
        logger.error(f"Pending reviews error: {e}")
        return {"reviews": [], "count": 0, "error": str(e)}


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


@app.post("/api/reviews/submit")
async def submit_review(request: Request):
    """Public endpoint: customers submit reviews from product page. Curated='pending' until approved."""
    try:
        body = await request.json()
        handle = body.get("product_handle", "").strip()
        name = body.get("name", "").strip()
        email = body.get("email", "").strip()
        rating = int(body.get("rating", 0))
        title = body.get("title", "").strip()
        review_body = body.get("body", "").strip()
        order_number = body.get("order_number", "").strip()
        # source: "site" (direct from product page) or "flow-email" (from post-delivery email)
        source = body.get("source", "site").strip()
        if source not in ("site", "flow-email"):
            source = "site"

        # Validation
        if not handle or not name or not rating or not review_body:
            return {"success": False, "error": "Champs obligatoires manquants"}
        if rating < 1 or rating > 5:
            return {"success": False, "error": "Note entre 1 et 5"}
        if len(review_body) < 10:
            return {"success": False, "error": "Votre avis doit contenir au moins 10 caractères"}
        if len(name) < 2:
            return {"success": False, "error": "Nom trop court"}

        # Order number is REQUIRED
        if not order_number:
            return {"success": False, "error": "Le numéro de commande est obligatoire pour laisser un avis."}

        # Anti-spam: check honeypot field
        if body.get("website", ""):
            return {"success": True, "message": "Merci pour votre avis !"}  # Silent reject

        # Verify order exists in Shopify
        try:
            clean_order = order_number.lstrip("#").strip()
            gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
            headers_gql = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=8) as c:
                oq = await c.post(gql_url, json={
                    "query": '{ orders(first:1, query:"name:%s") { edges { node { name } } } }' % clean_order.replace('"', '')
                }, headers=headers_gql)
                o_edges = oq.json().get("data", {}).get("orders", {}).get("edges", [])
                if not o_edges:
                    return {"success": False, "error": "Numéro de commande introuvable. Vérifiez votre email de confirmation."}
        except Exception as ve:
            logger.warning(f"Order verification failed for {order_number}: {ve}")
            # Allow submission if API is temporarily unavailable (don't block customer)

        # Rate limit: max 3 reviews per email per day
        db = get_db()
        if not db:
            return {"success": False, "error": "Service temporairement indisponible"}
        cur = db.cursor()
        # Ensure order_number column + unique index exist
        try:
            cur.execute("ALTER TABLE product_reviews ADD COLUMN IF NOT EXISTS order_number TEXT")
            cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_reviews_unique
                ON product_reviews(product_handle, reviewer_name, rating, title)""")
            db.commit()
        except Exception:
            try: db.rollback()
            except: pass
        if email:
            cur.execute("""SELECT COUNT(*) FROM product_reviews
                WHERE reviewer_email = %s AND created_at > NOW() - INTERVAL '24 hours'""", (email,))
            count = cur.fetchone()[0]
            if count >= 3:
                cur.close(); db.close()
                return {"success": False, "error": "Vous avez déjà laissé plusieurs avis récemment"}

        # Insert with curated='pending'
        cur.execute("""INSERT INTO product_reviews
            (title, body, rating, review_date, source, curated, reviewer_name, reviewer_email, product_handle, order_number)
            VALUES (%s, %s, %s, NOW(), %s, 'pending', %s, %s, %s, %s)
            ON CONFLICT (product_handle, reviewer_name, rating, title) DO NOTHING""",
            (title, review_body, rating, source, name, email, handle, order_number or None))
        db.commit()
        inserted = cur.rowcount > 0
        cur.close(); db.close()

        if inserted:
            log_activity("review_submit", f"Nouvel avis ({rating}★) pour {handle} par {name} [source:{source}]", {"handle": handle, "rating": rating, "source": source})
            # Send email notification to info@planetebeauty.com
            try:
                if SMTP_HOST:
                    from email.mime.multipart import MIMEMultipart
                    source_label = "page produit" if source == "site" else "email post-livraison"
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = f"Nouvel avis ({rating}★) - {handle} par {name}"
                    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
                    msg["To"] = "info@planetebeauty.com"
                    email_display = email or "pas d'email"
                    order_line = f"<p><strong>Commande :</strong> {order_number}</p>" if order_number else ""
                    title_line = f"<p><strong>Titre :</strong> {title}</p>" if title else ""
                    stars = "\u2B50" * rating
                    notif_html = f"""<div style="font-family:Arial,sans-serif;max-width:500px">
                        <h3 style="color:#C8984E">Nouvel avis client en attente</h3>
                        <p><strong>Produit :</strong> {handle}</p>
                        <p><strong>Client :</strong> {name} ({email_display})</p>
                        <p><strong>Note :</strong> {stars}</p>
                        <p><strong>Source :</strong> {source_label}</p>
                        {order_line}
                        {title_line}
                        <p><strong>Avis :</strong><br>{review_body}</p>
                        <hr>
                        <p style="font-size:12px;color:#888">Rendez-vous dans l'onglet Avis du dashboard STELLA V8 pour moderer.</p>
                    </div>"""
                    msg.attach(MIMEText(notif_html, "html", "utf-8"))
                    await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                                          username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
                    logger.info(f"Review notification sent to info@planetebeauty.com for {handle}")
            except Exception as mail_err:
                logger.warning(f"Review notification email failed: {mail_err}")

            return {"success": True, "message": "Merci pour votre avis ! Il sera publié après vérification."}
        else:
            return {"success": True, "message": "Merci ! Vous avez déjà laissé un avis similaire."}
    except Exception as e:
        logger.error(f"Review submit error: {e}", exc_info=True)
        return {"success": False, "error": f"Erreur: {str(e)}"}


@app.post("/api/reviews/approve")
async def approve_review(request: Request):
    """Dashboard: approve a review. Optionally credit 5€ store credit (30 days) if with_credit=true."""
    api_key = request.headers.get("x-api-key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        return {"success": False, "error": "Unauthorized"}
    try:
        body = await request.json()
        review_id = body.get("review_id")
        with_credit = body.get("with_credit", False)
        if not review_id:
            return {"success": False, "error": "review_id requis"}

        db = get_db()
        if not db:
            return {"success": False, "error": "DB indisponible"}
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get the review
        cur.execute("SELECT * FROM product_reviews WHERE id = %s", (review_id,))
        review = cur.fetchone()
        if not review:
            cur.close(); db.close()
            return {"success": False, "error": "Avis introuvable"}
        if review["curated"] == "ok":
            cur.close(); db.close()
            return {"success": False, "error": "Avis deja publie"}

        # 1. Publish the review
        cur.execute("UPDATE product_reviews SET curated = 'ok' WHERE id = %s", (review_id,))
        db.commit()
        cur.close(); db.close()

        # 2. Credit 5€ store credit (30 days) ONLY if with_credit=true
        credit_ok = False
        credit_error = None

        if with_credit:
            email = (review.get("reviewer_email") or "").strip()
            if email:
                gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
                headers_gql = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

                async with httpx.AsyncClient(timeout=10) as c:
                    search_r = await c.post(gql_url, json={
                        "query": '{ customers(first:1, query:"email:%s") { edges { node { id } } } }' % email
                    }, headers=headers_gql)
                    edges = search_r.json().get("data", {}).get("customers", {}).get("edges", [])

                    if edges:
                        customer_gid = edges[0]["node"]["id"]
                        acc_r = await c.post(gql_url, json={
                            "query": '{customer(id: "%s") { storeCreditAccounts(first:1) { edges { node { id } } } }}' % customer_gid
                        }, headers=headers_gql)
                        acc_edges = acc_r.json().get("data", {}).get("customer", {}).get("storeCreditAccounts", {}).get("edges", [])
                        target_id = acc_edges[0]["node"]["id"] if acc_edges else customer_gid

                        from datetime import timedelta
                        expires_at = (datetime.utcnow() + timedelta(days=30)).strftime("%Y-%m-%dT23:59:59Z")

                        credit_mutation = """mutation($id: ID!, $creditInput: StoreCreditAccountCreditInput!) {
                            storeCreditAccountCredit(id: $id, creditInput: $creditInput) {
                                storeCreditAccountTransaction { id amount { amount currencyCode } }
                                userErrors { field message }
                            }
                        }"""
                        cr = await c.post(gql_url, json={
                            "query": credit_mutation,
                            "variables": {
                                "id": target_id,
                                "creditInput": {
                                    "creditAmount": {"amount": "5.00", "currencyCode": "EUR"},
                                    "expiresAt": expires_at
                                }
                            }
                        }, headers=headers_gql)
                        cr_data = cr.json()
                        u_errors = cr_data.get("data", {}).get("storeCreditAccountCredit", {}).get("userErrors", [])
                        if not u_errors and cr_data.get("data", {}).get("storeCreditAccountCredit", {}).get("storeCreditAccountTransaction"):
                            credit_ok = True
                            log_activity("review_credit", f"Avis approuve: 5€ credite a {email} pour {review['product_handle']}",
                                         {"email": email, "review_id": review_id, "amount": 5.00, "expires": expires_at},
                                         source="dashboard", customer_email=email)
                        else:
                            credit_error = u_errors[0].get("message", "Erreur credit") if u_errors else "Erreur mutation"
                    else:
                        credit_error = f"Client non trouve pour {email}"
            else:
                credit_error = "Pas d'email — credit non attribue"

        msg = "Avis publie"
        if with_credit:
            msg += " + 5€ credite (30j)" if credit_ok else f" (credit: {credit_error})"
        return {
            "success": True,
            "published": True,
            "credit_ok": credit_ok,
            "credit_error": credit_error,
            "message": msg
        }
    except Exception as e:
        logger.error(f"Review approve error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/reviews/reject")
async def reject_review(request: Request):
    """Dashboard: reject/delete a pending review."""
    api_key = request.headers.get("x-api-key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        return {"success": False, "error": "Unauthorized"}
    try:
        body = await request.json()
        review_id = body.get("review_id")
        if not review_id:
            return {"success": False, "error": "review_id requis"}
        db = get_db()
        if not db:
            return {"success": False, "error": "DB indisponible"}
        cur = db.cursor()
        cur.execute("DELETE FROM product_reviews WHERE id = %s AND curated = 'pending'", (review_id,))
        deleted = cur.rowcount > 0
        db.commit()
        cur.close(); db.close()
        return {"success": True, "deleted": deleted}
    except Exception as e:
        logger.error(f"Review reject error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/google-optin")
async def google_optin_page(request: Request):
    """Public: render Google Customer Reviews opt-in module.
    Called from checkout UI extension on thank-you page.
    Accepts ?order_id=X&email=X&country=XX&delivery_date=YYYY-MM-DD"""
    order_id = request.query_params.get("order_id", "")
    email = request.query_params.get("email", "")
    country = request.query_params.get("country", "FR")
    delivery_date = request.query_params.get("delivery_date", "")

    if not delivery_date:
        from datetime import timedelta
        delivery_date = (datetime.utcnow() + timedelta(days=10)).strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Avis Google - PlaneteBeauty</title>
    <style>
        body {{ font-family: 'Outfit', -apple-system, sans-serif; background: #FAF7F2; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
        .container {{ max-width: 480px; padding: 40px 24px; text-align: center; }}
        .logo {{ font-family: 'Cormorant Garamond', serif; font-size: 28px; color: #C8984E; margin-bottom: 24px; }}
        h1 {{ font-size: 20px; font-weight: 400; color: #1A1A1A; margin-bottom: 12px; font-family: 'Cormorant Garamond', serif; }}
        p {{ font-size: 14px; color: #6B6560; line-height: 1.6; margin-bottom: 24px; }}
        .thanks {{ display: none; font-size: 16px; color: #34A853; margin-top: 24px; }}
        .loading {{ font-size: 13px; color: #9B9590; }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Outfit:wght@300;400;500&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="logo">PlaneteBeauty</div>
        <h1>Merci pour votre commande !</h1>
        <p>Nous souhaitons vous offrir la meilleure experience possible.<br>
        Acceptez de recevoir un court questionnaire Google pour nous aider.</p>
        <div class="loading">Chargement du formulaire Google...</div>
        <div class="thanks" id="thanks-msg">Merci ! Vous pouvez fermer cette page.</div>
    </div>

    <script src="https://apis.google.com/js/platform.js?onload=renderOptIn" async defer></script>
    <script>
    window.renderOptIn = function() {{
        document.querySelector('.loading').style.display = 'none';
        window.gapi.load('surveyoptin', function() {{
            window.gapi.surveyoptin.render({{
                "merchant_id": 277377202,
                "order_id": "{order_id}",
                "email": "{email}",
                "delivery_country": "{country}",
                "estimated_delivery_date": "{delivery_date}"
            }});
        }});
    }};
    window.___gcfg = {{ lang: 'fr' }};
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/review-redirect")
async def review_redirect(request: Request):
    """Public: redirect from Flow email to product page with ?review=email.
    Accepts ?order=PB-1234 — looks up the order, gets first product handle, redirects."""
    from fastapi.responses import RedirectResponse
    fallback = "https://planetebeauty.com?review=email"
    order_name = request.query_params.get("order", "").strip()
    if not order_name:
        return RedirectResponse(url=fallback)

    try:
        gql_url = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
        headers_gql = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

        # Search order by name
        query = """{ orders(first: 1, query: "name:%s") {
            edges { node { lineItems(first: 5) {
                edges { node { product { handle } } }
            } } }
        } }""" % order_name.replace('"', '\\"')

        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.post(gql_url, json={"query": query}, headers=headers_gql)
            data = r.json()
            edges = data.get("data", {}).get("orders", {}).get("edges", [])
            if edges:
                line_items = edges[0]["node"]["lineItems"]["edges"]
                for li in line_items:
                    handle = li.get("node", {}).get("product", {}).get("handle", "")
                    # Skip free gifts and non-product items
                    if handle and not handle.startswith("docapp-free-gift"):
                        product_url = f"https://planetebeauty.com/products/{handle}?review=email"
                        logger.info(f"Review redirect: {order_name} -> {handle}")
                        return RedirectResponse(url=product_url)

        # Fallback: no product found
        logger.warning(f"Review redirect: no product found for order {order_name}")
        return RedirectResponse(url=fallback)
    except Exception as e:
        logger.error(f"Review redirect error: {e}")
        return RedirectResponse(url=fallback)


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

    # Check if product has a Try Me variant → pre-generate card assets
    try:
        variants = pdata.get("variants", [])
        has_tryme = any("try me" in (v.get("title") or "").lower() for v in variants)
        if has_tryme:
            pid = str(pdata.get("id", ""))
            handle = pdata.get("handle", "")
            asyncio.create_task(_pregenerate_tryme_card(pid, ptitle, handle))
    except Exception as e:
        logger.warning(f"Try Me card pregeneration check: {e}")

    # Force DRAFT on new products if they are ACTIVE and incomplete
    # Shopify webhook header X-Shopify-Topic tells us if it's create or update
    topic = request.headers.get("X-Shopify-Topic", "")
    if "create" in topic.lower():
        asyncio.create_task(_force_draft_if_incomplete(pdata))

    # Auto-enrichment check: verify product completeness
    asyncio.create_task(_check_product_enrichment(pdata))

    return {"ok": True, "regenerating": True}

async def _force_draft_if_incomplete(pdata):
    """Force new products to DRAFT status. Products must be 100% complete before going ACTIVE."""
    try:
        pid = pdata.get("admin_graphql_api_id", f"gid://shopify/Product/{pdata.get('id','')}")
        status = pdata.get("status", "")
        ptitle = pdata.get("title", "")

        if status and status.lower() == "active":
            async with httpx.AsyncClient(timeout=10) as client:
                mut = 'mutation { productUpdate(input: {id: "%s", status: DRAFT}) { product { id status } userErrors { message } } }' % pid
                await client.post(SHOPIFY_GRAPHQL_URL,
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    json={"query": mut})
                logger.info(f"Forced DRAFT on new product: {ptitle}")
                log_activity("force_draft", f"Nouveau produit forcé en DRAFT: {ptitle}",
                    {"product": ptitle, "reason": "Règle: jamais ACTIVE tant que pas 100% complet"},
                    source="webhook", product_title=ptitle)
    except Exception as e:
        logger.warning(f"Force draft: {e}")

async def _check_product_enrichment(pdata):
    """Check if a product is fully enriched. Tag 'a-enrichir' if not.
    Checks: vendor, description, SEO, images notes, images produit (carré 2000x2000),
    metafields parfum, variante Try Me, canaux de vente, collections/catégories."""
    try:
        pid = pdata.get("admin_graphql_api_id", f"gid://shopify/Product/{pdata.get('id','')}")
        ptitle = pdata.get("title", "")
        vendor = pdata.get("vendor", "")
        product_type = pdata.get("product_type", "")

        # Skip non-perfume products
        perfume_types = ["Extrait de Parfum", "Eau de Parfum", "Eau de Parfum Intense", "Eau de Toilette", "Eau de Cologne"]
        if product_type not in perfume_types:
            return

        issues = []

        # 1. Vendor uppercase
        if vendor and vendor != vendor.upper():
            issues.append("vendor pas MAJUSCULES")

        # 2. productType exact
        if product_type not in perfume_types:
            issues.append(f"productType '{product_type}' non standard")

        # 3. Deep check via GraphQL
        async with httpx.AsyncClient(timeout=20) as client:
            gql = """{ product(id: "%s") {
                descriptionHtml status publishedOnCurrentPublication
                seo { title description }
                images(first: 5) { edges { node { width height url } } }
                variants(first: 5) { edges { node { title price availableForSale } } }
                collections(first: 5) { edges { node { title } } }
                img_tete: metafield(namespace:"parfum", key:"image_note_tete") { value }
                img_coeur: metafield(namespace:"parfum", key:"image_note_coeur") { value }
                img_fond: metafield(namespace:"parfum", key:"image_note_fond") { value }
                note_tete: metafield(namespace:"parfum", key:"note_tete_principale") { value }
                note_coeur: metafield(namespace:"parfum", key:"note_coeur_principale") { value }
                note_fond: metafield(namespace:"parfum", key:"note_fond_principale") { value }
                parfumeur: metafield(namespace:"parfum", key:"parfumeur") { value }
                famille: metafield(namespace:"parfum", key:"famille_olfactive") { value }
                accord: metafield(namespace:"parfum", key:"accord_principal") { value }
                genre: metafield(namespace:"parfum", key:"genre") { value }
                intensite: metafield(namespace:"parfum", key:"intensite") { value }
                sillage: metafield(namespace:"parfum", key:"sillage_level") { value }
                tags
            } }""" % pid
            r = await client.post(SHOPIFY_GRAPHQL_URL,
                headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                json={"query": gql})
            p = r.json().get("data", {}).get("product", {})
            if not p:
                return

            # 4. Description 3 paragraphs
            desc = p.get("descriptionHtml", "") or ""
            if desc.count("<p") < 1:
                issues.append("description vide (min 1 paragraphe)")

            # 5. SEO
            seo = p.get("seo", {}) or {}
            if not seo.get("title"):
                issues.append("SEO title vide")
            if not seo.get("description"):
                issues.append("SEO meta vide")

            # 6. Images produit — minimum 3, carrées
            images = [e["node"] for e in p.get("images", {}).get("edges", [])]
            if len(images) < 3:
                issues.append(f"seulement {len(images)} images (min 3)")
            for i, img in enumerate(images):
                w, h = img.get("width", 0), img.get("height", 0)
                if w != h:
                    issues.append(f"image {i+1} pas carrée ({w}x{h})")
                elif w < 2048:
                    issues.append(f"image {i+1} trop petite ({w}px, min 2000)")

            # 7. Images notes olfactives
            if p.get("note_tete", {}).get("value"):
                if not p.get("img_tete", {}).get("value"):
                    issues.append("image_note_tete manquante")
                if not p.get("img_coeur", {}).get("value"):
                    issues.append("image_note_coeur manquante")
                if not p.get("img_fond", {}).get("value"):
                    issues.append("image_note_fond manquante")

            # 8. Metafields parfum essentiels (7 champs minimum)
            essential_mf = ["note_tete", "note_coeur", "note_fond", "parfumeur", "famille", "accord", "genre"]
            missing_mf = [k for k in essential_mf if not p.get(k, {}).get("value")]
            if missing_mf:
                issues.append(f"metafields manquants: {', '.join(missing_mf)}")

            # 9. Variante Try Me
            variants = [e["node"] for e in p.get("variants", {}).get("edges", [])]
            has_tryme = any("try me" in (v.get("title") or "").lower() for v in variants)
            has_standard = any("try me" not in (v.get("title") or "").lower() for v in variants)
            if has_standard and not has_tryme:
                issues.append("pas de variante Try Me")

            # 10. Collections/catégories
            collections = [e["node"]["title"] for e in p.get("collections", {}).get("edges", [])]
            if not collections:
                issues.append("pas dans aucune collection")

            # 11. Tags format
            tags = p.get("tags", [])
            has_famille_tag = any(t.startswith("Famille:") for t in tags)
            has_genre_tag = any(t.startswith("Genre:") for t in tags)
            if not has_famille_tag:
                issues.append("tag Famille:X manquant")
            if not has_genre_tag:
                issues.append("tag Genre:X manquant")

            # === TAG MANAGEMENT ===
            current_tags = p.get("tags", [])
            needs_tag = len(issues) > 0
            has_tag = "a-enrichir" in current_tags

            if needs_tag and not has_tag:
                new_tags = current_tags + ["a-enrichir"]
                mut = 'mutation { productUpdate(input: {id: "%s", tags: %s}) { product { id } userErrors { message } } }' % (pid, json.dumps(new_tags))
                await client.post(SHOPIFY_GRAPHQL_URL,
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    json={"query": mut})
                logger.info(f"Enrichment: tagged '{ptitle}' a-enrichir — {', '.join(issues[:5])}")
                log_activity("enrichment_check", f"Produit incomplet: {ptitle} — {len(issues)} problèmes",
                    {"product": ptitle, "issues": issues}, source="webhook", product_title=ptitle)

            elif not needs_tag and has_tag:
                new_tags = [t for t in current_tags if t != "a-enrichir"]
                mut = 'mutation { productUpdate(input: {id: "%s", tags: %s}) { product { id } userErrors { message } } }' % (pid, json.dumps(new_tags))
                await client.post(SHOPIFY_GRAPHQL_URL,
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    json={"query": mut})
                logger.info(f"Enrichment: '{ptitle}' complet, tag a-enrichir retiré")

    except Exception as e:
        logger.warning(f"Enrichment check: {e}")

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
    """Shopify webhook: order paid → respond 200 immediately, process cashback in background."""
    import asyncio
    logger.info(f"[WEBHOOK] order-paid received")
    try:
        raw_body = await request.body()
        body = json.loads(raw_body)
    except Exception as e:
        logger.error(f"[WEBHOOK] order-paid parse error: {e}")
        return {"ok": False, "error": "Invalid JSON"}

    # Respond 200 immediately — Shopify requires < 5 seconds
    order_name = body.get("name", "?")
    logger.info(f"[WEBHOOK] order-paid {order_name} — processing in background")
    asyncio.create_task(_process_cashback(body))
    return {"ok": True, "queued": order_name}


async def _process_cashback(body):
    """Background task: calculate and credit cashback after webhook returns 200."""
    import traceback
    try:
        await _process_cashback_inner(body)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"[CASHBACK] Background error: {e}\n{tb}")
        log_activity("cashback_error", f"Erreur cashback: {e}",
                     {"order_name": body.get("name", ""), "error": str(e), "traceback": tb[-500:]},
                     source="webhook", order_name=body.get("name", ""))


async def _process_cashback_inner(body):
    """Inner cashback processing logic."""

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

    # Deduplication: Redis lock (prevents race condition with concurrent webhooks)
    redis_lock_key = f"cashback_lock:{order_id}"
    try:
        from redis import Redis
        rc = Redis.from_url(os.getenv("REDIS_URL", ""), decode_responses=True)
        # SET NX = only set if not exists, EX = expire in 60s
        lock_acquired = rc.set(redis_lock_key, "1", nx=True, ex=60)
        if not lock_acquired:
            logger.info(f"[CASHBACK] {order_name} lock exists — skipping duplicate")
            return
    except Exception as e:
        logger.warning(f"[CASHBACK] Redis lock failed: {e}, continuing anyway")

    # Skip orders matching excluded tags
    excluded_tags = [t.strip().lower() for t in settings["excluded_tags"].split(",") if t.strip()]
    order_tags = [t.strip().lower() for t in tags.split(",") if t.strip()]
    if any(et in order_tags for et in excluded_tags):
        return {"ok": True, "skipped": True, "reason": f"excluded tag match"}

    # Calculate cashback base:
    # subtotal_price = produits après remises (hors port, hors taxes)
    # Exclude Try Me items, shipping items, and items tagged no-discount/tryme
    subtotal = float(body.get("subtotal_price", "0"))  # Produits après remises, hors port
    total_discounts = float(body.get("total_discounts", "0"))  # Remises déjà soustraites du subtotal

    # Calculate eligible subtotal (exclude Try Me / Try Me Upsell / no-discount / gift items)
    tryme_total = 0.0
    for li in body.get("line_items", []):
        title = (li.get("title") or "").lower()
        variant_title = (li.get("variant_title") or "").lower()
        product_type = (li.get("product_type") or "").lower()

        # 1. Try Me Upsell panier — line property _tryme_upsell == "true"
        is_upsell = any(
            (p.get("name") == "_tryme_upsell" and str(p.get("value", "")).lower() == "true")
            for p in (li.get("properties") or [])
        )

        # 2. Variante Try Me classique
        is_tryme_variant = (
            "try me" in title or "tryme" in title
            or "try me" in variant_title or "tryme" in variant_title
            or "2 ml" in variant_title
        )

        # 3. Tags ligne (Shopify expose rarement les tags sur line_items mais on check par securite)
        li_tags_raw = li.get("tags") or []
        if isinstance(li_tags_raw, str):
            li_tags_list = [t.strip().lower() for t in li_tags_raw.split(",") if t.strip()]
        else:
            li_tags_list = [str(t).lower() for t in li_tags_raw]
        has_excluded_tag = any(
            tag in li_tags_list
            for tag in ("tryme", "no-cashback", "no-discount", "gift", "exclusion-promo")
        )

        # 4. Mots-cles cadeaux / echantillons / mystere
        is_gift_like = any(
            kw in title
            for kw in ("echantillon", "échantillon", "mystère", "mystere", "cadeau", "gift", "shipping", "livraison")
        ) or product_type in ("try me", "gift")

        if is_upsell or is_tryme_variant or has_excluded_tag or is_gift_like:
            tryme_total += float(li.get("price", "0")) * int(li.get("quantity", 1))

    # Detect store credit used in this order (payment gateway = "gift_card" or "store_credit")
    store_credit_used = 0.0
    for txn in body.get("payment_gateway_names", []):
        if "gift_card" in txn.lower() or "store_credit" in txn.lower():
            # Look in transactions for the store credit amount
            break
    # More precise: check refunds/transactions for store credit
    for line in (body.get("payment_terms") or {}).get("payment_schedules", []) or []:
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
    cashback_base = current_subtotal - tryme_total  # Subtotal minus Try Me items, before shipping
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
                api_subtotal = float(order_data.get("currentSubtotalPriceSet", {}).get("shopMoney", {}).get("amount", cashback_base))
                # Subtract Try Me total from API subtotal (API includes everything)
                cashback_base = api_subtotal - tryme_total
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
                expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT NOW(), status TEXT DEFAULT 'active',
                email_sent BOOLEAN DEFAULT FALSE, reminder_sent_at TIMESTAMP)""")
            # Add columns if missing (for existing tables)
            for col_sql in [
                "ALTER TABLE cashback_rewards ADD COLUMN IF NOT EXISTS email_sent BOOLEAN DEFAULT FALSE",
                "ALTER TABLE cashback_rewards ADD COLUMN IF NOT EXISTS reminder_sent_at TIMESTAMP",
            ]:
                try:
                    cur.execute(col_sql)
                except Exception:
                    pass
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

    # 5. Send cashback notification email to customer
    email_ok = False
    if customer_email:
        email_ok = await _send_cashback_email(customer_email, cashback_amount, expires_at, order_name)
        if email_ok:
            db2 = get_db()
            if db2:
                try:
                    c2 = db2.cursor()
                    c2.execute("UPDATE cashback_rewards SET email_sent = TRUE WHERE order_id = %s", (order_id,))
                    db2.commit(); c2.close(); db2.close()
                except Exception:
                    try: db2.close()
                    except: pass

    return {"ok": True, "cashback": cashback_amount, "base": cashback_base, "storeCreditUsed": store_credit_used, "email_sent": email_ok}


async def _send_cashback_email(to_email: str, amount: float, expires_at: str, order_name: str):
    """Send cashback notification email to customer after store credit is applied."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_FROM_EMAIL:
        logger.warning("Cashback email: SMTP not configured")
        return False

    formatted = f"{amount:.2f}".replace(".", ",")
    # Parse expiry date for display
    try:
        from datetime import datetime as dt
        exp_date = dt.strptime(expires_at[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        exp_date = expires_at[:10]

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = f"Votre cashback de {formatted} € est disponible !"

    text_body = f"""Merci pour votre commande {order_name} !

Votre cashback de {formatted} EUR a ete credite sur votre compte PlaneteBeauty.

Ce credit magasin est utilisable sur votre prochaine commande des 70 EUR d'achat.
Il est valable jusqu'au {exp_date}.

Pour l'utiliser : passez une commande, le credit apparaitra automatiquement dans les moyens de paiement au checkout.

Bonne decouverte olfactive !
PlaneteBeauty — planetebeauty.com
"""

    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f8f6f3;">
<div style="max-width:560px;margin:20px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
  <div style="background:#1a1a2e;padding:28px 32px;text-align:center;">
    <h1 style="color:#d4af37;margin:0;font-size:22px;letter-spacing:1px;">PLAN&Egrave;TEBEAUTY</h1>
  </div>
  <div style="padding:32px;">
    <h2 style="color:#1a1a2e;margin:0 0 16px;font-size:20px;">Votre cashback est disponible !</h2>
    <p style="color:#444;line-height:1.6;margin:0 0 20px;">
      Merci pour votre commande <strong>{order_name}</strong>. En tant que cliente fid&egrave;le, nous avons cr&eacute;dit&eacute; votre compte :
    </p>
    <div style="background:linear-gradient(135deg,#1a1a2e,#2d2d4e);border-radius:10px;padding:24px;text-align:center;margin:0 0 20px;">
      <div style="color:#d4af37;font-size:36px;font-weight:700;margin:0 0 4px;">{formatted} &euro;</div>
      <div style="color:#aaa;font-size:13px;">de cr&eacute;dit magasin</div>
    </div>
    <div style="background:#faf8f5;border-radius:8px;padding:16px;margin:0 0 20px;">
      <p style="margin:0 0 8px;color:#666;font-size:14px;"><strong>Comment l'utiliser ?</strong></p>
      <p style="margin:0 0 4px;color:#666;font-size:13px;">&bull; Passez une commande d'au moins 70 &euro;</p>
      <p style="margin:0 0 4px;color:#666;font-size:13px;">&bull; Au checkout, s&eacute;lectionnez &laquo; Cr&eacute;dit magasin &raquo;</p>
      <p style="margin:0;color:#666;font-size:13px;">&bull; Valable jusqu'au <strong>{exp_date}</strong></p>
    </div>
    <div style="text-align:center;margin:24px 0;">
      <a href="https://planetebeauty.com" style="display:inline-block;background:#d4af37;color:#1a1a2e;text-decoration:none;padding:12px 32px;border-radius:6px;font-weight:600;font-size:14px;">Decouvrir nos nouveautes</a>
    </div>
  </div>
  <div style="background:#f8f6f3;padding:16px 32px;text-align:center;border-top:1px solid #eee;">
    <p style="margin:0;color:#999;font-size:11px;">
      Plan&egrave;teBeauty &mdash; Parfumerie de niche<br>
      Livraison 24h &middot; Try&amp;Buy &middot; Cashback 5%
    </p>
  </div>
</div>
</body></html>"""

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                              username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
        logger.info(f"Cashback email sent to {to_email}: {formatted}€ (order {order_name})")
        return True
    except Exception as e:
        logger.error(f"Cashback email failed for {to_email}: {e}")
        return False


async def _send_google_review_email(to_email: str, first_name: str, order_name: str) -> bool:
    """Send post-delivery Google review request email via SMTP (Gmail)."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_FROM_EMAIL:
        logger.warning("Google review email: SMTP not configured")
        return False

    greeting = f"Bonjour {first_name}" if first_name else "Bonjour"

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = "Votre avis compte — PlanèteBeauty"

    text_body = f"""{greeting},

Merci pour votre commande {order_name}. Nous esperons que vous etes ravi(e) de votre achat.

Pourriez-vous prendre 30 secondes pour partager votre experience sur Google ?
Votre avis nous aide enormement et permet a d'autres clients de decouvrir notre parfumerie de niche.

----------------------------------------
EN REMERCIEMENT : 5 EUR DE CREDIT MAGASIN

Laissez un avis Google et recevez 5 EUR de credit magasin a utiliser sur votre prochaine commande, des que votre avis est valide.

IMPORTANT — pour que nous puissions vous attribuer le credit, indiquez IMPERATIVEMENT votre numero de commande dans votre avis :

Votre numero de commande : {order_name}
(format : # suivi de 5 chiffres, exemple #20250)

Sans ce numero, le credit ne pourra pas etre attribue.
----------------------------------------

Laisser un avis : {GOOGLE_REVIEW_URL}

Merci infiniment,
L'equipe PlaneteBeauty — planetebeauty.com
"""

    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f8f6f3;">
<div style="max-width:560px;margin:20px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
  <div style="background:#1a1a2e;padding:28px 32px;text-align:center;">
    <h1 style="color:#d4af37;margin:0;font-size:22px;letter-spacing:1px;">PLAN&Egrave;TEBEAUTY</h1>
  </div>
  <div style="padding:32px;">
    <h2 style="color:#1a1a2e;margin:0 0 16px;font-size:20px;">Votre avis compte &eacute;norm&eacute;ment</h2>
    <p style="color:#444;line-height:1.6;margin:0 0 16px;">
      {greeting},
    </p>
    <p style="color:#444;line-height:1.6;margin:0 0 20px;">
      Merci pour votre commande <strong>{order_name}</strong>. Nous esp&eacute;rons que vous &ecirc;tes ravi(e) de votre achat.
    </p>
    <p style="color:#444;line-height:1.6;margin:0 0 24px;">
      Pourriez-vous prendre <strong>30 secondes</strong> pour partager votre exp&eacute;rience sur Google&nbsp;? Votre avis nous aide &eacute;norm&eacute;ment et permet &agrave; d'autres clients de d&eacute;couvrir notre parfumerie de niche.
    </p>
    <div style="border:1px solid #d4af37;border-radius:10px;padding:24px;margin:24px 0;background:#fdfaf2;">
      <p style="margin:0 0 6px;color:#1a1a2e;font-size:11px;letter-spacing:2.5px;font-weight:600;text-transform:uppercase;text-align:center;">En remerciement</p>
      <p style="margin:0 0 18px;color:#1a1a2e;font-size:18px;line-height:1.4;font-weight:600;text-align:center;">
        Recevez <span style="color:#d4af37;">5&nbsp;&euro; de cr&eacute;dit magasin</span><br>sur votre prochaine commande
      </p>
      <p style="margin:0 0 10px;color:#555;font-size:13px;line-height:1.6;">
        Pour que nous puissions vous attribuer le cr&eacute;dit, indiquez <strong>imp&eacute;rativement</strong> votre num&eacute;ro de commande dans votre avis Google&nbsp;:
      </p>
      <div style="text-align:center;margin:14px 0 8px;">
        <span style="display:inline-block;background:#1a1a2e;color:#d4af37;padding:10px 24px;border-radius:6px;font-size:18px;font-weight:bold;letter-spacing:1.5px;font-family:Menlo,Consolas,monospace;">{order_name}</span>
      </div>
      <p style="margin:8px 0 0;color:#999;font-size:11px;line-height:1.5;text-align:center;">
        Format&nbsp;: # suivi de 5 chiffres (exemple #20250)<br>Sans ce num&eacute;ro, le cr&eacute;dit ne peut pas &ecirc;tre attribu&eacute;.
      </p>
    </div>
    <div style="text-align:center;margin:24px 0;">
      <a href="{GOOGLE_REVIEW_URL}" style="display:inline-block;background:#d4af37;color:#1a1a2e;text-decoration:none;padding:14px 36px;border-radius:6px;font-weight:600;font-size:15px;">&#9733; Laisser un avis Google</a>
    </div>
    <p style="color:#888;line-height:1.6;margin:24px 0 0;font-size:13px;">
      Merci infiniment pour votre confiance,<br>
      L'&eacute;quipe Plan&egrave;teBeauty
    </p>
  </div>
  <div style="background:#f8f6f3;padding:16px 32px;text-align:center;border-top:1px solid #eee;">
    <p style="margin:0;color:#999;font-size:11px;">
      Plan&egrave;teBeauty &mdash; Parfumerie de niche<br>
      Livraison 24h &middot; Try&amp;Buy &middot; Cashback 5%
    </p>
  </div>
</div>
</body></html>"""

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                              username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
        logger.info(f"Google review email sent to {to_email} (order {order_name})")
        return True
    except Exception as e:
        logger.error(f"Google review email failed for {to_email}: {e}")
        return False


@app.post("/api/webhook/google-review-request")
async def webhook_google_review_request(request: Request):
    """Shopify Flow → send Google review request email via Gmail SMTP.

    Expected JSON body: {"to": "...", "first_name": "...", "order_name": "#1234"}
    Auth: header X-Flow-Token must match FLOW_WEBHOOK_TOKEN env var.
    Idempotent: deduped on (order_name, to) for 30 days via PostgreSQL.
    """
    import asyncio

    if not FLOW_WEBHOOK_TOKEN:
        logger.error("[FLOW] google-review-request: FLOW_WEBHOOK_TOKEN not set")
        raise HTTPException(503, "Flow webhook not configured")
    if request.headers.get("x-flow-token") != FLOW_WEBHOOK_TOKEN:
        raise HTTPException(401, "Invalid token")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    to_email = (body.get("to") or "").strip()
    first_name = (body.get("first_name") or "").strip()
    order_name = (body.get("order_name") or "").strip()

    if not to_email:
        raise HTTPException(400, "Missing 'to' field")

    # Dedup: don't send twice for the same (order, recipient)
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS google_review_emails (
                id SERIAL PRIMARY KEY,
                to_email TEXT NOT NULL,
                order_name TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(to_email, order_name))""")
            db.commit()
            cur.execute("SELECT 1 FROM google_review_emails WHERE to_email=%s AND order_name=%s",
                        (to_email, order_name))
            if cur.fetchone():
                cur.close(); db.close()
                logger.info(f"[FLOW] google-review-request {order_name} → {to_email} already sent, skipping")
                return {"ok": True, "skipped": True, "reason": "already sent"}
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
            logger.warning(f"[FLOW] google-review dedup check failed: {e}")

    async def _send_and_log():
        ok = await _send_google_review_email(to_email, first_name, order_name)
        if ok:
            db2 = get_db()
            if db2:
                try:
                    c2 = db2.cursor()
                    c2.execute("""INSERT INTO google_review_emails (to_email, order_name)
                        VALUES (%s, %s) ON CONFLICT (to_email, order_name) DO NOTHING""",
                        (to_email, order_name))
                    db2.commit(); c2.close(); db2.close()
                except Exception as e:
                    try: db2.close()
                    except: pass
                    logger.warning(f"[FLOW] google-review log failed: {e}")
            log_activity("google_review_email", f"Mail avis Google envoyé pour {order_name}",
                         {"to": to_email, "order_name": order_name},
                         source="flow_webhook", customer_email=to_email, order_name=order_name)

    asyncio.create_task(_send_and_log())
    return {"ok": True, "queued": order_name or to_email}


@app.post("/api/admin/google-review/backfill")
async def google_review_backfill(request: Request):
    """Backfill: send Google review request mail to all customers delivered since `since`,
    where delivery happened at least `min_days_after_delivery` days ago.

    Auth: header X-Flow-Token must match FLOW_WEBHOOK_TOKEN.
    Body params (JSON, all optional):
      - since: ISO date (default "2026-04-13")
      - min_days_after_delivery: int (default 2)
      - dry_run: bool (default false) — when true, lists eligible orders without sending
      - limit: int (default 500) — max orders to process this run

    Dedup is automatic via the google_review_emails table.
    """
    if not FLOW_WEBHOOK_TOKEN:
        raise HTTPException(503, "FLOW_WEBHOOK_TOKEN not set")
    if request.headers.get("x-flow-token") != FLOW_WEBHOOK_TOKEN:
        raise HTTPException(401, "Invalid token")

    try:
        body = await request.json()
    except Exception:
        body = {}

    since = body.get("since", "2026-04-13")
    min_days = int(body.get("min_days_after_delivery", 2))
    min_days_fulfilled = int(body.get("min_days_after_fulfilled", 7))
    dry_run = bool(body.get("dry_run", False))
    limit = int(body.get("limit", 500))
    exclude_orders = body.get("exclude_orders", []) or []
    exclude_set = {str(o).strip().lstrip("#") for o in exclude_orders if str(o).strip()}

    from datetime import datetime as dt, timedelta
    cutoff = dt.utcnow() - timedelta(days=min_days)
    cutoff_fulfilled = dt.utcnow() - timedelta(days=min_days_fulfilled)

    gql_url = SHOPIFY_GRAPHQL_URL
    headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}
    search_query = f"fulfillment_status:fulfilled created_at:>={since}"

    eligible = []
    skipped_no_delivery = 0
    skipped_too_recent = 0
    skipped_no_email = 0
    skipped_excluded = 0
    cursor = None
    page = 0
    fetched = 0

    async with httpx.AsyncClient(timeout=30) as c:
        while fetched < limit:
            page += 1
            after_clause = f', after: "{cursor}"' if cursor else ""
            gql = f"""{{
              orders(first: 50, query: "{search_query}"{after_clause}, sortKey: CREATED_AT) {{
                pageInfo {{ hasNextPage endCursor }}
                edges {{
                  node {{
                    id name createdAt
                    customer {{ email firstName }}
                    fulfillments {{ deliveredAt status createdAt }}
                  }}
                }}
              }}
            }}"""
            try:
                r = await c.post(gql_url, json={"query": gql}, headers=headers)
                data = r.json().get("data", {}).get("orders", {})
            except Exception as e:
                logger.error(f"[BACKFILL] GraphQL error page {page}: {e}")
                break

            edges = data.get("edges", [])
            if not edges:
                break

            for edge in edges:
                fetched += 1
                if fetched > limit:
                    break
                node = edge["node"]
                cust = node.get("customer") or {}
                to_email = (cust.get("email") or "").strip()
                first_name = (cust.get("firstName") or "").strip()
                order_name = node.get("name", "")

                if exclude_set and order_name.lstrip("#") in exclude_set:
                    skipped_excluded += 1
                    continue

                if not to_email:
                    skipped_no_email += 1
                    continue

                # Delivered date if available, else fall back to fulfillment date
                # (Mondial Relay/relais souvent ne push pas l'event DELIVERED)
                delivered_at = None
                fulfilled_at = None
                for f in node.get("fulfillments", []) or []:
                    if f.get("status") != "SUCCESS":
                        continue
                    if not delivered_at and f.get("deliveredAt"):
                        delivered_at = f["deliveredAt"]
                    if not fulfilled_at and f.get("createdAt"):
                        fulfilled_at = f["createdAt"]

                event_date = None
                event_source = None

                if delivered_at:
                    try:
                        d = dt.strptime(delivered_at[:19], "%Y-%m-%dT%H:%M:%S")
                        if d <= cutoff:
                            event_date = delivered_at
                            event_source = "delivered"
                        else:
                            skipped_too_recent += 1
                            continue
                    except Exception:
                        pass

                if not event_date and fulfilled_at:
                    try:
                        fd = dt.strptime(fulfilled_at[:19], "%Y-%m-%dT%H:%M:%S")
                        if fd <= cutoff_fulfilled:
                            event_date = fulfilled_at
                            event_source = "fulfilled_fallback"
                    except Exception:
                        pass

                if not event_date:
                    skipped_no_delivery += 1
                    continue

                eligible.append({
                    "order_name": order_name,
                    "to": to_email,
                    "first_name": first_name,
                    "delivered_at": event_date,
                    "source": event_source,
                })

            page_info = data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

    if dry_run:
        eligible_via_delivered = sum(1 for e in eligible if e.get("source") == "delivered")
        eligible_via_fallback = sum(1 for e in eligible if e.get("source") == "fulfilled_fallback")
        return {
            "ok": True,
            "dry_run": True,
            "since": since,
            "min_days_after_delivery": min_days,
            "min_days_after_fulfilled": min_days_fulfilled,
            "fetched": fetched,
            "eligible": len(eligible),
            "eligible_via_delivered": eligible_via_delivered,
            "eligible_via_fulfilled_fallback": eligible_via_fallback,
            "skipped": {
                "no_email": skipped_no_email,
                "no_delivery_date": skipped_no_delivery,
                "too_recent": skipped_too_recent,
                "excluded": skipped_excluded,
            },
            "sample": eligible[:10],
        }

    # Send emails (sequential to respect Gmail rate limits ~1 mail/sec)
    sent = 0
    already_sent = 0
    failed = 0
    db = get_db()
    if db:
        try:
            cur = db.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS google_review_emails (
                id SERIAL PRIMARY KEY,
                to_email TEXT NOT NULL,
                order_name TEXT NOT NULL,
                sent_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(to_email, order_name))""")
            db.commit(); cur.close(); db.close()
        except Exception:
            try: db.close()
            except: pass

    import asyncio
    for item in eligible:
        # Dedup check
        db2 = get_db()
        already = False
        if db2:
            try:
                c2 = db2.cursor()
                c2.execute("SELECT 1 FROM google_review_emails WHERE to_email=%s AND order_name=%s",
                           (item["to"], item["order_name"]))
                already = bool(c2.fetchone())
                c2.close(); db2.close()
            except Exception:
                try: db2.close()
                except: pass

        if already:
            already_sent += 1
            continue

        ok = await _send_google_review_email(item["to"], item["first_name"], item["order_name"])
        if ok:
            sent += 1
            db3 = get_db()
            if db3:
                try:
                    c3 = db3.cursor()
                    c3.execute("""INSERT INTO google_review_emails (to_email, order_name)
                        VALUES (%s, %s) ON CONFLICT (to_email, order_name) DO NOTHING""",
                        (item["to"], item["order_name"]))
                    db3.commit(); c3.close(); db3.close()
                except Exception:
                    try: db3.close()
                    except: pass
            await asyncio.sleep(1.0)  # throttle to stay under Gmail per-second limits
        else:
            failed += 1

    log_activity("google_review_backfill",
                 f"Backfill avis Google: {sent} envoyés, {already_sent} déjà fait, {failed} échecs",
                 {"since": since, "min_days": min_days, "fetched": fetched,
                  "sent": sent, "already_sent": already_sent, "failed": failed},
                 source="admin")

    return {
        "ok": True,
        "since": since,
        "min_days_after_delivery": min_days,
        "min_days_after_fulfilled": min_days_fulfilled,
        "fetched": fetched,
        "eligible": len(eligible),
        "eligible_via_delivered": sum(1 for e in eligible if e.get("source") == "delivered"),
        "eligible_via_fulfilled_fallback": sum(1 for e in eligible if e.get("source") == "fulfilled_fallback"),
        "sent": sent,
        "already_sent": already_sent,
        "failed": failed,
        "skipped": {
            "no_email": skipped_no_email,
            "no_delivery_date": skipped_no_delivery,
            "too_recent": skipped_too_recent,
        },
    }


# === GOOGLE REVIEWS POLLING (notif Gmail → store credit + product reviews) ===

@app.get("/api/admin/google-review/stats")
async def admin_google_review_stats(request: Request):
    """Return aggregate stats for the Google Reviews credit pipeline.

    Auth: X-API-Key header. Used by scheduled audit agents.
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(401, "Invalid API key")
    db = get_db()
    if not db:
        return {"ok": False, "error": "no db"}
    try:
        cur = db.cursor()
        cur.execute("""SELECT status, COUNT(*)::int FROM google_reviews_processed GROUP BY status""")
        by_status = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute("""SELECT order_name, reviewer_name, rating, credited_amount, processed_at
                       FROM google_reviews_processed
                       WHERE status='credited' ORDER BY processed_at DESC LIMIT 50""")
        credited_list = [
            {"order_name": r[0], "reviewer_name": r[1], "rating": r[2],
             "amount": float(r[3]) if r[3] is not None else None,
             "processed_at": r[4].isoformat() if r[4] else None}
            for r in cur.fetchall()
        ]
        cur.execute("SELECT COUNT(*)::int FROM google_reviews_processed")
        total = cur.fetchone()[0]
        # Backfill emails sent reference (from log_activity entries)
        cur.execute("""SELECT COUNT(*)::int FROM activity_log
                       WHERE action='google_review_backfill' AND created_at > NOW() - INTERVAL '30 days'""")
        try:
            backfill_runs = cur.fetchone()[0]
        except Exception:
            backfill_runs = None
        cur.close(); db.close()
        return {
            "ok": True,
            "total_processed": total,
            "by_status": by_status,
            "credited_sample": credited_list,
            "backfill_runs_last_30d": backfill_runs,
            "emails_sent_backfill_29_04": 155,
        }
    except Exception as e:
        try: db.close()
        except: pass
        return {"ok": False, "error": str(e)[:300]}


@app.get("/api/admin/gmail/debug")
async def admin_gmail_debug(request: Request, q: str = "newer_than:7d", limit: int = 10):
    """Diagnostic: list recent Gmail messages matching query. Returns subject + from + date."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(401, "Invalid API key")

    try:
        access_token = await _gmail_access_token()
        msgs = await _gmail_search(access_token, q, max_results=limit)
        out = []
        for m in msgs[:limit]:
            try:
                full = await _gmail_get_message(access_token, m["id"])
                hdrs = {h["name"].lower(): h["value"] for h in full.get("payload", {}).get("headers", [])}
                out.append({
                    "id": m["id"],
                    "subject": hdrs.get("subject", "")[:120],
                    "from": hdrs.get("from", "")[:80],
                    "date": hdrs.get("date", ""),
                })
            except Exception as e:
                out.append({"id": m["id"], "error": str(e)[:100]})
        return {"ok": True, "query": q, "count": len(msgs), "samples": out}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


async def _gmail_access_token() -> str:
    """Refresh Gmail OAuth access token (scope gmail.readonly).

    Uses GMAIL_* env vars (separate OAuth client from Google Ads/Analytics universal token,
    since the universal token does not include gmail.readonly scope).
    """
    refresh_token = os.getenv("GMAIL_REFRESH_TOKEN", "")
    client_id = os.getenv("GMAIL_CLIENT_ID", "")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET", "")
    if not refresh_token or not client_id or not client_secret:
        raise ValueError("Gmail OAuth not configured (GMAIL_REFRESH_TOKEN/GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET missing)")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        r.raise_for_status()
        return r.json()["access_token"]


async def _gmail_search(access_token: str, query: str, max_results: int = 50, include_spam_trash: bool = True) -> list:
    """Search Gmail messages, return list of {id, threadId}.

    include_spam_trash=True permet de couvrir les emails archivés/déplacés
    par les workflows automatiques info@ (qui poussent en Corbeille).
    """
    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    params = {"q": query, "maxResults": max_results}
    if include_spam_trash:
        params["includeSpamTrash"] = "true"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, params=params,
                        headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json().get("messages", []) or []


async def _gmail_get_message(access_token: str, msg_id: str) -> dict:
    """Get full Gmail message content."""
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(url, params={"format": "full"},
                        headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json()


def _decode_gmail_part(part: dict) -> str:
    """Recursively extract text from Gmail message parts (decodes base64url)."""
    import base64
    body_data = part.get("body", {}).get("data", "")
    out = ""
    if body_data:
        try:
            out = base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")
        except Exception:
            out = ""
    for sub in part.get("parts", []) or []:
        out += "\n" + _decode_gmail_part(sub)
    return out


def _parse_google_review_email(message: dict) -> dict:
    """Extract reviewer_name, rating, review_text, order_name from Google Business notif email.

    Supports:
    - Direct notif: "fernanda a laissé un avis sur PLANETEBEAUTY.COM"
    - Manual forward: "Fwd: fernanda a laissé un avis..."
    - Auto-reply: "NOUS AVONS PRIS EN COMPTE... Re: Fwd: fernanda a laissé un avis..."
    """
    import re
    headers = {h["name"].lower(): h["value"] for h in message.get("payload", {}).get("headers", [])}
    subject = headers.get("subject", "") or ""

    # Reviewer name = 1-2 words just before "a laissé un avis", skipping Re:/Fwd: prefixes
    reviewer_name = None
    parts = re.split(r"\s+a\s+laiss[ée]\s+un\s+avis", subject, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) > 1:
        before = parts[0].strip()
        words = [w for w in before.split() if w.lower().rstrip(":") not in ("re", "fwd", "fw", "tr")]
        if words:
            # Take last 1-2 words (most reviewer names are firstname OR firstname lastname)
            if len(words) >= 2 and words[-2][:1].isalpha():
                reviewer_name = f"{words[-2]} {words[-1]}"
            else:
                reviewer_name = words[-1]
            # Strip trailing punctuation
            reviewer_name = reviewer_name.strip(",.;:")

    # Body content (HTML + text parts)
    raw_text = _decode_gmail_part(message.get("payload", {}))

    # Rating: "nouvel avis avec X étoile(s)"
    rating_match = re.search(r"avec\s+(\d)\s+[ée]toile", raw_text, re.IGNORECASE)
    rating = int(rating_match.group(1)) if rating_match else None

    # Strip HTML
    clean_text = re.sub(r"<[^>]+>", " ", raw_text)
    clean_text = re.sub(r"&[a-z]+;", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # Order number — accept #XXXXX, # XXXXX, n°XXXXX
    order_match = re.search(r"#\s*(\d{5})", clean_text)
    order_name = "#" + order_match.group(1) if order_match else None

    # Review text — extract block between reviewer name and "Répondre" / "Afficher tous"
    review_text = None
    if reviewer_name:
        m = re.search(
            rf"{re.escape(reviewer_name)}.*?[ée]toile[s]?\s*(.+?)(?:R[ée]pondre|Afficher tous|Voir l|©)",
            clean_text, re.IGNORECASE | re.DOTALL
        )
        if m:
            review_text = m.group(1).strip()[:2000]

    return {
        "reviewer_name": reviewer_name,
        "rating": rating,
        "review_text": review_text,
        "order_name": order_name,
        "subject": subject,
    }


async def _send_google_review_credit_confirmation_email(to_email: str, first_name: str, order_name: str) -> bool:
    """Email confirmation client après crédit 5€ pour avis Google."""
    if not SMTP_HOST or not SMTP_USER or not SMTP_FROM_EMAIL or not to_email:
        return False
    greeting = f"Bonjour {first_name}" if first_name else "Bonjour"

    msg = MIMEMultipart("alternative")
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
    msg["To"] = to_email
    msg["Subject"] = "Merci pour votre avis Google — vos 5 € sont crédités"

    text_body = f"""{greeting},

Merci infiniment pour votre avis Google sur PlaneteBeauty.

Conformement a notre engagement, 5 EUR de credit magasin viennent d'etre credites sur votre compte client (commande {order_name}).

Vous pouvez l'utiliser sur votre prochaine commande, valable 30 jours, sur planetebeauty.com.

A tres bientot,
L'equipe PlaneteBeauty
"""

    html_body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;font-family:Arial,Helvetica,sans-serif;background:#f8f6f3;">
<div style="max-width:560px;margin:20px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,0.08);">
  <div style="background:#1a1a2e;padding:28px 32px;text-align:center;">
    <h1 style="color:#d4af37;margin:0;font-size:22px;letter-spacing:1px;">PLAN&Egrave;TEBEAUTY</h1>
  </div>
  <div style="padding:32px;">
    <h2 style="color:#1a1a2e;margin:0 0 16px;font-size:20px;">Merci pour votre avis</h2>
    <p style="color:#444;line-height:1.6;margin:0 0 16px;">{greeting},</p>
    <p style="color:#444;line-height:1.6;margin:0 0 24px;">Votre avis Google sur Plan&egrave;teBeauty compte &eacute;norm&eacute;ment. Merci d'avoir pris le temps de partager votre exp&eacute;rience.</p>
    <div style="border:1px solid #d4af37;border-radius:10px;padding:24px;margin:24px 0;background:#fdfaf2;text-align:center;">
      <p style="margin:0 0 6px;color:#1a1a2e;font-size:11px;letter-spacing:2.5px;font-weight:600;text-transform:uppercase;">Cr&eacute;dit&eacute;</p>
      <p style="margin:0 0 10px;color:#d4af37;font-size:32px;line-height:1;font-weight:700;">5,00 &euro;</p>
      <p style="margin:0;color:#666;font-size:13px;">de cr&eacute;dit magasin sur votre compte<br>commande <strong>{order_name}</strong> &middot; valable 30 jours</p>
    </div>
    <p style="color:#444;line-height:1.6;margin:0 0 12px;">Le cr&eacute;dit s'applique automatiquement lors de votre prochaine commande sur <a href="https://planetebeauty.com" style="color:#d4af37;">planetebeauty.com</a>.</p>
    <p style="color:#888;line-height:1.6;margin:24px 0 0;font-size:13px;">&Agrave; tr&egrave;s bient&ocirc;t,<br>L'&eacute;quipe Plan&egrave;teBeauty</p>
  </div>
  <div style="background:#f8f6f3;padding:16px 32px;text-align:center;border-top:1px solid #eee;">
    <p style="margin:0;color:#999;font-size:11px;">Plan&egrave;teBeauty &mdash; Parfumerie de niche</p>
  </div>
</div>
</body></html>"""

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                              username=SMTP_USER, password=SMTP_PASS, use_tls=False, start_tls=True)
        return True
    except Exception as e:
        logger.error(f"Google review credit confirmation email failed for {to_email}: {e}")
        return False


@app.post("/api/cron/google-reviews-poll")
async def cron_google_reviews_poll(request: Request):
    """Cron 30 min: lit Gmail forward des notifs Google Business → match #XXXXX → crédite 5€ + transfert avis produits.

    Source: emails forwardés depuis bmapbenoit@gmail.com (filtre Gmail) vers info@planetebeauty.com.
    From header: businessprofile-noreply@google.com.
    Idempotent via google_reviews_processed (gmail_message_id UNIQUE).
    """
    api_key = request.headers.get("X-API-Key", "")
    if api_key != "stella-mem-2026-planetebeauty":
        raise HTTPException(401, "Invalid API key")

    results = {"checked": 0, "new": 0, "credited": 0, "needs_manual": 0, "duplicates": 0, "errors": []}

    try:
        access_token = await _gmail_access_token()
    except Exception as e:
        return {"status": "error", "error": f"Gmail OAuth: {e}"}

    db = get_db()
    if not db:
        return {"status": "error", "error": "no db"}
    try:
        cur = db.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS google_reviews_processed (
            id SERIAL PRIMARY KEY,
            gmail_message_id TEXT UNIQUE NOT NULL,
            reviewer_name TEXT,
            rating INTEGER,
            review_text TEXT,
            order_name TEXT,
            customer_id TEXT,
            status TEXT,
            credited_amount NUMERIC(10,2),
            processed_at TIMESTAMP DEFAULT NOW()
        )""")
        db.commit()
    except Exception as e:
        try: db.close()
        except: pass
        return {"status": "error", "error": f"DB init: {e}"}

    try:
        # Match: auto-forwards (preserves From), manual forwards (subject preserved), even auto-replies
        # with original content quoted in body (subject contains the review pattern).
        # includeSpamTrash=True car workflow info@ pousse les emails traités en Corbeille.
        messages = await _gmail_search(
            access_token,
            'subject:"a laissé un avis" newer_than:60d',
            max_results=50
        )
    except Exception as e:
        cur.close(); db.close()
        return {"status": "error", "error": f"Gmail search: {e}"}

    results["checked"] = len(messages)
    gql_url = SHOPIFY_GRAPHQL_URL
    headers_gql = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30) as client:
        for msg_meta in messages:
            msg_id = msg_meta.get("id")
            if not msg_id:
                continue

            cur.execute("SELECT 1 FROM google_reviews_processed WHERE gmail_message_id=%s", (msg_id,))
            if cur.fetchone():
                results["duplicates"] += 1
                continue

            results["new"] += 1

            try:
                msg = await _gmail_get_message(access_token, msg_id)
                parsed = _parse_google_review_email(msg)
            except Exception as e:
                results["errors"].append(f"msg {msg_id} parse: {str(e)[:100]}")
                continue

            order_name = parsed.get("order_name")
            reviewer_name = parsed.get("reviewer_name") or "Client Google"
            rating = parsed.get("rating") or 5
            review_text = parsed.get("review_text") or ""

            if not order_name:
                cur.execute("""INSERT INTO google_reviews_processed
                    (gmail_message_id, reviewer_name, rating, review_text, status)
                    VALUES (%s,%s,%s,%s,'needs_manual_review') ON CONFLICT (gmail_message_id) DO NOTHING""",
                    (msg_id, reviewer_name, rating, review_text))
                db.commit()
                results["needs_manual"] += 1
                continue

            try:
                order_q = await client.post(gql_url, json={"query": f'''{{
                    orders(first:1, query:"name:{order_name}") {{
                        edges {{ node {{
                            id name displayFulfillmentStatus
                            customer {{ id email firstName }}
                            lineItems(first:20) {{ edges {{ node {{ title product {{ handle id }} }} }} }}
                        }} }}
                    }}
                }}'''}, headers=headers_gql)
                edges = order_q.json().get("data", {}).get("orders", {}).get("edges", []) or []
            except Exception as e:
                results["errors"].append(f"{order_name} lookup: {str(e)[:100]}")
                continue

            if not edges:
                cur.execute("""INSERT INTO google_reviews_processed
                    (gmail_message_id, reviewer_name, rating, review_text, order_name, status)
                    VALUES (%s,%s,%s,%s,%s,'order_not_found') ON CONFLICT (gmail_message_id) DO NOTHING""",
                    (msg_id, reviewer_name, rating, review_text, order_name))
                db.commit()
                results["needs_manual"] += 1
                continue

            order = edges[0]["node"]
            customer = order.get("customer") or {}
            customer_id = customer.get("id")
            customer_email = customer.get("email", "") or ""
            customer_first_name = customer.get("firstName", "") or ""

            if not customer_id:
                cur.execute("""INSERT INTO google_reviews_processed
                    (gmail_message_id, reviewer_name, rating, review_text, order_name, status)
                    VALUES (%s,%s,%s,%s,%s,'no_customer') ON CONFLICT (gmail_message_id) DO NOTHING""",
                    (msg_id, reviewer_name, rating, review_text, order_name))
                db.commit()
                results["needs_manual"] += 1
                continue

            cur.execute("SELECT 1 FROM google_reviews_processed WHERE order_name=%s AND status='credited'", (order_name,))
            if cur.fetchone():
                cur.execute("""INSERT INTO google_reviews_processed
                    (gmail_message_id, reviewer_name, rating, review_text, order_name, customer_id, status)
                    VALUES (%s,%s,%s,%s,%s,%s,'duplicate_order') ON CONFLICT (gmail_message_id) DO NOTHING""",
                    (msg_id, reviewer_name, rating, review_text, order_name, customer_id))
                db.commit()
                results["duplicates"] += 1
                continue

            try:
                credit_mutation = """mutation($id: ID!, $credit: StoreCreditAccountCreditInput!) {
                    storeCreditAccountCredit(id: $id, creditInput: $credit) {
                        storeCreditAccountTransaction { id }
                        userErrors { field message }
                    }
                }"""
                cr = await client.post(gql_url, json={
                    "query": credit_mutation,
                    "variables": {
                        "id": customer_id,
                        "credit": {"creditAmount": {"amount": "5.00", "currencyCode": "EUR"}}
                    }
                }, headers=headers_gql)
                cr_data = cr.json().get("data", {}).get("storeCreditAccountCredit", {}) or {}
                cr_errors = cr_data.get("userErrors", []) or []
            except Exception as e:
                results["errors"].append(f"{order_name} credit: {str(e)[:100]}")
                continue

            if cr_errors:
                msg_err = cr_errors[0].get("message", "")
                results["errors"].append(f"{order_name}: {msg_err}")
                cur.execute("""INSERT INTO google_reviews_processed
                    (gmail_message_id, reviewer_name, rating, review_text, order_name, customer_id, status)
                    VALUES (%s,%s,%s,%s,%s,%s,'credit_failed') ON CONFLICT (gmail_message_id) DO NOTHING""",
                    (msg_id, reviewer_name, rating, review_text, order_name, customer_id))
                db.commit()
                continue

            review_title = (review_text[:80] + "...") if len(review_text) > 80 else (review_text or "Avis Google")
            line_items = order.get("lineItems", {}).get("edges", []) or []
            for item in line_items:
                p = item.get("node", {}).get("product", {}) or {}
                p_handle = p.get("handle", "")
                p_id = p.get("id", "")
                if p_handle:
                    try:
                        cur.execute("""INSERT INTO product_reviews
                            (title, body, rating, review_date, source, curated, reviewer_name, reviewer_email, product_id, product_handle, order_number)
                            VALUES (%s,%s,%s,NOW(),%s,%s,%s,%s,%s,%s,%s)""",
                            (review_title, review_text, rating, "google_review", "ok",
                             reviewer_name, customer_email, p_id, p_handle, order_name))
                    except Exception as e:
                        logger.warning(f"Review insert {p_handle}: {e}")

            cur.execute("""INSERT INTO google_reviews_processed
                (gmail_message_id, reviewer_name, rating, review_text, order_name, customer_id, status, credited_amount)
                VALUES (%s,%s,%s,%s,%s,%s,'credited',%s) ON CONFLICT (gmail_message_id) DO NOTHING""",
                (msg_id, reviewer_name, rating, review_text, order_name, customer_id, 5.00))
            db.commit()

            results["credited"] += 1
            log_activity("google_review_credit",
                         f"Avis Google {order_name}: 5€ crédité à {customer_email}",
                         {"order_name": order_name, "reviewer": reviewer_name, "rating": rating, "amount": 5.00},
                         source="cron", customer_email=customer_email, order_name=order_name)

            try:
                await _send_google_review_credit_confirmation_email(customer_email, customer_first_name, order_name)
            except Exception as e:
                logger.warning(f"Confirmation email {customer_email}: {e}")

    cur.close(); db.close()

    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor()
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s, %s)",
                ("google-reviews-poll", json.dumps(results)))
            db2.commit(); cur2.close(); db2.close()
        except:
            try: db2.close()
            except: pass

    return {"status": "ok", **results}


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
    """Dashboard data for cashback tab in STELLA V8."""
    db = get_db()
    cb_settings = get_cashback_settings()
    stats = {
        "total_rewarded": 0, "total_amount": 0, "total_used": 0, "total_revoked": 0,
        "min_use": cb_settings["min_order_use"],
        "recent": [], "pending": [], "expiring_soon": []
    }
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Global stats
            cur.execute("SELECT COUNT(*) as c, COALESCE(SUM(cashback_amount),0) as total FROM cashback_rewards")
            row = cur.fetchone()
            stats["total_rewarded"] = row["c"]
            stats["total_amount"] = float(row["total"])
            cur.execute("SELECT COUNT(*) as c FROM cashback_rewards WHERE status='used'")
            stats["total_used"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM cashback_rewards WHERE status='revoked'")
            stats["total_revoked"] = cur.fetchone()["c"]

            # Recent cashback flow (last 30)
            cur.execute("""SELECT customer_email, order_name, cashback_base, cashback_amount,
                          store_credit_used, status, expires_at, created_at,
                          COALESCE(email_sent, false) as email_sent,
                          reminder_sent_at
                          FROM cashback_rewards ORDER BY created_at DESC LIMIT 30""")
            stats["recent"] = [dict(r) for r in cur.fetchall()]

            # Pending (active, not expired, not used)
            cur.execute("""SELECT customer_email, order_name, cashback_amount, expires_at, created_at
                          FROM cashback_rewards
                          WHERE status = 'active' AND expires_at > NOW()
                          ORDER BY expires_at ASC""")
            stats["pending"] = [dict(r) for r in cur.fetchall()]

            # Expiring within 7 days
            cur.execute("""SELECT customer_email, order_name, cashback_amount, expires_at,
                          COALESCE(reminder_sent_at IS NOT NULL, false) as reminder_sent
                          FROM cashback_rewards
                          WHERE status = 'active' AND expires_at > NOW()
                          AND expires_at < NOW() + INTERVAL '7 days'
                          ORDER BY expires_at ASC""")
            stats["expiring_soon"] = [dict(r) for r in cur.fetchall()]

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


# ══════════════════════ SHIPPING DISCOUNT CONFIG ══════════════════════

SHIPPING_DISCOUNT_ID = "gid://shopify/DiscountAutomaticNode/1880224399702"

@app.get("/api/shipping/settings")
async def get_shipping_settings():
    """Get shipping discount config from Shopify metafield."""
    defaults = {"threshold": 99, "amount": 5}
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={
                "query": f'{{ discountNode(id: "{SHIPPING_DISCOUNT_ID}") {{ metafield(namespace: "$app:pb-shipping-discount", key: "function-configuration") {{ value }} }} }}'
            }, headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            mf = r.json().get("data", {}).get("discountNode", {}).get("metafield")
            if mf and mf.get("value"):
                config = json.loads(mf["value"])
                return {"threshold": config.get("threshold", 99), "amount": config.get("amount", 5)}
    except Exception as e:
        logger.error(f"Shipping settings read error: {e}")
    return defaults


@app.post("/api/shipping/settings")
async def save_shipping_settings(request: Request):
    """Save shipping discount config to Shopify metafield on discount node."""
    body = await request.json()
    threshold = float(body.get("threshold", 99))
    amount = float(body.get("amount", 5))
    if threshold <= 0 or amount <= 0:
        raise HTTPException(400, "Valeurs invalides")

    config_json = json.dumps({"threshold": threshold, "amount": amount})
    try:
        mutation = """mutation($metafields: [MetafieldsSetInput!]!) {
          metafieldsSet(metafields: $metafields) {
            metafields { id namespace key }
            userErrors { field message }
          }
        }"""
        variables = {"metafields": [{
            "ownerId": SHIPPING_DISCOUNT_ID,
            "namespace": "$app:pb-shipping-discount",
            "key": "function-configuration",
            "type": "json",
            "value": config_json,
        }]}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": variables},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("metafieldsSet", {}).get("userErrors", [])
            if errors:
                return {"success": False, "errors": errors}

        # Update discount title to reflect new values
        title_mutation = f"""mutation {{
          discountAutomaticAppUpdate(id: "{SHIPPING_DISCOUNT_ID}", automaticAppDiscount: {{
            title: "-{int(amount)}€ sur la livraison dès {int(threshold)}€"
          }}) {{ automaticAppDiscount {{ discountId title }} userErrors {{ message }} }}
        }}"""
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(SHOPIFY_GRAPHQL_URL, json={"query": title_mutation},
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})

        log_activity("shipping_config", f"Shipping discount mis à jour: -{int(amount)}€ dès {int(threshold)}€",
                     {"threshold": threshold, "amount": amount}, source="api")
        return {"success": True, "threshold": threshold, "amount": amount}
    except Exception as e:
        logger.error(f"Shipping settings save error: {e}")
        return {"success": False, "error": str(e)}


# ══════════════════════ PROMO CODES CONFIG ══════════════════════

PROMO_DISCOUNT_ID = "gid://shopify/DiscountAutomaticNode/1880205492566"
PROMO_MF_NS = "planete-beaute"
PROMO_MF_KEY = "discount-codes-config"

DEFAULT_PROMO_CONFIG = {
    "codes": {
        "PB580": {"percent": 5.0, "minSubtotal": 80.0, "message": "-5% avec le code PB580"},
        "PB10180": {"percent": 10.0, "minSubtotal": 180.0, "message": "-10% avec le code PB10180"},
    },
    "excludedVendors": ["Creed", "Roja Parfums", "Clive Christian"]
}


@app.get("/api/promo/settings")
async def get_promo_settings():
    """Get promo codes config from Shopify metafield."""
    try:
        query = f"""{{ discountNode(id: "{PROMO_DISCOUNT_ID}") {{
            metafield(namespace: "{PROMO_MF_NS}", key: "{PROMO_MF_KEY}") {{ value }}
        }} }}"""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": query},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            mf = r.json().get("data", {}).get("discountNode", {}).get("metafield")
            if mf and mf.get("value"):
                return json.loads(mf["value"])
    except Exception as e:
        logger.error(f"Promo settings read error: {e}")
    return DEFAULT_PROMO_CONFIG


@app.post("/api/promo/settings")
async def save_promo_settings(request: Request):
    """Save promo codes config to Shopify metafield."""
    body = await request.json()
    codes = body.get("codes", {})
    excluded_vendors = body.get("excludedVendors", [])

    if not codes:
        raise HTTPException(400, "Au moins un code requis")

    # Validate each code
    for code_name, rule in codes.items():
        if not rule.get("percent") or not rule.get("minSubtotal"):
            raise HTTPException(400, f"Code {code_name}: percent et minSubtotal requis")

    config = {"codes": codes, "excludedVendors": excluded_vendors}
    config_json = json.dumps(config)

    try:
        mutation = """mutation($metafields: [MetafieldsSetInput!]!) {
          metafieldsSet(metafields: $metafields) {
            metafields { id namespace key }
            userErrors { field message }
          }
        }"""
        variables = {"metafields": [{
            "ownerId": PROMO_DISCOUNT_ID,
            "namespace": PROMO_MF_NS,
            "key": PROMO_MF_KEY,
            "type": "json",
            "value": config_json,
        }]}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": variables},
                           headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            resp = r.json()
            errors = resp.get("data", {}).get("metafieldsSet", {}).get("userErrors", [])
            if errors:
                return {"success": False, "errors": errors}

            # Also sync to SHOP metafield so Liquid can read it
            shop_r = await c.post(SHOPIFY_GRAPHQL_URL, json={"query": "{ shop { id } }"},
                                 headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})
            shop_id = shop_r.json().get("data", {}).get("shop", {}).get("id", "")
            if shop_id:
                shop_mf_vars = {"metafields": [{
                    "ownerId": shop_id,
                    "namespace": "planete-beaute",
                    "key": "promo-codes",
                    "type": "json",
                    "value": config_json,
                }]}
                await c.post(SHOPIFY_GRAPHQL_URL, json={"query": mutation, "variables": shop_mf_vars},
                            headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"})

        code_count = len(codes)
        log_activity("promo_config", f"Codes promo mis à jour: {code_count} codes actifs (shop metafield synced)",
                     {"codes": list(codes.keys())}, source="api")
        return {"success": True, "codes_count": code_count}
    except Exception as e:
        logger.error(f"Promo settings save error: {e}")
        return {"success": False, "error": str(e)}


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

        # Recent 50
        cur.execute("""
            SELECT email, product_title, product_handle, subscribed_at, status
            FROM bis_subscriptions
            ORDER BY subscribed_at DESC LIMIT 50
        """)
        recent = cur.fetchall()

        # Total notified + total subscriptions
        cur.execute("SELECT COUNT(*) as c FROM bis_subscriptions WHERE status='notified'")
        total_notified = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM bis_subscriptions")
        total_subs = cur.fetchone()["c"]

        cur.close(); db.close()

        # Serialize datetimes
        for p in products:
            if p.get("last_sub"):
                p["last_sub"] = p["last_sub"].isoformat()
        for r in recent:
            if r.get("subscribed_at"):
                r["subscribed_at"] = r["subscribed_at"].isoformat()

        return {
            "total_active": total,
            "total_products": len(products),
            "total_notified": total_notified,
            "total_subscriptions": total_subs,
            "products": products,
            "recent": recent
        }
    except Exception as e:
        try: db.close()
        except: pass
        logger.error(f"BIS dashboard: {e}")
        return {"total_active": 0, "total_products": 0, "total_notified": 0, "total_subscriptions": 0, "products": [], "recent": [], "error": str(e)}


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
    """Check SMTP configuration status."""
    configured = bool(SMTP_HOST and SMTP_USER and SMTP_FROM_EMAIL)
    return {
        "status": "ok" if configured else "not_configured",
        "configured": configured,
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
    """KPIs temps reel — STELLA OS dashboard complet."""
    result = {"revenue_today": 0, "orders_today": 0, "avg_order_value": 0,
              "cashback_generated_today": 0, "cashback_active_total": 0, "cashback_expiring_soon": 0,
              "bis_active": 0, "tryme_active": 0, "reviews_pending": 0,
              "catalogue_issues": 0, "quiz_completions_today": 0, "quiz_views_today": 0}
    try:
        from zoneinfo import ZoneInfo
        tz_paris = ZoneInfo("Europe/Paris")
        today_paris = datetime.now(tz_paris).replace(hour=0, minute=0, second=0, microsecond=0)
        today_iso = today_paris.strftime("%Y-%m-%dT%H:%M:%S%z")
        today_iso = today_iso[:-2] + ":" + today_iso[-2:]  # +0200 → +02:00
        query = f'''{{ orders(first: 250, query: "created_at:>='{today_iso}'") {{ edges {{ node {{ name totalPriceSet {{ shopMoney {{ amount }} }} displayFinancialStatus }} }} }} }}'''
        data = await shopify_graphql(query)
        orders = data.get("data", {}).get("orders", {}).get("edges", [])
        paid = [o for o in orders if o["node"].get("displayFinancialStatus") in ["PAID", "PARTIALLY_PAID", "PARTIALLY_REFUNDED"]]
        result["orders_today"] = len(paid)
        result["revenue_today"] = round(sum(float(o["node"]["totalPriceSet"]["shopMoney"]["amount"]) for o in paid), 2)
        result["avg_order_value"] = round(result["revenue_today"] / max(result["orders_today"], 1), 2)
    except Exception as e:
        logger.warning(f"KPI shopify error: {e}")

    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            # Cashback today
            cur.execute("SELECT COALESCE(SUM((details->>'cashback_amount')::numeric), 0) as total FROM activity_log WHERE type='cashback_credit' AND timestamp >= CURRENT_DATE")
            result["cashback_generated_today"] = float(cur.fetchone()["total"])
            # Cashback encours actifs (montant total)
            cur.execute("SELECT COALESCE(SUM(cashback_amount), 0) as total, COUNT(*) as cnt FROM cashback_rewards WHERE status='active' AND expires_at > NOW()")
            row = cur.fetchone()
            result["cashback_active_total"] = float(row["total"])
            # Cashback expirant dans 7 jours
            cur.execute("SELECT COUNT(*) as c FROM cashback_rewards WHERE status='active' AND expires_at BETWEEN NOW() AND NOW() + INTERVAL '7 days'")
            result["cashback_expiring_soon"] = cur.fetchone()["c"]
            # BIS actifs
            cur.execute("SELECT COUNT(*) as c FROM bis_subscriptions WHERE status='active'")
            result["bis_active"] = cur.fetchone()["c"]
            # Try Me actifs (pending = non utilises, non expires)
            cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='pending' AND expires_at > NOW()")
            result["tryme_active"] = cur.fetchone()["c"]
            # Avis en attente
            cur.execute("SELECT COUNT(*) as c FROM product_reviews WHERE curated='pending'")
            result["reviews_pending"] = cur.fetchone()["c"]
            # Problemes catalogue (dernier audit)
            cur.execute("SELECT result_data FROM cron_results WHERE cron_name='audit-qualite' ORDER BY executed_at DESC LIMIT 1")
            audit_row = cur.fetchone()
            if audit_row:
                audit_data = audit_row["result_data"]
                if isinstance(audit_data, str):
                    try: audit_data = json.loads(audit_data)
                    except: audit_data = {}
                result["catalogue_issues"] = audit_data.get("total_issues", len(audit_data.get("issues", [])))
            cur.close(); db.close()
        except Exception as e:
            logger.warning(f"KPI db error: {e}")
            try: db.close()
            except: pass

    # Quiz stats from Redis
    rc = get_redis()
    if rc:
        try:
            today_key = datetime.now().strftime("%Y-%m-%d")
            result["quiz_views_today"] = int(rc.get(f"quiz:quiz_view:{today_key}") or 0)
            result["quiz_completions_today"] = int(rc.get(f"quiz:quiz_complete:{today_key}") or 0)
        except: pass
    return result

# ══════════════════════ MARKETING INTELLIGENCE ══════════════════════

GADS_MCC_ID = os.getenv("GADS_MCC_ID", "9032082552")
GA4_PROPERTY_ID = os.getenv("GA4_PROPERTY_ID", "427142120")
GMERCHANT_ID = os.getenv("GMERCHANT_ID", "277377202")
GSC_SITE = os.getenv("GSC_SITE", "https://www.planetebeauty.com/")
GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "")

async def _google_access_token() -> str:
    """Refresh Google OAuth universal access token."""
    token = GOOGLE_REFRESH_TOKEN or GADS_REFRESH_TOKEN
    if not token: raise ValueError("No Google refresh token configured")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post("https://oauth2.googleapis.com/token", data={
            "client_id": GADS_CLIENT_ID,
            "client_secret": GADS_CLIENT_SECRET,
            "refresh_token": token,
            "grant_type": "refresh_token"
        })
        r.raise_for_status()
        return r.json()["access_token"]

async def _gads_query(access_token: str, gaql: str) -> dict:
    """Execute a GAQL query against Google Ads REST API v20."""
    url = f"https://googleads.googleapis.com/v20/customers/{GADS_CUSTOMER_ID}/googleAds:search"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json={"query": gaql}, headers={
            "Authorization": f"Bearer {access_token}",
            "developer-token": GADS_DEVELOPER_TOKEN,
            "login-customer-id": GADS_MCC_ID,
        })
        r.raise_for_status()
        return r.json()

async def _ga4_report(access_token: str, body: dict) -> dict:
    """Run a GA4 Data API report."""
    url = f"https://analyticsdata.googleapis.com/v1beta/properties/{GA4_PROPERTY_ID}:runReport"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json=body, headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json()

async def _gsc_query(access_token: str, body: dict) -> dict:
    """Run a Search Console query."""
    from urllib.parse import quote
    url = f"https://www.googleapis.com/webmasters/v3/sites/{quote(GSC_SITE, safe='')}/searchAnalytics/query"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(url, json=body, headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json()

async def _merchant_get(access_token: str, path: str) -> dict:
    """Call Merchant Center API."""
    url = f"https://shoppingcontent.googleapis.com/content/v2.1/{GMERCHANT_ID}/{path}"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(url, headers={"Authorization": f"Bearer {access_token}"})
        r.raise_for_status()
        return r.json()

def _paris_month_range():
    """Return first_ymd, today_ymd, first_iso (Shopify-compatible), days_elapsed, days_in_month."""
    from zoneinfo import ZoneInfo
    import calendar
    tz = ZoneInfo("Europe/Paris")
    now = datetime.now(tz)
    first = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    first_iso = first.strftime("%Y-%m-%dT%H:%M:%S%z")
    first_iso = first_iso[:-2] + ":" + first_iso[-2:]
    return first.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), first_iso, now.day, calendar.monthrange(now.year, now.month)[1], now.strftime("%B %Y")

@app.get("/api/marketing/intelligence")
async def marketing_intelligence():
    """Full marketing intelligence — Ads, GA4, Search Console, Merchant Center."""
    rc = get_redis()
    if rc:
        try:
            cached = rc.get("dashboard:marketing:intel")
            if cached: return json.loads(cached)
        except: pass

    first_ymd, today_ymd, first_iso, days_elapsed, days_in_month, month_label = _paris_month_range()
    token = await _google_access_token()

    result = {"month_label": month_label, "days_elapsed": days_elapsed, "days_in_month": days_in_month,
              "budget": {}, "campaigns": [], "asset_groups": [], "conversion_tracking": {},
              "funnel": {}, "channels": [], "seo": {}, "merchant": {}, "recommendations": [], "errors": []}

    # ── 1. ADS: Budget ratio + daily breakdown ──
    try:
        gads_daily = await _gads_query(token, f"SELECT metrics.cost_micros, segments.date FROM customer WHERE segments.date BETWEEN '{first_ymd}' AND '{today_ymd}'")
        daily_spend = {}
        total_micros = 0
        for row in gads_daily.get("results", []):
            cost = int(row.get("metrics", {}).get("costMicros", 0))
            day = row.get("segments", {}).get("date", "")
            total_micros += cost
            if day: daily_spend[day] = daily_spend.get(day, 0) + cost
        ads_spend = round(total_micros / 1_000_000, 2)

        # Shopify revenue same period
        shopify_q = f'''{{ orders(first: 250, query: "created_at:>='{first_iso}'") {{ edges {{ node {{ createdAt totalPriceSet {{ shopMoney {{ amount }} }} displayFinancialStatus }} }} }} }}'''
        shopify_data = await shopify_graphql(shopify_q)
        daily_rev = {}
        total_rev = 0
        for e in shopify_data.get("data", {}).get("orders", {}).get("edges", []):
            n = e["node"]
            if n.get("displayFinancialStatus") in {"PAID", "PARTIALLY_PAID", "PARTIALLY_REFUNDED"}:
                amt = float(n["totalPriceSet"]["shopMoney"]["amount"])
                total_rev += amt
                day = n["createdAt"][:10]
                daily_rev[day] = daily_rev.get(day, 0) + amt
        revenue = round(total_rev, 2)
        ratio = round(ads_spend / revenue * 100, 2) if revenue > 0 else 0
        proj_spend = round(ads_spend / max(days_elapsed, 1) * days_in_month, 2)
        proj_rev = round(revenue / max(days_elapsed, 1) * days_in_month, 2)
        proj_ratio = round(proj_spend / proj_rev * 100, 2) if proj_rev > 0 else 0

        result["budget"] = {
            "ads_spend": ads_spend, "revenue": revenue, "ratio_pct": ratio,
            "threshold_pct": 5.0, "status": "danger" if ratio >= 5 else "warning" if ratio >= 4 else "ok",
            "projected_spend": proj_spend, "projected_revenue": proj_rev, "projected_ratio_pct": proj_ratio,
            "daily": [{"date": d, "spend": round(v / 1_000_000, 2), "revenue": round(daily_rev.get(d, 0), 2)}
                      for d, v in sorted(daily_spend.items())]
        }
    except Exception as e:
        logger.warning(f"Marketing intel budget error: {e}")
        result["errors"].append(f"Budget: {str(e)[:200]}")

    # ── 2. ADS: Campaigns performance ──
    try:
        camps = await _gads_query(token, f"""
            SELECT campaign.name, campaign.status, campaign.advertising_channel_type,
                   metrics.cost_micros, metrics.clicks, metrics.impressions,
                   metrics.conversions, metrics.conversions_value, metrics.average_cpc, metrics.ctr,
                   metrics.cost_per_conversion
            FROM campaign WHERE segments.date DURING THIS_MONTH AND campaign.status != REMOVED
            ORDER BY metrics.cost_micros DESC""")
        for r in camps.get("results", []):
            c, m = r.get("campaign", {}), r.get("metrics", {})
            cost = int(m.get("costMicros", 0)) / 1e6
            conv_val = float(m.get("conversionsValue", 0))
            roas = round(conv_val / cost, 1) if cost > 0 else 0
            result["campaigns"].append({
                "name": c.get("name", ""), "status": c.get("status", ""),
                "type": c.get("advertisingChannelType", ""),
                "cost": round(cost, 2), "clicks": int(m.get("clicks", 0)),
                "impressions": int(m.get("impressions", 0)),
                "conversions": round(float(m.get("conversions", 0)), 1),
                "conv_value": round(conv_val, 2), "roas": roas,
                "cpc": round(int(m.get("averageCpc", 0)) / 1e6, 2),
                "ctr": round(float(m.get("ctr", 0)) * 100, 2),
                "cpa": round(int(m.get("costPerConversion", 0)) / 1e6, 2) if m.get("costPerConversion") else 0
            })
    except Exception as e:
        result["errors"].append(f"Campaigns: {str(e)[:200]}")

    # ── 3. ADS: Asset groups (PMax breakdown) ──
    try:
        ags = await _gads_query(token, f"""
            SELECT asset_group.name, asset_group.status,
                   metrics.cost_micros, metrics.conversions, metrics.conversions_value,
                   metrics.clicks, metrics.impressions
            FROM asset_group WHERE segments.date DURING THIS_MONTH AND campaign.status = ENABLED
            ORDER BY metrics.cost_micros DESC LIMIT 20""")
        for r in ags.get("results", []):
            ag, m = r.get("assetGroup", {}), r.get("metrics", {})
            cost = int(m.get("costMicros", 0)) / 1e6
            val = float(m.get("conversionsValue", 0))
            result["asset_groups"].append({
                "name": ag.get("name", ""), "status": ag.get("status", ""),
                "cost": round(cost, 2), "clicks": int(m.get("clicks", 0)),
                "impressions": int(m.get("impressions", 0)),
                "conversions": round(float(m.get("conversions", 0)), 1),
                "conv_value": round(val, 2),
                "roas": round(val / cost, 1) if cost > 0 else 0
            })
    except Exception as e:
        result["errors"].append(f"Asset groups: {str(e)[:200]}")

    # ── 4. ADS: Conversion tracking audit ──
    try:
        conv_actions = await _gads_query(token, """
            SELECT conversion_action.name, conversion_action.type, conversion_action.status,
                   conversion_action.category, conversion_action.include_in_conversions_metric
            FROM conversion_action WHERE conversion_action.status = ENABLED""")
        primary = []
        secondary = []
        for r in conv_actions.get("results", []):
            ca = r.get("conversionAction", {})
            entry = {"name": ca.get("name", ""), "type": ca.get("type", ""),
                     "category": ca.get("category", ""), "primary": ca.get("includeInConversionsMetric", False)}
            if entry["primary"]: primary.append(entry)
            else: secondary.append(entry)
        result["conversion_tracking"] = {
            "primary_actions": primary, "secondary_actions": secondary,
            "total_enabled": len(primary) + len(secondary),
            "purchase_tracking": any(a["category"] == "PURCHASE" for a in primary)
        }
    except Exception as e:
        result["errors"].append(f"Conversion tracking: {str(e)[:200]}")

    # ── 5. GA4: Funnel e-commerce + channels ──
    try:
        funnel = await _ga4_report(token, {
            "dateRanges": [{"startDate": first_ymd, "endDate": today_ymd}],
            "metrics": [{"name": "sessions"}, {"name": "addToCarts"}, {"name": "checkouts"},
                        {"name": "ecommercePurchases"}, {"name": "purchaseRevenue"},
                        {"name": "totalUsers"}, {"name": "bounceRate"}],
            "dimensions": [{"name": "sessionDefaultChannelGroup"}]
        })
        totals = {"sessions": 0, "atc": 0, "checkout": 0, "purchases": 0, "revenue": 0}
        for r in funnel.get("rows", []):
            ch = r["dimensionValues"][0]["value"]
            vals = [v["value"] for v in r["metricValues"]]
            sessions, atc, checkout, purchases = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])
            rev, users, bounce = float(vals[4]), int(vals[5]), float(vals[6])
            totals["sessions"] += sessions; totals["atc"] += atc
            totals["checkout"] += checkout; totals["purchases"] += purchases; totals["revenue"] += rev
            result["channels"].append({
                "name": ch, "sessions": sessions, "users": users, "atc": atc,
                "checkout": checkout, "purchases": purchases, "revenue": round(rev, 2),
                "bounce_rate": round(bounce * 100, 1),
                "atc_rate": round(atc / max(sessions, 1) * 100, 1),
                "conv_rate": round(purchases / max(sessions, 1) * 100, 2)
            })
        result["channels"].sort(key=lambda x: -x["sessions"])
        # Funnel totals + drop-off rates
        result["funnel"] = {
            **totals, "revenue": round(totals["revenue"], 2),
            "atc_rate": round(totals["atc"] / max(totals["sessions"], 1) * 100, 1),
            "checkout_rate": round(totals["checkout"] / max(totals["atc"], 1) * 100, 1),
            "purchase_rate": round(totals["purchases"] / max(totals["checkout"], 1) * 100, 1),
            "overall_conv": round(totals["purchases"] / max(totals["sessions"], 1) * 100, 2)
        }
    except Exception as e:
        result["errors"].append(f"GA4 funnel: {str(e)[:200]}")

    # ── 6. GA4: Tracking audit (events firing correctly?) ──
    try:
        events = await _ga4_report(token, {
            "dateRanges": [{"startDate": first_ymd, "endDate": today_ymd}],
            "metrics": [{"name": "eventCount"}, {"name": "purchaseRevenue"}],
            "dimensions": [{"name": "eventName"}],
            "dimensionFilter": {"filter": {"fieldName": "eventName", "inListFilter": {
                "values": ["purchase", "add_to_cart", "begin_checkout", "view_item",
                           "add_payment_info", "add_shipping_info", "page_view", "session_start"]}}}
        })
        event_counts = {}
        for r in events.get("rows", []):
            ev = r["dimensionValues"][0]["value"]
            event_counts[ev] = int(r["metricValues"][0]["value"])
        result["funnel"]["events"] = event_counts
        # Check tracking health
        tracking_issues = []
        if event_counts.get("purchase", 0) == 0: tracking_issues.append("Aucun event purchase detecte")
        if event_counts.get("add_to_cart", 0) == 0: tracking_issues.append("Aucun event add_to_cart detecte")
        if event_counts.get("begin_checkout", 0) > 0 and event_counts.get("add_payment_info", 0) == 0:
            tracking_issues.append("begin_checkout present mais add_payment_info absent")
        result["funnel"]["tracking_issues"] = tracking_issues
    except Exception as e:
        result["errors"].append(f"GA4 events: {str(e)[:200]}")

    # ── 7. Search Console: SEO performance ──
    try:
        # Top queries
        gsc_queries = await _gsc_query(token, {
            "startDate": first_ymd, "endDate": today_ymd,
            "dimensions": ["query"], "rowLimit": 20, "type": "web"
        })
        # Top pages
        gsc_pages = await _gsc_query(token, {
            "startDate": first_ymd, "endDate": today_ymd,
            "dimensions": ["page"], "rowLimit": 15, "type": "web"
        })
        result["seo"] = {
            "top_queries": [{"query": r["keys"][0], "clicks": r["clicks"], "impressions": r["impressions"],
                             "ctr": round(r["ctr"] * 100, 1), "position": round(r["position"], 1)}
                            for r in gsc_queries.get("rows", [])],
            "top_pages": [{"page": r["keys"][0].replace(GSC_SITE.rstrip("/"), ""),
                           "clicks": r["clicks"], "impressions": r["impressions"],
                           "ctr": round(r["ctr"] * 100, 1), "position": round(r["position"], 1)}
                          for r in gsc_pages.get("rows", [])],
        }
    except Exception as e:
        result["errors"].append(f"Search Console: {str(e)[:200]}")

    # ── 8. Merchant Center: Product health ──
    try:
        statuses = await _merchant_get(token, "productstatuses?maxResults=250")
        products = statuses.get("resources", [])
        ok = disapproved = warnings_count = 0
        issues_map = {}
        for p in products:
            has_disapproval = any(ds.get("status") == "disapproved" for ds in p.get("destinationStatuses", []))
            if has_disapproval:
                disapproved += 1
            else:
                item_issues = p.get("itemLevelIssues", [])
                if item_issues:
                    warnings_count += 1
                    for iss in item_issues:
                        code = iss.get("code", "unknown")
                        issues_map[code] = issues_map.get(code, 0) + 1
                else:
                    ok += 1
        result["merchant"] = {
            "total_products": len(products), "ok": ok,
            "warnings": warnings_count, "disapproved": disapproved,
            "top_issues": [{"code": k, "count": v} for k, v in sorted(issues_map.items(), key=lambda x: -x[1])[:10]]
        }
    except Exception as e:
        result["errors"].append(f"Merchant Center: {str(e)[:200]}")

    # ── 9. Auto-generate recommendations ──
    recs = []
    b = result.get("budget", {})
    if b.get("ratio_pct", 0) >= 4:
        recs.append({"priority": "high", "area": "budget", "text": f"Ratio Ads/CA a {b['ratio_pct']}% — proche du seuil 5%. Surveiller."})
    for camp in result.get("campaigns", []):
        if camp["status"] == "ENABLED" and camp["cost"] > 50 and camp["roas"] < 3:
            recs.append({"priority": "high", "area": "campaign", "text": f"Campagne '{camp['name']}' ROAS faible ({camp['roas']}x). Optimiser ou pauser."})
        if camp["status"] == "ENABLED" and camp["cost"] > 50 and camp["roas"] > 8:
            recs.append({"priority": "medium", "area": "campaign", "text": f"Campagne '{camp['name']}' ROAS excellent ({camp['roas']}x). Potentiel d'augmentation budget."})
    ct = result.get("conversion_tracking", {})
    if not ct.get("purchase_tracking"):
        recs.append({"priority": "critical", "area": "tracking", "text": "Aucune action de conversion PURCHASE en primaire. Le suivi d'achat est peut-etre casse."})
    for ch in result.get("channels", []):
        if ch["sessions"] > 100 and ch["bounce_rate"] > 80:
            recs.append({"priority": "medium", "area": "traffic", "text": f"Canal '{ch['name']}' : {ch['bounce_rate']}% bounce sur {ch['sessions']} sessions. Trafic de mauvaise qualite."})
        if ch["sessions"] > 50 and ch["atc"] > 10 and ch["purchases"] == 0:
            recs.append({"priority": "medium", "area": "funnel", "text": f"Canal '{ch['name']}' : {ch['atc']} ATC mais 0 achat. Probleme checkout?"})
    m = result.get("merchant", {})
    if m.get("disapproved", 0) > 0:
        recs.append({"priority": "high", "area": "merchant", "text": f"{m['disapproved']} produits disapproved dans Merchant Center. Visibilite Shopping reduite."})
    if m.get("warnings", 0) > 20:
        recs.append({"priority": "medium", "area": "merchant", "text": f"{m['warnings']} produits avec warnings Merchant Center. Corriger pour ameliorer qualite du flux."})
    funnel = result.get("funnel", {})
    if funnel.get("checkout_rate", 100) < 50:
        recs.append({"priority": "medium", "area": "funnel", "text": f"Taux ATC→Checkout = {funnel['checkout_rate']}%. Friction possible dans le panier."})
    if funnel.get("purchase_rate", 100) < 30:
        recs.append({"priority": "medium", "area": "funnel", "text": f"Taux Checkout→Achat = {funnel['purchase_rate']}%. Friction au paiement."})
    result["recommendations"] = sorted(recs, key=lambda x: {"critical": 0, "high": 1, "medium": 2}.get(x["priority"], 3))

    # Cache 15 min
    if rc:
        try: rc.setex("dashboard:marketing:intel", 900, json.dumps(result))
        except: pass
    return result

# Legacy endpoint — redirects
@app.get("/api/google-ads/dashboard")
async def google_ads_dashboard():
    """Legacy — returns budget section from marketing intelligence."""
    data = await marketing_intelligence()
    b = data.get("budget", {})
    b["month_label"] = data.get("month_label", "")
    b["days_elapsed"] = data.get("days_elapsed", 0)
    b["days_in_month"] = data.get("days_in_month", 0)
    b["error"] = data.get("errors", [None])[0] if data.get("errors") else None
    return b

# ══════════════════════ ORDERS DASHBOARD ══════════════════════

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
        from zoneinfo import ZoneInfo
        tz_paris = ZoneInfo("Europe/Paris")
        today_paris = datetime.now(tz_paris).replace(hour=0, minute=0, second=0, microsecond=0)
        today_iso = today_paris.strftime("%Y-%m-%dT%H:%M:%S%z")
        today_iso = today_iso[:-2] + ":" + today_iso[-2:]
        query = f'''{{ orders(first: 50, query: "created_at:>='{today_iso}'", sortKey: CREATED_AT, reverse: true) {{ edges {{ node {{ id name createdAt totalPriceSet {{ shopMoney {{ amount currencyCode }} }} displayFinancialStatus displayFulfillmentStatus customer {{ email displayName }} tags }} }} }} }}'''
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
                result["products_with_issues"] = audit_data.get("total_issues", audit_data.get("issues_count", len(audit_data.get("issues", []))))
                result["issues"] = audit_data.get("issues", [])[:50]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass
    try:
        query = '{ productsCount(limit: null) { count } }'
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

# ══════════════════════════════════════════════════════════════
# STELLA OS — Operations Management System (07/04/2026)
# Source de vérité : PostgreSQL. Rollbacks 24/7 via Railway.
# ══════════════════════════════════════════════════════════════

def ops_log(entity_code, actor, action, operation_id=None, severity="info", details=None):
    """Log dans ops_audit_log."""
    db = get_db()
    if not db: return
    try:
        cur = db.cursor()
        cur.execute("""INSERT INTO ops_audit_log (entity_code, actor, action, operation_id, severity, details)
            VALUES (%s,%s,%s,%s,%s,%s)""", (entity_code, actor, action, operation_id, severity, json.dumps(details or {})))
        db.commit(); cur.close(); db.close()
    except Exception as e:
        logger.error(f"ops_log error: {e}")
        try: db.close()
        except: pass

# ── ENDPOINTS API /api/ops/* ──

@app.post("/api/ops/create")
async def ops_create(request: Request):
    """Créer une opération (promo, deploy, feature, fix, config)."""
    body = await request.json()
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor()
        cur.execute("""INSERT INTO operations (entity_code, name, type, status, expires_at, created_by, description, metadata)
            VALUES (%s,%s,%s,'active',%s,%s,%s,%s) RETURNING id""",
            (body.get("entity_code", "bhtc_fr"), body["name"], body["type"],
             body.get("expires_at"), body.get("created_by", "stella"),
             body.get("description", ""), json.dumps(body.get("metadata", {}))))
        op_id = cur.fetchone()[0]
        # Sauvegarder les fichiers modifiés
        for f in body.get("files", []):
            cur.execute("""INSERT INTO operations_files (operation_id, file_path, file_type, state_before, state_after, rollback_action)
                VALUES (%s,%s,%s,%s,%s,%s)""",
                (op_id, f["file_path"], f.get("file_type", ""),
                 f.get("state_before", ""), f.get("state_after", ""), f.get("rollback_action", "")))
        db.commit(); cur.close(); db.close()
        ops_log(body.get("entity_code", "bhtc_fr"), body.get("created_by", "stella"),
                "operation_created", op_id, "info", {"name": body["name"], "type": body["type"]})
        return {"success": True, "operation_id": op_id}
    except Exception as e:
        logger.error(f"ops_create error: {e}")
        try: db.close()
        except: pass
        return {"error": str(e)}

@app.get("/api/ops/active")
async def ops_active(entity: str = None):
    """Lister les opérations actives."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if entity:
            cur.execute("SELECT * FROM operations WHERE status='active' AND entity_code=%s ORDER BY created_at DESC", (entity,))
        else:
            cur.execute("SELECT * FROM operations WHERE status='active' ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"operations": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"ops_active error: {e}")
        return {"error": str(e)}

@app.get("/api/ops/detail/{op_id}")
async def ops_detail(op_id: int):
    """Détails d'une opération avec ses fichiers et audit."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM operations WHERE id=%s", (op_id,))
        op = cur.fetchone()
        if not op:
            cur.close(); db.close()
            return {"error": "Operation not found"}
        cur.execute("SELECT * FROM operations_files WHERE operation_id=%s", (op_id,))
        files = cur.fetchall()
        cur.execute("SELECT * FROM ops_audit_log WHERE operation_id=%s ORDER BY timestamp DESC LIMIT 50", (op_id,))
        audit = cur.fetchall()
        cur.close(); db.close()
        return {"operation": dict(op), "files": [dict(f) for f in files], "audit": [dict(a) for a in audit]}
    except Exception as e:
        logger.error(f"ops_detail error: {e}")
        return {"error": str(e)}

@app.post("/api/ops/complete/{op_id}")
async def ops_complete(op_id: int):
    """Marquer une opération comme terminée."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor()
        cur.execute("UPDATE operations SET status='completed', completed_at=NOW() WHERE id=%s", (op_id,))
        db.commit(); cur.close(); db.close()
        ops_log("bhtc_fr", "stella", "operation_completed", op_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"ops_complete error: {e}")
        return {"error": str(e)}

@app.post("/api/ops/schedule/{op_id}")
async def ops_schedule(op_id: int, request: Request):
    """Planifier une action (rollback, notification, check)."""
    body = await request.json()
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor()
        cur.execute("""INSERT INTO scheduled_actions (operation_id, entity_code, action_type, scheduled_at, action_payload)
            VALUES (%s,%s,%s,%s,%s) RETURNING id""",
            (op_id, body.get("entity_code", "bhtc_fr"), body.get("action_type", "rollback"),
             body["scheduled_at"], json.dumps(body.get("action_payload", {}))))
        action_id = cur.fetchone()[0]
        db.commit(); cur.close(); db.close()
        ops_log(body.get("entity_code", "bhtc_fr"), "stella", "action_scheduled", op_id, "info",
                {"action_id": action_id, "type": body.get("action_type"), "scheduled_at": body["scheduled_at"]})
        return {"success": True, "action_id": action_id}
    except Exception as e:
        logger.error(f"ops_schedule error: {e}")
        return {"error": str(e)}

@app.get("/api/ops/scheduled")
async def ops_scheduled_list():
    """Lister les actions planifiées en attente."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM scheduled_actions WHERE status IN ('pending','failed') ORDER BY scheduled_at ASC")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"scheduled_actions": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"ops_scheduled error: {e}")
        return {"error": str(e)}

# ── ÉTAT DU SITE ──

@app.get("/api/ops/state")
async def ops_state(entity: str = "bhtc_fr"):
    """État live complet du site."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM site_state WHERE entity_code=%s ORDER BY category, key", (entity,))
        states = cur.fetchall()
        cur.execute("SELECT * FROM operations WHERE entity_code=%s AND status='active' ORDER BY created_at DESC", (entity,))
        active_ops = cur.fetchall()
        cur.execute("""SELECT * FROM scheduled_actions WHERE entity_code=%s AND status='pending'
            ORDER BY scheduled_at ASC""", (entity,))
        pending = cur.fetchall()
        cur.close(); db.close()
        # Grouper par catégorie
        grouped = {}
        for s in states:
            cat = s["category"]
            if cat not in grouped: grouped[cat] = []
            grouped[cat].append(dict(s))
        return {"entity": entity, "state": grouped, "active_operations": [dict(o) for o in active_ops],
                "pending_actions": [dict(p) for p in pending]}
    except Exception as e:
        logger.error(f"ops_state error: {e}")
        return {"error": str(e)}

@app.post("/api/ops/state/update")
async def ops_state_update(request: Request):
    """Mettre à jour un élément de l'état du site (upsert)."""
    body = await request.json()
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor()
        cur.execute("""INSERT INTO site_state (entity_code, category, key, value, expires_at)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (entity_code, category, key) DO UPDATE SET value=%s, updated_at=NOW(), expires_at=%s""",
            (body.get("entity_code", "bhtc_fr"), body["category"], body["key"],
             json.dumps(body["value"]), body.get("expires_at"),
             json.dumps(body["value"]), body.get("expires_at")))
        db.commit(); cur.close(); db.close()
        return {"success": True}
    except Exception as e:
        logger.error(f"ops_state_update error: {e}")
        return {"error": str(e)}

@app.delete("/api/ops/state/{state_id}")
async def ops_state_delete(state_id: int):
    """Retirer un élément de l'état du site."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor()
        cur.execute("DELETE FROM site_state WHERE id=%s", (state_id,))
        db.commit(); cur.close(); db.close()
        return {"success": True}
    except Exception as e:
        logger.error(f"ops_state_delete error: {e}")
        return {"error": str(e)}

# ── BRIEFING & AUDIT ──

@app.get("/api/ops/briefing")
async def ops_briefing(entity: str = "bhtc_fr"):
    """Briefing complet pour démarrage session Claude."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Opérations actives
        cur.execute("SELECT id, name, type, status, created_at, expires_at FROM operations WHERE entity_code=%s AND status='active' ORDER BY created_at DESC", (entity,))
        active_ops = cur.fetchall()
        # Actions planifiées dans les 48h
        cur.execute("""SELECT sa.*, o.name as operation_name FROM scheduled_actions sa
            LEFT JOIN operations o ON sa.operation_id = o.id
            WHERE sa.entity_code=%s AND sa.status='pending' AND sa.scheduled_at <= NOW() + INTERVAL '48 hours'
            ORDER BY sa.scheduled_at ASC""", (entity,))
        upcoming = cur.fetchall()
        # Alertes récentes (24h)
        cur.execute("""SELECT * FROM ops_audit_log WHERE entity_code=%s AND severity IN ('warning','error','critical')
            AND timestamp >= NOW() - INTERVAL '24 hours' ORDER BY timestamp DESC""", (entity,))
        alerts = cur.fetchall()
        # Dernières opérations terminées (7 jours)
        cur.execute("""SELECT id, name, type, status, completed_at FROM operations WHERE entity_code=%s
            AND status IN ('completed','rolled_back') AND completed_at >= NOW() - INTERVAL '7 days'
            ORDER BY completed_at DESC LIMIT 10""", (entity,))
        recent = cur.fetchall()
        # État du site
        cur.execute("SELECT category, key, value, expires_at FROM site_state WHERE entity_code=%s ORDER BY category", (entity,))
        site = cur.fetchall()
        cur.close(); db.close()
        return {
            "entity": entity,
            "active_operations": [dict(o) for o in active_ops],
            "upcoming_actions_48h": [dict(u) for u in upcoming],
            "alerts_24h": [dict(a) for a in alerts],
            "recent_completed": [dict(r) for r in recent],
            "site_state": [dict(s) for s in site]
        }
    except Exception as e:
        logger.error(f"ops_briefing error: {e}")
        return {"error": str(e)}

@app.get("/api/ops/audit")
async def ops_audit(entity: str = None, limit: int = 100):
    """Journal d'audit (filtrable)."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if entity:
            cur.execute("SELECT * FROM ops_audit_log WHERE entity_code=%s ORDER BY timestamp DESC LIMIT %s", (entity, limit))
        else:
            cur.execute("SELECT * FROM ops_audit_log ORDER BY timestamp DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"audit_log": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"ops_audit error: {e}")
        return {"error": str(e)}

@app.get("/api/ops/entities")
async def ops_entities():
    """Lister les entités."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM entity_registry WHERE active=TRUE ORDER BY code")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"entities": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"ops_entities error: {e}")
        return {"error": str(e)}

@app.get("/api/ops/playbook")
async def ops_playbook_list():
    """Lister tous les processus du playbook."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT process_name, process_type, description, trigger_conditions FROM process_playbook ORDER BY process_type, process_name")
        rows = cur.fetchall()
        cur.close(); db.close()
        return {"playbook": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/ops/playbook/{name}")
async def ops_playbook_detail(name: str):
    """Recuperer les etapes d'un processus + regles metier associees."""
    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM process_playbook WHERE process_name=%s", (name,))
        row = cur.fetchone()
        cur.close(); db.close()
        if not row: return {"error": f"Process '{name}' not found"}
        result = dict(row)
        # Ajouter les regles metier pertinentes selon le type de processus
        result["rules"] = PROCESS_RULES.get(name, [])
        result["reminder"] = "AVANT DE CODER : verifier chaque regle ci-dessous. Si une regle contredit ce que tu fais, ARRETE et corrige."
        return result
    except Exception as e:
        return {"error": str(e)}

# Regles metier associees a chaque processus — source de verite
PROCESS_RULES = {
    "creation_produit": [
        "Images : 3 images carre 2000x2000 WebP. Image 1 = flacon fond blanc zoom 80%. Image 2 = flacon+boite. Image 3 = lifestyle. Alt text SEO.",
        "Description : 1 paragraphe narratif obligatoire (accroche narrative 2-3 phrases, citation via metafield).",
        "SEO : title max 70c format [Nom] – [MARQUE] | [Conc] [Vol]ml | PlaneteBeauty. Meta max 155c.",
        "Vendor TOUJOURS en MAJUSCULES. productType = concentration exacte.",
        "Tags format standardise : Famille:X, Saison:X, Genre:X, Concentration:X, Occasion:X, Accord:X.",
        "Try Me : prix = 5% du format standard le plus cher arrondi euro sup. 1 par reference (EDP != Extrait).",
        "Metafields : 32 champs namespace parfum. Valeurs standardisees sillage (1-4), intensite (1-5).",
        "Sources : site officiel marque → Fragrantica Chrome MCP → Parfumo. JAMAIS inventer.",
    ],
    "creation_marque": [
        "Creer la collection marque.",
        "Enrichir chaque produit (pipeline 10 etapes — appeler playbook creation_produit).",
        "Creer article de blog pour la marque.",
        "Ajouter dans liste collections homepage + menu navigation.",
        "Mettre a jour compteur 'Nos XX Marques' homepage.",
        "Mettre a jour la page collection (description, image).",
    ],
    "operation_promo": [
        "AVANT de modifier quoi que ce soit : POST /api/ops/create avec state_before de CHAQUE fichier.",
        "Fichiers typiquement impactes : header-group.json, index.json, cart.json, pb-discount-code.liquid, metafield promo-codes.",
        "PLANIFIER le rollback : POST /api/ops/schedule avec date d'expiration.",
        "La carte Try Me est une SURPRISE — JAMAIS mentionnee dans une promo.",
        "Codes promo = product discount. Exclusions : Try Me, echantillons, coffrets, Creed/Roja/Clive Christian.",
        "Apres mise en place : verifier visuellement homepage, page produit, panier.",
    ],
    "modification_theme": [
        "AVANT : lire la carte des dependances (registre §5). Identifier TOUS les systemes impactes.",
        "APRES TOUTE modif : tester ATC sur 3 produits (formValid=true, bisEmailInForm=false).",
        "Tester panier (produits, quantites, cadeaux auto, codes promo, sous-total).",
        "Tester checkout (redirection, paiement).",
        "REVERT IMMEDIAT si echec.",
        "Le bloc BIS ne doit JAMAIS etre dans le form ATC (incident 03/04 = 23h ATC casse = ~2500€ perdu).",
        "Les cartes Try Me sont generiques Vistaprint — PAS de cartes digitales dans le dashboard.",
    ],
    "deploy_railway": [
        "Syntax check : python3 -c 'import py_compile; py_compile.compile(\"main.py\", doraise=True)'",
        "git push origin main (auto-deploy Railway).",
        "Attendre 45s puis verifier /api/ops/health.",
        "Mettre a jour site_state (deploy commit).",
        "Mettre a jour registre technique.",
    ],
    "deploy_shopify_function": [
        "Build : cargo build --target wasm32-wasip1.",
        "Deploy : shopify app deploy --force --no-release.",
        "Release : shopify app release --version=stella-v8-XX --allow-updates.",
        "Verifier Function ACTIVE dans Shopify Admin.",
        "Tester parcours client complet (ATC, panier, checkout, codes promo).",
        "Mettre a jour registre technique (version, IDs).",
    ],
}

@app.get("/api/ops/suggestions")
async def ops_suggestions():
    """Intelligence STELLA — suggestions automatiques basees sur les donnees."""
    suggestions = []
    db = get_db()
    if not db: return {"suggestions": []}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 1. Cashback expirant bientot sans relance
        cur.execute("""SELECT COUNT(*) as c FROM cashback_rewards
            WHERE status='active' AND expires_at BETWEEN NOW() AND NOW() + INTERVAL '3 days'
            AND (reminder_sent_at IS NULL)""")
        cb_expiring = cur.fetchone()["c"]
        if cb_expiring > 0:
            suggestions.append({
                "type": "action", "priority": "high", "system": "cashback",
                "message": f"{cb_expiring} cashback(s) expirent dans 3 jours sans relance envoyee",
                "action": "Envoyer les relances",
                "action_tab": "cashback"
            })

        # 2. Avis en attente de moderation
        cur.execute("SELECT COUNT(*) as c FROM product_reviews WHERE curated='pending'")
        pending_reviews = cur.fetchone()["c"]
        if pending_reviews > 0:
            suggestions.append({
                "type": "action", "priority": "medium", "system": "reviews",
                "message": f"{pending_reviews} avis en attente de moderation",
                "action": "Moderer les avis",
                "action_tab": "reviews"
            })

        # 3. Try Me non convertis qui expirent bientot
        cur.execute("""SELECT COUNT(*) as c FROM tryme_purchases
            WHERE status='pending' AND expires_at BETWEEN NOW() AND NOW() + INTERVAL '5 days'""")
        tryme_expiring = cur.fetchone()["c"]
        if tryme_expiring > 0:
            suggestions.append({
                "type": "insight", "priority": "medium", "system": "tryme",
                "message": f"{tryme_expiring} code(s) Try Me expirent dans 5 jours sans conversion",
                "action": "Voir les codes",
                "action_tab": "tryme"
            })

        # 4. Produits avec beaucoup de demandes BIS
        cur.execute("""SELECT product_handle, COUNT(*) as cnt FROM bis_subscriptions
            WHERE status='active' GROUP BY product_handle HAVING COUNT(*) >= 3 ORDER BY cnt DESC LIMIT 5""")
        hot_bis = cur.fetchall()
        for p in hot_bis:
            suggestions.append({
                "type": "insight", "priority": "high", "system": "bis",
                "message": f"{p['product_handle']} : {p['cnt']} clients attendent le retour en stock",
                "action": "Contacter le fournisseur",
                "action_tab": "bis"
            })

        # 5. Problemes catalogue non resolus
        cur.execute("SELECT result_data FROM cron_results WHERE cron_name='audit-qualite' ORDER BY executed_at DESC LIMIT 1")
        audit = cur.fetchone()
        if audit:
            audit_data = audit["result_data"]
            if isinstance(audit_data, str):
                try: audit_data = json.loads(audit_data)
                except: audit_data = {}
            issues = audit_data.get("issues", [])
            seo_issues = [i for i in issues if any("SEO" in p for p in (i.get("issues") or i.get("problems") or []))]
            if seo_issues:
                suggestions.append({
                    "type": "action", "priority": "high", "system": "catalogue",
                    "message": f"{len(seo_issues)} produit(s) avec SEO manquant — impact referencement Google",
                    "action": "Corriger le SEO",
                    "action_tab": "catalogue"
                })
            other_issues = [i for i in issues if not any("SEO" in p for p in (i.get("issues") or i.get("problems") or []))]
            if other_issues:
                suggestions.append({
                    "type": "insight", "priority": "low", "system": "catalogue",
                    "message": f"{len(other_issues)} produit(s) avec metafields incomplets (parfumeur, famille, accord)",
                    "action": "Enrichir les produits",
                    "action_tab": "catalogue"
                })

        # 6. Rollbacks en attente ou echoues
        cur.execute("SELECT COUNT(*) as c FROM scheduled_actions WHERE status='failed'")
        failed_actions = cur.fetchone()["c"]
        if failed_actions > 0:
            suggestions.append({
                "type": "alert", "priority": "critical", "system": "ops",
                "message": f"{failed_actions} action(s) planifiee(s) en echec — intervention requise",
                "action": "Voir les operations",
                "action_tab": "ops"
            })

        # 7. Cashback taux d'utilisation faible
        cur.execute("SELECT COUNT(*) FILTER (WHERE status='used') as used, COUNT(*) FILTER (WHERE status IN ('expired','revoked')) as lost FROM cashback_rewards")
        usage = cur.fetchone()
        total_closed = (usage["used"] or 0) + (usage["lost"] or 0)
        if total_closed > 10:
            rate = round((usage["used"] or 0) / total_closed * 100)
            if rate < 30:
                suggestions.append({
                    "type": "insight", "priority": "medium", "system": "cashback",
                    "message": f"Taux d'utilisation cashback : {rate}% — beaucoup de cashback expirent inutilises",
                    "action": "Ameliorer les relances",
                    "action_tab": "cashback"
                })

        cur.close(); db.close()
    except Exception as e:
        logger.error(f"suggestions error: {e}")
        try: db.close()
        except: pass

    # Quiz - faible taux de conversion
    rc = get_redis()
    if rc:
        try:
            today_key = datetime.now().strftime("%Y-%m-%d")
            views = int(rc.get(f"quiz:quiz_view:{today_key}") or 0)
            atc = int(rc.get(f"quiz:quiz_add_to_cart:{today_key}") or 0)
            if views > 10 and atc == 0:
                suggestions.append({
                    "type": "insight", "priority": "low", "system": "quiz",
                    "message": f"{views} vues quiz aujourd'hui mais 0 ajout au panier",
                    "action": "Verifier le scoring quiz",
                    "action_tab": "quiz"
                })
        except: pass

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 9))
    return {"suggestions": suggestions, "count": len(suggestions)}

@app.post("/api/ops/context")
async def ops_context(request: Request):
    """Analyse le message utilisateur et retourne le contexte pertinent (regles, playbook, etat)."""
    body = await request.json()
    message = (body.get("message", "") or "").lower()

    # Mapping mots-cles → sections regles metier + playbook
    KEYWORD_MAP = {
        "images": {"rules": "images", "section": "1. IMAGES PRODUITS"},
        "image": {"rules": "images", "section": "1. IMAGES PRODUITS"},
        "photo": {"rules": "images", "section": "1. IMAGES PRODUITS"},
        "webp": {"rules": "images", "section": "1. IMAGES PRODUITS"},
        "code promo": {"rules": "codes_promo", "playbook": "operation_promo", "section": "2. CODES PROMO"},
        "promo": {"rules": "codes_promo", "playbook": "operation_promo", "section": "2. CODES PROMO"},
        "pb580": {"rules": "codes_promo", "section": "2. CODES PROMO"},
        "pb10180": {"rules": "codes_promo", "section": "2. CODES PROMO"},
        "discount": {"rules": "codes_promo", "section": "2. CODES PROMO"},
        "try me": {"rules": "tryme", "section": "3. TRY ME"},
        "tryme": {"rules": "tryme", "section": "3. TRY ME"},
        "echantillon": {"rules": "tryme", "section": "3. TRY ME"},
        "cashback": {"rules": "cashback", "section": "4. CASHBACK"},
        "store credit": {"rules": "cashback", "section": "4. CASHBACK"},
        "credit": {"rules": "cashback", "section": "4. CASHBACK"},
        "livraison": {"rules": "livraison", "section": "5. LIVRAISON"},
        "shipping": {"rules": "livraison", "section": "5. LIVRAISON"},
        "frais de port": {"rules": "livraison", "section": "5. LIVRAISON"},
        "mondial relay": {"rules": "livraison", "section": "5. LIVRAISON"},
        "cadeau": {"rules": "cadeaux", "section": "6. CADEAUX AUTOMATIQUES"},
        "free gift": {"rules": "cadeaux", "section": "6. CADEAUX AUTOMATIQUES"},
        "mystere": {"rules": "cadeaux", "section": "6. CADEAUX AUTOMATIQUES"},
        "coffret": {"rules": "cadeaux", "section": "6. CADEAUX AUTOMATIQUES"},
        "tag": {"rules": "tags", "section": "7. TAGS SHOPIFY"},
        "collection": {"rules": "tags", "section": "7. TAGS SHOPIFY"},
        "filtre": {"rules": "tags", "section": "7. TAGS SHOPIFY"},
        "metafield": {"rules": "metafields", "section": "8. METAFIELDS"},
        "seo": {"rules": "seo", "section": "9. SEO"},
        "title": {"rules": "seo", "section": "9. SEO"},
        "meta description": {"rules": "seo", "section": "9. SEO"},
        "produit": {"rules": "pipeline_produit", "playbook": "creation_produit", "section": "10. PIPELINE CREATION PRODUIT"},
        "enrichir": {"rules": "pipeline_produit", "playbook": "creation_produit", "section": "10. PIPELINE CREATION PRODUIT"},
        "enrichissement": {"rules": "pipeline_produit", "playbook": "creation_produit", "section": "10. PIPELINE CREATION PRODUIT"},
        "creer un produit": {"rules": "pipeline_produit", "playbook": "creation_produit", "section": "10. PIPELINE CREATION PRODUIT"},
        "nouveau produit": {"rules": "pipeline_produit", "playbook": "creation_produit", "section": "10. PIPELINE CREATION PRODUIT"},
        "marque": {"rules": "pipeline_marque", "playbook": "creation_marque", "section": "11. PIPELINE NOUVELLE MARQUE"},
        "nouvelle marque": {"rules": "pipeline_marque", "playbook": "creation_marque", "section": "11. PIPELINE NOUVELLE MARQUE"},
        "theme": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. MODIFICATIONS THEME"},
        "liquid": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. MODIFICATIONS THEME"},
        "template": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. MODIFICATIONS THEME"},
        "section": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. MODIFICATIONS THEME"},
        "snippet": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. MODIFICATIONS THEME"},
        "deploy": {"playbook": "deploy_railway", "section": "DEPLOY"},
        "railway": {"playbook": "deploy_railway", "section": "DEPLOY"},
        "function": {"playbook": "deploy_shopify_function", "section": "SHOPIFY FUNCTION"},
        "rust": {"playbook": "deploy_shopify_function", "section": "SHOPIFY FUNCTION"},
        "wasm": {"playbook": "deploy_shopify_function", "section": "SHOPIFY FUNCTION"},
        "avis": {"section": "AVIS / REVIEWS"},
        "review": {"section": "AVIS / REVIEWS"},
        "bis": {"section": "BACK IN STOCK"},
        "back in stock": {"section": "BACK IN STOCK"},
        "rupture": {"section": "BACK IN STOCK"},
        "quiz": {"section": "QUIZ OLFACTIF"},
        "dashboard": {"section": "DASHBOARD"},
        "facture": {"section": "ORDER PRINTER"},
        "order printer": {"section": "ORDER PRINTER"},
        "email": {"section": "12. EMAILS AUTOMATIQUES"},
        "newsletter": {"section": "12. EMAILS AUTOMATIQUES"},
        "panier abandonn": {"section": "12. EMAILS AUTOMATIQUES"},
        "stock": {"section": "13. RECEPTION STOCK"},
        "reception": {"section": "13. RECEPTION STOCK"},
        "reappro": {"section": "13. RECEPTION STOCK"},
        "test": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. TESTS THEME"},
        "atc": {"rules": "modification_theme", "playbook": "modification_theme", "section": "14. TESTS THEME"},
        "checkout": {"rules": "modification_theme", "section": "14. TESTS THEME"},
        "operation": {"section": "STELLA OS OPERATIONS"},
        "ops": {"section": "STELLA OS OPERATIONS"},
        "rollback": {"rules": "operation_promo", "playbook": "operation_promo", "section": "STELLA OS OPERATIONS"},
        "stellaos": {"section": "STELLA OS"},
        "memoire": {"section": "MEMOIRE & SAUVEGARDE"},
        "sauvegarde": {"section": "MEMOIRE & SAUVEGARDE"},
        "checkpoint": {"section": "MEMOIRE & SAUVEGARDE"},
        "rag": {"section": "MEMOIRE & SAUVEGARDE"},
        "qdrant": {"section": "MEMOIRE & SAUVEGARDE"},
        "registre": {"section": "REGISTRE TECHNIQUE"},
        "commande": {"section": "COMMANDES"},
        "order": {"section": "COMMANDES"},
        "factur": {"section": "ORDER PRINTER / FACTURE"},
        "client": {"section": "CLIENTS"},
        "customer": {"section": "CLIENTS"},
    }

    # Identifier les sujets pertinents
    matched = {}
    for keyword, config in KEYWORD_MAP.items():
        if keyword in message:
            section = config.get("section", "")
            if section not in matched:
                matched[section] = config

    # Construire la reponse
    result = {
        "subjects_detected": list(matched.keys()),
        "rules": [],
        "playbooks": [],
    }

    # Recuperer les regles du playbook si applicable
    for section, config in matched.items():
        playbook_name = config.get("playbook")
        if playbook_name and playbook_name in PROCESS_RULES:
            result["playbooks"].append({
                "process": playbook_name,
                "rules": PROCESS_RULES[playbook_name]
            })

    # Ajouter le briefing ops
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT COUNT(*) as c FROM operations WHERE status='active'")
            result["active_ops"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM scheduled_actions WHERE status='pending'")
            result["pending_actions"] = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) as c FROM ops_audit_log WHERE severity IN ('warning','error','critical') AND timestamp >= NOW() - INTERVAL '24 hours'")
            result["alerts_24h"] = cur.fetchone()["c"]
            cur.close(); db.close()
        except Exception as e:
            try: db.close()
            except: pass

    # Recherche RAG — PostgreSQL direct (fiable, pas de dépendance Qdrant/embedding)
    if message and len(message) > 3:
        db2 = get_db()
        if db2:
            try:
                cur2 = db2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                words = [w.strip() for w in message.split() if len(w.strip()) > 2]
                if words:
                    conditions = " OR ".join(["(title ILIKE %s OR content ILIKE %s)"] * len(words))
                    params = []
                    for w in words[:8]:  # Max 8 mots-clés
                        params.extend([f"%{w}%", f"%{w}%"])
                    params.append(5)
                    cur2.execute(f"SELECT id, title, content, category, importance FROM claude_memories WHERE ({conditions}) ORDER BY importance DESC LIMIT %s", params)
                    memories = cur2.fetchall()
                    if memories:
                        result["rag_memories"] = [
                            {"title": m["title"], "content": (m["content"] or "")[:300], "category": m.get("category", "")}
                            for m in memories[:3]
                        ]
                cur2.close(); db2.close()
            except Exception as e:
                logger.debug(f"Context RAG search: {e}")
                try: db2.close()
                except: pass

    # Incohérences non résolues — dernier rapport coherence-check
    db3 = get_db()
    if db3:
        try:
            cur3 = db3.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur3.execute("SELECT result_data FROM cron_results WHERE cron_name='coherence-check' ORDER BY executed_at DESC LIMIT 1")
            row = cur3.fetchone()
            if row and row["result_data"]:
                report = row["result_data"] if isinstance(row["result_data"], dict) else json.loads(row["result_data"])
                incoherences = report.get("incoherences", [])
                errors = report.get("errors", [])
                if incoherences:
                    result["pending_incoherences"] = incoherences
                elif errors:
                    result["pending_incoherences"] = [{"domain": "legacy", "description": e} for e in errors]
            cur3.close(); db3.close()
        except Exception as e:
            logger.debug(f"Context incoherences: {e}")
            try: db3.close()
            except: pass

    return result

@app.get("/api/tryme/available")
async def tryme_available():
    """Retourne les variantes Try Me disponibles en stock 24H pour l'upsell panier."""
    try:
        gql = """{
          products(first: 250, query: "status:active") {
            nodes {
              id title handle
              variants(first: 10) {
                nodes {
                  id title price availableForSale
                  delai: metafield(namespace: "custom", key: "delai_d_expedition_variante") { value }
                }
              }
            }
          }
        }"""
        data = await shopify_graphql(gql)
        products = data.get("data", {}).get("products", {}).get("nodes", [])
        available = []
        for p in products:
            for v in p.get("variants", {}).get("nodes", []):
                vt = (v.get("title") or "").lower()
                # Filtrer : Try Me + en stock + expédition 24H
                delai = (v.get("delai") or {}).get("value", "")
                is_24h = "24H" in delai or "24h" in delai
                if "try me" in vt and v.get("availableForSale") and is_24h:
                    available.append({
                        "product_id": p["id"].split("/")[-1],
                        "product_title": p["title"],
                        "product_handle": p["handle"],
                        "variant_id": v["id"].split("/")[-1],
                        "variant_title": v["title"],
                        "price": v["price"]
                    })
        # Trier par nom de produit (regroupe les marques ensemble)
        available.sort(key=lambda x: x["product_title"])
        return {"available": available, "count": len(available)}
    except Exception as e:
        logger.error(f"tryme_available error: {e}")
        return {"available": [], "count": 0, "error": str(e)}

@app.get("/api/ops/health")
async def ops_health():
    """Santé globale du système."""
    db = get_db()
    rc = get_redis()
    health = {"database": bool(db), "redis": bool(rc), "crons": {}, "pending_actions": 0, "active_ops": 0}
    if db:
        try:
            cur = db.cursor()
            cur.execute("SELECT COUNT(*) FROM scheduled_actions WHERE status='pending'")
            health["pending_actions"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM operations WHERE status='active'")
            health["active_ops"] = cur.fetchone()[0]
            cur.execute("SELECT cron_name, MAX(executed_at) as last_run FROM cron_results GROUP BY cron_name")
            for row in cur.fetchall():
                health["crons"][row[0]] = str(row[1])
            cur.close(); db.close()
        except Exception as e:
            health["error"] = str(e)
            try: db.close()
            except: pass
    return health

# ── CRONS STELLA OS ──

@app.post("/api/cron/action-executor")
async def cron_action_executor():
    """Exécute les actions planifiées dont la date est passée. Toutes les 5 min."""
    db = get_db()
    if not db: return {"status": "no_db"}
    results = []
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""SELECT * FROM scheduled_actions
            WHERE status='pending' AND scheduled_at <= NOW()
            ORDER BY scheduled_at ASC LIMIT 10""")
        actions = cur.fetchall()
        if not actions:
            cur.close(); db.close()
            return {"status": "ok", "executed": 0}

        for action in actions:
            action = dict(action)
            action_id = action["id"]
            try:
                # Marquer en cours
                cur.execute("UPDATE scheduled_actions SET status='executing' WHERE id=%s", (action_id,))
                db.commit()

                payload = action["action_payload"] if isinstance(action["action_payload"], dict) else json.loads(action["action_payload"])
                action_type = action["action_type"]

                if action_type == "rollback":
                    # Exécuter le rollback via Shopify API
                    rollback_result = await _execute_rollback(action["operation_id"], payload, cur, db)
                    results.append({"action_id": action_id, "type": "rollback", "result": rollback_result})
                elif action_type == "notification":
                    # Envoyer email
                    await _send_ops_email(payload.get("to", "info@planetebeauty.com"),
                                          payload.get("subject", "STELLA OS Notification"),
                                          payload.get("body", ""))
                    results.append({"action_id": action_id, "type": "notification", "result": "sent"})

                # Marquer comme terminé
                cur.execute("""UPDATE scheduled_actions SET status='completed', executed_at=NOW(),
                    result=%s WHERE id=%s""", (json.dumps({"success": True}), action_id))
                db.commit()
                ops_log(action.get("entity_code", "bhtc_fr"), "cron:action-executor",
                        f"{action_type}_executed", action.get("operation_id"), "info",
                        {"action_id": action_id})

            except Exception as e:
                logger.error(f"Action {action_id} failed: {e}")
                retry = action.get("retry_count", 0) + 1
                max_r = action.get("max_retries", 3)
                if retry < max_r:
                    # Re-planifier dans 30 min
                    cur.execute("""UPDATE scheduled_actions SET status='pending', retry_count=%s,
                        scheduled_at=NOW() + INTERVAL '30 minutes', error_message=%s WHERE id=%s""",
                        (retry, str(e), action_id))
                else:
                    cur.execute("""UPDATE scheduled_actions SET status='failed', retry_count=%s,
                        error_message=%s WHERE id=%s""", (retry, str(e), action_id))
                    ops_log(action.get("entity_code", "bhtc_fr"), "cron:action-executor",
                            f"{action_type}_failed", action.get("operation_id"), "error",
                            {"action_id": action_id, "error": str(e), "retries": retry})
                    # Email alerte
                    await _send_ops_email("info@planetebeauty.com",
                        f"STELLA OS ALERTE — Action échouée #{action_id}",
                        f"L'action {action_type} pour l'opération #{action.get('operation_id')} a échoué après {retry} tentatives.\n\nErreur : {e}")
                db.commit()
                results.append({"action_id": action_id, "error": str(e)})

        cur.close(); db.close()
        return {"status": "ok", "executed": len(results), "results": results}
    except Exception as e:
        logger.error(f"action_executor error: {e}")
        try: db.close()
        except: pass
        return {"status": "error", "error": str(e)}

async def _execute_rollback(operation_id, payload, cur, db):
    """Exécuter un rollback : restaurer les fichiers modifiés via Shopify API."""
    if not operation_id:
        return {"error": "no operation_id"}

    cur.execute("SELECT * FROM operations_files WHERE operation_id=%s AND rolled_back=FALSE", (operation_id,))
    files = cur.fetchall()
    rolled = []

    for f in files:
        f = dict(f) if hasattr(f, 'keys') else {"id": f[0], "file_path": f[2], "file_type": f[3], "state_before": f[4]}
        file_id = f["id"]
        file_path = f["file_path"]
        file_type = f.get("file_type", "")
        state_before = f.get("state_before", "")

        try:
            if file_type in ("theme_json", "liquid", "snippet") and state_before:
                # Restaurer via Theme API
                theme_id = "197543559510"
                gql = """mutation { themeFilesUpsert(themeId: "gid://shopify/OnlineStoreTheme/%s", files: [{
                    filename: "%s", body: { type: TEXT, value: %s }
                }]) { upsertedThemeFiles { filename } userErrors { message } } }""" % (
                    theme_id, file_path, json.dumps(state_before))
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/2026-01/graphql.json",
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                        json={"query": gql})
                rolled.append({"file": file_path, "status": "restored"})

            elif file_type == "metafield" and state_before:
                # Restaurer metafield
                meta = json.loads(state_before) if isinstance(state_before, str) else state_before
                gql = """mutation { metafieldsSet(metafields: [{ ownerId: "%s", namespace: "%s", key: "%s",
                    type: "%s", value: %s }]) { metafields { id } userErrors { message } } }""" % (
                    meta["ownerId"], meta["namespace"], meta["key"], meta["type"], json.dumps(meta["value"]))
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/2026-01/graphql.json",
                        headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                        json={"query": gql})
                rolled.append({"file": file_path, "status": "restored"})

            # Marquer comme rollbacké
            cur.execute("UPDATE operations_files SET rolled_back=TRUE, rolled_back_at=NOW() WHERE id=%s", (file_id,))
            db.commit()

        except Exception as e:
            rolled.append({"file": file_path, "status": "error", "error": str(e)})
            logger.error(f"Rollback file {file_path}: {e}")

    # Mettre à jour l'opération
    cur.execute("UPDATE operations SET status='rolled_back', completed_at=NOW() WHERE id=%s", (operation_id,))
    db.commit()
    return {"files_rolled": len(rolled), "details": rolled}

async def _send_ops_email(to, subject, body):
    """Envoyer un email STELLA OS."""
    try:
        import aiosmtplib
        from email.mime.text import MIMEText
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASS", "")
        if not smtp_user:
            logger.warning("No SMTP config for ops email")
            return
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to
        await aiosmtplib.send(msg, hostname=smtp_host, port=smtp_port,
                              username=smtp_user, password=smtp_pass, start_tls=True)
        logger.info(f"Ops email sent: {subject}")
    except Exception as e:
        logger.error(f"Ops email error: {e}")

@app.post("/api/cron/drift-detector")
async def cron_drift_detector():
    """Détecte les incohérences entre site_state et la réalité Shopify. Quotidien 7h Paris."""
    db = get_db()
    if not db: return {"status": "no_db"}
    drifts = []
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # 1. Promos expirées encore marquées actives
        cur.execute("SELECT * FROM site_state WHERE category='promo_active' AND expires_at IS NOT NULL AND expires_at < NOW()")
        expired = cur.fetchall()
        for exp in expired:
            drifts.append({"type": "expired_promo", "key": exp["key"], "expired_at": str(exp["expires_at"])})
            # Auto-cleanup
            cur.execute("DELETE FROM site_state WHERE id=%s", (exp["id"],))
            ops_log(exp.get("entity_code", "bhtc_fr"), "cron:drift-detector", "expired_promo_cleaned",
                    severity="warning", details={"key": exp["key"]})

        # 2. Vérifier metafield promo-codes vs site_state
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                gql = '{ shop { metafield(namespace:"planete-beaute", key:"promo-codes") { value } } }'
                r = await client.post(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/2026-01/graphql.json",
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    json={"query": gql})
                data = r.json()
                mf_value = data.get("data", {}).get("shop", {}).get("metafield", {})
                if mf_value:
                    codes_live = json.loads(mf_value.get("value", "{}")).get("codes", {})
                    # Vérifier chaque code live a un discount Shopify actif
                    for code_name in codes_live:
                        gql2 = '{ discountNodes(first:1, query:"%s") { nodes { id discount { ... on DiscountCodeBasic { status endsAt } } } } }' % code_name
                        r2 = await client.post(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/2026-01/graphql.json",
                            headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                            json={"query": gql2})
                        d2 = r2.json()
                        nodes = d2.get("data", {}).get("discountNodes", {}).get("nodes", [])
                        if not nodes:
                            drifts.append({"type": "metafield_code_no_discount", "code": code_name,
                                           "detail": "Code dans metafield promo-codes mais pas de discount Shopify actif"})
                            ops_log("bhtc_fr", "cron:drift-detector", "code_without_discount",
                                    severity="error", details={"code": code_name})
        except Exception as e:
            logger.error(f"Drift check metafield: {e}")

        db.commit()
        cur.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s,%s)",
                    ("drift-detector", json.dumps({"drifts_found": len(drifts), "details": drifts})))
        db.commit(); cur.close(); db.close()

        # Email si des drifts trouvés
        if drifts:
            drift_text = "\n".join([f"- {d['type']}: {d.get('key', d.get('code', ''))}" for d in drifts])
            await _send_ops_email("info@planetebeauty.com",
                f"STELLA OS — {len(drifts)} incohérence(s) détectée(s)",
                f"Le drift detector a trouvé {len(drifts)} incohérence(s):\n\n{drift_text}\n\nActions correctives appliquées automatiquement quand possible.")

        return {"status": "ok", "drifts_found": len(drifts), "details": drifts}
    except Exception as e:
        logger.error(f"drift_detector error: {e}")
        try: db.close()
        except: pass
        return {"status": "error", "error": str(e)}

@app.post("/api/cron/daily-briefing")
async def cron_daily_briefing():
    """Envoie le rapport quotidien à Benoit. Quotidien 8h Paris."""
    db = get_db()
    if not db: return {"status": "no_db"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Opérations actives
        cur.execute("SELECT name, type, expires_at FROM operations WHERE status='active' ORDER BY expires_at ASC NULLS LAST")
        active_ops = cur.fetchall()

        # Rollbacks exécutés hier
        cur.execute("""SELECT sa.action_type, o.name, sa.executed_at, sa.status FROM scheduled_actions sa
            LEFT JOIN operations o ON sa.operation_id = o.id
            WHERE sa.executed_at >= NOW() - INTERVAL '24 hours' ORDER BY sa.executed_at DESC""")
        yesterday_actions = cur.fetchall()

        # Actions en attente (48h)
        cur.execute("""SELECT sa.action_type, o.name, sa.scheduled_at FROM scheduled_actions sa
            LEFT JOIN operations o ON sa.operation_id = o.id
            WHERE sa.status='pending' AND sa.scheduled_at <= NOW() + INTERVAL '48 hours'
            ORDER BY sa.scheduled_at ASC""")
        upcoming = cur.fetchall()

        # Alertes 24h
        cur.execute("""SELECT action, severity, details FROM ops_audit_log
            WHERE severity IN ('warning','error','critical') AND timestamp >= NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC""")
        alerts = cur.fetchall()

        # CA hier (depuis activity_log)
        cur.execute("""SELECT COUNT(*) as count FROM activity_log
            WHERE type='cashback_credited' AND timestamp >= NOW() - INTERVAL '24 hours'""")
        cashback_count = cur.fetchone()

        cur.close(); db.close()

        # Composer le rapport
        lines = ["STELLA OS — Rapport quotidien", "=" * 40, ""]
        if active_ops:
            lines.append(f"OPÉRATIONS ACTIVES ({len(active_ops)}):")
            for op in active_ops:
                exp = f" — expire {str(op['expires_at'])[:16]}" if op.get("expires_at") else ""
                lines.append(f"  - [{op['type']}] {op['name']}{exp}")
            lines.append("")
        else:
            lines.append("Aucune opération active.\n")

        if upcoming:
            lines.append(f"ACTIONS PRÉVUES (48h) ({len(upcoming)}):")
            for u in upcoming:
                lines.append(f"  - {u['action_type']} : {u.get('name', '?')} — {str(u['scheduled_at'])[:16]}")
            lines.append("")

        if yesterday_actions:
            lines.append(f"ACTIONS EXÉCUTÉES HIER ({len(yesterday_actions)}):")
            for ya in yesterday_actions:
                lines.append(f"  - {ya['action_type']} : {ya.get('name', '?')} — {ya['status']}")
            lines.append("")

        if alerts:
            lines.append(f"ALERTES ({len(alerts)}):")
            for a in alerts:
                lines.append(f"  - [{a['severity']}] {a['action']}")
            lines.append("")

        if cashback_count:
            lines.append(f"Cashback crédités hier : {cashback_count.get('count', 0)}")

        body = "\n".join(lines)
        await _send_ops_email("info@planetebeauty.com", "STELLA OS — Rapport quotidien", body)

        # Log cron
        db2 = get_db()
        if db2:
            cur2 = db2.cursor()
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s,%s)",
                        ("daily-briefing", json.dumps({"sent": True, "ops": len(active_ops), "alerts": len(alerts)})))
            db2.commit(); cur2.close(); db2.close()

        return {"status": "ok", "active_ops": len(active_ops), "alerts": len(alerts)}
    except Exception as e:
        logger.error(f"daily_briefing error: {e}")
        try: db.close()
        except: pass
        return {"status": "error", "error": str(e)}

@app.post("/api/cron/coherence-check")
async def cron_coherence_check():
    """Toutes les 6h — cross-check complet multi-sources. Shopify = source de vérité."""
    results = {"checks": [], "errors": [], "warnings": [], "incoherences": [], "elements_ok": []}
    shopify_headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}
    shopify_gql = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

    async with httpx.AsyncClient(timeout=20) as client:

        # ══ A. CODES PROMO — cross-check 3 sources ══
        try:
            # Source 1: Shopify metafield
            gql = '{ shop { metafield(namespace:"planete-beaute", key:"promo-codes") { value } } }'
            r = await client.post(shopify_gql, headers=shopify_headers, json={"query": gql})
            mf = r.json().get("data", {}).get("shop", {}).get("metafield", {})
            mf_codes = set(json.loads(mf.get("value", "{}")).get("codes", {}).keys()) if mf else set()

            # Source 2: site_state Railway
            db = get_db()
            state_codes = set()
            if db:
                cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT key FROM site_state WHERE entity_code='bhtc_fr' AND category='discount_active'")
                state_codes = {r["key"] for r in cur.fetchall()}
                cur.close(); db.close()

            # Source 3: Shopify Discounts (vérifier chaque code du metafield)
            shopify_active = set()
            for code_name in mf_codes:
                gql2 = '{ discountNodes(first:1, query:"%s") { nodes { id discount { ... on DiscountCodeBasic { status } } } } }' % code_name
                r2 = await client.post(shopify_gql, headers=shopify_headers, json={"query": gql2})
                nodes = r2.json().get("data", {}).get("discountNodes", {}).get("nodes", [])
                if nodes:
                    shopify_active.add(code_name)

            # Cross-check
            if mf_codes == state_codes == shopify_active:
                results["elements_ok"].append(f"{len(mf_codes)} codes promo coherents (metafield = site_state = Shopify): {', '.join(sorted(mf_codes))}")
            else:
                if mf_codes != shopify_active:
                    diff = mf_codes - shopify_active
                    if diff:
                        results["incoherences"].append({"domain": "codes_promo", "description": f"Codes dans metafield mais SANS discount Shopify: {diff}", "source_verite": f"Shopify actifs: {shopify_active}", "action": "Creer les discounts manquants ou retirer du metafield"})
                if mf_codes != state_codes:
                    results["incoherences"].append({"domain": "codes_promo", "description": f"Metafield ({mf_codes}) != site_state ({state_codes})", "source_verite": f"Metafield = source", "action": "Sync site_state"})
        except Exception as e:
            results["warnings"].append(f"Check promo: {e}")

        # ══ B. PRODUITS RECENTS (7 jours) — images notes + SEO + description ══
        try:
            gql = """{ products(first: 20, query: "updated_at:>%s", sortKey: UPDATED_AT, reverse: true) {
                nodes { id title status vendor
                    seo { title description }
                    descriptionHtml
                    img_tete: metafield(namespace:"parfum", key:"image_note_tete") { value }
                    img_coeur: metafield(namespace:"parfum", key:"image_note_coeur") { value }
                    img_fond: metafield(namespace:"parfum", key:"image_note_fond") { value }
                    note_tete: metafield(namespace:"parfum", key:"note_tete_principale") { value }
                    note_coeur: metafield(namespace:"parfum", key:"note_coeur_principale") { value }
                    note_fond: metafield(namespace:"parfum", key:"note_fond_principale") { value }
                }
            } }""" % (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            r = await client.post(shopify_gql, headers=shopify_headers, json={"query": gql})
            products = r.json().get("data", {}).get("products", {}).get("nodes", [])

            produits_ok = 0
            for p in products:
                issues = []
                title = p.get("title", "?")

                # Images notes
                has_notes = p.get("note_tete") or p.get("note_coeur") or p.get("note_fond")
                if has_notes:
                    if not p.get("img_tete", {}).get("value"):
                        issues.append("image_note_tete manquante")
                    if not p.get("img_coeur", {}).get("value"):
                        issues.append("image_note_coeur manquante")
                    if not p.get("img_fond", {}).get("value"):
                        issues.append("image_note_fond manquante")

                # SEO
                seo = p.get("seo", {}) or {}
                if not seo.get("title"):
                    issues.append("SEO title vide")
                if not seo.get("description"):
                    issues.append("SEO meta vide")

                # Description
                desc = p.get("descriptionHtml", "") or ""
                if desc.count("<p>") < 1 and desc.count("<p") < 1:
                    issues.append("Description vide (min 1 paragraphe)")

                # Vendor
                vendor = p.get("vendor", "")
                if vendor != vendor.upper():
                    issues.append(f"Vendor pas MAJUSCULES: {vendor}")

                if issues:
                    results["incoherences"].append({
                        "domain": "produits",
                        "description": f"{title}: {', '.join(issues)}",
                        "action": "Corriger avant mise en ACTIVE"
                    })
                else:
                    produits_ok += 1

            if produits_ok > 0:
                results["elements_ok"].append(f"{produits_ok}/{len(products)} produits recents complets")
        except Exception as e:
            results["warnings"].append(f"Check produits: {e}")

        # ══ C. OPERATIONS — ops obsoletes ══
        db = get_db()
        if db:
            try:
                cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Ops actives avec expires_at depasse
                cur.execute("SELECT id, name FROM operations WHERE status='active' AND expires_at IS NOT NULL AND expires_at < NOW()")
                expired_ops = cur.fetchall()
                for op in expired_ops:
                    results["incoherences"].append({"domain": "operations", "description": f"Op '{op['name']}' (id:{op['id']}) active mais expiree", "action": "Completer ou supprimer"})

                # Actions echouees
                cur.execute("SELECT COUNT(*) as c FROM scheduled_actions WHERE status='failed'")
                failed = cur.fetchone()["c"]
                if failed > 0:
                    results["incoherences"].append({"domain": "operations", "description": f"{failed} action(s) planifiee(s) en echec", "action": "Investiguer et relancer ou supprimer"})

                # Promos expirees dans site_state
                cur.execute("SELECT key FROM site_state WHERE category='promo_active' AND expires_at IS NOT NULL AND expires_at < NOW()")
                for exp in cur.fetchall():
                    cur.execute("DELETE FROM site_state WHERE category='promo_active' AND key=%s", (exp['key'],))
                    results["warnings"].append(f"Promo expiree {exp['key']} nettoyee de site_state")

                db.commit(); cur.close(); db.close()
            except Exception as e:
                results["warnings"].append(f"Check ops: {e}")
                try: db.close()
                except: pass

        # ══ D. HEARTBEAT MAC — vérifier que les checks locaux tournent ══
        db_hb = get_db()
        if db_hb:
            try:
                cur_hb = db_hb.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur_hb.execute("SELECT created_at FROM claude_memories WHERE title='heartbeat-mac' ORDER BY created_at DESC LIMIT 1")
                hb = cur_hb.fetchone()
                if hb and hb["created_at"]:
                    hours_ago = (datetime.now(timezone.utc) - hb["created_at"].replace(tzinfo=timezone.utc)).total_seconds() / 3600
                    if hours_ago > 4:
                        results["incoherences"].append({"domain": "monitoring", "description": f"Heartbeat Mac absent depuis {hours_ago:.0f}h (seuil 4h)", "action": "Mac eteint ou Claude Code ferme — checks locaux ne tournent pas"})
                    else:
                        results["elements_ok"].append(f"Heartbeat Mac OK ({hours_ago:.0f}h)")
                else:
                    results["warnings"].append("Aucun heartbeat Mac trouve — checks locaux jamais executes?")
                cur_hb.close(); db_hb.close()
            except Exception as e:
                results["warnings"].append(f"Heartbeat check: {e}")
                try: db_hb.close()
                except: pass

        # ══ E. MEMOIRE QDRANT — doublons + credentials retrouvables ══
        db = get_db()
        if db:
            try:
                cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                # Doublons
                cur.execute("SELECT title, COUNT(*) as c FROM claude_memories GROUP BY title HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 10")
                dupes = cur.fetchall()
                dupe_count = sum(d["c"] - 1 for d in dupes)
                if dupe_count > 0:
                    results["incoherences"].append({"domain": "memoire", "description": f"{dupe_count} doublons dans Qdrant ({len(dupes)} titres)", "action": "memory-consolidate va nettoyer"})
                else:
                    results["elements_ok"].append("Zero doublon memoire")

                # Credentials retrouvables
                cur.execute("SELECT COUNT(*) as c FROM claude_memories WHERE (title ILIKE '%openai%' OR content ILIKE '%sk-proj%') AND importance >= 8")
                if cur.fetchone()["c"] > 0:
                    results["elements_ok"].append("Cle OpenAI retrouvable dans RAG")
                else:
                    results["incoherences"].append({"domain": "memoire", "description": "Cle OpenAI introuvable dans RAG (importance >= 8)", "action": "Sauvegarder la cle avec importance 10"})

                # Stats memoire
                cur.execute("SELECT COUNT(*) as c FROM claude_memories")
                total = cur.fetchone()["c"]
                results["elements_ok"].append(f"{total} memoires RAG total")

                cur.close(); db.close()
            except Exception as e:
                results["warnings"].append(f"Check memoire: {e}")
                try: db.close()
                except: pass

        # ══ F. SMTP (conditionnel) ══
        smtp_user = os.getenv("SMTP_USER", "")
        if smtp_user:
            try:
                import aiosmtplib
                smtp = aiosmtplib.SMTP(hostname=os.getenv("SMTP_HOST", "smtp.gmail.com"), port=int(os.getenv("SMTP_PORT", "587")))
                await smtp.connect()
                await smtp.starttls()
                await smtp.login(smtp_user, os.getenv("SMTP_PASS", ""))
                await smtp.quit()
                results["elements_ok"].append("SMTP OK")
            except Exception as e:
                results["warnings"].append(f"SMTP: {e}")

        # ══ G. THEME — version coherente ══
        db = get_db()
        if db:
            try:
                cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT value FROM site_state WHERE entity_code='bhtc_fr' AND category='theme' AND key='main_theme'")
                row = cur.fetchone()
                if row:
                    theme_state = row["value"] if isinstance(row["value"], dict) else json.loads(row["value"])
                    results["elements_ok"].append(f"Theme {theme_state.get('name', '?')} ID {theme_state.get('id', '?')}")
                cur.close(); db.close()
            except:
                try: db.close()
                except: pass

    # ══ RESULTATS ══
    has_incoherences = len(results["incoherences"]) > 0
    severity = "error" if has_incoherences else ("warning" if results["warnings"] else "info")

    ops_log("bhtc_fr", "cron:coherence-check", "coherence_check_complete",
            severity=severity,
            details={"checks_ok": len(results["elements_ok"]), "incoherences": len(results["incoherences"]), "warnings": len(results["warnings"])})

    # Email automatique si nouvelles incohérences détectées
    if has_incoherences:
        # Comparer avec le dernier rapport pour ne pas spammer
        db_prev = get_db()
        prev_count = 0
        if db_prev:
            try:
                cur_prev = db_prev.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur_prev.execute("SELECT result_data FROM cron_results WHERE cron_name='coherence-check' ORDER BY executed_at DESC LIMIT 1")
                prev = cur_prev.fetchone()
                if prev and prev["result_data"]:
                    prev_data = prev["result_data"] if isinstance(prev["result_data"], dict) else json.loads(prev["result_data"])
                    prev_count = len(prev_data.get("incoherences", []))
                cur_prev.close(); db_prev.close()
            except:
                try: db_prev.close()
                except: pass

        current_count = len(results["incoherences"])
        # Alerter seulement si nouvelles incohérences (pas les mêmes qu'avant)
        if current_count > prev_count:
            new_issues = current_count - prev_count
            inco_text = "\n".join(f"• [{i.get('domain','')}] {i.get('description','')}" for i in results["incoherences"][:10])
            ok_text = "\n".join(f"✅ {e}" for e in results["elements_ok"][:5])
            await _send_ops_email("info@planetebeauty.com",
                f"STELLA — {new_issues} nouvelle(s) incohérence(s) détectée(s)",
                f"Contrôle automatique ({current_count} incohérences, {len(results['elements_ok'])} éléments OK)\n\nIncohérences:\n{inco_text}\n\nOK:\n{ok_text}")

    # Sauvegarder
    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor()
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s,%s)",
                        ("coherence-check", json.dumps(results)))
            db2.commit(); cur2.close(); db2.close()
        except: pass

    return results

@app.post("/api/cron/memory-audit")
async def cron_memory_audit():
    """Hebdo lundi — vérifie la cohérence des couches mémoire."""
    results = {"checks": [], "issues": []}

    db = get_db()
    if not db: return {"error": "No database"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # 1. Vérifier que des opérations récentes existent (signe de vie)
        cur.execute("SELECT COUNT(*) as c FROM operations WHERE created_at >= NOW() - INTERVAL '7 days'")
        ops_week = cur.fetchone()["c"]
        if ops_week == 0:
            results["issues"].append("Aucune opération tracée cette semaine — les actions sont-elles enregistrées ?")
        else:
            results["checks"].append(f"{ops_week} opérations tracées cette semaine")

        # 2. Vérifier que l'audit trail a des entrées
        cur.execute("SELECT COUNT(*) as c FROM ops_audit_log WHERE timestamp >= NOW() - INTERVAL '7 days'")
        audit_week = cur.fetchone()["c"]
        results["checks"].append(f"{audit_week} entrées audit trail cette semaine")

        # 3. Vérifier site_state n'est pas vide
        cur.execute("SELECT COUNT(*) as c FROM site_state")
        state_count = cur.fetchone()["c"]
        if state_count == 0:
            results["issues"].append("site_state est vide — l'état du site n'est pas suivi")
        else:
            results["checks"].append(f"{state_count} éléments dans site_state")

        # 4. Vérifier que le RAG a des entrées récentes
        cur.execute("SELECT COUNT(*) as c FROM claude_memories WHERE created_at >= NOW() - INTERVAL '7 days'")
        rag_week = cur.fetchone()["c"]
        if rag_week == 0:
            results["issues"].append("Aucune mémoire RAG sauvegardée cette semaine — la mémoire sémantique stagne")
        else:
            results["checks"].append(f"{rag_week} mémoires RAG ajoutées cette semaine")

        # 5. Vérifier playbook à jour
        cur.execute("SELECT COUNT(*) as c FROM process_playbook")
        playbook_count = cur.fetchone()["c"]
        results["checks"].append(f"{playbook_count} processus dans le playbook")

        cur.close(); db.close()
    except Exception as e:
        results["issues"].append(f"Erreur audit: {e}")
        try: db.close()
        except: pass

    # Email rapport
    checks_text = "\n".join(f"✅ {c}" for c in results["checks"])
    issues_text = "\n".join(f"⚠️ {i}" for i in results["issues"]) if results["issues"] else "Aucun problème détecté."
    await _send_ops_email("info@planetebeauty.com",
        "STELLA OS — Audit mémoire hebdomadaire",
        f"Rapport santé mémoire STELLA :\n\n{checks_text}\n\nProblèmes :\n{issues_text}")

    # Sauvegarder
    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor()
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s,%s)",
                        ("memory-audit", json.dumps(results)))
            db2.commit(); cur2.close(); db2.close()
        except: pass

    return results

@app.post("/api/cron/memory-consolidate")
async def cron_memory_consolidate():
    """Toutes les 2h — Nourrir, organiser, optimiser, vérifier, homogénéiser, contrôler la mémoire."""
    results = {"actions": [], "stats": {}}
    db = get_db()
    if not db: return {"error": "no_db"}
    try:
        cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # ── 1. NOURRIR — Capturer les événements récents non encore mémorisés ──
        # Vérifier les activités des 2 dernières heures pas encore dans ops_audit_log
        cur.execute("""SELECT type, action, details, timestamp FROM activity_log
            WHERE timestamp >= NOW() - INTERVAL '2 hours'
            AND type IN ('cashback_credit', 'tryme_created', 'order_paid', 'review_submitted', 'bis_notified')
            ORDER BY timestamp DESC LIMIT 20""")
        recent_events = cur.fetchall()
        results["stats"]["events_2h"] = len(recent_events)

        # Tracer les événements significatifs dans ops_audit_log s'ils n'y sont pas
        for ev in recent_events:
            cur.execute("""INSERT INTO ops_audit_log (entity_code, actor, action, severity, details)
                SELECT 'bhtc_fr', 'cron:consolidate', %s, 'info', %s
                WHERE NOT EXISTS (SELECT 1 FROM ops_audit_log WHERE action=%s AND timestamp >= NOW() - INTERVAL '2 hours')""",
                (f"event:{ev['type']}", json.dumps({"action": ev["action"]}), f"event:{ev['type']}"))

        # ── 2. ORGANISER — Classifier les mémoires RAG par catégorie ──
        # Compter les mémoires par catégorie
        cur.execute("SELECT category, COUNT(*) as c FROM claude_memories GROUP BY category ORDER BY c DESC")
        mem_categories = cur.fetchall()
        results["stats"]["rag_by_category"] = {r["category"]: r["c"] for r in mem_categories}
        results["stats"]["rag_total"] = sum(r["c"] for r in mem_categories)

        # ── 3. OPTIMISER — Supprimer les doublons (garder le plus récent) ──
        cur.execute("""SELECT title, COUNT(*) as c FROM claude_memories
            GROUP BY title HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 20""")
        duplicates = cur.fetchall()
        total_deleted = 0
        if duplicates:
            for d in duplicates:
                # Garder le plus récent (id le plus grand), supprimer les autres
                cur.execute("""DELETE FROM claude_memories WHERE title = %s AND id NOT IN (
                    SELECT MAX(id) FROM claude_memories WHERE title = %s)""",
                    (d["title"], d["title"]))
                deleted = cur.rowcount
                total_deleted += deleted
            results["actions"].append(f"{total_deleted} doublons supprimés ({len(duplicates)} titres)")
            results["stats"]["duplicates_cleaned"] = total_deleted

        # ── 4. VÉRIFIER — Cohérence entre les couches ──
        # site_state vs opérations actives
        cur.execute("SELECT COUNT(*) as c FROM site_state WHERE entity_code='bhtc_fr'")
        state_count = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) as c FROM operations WHERE status='active'")
        active_ops = cur.fetchone()["c"]
        results["stats"]["site_state_count"] = state_count
        results["stats"]["active_ops"] = active_ops

        if state_count == 0:
            results["actions"].append("site_state VIDE — l'état du site n'est pas suivi")

        # ── 5. HOMOGÉNÉISER — S'assurer que les données critiques sont partout ──
        # Vérifier que les codes promo dans site_state matchent le metafield Shopify
        cur.execute("SELECT key, value FROM site_state WHERE entity_code='bhtc_fr' AND category='discount_active'")
        state_codes = {r["key"] for r in cur.fetchall()}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                gql = '{ shop { metafield(namespace:"planete-beaute", key:"promo-codes") { value } } }'
                r = await client.post(f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json",
                    headers={"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"},
                    json={"query": gql})
                mf = r.json().get("data", {}).get("shop", {}).get("metafield", {})
                if mf:
                    mf_codes = set(json.loads(mf.get("value", "{}")).get("codes", {}).keys())
                    # Sync : metafield → site_state
                    for code in mf_codes - state_codes:
                        cur.execute("""INSERT INTO site_state (entity_code, category, key, value)
                            VALUES ('bhtc_fr', 'discount_active', %s, %s)
                            ON CONFLICT (entity_code, category, key) DO NOTHING""",
                            (code, json.dumps({"source": "auto-sync from metafield"})))
                        results["actions"].append(f"Code {code} ajouté dans site_state (sync metafield)")
                    for code in state_codes - mf_codes:
                        cur.execute("DELETE FROM site_state WHERE entity_code='bhtc_fr' AND category='discount_active' AND key=%s", (code,))
                        results["actions"].append(f"Code {code} retiré de site_state (absent du metafield)")
        except Exception as e:
            results["actions"].append(f"Erreur sync metafield: {e}")

        # ── 6. CONTRÔLER — Vérifier les seuils critiques ──
        # Cashback : montant total encours
        cur.execute("SELECT COALESCE(SUM(cashback_amount), 0) as total FROM cashback_rewards WHERE status='active'")
        cashback_total = float(cur.fetchone()["total"])
        results["stats"]["cashback_encours"] = cashback_total
        if cashback_total > 5000:
            results["actions"].append(f"ALERTE: cashback encours = {cashback_total:.0f}€ (seuil 5000€)")

        # Try Me : codes actifs
        cur.execute("SELECT COUNT(*) as c FROM tryme_purchases WHERE status='pending' AND expires_at > NOW()")
        tryme_active = cur.fetchone()["c"]
        results["stats"]["tryme_active"] = tryme_active

        # ── 7. PRÉPARER LE CONTEXTE — Mettre à jour site_state avec les stats fraîches ──
        cur.execute("""INSERT INTO site_state (entity_code, category, key, value, updated_at)
            VALUES ('bhtc_fr', 'stats', 'memory_health', %s, NOW())
            ON CONFLICT (entity_code, category, key) DO UPDATE SET value=%s, updated_at=NOW()""",
            (json.dumps(results["stats"]), json.dumps(results["stats"])))

        db.commit(); cur.close(); db.close()
    except Exception as e:
        results["actions"].append(f"Erreur consolidation: {e}")
        try: db.close()
        except: pass

    # Logger
    ops_log("bhtc_fr", "cron:memory-consolidate", "memory_consolidated",
            severity="info", details={"actions": len(results["actions"]), "stats": results["stats"]})

    # Sauvegarder résultat
    db2 = get_db()
    if db2:
        try:
            cur2 = db2.cursor()
            cur2.execute("INSERT INTO cron_results (cron_name, result_data) VALUES (%s,%s)",
                        ("memory-consolidate", json.dumps(results)))
            db2.commit(); cur2.close(); db2.close()
        except: pass

    return results

# ══════════════════════ CRON: GOOGLE ADS BUDGET GUARD ══════════════════════

@app.post("/api/cron/gads-budget-guard")
async def cron_gads_budget_guard():
    """Quotidien 8h — Verifie que depenses Ads < 5% du CA Shopify mensuel."""
    if not GADS_REFRESH_TOKEN and not GOOGLE_REFRESH_TOKEN:
        return {"skipped": True, "reason": "no_google_credentials"}
    try:
        data = await marketing_intelligence()
        b = data.get("budget", {})
        ratio = b.get("ratio_pct", 0)
        projected = b.get("projected_ratio_pct", 0)
        status = b.get("status", "ok")

        # Store cron result
        db = get_db()
        if db:
            try:
                cur = db.cursor()
                cur.execute("INSERT INTO cron_results (cron_name, result_data, executed_at) VALUES (%s, %s, NOW())",
                            ("gads-budget-guard", json.dumps(data)))
                db.commit(); cur.close(); db.close()
            except: pass

        # Alert if ratio >= 4%
        if ratio >= 4.0:
            severity = "danger" if ratio >= 5.0 else "warning"
            log_activity("gads_alert", f"Google Ads: ratio {ratio:.2f}% du CA (seuil 5%). Projection: {projected:.2f}%",
                         {"ratio": ratio, "projected": projected, "spend": b.get("ads_spend", 0),
                          "revenue": b.get("revenue", 0)},
                         source="cron", status=severity)

            # Email alert if >= 5%
            if ratio >= 5.0 and SMTP_HOST:
                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = f"ALERTE Google Ads : {ratio:.1f}% du CA"
                    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
                    msg["To"] = SMTP_FROM_EMAIL
                    body = f"""<div style="font-family:sans-serif;max-width:500px;margin:0 auto">
                      <h2 style="color:#dc2626">Budget Ads depasse le seuil</h2>
                      <p>Ratio actuel : <strong>{ratio:.2f}%</strong> (seuil : 5%)</p>
                      <p>Depenses mois : <strong>{data.get('ads_spend_month', 0):.2f} EUR</strong></p>
                      <p>CA mois : <strong>{data.get('revenue_month', 0):.2f} EUR</strong></p>
                      <p>Projection fin de mois : <strong>{projected:.2f}%</strong></p>
                    </div>"""
                    msg.attach(MIMEText(body, "html"))
                    await aiosmtplib.send(msg, hostname=SMTP_HOST, port=SMTP_PORT,
                                          username=SMTP_USER, password=SMTP_PASS, use_tls=True)
                    logger.info(f"GADS alert email sent: ratio={ratio:.2f}%")
                except Exception as e:
                    logger.warning(f"GADS alert email error: {e}")

        return {"ratio": ratio, "projected": projected, "status": status, "alerted": ratio >= 4.0}
    except Exception as e:
        logger.error(f"GADS budget guard error: {e}")
        return {"error": str(e)}

# ══════ SELF-AUDIT — État réel du système (remplace le guide statique) ══════

@app.get("/api/self-audit")
async def self_audit():
    """Retourne l'état RÉEL de chaque composant en lisant le code et les services live.
    Source de vérité = code + API, PAS un fichier guide statique."""
    audit = {"generated_at": datetime.now().isoformat(), "components": []}
    shopify_headers = {"X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN, "Content-Type": "application/json"}
    shopify_gql = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

    # ── 1. ENDPOINTS — Lister tous les @app.route enregistrés ──
    routes = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = list(route.methods - {"HEAD", "OPTIONS"}) if route.methods else []
            if methods and not route.path.startswith("/static"):
                routes.append({"path": route.path, "methods": methods})
    audit["endpoints_count"] = len(routes)
    audit["endpoints"] = sorted(routes, key=lambda r: r["path"])

    # ── 2. WEBHOOKS — Lister les webhooks Shopify enregistrés ──
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            gql = '{ webhookSubscriptions(first: 20) { edges { node { topic endpoint { ... on WebhookHttpEndpoint { callbackUrl } } } } } }'
            r = await client.post(shopify_gql, headers=shopify_headers, json={"query": gql})
            wh = r.json().get("data", {}).get("webhookSubscriptions", {}).get("edges", [])
            webhooks = [{"topic": w["node"]["topic"], "url": w["node"]["endpoint"]["callbackUrl"]} for w in wh]
            audit["webhooks"] = webhooks
    except Exception as e:
        audit["webhooks"] = {"error": str(e)}

    # ── 3. CRONS — État des crons APScheduler ──
    db = get_db()
    if db:
        try:
            cur = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT cron_name, MAX(executed_at) as last_run, COUNT(*) as runs FROM cron_results GROUP BY cron_name ORDER BY cron_name")
            crons = cur.fetchall()
            audit["crons"] = [{"name": c["cron_name"], "last_run": c["last_run"].isoformat() if c["last_run"] else None, "total_runs": c["runs"]} for c in crons]
            cur.close(); db.close()
        except Exception as e:
            audit["crons"] = {"error": str(e)}
            try: db.close()
            except: pass

    # ── 4. COMPOSANTS FONCTIONNELS — Test chaque service ──
    components = []

    # Cashback
    try:
        settings = get_cashback_settings()
        components.append({"name": "Cashback", "status": "ok", "config": {"rate": settings.get("cashback_rate"), "expiry": settings.get("expiry_days"), "min_use": settings.get("min_order_use")}})
    except:
        components.append({"name": "Cashback", "status": "error", "detail": "get_cashback_settings() failed"})

    # Codes promo
    try:
        db2 = get_db()
        if db2:
            cur2 = db2.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur2.execute("SELECT key, value FROM site_state WHERE category='discount_active'")
            codes = {r["key"]: r["value"] for r in cur2.fetchall()}
            cur2.close(); db2.close()
            components.append({"name": "Codes Promo", "status": "ok", "codes": list(codes.keys())})
    except:
        components.append({"name": "Codes Promo", "status": "error"})

    # BIS
    try:
        db3 = get_db()
        if db3:
            cur3 = db3.cursor()
            cur3.execute("SELECT COUNT(*) FROM bis_subscriptions WHERE notified_at IS NULL")
            bis_count = cur3.fetchone()[0]
            cur3.close(); db3.close()
            components.append({"name": "BIS", "status": "ok", "active_subscribers": bis_count})
    except:
        components.append({"name": "BIS", "status": "error", "detail": "table bis_subscriptions missing?"})

    # Quiz
    rc = get_redis()
    if rc:
        quiz_last = rc.get("quiz:regen:last")
        quiz_count = rc.get("quiz:regen:count")
        components.append({"name": "Quiz", "status": "ok", "last_regen": int(quiz_last) if quiz_last else None, "products": int(quiz_count) if quiz_count else None})
    else:
        components.append({"name": "Quiz", "status": "unknown", "detail": "Redis unavailable"})

    # Shipping
    try:
        db4 = get_db()
        if db4:
            cur4 = db4.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur4.execute("SELECT value FROM site_state WHERE key='shipping_settings' LIMIT 1")
            row = cur4.fetchone()
            cur4.close(); db4.close()
            if row:
                components.append({"name": "Livraison", "status": "ok", "config": row["value"]})
            else:
                components.append({"name": "Livraison", "status": "ok", "detail": "defaults"})
    except:
        components.append({"name": "Livraison", "status": "error"})

    # Google Ads
    has_gads = bool(os.getenv("GADS_CUSTOMER_ID")) and bool(os.getenv("GOOGLE_REFRESH_TOKEN") or os.getenv("GADS_REFRESH_TOKEN"))
    components.append({"name": "Google Ads", "status": "configured" if has_gads else "not_configured", "customer_id": os.getenv("GADS_CUSTOMER_ID", "")})

    # Memory
    try:
        db5 = get_db()
        if db5:
            cur5 = db5.cursor()
            cur5.execute("SELECT COUNT(*) FROM claude_memories")
            mem_count = cur5.fetchone()[0]
            cur5.close(); db5.close()
            components.append({"name": "Memoire RAG", "status": "ok", "count": mem_count})
    except:
        components.append({"name": "Memoire RAG", "status": "error"})

    audit["components"] = components
    return audit

# ══════ FIN STELLA OS ══════

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
