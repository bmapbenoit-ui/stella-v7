"""
STELLA V7 — Shopify Embedded App
FastAPI backend: Shopify session token auth + proxy to Context Engine
+ SHOPIFY API INTEGRATION for real-time data
"""
import os, time, hmac, hashlib, base64, json, logging, re
from typing import Optional, Dict, List
import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stella-shopify")

# ══════════════════════════════════════════════
# CONFIG (Railway environment variables)
# ══════════════════════════════════════════════
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")
SHOPIFY_STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN", SHOPIFY_SHOP)
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))

SHOPIFY_API_VERSION = "2026-01"
SHOPIFY_GRAPHQL_URL = f"https://{SHOPIFY_STORE_DOMAIN}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

app = FastAPI(title="STELLA Shopify App")


# ══════════════════════════════════════════════
# SHOPIFY GRAPHQL CLIENT
# ══════════════════════════════════════════════
async def shopify_graphql(query: str, variables: dict = None) -> dict:
    """Execute a Shopify GraphQL query with the admin access token."""
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
            data = r.json()
            if "errors" in data:
                logger.warning(f"Shopify GraphQL errors: {data['errors']}")
            return data
        except Exception as e:
            logger.error(f"Shopify GraphQL error: {e}")
            return {"error": str(e)}


# ══════════════════════════════════════════════
# SHOPIFY DATA FETCHERS
# ══════════════════════════════════════════════
async def fetch_shop_info() -> str:
    data = await shopify_graphql("{ shop { name myshopifyDomain plan { displayName } currencyCode } }")
    shop = data.get("data", {}).get("shop", {})
    return f"Boutique: {shop.get('name')} ({shop.get('myshopifyDomain')}), Plan: {shop.get('plan', {}).get('displayName')}, Devise: {shop.get('currencyCode')}"


async def fetch_products_summary() -> str:
    data = await shopify_graphql("""
    {
      all: products(first: 250) {
        edges { node { title status vendor productType totalInventory } }
      }
    }
    """)
    products = [e["node"] for e in data.get("data", {}).get("all", {}).get("edges", [])]
    active = sum(1 for p in products if p["status"] == "ACTIVE")
    draft = sum(1 for p in products if p["status"] == "DRAFT")
    archived = sum(1 for p in products if p["status"] == "ARCHIVED")
    total = len(products)

    brands = {}
    for p in products:
        brand = p.get("vendor", "Inconnu")
        brands[brand] = brands.get(brand, 0) + 1
    top_brands = sorted(brands.items(), key=lambda x: -x[1])[:10]

    low_stock = [p for p in products if p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]
    out_of_stock = [p for p in products if p.get("totalInventory") is not None and p["totalInventory"] <= 0]

    lines = [
        f"DONNEES SHOPIFY EN TEMPS REEL ({SHOPIFY_STORE_DOMAIN}):",
        f"Total produits: {total} (Actifs: {active}, Brouillons: {draft}, Archives: {archived})",
        f"Marques ({len(brands)}): {', '.join(f'{b} ({c})' for b,c in top_brands)}",
    ]
    if low_stock:
        lines.append(f"ALERTE STOCK BAS ({len(low_stock)}): {', '.join(p['title'][:40] for p in low_stock[:5])}")
    if out_of_stock:
        lines.append(f"RUPTURE STOCK ({len(out_of_stock)}): {', '.join(p['title'][:40] for p in out_of_stock[:5])}")
    return "\n".join(lines)


async def fetch_orders_summary() -> str:
    data = await shopify_graphql("""
    {
      orders(first: 50, sortKey: CREATED_AT, reverse: true) {
        edges {
          node {
            name createdAt displayFinancialStatus displayFulfillmentStatus
            totalPriceSet { shopMoney { amount currencyCode } }
            lineItems(first: 5) { edges { node { title quantity } } }
          }
        }
      }
    }
    """)
    orders = [e["node"] for e in data.get("data", {}).get("orders", {}).get("edges", [])]
    if not orders:
        return "COMMANDES SHOPIFY: Aucune commande recente."

    total_revenue = sum(float(o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", 0)) for o in orders)
    currency = orders[0].get("totalPriceSet", {}).get("shopMoney", {}).get("currencyCode", "EUR")
    paid = sum(1 for o in orders if o.get("displayFinancialStatus") == "PAID")
    pending = sum(1 for o in orders if o.get("displayFinancialStatus") in ("PENDING", "AUTHORIZED"))
    refunded = sum(1 for o in orders if o.get("displayFinancialStatus") == "REFUNDED")
    fulfilled = sum(1 for o in orders if o.get("displayFulfillmentStatus") == "FULFILLED")
    unfulfilled = sum(1 for o in orders if o.get("displayFulfillmentStatus") in ("UNFULFILLED", None))

    lines = [
        f"COMMANDES SHOPIFY (50 dernieres):",
        f"Total: {len(orders)} commandes, CA: {total_revenue:.2f} {currency}",
        f"Paiement: {paid} payees, {pending} en attente, {refunded} remboursees",
        f"Expedition: {fulfilled} expediees, {unfulfilled} a expedier",
        "5 dernieres commandes:"
    ]
    for o in orders[:5]:
        amount = o.get("totalPriceSet", {}).get("shopMoney", {}).get("amount", "?")
        items = [f"{i['node']['title'][:30]} x{i['node']['quantity']}" for i in o.get("lineItems", {}).get("edges", [])[:3]]
        lines.append(f"  {o['name']} | {amount} {currency} | {o.get('displayFinancialStatus','?')} | {', '.join(items)}")
    return "\n".join(lines)


async def fetch_inventory_alerts() -> str:
    data = await shopify_graphql("""
    {
      products(first: 250) {
        edges {
          node {
            title vendor status totalInventory
            variants(first: 5) { edges { node { title inventoryQuantity sku price } } }
          }
        }
      }
    }
    """)
    products = [e["node"] for e in data.get("data", {}).get("products", {}).get("edges", [])]
    critical = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and p["totalInventory"] <= 0]
    low = [p for p in products if p.get("status") == "ACTIVE" and p.get("totalInventory") is not None and 0 < p["totalInventory"] <= 5]

    lines = [f"ALERTES STOCK SHOPIFY:", f"Ruptures: {len(critical)} produits"]
    for p in critical[:10]:
        lines.append(f"  RUPTURE: {p['title'][:50]} ({p.get('vendor','?')})")
    lines.append(f"Stock bas (<=5): {len(low)} produits")
    for p in low[:10]:
        lines.append(f"  BAS ({p.get('totalInventory')}): {p['title'][:50]} ({p.get('vendor','?')})")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# SMART CONTEXT ENRICHMENT
# ══════════════════════════════════════════════
PRODUCT_KW = ["produit", "product", "catalogue", "stock", "inventaire", "marque", "brand", "vendor", "rupture", "combien de produit", "fiche", "actif"]
ORDER_KW = ["commande", "order", "vente", "revenue", "chiffre", "ca ", "c.a.", "panier", "expedition", "fulfil", "rembours"]
STOCK_KW = ["stock", "inventaire", "rupture", "out of stock", "alerte", "reapprovision", "bas"]
SHOP_KW = ["boutique", "shop", "magasin", "plan shopify", "planetebeauty"]

def detect_shopify_intent(message: str) -> List[str]:
    msg = message.lower()
    intents = []
    if any(k in msg for k in PRODUCT_KW):
        intents.append("products")
    if any(k in msg for k in ORDER_KW):
        intents.append("orders")
    if any(k in msg for k in STOCK_KW):
        intents.append("stock")
    if any(k in msg for k in SHOP_KW):
        intents.append("shop")
    # Generic "shopify" or "données" triggers full fetch
    if "shopify" in msg or ("donn" in msg and ("acc" in msg or "temps" in msg)):
        intents = list(set(intents + ["products", "orders", "shop"]))
    return intents


async def build_shopify_context(intents: List[str]) -> str:
    if not SHOPIFY_ACCESS_TOKEN:
        return "[SHOPIFY NON CONNECTE: SHOPIFY_ACCESS_TOKEN manquant]"
    if not intents:
        return ""
    parts = []
    try:
        if "products" in intents:
            parts.append(await fetch_products_summary())
        if "orders" in intents:
            parts.append(await fetch_orders_summary())
        if "stock" in intents:
            parts.append(await fetch_inventory_alerts())
        if "shop" in intents:
            parts.append(await fetch_shop_info())
    except Exception as e:
        parts.append(f"[Erreur Shopify: {e}]")
        logger.error(f"Shopify context error: {e}")
    return "\n\n".join(parts)


# ══════════════════════════════════════════════
# SHOPIFY SESSION TOKEN VERIFICATION
# ══════════════════════════════════════════════
def decode_session_token(token: str) -> dict:
    if not SHOPIFY_API_SECRET:
        raise HTTPException(500, "SHOPIFY_API_SECRET not configured")
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT")
        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = base64.urlsafe_b64encode(
            hmac.new(SHOPIFY_API_SECRET.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        if not hmac.compare_digest(expected, sig_b64.rstrip("=")):
            raise ValueError("Bad signature")
        pad = 4 - len(payload_b64) % 4
        if pad != 4:
            payload_b64 += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        now = time.time()
        if payload.get("exp", 0) < now - 10:
            raise ValueError("Expired")
        if payload.get("nbf", 0) > now + 10:
            raise ValueError("Not yet valid")
        if payload.get("aud") != SHOPIFY_API_KEY:
            raise ValueError(f"Wrong audience: {payload.get('aud')}")
        return payload
    except ValueError as e:
        raise HTTPException(401, f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(401, f"Token error: {e}")


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
# API ROUTES
# ══════════════════════════════════════════════
class ChatReq(BaseModel):
    message: str
    project: str = "GENERAL"

class ActionReq(BaseModel):
    action: str


@app.post("/api/chat")
async def chat(req: ChatReq, session: dict = Depends(verify_request)):
    """Chat with STELLA — enriched with live Shopify data."""
    # Detect if message needs Shopify data
    intents = detect_shopify_intent(req.message)
    shopify_context = ""
    if intents:
        logger.info(f"Shopify intents: {intents} for: {req.message[:80]}")
        shopify_context = await build_shopify_context(intents)

    # Build enriched task_context for Context Engine
    enriched_task_context = ""
    if shopify_context:
        enriched_task_context = f"""
--- DONNEES SHOPIFY TEMPS REEL ---
{shopify_context}
--- FIN DONNEES SHOPIFY ---
INSTRUCTION: Utilise ces donnees Shopify REELLES pour repondre. Ne dis JAMAIS que tu n'as pas acces aux donnees Shopify."""

    async with httpx.AsyncClient(timeout=120.0) as c:
        try:
            r = await c.post(f"{CONTEXT_ENGINE_URL}/chat", json={
                "message": req.message,
                "project": req.project,
                "user": session.get("sub", "unknown"),
                "task_context": enriched_task_context,
            })
            result = r.json()
            if intents:
                result["shopify_data"] = True
                result["shopify_intents"] = intents
            return result
        except Exception as e:
            raise HTTPException(502, f"Context Engine error: {e}")


@app.post("/api/action")
async def action(req: ActionReq, session: dict = Depends(verify_request)):
    """Quick actions."""
    if req.action == "shopify_products":
        return {"action": "shopify_products", "data": await fetch_products_summary()}
    if req.action == "shopify_orders":
        return {"action": "shopify_orders", "data": await fetch_orders_summary()}
    if req.action == "shopify_stock":
        return {"action": "shopify_stock", "data": await fetch_inventory_alerts()}

    endpoints = {
        "queue_stats": "/admin/queue-stats",
        "health": "/health",
        "next_product": "/admin/next-product",
    }
    ep = endpoints.get(req.action)
    if not ep:
        raise HTTPException(400, f"Unknown action: {req.action}")
    async with httpx.AsyncClient(timeout=30.0) as c:
        try:
            r = await c.get(f"{CONTEXT_ENGINE_URL}{ep}")
            return r.json()
        except Exception as e:
            raise HTTPException(502, f"Action error: {e}")


@app.get("/api/shopify/status")
async def shopify_status():
    """Check Shopify API connection status."""
    if not SHOPIFY_ACCESS_TOKEN:
        return {"connected": False, "error": "SHOPIFY_ACCESS_TOKEN missing"}
    try:
        data = await shopify_graphql("{ shop { name } }")
        if "errors" in data:
            return {"connected": False, "error": str(data["errors"])}
        return {"connected": True, "shop": data.get("data", {}).get("shop", {}).get("name")}
    except Exception as e:
        return {"connected": False, "error": str(e)}


@app.get("/api/memory/{project}")
async def memory(project: str, session: dict = Depends(verify_request)):
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.get(f"{CONTEXT_ENGINE_URL}/memory/{project}")
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
    return {
        "status": "ok",
        "app": "stella-shopify",
        "dev_mode": DEV_MODE,
        "shopify_api": "connected" if SHOPIFY_ACCESS_TOKEN else "no_token",
    }


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
