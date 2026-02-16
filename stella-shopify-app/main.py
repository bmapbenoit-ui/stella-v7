"""
STELLA V7 — Shopify Embedded App
FastAPI backend: Shopify session token auth + proxy to Context Engine
"""
import os, time, hmac, hashlib, base64, json, logging
from typing import Optional
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
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "https://context-engine-production-e525.up.railway.app")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(title="STELLA Shopify App")


# ══════════════════════════════════════════════
# SHOPIFY SESSION TOKEN VERIFICATION
# ══════════════════════════════════════════════
def decode_session_token(token: str) -> dict:
    """Decode and verify Shopify App Bridge session token (JWT)."""
    if not SHOPIFY_API_SECRET:
        raise HTTPException(500, "SHOPIFY_API_SECRET not configured")
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT")

        header_b64, payload_b64, sig_b64 = parts

        # Verify HMAC-SHA256 signature
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected = base64.urlsafe_b64encode(
            hmac.new(SHOPIFY_API_SECRET.encode(), signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode()

        if not hmac.compare_digest(expected, sig_b64.rstrip("=")):
            raise ValueError("Bad signature")

        # Decode payload
        pad = 4 - len(payload_b64) % 4
        if pad != 4:
            payload_b64 += "=" * pad
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))

        # Check expiry (10s clock skew)
        now = time.time()
        if payload.get("exp", 0) < now - 10:
            raise ValueError("Expired")
        if payload.get("nbf", 0) > now + 10:
            raise ValueError("Not yet valid")

        # Check audience = our API key
        if payload.get("aud") != SHOPIFY_API_KEY:
            raise ValueError(f"Wrong audience: {payload.get('aud')}")

        return payload
    except ValueError as e:
        raise HTTPException(401, f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(401, f"Token error: {e}")


async def verify_request(request: Request) -> dict:
    """Extract session token from Authorization header."""
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
    """Chat with STELLA via Context Engine."""
    async with httpx.AsyncClient(timeout=120.0) as c:
        try:
            r = await c.post(f"{CONTEXT_ENGINE_URL}/chat", json={
                "message": req.message,
                "project": req.project,
                "user": session.get("sub", "unknown"),
            })
            return r.json()
        except Exception as e:
            raise HTTPException(502, f"Context Engine error: {e}")


@app.post("/api/action")
async def action(req: ActionReq, session: dict = Depends(verify_request)):
    """Quick actions: queue_stats, health, next_product, enrich."""
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
    return {"status": "ok", "app": "stella-shopify", "dev_mode": DEV_MODE}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
