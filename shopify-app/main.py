"""
STELLA V7 — Shopify Embedded App
Backend FastAPI: auth Shopify + proxy vers Context Engine
"""
import os
import time
import hmac
import hashlib
import base64
import json
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stella-shopify")

# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "http://context-engine.railway.internal:8080")
ALLOWED_SHOP = SHOPIFY_SHOP  # Single-store app
PORT = int(os.getenv("PORT", "8080"))

# For dev/testing - bypass auth
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

app = FastAPI(title="STELLA Shopify App", version="1.0")


# ══════════════════════════════════════════════
# SHOPIFY SESSION TOKEN VALIDATION
# ══════════════════════════════════════════════
def decode_session_token(token: str) -> dict:
    """Decode and verify a Shopify session token (JWT)."""
    if not SHOPIFY_API_SECRET:
        raise HTTPException(status_code=500, detail="SHOPIFY_API_SECRET not configured")
    
    try:
        # Split JWT
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        
        header_b64, payload_b64, signature_b64 = parts
        
        # Verify signature using HMAC-SHA256
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        secret = SHOPIFY_API_SECRET.encode("utf-8")
        expected_sig = base64.urlsafe_b64encode(
            hmac.new(secret, signing_input, hashlib.sha256).digest()
        ).rstrip(b"=").decode("utf-8")
        
        # Compare signatures
        actual_sig = signature_b64.rstrip("=")
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")
        
        # Decode payload
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        
        # Verify claims
        now = time.time()
        
        # Check expiration (allow 10s clock skew)
        if payload.get("exp", 0) < now - 10:
            raise ValueError("Token expired")
        
        # Check not-before
        if payload.get("nbf", 0) > now + 10:
            raise ValueError("Token not yet valid")
        
        # Check audience matches our API key
        if payload.get("aud") != SHOPIFY_API_KEY:
            raise ValueError(f"Invalid audience: {payload.get('aud')}")
        
        # Check issuer/dest is our shop
        dest = payload.get("dest", "")
        if ALLOWED_SHOP not in dest:
            raise ValueError(f"Invalid shop: {dest}")
        
        return payload
        
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token decode error: {e}")


async def verify_shopify_request(request: Request) -> Optional[dict]:
    """Extract and verify Shopify session token from request."""
    if DEV_MODE:
        return {"sub": "dev_user", "dest": SHOPIFY_SHOP, "iss": SHOPIFY_SHOP}
    
    # App Bridge automatically adds session token in Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        return decode_session_token(token)
    
    # Also check query param (for initial load)
    token = request.query_params.get("id_token")
    if token:
        return decode_session_token(token)
    
    raise HTTPException(status_code=401, detail="No session token provided")


# ══════════════════════════════════════════════
# API MODELS
# ══════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str
    project: str = "GENERAL"


class QuickAction(BaseModel):
    action: str  # "queue_stats", "brands", "health", "next_product"


# ══════════════════════════════════════════════
# ROUTES — API (authenticated)
# ══════════════════════════════════════════════
@app.post("/api/chat")
async def chat(req: ChatRequest, session: dict = Depends(verify_shopify_request)):
    """Proxy chat to Context Engine with Shopify auth."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{CONTEXT_ENGINE_URL}/chat", json={
                "message": req.message,
                "project": req.project,
                "user": session.get("sub", "unknown"),
            })
            return resp.json()
        except httpx.ConnectError:
            # Fallback to public URL if internal doesn't work
            public_url = os.getenv("CONTEXT_ENGINE_PUBLIC_URL", 
                                   "https://context-engine-production-e525.up.railway.app")
            resp = await client.post(f"{public_url}/chat", json={
                "message": req.message,
                "project": req.project,
            })
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Context Engine error: {e}")


@app.post("/api/action")
async def quick_action(req: QuickAction, session: dict = Depends(verify_shopify_request)):
    """Execute quick actions (queue stats, etc.)."""
    action_map = {
        "queue_stats": "/admin/queue-stats",
        "brands": "/brands",
        "health": "/health",
        "next_product": "/admin/next-product",
    }
    
    endpoint = action_map.get(req.action)
    if not endpoint:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}{endpoint}")
            return resp.json()
        except httpx.ConnectError:
            public_url = os.getenv("CONTEXT_ENGINE_PUBLIC_URL",
                                   "https://context-engine-production-e525.up.railway.app")
            resp = await client.get(f"{public_url}{endpoint}")
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Action error: {e}")


@app.get("/api/memory/{project}")
async def get_memory(project: str, session: dict = Depends(verify_shopify_request)):
    """Get memory state for a project."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}/memory/{project}")
            return resp.json()
        except:
            public_url = os.getenv("CONTEXT_ENGINE_PUBLIC_URL",
                                   "https://context-engine-production-e525.up.railway.app")
            resp = await client.get(f"{public_url}/memory/{project}")
            return resp.json()


# ══════════════════════════════════════════════
# ROUTES — App pages (unauthenticated for embedding)
# ══════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def app_home(request: Request):
    """Serve the embedded app frontend."""
    # Read and inject config into the HTML
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r") as f:
        html = f.read()
    
    # Inject Shopify config
    html = html.replace("__SHOPIFY_API_KEY__", SHOPIFY_API_KEY)
    html = html.replace("__SHOPIFY_SHOP__", SHOPIFY_SHOP)
    
    # Get host param from Shopify for App Bridge
    host = request.query_params.get("host", "")
    html = html.replace("__SHOPIFY_HOST__", host)
    
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "app": "stella-shopify",
        "shop": SHOPIFY_SHOP,
        "context_engine": CONTEXT_ENGINE_URL,
        "dev_mode": DEV_MODE,
    }


# Static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
