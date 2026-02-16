"""
STELLA V7 — Shopify Embedded App
Chat interface dans le Shopify Admin de PlaneteBeauty
"""
import os, json, time, hmac, hashlib, base64, urllib.parse
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# === CONFIG ===
SHOPIFY_API_KEY = os.getenv("SHOPIFY_API_KEY", "")
SHOPIFY_API_SECRET = os.getenv("SHOPIFY_API_SECRET", "")
SHOPIFY_SCOPES = "read_products,write_products,read_orders"
SHOPIFY_SHOP = os.getenv("SHOPIFY_SHOP", "planetemode.myshopify.com")
HOST = os.getenv("HOST", "https://stella-shopify-app-production.up.railway.app")

# Context Engine URL (internal Railway network)
CONTEXT_ENGINE_URL = os.getenv("CONTEXT_ENGINE_URL", "http://context-engine.railway.internal:8080")
# Fallback to public URL if internal doesn't work
CONTEXT_ENGINE_PUBLIC = os.getenv("CONTEXT_ENGINE_PUBLIC", "https://context-engine-production-e525.up.railway.app")

# Store access token in memory (single-tenant app)
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN", "")

app = FastAPI(title="STELLA V7 — Shopify App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SHOPIFY AUTH HELPERS ===

def verify_hmac(query_params: dict) -> bool:
    """Verify Shopify HMAC signature"""
    if not SHOPIFY_API_SECRET:
        return True  # Skip in dev mode
    
    hmac_value = query_params.pop("hmac", "")
    sorted_params = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
    computed = hmac.new(
        SHOPIFY_API_SECRET.encode(),
        sorted_params.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(computed, hmac_value)


def verify_session_token(token: str) -> dict:
    """Verify Shopify Session Token (JWT)"""
    if not SHOPIFY_API_SECRET:
        return {"shop": SHOPIFY_SHOP}  # Dev mode
    
    try:
        # Shopify session tokens are JWTs signed with API secret
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT")
        
        # Decode header and payload
        payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        
        # Verify signature
        signing_input = f"{parts[0]}.{parts[1]}".encode()
        signature = base64.urlsafe_b64decode(parts[2] + "=" * (4 - len(parts[2]) % 4))
        expected = hmac.new(SHOPIFY_API_SECRET.encode(), signing_input, hashlib.sha256).digest()
        
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        
        # Check expiry
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")
        
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid session token: {e}")


async def get_current_shop(request: Request) -> str:
    """Extract and verify shop from request"""
    # Check Authorization header (session token from App Bridge)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        payload = verify_session_token(token)
        return payload.get("dest", "").replace("https://", "").replace("http://", "")
    
    # Check query params (for embedded app frame)
    shop = request.query_params.get("shop", SHOPIFY_SHOP)
    return shop


# === MODELS ===

class ChatMessage(BaseModel):
    message: str
    project: Optional[str] = "GENERAL"


# === ROUTES ===

@app.get("/")
async def root(request: Request):
    """Serve the embedded app or redirect to install"""
    shop = request.query_params.get("shop", "")
    
    # If accessed from Shopify Admin, serve the app
    return HTMLResponse(content=get_app_html())


@app.get("/auth")
async def auth_start(request: Request):
    """Start Shopify OAuth flow"""
    shop = request.query_params.get("shop", SHOPIFY_SHOP)
    
    import secrets
    nonce = secrets.token_hex(16)
    
    redirect_uri = f"{HOST}/auth/callback"
    auth_url = (
        f"https://{shop}/admin/oauth/authorize?"
        f"client_id={SHOPIFY_API_KEY}&"
        f"scope={SHOPIFY_SCOPES}&"
        f"redirect_uri={redirect_uri}&"
        f"state={nonce}"
    )
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle Shopify OAuth callback"""
    global ACCESS_TOKEN
    
    params = dict(request.query_params)
    code = params.get("code", "")
    shop = params.get("shop", "")
    
    if not code or not shop:
        raise HTTPException(400, "Missing code or shop")
    
    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://{shop}/admin/oauth/access_token",
            json={
                "client_id": SHOPIFY_API_KEY,
                "client_secret": SHOPIFY_API_SECRET,
                "code": code,
            }
        )
        data = resp.json()
        ACCESS_TOKEN = data.get("access_token", "")
    
    # Redirect to app
    return RedirectResponse(f"https://{shop}/admin/apps/{SHOPIFY_API_KEY}")


@app.post("/api/chat")
async def chat(msg: ChatMessage, request: Request):
    """Chat with STELLA — proxies to Context Engine"""
    # Verify the request comes from our Shopify app
    shop = await get_current_shop(request)
    
    # Forward to Context Engine
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Try internal Railway URL first
            resp = await client.post(
                f"{CONTEXT_ENGINE_URL}/chat",
                json={
                    "message": msg.message,
                    "project": msg.project,
                    "task_context": f"Chat Shopify — {shop}"
                }
            )
            return resp.json()
        except Exception:
            # Fallback to public URL
            resp = await client.post(
                f"{CONTEXT_ENGINE_PUBLIC}/chat",
                json={
                    "message": msg.message,
                    "project": msg.project,
                    "task_context": f"Chat Shopify — {shop}"
                }
            )
            return resp.json()


@app.get("/api/memory/{project}")
async def get_memory(project: str, request: Request):
    """Get STELLA memory for a project"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}/memory/{project}")
            return resp.json()
        except Exception:
            resp = await client.get(f"{CONTEXT_ENGINE_PUBLIC}/memory/{project}")
            return resp.json()


@app.get("/api/queue-stats")
async def queue_stats(request: Request):
    """Get product queue stats"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}/admin/queue-stats")
            return resp.json()
        except Exception:
            resp = await client.get(f"{CONTEXT_ENGINE_PUBLIC}/admin/queue-stats")
            return resp.json()


@app.get("/api/brands")
async def brands(request: Request):
    """Get configured brands"""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}/brands")
            return resp.json()
        except Exception:
            resp = await client.get(f"{CONTEXT_ENGINE_PUBLIC}/brands")
            return resp.json()


@app.get("/api/health")
async def health():
    """Health check"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{CONTEXT_ENGINE_URL}/health")
            ce_health = resp.json()
        except Exception:
            try:
                resp = await client.get(f"{CONTEXT_ENGINE_PUBLIC}/health")
                ce_health = resp.json()
            except Exception:
                ce_health = {"status": "unreachable"}
    
    return {
        "status": "ok",
        "context_engine": ce_health,
        "shop": SHOPIFY_SHOP,
        "api_key_set": bool(SHOPIFY_API_KEY),
    }


def get_app_html():
    """Return the embedded app HTML"""
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>STELLA V7 — PlaneteBeauty</title>
    
    <!-- Shopify App Bridge -->
    <script src="https://cdn.shopify.com/shopifycloud/app-bridge.js"></script>
    
    <!-- Polaris -->
    <link rel="stylesheet" href="https://unpkg.com/@shopify/polaris@12.0.0/build/esm/styles.css">
    
    <!-- React -->
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif; background: #F6F6F7; }}
        
        .stella-container {{
            max-width: 960px;
            margin: 0 auto;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .stella-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 16px;
        }}
        
        .stella-logo {{
            width: 40px; height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, #C8984E, #8B6914);
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 20px; font-weight: 700;
        }}
        
        .stella-title {{ font-size: 18px; font-weight: 700; color: #1A1A1A; }}
        .stella-subtitle {{ font-size: 12px; color: #6B7280; }}
        
        .stella-status {{
            display: flex; align-items: center; gap: 6px;
            margin-left: auto;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px; font-weight: 600;
        }}
        .stella-status.online {{ background: #E8F5E9; color: #2E7D32; }}
        .stella-status.offline {{ background: #FEE; color: #C62828; }}
        .status-dot {{
            width: 6px; height: 6px; border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}
        
        .chat-area {{
            flex: 1;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .messages {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        
        .message {{
            display: flex;
            gap: 10px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }}
        @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}
        
        .message.user {{ align-self: flex-end; flex-direction: row-reverse; }}
        .message.stella {{ align-self: flex-start; }}
        
        .message-avatar {{
            width: 32px; height: 32px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 14px; flex-shrink: 0;
        }}
        .message.stella .message-avatar {{
            background: linear-gradient(135deg, #C8984E, #8B6914);
            color: white;
        }}
        .message.user .message-avatar {{
            background: #E3E8EF;
            color: #374151;
        }}
        
        .message-bubble {{
            padding: 12px 16px;
            border-radius: 12px;
            font-size: 14px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .message.stella .message-bubble {{
            background: #F3F4F6;
            color: #1A1A1A;
            border-bottom-left-radius: 4px;
        }}
        .message.user .message-bubble {{
            background: #C8984E;
            color: white;
            border-bottom-right-radius: 4px;
        }}
        
        .message-meta {{
            font-size: 10px;
            color: #9CA3AF;
            margin-top: 4px;
        }}
        .message.user .message-meta {{ text-align: right; }}
        
        .input-area {{
            padding: 16px 20px;
            border-top: 1px solid #E5E7EB;
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }}
        
        .input-area textarea {{
            flex: 1;
            border: 1px solid #D1D5DB;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 14px;
            font-family: inherit;
            resize: none;
            min-height: 42px;
            max-height: 120px;
            outline: none;
            transition: border-color 0.2s;
        }}
        .input-area textarea:focus {{ border-color: #C8984E; }}
        
        .send-btn {{
            width: 42px; height: 42px;
            border-radius: 10px;
            border: none;
            background: #C8984E;
            color: white;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
            flex-shrink: 0;
        }}
        .send-btn:hover {{ background: #B8883E; }}
        .send-btn:disabled {{ background: #D1D5DB; cursor: not-allowed; }}
        
        .project-selector {{
            display: flex;
            gap: 6px;
            padding: 12px 20px;
            border-bottom: 1px solid #F3F4F6;
            overflow-x: auto;
        }}
        .project-btn {{
            padding: 5px 12px;
            border-radius: 20px;
            border: 1px solid #E5E7EB;
            background: white;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
        }}
        .project-btn.active {{
            background: #C8984E;
            color: white;
            border-color: #C8984E;
        }}
        .project-btn:hover:not(.active) {{ border-color: #C8984E; color: #C8984E; }}
        
        .typing-indicator {{
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }}
        .typing-dot {{
            width: 6px; height: 6px;
            border-radius: 50%;
            background: #9CA3AF;
            animation: typing 1.4s infinite;
        }}
        .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
        @keyframes typing {{ 0%,60%,100%{{opacity:0.3}} 30%{{opacity:1}} }}
        
        .quick-actions {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            padding: 0 20px 16px;
        }}
        .quick-action {{
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            background: #FAFAFA;
            font-size: 12px;
            color: #6B7280;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .quick-action:hover {{
            border-color: #C8984E;
            color: #C8984E;
            background: #FFF8F0;
        }}
        
        .stats-bar {{
            display: flex;
            gap: 16px;
            padding: 8px 20px;
            background: #FAFAFA;
            border-bottom: 1px solid #F3F4F6;
            font-size: 11px;
            color: #6B7280;
        }}
        .stat {{ display: flex; align-items: center; gap: 4px; }}
        .stat-value {{ font-weight: 700; color: #1A1A1A; }}
    </style>
</head>
<body>
    <div id="app"></div>
    
    <script type="text/babel">
        const {{ useState, useEffect, useRef, useCallback }} = React;
        
        const PROJECTS = [
            {{ id: "GENERAL", label: "Général", icon: "💬" }},
            {{ id: "FICHES_V6", label: "Fiches V6", icon: "📋" }},
            {{ id: "GOOGLE_ADS", label: "Google Ads", icon: "📊" }},
            {{ id: "FINANCES", label: "Finances", icon: "💰" }},
            {{ id: "FOURNISSEURS", label: "Fournisseurs", icon: "📦" }},
        ];
        
        const QUICK_ACTIONS = [
            "Combien de produits enrichis ?",
            "Quel est le prochain produit à traiter ?",
            "Résumé de la journée",
            "Stats par marque",
            "Que sais-tu sur les BADAR ?",
        ];
        
        function App() {{
            const [messages, setMessages] = useState([
                {{
                    role: "stella",
                    text: "Bonjour Benoit ! Je suis STELLA V7, le cerveau permanent de PlaneteBeauty.\\n\\nJe me souviens de tout : 337 produits dans le catalogue, 49 marques, le template V6, tes décisions stratégiques...\\n\\nComment puis-je t'aider ?",
                    time: new Date().toLocaleTimeString("fr-FR", {{ hour: "2-digit", minute: "2-digit" }}),
                    project: "GENERAL"
                }}
            ]);
            const [input, setInput] = useState("");
            const [loading, setLoading] = useState(false);
            const [project, setProject] = useState("GENERAL");
            const [stats, setStats] = useState(null);
            const [ceHealth, setCeHealth] = useState(null);
            const messagesEndRef = useRef(null);
            const textareaRef = useRef(null);
            
            // Auto-scroll
            useEffect(() => {{
                messagesEndRef.current?.scrollIntoView({{ behavior: "smooth" }});
            }}, [messages]);
            
            // Load stats on mount
            useEffect(() => {{
                fetch("/api/queue-stats").then(r => r.json()).then(setStats).catch(() => {{}});
                fetch("/api/health").then(r => r.json()).then(setCeHealth).catch(() => {{}});
            }}, []);
            
            const sendMessage = useCallback(async (text) => {{
                if (!text.trim() || loading) return;
                
                const userMsg = {{
                    role: "user",
                    text: text.trim(),
                    time: new Date().toLocaleTimeString("fr-FR", {{ hour: "2-digit", minute: "2-digit" }}),
                    project
                }};
                
                setMessages(prev => [...prev, userMsg]);
                setInput("");
                setLoading(true);
                
                // Auto-resize textarea
                if (textareaRef.current) {{
                    textareaRef.current.style.height = "42px";
                }}
                
                try {{
                    const resp = await fetch("/api/chat", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify({{ message: text.trim(), project }})
                    }});
                    const data = await resp.json();
                    
                    setMessages(prev => [...prev, {{
                        role: "stella",
                        text: data.answer || data.error || "Pas de réponse",
                        time: new Date().toLocaleTimeString("fr-FR", {{ hour: "2-digit", minute: "2-digit" }}),
                        project,
                        llm: data.llm_used
                    }}]);
                }} catch (err) {{
                    setMessages(prev => [...prev, {{
                        role: "stella",
                        text: "❌ Erreur de connexion au Context Engine. Vérifie que le service est actif sur Railway.",
                        time: new Date().toLocaleTimeString("fr-FR", {{ hour: "2-digit", minute: "2-digit" }}),
                        project
                    }}]);
                }}
                
                setLoading(false);
            }}, [loading, project]);
            
            const handleKeyDown = (e) => {{
                if (e.key === "Enter" && !e.shiftKey) {{
                    e.preventDefault();
                    sendMessage(input);
                }}
            }};
            
            const handleTextareaInput = (e) => {{
                setInput(e.target.value);
                e.target.style.height = "42px";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }};
            
            const totalCompleted = stats ? stats.brands?.reduce((s, b) => s + (b.completed || 0), 0) : 0;
            const totalPending = stats ? stats.brands?.reduce((s, b) => s + (b.pending || 0), 0) : 0;
            const isOnline = ceHealth?.context_engine?.status === "ok";
            
            return (
                <div className="stella-container">
                    {{/* HEADER */}}
                    <div className="stella-header">
                        <div className="stella-logo">★</div>
                        <div>
                            <div className="stella-title">STELLA V7</div>
                            <div className="stella-subtitle">Cerveau permanent — PlaneteBeauty</div>
                        </div>
                        <div className={{`stella-status ${{isOnline !== false ? "online" : "offline"}}`}}>
                            <span className="status-dot"></span>
                            {{isOnline !== false ? "En ligne" : "Hors ligne"}}
                        </div>
                    </div>
                    
                    {{/* CHAT AREA */}}
                    <div className="chat-area">
                        {{/* Stats bar */}}
                        {{stats && (
                            <div className="stats-bar">
                                <div className="stat">Produits: <span className="stat-value">&nbsp;{{stats.total}}</span></div>
                                <div className="stat">Enrichis: <span className="stat-value" style={{{{color:"#2E7D32"}}}}>&nbsp;{{totalCompleted}}</span></div>
                                <div className="stat">En attente: <span className="stat-value" style={{{{color:"#E8871E"}}}}>&nbsp;{{totalPending}}</span></div>
                                <div className="stat">Marques: <span className="stat-value">&nbsp;{{stats.brands?.length}}</span></div>
                            </div>
                        )}}
                        
                        {{/* Project selector */}}
                        <div className="project-selector">
                            {{PROJECTS.map(p => (
                                <button
                                    key={{p.id}}
                                    className={{`project-btn ${{project === p.id ? "active" : ""}}`}}
                                    onClick={{() => setProject(p.id)}}
                                >
                                    {{p.icon}} {{p.label}}
                                </button>
                            ))}}
                        </div>
                        
                        {{/* Messages */}}
                        <div className="messages">
                            {{messages.map((msg, i) => (
                                <div key={{i}} className={{`message ${{msg.role}}`}}>
                                    <div className="message-avatar">
                                        {{msg.role === "stella" ? "★" : "B"}}
                                    </div>
                                    <div>
                                        <div className="message-bubble">{{msg.text}}</div>
                                        <div className="message-meta">
                                            {{msg.time}}
                                            {{msg.llm && ` · ${{msg.llm}}`}}
                                            {{msg.project && msg.project !== "GENERAL" && ` · ${{msg.project}}`}}
                                        </div>
                                    </div>
                                </div>
                            ))}}
                            
                            {{loading && (
                                <div className="message stella">
                                    <div className="message-avatar">★</div>
                                    <div className="message-bubble">
                                        <div className="typing-indicator">
                                            <div className="typing-dot"></div>
                                            <div className="typing-dot"></div>
                                            <div className="typing-dot"></div>
                                        </div>
                                    </div>
                                </div>
                            )}}
                            
                            <div ref={{messagesEndRef}} />
                        </div>
                        
                        {{/* Quick actions (show only when few messages) */}}
                        {{messages.length <= 1 && (
                            <div className="quick-actions">
                                {{QUICK_ACTIONS.map((q, i) => (
                                    <button key={{i}} className="quick-action" onClick={{() => sendMessage(q)}}>
                                        {{q}}
                                    </button>
                                ))}}
                            </div>
                        )}}
                        
                        {{/* Input */}}
                        <div className="input-area">
                            <textarea
                                ref={{textareaRef}}
                                value={{input}}
                                onChange={{handleTextareaInput}}
                                onKeyDown={{handleKeyDown}}
                                placeholder={{`Parle à STELLA (${{PROJECTS.find(p => p.id === project)?.label}})...`}}
                                rows="1"
                            />
                            <button
                                className="send-btn"
                                onClick={{() => sendMessage(input)}}
                                disabled={{loading || !input.trim()}}
                            >
                                →
                            </button>
                        </div>
                    </div>
                </div>
            );
        }}
        
        ReactDOM.render(<App />, document.getElementById("app"));
    </script>
</body>
</html>"""


# === MOUNT STATIC ===
# Try to serve static files if they exist
import pathlib
static_path = pathlib.Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
