"""
app.py
------
The web app with passkey authentication and all routes.

Pages:
  GET  /login     -> login screen (public)
  GET  /          -> home
  GET  /add       -> add a person + name directory
  GET  /search    -> search a face
  GET  /log       -> access log

API:
  POST /api/login   -> authenticate (email + passkey)
  POST /api/logout  -> clear session
  POST /api/add     -> add photos under a name (with thumbnails)
  POST /api/search  -> identify a face
  GET  /api/names   -> list all names with count + last_updated
  GET  /api/person/{name} -> get thumbnails for a person
  GET  /api/log     -> get access log entries
  GET  /health      -> status + face count
"""

import os
import hmac
import hashlib
import base64
import json
import time
from typing import List
from urllib.parse import quote

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

import face_engine
import database

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

RECOGNITION_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.40"))
ACCESS_PASSKEY = os.environ.get("ACCESS_PASSKEY", "")  # set in HF Secrets
SESSION_SECRET = os.environ.get("SESSION_SECRET", ACCESS_PASSKEY or "dev-secret")
IS_PRODUCTION = bool(os.environ.get("QDRANT_URL"))  # HF Spaces sets this


# ---- session helpers ----

def sign_session(email: str) -> str:
    payload = json.dumps({"email": email, "ts": int(time.time())})
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode()
    sig = hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()[:24]
    return f"{payload_b64}.{sig}"


def verify_session(cookie: str) -> str | None:
    try:
        payload_b64, sig = cookie.rsplit(".", 1)
        expected = hmac.new(SESSION_SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload.get("email")
    except Exception:
        return None


# ---- auth middleware ----

PROTECTED_EXACT = {"/add", "/log", "/api/add", "/api/names", "/api/log", "/api/logout"}
PROTECTED_PREFIX = ("/api/person/",)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        needs_auth = path in PROTECTED_EXACT or any(path.startswith(p) for p in PROTECTED_PREFIX)
        if not needs_auth:
            return await call_next(request)

        cookie = request.cookies.get("session")
        email = verify_session(cookie) if cookie else None
        if not email:
            if path.startswith("/api/"):
                return JSONResponse({"error": "Not authenticated"}, status_code=401)
            return RedirectResponse("/login")

        request.state.email = email
        return await call_next(request)


app = FastAPI(title="Face Index")
app.add_middleware(AuthMiddleware)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# ---- pages ----

@app.get("/login")
def login_page():
    return FileResponse(os.path.join(WEB_DIR, "login.html"))

@app.get("/")
def home():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))

@app.get("/add")
def add_page():
    return FileResponse(os.path.join(WEB_DIR, "add.html"))

@app.get("/search")
def search_page():
    return FileResponse(os.path.join(WEB_DIR, "search.html"))

@app.get("/log")
def log_page():
    return FileResponse(os.path.join(WEB_DIR, "log.html"))


# ---- auth API ----

@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    passkey = (body.get("passkey") or "").strip()

    if not email or "@" not in email:
        return JSONResponse({"error": "Enter a valid email."}, status_code=400)
    if not ACCESS_PASSKEY:
        return JSONResponse({"error": "Passkey not configured on the server."}, status_code=500)
    if not hmac.compare_digest(passkey, ACCESS_PASSKEY):
        return JSONResponse({"error": "Incorrect passkey."}, status_code=403)

    database.log_access(email)
    token = sign_session(email)
    resp = JSONResponse({"ok": True, "email": email})
    resp.set_cookie(
        "session", token, httponly=True,
        samesite="none" if IS_PRODUCTION else "lax",
        secure=IS_PRODUCTION,
        max_age=86400 * 7,
    )
    return resp


@app.post("/api/logout")
async def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session", samesite="none" if IS_PRODUCTION else "lax", secure=IS_PRODUCTION)
    return resp


# ---- health ----

@app.get("/health")
def health():
    try:
        return {"status": "ok", "faces_indexed": database.count()}
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ---- faces API ----

@app.post("/api/add")
async def api_add(name: str = Form(...), files: List[UploadFile] = File(...)):
    name = (name or "").strip()
    if not name:
        return JSONResponse({"error": "Enter a name first."}, status_code=400)
    if not files:
        return JSONResponse({"error": "Add at least one photo."}, status_code=400)

    embeddings, sources, thumbnails, skipped = [], [], [], []
    for f in files:
        data = await f.read()
        emb = face_engine.embedding_from_bytes(data)
        if emb is None:
            skipped.append({"file": f.filename, "reason": "No clear face detected"})
        else:
            embeddings.append(emb)
            sources.append(f.filename)
            thumb = face_engine.make_thumbnail(data)
            thumbnails.append(thumb)

    added = database.add_faces(name, embeddings, sources=sources, thumbnails=thumbnails)
    return {
        "name": name,
        "added": added,
        "skipped": skipped,
        "total_indexed": database.count(),
    }


@app.post("/api/search")
async def api_search(file: UploadFile = File(...)):
    data = await file.read()
    emb = face_engine.embedding_from_bytes(data)
    if emb is None:
        return JSONResponse(
            {"error": "No clear face detected in that photo. Try another one."},
            status_code=422,
        )

    results = database.search(emb, limit=3)
    if not results:
        return {"unknown": True, "reason": "empty_database",
                "threshold": RECOGNITION_THRESHOLD, "results": []}

    unknown = results[0]["score"] < RECOGNITION_THRESHOLD
    return {"unknown": unknown, "threshold": RECOGNITION_THRESHOLD, "results": results}


@app.get("/api/names")
def api_names():
    return database.list_names()


@app.get("/api/person/{name}")
def api_person(name: str):
    return database.get_person_thumbnails(name)


@app.put("/api/person/{name}")
async def api_rename_person(name: str, request: Request):
    body = await request.json()
    new_name = (body.get("new_name") or "").strip()
    if not new_name:
        return JSONResponse({"error": "Enter a new name."}, status_code=400)
    updated = database.rename_person(name, new_name)
    return {"old_name": name, "new_name": new_name, "updated": updated}


@app.delete("/api/person/{name}")
async def api_delete_person(name: str):
    deleted = database.delete_person(name)
    return {"name": name, "deleted": deleted}


@app.get("/api/log")
def api_log():
    return database.get_access_log()