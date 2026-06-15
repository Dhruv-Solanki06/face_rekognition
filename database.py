"""
database.py
-----------
Talks to the Qdrant vector database. Handles faces, name directory, and access log.

Collections:
  faces_collection  — face fingerprints + thumbnails + metadata
  access_log        — login events (email + timestamp)
"""

import os
import uuid
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "faces_collection")
LOG_COLLECTION = "access_log"
VECTOR_SIZE = 512
LOG_VECTOR_SIZE = 1  # dummy dimension for the log (Qdrant requires vectors)

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        if url:
            _client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=60)
        else:
            path = os.environ.get("QDRANT_PATH")
            _client = QdrantClient(path=path) if path else QdrantClient(":memory:")
    return _client


def ensure_collection() -> None:
    c = get_client()
    try:
        c.get_collection(COLLECTION_NAME)
    except Exception:
        c.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def ensure_log_collection() -> None:
    c = get_client()
    try:
        c.get_collection(LOG_COLLECTION)
    except Exception:
        c.create_collection(
            collection_name=LOG_COLLECTION,
            vectors_config=VectorParams(size=LOG_VECTOR_SIZE, distance=Distance.COSINE),
        )


# ---- faces ----

def add_faces(name: str, embeddings: List[np.ndarray],
              sources: Optional[List[str]] = None,
              thumbnails: Optional[List[str]] = None) -> int:
    if not embeddings:
        return 0
    ensure_collection()
    c = get_client()
    now = datetime.now(timezone.utc).isoformat()
    points = []
    for i, emb in enumerate(embeddings):
        src = sources[i] if sources and i < len(sources) else None
        thumb = thumbnails[i] if thumbnails and i < len(thumbnails) else None
        payload = {"name": name, "source": src, "added_at": now}
        if thumb:
            payload["thumbnail"] = thumb
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=np.asarray(emb, dtype=np.float32).tolist(),
            payload=payload,
        ))
    c.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def confidence_from_score(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0)) * 100.0


def search(embedding: np.ndarray, limit: int = 3) -> List[dict]:
    ensure_collection()
    c = get_client()
    hits = c.query_points(
        collection_name=COLLECTION_NAME,
        query=np.asarray(embedding, dtype=np.float32).tolist(),
        limit=limit,
    ).points
    results = []
    for hit in hits:
        payload = hit.payload or {}
        score = float(hit.score)
        results.append({
            "name": payload.get("name", "Unknown"),
            "score": score,
            "confidence": round(confidence_from_score(score), 2),
            "source": payload.get("source"),
        })
    return results


def count() -> int:
    ensure_collection()
    return get_client().count(collection_name=COLLECTION_NAME).count


def rename_person(old_name: str, new_name: str) -> int:
    """Rename every point belonging to old_name to new_name."""
    ensure_collection()
    c = get_client()
    ids = []
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=COLLECTION_NAME, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            if (p.payload or {}).get("name") == old_name:
                ids.append(p.id)
        if offset is None:
            break
    if ids:
        for i in range(0, len(ids), 100):
            c.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"name": new_name},
                points=ids[i:i+100],
            )
    return len(ids)


def delete_person(name: str) -> int:
    """Delete every point belonging to a person."""
    ensure_collection()
    c = get_client()
    ids = []
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=COLLECTION_NAME, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            if (p.payload or {}).get("name") == name:
                ids.append(p.id)
        if offset is None:
            break
    if ids:
        from qdrant_client.models import PointIdsList
        c.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids),
        )
    return len(ids)


def list_names() -> List[Dict]:
    """
    Return a list of every person on record:
      [{name, count, last_updated}, ...]
    Fetches all points (payload only, no vectors) and aggregates.
    """
    ensure_collection()
    c = get_client()
    names: Dict[str, dict] = {}
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            nm = pl.get("name", "Unknown")
            added = pl.get("added_at", "")
            if nm not in names:
                names[nm] = {"name": nm, "count": 0, "last_updated": added}
            names[nm]["count"] += 1
            if added > names[nm]["last_updated"]:
                names[nm]["last_updated"] = added
        if offset is None:
            break
    return list(names.values())


def get_person_thumbnails(name: str) -> List[Dict]:
    """Return all thumbnails + metadata for a specific person."""
    ensure_collection()
    c = get_client()
    results = []
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            if pl.get("name") == name:
                results.append({
                    "id": str(p.id),
                    "source": pl.get("source"),
                    "added_at": pl.get("added_at", ""),
                    "thumbnail": pl.get("thumbnail"),
                })
        if offset is None:
            break
    return results


# ---- access log ----

def log_access(email: str) -> None:
    ensure_log_collection()
    c = get_client()
    c.upsert(collection_name=LOG_COLLECTION, points=[PointStruct(
        id=str(uuid.uuid4()),
        vector=[0.0],
        payload={"email": email, "timestamp": datetime.now(timezone.utc).isoformat()},
    )])


def get_access_log() -> List[Dict]:
    """Return all login events, newest first."""
    ensure_log_collection()
    c = get_client()
    entries = []
    offset = None
    while True:
        points, offset = c.scroll(
            collection_name=LOG_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            entries.append({
                "email": pl.get("email", ""),
                "timestamp": pl.get("timestamp", ""),
            })
        if offset is None:
            break
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries