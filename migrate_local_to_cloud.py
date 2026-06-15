"""
migrate_local_to_cloud.py
--------------------------
ONE-TIME script. Run this ONCE on your Mac to copy the faces you've already
indexed in your local (Docker) Qdrant up into the shared Qdrant Cloud database.

It does NOT re-process any photos and does NOT delete anything. It simply copies
every stored fingerprint and its name across. After it finishes once, you never
need to run it again.

HOW TO USE (on your Mac):
  1. Make sure your local Qdrant (Docker) is running, the same as before.
  2. Paste your Qdrant Cloud details into the two lines marked >>> below.
  3. In Terminal, in this folder, run:  python migrate_local_to_cloud.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ---------------------------------------------------------------------------
# >>> PASTE YOUR QDRANT CLOUD DETAILS HERE (this file stays on your Mac only):
CLOUD_URL = "https://YOUR-CLUSTER-URL.qdrant.io"
CLOUD_API_KEY = "YOUR-QDRANT-CLOUD-API-KEY"
# ---------------------------------------------------------------------------

# Your existing local database (unchanged from the original project):
LOCAL_HOST = "localhost"
LOCAL_PORT = 6333

COLLECTION_NAME = "faces_collection"
VECTOR_SIZE = 512
BATCH = 100  # copy in small groups so memory stays low


def ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def migrate(src: QdrantClient, dst: QdrantClient,
            collection: str = COLLECTION_NAME, batch: int = BATCH) -> int:
    """Copy every point (vector + name) from src to dst. Returns how many copied."""
    ensure_collection(dst)
    copied = 0
    offset = None
    while True:
        points, offset = src.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        dst.upsert(collection_name=collection, points=points)
        copied += len(points)
        print(f"  ...copied {copied} faces so far")
        if offset is None:
            break
    return copied


def main():
    print("Connecting to your local database...")
    src = QdrantClient(LOCAL_HOST, port=LOCAL_PORT)

    print("Connecting to Qdrant Cloud...")
    dst = QdrantClient(url=CLOUD_URL, api_key=CLOUD_API_KEY, timeout=120)

    print("Copying faces (this does not change your local data)...")
    total = migrate(src, dst)

    print("\n" + "=" * 50)
    print(f"  Done. {total} faces are now in the cloud database.")
    print(f"  Cloud now holds: {dst.count(collection_name=COLLECTION_NAME).count} faces.")
    print("=" * 50)


if __name__ == "__main__":
    main()
