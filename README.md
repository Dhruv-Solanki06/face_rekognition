---
title: Face Index
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Face Index

A shared, on-device-style face search tool for a small team.

- **Add a person** — upload one or many photos under a single name.
- **Search a face** — upload a photo to find the closest match, with a confidence reading.

All data is shared through a Qdrant Cloud database, so anything one person adds is
available to everyone.

## Configuration (set as Space *Secrets*, not in code)

| Secret | Description |
| --- | --- |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Your Qdrant Cloud API key |

Optional setting (Space *Variables*):

| Variable | Default | Description |
| --- | --- | --- |
| `RECOGNITION_THRESHOLD` | `0.40` | Cosine score below which a face is reported as "Unknown person" |
