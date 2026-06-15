# Hugging Face Spaces (Docker SDK) image for the Face Index app.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/user

# System libraries: build tools (InsightFace compiles on install),
# plus libs needed by onnxruntime (libgomp1) and OpenCV (libglib2.0-0).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The user Hugging Face Spaces runs the container as.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# --- Python dependencies ---
COPY requirements.txt .
# Build prerequisites for InsightFace, then InsightFace itself (no isolation so it
# can see numpy + Cython), then everything else, then re-pin numpy to the 1.x line.
RUN pip install numpy==1.26.4 Cython \
    && pip install --no-build-isolation insightface==0.7.3 \
    && pip install -r requirements.txt \
    && pip install numpy==1.26.4

# Bake the face model into the image so cold starts are instant (no runtime download).
RUN python -c "import warnings; warnings.filterwarnings('ignore'); \
from insightface.app import FaceAnalysis; \
a = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
a.prepare(ctx_id=0, det_size=(640, 640)); print('Model baked into image.')"

# --- App code ---
COPY . .

# Hand everything (including the downloaded model in /home/user/.insightface) to the user.
RUN chown -R user:user /home/user
USER user

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
