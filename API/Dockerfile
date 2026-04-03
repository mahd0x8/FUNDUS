# ── Backend — FastAPI inference service ──────────────────────────────────────
#
# Default: CPU-only PyTorch (smaller image, works everywhere).
# For GPU support see docker-compose.yml notes.

FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first so subsequent `pip install -r requirements.txt`
# doesn't pull in the larger CUDA wheel.
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch/torchvision already satisfied above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY . .

# Model checkpoint is mounted at runtime (see docker-compose.yml)
ENV CHECKPOINT_PATH=/app/model/best_classifier.pt
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
