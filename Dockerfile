FROM python:3.11-slim AS builder

WORKDIR /app

# Pure-Python/onnxruntime wheels - no build toolchain needed.
COPY requirements-production.txt .
RUN pip install --no-cache-dir -r requirements-production.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Torch-free serving code: int8 ONNX encoder + numpy tail + heads.
COPY server.py inference_onnx.py text_norm.py aisle_map.py ./

# Code license (AGPL-3.0) + Gemma model NOTICE travel with the image.
COPY LICENSE NOTICE ./

# The exported int8 ONNX artifact (built by export_onnx.py). ~330 MB.
COPY aisle_model_onnx/ ./aisle_model_onnx/

# Cap CPU threads: without this, single-request latency balloons on many-core hosts.
ENV OMP_NUM_THREADS=4

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
