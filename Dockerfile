FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    numpy \
    pillow \
    python-multipart \
    "tritonclient[grpc]"

COPY api.py /app/api.py
COPY infer_triton.py /app/infer_triton.py
COPY imagenet_classes.txt /app/imagenet_classes.txt

EXPOSE 5000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
