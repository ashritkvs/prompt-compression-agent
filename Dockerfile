FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Hugging Face Spaces runs the container with a restricted HOME, so point the
# model/download caches at a guaranteed-writable location. Harmless elsewhere.
ENV HF_HOME=/tmp/hf
ENV XDG_CACHE_HOME=/tmp/cache

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
