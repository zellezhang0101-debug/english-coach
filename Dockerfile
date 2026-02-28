ARG PY_BASE_IMAGE=python:3.12-slim
FROM ${PY_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (minimal). Add build tools only if you later need them.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY README.md /app/README.md

# Optional Ming-UniAudio wrapper files (not required for production).
COPY ming_uniaudio_server.py /app/ming_uniaudio_server.py
COPY requirements_ming_uniaudio_server.txt /app/requirements_ming_uniaudio_server.txt

EXPOSE 8000

# Run Flask app via gunicorn (production WSGI).
# Many cloud platforms inject $PORT; default to 8000 for local/docker-compose.
CMD ["sh", "-c", "gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:${PORT:-8000} app:app"]

