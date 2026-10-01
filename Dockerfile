# Chapter and Verse API. Runs the same way locally (docker compose) and on Cloud Run.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py schema.sql ./
COPY static ./static

RUN useradd --create-home --uid 10001 app
USER app

# Cloud Run sets PORT; locally it defaults to 8080.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*' --no-access-log"]
