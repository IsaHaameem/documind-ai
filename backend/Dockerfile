FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip check
RUN useradd --uid 10001 --create-home app && mkdir /model-cache && chown app /model-cache
COPY documind documind
COPY alembic.ini .
COPY migrations migrations
COPY evaluation evaluation
USER app
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn documind.main:app --host 0.0.0.0 --port 8000 --limit-concurrency 64"]
