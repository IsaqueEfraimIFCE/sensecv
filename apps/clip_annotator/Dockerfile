FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clip_annotator ./clip_annotator

EXPOSE 8080

# One worker on purpose: the Drive event store keeps a write-through cache in
# process memory (single-writer design). Threads give the concurrency, and
# video range requests are many and short.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 120 'clip_annotator.wsgi:create_app_from_env()'"]
