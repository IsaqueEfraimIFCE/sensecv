FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frame_annotator ./frame_annotator

EXPOSE 8080

# One worker on purpose: the Drive annotation store keeps a write-through cache
# in process memory (single-writer design). Threads give the concurrency.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 120 'frame_annotator.wsgi:create_app_from_env()'"]
