FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clip_annotator ./clip_annotator

EXPOSE 8080

# One worker on purpose: the annotation store is single-writer (per-annotator
# locks and the Drive write-behind cache live in process memory). Threads give
# the concurrency: each annotator's browser keeps a few video range requests
# open, so 32 threads leave room for 10 people at once.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads ${GUNICORN_THREADS:-32} --timeout 120 'clip_annotator.wsgi:create_app_from_env()'"]
