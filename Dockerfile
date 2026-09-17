FROM python:3.11-slim

# ffmpeg is required by yt-dlp for subtitle conversion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package + its dependencies (yt-dlp is a declared dependency).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Persisted state — mount volumes here in production and back them up.
ENV YTRAG_DATA_DIR=/data \
    YTRAG_CHROMA_DIR=/chroma \
    PYTHONIOENCODING=utf-8
VOLUME ["/data", "/chroma"]

EXPOSE 8000

# SINGLE worker only: each Telegram bot runs a getUpdates long-poller on a daemon
# thread inside the process; multiple workers would double-poll and drop updates.
CMD ["uvicorn", "ytrag.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
