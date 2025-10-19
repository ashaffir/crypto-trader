FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY configs /app/configs
COPY src /app/src
COPY ui /app/ui

# Default paths
ENV CONFIG_PATH=/app/configs/config.yaml
ENV LOGBOOK_DIR=/app/data/logbook
ENV PYTHONPATH=/app

RUN mkdir -p /app/data/logbook

CMD ["python", "-m", "src.app"]
