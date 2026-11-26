FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap util-linux ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/scicode
COPY eval/scicode_verified/requirements-2024.txt eval/scicode_verified/requirements-2025.txt /tmp/
RUN python -m venv /opt/science2024 && python -m venv /opt/science2025 \
    && /opt/science2024/bin/pip install --no-cache-dir -r /tmp/requirements-2024.txt \
    && /opt/science2025/bin/pip install --no-cache-dir -r /tmp/requirements-2025.txt
COPY eval/scicode_verified/run.py eval/scicode_verified/pipeline.py /opt/scicode/eval/scicode_verified/
