# ============================================================
# Trading Backend — Docker Build
# ============================================================
# Runs the autonomous trading cycle (collect → analyze → trade)
# as a standalone container with its own copy of the backend code.
#
# Build:
#   cd sun/trading-service
#   docker build -t trading-service .
# ============================================================

FROM python:3.11-slim AS deps

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Production runner ─────────────────────────────────────────
FROM python:3.11-slim AS runner
WORKDIR /app

# Install wget for healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system --gid 1001 appgrp \
    && useradd --system --uid 1001 --gid appgrp -m -d /home/appusr appusr

# Create logs directory
RUN mkdir -p /app/logs && chown -R appusr:appgrp /app/logs

# ── Copy Python venv ──────────────────────────────────────────
COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Copy backend source ──────────────────────────────────────
COPY app/ ./app/
COPY scripts/ ./scripts/

# ── Copy the cycle backend entrypoint ────────────────────────
COPY cycle_main.py ./cycle_main.py
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

RUN mkdir -p /app/logs/cycles /app/logs/v2 /app/memory

RUN chown -R appusr:appgrp /app

ENV PYTHONPATH="/app"
ENV SHARED_CODEBASE_PATH="/app"

USER appusr

HEALTHCHECK --interval=60s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "print('cycle-backend alive')" || exit 1

CMD ["./entrypoint.sh"]
