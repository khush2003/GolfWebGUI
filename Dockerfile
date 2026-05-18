# syntax=docker/dockerfile:1.7

FROM node:24-alpine AS client-build
WORKDIR /client
COPY client/package.json client/package-lock.json* ./
RUN npm install --include=dev --no-audit --no-fund
COPY client/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=3000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY server.py /app/server.py
COPY scripts/ /app/scripts/
COPY --from=client-build /client/dist /app/client/dist

EXPOSE 3000

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]
