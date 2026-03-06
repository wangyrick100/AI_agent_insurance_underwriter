FROM python:3.11-slim AS backend

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir .
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "underwriting_suite.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Next.js UI ───────────────────────────────
FROM node:20-alpine AS ui-builder
WORKDIR /ui
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci
COPY ui/ .
RUN npm run build

FROM node:20-alpine AS ui
WORKDIR /ui
COPY --from=ui-builder /ui/.next .next
COPY --from=ui-builder /ui/public public
COPY --from=ui-builder /ui/node_modules node_modules
COPY --from=ui-builder /ui/package.json .
EXPOSE 3000
CMD ["npm", "start"]
