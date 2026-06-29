# Observability Plan

This document defines the observability strategy across three pillars:
metrics, logs, and tracing. It follows the same structure as the
evaluation plan: what to measure, how to measure it, and what tooling
to use.

---

## 1. Metrics — Prometheus + Grafana

### What is not covered today

There is no metrics instrumentation in the current stack. The `/health`
endpoint confirms PostgreSQL connectivity but provides no time-series
data on throughput, latency, or error rates.

### What to measure

| Metric | Type | Purpose |
|---|---|---|
| HTTP request rate | Counter | Overall throughput |
| HTTP request duration | Histogram | P95/P99 API latency per route |
| Chat errors | Counter | Error breakdown by type |
| LangGraph node duration | Histogram | Per-node LLM latency (chat, extract, duplicate_check, save) |
| Requests saved | Counter | Successful end-to-end submissions |
| Duplicate detections | Counter | Duplicate pipeline hit rate |
| Active sessions | Gauge | Sessions in the rate-limit window |
| PostgreSQL connections | Gauge | DB pool pressure (via sidecar exporter) |

### Implementation approach

1. Add `prometheus-fastapi-instrumentator` (preferred) or `prometheus-client` directly to dependencies. The instrumentator auto-instruments all HTTP routes and exposes a `/metrics` endpoint with zero boilerplate; it is built on top of `prometheus-client` and both can be used together. If the instrumentator proves incompatible, fall back to `prometheus-client` alone and write the HTTP middleware manually.
2. Define custom application-level counters, histograms, and gauges for the business metrics above using `prometheus-client` directly; increment them inside the chat endpoint and workflow nodes.
3. Add a `postgres-exporter` sidecar that scrapes `pg_stat_*` views.
4. Configure Prometheus with a scrape config targeting the API on port 8000 and the exporter.
5. Import community Grafana dashboards for FastAPI (ID 17658) and PostgreSQL (ID 9628); add custom panels for the business metrics.
6. Add three alerting rules: high error rate, slow LangGraph node (P95 > 10 s), PostgreSQL exporter down.

---

## 2. Logs — Fluent Bit + Elasticsearch + Kibana

### What is not covered today

Logs are emitted as plain text via Python's `basicConfig`. There is no
centralised log storage or search capability.

### What to capture

| Signal | Purpose |
|---|---|
| ERROR-level logs | Application errors and tracebacks |
| Duplicate detection events | How often and which sessions trigger the pipeline |
| Blocked prompt injections | Guardrail efficacy over time |
| Rate-limit hits | Abuse patterns per session |
| LangGraph node logs | Per-turn workflow trace |

### Implementation approach

1. Replace `basicConfig` with a JSON log formatter (`python-json-logger`) so every log call emits a structured JSON record to stdout. No Dockerfile changes needed — Gunicorn already writes to stdout.
2. Deploy **Fluent Bit** as a container. It reads Docker container stdout logs directly via its built-in Docker input plugin and forwards them to Elasticsearch — no additional agent needed.
3. Deploy **Elasticsearch** as the log store (single-node for dev; enable `xpack.security` + TLS in production).
4. Deploy **Kibana**, create an index pattern `chatbot-logs-*`, and build saved searches for the signals above.

---

## 3. Tracing — LangSmith (already wired)

### What is not covered today

`langsmith_tracing` is defined in `Settings` and the API key and
project fields are already read from environment variables. The feature
is wired but not enabled in production.

### Implementation approach

Enable by setting `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and
`LANGSMITH_PROJECT` in `.env`. No code change is required.

LangSmith captures every LangGraph node invocation with full
input/output, latency per node, and token usage — complementing the
Prometheus histograms with per-request drill-down and a visual graph
execution trace.

---

## 4. Infrastructure additions (docker-compose)

Six new services alongside the existing `postgres` and `api`:

| Service | Image | Purpose |
|---|---|---|
| `prometheus` | `prom/prometheus` | Scrapes and stores metrics |
| `grafana` | `grafana/grafana` | Dashboards and alerting |
| `postgres-exporter` | `prometheuscommunity/postgres-exporter` | Exposes PostgreSQL metrics |
| `elasticsearch` | `docker.elastic.co/elasticsearch` | Log storage and indexing |
| `kibana` | `docker.elastic.co/kibana` | Log search and visualisation |
| `fluent-bit` | `fluent/fluent-bit` | Ships container logs to Elasticsearch |

New named volumes: `prometheus_data`, `grafana_data`, `es_data`.

---

## 5. Implementation order

| Step | Task | Effort |
|---|---|---|
| 1 | Add `prometheus-fastapi-instrumentator` + `python-json-logger` to `pyproject.toml` | Trivial |
| 2 | Replace plain-text logger with JSON formatter in `main.py` | Low |
| 3 | Wire Instrumentator and define custom metrics in `main.py` and `workflow.py` | Low |
| 4 | Create `observability/` config files (Prometheus scrape config, Fluent Bit config, alert rules) | Low |
| 5 | Extend `docker-compose.yml` with the 6 new services and 3 new volumes | Low |
| 6 | Spin up, import Grafana dashboards, create Kibana index pattern | Trivial |
| 7 | Enable LangSmith tracing via `.env` | Trivial |

---

## Summary

| Pillar | Tooling | Current gap | Fix | Effort |
|---|---|---|---|---|
| Metrics | Prometheus + Grafana | No instrumentation | `prometheus-fastapi-instrumentator` (or `prometheus-client` directly as fallback) + custom metrics | Low |
| Logs | Fluent Bit → Elasticsearch → Kibana | Plain-text stdout, no centralised search | JSON formatter + Fluent Bit | Low |
| Tracing | LangSmith | Wired but disabled | Set env vars in `.env` | Trivial |
| Alerting | Prometheus alert rules | No alerts | 3 rules (error rate, node latency, DB down) | Low |
