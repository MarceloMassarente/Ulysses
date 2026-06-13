# Ulysses Legal NER

Microservico de extracao de entidades juridicas (FastAPI + legal-bert-ner).

## Deploy Railway (mesmo projeto que RAGjuridico)

1. No **mesmo** projeto Railway do RAGjuridico: **New** → **GitHub Repo** → este repositorio
2. Nome do servico: **`ulysses`**
3. **Config file path**: `railway.toml`
4. **Public domain**: desligado (rede privada)
5. **RAM**: minimo 8 GB, configurado no dashboard Railway

Detalhes: [deploy/railway/README.md](deploy/railway/README.md)

## Integracao RAGjuridico

No servico `api` e `worker` do RAGjuridico (Shared Variables):

```env
LEGAL_NER_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract
LEGAL_NER_BATCH_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract_batch
LEGAL_NER_INCLUDE_REGEX=false
```

## Endpoints

- `GET /health` — readiness (conexao falha ate o app subir; depois 503 ate modelo carregar)
- `POST /api/v1/extract` — body: `{ "text", "confidence_threshold", "include_regex" }`
- `POST /api/v1/extract_batch` — body: `{ "texts", "confidence_threshold", "include_regex" }`; `texts` aceita ate `LEGAL_NER_MAX_BATCH_ITEMS`

Por padrao o servico limita cada texto a `LEGAL_NER_MAX_INPUT_CHARS=50000`, admite ate 2 requests por processo (`LEGAL_NER_MAX_IN_FLIGHT=2`) e aguarda ate 30s por slot antes de retornar 503. O container sobe com um unico worker uvicorn para evitar multiplas copias do modelo; a inferencia continua protegida por lock interno, entao throughput deve vir de batches e replicas, nao de mais workers.

### Observabilidade

`GET /health` reporta `in_flight` (slots de inferencia ocupados) e `requests` (contadores acumulados `requests_total`, `batch_requests_total`, `errors_503`, `errors_504`). Cada request emite um log estruturado `ner_request` com o breakdown de tempos, util para atribuir ganho a batch vs. concorrencia:

```
ner_request endpoint=extract_batch outcome=success total_ms=842 in_flight=1 batch_size=32 normalize_ms=4.1 infer_ms=790.3 regex_ms=12.7 postprocess_ms=3.5 entities=118
```

Como o `_infer_lock` segura o lock durante o batch inteiro, um `batch_size` alto que ultrapasse `LEGAL_NER_QUEUE_TIMEOUT_SECONDS` faz requests concorrentes retornarem 503 — acompanhe `errors_503` e o `infer_ms` do batch para dimensionar `LEGAL_NER_MAX_BATCH_ITEMS`.

## Local

```bash
docker compose up --build
# http://localhost:5522/health
```
