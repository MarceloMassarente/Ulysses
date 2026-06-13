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
```

## Endpoints

- `GET /health` — readiness (conexao falha ate o app subir; depois 503 ate modelo carregar)
- `POST /api/v1/extract` — body: `{ "text", "confidence_threshold", "include_regex" }`

Por padrao o servico limita `text` a `LEGAL_NER_MAX_INPUT_CHARS=50000`, admite ate 4 requests por processo (`LEGAL_NER_MAX_IN_FLIGHT=4`) e aguarda ate 5s por slot antes de retornar 503. O container sobe com um unico worker uvicorn para evitar multiplas copias do modelo; o tokenizer continua protegido por lock interno.

## Local

```bash
docker compose up --build
# http://localhost:5522/health
```
