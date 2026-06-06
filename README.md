# Ulysses Legal NER

Microservico de extracao de entidades juridicas (FastAPI + legal-bert-ner).

## Deploy Railway (mesmo projeto que RAGjuridico)

1. No **mesmo** projeto Railway do RAGjuridico: **New** → **GitHub Repo** → este repositorio
2. Nome do servico: **`ulysses`**
3. **Config file path**: `railway.toml`
4. **Public domain**: desligado (rede privada)
5. **RAM**: minimo 2 GB

Detalhes: [deploy/railway/README.md](deploy/railway/README.md)

## Integracao RAGjuridico

No servico `api` e `worker` do RAGjuridico (Shared Variables):

```env
LEGAL_NER_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract
```

## Endpoints

- `GET /health` — readiness (503 ate modelo carregar)
- `POST /api/v1/extract` — body: `{ "text", "confidence_threshold", "include_regex" }`

## Local

```bash
docker compose up --build
# http://localhost:5522/health
```
