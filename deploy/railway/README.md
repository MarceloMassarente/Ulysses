# Ulysses — deploy Railway

Microserviço Legal NER consumido pelo RAGjuridico via `LEGAL_NER_ENDPOINT`.

## Setup no mesmo projeto Railway

1. **New** → **GitHub Repo** → `MarceloMassarente/Ulysses`
2. Nome do servico: **`ulysses`** (para `${{ulysses.RAILWAY_PRIVATE_DOMAIN}}`)
3. **Config file path**: `railway.toml`
4. **Sem** dominio publico
5. **RAM**: minimo **2 GB** (modelo PyTorch em memoria)
6. Variaveis opcionais: `deploy/railway/env.example`

## Integracao RAGjuridico

No servico `api` e `worker` do RAGjuridico:

```env
LEGAL_NER_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract
```

Endpoint exposto: `POST /api/v1/extract` (payload: `text`, `confidence_threshold`, `include_regex`).

Health: `GET /health` (503 ate o modelo carregar — timeout 300s no primeiro deploy).

Guia stack completo: `RAGjuridico/docs/operations/deploy-railway-stack.md`
