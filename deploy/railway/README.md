# Ulysses — deploy Railway

Microserviço Legal NER consumido pelo RAGjuridico via `LEGAL_NER_ENDPOINT`.

## Setup no mesmo projeto Railway

1. **New** → **GitHub Repo** → `MarceloMassarente/Ulysses`
2. Nome do servico: **`ulysses`** (para `${{ulysses.RAILWAY_PRIVATE_DOMAIN}}`)
3. **Config file path**: `railway.toml`
4. **Sem** dominio publico
5. **RAM**: minimo **8 GB** no dashboard Railway (modelo PyTorch em memoria)
6. Variaveis opcionais: `deploy/railway/env.example`

## Integracao RAGjuridico

No servico `api` e `worker` do RAGjuridico:

```env
LEGAL_NER_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract
LEGAL_NER_BATCH_ENDPOINT=http://${{ulysses.RAILWAY_PRIVATE_DOMAIN}}/api/v1/extract_batch
LEGAL_NER_INCLUDE_REGEX=false
```

Endpoints expostos:

- `POST /api/v1/extract` (payload: `text`, `confidence_threshold`, `include_regex`)
- `POST /api/v1/extract_batch` (payload: `texts`, `confidence_threshold`, `include_regex`)

Use o endpoint batch no worker do RAGjuridico para processar chunks em lotes. `include_regex=false` evita duplicar regex quando o RAGjuridico ja extrai citacoes e identificadores localmente.

Health: `GET /health` (durante cold start pode haver falha de conexao ate o app subir; depois 503 ate o modelo carregar).

Guia stack completo: `RAGjuridico/docs/operations/deploy-railway-stack.md`
