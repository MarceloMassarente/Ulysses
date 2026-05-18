"""
compare_ner_regex.py
Pipeline comparativo: REGEX vs NER (pypdf bruto) vs NER (Docling Markdown)

Fluxo:
  1. Extrai texto bruto do PDF via pypdf  (baseline rápido)
  2. Envia o PDF para o Docling em /v1/convert/file e obtém Markdown limpo
  3. Envia ambos ao microsserviço Legal-NER e compara com padrões Regex

Uso:
    python compare_ner_regex.py [caminho_do_pdf]
"""

import re
import sys
import io
import time
import os
import json

import requests

# Force UTF-8 output on Windows to avoid cp1252 codec errors with special chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── Configuração ───────────────────────────────────────────────────────────
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\marce\Documentos\OB\Dossie\Réplica - MARCELO Casa de Itapira.pdf"
)
NER_API   = "http://127.0.0.1:5522/api/v1/extract"
DOCLING   = "http://192.168.1.221:5001"

NER_THRESHOLD = 0.55   # confiança mínima para exibir uma entidade

# ─── Padrões Regex jurídicos ─────────────────────────────────────────────────
REGEX_PATTERNS = {
    "CPF":              r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    "CNPJ":             r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
    "PROCESSO":         r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b",
    "LEGISLACAO":       r"\b(?:[Aa]rt(?:\.?o?|igo)?\.?\s+\d+[°º]?|[Ll]ei\s+n?\.?\s*\d+(?:[.\d]+)*(?:/\d+)?)\b",
    "DATA":             r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    "VALOR":            r"\bR\$\s*[\d.,]+\b",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_via_pypdf(pdf_path: str) -> str:
    """Extrai texto bruto do PDF via pypdf."""
    try:
        import pypdf
    except ImportError:
        print("Instalando pypdf...")
        os.system("pip install pypdf -q")
        import pypdf

    reader = pypdf.PdfReader(pdf_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def convert_via_docling(pdf_path: str) -> str | None:
    """Envia o PDF ao Docling-serve e retorna o Markdown gerado."""
    endpoint = f"{DOCLING}/v1/convert/file"
    print(f"\n-> Enviando PDF para Docling: {endpoint}")
    try:
        with open(pdf_path, "rb") as f:
            t0 = time.time()
            resp = requests.post(
                endpoint,
                files={"files": (os.path.basename(pdf_path), f, "application/pdf")},
                # placeholder: images become ![Image](<!-- image -->) tokens, no base64 blobs
                data={"target_type": "inbody", "image_export_mode": "placeholder"},
                timeout=300,
            )
        elapsed = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(f"  X Docling retornou HTTP {resp.status_code}: {resp.text[:300]}")
            return None

        data = resp.json()
        doc = data.get("document") or {}
        md_text = doc.get("md_content") or doc.get("markdown") or doc.get("text_content")

        if not md_text:
            docs = data.get("documents") or data.get("results") or []
            if docs:
                first = docs[0]
                md_text = first.get("md_content") or first.get("markdown")

        if md_text:
            md_text = re.sub(r'\n{3,}', '\n\n', md_text)  # collapse excessive blank lines
            print(f"  OK Docling converteu em {elapsed:.0f}ms  ({len(md_text):,} chars)")
        else:
            print(f"  X Docling respondeu OK mas md_content nao encontrado.")
            print(f"     Chaves disponiveis: {list(data.keys())}")
            if "document" in data:
                print(f"     Chaves em 'document': {list(data['document'].keys())}")
        return md_text

    except requests.exceptions.ConnectionError:
        print(f"  X Nao foi possivel conectar ao Docling em {DOCLING}.")
        return None
    except Exception as e:
        print(f"  X Erro ao chamar Docling: {e}")
        return None



def extract_ner(text: str, label: str) -> tuple[list, float]:
    """Chama a API Legal-NER e retorna (entidades, tempo_ms)."""
    t0 = time.time()
    try:
        resp = requests.post(
            NER_API,
            json={"text": text, "confidence_threshold": NER_THRESHOLD},
            timeout=300,
        )
        elapsed = (time.time() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            print(f"  ✓ NER [{label}] → {elapsed:.0f}ms HTTP  |  "
                  f"{data['processing_time_ms']}ms CPU  |  "
                  f"{len(data['entities'])} entidades encontradas")
            return data["entities"], elapsed
        else:
            print(f"  ✗ NER retornou HTTP {resp.status_code}: {resp.text[:200]}")
            return [], elapsed
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Não foi possível conectar ao NER em {NER_API}.")
        return [], 0.0


def regex_extract(text: str) -> list[dict]:
    """Extrai entidades por Regex e retorna lista de dicts."""
    results = []
    for label, pattern in REGEX_PATTERNS.items():
        for match in sorted(set(re.findall(pattern, text))):
            results.append({"type": label, "word": match.strip()})
    return sorted(results, key=lambda x: (x["type"], x["word"]))


def print_section(title: str, entities: list[dict], key_type="ner"):
    """Imprime seção do relatório."""
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    if not entities:
        print("  (nenhuma entidade encontrada)")
        return
    if key_type == "regex":
        by_type: dict[str, list] = {}
        for e in entities:
            by_type.setdefault(e["type"], []).append(e["word"])
        for t, words in sorted(by_type.items()):
            for w in words:
                print(f"  [{t:20s}] {w}")
    else:
        by_group: dict[str, list] = {}
        for e in entities:
            by_group.setdefault(e["entity_group"], []).append((e["word"], e["score"]))
        for group, items in sorted(by_group.items()):
            items_sorted = sorted(items, key=lambda x: -x[1])
            for word, score in items_sorted:
                print(f"  [{group:20s}] {word}  ({score:.2f})")


def ner_delta(entities_pypdf: list, entities_docling: list) -> dict:
    """Compara os dois conjuntos e retorna entidades exclusivas de cada um."""
    set_pypdf   = {(e["entity_group"], e["word"]) for e in entities_pypdf}
    set_docling = {(e["entity_group"], e["word"]) for e in entities_docling}
    return {
        "only_in_pypdf":   set_pypdf   - set_docling,
        "only_in_docling": set_docling - set_pypdf,
        "common":          set_pypdf   & set_docling,
    }


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main():
    if not os.path.exists(PDF_PATH):
        print(f"✗ Arquivo não encontrado: {PDF_PATH}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  PIPELINE: Regex vs NER (pypdf) vs NER (Docling)")
    print(f"  Documento: {os.path.basename(PDF_PATH)}")
    print(f"{'═'*60}")

    # ── 1. Extração pypdf ──────────────────────────────────────────────────
    print("\n[1/4] Extraindo texto bruto via pypdf...")
    t0 = time.time()
    raw_text = extract_via_pypdf(PDF_PATH)
    pypdf_ms = (time.time() - t0) * 1000
    print(f"  ✓ {len(raw_text):,} chars extraídos em {pypdf_ms:.0f}ms")

    # ── 2. Regex ───────────────────────────────────────────────────────────
    print("\n[2/4] Executando extração Regex...")
    t0 = time.time()
    regex_entities = regex_extract(raw_text)
    regex_ms = (time.time() - t0) * 1000
    print(f"  ✓ {len(regex_entities)} matches em {regex_ms:.2f}ms")

    # ── 3. NER sobre texto bruto pypdf ────────────────────────────────────
    print("\n[3/4] Executando NER sobre texto bruto (pypdf)...")
    ner_raw_entities, ner_raw_ms = extract_ner(raw_text, "pypdf bruto")

    # ── 4. Docling → Markdown → NER ───────────────────────────────────────
    print("\n[4/4] Convertendo via Docling e executando NER sobre Markdown...")
    md_text = convert_via_docling(PDF_PATH)
    ner_md_entities = []
    ner_md_ms = 0.0
    if md_text:
        ner_md_entities, ner_md_ms = extract_ner(md_text, "Docling Markdown")
    else:
        print("  ✗ Pulando NER sobre Markdown (Docling falhou).")

    # ── Relatório ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print("  RELATÓRIO COMPARATIVO FINAL")
    print(f"{'═'*60}")

    print_section("① REGEX (padrões estruturados)", regex_entities, key_type="regex")
    print_section("② NER — texto bruto pypdf",       ner_raw_entities, key_type="ner")
    if ner_md_entities:
        print_section("③ NER — Markdown Docling", ner_md_entities, key_type="ner")

        delta = ner_delta(ner_raw_entities, ner_md_entities)
        print(f"\n{'─'*60}")
        print(f"  DELTA: pypdf vs Docling")
        print(f"{'─'*60}")
        print(f"  Entidades em comum:       {len(delta['common'])}")
        print(f"  Só no pypdf (perdidas):   {len(delta['only_in_pypdf'])}")
        for g, w in sorted(delta["only_in_pypdf"]):
            print(f"    [-] [{g}] {w}")
        print(f"  Só no Docling (ganhas):   {len(delta['only_in_docling'])}")
        for g, w in sorted(delta["only_in_docling"]):
            print(f"    [+] [{g}] {w}")

    print(f"\n{'═'*60}")
    print("  RESUMO DE PERFORMANCE")
    print(f"{'═'*60}")
    print(f"  pypdf extração          : {pypdf_ms:>8.0f} ms")
    print(f"  Regex                   : {regex_ms:>8.2f} ms")
    print(f"  NER texto bruto (HTTP)  : {ner_raw_ms:>8.0f} ms")
    if ner_md_entities:
        print(f"  NER Markdown Docling    : {ner_md_ms:>8.0f} ms  (inclui latência de rede)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
