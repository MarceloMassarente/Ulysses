import re
import pypdf
import requests
import json
import time
import os

pdf_path = r"C:\Users\marce\Documentos\OB\Dossie\Réplica - MARCELO Casa de Itapira.pdf"
api_url = "http://192.168.1.221:5522/api/v1/extract"

if not os.path.exists(pdf_path):
    print(f"Erro: O arquivo PDF não foi encontrado no caminho: {pdf_path}")
    exit(1)

print("Extraindo texto do PDF...")
try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
except Exception as e:
    print(f"Erro ao ler o arquivo PDF: {e}")
    exit(1)

print(f"Texto extraído com sucesso! Total de caracteres: {len(text)}")

# --- REGEX EXTRACTION ---
print("\nExecutando extração via REGEX...")
start_regex = time.time()

# Padrões comuns para o domínio jurídico brasileiro
patterns = {
    "CPF": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
    "CNPJ": r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b",
    "PROCESSO": r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b",
    "LEGISLACAO (Art/Lei)": r"\b(?:[Aa]rt(?:\.o|igo)?\s+\d+|[Ll]ei\s+\d+(?:\.\d+)*(?:/\d+)?)\b"
}

regex_entities = []
for name, pattern in patterns.items():
    matches = re.findall(pattern, text)
    for match in set(matches):  # Pegando apenas correspondências únicas por tipo
        regex_entities.append({"type": name, "word": match})
        
regex_time = (time.time() - start_regex) * 1000
print(f"Regex concluído em {regex_time:.2f}ms.")

# --- NER EXTRACTION ---
print("\nExecutando extração via LEGAL NER API (INT8 Otimizado)...")
start_ner = time.time()
try:
    # Usando threshold de 0.50 para pegar uma boa quantidade de entidades
    response = requests.post(api_url, json={"text": text, "confidence_threshold": 0.50})
    ner_time = (time.time() - start_ner) * 1000
    if response.status_code == 200:
        ner_data = response.json()
        ner_entities = ner_data.get("entities", [])
        print(f"NER API concluído em {ner_time:.2f}ms (Tempo de processamento puro da rede neural: {ner_data.get('processing_time_ms')}ms).")
    else:
        print(f"Erro na API NER: Status {response.status_code} - {response.text}")
        ner_entities = []
except Exception as e:
    print(f"Falha ao conectar na API NER (ela está rodando?): {e}")
    ner_entities = []

# --- REPORT ---
print("\n" + "="*60)
print(" RELATÓRIO COMPARATIVO: REGEX vs LEGAL NER")
print("="*60)

print("\n--- [1] ENTIDADES ENCONTRADAS POR REGEX ---")
if regex_entities:
    for ent in sorted(regex_entities, key=lambda x: x['type']):
        print(f" - [{ent['type']}] -> {ent['word']}")
else:
    print(" Nenhuma entidade encontrada via Regex.")

print("\n--- [2] ENTIDADES ENCONTRADAS POR LEGAL NER ---")
if ner_entities:
    # Agrupar entidades para evitar repetição no relatório e remover subwords artifacts
    unique_ner = {}
    for ent in ner_entities:
        word = ent['word'].replace("##", "").strip()
        if len(word) <= 1:
            continue
        key = (ent['entity_group'], word)
        if key not in unique_ner or ent['score'] > unique_ner[key]:
            unique_ner[key] = ent['score']
            
    sorted_ner = sorted(unique_ner.items(), key=lambda x: x[0][0])
    for (group, word), score in sorted_ner:
        print(f" - [{group}] -> {word} (Confiança: {score:.2f})")
else:
    print(" Nenhuma entidade encontrada via NER.")
    
print("\n" + "="*60)
print(" RESUMO DE PERFORMANCE:")
print(f" - Tempo Regex: {regex_time:.2f} ms")
print(f" - Tempo NER (Rede + Chamada HTTP): {ner_time:.2f} ms")
if ner_entities:
    print(f" - Tempo Inferência Puramente na CPU: {ner_data.get('processing_time_ms')} ms")
print("="*60)
