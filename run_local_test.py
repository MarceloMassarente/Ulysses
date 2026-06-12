import json
import subprocess
import time

import requests


def main() -> int:
    print("Starting Uvicorn server...")
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "5522"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url = "http://127.0.0.1:5522/api/v1/extract"
    health_url = "http://127.0.0.1:5522/health"

    payload = {
        "text": (
            "O Tribunal de Contas da União, no Acórdão 1234/2020-Plenário, "
            "determinou à Empresa Brasileira de Infraestrutura que cumpra o "
            "Art. 5º da Lei 8.666/1993, sob pena de multa de R$ 50.000,00."
        ),
        "confidence_threshold": 0.85,
        "include_regex": True,
    }

    headers = {"Content-Type": "application/json"}

    try:
        print("Aguardando o serviço iniciar (healthcheck)...")
        for _ in range(60):
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print("Serviço está pronto!")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        else:
            print("O serviço não respondeu no tempo esperado. Verifique os logs.")
            print("STDOUT:", server_process.stdout.read().decode(errors="replace"))
            print("STDERR:", server_process.stderr.read().decode(errors="replace"))
            return 1

        print("\nEnviando texto para extração de entidades...")
        print(f"Texto: {payload['text']}")

        response = requests.post(url, headers=headers, json=payload, timeout=120)

        print(f"\nStatus Code: {response.status_code}")
        if response.status_code == 200:
            print("Resposta:")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print("Erro:")
            print(response.text)
            return 1
        return 0
    finally:
        print("\nEncerrando o servidor Uvicorn...")
        server_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
