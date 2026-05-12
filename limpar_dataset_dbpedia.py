import json
import time
import requests
from tqdm import tqdm

ARQUIVO_ENTRADA = "test-data.json"
ARQUIVO_SAIDA = "lcquad_test_purgado.json"

print("🧹 Inicializando Sanitizador DBpedia (Modo Turbo)...")

# 1. CRIAR UMA SESSÃO PERSISTENTE (A mágica da velocidade)
sessao = requests.Session()

# 2. DISFARÇAR DE NAVEGADOR (Fura a fila de baixa prioridade)
sessao.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/sparql-results+json'
})

def testar_query_turbo(query_sparql, max_retries=3):
    """
    Testa a query na DBpedia usando a Sessão Persistente do Requests.
    """
    url = "http://dbpedia.org/sparql"
    
    for tentativa in range(max_retries):
        try:
            # Enviamos a query via GET como o navegador faz
            resposta = sessao.get(url, params={'query': query_sparql, 'format': 'json'}, timeout=10)
            
            # Se a DBpedia pedir para ir mais devagar (Erro 429 ou 503)
            if resposta.status_code in [429, 503]:
                time.sleep(2)
                continue
                
            # Se a query tem erro de sintaxe
            if resposta.status_code == 400:
                return False
                
            resultados = resposta.json()
            
            # Caso 1: SELECT
            if "results" in resultados and "bindings" in resultados["results"]:
                return len(resultados["results"]["bindings"]) > 0
            
            # Caso 2: ASK
            elif "boolean" in resultados:
                return True
                
            return False
            
        except requests.exceptions.RequestException:
            # Falha de rede pura, espera e tenta de novo
            time.sleep(1)
            
    return False

def purgar_dataset():
    try:
        with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {ARQUIVO_ENTRADA} não encontrado.")
        return

    print(f"📦 Total de perguntas originais: {len(dados)}")
    dados_sobreviventes = []

    for item in tqdm(dados, desc="Validando Queries"):
        query = item.get("sparql_query", "")
        
        if query:
            # Substituímos a função lenta pela Turbo
            if testar_query_turbo(query):
                dados_sobreviventes.append(item)
                
        # Mantemos uma pausa minúscula (agora de 0.05) só por educação cívica
        time.sleep(0.05) 

    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(dados_sobreviventes, f, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print("✅ PURGA CONCLUÍDA COM SUCESSO!")
    print(f"📉 Sobreviveram: {len(dados_sobreviventes)} de {len(dados)} perguntas.")
    print(f"🔥 Taxa de sobrevivência: {(len(dados_sobreviventes) / len(dados)) * 100:.1f}%")

if __name__ == "__main__":
    purgar_dataset()