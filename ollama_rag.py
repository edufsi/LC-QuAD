import os
import json
import faiss
import requests
import time
import numpy as np
import ollama  # NOVA BIBLIOTECA

print("A")
from sentence_transformers import SentenceTransformer
print("B")
# ==========================================
# CONFIGURAÇÕES
# ==========================================
MODELO_OLLAMA = 'llama3.1' # Pode testar 'mistral' ou 'qwen2.5' depois
URL_SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"
CONFIANCA_SPOTLIGHT = 0.6

FICHEIRO_VOCABULARIO = "vocabulario_universal_dbpedia.txt"
FICHEIRO_INDICE_FAISS = "indice_relacoes.faiss"
FICHEIRO_LISTA_RELACOES = "lista_relacoes.json"

# ==========================================
# MÓDULO 1: INDEXAÇÃO VETORIAL (OFFLINE) - IGUAL
# ==========================================
print("Carregando o modelo de embeddings (MiniLM)...")
# O SentenceTransformer usará automaticamente o CUDA (RTX 4060) se o PyTorch estiver configurado para GPU

modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

def preparar_banco_vetorial():
    if os.path.exists(FICHEIRO_INDICE_FAISS) and os.path.exists(FICHEIRO_LISTA_RELACOES):
        print("⚡ Índice FAISS encontrado no disco. Carregando em memória...")
        indice = faiss.read_index(FICHEIRO_INDICE_FAISS)
        with open(FICHEIRO_LISTA_RELACOES, 'r', encoding='utf-8') as f:
            relacoes = json.load(f)
        return indice, relacoes

    print("⚙️ Índice FAISS não encontrado. Criando agora...")
    try:
        with open(FICHEIRO_VOCABULARIO, 'r', encoding='utf-8') as f:
            relacoes = [linha.strip() for linha in f if linha.strip()]
    except FileNotFoundError:
        print(f"Erro: O ficheiro {FICHEIRO_VOCABULARIO} não foi encontrado.")
        exit()

    print(f"Gerando embeddings para {len(relacoes)} relações...")
    vetores = modelo_embedding.encode(relacoes, show_progress_bar=True)
    vetores_float32 = np.array(vetores).astype('float32')
    dimensao = vetores_float32.shape[1]
    indice = faiss.IndexFlatL2(dimensao)
    indice.add(vetores_float32)
    faiss.write_index(indice, FICHEIRO_INDICE_FAISS)
    with open(FICHEIRO_LISTA_RELACOES, 'w', encoding='utf-8') as f:
        json.dump(relacoes, f)
        
    print("✅ Índice FAISS criado com sucesso!")
    return indice, relacoes

# ==========================================
# MÓDULO 2: RECUPERAÇÃO E RAG (ONLINE) COM OLLAMA
# ==========================================
def recuperar_top_k_relacoes(pergunta, indice, relacoes, k=15):
    vetor_pergunta = modelo_embedding.encode([pergunta])
    vetor_pergunta = np.array(vetor_pergunta).astype('float32')
    distancias, indices_recuperados = indice.search(vetor_pergunta, k)
    return [relacoes[idx] for idx in indices_recuperados[0]]

def extrair_entidades_spotlight(pergunta):
    headers = {'Accept': 'application/json'}
    params = {'text': pergunta, 'confidence': CONFIANCA_SPOTLIGHT}
    try:
        resposta = requests.get(URL_SPOTLIGHT, headers=headers, params=params, timeout=10)
        dados = resposta.json()
        entidades = []
        if 'Resources' in dados:
            for recurso in dados['Resources']:
                uri = recurso['@URI'].replace("http://dbpedia.org/resource/", "dbr:")
                entidades.append(uri)
        return list(set(entidades))
    except Exception as e:
        print(f"Aviso do Spotlight: {e}")
        return []

def executar_pipeline_rag_ollama(pergunta, indice, relacoes_completas):
    print(f"\n📝 Pergunta: {pergunta}")
    
    entidades = extrair_entidades_spotlight(pergunta)
    top_relacoes = recuperar_top_k_relacoes(pergunta, indice, relacoes_completas, k=15)
    
    print(f"🎯 Entidades: {entidades}")
    print(f"🔎 Relações Lidas: {top_relacoes[:3]}...")
    
    # Modelos Open-Source precisam de prompts mais restritivos ("System Prompts" fortes)
    # para não começarem a "conversar" com o usuário (ex: "Here is your answer:")
    prompt = f"""
    You are an expert DBpedia Knowledge Graph reasoning system.
    Your ONLY task is to map the user's question to the correct semantic path using the provided entities and relations.
    
    Candidate Entities (evaluate if they make sense):
    {entidades}
    
    Allowed DBpedia Relations (pick from this list ONLY):
    {top_relacoes}
    
    User Question: "{pergunta}"
    
    Output format strictly as follows (one per line, NO OTHER TEXT):
    - Single path: Entity -> Relation
    - Multi-hop: Entity -> Relation1 -> Relation2
    - Intersection: 
      Entity1 -> Relation1
      Entity2 -> Relation2
      
    DO NOT output introductory sentences. DO NOT explain your reasoning. JUST OUTPUT THE PATHS.
    """
    
    print(f"🧠 Inferindo no Ollama local ({MODELO_OLLAMA})...")
    # Chamada local! Rápida, privada e gratuita.
    resposta = ollama.generate(model=MODELO_OLLAMA, prompt=prompt)
    resultado = resposta['response'].strip()
    
    print(f"✅ Resposta LLM: \n{resultado}")
    return resultado

# ==========================================
# TESTE
# ==========================================
if __name__ == "__main__":
    indice_faiss, lista_relacoes = preparar_banco_vetorial()
    
    perguntas = [
        "Who is the architect of the New Sanno Hotel?",
        "Where was George Washington born?",
        "Name the common editor of Easy Street (film) and Work (film)?"
    ]
    
    # Repare: Removemos o time.sleep() porque a sua GPU não tem cota! Pode rodar a 100%
    for p in perguntas:
        inicio = time.time()
        executar_pipeline_rag_ollama(p, indice_faiss, lista_relacoes)
        fim = time.time()
        print(f"⏱️ Tempo de resposta (RAG + Ollama): {fim - inicio:.2f} segundos")