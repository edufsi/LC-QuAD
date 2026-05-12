import json
import time
import os
import ollama

# ==========================================
# CONFIGURAÇÕES
# ==========================================
MODELO_OLLAMA = 'llama3.1'
ARQUIVO_TESTE = 'lcquad-uris-test-data.json' # Altere se o nome do seu arquivo for diferente
ARQUIVO_SAIDA = 'resultados_baseline_zeroshot_puro.json'

def executar_zeroshot_puro(pergunta):
    """Executa a pergunta no Ollama sem nenhum contexto externo (Zero-Shot puro)."""
    prompt = f"""
    You are an expert DBpedia Knowledge Graph reasoning system.
    Your task is to map the user's question to the correct semantic path using valid DBpedia URIs.
    
    Rules:
    - Use 'dbr:' prefix for entities (e.g., dbr:George_Washington).
    - Use 'dbo:' or 'dbp:' prefix for relations (e.g., dbo:birthPlace).
    - You must infer the correct entities and relations strictly from your internal knowledge of DBpedia.
    
    User Question: "{pergunta}"
    
    Output format strictly as follows (one per line, NO OTHER TEXT):
    - Single path: Entity -> Relation
    - Multi-hop: Entity -> Relation1 -> Relation2
    - Intersection: 
      Entity1 -> Relation1
      Entity2 -> Relation2
      
    DO NOT output introductory sentences. DO NOT explain your reasoning. DO NOT add any extra text. JUST OUTPUT THE PATHS.
    """
    
    # Inferência no modelo local
    resposta = ollama.generate(model=MODELO_OLLAMA, prompt=prompt)
    return resposta['response'].strip()

# ==========================================
# PIPELINE DE EXECUÇÃO EM LOTE
# ==========================================
if __name__ == "__main__":
    # 1. Carrega os dados de teste
    try:
        with open(ARQUIVO_TESTE, 'r', encoding='utf-8') as f:
            dados_teste = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {ARQUIVO_TESTE} não encontrado. Coloque-o na mesma pasta.")
        exit()

    resultados = []
    ids_processados = set()
    
    # 2. Sistema de Checkpoint (Retoma de onde parou)
    if os.path.exists(ARQUIVO_SAIDA):
        with open(ARQUIVO_SAIDA, 'r', encoding='utf-8') as f:
            try:
                resultados = json.load(f)
                ids_processados = {item['id'] for item in resultados}
                print(f"🔄 Retomando execução. {len(ids_processados)} perguntas já processadas.")
            except json.JSONDecodeError:
                print("⚠️ Aviso: Arquivo de saída anterior corrompido. Iniciando um novo.")

    print(f"\n🚀 Iniciando Baseline Zero-Shot Puro com {MODELO_OLLAMA}...")
    print("Pressione Ctrl+C a qualquer momento para parar (o progresso é salvo a cada pergunta).\n")

    # 3. Loop de Execução
    try:
        for item in dados_teste:
            q_id = item['id']
            pergunta = item['question']
            
            # Pula as perguntas já respondidas
            if q_id in ids_processados:
                continue
                
            print(f"📝 Pergunta [{q_id}]: {pergunta}")
            
            inicio = time.time()
            resposta_llm = executar_zeroshot_puro(pergunta)
            fim = time.time()
            
            print(f"✅ Resposta: {resposta_llm}")
            print(f"⏱️ Tempo: {fim - inicio:.2f}s\n")
            print("-" * 50)
            
            # Monta o objeto com o gabarito para futura avaliação
            registro = {
                'id': q_id,
                'question': pergunta,
                'baseline_answer': resposta_llm,
                'target_uris': item.get('target_uris', [])
            }
            resultados.append(registro)
            
            # Salva no disco imediatamente após cada inferência
            with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=4, ensure_ascii=False)
                
    except KeyboardInterrupt:
        print("\n⏸️ Execução interrompida pelo usuário. Progresso salvo de forma segura.")

    if len(resultados) == len(dados_teste):
        print(f"\n🎉 Execução do Baseline concluída! Resultados salvos em: {ARQUIVO_SAIDA}")