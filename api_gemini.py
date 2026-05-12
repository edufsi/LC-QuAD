import requests
import google.generativeai as genai
import os
import json
import time
from google.api_core.exceptions import ResourceExhausted

# 1. Configuração do Gemini API
CHAVE_API = "AIzaSyAqVAk1oLyb1nzf3US0uaH4Feo4uPzzlOk" # COLOQUE A SUA CHAVE AQUI
genai.configure(api_key=CHAVE_API)

# Usamos o Flash porque é gratuito, ultrarrápido e tem janela de 1M tokens
modelo = genai.GenerativeModel('gemini-2.5-flash')

# 2. Configuração do Spotlight
URL_SPOTLIGHT = "https://api.dbpedia-spotlight.org/en/annotate"
CONFIANCA_SPOTLIGHT = 0.6 # Alta confiança para evitar lixo

def extrair_entidades_spotlight(pergunta):
    """Usa o DBpedia Spotlight para achar as URIs das entidades na pergunta."""
    print("🔍 Consultando Spotlight...")
    headers = {'Accept': 'application/json'}
    params = {'text': pergunta, 'confidence': CONFIANCA_SPOTLIGHT}
    
    try:
        resposta = requests.get(URL_SPOTLIGHT, headers=headers, params=params)
        dados = resposta.json()
        
        entidades = []
        if 'Resources' in dados:
            for recurso in dados['Resources']:
                uri = recurso['@URI'].replace("http://dbpedia.org/resource/", "dbr:")
                entidades.append(uri)
        return list(set(entidades))
    except Exception as e:
        print(f"Erro no Spotlight: {e}")
        return []

def carregar_vocabulario():
    """Carrega as 65.000 relações do arquivo de texto."""
    arquivo = "vocabulario_universal_dbpedia.txt"
    try:
        with open(arquivo, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"⚠️ Aviso: Arquivo {arquivo} não encontrado!")
        return "dbo:architect\ndbo:director\ndbo:birthPlace\n"

def executar_baseline_llm(pergunta, vocabulario_relacoes):
    print(f"\n📝 Pergunta: {pergunta}")
    
    entidades = extrair_entidades_spotlight(pergunta)
    print(f"🎯 Entidades encontradas: {entidades}")
    
    prompt = f"""
    Você é um sistema de Question Answering especializado na DBpedia.
    O seu objetivo é mapear a pergunta do usuário para as URIs corretas do Grafo de Conhecimento.
    
    O nosso sistema de extração sugeriu as seguintes entidades candidatas que PODEM estar presentes na pergunta:
    {entidades}
    
    Abaixo está a lista EXAUSTIVA de todas as propriedades/relações que existem na DBpedia. 
    Você deve obrigatoriamente escolher a(s) relação(ões) a partir desta lista (não invente nada que não esteja aqui):
    
    [LISTA DE RELAÇÕES DA DBPEDIA]
    {vocabulario_relacoes}
    [/FIM DA LISTA]
    
    Pergunta do usuário: "{pergunta}"
    
    Atenção: A pergunta pode ser simples (1 entidade) ou complexa (múltiplas entidades, interseções ou caminhos multi-hop).
    Avalie criticamente as entidades sugeridas e a lista de relações para montar a estrutura lógica correta.
    
    Responda APENAS com as sequências lógicas, utilizando um dos seguintes formatos (um por linha):
    - Caminho Simples: Entidade -> Relacao
    - Caminho Multi-hop: Entidade -> Relacao1 -> Relacao2
    - Interseção (múltiplas entidades): 
      Entidade1 -> Relacao1
      Entidade2 -> Relacao2
      
    Não inclua nenhuma outra palavra, saudação ou explicação. Responda apenas com a estrutura em formato de texto cru.
    """
    
    print("🧠 Enviando os tokens para o Gemini...")
    resposta = modelo.generate_content(prompt)
    resultado_texto = resposta.text.strip()
    
    print(f"✅ Resposta do Baseline: {resultado_texto}")
    return resultado_texto

# --- PIPELINE DE EXECUÇÃO EM LOTE ---
if __name__ == "__main__":
    ARQUIVO_TESTE = 'lcquad-uris-test-data.json' # Altere se o nome for diferente
    ARQUIVO_SAIDA = 'baseline_resultados_zeroshot.json'
    
    # Carrega o vocabulário uma única vez para poupar I/O do disco
    print("Carregando vocabulário gigante na memória...")
    vocabulario = carregar_vocabulario()
    
    # Carrega os dados de teste
    try:
        with open(ARQUIVO_TESTE, 'r', encoding='utf-8') as f:
            dados_teste = json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo {ARQUIVO_TESTE} não encontrado. Coloque o dataset na mesma pasta.")
        exit()

    # Sistema de Checkpoint (Retoma de onde parou)
    resultados = []
    ids_processados = set()
    
    if os.path.exists(ARQUIVO_SAIDA):
        with open(ARQUIVO_SAIDA, 'r', encoding='utf-8') as f:
            try:
                resultados = json.load(f)
                ids_processados = {item['id'] for item in resultados}
                print(f"🔄 Retomando execução. {len(ids_processados)} perguntas já processadas.")
            except json.JSONDecodeError:
                print("Aviso: Arquivo de saída corrompido, iniciando um novo.")

    # Loop de Execução
    for item in dados_teste:
        q_id = item['id']
        pergunta = item['question']
        
        # Ignora se já respondeu numa execução anterior
        if q_id in ids_processados:
            continue
            
        sucesso = False
        while not sucesso:
            try:
                resposta_llm = executar_baseline_llm(pergunta, vocabulario)
                
                # Monta o objeto com a pergunta, resposta do modelo e a resposta esperada
                registro = {
                    'id': q_id,
                    'question': pergunta,
                    'baseline_answer': resposta_llm,
                    'target_uris': item.get('target_uris', []) # O gabarito para avaliarmos depois
                }
                resultados.append(registro)
                
                # Salva no disco imediatamente
                with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
                    json.dump(resultados, f, indent=4, ensure_ascii=False)
                
                sucesso = True
                
                # Pausa amigável para não sobrecarregar
                time.sleep(1) 
                
            except ResourceExhausted as e:
                # Imprime o erro exato. Se falar "per day", o limite diário foi atingido.
                print(f"⏳ Cota estourada! Mensagem do Google: {e}")
                print("Aguardando 60 segundos...")
                time.sleep(60)
            except Exception as e:
                print(f"❌ Erro inesperado na pergunta {q_id}: {e}")
                # Em caso de erro bizarro (ex: quebra de internet), salva o erro e continua
                resultados.append({
                    'id': q_id,
                    'question': pergunta,
                    'baseline_answer': f"ERRO: {e}",
                    'target_uris': item.get('target_uris', [])
                })
                with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
                    json.dump(resultados, f, indent=4, ensure_ascii=False)
                sucesso = True

    print("\n🎉 Execução do Baseline concluída com sucesso!")
    print(f"Resultados salvos em: {ARQUIVO_SAIDA}")