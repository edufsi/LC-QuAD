import json
import requests
import time

# Configuração da API pública do DBpedia Spotlight
SPOTLIGHT_URL = "https://api.dbpedia-spotlight.org/en/annotate"

def rodar_spotlight_baseline_e_ruido(arq_lcquad_original, arq_saida):
    print("1. Carregando o LC-QuAD original para ler as perguntas...")
    with open(arq_lcquad_original, 'r', encoding='utf-8') as f:
        dados_originais = json.load(f)

    dataset_spotlight = []
    
    # Exigimos que a API nos responda em JSON limpo
    headers = {'Accept': 'application/json'}

    print(f"\n2. Iniciando extração via DBpedia Spotlight (Total: {len(dados_originais)} perguntas)")
    print("   Isso pode demorar um pouco devido aos limites da API pública...")
    
    for i, item in enumerate(dados_originais, 1):
        pergunta = item.get('corrected_question', '')
        id_pergunta = item.get('_id')
        
        if not pergunta:
            continue
            
        # Confidence 0.3: Ideal para capturar a resposta certa E o ruído realista (os falsos positivos)
        params = {
            'text': pergunta,
            'confidence': 0.3 
        }

        tentativas = 0
        sucesso = False
        uris_encontradas = []

        # Sistema de tolerância a falhas para não perder o progresso se a internet oscilar
        while not sucesso and tentativas < 3:
            try:
                resposta = requests.get(SPOTLIGHT_URL, headers=headers, params=params, timeout=10)
                
                if resposta.status_code == 200:
                    dados_api = resposta.json()
                    
                    if 'Resources' in dados_api:
                        # Extrai as URIs e usa dict.fromkeys para remover duplicatas mantendo a ordem
                        uris_encontradas = list(dict.fromkeys([res['@URI'] for res in dados_api['Resources']]))
                        
                    sucesso = True
                else:
                    raise Exception(f"Erro HTTP {resposta.status_code}")
                    
            except Exception as e:
                tentativas += 1
                time.sleep(2) # Espera 2 segundos antes de tentar de novo

        # Salvamos TUDO que o Spotlight retornou para essa pergunta
        dataset_spotlight.append({
            "id": id_pergunta,
            "question": pergunta,
            "spotlight_uris": uris_encontradas
        })

        # Feedback visual a cada 100 perguntas
        if i % 100 == 0:
            print(f"   [{i}/{len(dados_originais)}] perguntas processadas...")
            
        # Pausa obrigatória de 0.3s para não tomar bloqueio de IP da DBpedia
        time.sleep(0.3) 

    print(f"\n3. Extração concluída com sucesso!")
    with open(arq_saida, 'w', encoding='utf-8') as f:
        json.dump(dataset_spotlight, f, indent=4, ensure_ascii=False)
        
    print(f"   -> Arquivo unificado salvo em: {arq_saida}")

# ======== COMO USAR ========
# 1. O JSON gigante original do LC-QuAD que tem as perguntas
ARQUIVO_ORIGINAL = 'train-data.json' 
# 2. O arquivo de saída com as predições do Spotlight
ARQUIVO_SAIDA = 'lcquad_spotlight_confiante_completo_0.3_train.json'

rodar_spotlight_baseline_e_ruido(ARQUIVO_ORIGINAL, ARQUIVO_SAIDA)