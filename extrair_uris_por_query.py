import json
import re

def extrair_uris_por_query(caminho_entrada, caminho_saida):
    # Regex para capturar qualquer URL que comece com http dentro de < >
    padrao_uri = re.compile(r'<(http[^>]+)>')
    
    # Lista que vai armazenar o nosso novo dataset processado
    dataset_processado = []
    
    # 1. Abre o JSON original
    with open(caminho_entrada, 'r', encoding='utf-8') as f:
        dados_lcquad = json.load(f)
        
    print(f"Processando {len(dados_lcquad)} perguntas do dataset...")
    
    # 2. Varre cada item do dataset
    for item in dados_lcquad:
        id_pergunta = item.get('_id')
        pergunta = item.get('corrected_question')
        query_sparql = item.get('sparql_query', '')
        
        # 3. Extrai as URIs da string do SPARQL
        uris_encontradas = padrao_uri.findall(query_sparql)
        
        # Opcional: Remover duplicatas caso a mesma URI apareça duas vezes na query
        # Mantendo a ordem original (se a ordem importar, use dict.fromkeys)
        uris_unicas = list(dict.fromkeys(uris_encontradas))
        
        # 4. Monta o novo objeto isolando o que importa
        novo_item = {
            "id": id_pergunta,
            "question": pergunta,
            "target_uris": uris_unicas
        }
        
        dataset_processado.append(novo_item)
        
    # 5. Salva o resultado em um novo arquivo JSON
    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(dataset_processado, f, indent=4, ensure_ascii=False)
        
    print(f"Extração concluída! Arquivo salvo em: {caminho_saida}")

# ======== COMO USAR ========
# Substitua pelos nomes reais dos seus arquivos
ARQUIVO_ENTRADA = 'lcquad_train_purgado.json'
ARQUIVO_SAIDA = 'lcquad-uris-train-purgado.json'

extrair_uris_por_query(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)