import json
import csv
import os
import re

from recuperador_entidades import RecuperadorEntidades
from recuperador_relacoes_faiss import RecuperadorRelacoesFAISS

recuperador = RecuperadorEntidades("lcquad-uris-unified.json")
recuperador_relacoes = RecuperadorRelacoesFAISS("vocabulario_universal_dbpedia.txt")

def limpar_uri(uri):
    """Substitui URLs gigantes pelos prefixos curtos (dbr, dbo, dbp)."""
    uri = uri.strip('<>') 
    uri = uri.replace("http://dbpedia.org/resource/", "dbr:")
    uri = uri.replace("http://dbpedia.org/ontology/", "dbo:")
    uri = uri.replace("http://dbpedia.org/property/", "dbp:")
    uri = uri.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "rdf:type")
    uri = uri.replace("http://www.w3.org/2000/01/rdf-schema#type", "rdfs:type")
    return uri


def processar_query_para_t5(query_sparql):
    """Normaliza variáveis e lineariza a query, lidando com SELECT, COUNT e ASK."""
    if not query_sparql:
        return ""

    var_resposta = None
    is_ask = False
    is_count = False

    # 1. Identificar o TIPO de Query (ASK, COUNT ou SELECT normal)
    if re.search(r'\bASK\b', query_sparql, re.IGNORECASE):
        is_ask = True
    else:
        # Regex blindado para pegar SELECT ?uri, SELECT DISTINCT ?uri e SELECT COUNT(?uri)
        match_select = re.search(r'SELECT\s+(?:DISTINCT\s+)?(?:COUNT\s*\(\s*)?(\?[a-zA-Z0-9_]+)', query_sparql, re.IGNORECASE)
        if match_select:
            var_resposta = match_select.group(1)
            # Verifica se tinha a palavra COUNT perto do SELECT
            if re.search(r'COUNT\s*\(', query_sparql, re.IGNORECASE):
                is_count = True
        else:
            return "" # Se não for nenhum formato conhecido, pula (evita lixo)

    # 2. Normalização de Variáveis (Só acontece se não for ASK, pois ASK geralmente não tem var)
    query_normalizada = query_sparql
    if var_resposta:
        todas_variaveis = re.findall(r'\?[a-zA-Z0-9_]+', query_sparql)
        variaveis_unicas = list(dict.fromkeys(todas_variaveis))
        
        if(len(variaveis_unicas) > 2):
            print(f"  -> Variáveis encontradas: {variaveis_unicas}")
        mapa_variaveis = {}
        contador_var = 1
        
        for var in variaveis_unicas:
            if var == var_resposta:
                mapa_variaveis[var] = "[ANS]"
            else:
                mapa_variaveis[var] = f"[VAR{contador_var}]"
                contador_var += 1
                
        for var_original, var_nova in mapa_variaveis.items():
            query_normalizada = re.sub(rf'\{var_original}\b', var_nova, query_normalizada)

    # 3. Extração dos Triplos do bloco WHERE
    match_where = re.search(r'WHERE\s*{(.*?)}', query_normalizada, re.IGNORECASE | re.DOTALL)
    if not match_where:
        return ""
    
    where_clause = match_where.group(1).strip()
    
    # Captura triplos, lidando com pontuações extras que o LC-QuAD às vezes coloca
    padrao_triplo = r'(<[^>]+>|\[[A-Z0-9]+\])\s+(<[^>]+>|\[[A-Z0-9]+\])\s+(<[^>]+>|\[[A-Z0-9]+\])'
    triplos = re.findall(padrao_triplo, where_clause)
    
    caminhos = []
    for s, p, o in triplos:
        caminho = f"{limpar_uri(s)} >> {limpar_uri(p)} >> {limpar_uri(o)}"
        caminhos.append(caminho)
        
    # 4. Ordenação Alfabética e Montagem Final
    if not caminhos:
        return ""
        
    caminhos.sort()
    target_text = " [SEP] ".join(caminhos)
    
    # Adiciona o Prefixo de Intenção para ensinar ao T5 o tipo da query
    if is_ask:
        target_text = f"ASK >> {target_text}"
    elif is_count:
        target_text = f"COUNT >> {target_text}"
        
    return target_text


def preparar_dataset_t5(arquivo_json_entrada, arquivo_csv_saida):
    print(f"Lendo dados de: {arquivo_json_entrada}...")
    
    try:
        with open(arquivo_json_entrada, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erro: O arquivo {arquivo_json_entrada} não foi encontrado.")
        return

    registros_validos = []
    
    for item in dados:
        pergunta = item.get('question', item.get('corrected_question', '')).strip()
        query_bruta = item.get('sparql_query', '').strip()
        
        if pergunta and query_bruta:
            
            # 1. O ORÁCULO TRABALHA (Usando a pergunta em inglês!)
            entidades = recuperador.extrair_entidades(pergunta)

            if not entidades:
                print(f"⚠️ Aviso: Oráculo retornou None para a pergunta: '{pergunta}'. Substituindo por lista vazia.")

            str_entidades = ", ".join(entidades) if entidades else "None"
            

            # 2. Pega Relações Candidatas (RAG Vetorial - Top 5)
            #relacoes = recuperador_relacoes.buscar_top_k(pergunta, k=5)
            
            #if not relacoes:
            #    print(f"⚠️ Aviso: Recuperador de Relações retornou None para a pergunta: '{pergunta}'. Substituindo por lista vazia.")
            #    relacoes = []

            #str_relacoes = ", ".join(relacoes)


            # 2. O INPUT DO T5 (A Dica)
            # É AQUI que a mágica acontece. O modelo vai LER as entidades exatas.
            input_text = f"translate question to dbpedia path: {pergunta} | Entities possibly involved: {str_entidades}"
            
            #input_text = f"translate question to dbpedia path | Question: {pergunta} | Entities possibly involved: {str_entidades}"
            # 3. O TARGET DO T5 (O Caminho Lógico Puro)
            target_text = processar_query_para_t5(query_bruta)
            
            if target_text:
                registros_validos.append([input_text, target_text])

            else:
                print(f"⚠️ Aviso: Query SPARQL vazia ou mal formatada para a pergunta: '{pergunta}'")
                print(f"   Query original: '{query_bruta}'")


    print(f"Salvando {len(registros_validos)} exemplos no formato T5...")
    
    with open(arquivo_csv_saida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input_text', 'target_text']) 
        writer.writerows(registros_validos)
        
    print(f"✅ Arquivo {arquivo_csv_saida} gerado com sucesso!\n")

if __name__ == "__main__":
    # Caminhos dos arquivos JSON originais
    arquivo_treino = "train-data.json" 
    arquivo_teste = "test-data.json"

    # Gera o CSV de Treino
    if os.path.exists(arquivo_treino):
        preparar_dataset_t5(arquivo_treino, "t5_dataset_train.csv")
    else:
        print(f"⚠️ Arquivo de treino ({arquivo_treino}) não encontrado.")

    # Gera o CSV de Teste
    if os.path.exists(arquivo_teste):
        preparar_dataset_t5(arquivo_teste, "t5_dataset_test.csv")
    else:
        print(f"⚠️ Arquivo de teste ({arquivo_teste}) não encontrado.")
    
    # Amostra do resultado final
    if os.path.exists("t5_dataset_test.csv"):
        print("👀 Amostra do Dataset Linearizado (Target com Lógica de Triplos):")
        with open("t5_dataset_test.csv", 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            for i, linha in enumerate(reader):
                if i < 3:
                    print(f"  Input : {linha[0]}")
                    print(f"  Target: {linha[1]}\n")