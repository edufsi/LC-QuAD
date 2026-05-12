import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from SPARQLWrapper import SPARQLWrapper, JSON

# Configurações de Arquivos
ARQUIVO_GRAFO_BASE = 'subgrafo_spotlight_0.3_confianca_paralelo_train2.nt' # O grafo de 1.8M que você já tem
ARQUIVO_SAIDA = 'subgrafo_relacoes_extra_subgrafo_spotlight_0.3_confianca_paralelo_train2.nt'       # Onde o reforço será salvo
ARQUIVO_LOG = 'log_relacoes_sucesso_subgrafo_spotlight_0.3_confianca_paralelo_train.txt'           # Diário de bordo
ENDPOINT_URL = "http://dbpedia.org/sparql"

# Limite de exemplos a puxar por relação. 
# 300 é um número excelente para dar "peso" ao TransE sem explodir o tamanho do grafo.
LIMITE_EXEMPLOS = 300 

lock_arquivo = threading.Lock()
lock_log = threading.Lock()

def extrair_predicados_locais(arquivo_nt):
    print(f"Lendo {arquivo_nt} para extrair relações únicas...")
    predicados = set()
    
    with open(arquivo_nt, 'r', encoding='utf-8') as f:
        for linha in f:
            partes = linha.strip().split(' ')
            # No formato N-Triples, a relação é sempre o segundo elemento (índice 1)
            if len(partes) >= 3:
                predicado_bruto = partes[1]
                # Limpa os sinais de < e > para ter a URI limpa
                predicado_uri = predicado_bruto.strip('<>')
                predicados.add(predicado_uri)
                
    print(f"-> Encontradas {len(predicados)} relações únicas no seu grafo base.")
    return predicados

def processar_relacao(uri_relacao):
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.addCustomHttpHeader("User-Agent", "LCQuAD_RelationAugmentation_Bot/1.0")
    
    # Query: "Me dê X exemplos de Sujeito e Objeto que usam essa Relação. Apenas URIs (sem textos/datas)."
    query = f"""
    SELECT ?s ?o WHERE {{
        ?s <{uri_relacao}> ?o .
        FILTER(isIRI(?s) && isIRI(?o))
    }} LIMIT {LIMITE_EXEMPLOS}
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    tentativas = 0
    espera = 1
    
    while tentativas < 5:
        try:
            resultados = sparql.query().convert()
            triplas = []
            
            for binding in resultados["results"]["bindings"]:
                s = binding["s"]["value"]
                o = binding["o"]["value"]
                # Monta a tripla de volta: <Sujeito> <Relação> <Objeto> .
                triplas.append(f"<{s}> <{uri_relacao}> <{o}> .\n")
                
            if triplas:
                with lock_arquivo:
                    with open(ARQUIVO_SAIDA, 'a', encoding='utf-8') as f:
                        f.writelines(triplas)
                        
            with lock_log:
                with open(ARQUIVO_LOG, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"{uri_relacao}\n")
                    
            return True, uri_relacao, len(triplas)
            
        except Exception as e:
            tentativas += 1
            time.sleep(espera)
            espera *= 2 
            
    return False, uri_relacao, 0

def iniciar_aumento_de_dados(max_workers=5):
    # 1. Lê o seu arquivo de 1.8M e descobre quais relações existem lá
    predicados_semente = extrair_predicados_locais(ARQUIVO_GRAFO_BASE)
    
    # 2. Sistema de recuperação (Checkpoint)
    predicados_ja_feitos = set()
    if os.path.exists(ARQUIVO_LOG):
        with open(ARQUIVO_LOG, 'r', encoding='utf-8') as f_log:
            predicados_ja_feitos = set(linha.strip() for linha in f_log)
        print(f"-> Retomando progresso: {len(predicados_ja_feitos)} relações já foram reforçadas anteriormente.")
        
    predicados_pendentes = predicados_semente - predicados_ja_feitos
    total_pendentes = len(predicados_pendentes)
    
    if total_pendentes == 0:
        print("Todas as relações já receberam reforço. Nada a fazer!")
        return

    print(f"\nIniciando Thread Pool com {max_workers} trabalhadores...")
    print(f"Buscando até {LIMITE_EXEMPLOS} exemplos para cada uma das {total_pendentes} relações restantes.\n")
    
    com_erro = []
    triplas_totais = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(processar_relacao, uri): uri for uri in predicados_pendentes}
        
        for i, futuro in enumerate(as_completed(futuros), 1):
            sucesso, uri, qtd_triplas = futuro.result()
            
            if sucesso:
                triplas_totais += qtd_triplas
                print(f"[{i}/{total_pendentes}] OK (+{qtd_triplas} triplas) | Rel: {uri}")
            else:
                com_erro.append(uri)
                print(f"[{i}/{total_pendentes}] FALHOU | Rel: {uri}")
                
            time.sleep(0.1)

    print("\n=== RESUMO DO DATA AUGMENTATION ===")
    print(f"Total de Triplas de Reforço Salvas: {triplas_totais}")
    
    if com_erro:
        with open('relacoes_com_erro.txt', 'w') as f:
            for u in com_erro:
                f.write(f"{u}\n")

# Inicie com 4 workers para respeitar os limites da DBpedia
iniciar_aumento_de_dados(max_workers=6)