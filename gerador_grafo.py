import json
import time
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_URL = "http://dbpedia.org/sparql"
ARQUIVO_ENTRADA = 'lcquad_spotlight_confiante_completo_0.3_train.json'
ARQUIVO_SAIDA = 'subgrafo_spotlight_0.3_confianca_paralelo_train3.nt'
ARQUIVO_LOG = 'log_spotlight_0.3_train_sucesso.txt' # O seu diário de bordo

# Cadeados para garantir que duas threads não escrevam nos arquivos ao mesmo tempo
lock_arquivo = threading.Lock()
lock_log = threading.Lock()

def processar_uri(uri):
    # Cada thread precisa do seu próprio objeto SPARQLWrapper para evitar conflitos
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.addCustomHttpHeader("User-Agent", "LCQuAD_Research_Bot_Threaded/1.0")
    
    query = f"""
    SELECT ?s ?p ?o WHERE {{
        {{ BIND(<{uri}> AS ?s) . <{uri}> ?p ?o . FILTER(isIRI(?o)) }}
        UNION
        {{ BIND(<{uri}> AS ?o) . ?s ?p <{uri}> . FILTER(isIRI(?s)) }}
    }} LIMIT 1000
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    tentativas = 0
    espera = 1 # Tempo inicial de espera em caso de erro (Backoff)
    
    while tentativas < 5:
        try:
            resultados = sparql.query().convert()
            triplas = []
            
            for binding in resultados["results"]["bindings"]:
                s = binding["s"]["value"]
                p = binding["p"]["value"]
                o = binding["o"]["value"]
                triplas.append(f"<{s}> <{p}> <{o}> .\n")
                
            # Se encontrou triplas, salva no arquivo .nt com o cadeado
            if triplas:
                with lock_arquivo:
                    with open(ARQUIVO_SAIDA, 'a', encoding='utf-8') as f:
                        f.writelines(triplas)
            
            # --- NOVIDADE: ANOTA NO LOG DE SUCESSO ---
            # Salva a URI no log mesmo se não tiver triplas (evita repeti-la no futuro)
            with lock_log:
                with open(ARQUIVO_LOG, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"{uri}\n")
                        
            return True, uri, len(triplas)
            
        except Exception as e:
            tentativas += 1
            # Backoff Exponencial: se o servidor negar, espera 2s, depois 4s, 8s...
            time.sleep(espera)
            espera *= 2 
            
    return False, uri, 0

def iniciar_extracao_paralela(max_workers=5):
    print("Carregando URIs semente...")
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
        dados = json.load(f)
        
    uris_semente = set()
    # Verifica a estrutura do JSON. Se a chave for diferente (ex: spotlight_uris), altere aqui
    chave_uris = 'target_uris' if 'target_uris' in dados[0] else 'spotlight_uris'
    for item in dados:
        uris_semente.update(item[chave_uris])
        
    # --- NOVIDADE: VERIFICA O LOG E PULA O QUE JÁ FOI FEITO ---
    uris_ja_processadas = set()
    if os.path.exists(ARQUIVO_LOG):
        with open(ARQUIVO_LOG, 'r', encoding='utf-8') as f_log:
            uris_ja_processadas = set(linha.strip() for linha in f_log)
        print(f"-> Diário de Bordo encontrado! Pulando {len(uris_ja_processadas)} URIs já finalizadas.")

    # Subtrai as processadas do total
    uris_pendentes = uris_semente - uris_ja_processadas
    total_uris = len(uris_pendentes)
    
    if total_uris == 0:
        print("\nTodas as URIs já foram processadas! Nenhum trabalho pendente.")
        return

    print(f"Total de URIs restantes para extrair: {total_uris}")
    print(f"Iniciando Thread Pool com {max_workers} trabalhadores simultâneos...")
    
    uris_com_erro = []
    triplas_totais = 0
    
    # ThreadPoolExecutor gerencia as filas de requisição usando apenas as PENDENTES
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(processar_uri, uri): uri for uri in uris_pendentes}
        
        # as_completed vai liberando o print assim que uma thread termina
        for i, futuro in enumerate(as_completed(futuros), 1):
            sucesso, uri, qtd_triplas = futuro.result()
            
            if sucesso:
                triplas_totais += qtd_triplas
                print(f"[{i}/{total_uris}] OK (+{qtd_triplas} triplas) | URI: {uri}")
            else:
                uris_com_erro.append(uri)
                print(f"[{i}/{total_uris}] FALHOU | URI: {uri}")
                
            # Pausa milimétrica de segurança a cada conclusão
            time.sleep(0.1)

    print("\n=== RESUMO DA SESSÃO DE EXTRAÇÃO ===")
    print(f"Total de Triplas Salvas nesta sessão: {triplas_totais}")
    print(f"Total de URIs que falharam (Timeout/Ban): {len(uris_com_erro)}")
    
    if uris_com_erro:
        with open('uris_pendentes_erro.txt', 'w') as f:
            for u in uris_com_erro:
                f.write(f"{u}\n")
        print("As URIs que falharam foram salvas em 'uris_pendentes_erro.txt' para você tentar depois.")

# Inicie com 4 ou 5 workers. Se perceber muitos erros de timeout, baixe para 3.
iniciar_extracao_paralela(max_workers=6)