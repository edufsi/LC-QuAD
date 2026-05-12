import time
from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_URL = "http://dbpedia.org/sparql"
ARQUIVO_SAIDA = "todas_as_relacoes_dbpedia_completo.txt"

def baixar_todas_as_relacoes_paginado():
    print("Iniciando o download paginado do vocabulário da DBpedia...")
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.addCustomHttpHeader("User-Agent", "GenericKGQA_Research_Bot/2.0")
    
    todas_relacoes = set()
    limite = 10000
    offset = 0
    continuar = True
    
    inicio = time.time()
    
    while continuar:
        print(f"Buscando do registro {offset} até {offset + limite}...")
        
        # A query agora usa LIMIT e OFFSET para fatiar o banco de dados
        query = f"""
        SELECT DISTINCT ?p WHERE {{
          {{ ?p a rdf:Property }}
          UNION
          {{ ?p a owl:ObjectProperty }}
          UNION
          {{ ?p a owl:DatatypeProperty }}
          FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/") || 
                 STRSTARTS(STR(?p), "http://dbpedia.org/property/"))
        }}
        ORDER BY ?p
        LIMIT {limite}
        OFFSET {offset}
        """
        
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        try:
            resultados = sparql.query().convert()
            bindings = resultados["results"]["bindings"]
            
            # Se a DBpedia devolver menos que o limite, significa que chegamos ao fim
            if len(bindings) == 0:
                continuar = False
                break
                
            for binding in bindings:
                uri = binding["p"]["value"]
                uri_limpa = uri.replace("http://dbpedia.org/ontology/", "dbo:")
                uri_limpa = uri_limpa.replace("http://dbpedia.org/property/", "dbp:")
                todas_relacoes.add(uri_limpa)
            
            offset += limite
            # Pausa de 1 segundo para não sermos bloqueados pelo servidor por spam
            time.sleep(1)
            
        except Exception as e:
            print(f"Erro ao consultar a DBpedia no offset {offset}: {e}")
            continuar = False

    # Salvando ordenado
    relacoes_ordenadas = sorted(list(todas_relacoes))
    
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        for r in relacoes_ordenadas:
            f.write(f"{r}\n")
            
    fim = time.time()
    print(f"\nSucesso absoluto! Extração concluída em {fim - inicio:.2f} segundos.")
    print(f"Total VERDADEIRO de relações salvas: {len(relacoes_ordenadas)}")
    print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")

baixar_todas_as_relacoes_paginado()