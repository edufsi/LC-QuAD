import time
import string
from SPARQLWrapper import SPARQLWrapper, JSON

ENDPOINT_URL = "http://dbpedia.org/sparql"
ARQUIVO_SAIDA = "vocabulario_universal_dbpedia.txt"

def baixar_vocabulario_universal():
    print("Iniciando o download do vocabulário genérico (Divisão Alfabética)...")
    sparql = SPARQLWrapper(ENDPOINT_URL)
    sparql.addCustomHttpHeader("User-Agent", "GenericKGQA_Research_Bot/4.0")
    
    todas_relacoes = set()
    inicio = time.time()

    # PASSO 1: Pegar a Ontologia Curada (dbo:) inteira de uma vez (são só ~3000, não trava)
    print("Baixando Ontologia Curada (dbo:)...")
    query_dbo = """
    SELECT DISTINCT ?p WHERE {
      ?p a rdf:Property .
      FILTER(STRSTARTS(STR(?p), "http://dbpedia.org/ontology/"))
    }
    """
    sparql.setQuery(query_dbo)
    sparql.setReturnFormat(JSON)
    try:
        res = sparql.query().convert()
        for binding in res["results"]["bindings"]:
            todas_relacoes.add(binding["p"]["value"].replace("http://dbpedia.org/ontology/", "dbo:"))
    except Exception as e:
        print(f"Erro no dbo: {e}")

    # PASSO 2: Pegar o restante universal e o caos das propriedades (dbp:) particionado por letra
    letras = list(string.ascii_lowercase) # ['a', 'b', 'c', ..., 'z']
    
    print("Baixando Propriedades Brutas (dbp:) e outras, particionadas por letra...")
    for letra in letras:
        print(f"  -> Buscando relações que começam com a letra '{letra}'...")
        
        # A sacada: Buscar relações cujo NOME local comece com a letra específica
        # Usamos LCASE para garantir que pegamos A e a
        query_alfabetica = f"""
        SELECT DISTINCT ?p WHERE {{
          {{ ?p a rdf:Property }} UNION {{ ?p a owl:ObjectProperty }} UNION {{ ?p a owl:DatatypeProperty }}
          
          # Filtra tudo que NÃO é dbo (pois já pegamos) e que o nome comece com a letra atual
          FILTER(!STRSTARTS(STR(?p), "http://dbpedia.org/ontology/"))
          
          # Pega a parte final da URI (depois da última barra ou hashtag)
          BIND(REPLACE(STR(?p), "^.*[/#]", "") AS ?nomeLocal)
          FILTER(STRSTARTS(LCASE(?nomeLocal), "{letra}"))
        }}
        """
        
        sparql.setQuery(query_alfabetica)
        
        try:
            res = sparql.query().convert()
            for binding in res["results"]["bindings"]:
                uri = binding["p"]["value"]
                # Limpeza padrão
                uri = uri.replace("http://dbpedia.org/property/", "dbp:")
                uri = uri.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "rdf:type")
                uri = uri.replace("http://xmlns.com/foaf/0.1/", "foaf:")
                todas_relacoes.add(uri)
                
            time.sleep(0.5) # Pausa amigável para o servidor
            
        except Exception as e:
            print(f"Erro na letra {letra}: {e}")

    # Salva o arquivo final
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        for relacao in sorted(todas_relacoes):
            f.write(f"{relacao}\n")

    fim = time.time()
    print(f"\nExtração Genérica Concluída em {fim - inicio:.2f} segundos!")
    print(f"Total de Relações Universais encontradas: {len(todas_relacoes)}")
    print(f"O vocabulário universal foi salvo em: {ARQUIVO_SAIDA}")

baixar_vocabulario_universal()