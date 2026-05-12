import re

def limpar_uri(uri):
    uri = uri.strip('<>') 
    uri = uri.replace("http://dbpedia.org/resource/", "dbr:")
    uri = uri.replace("http://dbpedia.org/ontology/", "dbo:")
    uri = uri.replace("http://dbpedia.org/property/", "dbp:")
    uri = uri.replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "rdf:type")
    uri = uri.replace("http://www.w3.org/2000/01/rdf-schema#type", "rdfs:type")
    return uri

def processar_query_para_t5(query_sparql):
    """Normaliza as variáveis e extrai os caminhos linearizados alfabeticamente."""
    
    # 1. Normalização de Variáveis
    match_select = re.search(r'SELECT\s+(?:DISTINCT\s+)?(\?[a-zA-Z0-9_]+)', query_sparql, re.IGNORECASE)
    if not match_select:
        return ""
        
    var_resposta = match_select.group(1)
    todas_variaveis = re.findall(r'\?[a-zA-Z0-9_]+', query_sparql)
    variaveis_unicas = list(dict.fromkeys(todas_variaveis))
    
    mapa_variaveis = {}
    contador_var = 1
    
    for var in variaveis_unicas:
        if var == var_resposta:
            mapa_variaveis[var] = "[ANS]"
        else:
            mapa_variaveis[var] = f"[VAR{contador_var}]"
            contador_var += 1
            
    query_normalizada = query_sparql
    for var_original, var_nova in mapa_variaveis.items():
        # Substitui garantindo que é a palavra inteira
        query_normalizada = re.sub(rf'\{var_original}\b', var_nova, query_normalizada)

    # 2. Extração dos Triplos
    match_where = re.search(r'WHERE\s*{(.*?)}', query_normalizada, re.IGNORECASE | re.DOTALL)
    if not match_where:
        return ""
    
    where_clause = match_where.group(1).strip()
    
    # Captura: (URI ou [VAR]) ...
    padrao_triplo = r'(<[^>]+>|\[[A-Z0-9]+\])\s+(<[^>]+>|\[[A-Z0-9]+\])\s+(<[^>]+>|\[[A-Z0-9]+\])'
    triplos = re.findall(padrao_triplo, where_clause)
    
    caminhos = []
    for s, p, o in triplos:
        caminho = f"{limpar_uri(s)} >> {limpar_uri(p)} >> {limpar_uri(o)}"
        caminhos.append(caminho)
        
    # 3. Ordenação Alfabética e Linearização
    caminhos.sort()
    return " [SEP] ".join(caminhos)

# === TESTE ===
query_complexa = " SELECT DISTINCT ?uri WHERE { <http://dbpedia.org/resource/Nolan> <http://dbpedia.org/ontology/child> ?x . ?x <http://dbpedia.org/ontology/birthPlace> ?uri } "
print("Query Original:", query_complexa.strip())
print("Target do T5  :", processar_query_para_t5(query_complexa))
# Resultado esperado: [VAR1] >> dbo:birthPlace >> [ANS] [SEP] dbr:Nolan >> dbo:child >> [VAR1]