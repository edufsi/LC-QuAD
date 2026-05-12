import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON

import faiss
import ollama
from sentence_transformers import SentenceTransformer
import re

class RecuperadorGrafoDBpedia:
    def __init__(self):
        print("🧠 Inicializando Graph-RAG (SPARQL + Embeddings)...")
        self.sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        self.sparql.setReturnFormat(JSON)
        # O modelo de embedding para fazer o filtro
        self.modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

    def obter_propriedades_reais(self, entidade_dbr):
        """Faz uma query SPARQL para pegar todas as propriedades de/para a entidade."""
        # Converte dbr:New_Sanno_Hotel para a URI completa
        uri = entidade_dbr.replace("dbr:", "http://dbpedia.org/resource/")
        uri = uri.replace("dbo:", "http://dbpedia.org/ontology/")
        
        # Pega propriedades onde a entidade é o Sujeito (Saída) OU o Objeto (Entrada)
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            {{ <{uri}> ?p ?o }}
            UNION
            {{ ?s ?p <{uri}> }}
            # Filtra lixo administrativo da Wikipedia
            FILTER (!regex(str(?p), "wikiPage|prov#|owl#|rdf-schema#"))
        }}
        """
        
        self.sparql.setQuery(query)
        try:
            print(query)
            resultados = self.sparql.query().convert()
            propriedades = []
            for resultado in resultados["results"]["bindings"]:
                prop = resultado["p"]["value"]
                # Mantém apenas propriedades dbo e dbp
                if "dbpedia.org/ontology/" in prop:
                    propriedades.append(prop.replace("http://dbpedia.org/ontology/", "dbo:"))
                elif "dbpedia.org/property/" in prop:
                    propriedades.append(prop.replace("http://dbpedia.org/property/", "dbp:"))
                    
            # Remove duplicatas
            return list(set(propriedades))
        except Exception as e:
            print(f"⚠️ Erro no SPARQL para {entidade_dbr}: {e}")
            return []

    def limpar_nome_propriedade(self, relacao):
        """dbo:navalArchitect -> naval architect"""
        nome_propriedade = relacao.split(':')[-1]
        nome_limpo = re.sub(r'([a-z])([A-Z])', r'\1 \2', nome_propriedade).lower()
        return nome_limpo.replace('_', ' ')

    def filtrar_top_k(self, pergunta, entidades_dbr, k=10):
        """Executa a sua ideia: Puxa do SPARQL e filtra por relevância vetorial."""
        todas_propriedades_reais = []
        
        
        # 1. Busca as propriedades exatas de todas as entidades
        for entidade in entidades_dbr:
            props = self.obter_propriedades_reais(entidade)
            todas_propriedades_reais.extend(props)
            
        todas_propriedades_reais = list(set(todas_propriedades_reais)) # Remove duplicatas
        
        if not todas_propriedades_reais:
            return []

        print(todas_propriedades_reais)

        # 2. O FILTRO: Calcula similaridade semântica só para este pequeno subconjunto
        textos_props = [self.limpar_nome_propriedade(p) for p in todas_propriedades_reais]
        
        emb_pergunta = self.modelo_embedding.encode(pergunta)
        emb_props = self.modelo_embedding.encode(textos_props)
        
        # Matemática de cosseno simples usando numpy (sem FAISS, pois o N é muito pequeno)
        norma_pergunta = np.linalg.norm(emb_pergunta)
        normas_props = np.linalg.norm(emb_props, axis=1)
        
        # Evita divisão por zero
        normas_props[normas_props == 0] = 1e-10 
        
        similaridades = np.dot(emb_props, emb_pergunta) / (normas_props * norma_pergunta)
        
        # Pega os índices dos K maiores scores
        top_indices = np.argsort(similaridades)[::-1][:k]
        
        # Retorna o Top K
        return [todas_propriedades_reais[i] for i in top_indices]

# === TESTE DO SEU PIPELINE ===
if __name__ == "__main__":
    grafo_rag = RecuperadorGrafoDBpedia()
    
    pergunta = "Which architect of Marine Corps Air Station Kaneohe Bay was also tenant of New Sanno hotel?"
    entidades = ["dbr:Marine_Corps_Air_Station_Kaneohe_Bay", "dbr:New_Sanno_Hotel"]
    
    print(f"\n📝 Pergunta: {pergunta}")
    print(f"🎯 Entidades: {entidades}")
    print("⏳ Consultando DBpedia SPARQL Endpoint...")
    
    # Executa a sua lógica
    top_relacoes = grafo_rag.filtrar_top_k(pergunta, entidades, k=10)
    
    print("\n✅ Top 10 Relações do RAG Ancorado no Grafo:")
    for r in top_relacoes:
        print(f" - {r}")