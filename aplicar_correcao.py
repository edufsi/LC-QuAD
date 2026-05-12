import json
import re
import numpy as np
import requests
from time import sleep
from tqdm import tqdm
import faiss
import ollama
from sentence_transformers import SentenceTransformer

class CorretorRelacoes:
    def __init__(self, limite_similaridade=0.6):
        print("🧠 Inicializando o motor semântico (SentenceTransformers)...")
        self.modelo = SentenceTransformer('all-MiniLM-L6-v2')
        self.limite_similaridade = limite_similaridade
        self.cache_entidades = {} # Cache para evitar queries repetidas
        
        self.sessao = requests.Session()
        self.sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
            'Accept': 'application/sparql-results+json'
        })

    def _limpar_uri(self, uri):
        """'dbo:birthPlace' -> 'birth place'"""
        nome = uri.split(':')[-1]
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', nome).lower()

    def _puxar_propriedades_dbpedia(self, entidade, max_retries=3):
        """Vai na DBpedia e puxa as propriedades válidas (Versão Otimizada)."""
        if entidade in self.cache_entidades:
            return self.cache_entidades[entidade]

        uri = entidade.replace("dbr:", "http://dbpedia.org/resource/")
        
        # OTIMIZAÇÃO SPARQL: Substituímos o lentíssimo REGEX por STRSTARTS
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            {{ <{uri}> ?p ?o }} UNION {{ ?s ?p <{uri}> }}
            FILTER (STRSTARTS(str(?p), "http://dbpedia.org/ontology/") || STRSTARTS(str(?p), "http://dbpedia.org/property/"))
        }}
        """
        
        url = "http://dbpedia.org/sparql"
        
        for tentativa in range(max_retries):
            try:
                resposta = self.sessao.get(url, params={'query': query, 'format': 'json'}, timeout=5)
                if resposta.status_code == 200:
                    resultados = resposta.json()
                    props = []
                    for r in resultados.get("results", {}).get("bindings", []):
                        p = r["p"]["value"]
                        if "ontology" in p: props.append(p.replace("http://dbpedia.org/ontology/", "dbo:"))
                        if "property" in p: props.append(p.replace("http://dbpedia.org/property/", "dbp:"))
                    
                    self.cache_entidades[entidade] = props
                    return props
                
                # Se a DBpedia reclamar, dormimos
                sleep(1)
            except Exception:
                sleep(1)
                
        # Cacheia falhas também para não perder tempo tentando de novo
        self.cache_entidades[entidade] = []
        return []

    def corrigir(self, tripla):
        """Recebe uma tripla (A, p, B) e tenta corrigir o 'p'."""
        # Se for muito curta, não faz nada
        if len(tripla) < 3: return tripla
        
        # CORREÇÃO DO ERRO: Evita o unpack de triplas malformadas (com mais de 3 itens)
        e1 = tripla[0]
        rel = tripla[1]
        e2 = tripla[2]
        resto_malformado = tripla[3:] # Guarda qualquer lixo extra que o modelo gerou
        
        # Só tentamos corrigir se for uma relação dbo: ou dbp:
        if not (rel.startswith("dbo:") or rel.startswith("dbp:")):
            return tripla

        # Descobre quem é a "Entidade Foco" (a que começa com dbr:)
        entidade_foco = None
        if e1.startswith("dbr:"): entidade_foco = e1
        elif e2.startswith("dbr:"): entidade_foco = e2
        
        # Se não tiver entidade, devolve como está
        if not entidade_foco: return tripla

        propriedades_reais = self._puxar_propriedades_dbpedia(entidade_foco)
        
        if not propriedades_reais: return tripla
        if rel in propriedades_reais: return tripla

        # Matemática Semântica
        palavra_alucinada = self._limpar_uri(rel)
        palavras_reais = [self._limpar_uri(p) for p in propriedades_reais]

        emb_alucinada = self.modelo.encode(palavra_alucinada)
        emb_reais = self.modelo.encode(palavras_reais)

        norma_a = np.linalg.norm(emb_alucinada)
        normas_r = np.linalg.norm(emb_reais, axis=1)
        normas_r[normas_r == 0] = 1e-10 
        
        similaridades = np.dot(emb_reais, emb_alucinada) / (normas_r * norma_a)
        indice_vencedor = np.argmax(similaridades)
        
        # Verifica se o threshold foi atingido
        if similaridades[indice_vencedor] >= self.limite_similaridade:
            relacao_nova = propriedades_reais[indice_vencedor]
            # Devolve a tripla corrigida preservando lixo extra, se houver
            return [e1, relacao_nova, e2] + resto_malformado
            
        return tripla


def processar_arquivo(arquivo_entrada, arquivo_saida):
    print(f"📂 Lendo resultados de: {arquivo_entrada}")
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    corretor = CorretorRelacoes(limite_similaridade=0.6)
    estatisticas = {"avaliados": 0, "corrigidos": 0}

    print("🚀 Iniciando correção de triplas (Modo Turbo SPARQL)...")
    
    for item in tqdm(dados, desc="Corrigindo Predições"):
        caminho_str = item.get('prediction', '')
        
        if not caminho_str: continue

        partes = caminho_str.split("[SEP]")
        caminho_corrigido = []
        
        for parte in partes:
            elementos = [e.strip() for e in parte.split(">>") if e.strip()]
            
            if len(elementos) >= 3:
                estatisticas["avaliados"] += 1
                nova_tripla = corretor.corrigir(elementos)
                
                if nova_tripla[1] != elementos[1]:
                    estatisticas["corrigidos"] += 1
                
                caminho_corrigido.append(" >> ".join(nova_tripla))
            else:
                caminho_corrigido.append(" >> ".join(elementos))
                
        item['prediction'] = " [SEP] ".join(caminho_corrigido)
        # Removemos o sleep artificial aqui do laço principal! 
        # Ele só vai esperar se der erro lá na requisição HTTP.

    print("\n" + "="*50)
    print("✅ PROCESSO DE CORREÇÃO CONCLUÍDO!")
    print(f"🔍 Triplas avaliadas: {estatisticas['avaliados']}")
    print(f"🛠️ Triplas corrigidas: {estatisticas['corrigidos']}")
    print(f"💾 Salvando em: {arquivo_saida}")
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    processar_arquivo("resultados_modelo_sem_rag_sem_trie.json", "resultados_corrigidos_modelo_sem_rag_sem_trie.json")