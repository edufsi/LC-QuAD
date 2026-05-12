import json
import re
import numpy as np
import requests
import logging
from time import sleep
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIGURAÇÃO DO LOG
# ==========================================
logging.basicConfig(
    filename='log_correcoes_diagnostico.txt',
    level=logging.INFO,
    format='%(message)s',
    encoding='utf-8'
)

def extrair_triplas(caminho_str):
    """Função auxiliar para quebrar strings em triplas."""
    triplas = []
    partes = caminho_str.split("[SEP]")
    for parte in partes:
        elementos = [e.strip() for e in parte.split(">>") if e.strip()]
        if len(elementos) >= 2:
            triplas.append(tuple(elementos))
    return triplas

class CorretorRelacoes:
    def __init__(self, limite_similaridade=0.6):
        print("🧠 Inicializando o motor semântico (SentenceTransformers)...")
        self.modelo = SentenceTransformer('all-MiniLM-L6-v2')
        self.limite_similaridade = limite_similaridade
        self.cache_entidades = {} 
        
        self.sessao = requests.Session()
        self.sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
            'Accept': 'application/sparql-results+json'
        })

    def _limpar_uri(self, uri):
        nome = uri.split(':')[-1]
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', nome).lower()

    def _puxar_propriedades_dbpedia(self, entidade, max_retries=2):
        if entidade in self.cache_entidades:
            return self.cache_entidades[entidade]

        uri = entidade.replace("dbr:", "http://dbpedia.org/resource/")
        
        query = f"""
        SELECT DISTINCT ?p WHERE {{
            {{ <{uri}> ?p ?o }} UNION {{ ?s ?p <{uri}> }}
            FILTER (STRSTARTS(str(?p), "http://dbpedia.org/ontology/") || STRSTARTS(str(?p), "http://dbpedia.org/property/"))
        }}
        """
        url = "http://dbpedia.org/sparql"
        
        for tentativa in range(max_retries):
            try:
                resposta = self.sessao.get(url, params={'query': query, 'format': 'json'}, timeout=4)
                if resposta.status_code == 200:
                    resultados = resposta.json()
                    props = []
                    for r in resultados.get("results", {}).get("bindings", []):
                        p = r["p"]["value"]
                        if "ontology" in p: props.append(p.replace("http://dbpedia.org/ontology/", "dbo:"))
                        if "property" in p: props.append(p.replace("http://dbpedia.org/property/", "dbp:"))
                    
                    self.cache_entidades[entidade] = props
                    return props
                sleep(1)
            except Exception:
                sleep(1)
                
        logging.warning(f"[FALHA DBpedia] Não foi possível consultar a entidade: {entidade}")
        self.cache_entidades[entidade] = []
        return []

    def _extrair_relacoes_alvo_da_entidade(self, gt_str, entidade_foco):
        """Descobre no gabarito qual era a relação específica ligada a esta entidade."""
        triplas_gt = extrair_triplas(gt_str)
        relacoes = []
        for t in triplas_gt:
            if len(t) >= 3:
                # Se a entidade é o sujeito ou objeto da tripla do gabarito, guarda a relação
                if t[0] == entidade_foco or t[2] == entidade_foco:
                    relacoes.append(t[1])
        # Retorna lista sem duplicatas, preservando a ordem
        return list(dict.fromkeys(relacoes))

    def corrigir(self, tripla, gt_str, id_pergunta="Desconhecido"):
        if len(tripla) < 3: return tripla
        
        e1, rel, e2 = tripla[0], tripla[1], tripla[2]
        resto_malformado = tripla[3:] 
        
        if not (rel.startswith("dbo:") or rel.startswith("dbp:")):
            return tripla

        entidade_foco = e1 if e1.startswith("dbr:") else e2 if e2.startswith("dbr:") else None
        if not entidade_foco: return tripla

        propriedades_reais = self._puxar_propriedades_dbpedia(entidade_foco)
        
        if not propriedades_reais: 
            return tripla
            
        if rel in propriedades_reais: 
            return tripla

        # --- DIAGNÓSTICO DE TEMPORAL DRIFT (Focado na Entidade) ---
        relacoes_esperadas = self._extrair_relacoes_alvo_da_entidade(gt_str, entidade_foco)
        esperadas_sobreviventes = [r for r in relacoes_esperadas if r in propriedades_reais]

        # --- Matemática Semântica ---
        palavra_alucinada = self._limpar_uri(rel)
        palavras_reais = [self._limpar_uri(p) for p in propriedades_reais]

        emb_alucinada = self.modelo.encode(palavra_alucinada)
        emb_reais = self.modelo.encode(palavras_reais)

        norma_a = np.linalg.norm(emb_alucinada)
        normas_r = np.linalg.norm(emb_reais, axis=1)
        normas_r[normas_r == 0] = 1e-10 
        
        similaridades = np.dot(emb_reais, emb_alucinada) / (normas_r * norma_a)
        
        # Pega os Top 3
        n_top = min(3, len(similaridades))
        indices_top = np.argsort(similaridades)[-n_top:][::-1]
        indice_vencedor = indices_top[0]
        
        top_info = ", ".join([f"{propriedades_reais[i]} ({similaridades[i]:.2f})" for i in indices_top])
        
        # O corretor decidiu fazer uma troca?
        if similaridades[indice_vencedor] >= self.limite_similaridade:
            relacao_nova = propriedades_reais[indice_vencedor]
            
            # Avalia o SUCESSO vs TEMPORAL DRIFT da correção
            if relacao_nova in relacoes_esperadas:
                status_temporal = f"🎉 CORREÇÃO PERFEITA! (O corretor acertou exatamente a relação do gabarito: {relacao_nova})"
            elif len(esperadas_sobreviventes) > 0:
                status_temporal = f"⚠️ ALVO PERDIDO! (A DBpedia tem {esperadas_sobreviventes}, mas o corretor escolheu {relacao_nova})"
            else:
                status_temporal = f"⏳ TEMPORAL DRIFT! (O gabarito exigia {relacoes_esperadas}, que não existem mais. O corretor salvou com {relacao_nova})"

            log_msg = (
                f"\n[CORRIGIDO] Pergunta ID: {id_pergunta} | Entidade-Foco: {entidade_foco}\n"
                f"  ❌ Alucinado: {rel}\n"
                f"  🎯 Gabarito p/ Entidade: {relacoes_esperadas}\n"
                f"  {status_temporal}\n"
                f"  🔍 Top 3 opções avaliadas: [{top_info}]"
            )
            logging.info(log_msg)
            
            return [e1, relacao_nova, e2] + resto_malformado
            
        else:
            # O corretor não atingiu o limite e rejeitou a troca
            if len(relacoes_esperadas) == 0:
                status_temporal = f"❓ ALUCINAÇÃO SEVERA (A entidade {entidade_foco} nem sequer devia estar no grafo gerado!)"
            elif len(esperadas_sobreviventes) > 0:
                status_temporal = f"⚠️ ALVO PERDIDO! (A DBpedia possui {esperadas_sobreviventes}, mas a similaridade com '{rel}' foi baixa demais para corrigir)"
            else:
                status_temporal = f"⏳ TEMPORAL DRIFT PURO! (O gabarito exigia {relacoes_esperadas}, que não existem mais na DBpedia)"

            log_msg = (
                f"\n[REJEITADO] Pergunta ID: {id_pergunta} | Entidade-Foco: {entidade_foco}\n"
                f"  ❌ Alucinado: {rel}\n"
                f"  🎯 Gabarito p/ Entidade: {relacoes_esperadas}\n"
                f"  {status_temporal}\n"
                f"  ⛔ Melhor Opção Rejeitada: {propriedades_reais[indice_vencedor]} (Score: {similaridades[indice_vencedor]:.2f} - ABAIXO DO LIMITE {self.limite_similaridade})\n"
                f"  🔍 Top 3 opções avaliadas: [{top_info}]"
            )
            logging.info(log_msg)
            
        return tripla


def processar_arquivo(arquivo_entrada, arquivo_saida):
    print(f"📂 Lendo resultados de: {arquivo_entrada}")
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    corretor = CorretorRelacoes(limite_similaridade=0.6)
    estatisticas = {"avaliados": 0, "corrigidos": 0}

    open('log_correcoes_diagnostico.txt', 'w', encoding='utf-8').close()
    logging.info(f"=== INÍCIO DO LOG DE CORREÇÕES (Threshold: {corretor.limite_similaridade}) ===\n")

    print("🚀 Iniciando correção de triplas...")
    
    for item in tqdm(dados, desc="Corrigindo Predições"):
        caminho_str = item.get('prediction', '')
        gt_str = item.get('ground_truth', '')
        id_pergunta = item.get('id', 'N/A')
        
        if not caminho_str: continue

        partes = caminho_str.split("[SEP]")
        caminho_corrigido = []
        
        for parte in partes:
            elementos = [e.strip() for e in parte.split(">>") if e.strip()]
            
            if len(elementos) >= 3:
                estatisticas["avaliados"] += 1
                # Agora passamos a string do ground_truth inteira para o corretor investigar localmente
                nova_tripla = corretor.corrigir(elementos, gt_str, id_pergunta)
                
                if nova_tripla[1] != elementos[1]:
                    estatisticas["corrigidos"] += 1
                
                caminho_corrigido.append(" >> ".join(nova_tripla))
            else:
                caminho_corrigido.append(" >> ".join(elementos))
                
        item['prediction'] = " [SEP] ".join(caminho_corrigido)

    print("\n" + "="*50)
    print("✅ PROCESSO DE CORREÇÃO CONCLUÍDO!")
    print(f"🔍 Triplas avaliadas: {estatisticas['avaliados']}")
    print(f"🛠️ Triplas corrigidas: {estatisticas['corrigidos']}")
    print(f"📄 Log detalhado salvo em: log_correcoes_diagnostico.txt")
    print(f"💾 Resultados salvos em: {arquivo_saida}")
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    processar_arquivo("resultados_modelo_sem_rag_sem_trie.json", "resultados_corrigidos_modelo_sem_rag_sem_trie_com_log.json")