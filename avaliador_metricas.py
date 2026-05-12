import json
import re
import os
from datetime import datetime

class DBpediaEvaluator:
    def __init__(self, titulo, descricao):
        self.titulo = titulo
        self.descricao = descricao
        self.metricas = {
            "exact_match": 0,
            "skeleton_match": 0,
            "precision_triples": 0,
            "recall_triples": 0,
            "hallucinations": 0
        }

    def _normalizar_caminho(self, caminho):
        """Ordena as triplas alfabeticamente para ignorar a ordem do [SEP]."""
        partes = [p.strip() for p in caminho.split("[SEP]") if p.strip()]
        partes.sort()
        return " [SEP] ".join(partes)

    def _extrair_esqueleto(self, caminho):
        """Remove URIs mantendo apenas a estrutura lógica e variáveis."""
        esqueleto = re.sub(r'(dbr:|dbo:|dbp:|rdf:)[^\s]+', '<URI>', caminho)
        return self._normalizar_caminho(esqueleto)

    def _get_triplas(self, caminho):
        triplas = []
        for p in caminho.split("[SEP]"):
            el = [e.strip() for e in p.split(">>") if e.strip()]
            if len(el) >= 2: triplas.append(tuple(el))
        return set(triplas)

    def avaliar(self, arquivo_json, output_txt):
        with open(arquivo_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        n = len(dados)
        total_p, total_r, total_h = 0, 0, 0
        em, sm = 0, 0

        for item in dados:
            gt = item['ground_truth']
            pred = item['prediction']

            # Exact Match (imune a ordem)
            if self._normalizar_caminho(gt) == self._normalizar_caminho(pred):
                em += 1
            
            # Skeleton Match (imune a ordem e URIs)
            if self._extrair_esqueleto(gt) == self._extrair_esqueleto(pred):
                sm += 1

            # Precisão/Recall de Triplas
            gt_t = self._get_triplas(gt)
            pred_t = self._get_triplas(pred)
            acertos = pred_t.intersection(gt_t)
            
            total_p += len(acertos) / len(pred_t) if pred_t else 0
            total_r += len(acertos) / len(gt_t) if gt_t else 0

            # Alucinações (URIs na pred que não estão no GT)
            uris_gt = {u for t in gt_t for u in t if ":" in u and not u.startswith("[")}
            uris_pred = {u for t in pred_t for u in t if ":" in u and not u.startswith("[")}
            total_h += len(uris_pred - uris_gt)

        # Formatação do Relatório
        relatorio = f"""
==================================================
📊 RELATÓRIO DE AVALIAÇÃO: {self.titulo}
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================

DESCRITIVO DO EXPERIMENTO:
{self.descricao}

MÉTRICAS (Amostras: {n}):
--------------------------------------------------
🎯 Exact Match (Lógica + URI): {(em/n)*100:.2f}%
🧩 Skeleton Match (Lógica Pura): {(sm/n)*100:.2f}%
✅ Precisão Média de Triplas: {total_p/n:.4f}
📈 Recall Médio de Triplas: {total_r/n:.4f}
👻 Média de URIs Alucinadas: {total_h/n:.2f}
==================================================
"""
        print(relatorio)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(relatorio)

if __name__ == "__main__":
    evaluator = DBpediaEvaluator(
        titulo="T5-Small vs DBpedia (Sem RAG)",
        descricao="Modelo treinado com 4k exemplos. Injeção de entidades via Oráculo.\n"
                  "Dataset original (não purgado) para manter densidade gramatical.\n"
                  "Modelo sem acesso a RAG de relações, sem Trie limitando o vocabulário de saída, sem corretor de saída"
    )
    evaluator.avaliar("resultados_corrigidos_modelo_sem_rag_sem_trie.json", "relatorio_final_modelo_sem_rag_sem_trie_corrigido.txt")