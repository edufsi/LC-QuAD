import json
import re
from datetime import datetime

def extrair_triplas(caminho_str):
    """Transforma o texto linear 'A >> p >> B [SEP] ...' numa lista de tuplas."""
    triplas = []
    partes = caminho_str.split("[SEP]")
    for parte in partes:
        elementos = [e.strip() for e in parte.split(">>") if e.strip()]
        if len(elementos) >= 2:
            triplas.append(tuple(elementos))
    return triplas

def extrair_todas_uris(triplas):
    """Pega todas as URIs (começam com dbr, dbo, dbp) ignorando variáveis"""
    uris = set()
    for t in triplas:
        for item in t:
            if ":" in item and not item.startswith("["): 
                uris.add(item)
    return uris

def extrair_esqueleto(caminho):
    """Substitui todas as URIs por <URI> mantendo a estrutura lógica."""
    esqueleto = re.sub(r'(dbr:|dbo:|dbp:|rdf:)[^\s]+', '<URI>', caminho)
    return esqueleto.strip()

def normalizar_ordem(caminho_str):
    """
    Torna a avaliação imune à ordem dos [SEP].
    Pega 'B [SEP] A', quebra, ordena alfabeticamente e retorna 'A [SEP] B'.
    """
    partes = [p.strip() for p in caminho_str.split("[SEP]") if p.strip()]
    partes.sort()
    return " [SEP] ".join(partes)

def normalizar_direcao_str(caminho_str):
    """
    Torna a avaliação imune à DIREÇÃO INTERNA da tripla.
    (A >> p >> B) vira igual a (B >> p >> A).
    Também aplica a imunidade à ordem dos [SEP].
    """
    triplas = extrair_triplas(caminho_str)
    partes_normalizadas = []
    
    for t in triplas:
        if len(t) >= 3:
            # Ordena alfabeticamente apenas o sujeito (t[0]) e o objeto (t[2])
            sujeito, objeto = sorted([t[0], t[2]])
            relacao = t[1]
            resto = t[3:] # Caso o modelo tenha gerado lixo a mais
            
            nova_t = [sujeito, relacao, objeto] + list(resto)
            partes_normalizadas.append(" >> ".join(nova_t))
        else:
            # Se a tripla estiver quebrada (só A >> p), mantém como está
            partes_normalizadas.append(" >> ".join(t))
            
    # Ordena as triplas para também ser imune à ordem dos [SEP]
    partes_normalizadas.sort()
    return " [SEP] ".join(partes_normalizadas)

def remover_tipos(caminho_str):
    """Remove qualquer tripla que declare tipos (rdf:type) para focar apenas na lógica de saltos."""
    partes = caminho_str.split("[SEP]")
    partes_filtradas = []
    for parte in partes:
        if "rdf:type" not in parte:
            partes_filtradas.append(parte.strip())
    return " [SEP] ".join(partes_filtradas)


def avaliar(arquivo_resultados, arquivo_saida, titulo_relatorio, descricao_modelo):
    with open(arquivo_resultados, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    total_precision = 0
    total_recall = 0
    total_alucinacoes_abs = 0
    total_exemplos = len(dados)
    
    exact_matches_estrito = 0
    exact_matches_direcao = 0
    
    skeleton_matches_estrito = 0
    skeleton_matches_direcao = 0
    
    exact_match_agnostico = 0
    skeleton_match_agnostico = 0

    # Variável para armazenar o log detalhado por pergunta
    log_detalhado = "\n\n==================================================\n"
    log_detalhado += "📋 DETALHAMENTO POR PERGUNTA E CLASSIFICAÇÃO\n"
    log_detalhado += "==================================================\n"

    for item in dados:
        gt_triplas = set(extrair_triplas(item['ground_truth']))
        pred_triplas = set(extrair_triplas(item['prediction']))
        

        # 1. Filtra os tipos para perdoar o modelo por ser "esperto"
        gt_sem_tipo = remover_tipos(item['ground_truth'])
        pred_sem_tipo = remover_tipos(item['prediction'])
        
        # 2. Agora calcula o Skeleton usando as strings limpas
        gabarito_esq_sem_tipo = normalizar_direcao_str(extrair_esqueleto(gt_sem_tipo))
        predicao_esq_sem_tipo = normalizar_direcao_str(extrair_esqueleto(pred_sem_tipo))
        
        is_skeleton_agnostico = (gabarito_esq_sem_tipo == predicao_esq_sem_tipo)

        if is_skeleton_agnostico:
            skeleton_match_agnostico += 1
        
        is_exact_agnostico = (gt_sem_tipo == pred_sem_tipo)

        if is_exact_agnostico:
            exact_match_agnostico += 1

        # -------------------------------------------------------------
        # 1. CÁLCULO DAS MÉTRICAS BOLEANAS
        # -------------------------------------------------------------
        gt_norm = normalizar_ordem(item['ground_truth'])
        pred_norm = normalizar_ordem(item['prediction'])
        is_exact_estrito = (gt_norm == pred_norm)
        if is_exact_estrito: exact_matches_estrito += 1

        gabarito_esq = normalizar_ordem(extrair_esqueleto(item['ground_truth']))
        predicao_esq = normalizar_ordem(extrair_esqueleto(item['prediction']))
        is_skeleton_estrito = (gabarito_esq == predicao_esq)
        if is_skeleton_estrito: skeleton_matches_estrito += 1

        gt_dir = normalizar_direcao_str(item['ground_truth'])
        pred_dir = normalizar_direcao_str(item['prediction'])
        is_exact_direcao = (gt_dir == pred_dir)
        if is_exact_direcao: exact_matches_direcao += 1
            
        gt_esq_dir = normalizar_direcao_str(extrair_esqueleto(item['ground_truth']))
        pred_esq_dir = normalizar_direcao_str(extrair_esqueleto(item['prediction']))
        is_skeleton_direcao = (gt_esq_dir == pred_esq_dir)
        if is_skeleton_direcao: skeleton_matches_direcao += 1

        # -------------------------------------------------------------
        # 2. DEFININDO A "MELHOR CLASSE" ATINGIDA (Hierarquia)
        # -------------------------------------------------------------
        if is_exact_estrito:
            melhor_classe = "🎯 Classe 1: Exact Match Estrito (Acertou Lógica e URIs perfeitamente)"
        elif is_skeleton_estrito:
            melhor_classe = "🧩 Classe 2: Skeleton Match Estrito (Acertou a Lógica, mas errou URIs/Vocabulário)"
        elif is_exact_direcao:
            melhor_classe = "🔄 Classe 3: Exact Match Flexível (Acertou Lógica e URIs, mas INVERTEU a direção/seta)"
        elif is_skeleton_direcao:
            melhor_classe = "🦴 Classe 4: Skeleton Match Flexível (Acertou a Lógica, mas errou URIs e INVERTEU a direção)"
        elif is_exact_agnostico:
            melhor_classe = "⚠️ Classe 5: Exact Match Agnóstico (Acertou a Lógica ignorando tipos)"
        elif is_skeleton_agnostico:
            melhor_classe = "⚠️ Classe 6: Skeleton Match Agnóstico (Acertou a Lógica mesmo ignorando tipos e URIs)"
        else:
            melhor_classe = "❌ Classe 5: Erro (Falha severa na estrutura lógica do grafo)"

        # -------------------------------------------------------------
        # 3. MÉTRICAS DE TRIPLAS (Precisão, Recall e Alucinações)
        # -------------------------------------------------------------
        acertos = pred_triplas.intersection(gt_triplas)
        precision = len(acertos) / len(pred_triplas) if len(pred_triplas) > 0 else 0
        recall = len(acertos) / len(gt_triplas) if len(gt_triplas) > 0 else 0
        total_precision += precision
        total_recall += recall

        uris_gt = extrair_todas_uris(gt_triplas)
        uris_pred = extrair_todas_uris(pred_triplas)
        inventadas = uris_pred - uris_gt
        total_alucinacoes_abs += len(inventadas)
        
        # -------------------------------------------------------------
        # 4. REGISTRO NO LOG DETALHADO
        # -------------------------------------------------------------
        pergunta_texto = item.get('input_prompt', item.get('question', 'N/A'))
        pergunta_texto = pergunta_texto.replace("translate question to dbpedia path: Question: ", "")
        
        log_detalhado += f"\n🔹 ID: {item.get('id', 'N/A')}\n"
        log_detalhado += f"📝 Pergunta: {pergunta_texto}\n"
        log_detalhado += f"🎯 Esperado: {item['ground_truth']}\n"
        log_detalhado += f"🤖 Obtido  : {item['prediction']}\n"
        log_detalhado += f"🏆 Diagnóstico: {melhor_classe}\n"
        log_detalhado += "-" * 70

    # --- MONTANDO O RELATÓRIO RESUMO ---
    relatorio_resumo = f"""==================================================
📊 RELATÓRIO DE MÉTRICAS DO GRAFO
==================================================
TÍTULO: {titulo_relatorio}
DATA: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
--------------------------------------------------
📝 CARACTERÍSTICAS / CONFIGURAÇÕES:
{descricao_modelo}
--------------------------------------------------
MÉTRICAS ALCANÇADAS (Baseado em {total_exemplos} amostras):

[MÉTRICAS SINTÁTICAS - DIREÇÃO IMPORTA]
🎯 Exact Match Estrito              : {(exact_matches_estrito/total_exemplos)*100:.2f}%
🧩 Skeleton Match Estrito           : {(skeleton_matches_estrito/total_exemplos)*100:.2f}%

[MÉTRICAS SEMÂNTICAS - IMUNE À INVERSÃO (A->B == B->A)]
🔄 Exact Match (Ignorando Direção)  : {(exact_matches_direcao/total_exemplos)*100:.2f}%
🦴 Skeleton Match (Ignorando Direção): {(skeleton_matches_direcao/total_exemplos)*100:.2f}%

[MÉTRICAS AGNÓSTICAS - IGNORANDO TIPOS E URIS]
⚠️ Exact Match Agnóstico            : {(exact_match_agnostico/total_exemplos)*100:.2f}%
⚠️ Skeleton Match Agnóstico         : {(skeleton_match_agnostico/total_exemplos)*100:.2f}%

[MÉTRICAS DE COMPONENTES]
✅ Precisão Média (Triplos)         : {total_precision/total_exemplos:.4f}
📈 Recall Médio (Triplos)           : {total_recall/total_exemplos:.4f}
👻 Média de URIs Alucinadas         : {total_alucinacoes_abs/total_exemplos:.2f}
=================================================="""

    # Imprime apenas o resumo no terminal
    print(relatorio_resumo)

    # Junta o resumo com o log detalhado e salva no arquivo
    relatorio_completo = relatorio_resumo + log_detalhado
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        f.write(relatorio_completo)
    print(f"\n💾 Relatório salvo com sucesso em: {arquivo_saida}")
    print(f"Abra o arquivo para ver o diagnóstico classificado pergunta a pergunta!")

if __name__ == "__main__":
    avaliar(
        arquivo_resultados="resultados_ollama_baseline.json", # Atualize para o nome correto
        arquivo_saida="relatorio_final_classificado_ollama.txt",
        titulo_relatorio="Avaliação Ollama - Categorização de Erros",
        descricao_modelo="Avaliação pós-correção com diagnóstico individual de classes de acerto."
    )