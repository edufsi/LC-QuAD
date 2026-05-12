# LC-QuAD (Fork de Experimentos KBQA)

Este projeto implementa um pipeline de **pergunta em linguagem natural → caminho/URIs da DBpedia**.

Pelo histórico em `relatorio.txt`, a linha atual validada é:
- treino do **T5-small** com os ~4k exemplos originais (sem purga),
- uso do **Oráculo de Entidades** no prompt,
- **sem RAG de relações** e **sem Trie** na configuração principal.

## Função de cada script Python

| Script | Função |
|---|---|
| `api_gemini.py` | Corrige relações alucinadas em predições (`dbo:/dbp:`) comparando com propriedades reais da DBpedia e similaridade semântica. |
| `aplicar_correcao.py` | Baseline zero-shot com Gemini + Spotlight + vocabulário completo de relações; roda em lote e salva resultados com checkpoint. |
| `avaliador_metricas.py` | Avaliador orientado a classe (`DBpediaEvaluator`) com Exact Match, Skeleton Match, precisão/recall de triplas e alucinações. |
| `avaliar_metricas.py` | Avaliação funcional das mesmas métricas, gerando relatório textual de desempenho de predições. |
| `baseline_ollama_zeroshot.py` | Baseline zero-shot puro em Ollama (sem contexto externo), em lote, com retomada por arquivo de saída. |
| `cuda_availability_check.py` | Verificação rápida de disponibilidade CUDA e nome da GPU no PyTorch. |
| `dbpedia_spotlight.py` | Extrai URIs candidatas do Spotlight para perguntas do LC-QuAD e gera dataset com ruído realista. |
| `extrair_uris_por_query.py` | Extrai URIs diretamente da `sparql_query` e gera JSON simplificado (`id`, `question`, `target_uris`). |
| `fine_tuning_t5.py` | Fine-tuning do T5 (`google-t5/t5-small`) com datasets CSV (`input_text`/`target_text`) e `Seq2SeqTrainer`. |
| `gerador_grafo.py` | Expande subgrafo DBpedia a partir de URIs semente, em paralelo, com retry/backoff e log de progresso. |
| `gerador_grafo_relacoes.py` | Faz data augmentation por relação: lê predicados do grafo local e busca exemplos adicionais na DBpedia. |
| `get_dbpedia_relations.py` | Baixa relações DBpedia por paginação (`LIMIT/OFFSET`) e salva lista consolidada. |
| `get_english_dbpedia_relations.py` | Constrói vocabulário universal de relações por varredura alfabética e normalização de prefixos. |
| `inferencia_framework.py` | Framework de benchmark com estratégias de geração (CSV puro e modo com Trie restritiva). |
| `limpar_dataset_dbpedia.py` | Purga dataset removendo exemplos cuja query não retorna resposta válida no endpoint DBpedia atual. |
| `linearizar_queries.py` | Converte SPARQL em alvo linearizado (`A >> p >> B [SEP] ...`) com normalização de variáveis. |
| `list_models.py` | Lista modelos do Gemini disponíveis para `generateContent`. |
| `ollama_rag.py` | Pipeline experimental RAG com embeddings + FAISS + Spotlight e geração local via Ollama. |
| `preparar_dados_t5.py` | Gera CSVs de treino/teste para T5: monta prompt com entidades (e suporte comentado a RAG de relações) e lineariza SPARQL. |
| `recuperador_entidades.py` | Recuperador “Oráculo” (perfect retriever): encontra a pergunta no dataset e retorna entidades `dbr:`. |
| `recuperador_relacoes.py` | Recuperador semântico de relações com FastEmbed (ONNX) e similaridade de cosseno em memória. |
| `recuperador_relacoes_faiss.py` | Recuperador de relações com SentenceTransformer + FAISS + limpeza linguística via spaCy. |
| `recuperador_relacoes_grafo.py` | Graph-RAG ancorado no endpoint: busca propriedades reais das entidades e ranqueia semanticamente. |
| `testar_t5_local.py` | Interface interativa mínima para inferência local do T5 sem injeções extras. |
| `testar_t5_local_com_recuperador_entidade_e_relacao.py` | Inferência T5 com injeção de entidades (oráculo) e relações candidatas (FAISS RAG). |
| `testar_t5_local_com_recuperador_entidades.py` | Inferência T5 com injeção apenas de entidades recuperadas pelo oráculo. |
| `testar_t5_local_com_recuperador_entidades_e_trie.py` | Inferência com decodificação restrita por Trie dinâmica (entidades + ontologia + tokens estruturais). |
| `teste_spacy.py` | Sandbox para extrair palavras-chave (verbos/substantivos) e testar limpeza semântica para RAG. |

