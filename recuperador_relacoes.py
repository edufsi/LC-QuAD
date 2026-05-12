import re
import numpy as np
from fastembed import TextEmbedding

class RecuperadorRelacoes:
    def __init__(self, arquivo_vocabulario):
        print("🧠 Inicializando o RAG Semântico (Motor ONNX/FastEmbed)...")
        # Usamos o mesmo modelo, mas via ONNX (Zero PyTorch dependências)
        self.modelo = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        
        self.relacoes_originais = []
        textos_para_embedding = []
        
        try:
            with open(arquivo_vocabulario, 'r', encoding='utf-8') as f:
                for linha in f:
                    relacao = linha.strip()
                    if relacao:
                        self.relacoes_originais.append(relacao)
                        # Limpeza: "dbo:riverMouth" -> "river mouth"
                        nome_propriedade = relacao.split(':')[-1] 
                        nome_limpo = re.sub(r'([a-z])([A-Z])', r'\1 \2', nome_propriedade).lower()
                        nome_limpo = nome_limpo.replace('_', ' ')
                        textos_para_embedding.append(nome_limpo)
                        
            print(f"📚 {len(self.relacoes_originais)} Relações processadas.")
        except FileNotFoundError:
            print(f"❌ Erro: {arquivo_vocabulario} não encontrado!")
            return

        print("⚡ Computando embeddings (Aceleração ONNX)...")
        # Fastembed gera um iterador de arrays numpy, convertemos para uma matriz 2D
        embeddings_lista = list(self.modelo.embed(textos_para_embedding))
        self.embeddings_corpus = np.vstack(embeddings_lista)
            
    def buscar_top_k(self, pergunta, k=10):
        # 1. Gera o embedding da pergunta
        query_emb = list(self.modelo.embed([pergunta]))[0]
        
        # 2. Matemática Pura: Similaridade de Cosseno com Numpy
        norma_corpus = np.linalg.norm(self.embeddings_corpus, axis=1)
        norma_query = np.linalg.norm(query_emb)
        
        # Produto escalar dividido pela magnitude
        scores = np.dot(self.embeddings_corpus, query_emb) / (norma_corpus * norma_query)
        
        # 3. Pega os K maiores índices
        top_indices = np.argsort(scores)[::-1][:k]
        
        return [self.relacoes_originais[i] for i in top_indices]

# === TESTE DO RAG ===
if __name__ == "__main__":
    rag = RecuperadorRelacoes("vocabulario_universal_dbpedia.txt")
    
    pergunta = "What is the river whose mouth is located in Murray Mouth?"
    print(f"\n📝 Pergunta: {pergunta}")
    
    top_relacoes = rag.buscar_top_k(pergunta, k=10)
    print("\n🎯 Top 10 Relações Recuperadas:")
    for r in top_relacoes:
        print(f" - {r}")