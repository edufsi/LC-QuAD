import os

from datasets import download
import spacy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import re
import json
import faiss
import ollama # GAMBIARRA ABSURDA: BIZARRAMENTE SE EU NÃO IMPORTAR O OLLAMA AQUI, O SENTENCE_TRANSFORMER CRASHA AO SER IMPORTADO
import numpy as np

print("A")
from sentence_transformers import SentenceTransformer
print("BB")
class RecuperadorRelacoesFAISS:
    def __init__(self, arquivo_vocabulario):
        print("🧠 Inicializando o RAG Semântico (Motor FAISS)...")
        self.modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.arquivo_indice = "indice_relacoes_t5.faiss"
        self.arquivo_lista = "lista_relacoes_t5.json"
        

        # === INICIALIZAÇÃO DO SPACY (1 ÚNICA VEZ) ===
        print("📚 Carregando motor linguístico spaCy...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ Modelo não encontrado. Baixando 'en_core_web_sm'...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

            
        # Se já existe um índice pré-computado, carrega (muito mais rápido!)
        if os.path.exists(self.arquivo_indice) and os.path.exists(self.arquivo_lista):
            print("⚡ Carregando FAISS do disco...")
            self.indice = faiss.read_index(self.arquivo_indice)
            with open(self.arquivo_lista, 'r', encoding='utf-8') as f:
                self.relacoes_originais = json.load(f)
        else:
            print("⚙️ Construindo índice FAISS do zero...")
            self.relacoes_originais = []
            textos_para_embedding = []
            
            try:
                with open(arquivo_vocabulario, 'r', encoding='utf-8') as f:
                    for linha in f:
                        relacao = linha.strip()
                        if relacao:
                            self.relacoes_originais.append(relacao)
                            
                            # A mágica do "desempacotamento" do CamelCase para melhorar a busca
                            nome_propriedade = relacao.split(':')[-1] 
                            nome_limpo = re.sub(r'([a-z])([A-Z])', r'\1 \2', nome_propriedade).lower()
                            nome_limpo = nome_limpo.replace('_', ' ')
                            textos_para_embedding.append(nome_limpo)
            except FileNotFoundError:
                print(f"❌ Erro: {arquivo_vocabulario} não encontrado!")
                return
                
            print(f"Gerando embeddings para {len(self.relacoes_originais)} relações...")
            vetores = self.modelo_embedding.encode(textos_para_embedding, show_progress_bar=True)
            vetores_float32 = np.array(vetores).astype('float32')
            
            dimensao = vetores_float32.shape[1]
            self.indice = faiss.IndexFlatL2(dimensao)
            self.indice.add(vetores_float32)
            
            # Salva para as próximas execuções
            faiss.write_index(self.indice, self.arquivo_indice)
            with open(self.arquivo_lista, 'w', encoding='utf-8') as f:
                json.dump(self.relacoes_originais, f)
            print("✅ Índice FAISS criado e salvo!")



    def limpar_pergunta(self, pergunta):
        """Método interno para remover entidades e focar na ação."""
        doc = self.nlp(pergunta)
        palavras_chave = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"] and not token.is_stop:
                palavras_chave.append(token.lemma_)
        return " ".join(palavras_chave)
    


    def buscar_top_k(self, pergunta, k=10):
        """Retorna as K relações mais relevantes focando apenas na semântica da ação."""
        # 1. A MÁGICA ACONTECE AQUI: Limpa a pergunta antes de vetorizar
        pergunta_limpa = self.limpar_pergunta(pergunta)
        
        # 2. Vetoriza apenas as palavras-chave (ex: "architect tenant hotel")
        vetor_pergunta = self.modelo_embedding.encode([pergunta_limpa])
        vetor_pergunta = np.array(vetor_pergunta).astype('float32')
        
        # 3. Busca no FAISS
        distancias, indices_recuperados = self.indice.search(vetor_pergunta, k)
        
        # 4. Mapeia de volta para as relações originais
        return [self.relacoes_originais[idx] for idx in indices_recuperados[0]]

# === TESTE ===
if __name__ == "__main__":
    rag = RecuperadorRelacoesFAISS("vocabulario_universal_dbpedia.txt")

    while True:
        pergunta = input("\nDigite uma pergunta (ou 'sair' para encerrar): ")
        if pergunta.lower() == "sair":
            break
        
        top_relacoes = rag.buscar_top_k(pergunta, k=10)
        print("\n🎯 Top 10 Relações Recuperadas:")
        for r in top_relacoes:
            print(f" - {r}")
   