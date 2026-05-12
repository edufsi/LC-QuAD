import os
# Impede o crash entre FAISS e PyTorch na inferência
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import json

import faiss
import ollama
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Importa os dois Oráculos
from recuperador_entidades import RecuperadorEntidades 
from recuperador_relacoes_faiss import RecuperadorRelacoesFAISS 

DIRETORIO_MODELO = "./modelo_t5_lcquad_finetuned"

print("🔮 Inicializando o Oráculo de Entidades...")
oraculo_entidades = RecuperadorEntidades("lcquad-uris-unified.json")

print("📚 Inicializando o RAG Semântico de Relações (FAISS + spaCy)...")
# Lembre-se de apontar para o seu txt com as relações
rag_relacoes = RecuperadorRelacoesFAISS("vocabulario_universal_dbpedia.txt")

print("🧠 Carregando o Especialista em DBpedia (T5)...")
tokenizador = T5Tokenizer.from_pretrained(DIRETORIO_MODELO)
modelo = T5ForConditionalGeneration.from_pretrained(DIRETORIO_MODELO)

device = "cuda" if torch.cuda.is_available() else "cpu"
modelo.to(device)
modelo.eval() 

def traduzir_pergunta(pergunta):
    # 1. Extrai Entidades
    entidades = oraculo_entidades.extrair_entidades(pergunta)
    str_entidades = ", ".join(entidades) if entidades else "None"
    
    # 2. Extrai Relações (Top 25, igual ao treino)
    # A classe FAISS já roda o spaCy internamente para limpar a pergunta
    relacoes = rag_relacoes.buscar_top_k(pergunta, k=5)
    str_relacoes = ", ".join(relacoes)
    
    # 3. Monta o Super-Prompt EXATAMENTE como foi treinado
    input_text = f"translate question to dbpedia path | Question: {pergunta} | Entities possibly involved: {str_entidades} | Relations possibly involved: {str_relacoes}"

    print(f"\n[Prompt Input] -> {input_text[:150]}... (truncado para visualização)")
    
    inputs = tokenizador(input_text, return_tensors="pt", max_length=256, truncation=True)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )
        
    resposta = tokenizador.decode(outputs[0], skip_special_tokens=True)
    return resposta

if __name__ == "__main__":
    print("\n✅ Modelo pronto! Digite 'sair' para encerrar.")
    print("-" * 50)
    
    while True:
        pergunta = input("\n📝 Pergunta (Inglês): ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
            
        resposta_t5 = traduzir_pergunta(pergunta)
        print(f"🎯 Target T5 : {resposta_t5}")