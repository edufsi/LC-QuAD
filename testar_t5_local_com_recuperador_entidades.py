import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Importe a sua classe Oráculo do arquivo que você criou
from recuperador_entidades import RecuperadorEntidades 

DIRETORIO_MODELO = "./modelo_t5_lcquad_finetuned"

print("🔮 Inicializando o Oráculo...")
# Importante: Para o teste interativo, usamos o JSON de TESTE
recuperador = RecuperadorEntidades("lcquad-uris-unified.json")

print("🧠 Carregando o Especialista em DBpedia...")
tokenizador = T5Tokenizer.from_pretrained(DIRETORIO_MODELO)
modelo = T5ForConditionalGeneration.from_pretrained(DIRETORIO_MODELO)

device = "cuda" if torch.cuda.is_available() else "cpu"
modelo.to(device)
modelo.eval() 

def traduzir_pergunta(pergunta):
    # 1. O Oráculo extrai as entidades da pergunta
    entidades = recuperador.extrair_entidades(pergunta)
    str_entidades = ", ".join(entidades) if entidades else "None"
    
    # 2. Montamos o prompt EXATAMENTE como foi treinado
    input_text = f"translate question to dbpedia path | Question: {pergunta} | Entities possibly involved: {str_entidades}"
    
    # Mostramos o que o modelo realmente está a ler
    print(f"\n[Prompt Input] -> {input_text}")
    
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
    print("⚠️ Lembre-se: O Oráculo só vai achar entidades para perguntas idênticas às do JSON de teste.")
    print("-" * 50)
    
    while True:
        pergunta = input("\n📝 Pergunta (Inglês): ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
            
        resposta_t5 = traduzir_pergunta(pergunta)
        print(f"🎯 Target T5 : {resposta_t5}")