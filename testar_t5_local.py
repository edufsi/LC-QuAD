import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

DIRETORIO_MODELO = "./modelo_t5_lcquad_finetuned"

print("🧠 Carregando o Especialista em DBpedia...")
# Carregamos o modelo a partir da pasta onde foi salvo
tokenizador = T5Tokenizer.from_pretrained(DIRETORIO_MODELO)
modelo = T5ForConditionalGeneration.from_pretrained(DIRETORIO_MODELO)

# Passamos o modelo para a GPU para inferência instantânea
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo.to(device)
modelo.eval() # Coloca o modelo em modo de avaliação (desliga o aprendizado)

def traduzir_pergunta(pergunta):
    # O exato prefixo que usamos no treinamento
    input_text = f"translate question to dbpedia path: {pergunta}"
    
    # Tokenização
    inputs = tokenizador(input_text, return_tensors="pt", max_length=128, truncation=True)
    inputs = inputs.to(device)
    
    # Geração da resposta
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_length=128,
            num_beams=5,         # Explora 5 árvores de probabilidade e escolhe a melhor
            early_stopping=True
        )
        
    # Decodifica os IDs de volta para texto legível
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
        print(f"🎯 Target T5: {resposta_t5}")