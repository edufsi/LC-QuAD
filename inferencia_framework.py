import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

class GeradorBase:
    """Classe base para estratégias de geração."""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def gerar(self, row):
        # Deve ser implementado pelas subclasses
        raise NotImplementedError

# --- ESTRATÉGIA 1: Pura (Lê o prompt pronto do CSV) ---
class EstrategiaCSV(GeradorBase):
    def gerar(self, row):
        prompt = str(row['input_text'])
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=256, num_beams=5, early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- ESTRATÉGIA 2: Trie (Geração Restrita) ---
class EstrategiaTrie(GeradorBase):
    def __init__(self, model, tokenizer, device, trie_obj):
        super().__init__(model, tokenizer, device)
        self.trie = trie_obj

    def gerar(self, row):
        prompt = str(row['input_text'])
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(self.device)
        
        # Aqui injetamos a sua função de restrição de prefixo (Trie)
        outputs = self.model.generate(
            **inputs, 
            max_length=256, 
            num_beams=5,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get_next_tokens(sent.tolist())
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- ENGINE PRINCIPAL ---
def rodar_benchmark(estrategia, arquivo_csv, arquivo_saida):
    df = pd.read_csv(arquivo_csv)
    resultados = []
    
    print(f"🚀 Rodando benchmark: {type(estrategia).__name__}")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        predicao = estrategia.gerar(row)
        resultados.append({
            "id": index,
            "input_prompt": str(row.get('input_text', '')),
            "ground_truth": str(row.get('target_text', '')),
            "prediction": predicao
        })

    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f"✅ Resultados salvos em {arquivo_saida}")

if __name__ == "__main__":
    MODEL_PATH = "./modelo_t5_lcquad_finetuned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

    # EXEMPLO DE USO:
    # Para usar CSV puro:
    est = EstrategiaCSV(model, tokenizer, device)
    rodar_benchmark(est, "t5_dataset_test.csv", "resultados_simples.json")
    
    # Para usar Trie (exemplo hipotético):
    # minhatrie = MinhaTrieManager()
    # est_trie = EstrategiaTrie(model, tokenizer, device, minhatrie)
    # rodar_benchmark(est_trie, "t5_dataset_test.csv", "resultados_trie.json")