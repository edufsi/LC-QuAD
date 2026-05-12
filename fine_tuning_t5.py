import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# ==========================================
# CONFIGURAÇÕES DA RTX 4060 (8GB VRAM)
# ==========================================
MODELO_NOME = "google-t5/t5-small" # Troque para 't5-base' se quiser tentar um modelo maior depois
ARQUIVO_TREINO = "t5_dataset_train.csv"
ARQUIVO_TESTE = "t5_dataset_test.csv"
DIRETORIO_SAIDA = "./modelo_t5_lcquad_finetuned"

def preparar_dados_para_treino():
    print(f"📥 Carregando datasets e o tokenizador do {MODELO_NOME}...")
    dataset = load_dataset("csv", data_files={"train": ARQUIVO_TREINO, "test": ARQUIVO_TESTE})
    tokenizador = T5Tokenizer.from_pretrained(MODELO_NOME)

    def tokenizar_batch(batch):
        # Tokeniza a pergunta (Input)
        inputs = tokenizador(batch["input_text"], max_length=256, truncation=True, padding="max_length")
        # Tokeniza a estrutura SPARQL linearizada (Target)
        targets = tokenizador(batch["target_text"], max_length=256, truncation=True, padding="max_length")
        
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("⚙️ Tokenizando os dados (isso pode levar um minutinho)...")
    dataset_tokenizado = dataset.map(tokenizar_batch, batched=True, remove_columns=["input_text", "target_text"])
    
    return dataset_tokenizado, tokenizador

def treinar_modelo():
    dataset_tokenizado, tokenizador = preparar_dados_para_treino()
    
    print("🧠 Carregando o modelo T5 para a GPU...")
    modelo = T5ForConditionalGeneration.from_pretrained(MODELO_NOME)

    # Hiperparâmetros otimizados para 8GB VRAM
    argumentos_treino = Seq2SeqTrainingArguments(
        output_dir=DIRETORIO_SAIDA,
        eval_strategy="epoch",      # Avalia no fim de cada época
        learning_rate=3e-4,         # Taxa de aprendizado padrão para T5
        per_device_train_batch_size=16, # Se der Out of Memory, baixe para 8
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,         # Guarda apenas os 2 melhores modelos para poupar disco
        num_train_epochs=5,         # 5 épocas costuma ser suficiente para ele decorar a sintaxe
        predict_with_generate=True, # Necessário para modelos Seq2Seq
        fp16=True,                  # MAGIA NEGRA: Usa precisão de 16-bits. Salva VRAM e acelera o treino na RTX série 4000
        logging_steps=50,
    )

    trainer = Seq2SeqTrainer(
        model=modelo,
        args=argumentos_treino,
        train_dataset=dataset_tokenizado["train"],
        eval_dataset=dataset_tokenizado["test"],
        processing_class=tokenizador,
    )

    print("🚀 Iniciando o Fine-Tuning! Acompanhe a barra de progresso...")
    trainer.train()

    print(f"💾 Treino concluído! Salvando modelo final em: {DIRETORIO_SAIDA}")
    trainer.save_model(DIRETORIO_SAIDA)
    tokenizador.save_pretrained(DIRETORIO_SAIDA)

if __name__ == "__main__":
    # Verifica se a GPU está visível para garantir que não vamos treinar na CPU
    if not torch.cuda.is_available():
        print("❌ ERRO CRÍTICO: CUDA não encontrado. O PyTorch não está vendo a RTX 4060.")
        print("Cancele a execução ou o treino vai demorar dias na CPU.")
    else:
        print(f"✅ GPU Detectada: {torch.cuda.get_device_name(0)}")
        treinar_modelo()