import spacy

print("Carregando o motor linguístico...")
nlp = spacy.load("en_core_web_sm")

def extrair_conceitos_para_rag(pergunta):
    """
    Usa NLP para extrair apenas verbos e substantivos comuns, 
    removendo Nomes Próprios (Entidades), preposições e pronomes.
    """
    doc = nlp(pergunta)
    palavras_chave = []
    
    for token in doc:
        # Pega Substantivos comuns (NOUN) e Verbos (VERB)
        # Ignora Nomes Próprios (PROPN), Stopwords (the, is, of) e pontuação
        if token.pos_ in ["NOUN", "VERB"] and not token.is_stop:
            palavras_chave.append(token.lemma_) # Usa o lema (ex: "developed" -> "develop")
            
    # Junta tudo numa string limpa
    return " ".join(palavras_chave)


while True:
    pergunta = input("\nDigite uma pergunta (ou 'sair' para encerrar): ")
    if pergunta.lower() == "sair":
        break
    conceitos = extrair_conceitos_para_rag(pergunta)
    print(f"Foco para o FAISS: '{conceitos}'")