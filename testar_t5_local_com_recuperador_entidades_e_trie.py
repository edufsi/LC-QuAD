import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from recuperador_entidades import RecuperadorEntidades

DIRETORIO_MODELO = "./modelo_t5_lcquad_finetuned"
ARQUIVO_RELACOES = "vocabulario_universal_dbpedia.txt"

class TrieNFA:
    """Uma Trie Não-Determinística que permite encadear palavras válidas infinitamente."""
    def __init__(self):
        self.trie = {}

    def adicionar_sequencia(self, token_ids):
        node = self.trie
        for token in token_ids:
            if token not in node:
                node[token] = {}
            node = node[token]
        node['<EOS>'] = True # Marca o fim de uma palavra válida

    def obter_proximos_tokens_validos(self, current_ids):
        # Rastreia todos os estados possíveis na árvore
        estados_atuais = [self.trie]
        
        for token in current_ids:
            proximos_estados = []
            for estado in estados_atuais:
                # 1. Se o token continua a palavra atual
                if token in estado:
                    proximos_estados.append(estado[token])
                # 2. Se a palavra anterior acabou, podemos pular para a raiz e começar uma nova
                if '<EOS>' in estado and token in self.trie:
                    proximos_estados.append(self.trie[token])
            
            estados_atuais = proximos_estados
            if not estados_atuais:
                # Se o modelo tentar um caminho impossível, bloqueia tudo
                return [] 
                
        tokens_permitidos = set()
        for estado in estados_atuais:
            for chave in estado.keys():
                if chave == '<EOS>':
                    # Se estamos no fim de uma palavra, podemos começar qualquer palavra da raiz
                    for chave_raiz in self.trie.keys():
                        if chave_raiz != '<EOS>':
                            tokens_permitidos.add(chave_raiz)
                    tokens_permitidos.add(1) # O Token ID 1 é o </s> (Fim da frase no T5)
                else:
                    tokens_permitidos.add(chave)
                    
        return list(tokens_permitidos)

# ==========================================
# INICIALIZAÇÃO
# ==========================================
print("🔮 Inicializando o Oráculo e carregando Ontologia...")
recuperador = RecuperadorEntidades("lcquad-uris-unified.json")

try:
    with open(ARQUIVO_RELACOES, 'r', encoding='utf-8') as f:
        relacoes_dbpedia = [linha.strip() for linha in f if linha.strip()]
except FileNotFoundError:
    print(f"❌ Erro: {ARQUIVO_RELACOES} não encontrado. Precisamos dele para a Trie!")
    exit()

# Adicionamos propriedades vitais que podem não estar na lista base
relacoes_dbpedia.extend(["rdf:type", "rdfs:type"])

print("🧠 Carregando o Modelo T5...")
tokenizador = T5Tokenizer.from_pretrained(DIRETORIO_MODELO)
modelo = T5ForConditionalGeneration.from_pretrained(DIRETORIO_MODELO)
device = "cuda" if torch.cuda.is_available() else "cpu"
modelo.to(device)
modelo.eval()

def construir_trie_para_pergunta(entidades):
    """Constrói uma Trie dinâmica que conhece a DBpedia e as entidades da pergunta atual."""
    trie = TrieNFA()
    
    # 1. O Vocabulário Estrutural (A nossa gramática)
    estruturas = ["[ANS]", "[VAR1]", "[VAR2]", " [SEP] ", " >> ", "ASK >> ", "COUNT >> "]
    
    # 2. As Entidades da Pergunta (O modelo SÓ pode usar as entidades que injetamos)
    # 3. A Ontologia Inteira da DBpedia
    todas_palavras = estruturas + entidades + relacoes_dbpedia
    
    # Truque do T5: As palavras podem ser geradas com ou sem espaço no início.
    # Adicionamos ambas as variações à Trie para torná-la à prova de balas.
    for palavra in todas_palavras:
        trie.adicionar_sequencia(tokenizador.encode(palavra, add_special_tokens=False))
        trie.adicionar_sequencia(tokenizador.encode(" " + palavra, add_special_tokens=False))
        
    return trie

def traduzir_com_trie(pergunta):
    entidades = recuperador.extrair_entidades(pergunta)
    str_entidades = ", ".join(entidades) if entidades else "None"
    input_text = f"translate question to dbpedia path | Question: {pergunta} | Entities: {str_entidades}"
    
    inputs = tokenizador(input_text, return_tensors="pt", max_length=256, truncation=True).to(device)
    
    # Constrói a Trie específica para esta inferência
    trie_atual = construir_trie_para_pergunta(entidades)
    
    # A Função que o HuggingFace vai chamar a cada milissegundo antes de gerar uma letra
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        # Pega a sequência que o modelo gerou até agora (ignorando o token inicial <pad>)
        current_sequence = input_ids[1:].tolist()
        permitidos = trie_atual.obter_proximos_tokens_validos(current_sequence)
        
        # Fallback de segurança: se houver algum erro, permite encerrar a frase
        return permitidos if permitidos else [1] 

    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn # <--- A MÁGICA ACONTECE AQUI
        )
        
    resposta = tokenizador.decode(outputs[0], skip_special_tokens=True)
    return resposta

# ==========================================
# LOOP DE TESTE
# ==========================================
if __name__ == "__main__":
    print("\n✅ Trie-Constrained Decoder pronto! (Zero Alucinação Ortográfica Garantida)")
    print("-" * 50)
    
    while True:
        pergunta = input("\n📝 Pergunta (Inglês): ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
            
        resposta = traduzir_com_trie(pergunta)
        print(f"🛡️ Target Trie : {resposta}")