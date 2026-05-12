import json
import pandas as pd
from tqdm import tqdm
import ollama

# ==========================================
# CONFIGURAÇÕES
# ==========================================
MODELO_OLLAMA = 'llama3.1' # Certifique-se de ter rodado `ollama pull llama3.1` no terminal
ARQUIVO_CSV = "t5_dataset_test.csv"
ARQUIVO_SAIDA = "resultados_ollama_baseline.json"

def criar_prompt_llama(pergunta_com_entidades):
    """
    Constrói um System Prompt forte com Few-Shot Examples para forçar o Llama 
    a usar a mesma sintaxe linearizada que o T5 usa.
    """
    prompt = f"""You are a strict Semantic Parsing engine. Your ONLY task is to translate an English question into a linearized DBpedia graph path.

RULES:
1. NEVER write introductory text, explanations, or markdown code blocks (```).
2. ONLY output the raw path.
3. Use '>>' to connect nodes and relations.
4. Use '[SEP]' to intersect multiple paths.
5. Use '[ANS]' for the final answer node and '[VAR1]' for intermediate unknown nodes.
6. Use 'COUNT >>' or 'ASK >>' at the very beginning if the question requires counting or a boolean yes/no.

EXAMPLES:
Input: Question: Where was George Washington born? | Entities possibly involved: dbr:George_Washington
Output: dbr:George_Washington >> dbo:birthPlace >> [ANS]

Input: Question: How many mammals are in the Chordate phylum? | Entities possibly involved: dbr:Chordate
Output: COUNT >> [ANS] >> dbo:phylum >> dbr:Chordate [SEP] [ANS] >> rdf:type >> dbo:Mammal

Input: Question: What is the alma mater of the president whose vice president was Enrique? | Entities possibly involved: dbr:Enrique
Output: [VAR1] >> dbo:almaMater >> [ANS] [SEP] [VAR1] >> dbo:vicePresident >> dbr:Enrique [SEP] [VAR1] >> rdf:type >> dbo:President

NOW PARSE THIS INPUT:
{pergunta_com_entidades}
Output:"""
    return prompt

def rodar_baseline_ollama():
    print(f"📂 Lendo o dataset de teste ({ARQUIVO_CSV})...")
    try:
        df = pd.read_csv(ARQUIVO_CSV)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {ARQUIVO_CSV} não encontrado.")
        return

    resultados = []
    print(f"🚀 Iniciando inferência em {len(df)} exemplos no Ollama ({MODELO_OLLAMA})...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Gerando Predições"):
        prompt_t5 = str(row['input_text'])
        ground_truth = str(row['target_text'])
        
        # Limpamos o prefixo inútil para o Llama focar no que importa
        input_limpo = prompt_t5.replace("translate question to dbpedia path: ", "")
        
        prompt_final = criar_prompt_llama(input_limpo)
        
        # Chamada ao Ollama
        try:
            resposta = ollama.generate(
                model=MODELO_OLLAMA, 
                prompt=prompt_final,
                options={
                    "temperature": 0.0, # Temperatura 0 para ser determinístico/frio
                    "top_p": 0.9
                }
            )
            # Remove qualquer quebra de linha extra ou espaço que o Llama possa ter colocado
            predicao = resposta['response'].strip()
            
            # Filtro de segurança: se o modelo pedir desculpas ou der texto, tentamos limpar
            if "I cannot" in predicao or "Here is" in predicao:
                # Tenta extrair só a linha que tem '>>'
                linhas = predicao.split('\n')
                for linha in linhas:
                    if '>>' in linha:
                        predicao = linha.strip()
                        break
                        
        except Exception as e:
            print(f"\n❌ Erro na API do Ollama no index {index}: {e}")
            predicao = ""

        resultados.append({
            "id": index,
            "input_prompt": prompt_t5,
            "ground_truth": ground_truth,
            "prediction": predicao
        })

    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f"✅ Baseline do Ollama concluído e salvo em: {ARQUIVO_SAIDA}")
    print("Próximos passos sugeridos:")
    print(f"1. Rode o 'avaliador_pro.py' apontando para '{ARQUIVO_SAIDA}' para ver as métricas brutas.")
    print(f"2. Rode o 'aplicar_correcao.py' (O Investigador) em cima de '{ARQUIVO_SAIDA}' para ver se o Llama alucina menos vocabulário que o T5.")

if __name__ == "__main__":
    rodar_baseline_ollama()