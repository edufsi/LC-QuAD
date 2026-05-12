import json

class RecuperadorEntidades:
    """
    Classe base para recuperação de entidades. 
    Atualmente implementa a estratégia 'Oráculo' (Perfect Retriever) para testes.
    """
    def __init__(self, arquivo_json):
        self.dados_conhecidos = []
        try:
            with open(arquivo_json, 'r', encoding='utf-8') as f:
                self.dados_conhecidos = json.load(f)
            print(f"🔮 Oráculo inicializado com {len(self.dados_conhecidos)} perguntas conhecidas.")
        except FileNotFoundError:
            print(f"⚠️ Aviso: Arquivo {arquivo_json} não encontrado. Oráculo vazio.")

    def limpar_uri(self, uri):
        uri = uri.strip('<>') 
        return uri.replace("http://dbpedia.org/resource/", "dbr:")

    def extrair_entidades(self, pergunta):
        """
        Busca a pergunta no 'gabarito' e retorna apenas as URIs de entidades.
        Se não encontrar, retorna uma lista vazia.
        """

        for item in self.dados_conhecidos:
            q_original = item.get('question', item.get('corrected_question', '')).strip()
            
            if q_original.lower() == pergunta.lower():
                uris_brutas = item.get('target_uris', [])
                entidades = []
                
                for uri in uris_brutas:
                    # Filtra apenas o que é entidade (recurso do DBpedia)
                    if 'resource/' in uri or uri.startswith('dbr:'):
                        entidades.append(self.limpar_uri(uri))
                        
                # Remove duplicatas e retorna
                return list(set(entidades))

        print(f"⚠️ Oráculo: Pergunta não encontrada no gabarito. Retornando lista vazia para: '{pergunta}'")
        # Se a pergunta for totalmente nova e não estiver no JSON
        return []

# === TESTE RÁPIDO DO ORÁCULO ===
if __name__ == "__main__":
    recuperador = RecuperadorEntidades("lcquad-uris-test-data.json")
    
    pergunta_teste = "Where was the person born who died in Bryn Mawr Hospital?"
    entidades = recuperador.extrair_entidades(pergunta_teste)
    
    print(f"\nPergunta: {pergunta_teste}")
    print(f"Entidades Injetadas: {entidades}")