import google.generativeai as genai

CHAVE_API = "AIzaSyAqVAk1oLyb1nzf3US0uaH4Feo4uPzzlOk"
genai.configure(api_key=CHAVE_API)

print("Modelos disponíveis para geração de texto:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)