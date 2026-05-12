import torch

# Verifica se o suporte CUDA (NVIDIA GPU) está ativado
print("CUDA disponível:", torch.cuda.is_available())

# Se estiver ativado, mostra o nome da placa gráfica
if torch.cuda.is_available():
    print("Placa de Vídeo:", torch.cuda.get_device_name(0))