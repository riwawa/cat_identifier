"""
Diagnóstico: mede a proporção de pixels escuros em cada foto, agrupada por
classe (rodolfo/luna), para investigar se existe uma correlação 
entre "quantidade de área escura na imagem" e a classe.
"""
import classificacao_gato  
dataset_completo = classificacao_gato.dataset_completo
import cv2
import numpy as np
from collections import defaultdict
LIMIAR_ESCURO = 50  # 0-255. Pixels com brilho abaixo disso contam como "escuro".


def proporcao_pixels_escuros(caminho_imagem, limiar=LIMIAR_ESCURO):
    imagem = cv2.imread(caminho_imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    pixels_escuros = np.sum(cinza < limiar)
    total_pixels = cinza.size
    return pixels_escuros / total_pixels


# Percorre TODAS as imagens do dataset (não só a validação), agrupando por classe.
proporcoes_por_classe = defaultdict(list)

for caminho, indice_classe in dataset_completo.samples:
    nome_classe = dataset_completo.classes[indice_classe]
    proporcao = proporcao_pixels_escuros(caminho)
    proporcoes_por_classe[nome_classe].append(proporcao)

print(f"--- Proporção de pixels escuros (limiar={LIMIAR_ESCURO}) ---\n")

for nome_classe, valores in proporcoes_por_classe.items():
    media = np.mean(valores)
    desvio = np.std(valores)
    print(f"{nome_classe}:")
    print(f"  média:  {media:.1%}")
    print(f"  desvio: {desvio:.1%}")
    print(f"  min:    {min(valores):.1%}")
    print(f"  max:    {max(valores):.1%}\n")

print(proporcao_pixels_escuros("dataset/luna/IMG_4953.jpg"))
