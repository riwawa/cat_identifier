"""
carregar o dataset (rodolfo/ vs luna/) e preparar treino/validação.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



torch.manual_seed(42)
# pré-processamento: padronizar o formato das imagens para 224x224, converter para tensor e normalizar
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

caminho_dataset = "dataset"
dataset_completo = datasets.ImageFolder(root=caminho_dataset, transform=transformacoes)

print("Classes encontradas:", dataset_completo.classes)
print("Total de imagens:", len(dataset_completo))

# divisao de treino 80/20 para validação
tamanho_treino = int(len(dataset_completo) * 0.8)
tamanho_val = len(dataset_completo) - tamanho_treino

dataset_treino, dataset_val = random_split(dataset_completo, [tamanho_treino, tamanho_val])  

# dataloaders para treino e validação
loader_treino = DataLoader(dataset_treino, batch_size=8, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False)

print(f"Imagens de treino: {len(dataset_treino)} | Imagens de validação: {len(dataset_val)}")

"""
montar o modelo de classificação via transfer learning.
"""

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Rodando em:", device)

# carregue a ResNet18 pré-treinada no ImageNet.
modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# congela TODOS os parâmetros do modelo.
for param in modelo.parameters():
    param.requires_grad = False


# substituir a última camada (modelo.fc) por uma nova.
numero_classes = len(dataset_completo.classes)

modelo.fc = nn.Linear(modelo.fc.in_features, numero_classes)

# Move o modelo inteiro pro device escolhido (GPU ou CPU)
modelo = modelo.to(device)

# define a função de perda.
funcao_perda = nn.CrossEntropyLoss()

# define o otimizador apenas para os parâmetros da nova camada (modelo.fc).
otimizador = torch.optim.Adam(modelo.fc.parameters(), lr=0.001)

print("Modelo pronto. Camada final:", modelo.fc)

"""
loop de treino.
"""

numero_epocas = 10

modelo.train()

for epoca in range(numero_epocas):
    perda_acumulada = 0.0

    for imagens, rotulos in loader_treino:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)

        otimizador.zero_grad()

        saidas = modelo(imagens)
        perda = funcao_perda(saidas, rotulos) # compara predição com rótulo real

        perda.backward() # calcula o gradiente de cada peso treinável
        otimizador.step() # atualiza os pesos usando esse gradiente
        perda_acumulada += perda.item()

    perda_media = perda_acumulada / len(loader_treino)
    #print(f"Época {epoca+1}/{numero_epocas} - Perda média: {perda_media:.4f}")

"""
avaliar o modelo no conjunto de validação.
"""
modelo.eval()

total_imagens = 0
total_acertos = 0

with torch.no_grad():
    for imagens, rotulos in loader_val:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)

        saidas = modelo(imagens)
        predicoes = torch.argmax(saidas, dim=1)

        acertos_neste_batch = (predicoes == rotulos).sum().item()

        total_acertos += acertos_neste_batch
        total_imagens += rotulos.size(0)

acuracia = total_acertos / total_imagens
print(f"Acurácia na validação: {acuracia:.2%} ({total_acertos}/{total_imagens})")

"""
identificar quais imagens da validação foram classificadas errado.
"""

modelo.eval()

#contador global de imagens processadas, para rastrear a posição no dataset original.
indice_global = 0

with torch.no_grad():
    for imagens, rotulos in loader_val:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)

        saidas = modelo(imagens)
        predicoes = torch.argmax(saidas, dim=1)

        # TODO 2: Percorra cada imagem DENTRO deste batch individualmente.
        # Dica: range(len(rotulos)) te dá os índices 0, 1, 2... dentro do batch atual.
        for i in range(len(rotulos)):
            # Compare a predição desta imagem específica com o rótulo real.
            if predicoes[i] != rotulos[i]:
                posicao_no_completo = dataset_val.indices[indice_global]
                caminho_arquivo = dataset_completo.samples[posicao_no_completo][0]

                classe_real = dataset_completo.classes[rotulos[i]]
                classe_prevista = dataset_completo.classes[predicoes[i]]

                print(f"ERRO -> Arquivo: {caminho_arquivo}")
                print(f"        Real: {classe_real} | Previsto: {classe_prevista}\n")

            indice_global += 1

"""
extrator de embeddings
"""

import torch
from torchvision import models

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

extrator = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in extrator.parameters():
    param.requires_grad = False

extrator.fc = torch.nn.Identity()

extrator = extrator.to(device)
extrator.eval()


entrada_teste = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    saida_teste = extrator(entrada_teste)
print("Formato do embedding:", saida_teste.shape)

"""
calcular o vetor de referência (protótipo) de cada classe,
"""

import torch

embeddings_por_classe = {nome: [] for nome in dataset_completo.classes}

with torch.no_grad():
    for imagens, rotulos in loader_treino:
        imagens = imagens.to(device)

        embeddings = extrator(imagens)

        for i in range(len(rotulos)):
            nome_classe = dataset_completo.classes[rotulos[i]]

            embeddings_por_classe[nome_classe].append(embeddings[i])


prototypes = {}

for nome_classe, lista_embeddings in embeddings_por_classe.items():
    tensor_empilhado = torch.stack(lista_embeddings)
    prototypes[nome_classe] = tensor_empilhado.mean(dim=0)

for nome_classe, vetor in prototypes.items():
    print(f"Protótipo de '{nome_classe}': shape {vetor.shape}, "
          f"baseado em {len(embeddings_por_classe[nome_classe])} fotos")
    

"""
classificar imagens da validação comparando seus embeddings com
os protótipos, usando similaridade de cosseno. depois, medir a acurácia.
"""

def classificar_por_similaridade(embedding, prototypes):
    melhor_classe = None
    melhor_similaridade = -float("inf")  # começa "impossivelmente" baixo

    for nome_classe, vetor_prototype in prototypes.items():
        similaridade = F.cosine_similarity(embedding, vetor_prototype, dim=0)
        if similaridade > melhor_similaridade:
            melhor_similaridade = similaridade
            melhor_classe = nome_classe

    return melhor_classe, melhor_similaridade


total_imagens = 0
total_acertos = 0

extrator.eval()

with torch.no_grad():
    for imagens, rotulos in loader_val:
        imagens = imagens.to(device)

        for i in range(len(rotulos)):
            embedding = extrator(imagens[i].unsqueeze(0)).squeeze(0)

            embedding = embedding.squeeze(0)

            classe_prevista, similaridade = classificar_por_similaridade(embedding, prototypes)

            classe_real = dataset_completo.classes[rotulos[i]]

            if classe_prevista == classe_real:  
                total_acertos += 1

            total_imagens += 1

acuracia = total_acertos / total_imagens
print(f"Acurácia (embeddings): {acuracia:.2%} ({total_acertos}/{total_imagens})")
torch.save(modelo.state_dict(), "modelo_classificador.pth")
torch.save(prototypes, "prototypes.pth")
torch.save(dataset_completo.classes, "classes.pth")  