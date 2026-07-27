"""
testa o pipeline completo (detecção YOLO + classificação treinada) ao vivo,
usando a webcam.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

classificador = models.resnet18(weights=None) #vai carregar pesos do arquivo .pth
numero_classes = 2
classificador.fc = nn.Linear(classificador.fc.in_features, numero_classes)
classificador.load_state_dict(torch.load("modelo_classificador.pth", map_location=device))
classificador = classificador.to(device)
classificador.eval()

nomes_classes = torch.load("classes.pth")

detector = YOLO("yolov8n.pt")
id_classe_gato = [k for k, v in detector.names.items() if v == "cat"][0]

# --- Transformações -- 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# --- Loop de captura ao vivo ---
captura = cv2.VideoCapture(0)

while True:
    ok, frame = captura.read()
    if not ok:
        print("Não foi possível ler da webcam.")
        break

    resultados_deteccao = detector(frame, verbose=False) 

    for resultado in resultados_deteccao:
        for box in resultado.boxes:
            if int(box.cls[0]) != id_classe_gato:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # fatiamento do frame original para obter a região do gato detectado
            recorte = frame[y1:y2, x1:x2]
            if recorte.size == 0:
                continue  

            # OpenCV entrega BGR; PIL/torchvision esperam RGB -- conversão necessária
            recorte_rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
            imagem_pil = Image.fromarray(recorte_rgb)

            tensor_entrada = transformacoes(imagem_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                saida = classificador(tensor_entrada)
                indice_previsto = torch.argmax(saida, dim=1).item()
                nome_previsto = nomes_classes[indice_previsto]

            # Desenha a caixa e o rótulo no frame original
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, nome_previsto, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

    cv2.imshow("identificacao de gatos - pressione q para sair", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

captura.release()
cv2.destroyAllWindows()