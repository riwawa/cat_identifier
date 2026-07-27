"""Detecção de objeto com YOLO pré-treinado."""

from ultralytics import YOLO
import cv2

modelo = YOLO("yolov8n.pt")
indices_encontrados = [indice for indice, nome in modelo.names.items() if nome == "cat"]
id_classe_interesse = indices_encontrados[0] if indices_encontrados else None

caminho_imagem = "gato.jpeg"

resultados = modelo(caminho_imagem)
imagem = cv2.imread(caminho_imagem)

for resultado in resultados:
    for box in resultado.boxes:
        classe_detectada = int(box.cls[0])
        confianca = float(box.conf[0])
        
        if classe_detectada == id_classe_interesse:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"Classe: {modelo.names[classe_detectada]}, Confiança: {confianca:.2f}, Coordenadas: ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ----- recorte
recorte = imagem[y1:y2, x1:x2]
cv2.imwrite("teste_recorte.jpg", recorte)
