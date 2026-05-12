import cv2
import pandas as pd
import os
import mediapipe as mp

# Inicialização
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

#Câmera
cap = cv2.VideoCapture(0)

# Pasta de dados
if not os.path.exists('data'):
    os.makedirs('data')

csv_path = 'data/landmarks.csv'

# Criar os nomes das colunas
cols = []
for i in range(21):
    cols.extend([f'x{i}', f'y{i}', f'z{i}'])
cols.append('label')

print("\n--- MODO DE COLETA DE DADOS ---")
print("Aperte uma letra (A-Z) para salvar a amostra.")
print("Aperte ESC para finalizar.\n")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    current_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Coleta as coordenadas
            for lm in hand_landmarks.landmark:
                current_landmarks.extend([lm.x, lm.y, lm.z])

    cv2.imshow('Camera', image)
    
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    
    # Verifica se uma letra é pressionada
    elif 97 <= key <= 122:
        letra = chr(key).upper()
        
        if current_landmarks:
            # Cria linha com os dados no csv
            nova_linha = pd.DataFrame([current_landmarks + [letra]], columns=cols)
            
            # Salva em tempo real
            nova_linha.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
            
            print(f"Letra {letra} salva no arquivo!")
        else:
            print("Aviso: Mão não detectada. Tente novamente.")

cap.release()
cv2.destroyAllWindows()
print(f"\nProcesso finalizado. Verifique o arquivo em: {csv_path}")