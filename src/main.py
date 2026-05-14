import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import warnings
import time

# Configuração inicial
warnings.filterwarnings("ignore", category=UserWarning)

# Controle
palavra_formada = ""
letra_atual = ""
tempo_inicio = 0
tempo_para_confirmar = 2.0

# Model
model_path = 'models/libras_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(">>> SISTEMA ATIVADO. Pressione ESC para sair.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicao_viva = "..." 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            try:
                input_data = np.array(coords).reshape(1, -1)
                probabilities = model.predict_proba(input_data)
                max_prob = np.max(probabilities)
                
                if max_prob > 0.80:
                    predicao_viva = model.predict(input_data)[0]
                    
                    # Temporizador
                    if predicao_viva != "Nada":
                        if predicao_viva == letra_atual:
                            tempo_passado = time.time() - tempo_inicio
                            
                            largura_barra = int((tempo_passado / tempo_para_confirmar) * 300)
                            cv2.rectangle(image, (30, 110), (30 + min(largura_barra, 300), 120), (0, 255, 255), -1)
                            
                            if tempo_passado >= tempo_para_confirmar:
                                palavra_formada += predicao_viva
                                letra_atual = ""
                                tempo_inicio = time.time()
                        else:
                            letra_atual = predicao_viva
                            tempo_inicio = time.time()
                else:
                    predicao_viva = "..."
                    letra_atual = ""

            except Exception as e:
                print(f"ERRO NA PREDICAO: {e}")

    # Interface
    cv2.rectangle(image, (20, 20), (350, 130), (0, 0, 0), -1)
    cv2.putText(image, f"Letra: {predicao_viva}", (40, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    
    cv2.rectangle(image, (0, h - 70), (w, h), (50, 50, 50), -1)
    cv2.putText(image, f"TEXTO: {palavra_formada}", (20, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('Reconhecimento de Libras em Tempo Real', image)
    
    # Comandos do teclado
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord(' '): 
        palavra_formada += " "
    elif key == ord('d'):
        palavra_formada = palavra_formada[:-1]
    elif key == ord('c'):
        palavra_formada = ""

cap.release()
cv2.destroyAllWindows()