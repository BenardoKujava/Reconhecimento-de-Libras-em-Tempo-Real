import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import warnings


warnings.filterwarnings("ignore", category=UserWarning)

# Modelo
model_path = 'models/libras_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(">>> SISTEMA ATIVADO.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            try:
                input_data = np.array(coords).reshape(1, -1)
                
                probabilities = model.predict_proba(input_data)
                prediction = model.predict(input_data)[0]
                max_prob = np.max(probabilities)

                if max_prob > 0.70:
                    cv2.putText(image, f"{prediction} ({max_prob*100:.0f}%)", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            except Exception as e:
                print(f"ERRO NA PREDICAO: {e}")

    cv2.imshow('Reconhecimento de Libras em Tempo Real', image)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()