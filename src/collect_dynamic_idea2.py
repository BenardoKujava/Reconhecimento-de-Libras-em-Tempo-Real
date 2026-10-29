import os
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
csv_path = "data/landmarks_idea2.csv"


def extract_idea2_features(frame_buffer):
    frames = np.array(frame_buffer)
    f_last = frames[-1].tolist()
    delta_total = (frames[-1] - frames[0]).tolist()
    step_deltas = (frames[1:] - frames[:-1]).flatten().tolist()

    total_distances = []
    for point_idx in range(21):
        pts = frames[:, point_idx * 3 : point_idx * 3 + 3]
        dists = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        total_distances.append(np.sum(dists))

    return f_last + delta_total + step_deltas + total_distances


cols = [f"pos_{i}" for i in range(63)]
cols += [f"d_total_{i}" for i in range(63)]
cols += [f"step_{s}_{i}" for s in range(9) for i in range(63)]
cols += [f"dist_p{p}" for p in range(21)]
cols.append("label")

frame_buffer = deque(maxlen=10)

print("\n--- COLETA DE DADOS (POSIÇÃO + DESLOCAMENTO + VELOCIDADE) ---")
print("1. Pressione e SEGURE a tecla (ex: 'h') enquanto faz o gesto.")
print("2. Solte a tecla após concluir a trajetória.")
print("3. ESC para sair.\n")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            p9_x = hand_landmarks.landmark[9].x - base_x
            p9_y = hand_landmarks.landmark[9].y - base_y
            p9_z = hand_landmarks.landmark[9].z - base_z
            scale = np.sqrt(p9_x**2 + p9_y**2 + p9_z**2)
            if scale == 0:
                scale = 1.0

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([
                    (lm.x - base_x) / scale,
                    (lm.y - base_y) / scale,
                    (lm.z - base_z) / scale,
                ])

            frame_buffer.append(coords)

    cv2.putText(
        image,
        f"Buffer: {len(frame_buffer)}/10",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Coleta de Dados", image)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif 97 <= key <= 122:
        letra = chr(key).upper()

        if len(frame_buffer) == 10:
            features = extract_idea2_features(frame_buffer)
            nova_linha = pd.DataFrame([features + [letra]], columns=cols)
            nova_linha.to_csv(
                csv_path,
                mode="a",
                header=not os.path.exists(csv_path),
                index=False,
            )
            print(f"Amostra temporal da letra '{letra}' salva!")
        else:
            print("Aguarde o buffer atingir 10 frames para salvar.")

cap.release()
cv2.destroyAllWindows()