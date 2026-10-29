from collections import deque
import os
import pickle
import time
import warnings
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Libras - Trabalho de Conclusão de Curso", layout="wide")

# Lista de letras dinâmicas da Libras
LETRAS_DINAMICAS = ["H", "J", "K", "X", "Z", "Y"]
COOLDOWN_DINAMICO = 1.5


@st.cache_resource
def get_cap():
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)


camera = get_cap()

# Estados
if "palavra" not in st.session_state:
    st.session_state.palavra = ""
if "letra_atual" not in st.session_state:
    st.session_state.letra_atual = ""
if "tempo_inicio" not in st.session_state:
    st.session_state.tempo_inicio = 0
if "ultimo_disparo_dinamico" not in st.session_state:
    st.session_state.ultimo_disparo_dinamico = 0


@st.cache_resource
def load_model():
    model_path = "models/libras_model.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()


@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, model_complexity=0
    )


hands = load_mediapipe()
mp_drawing = mp.solutions.drawing_utils


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


st.title("Reconhecimento de Libras em Tempo Real (Estático + Dinâmico)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Painel de Controle")
    placeholder_letra = st.empty()
    placeholder_texto = st.empty()

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Espaço", use_container_width=True):
            st.session_state.palavra += " "
    with c2:
        if st.button("Apagar", use_container_width=True):
            st.session_state.palavra = st.session_state.palavra[:-1]

    if st.button("Limpar Tudo", type="primary", use_container_width=True):
        st.session_state.palavra = ""

    st.divider()
    threshold = st.slider("Sensibilidade (Confiança)", 0.0, 1.0, 0.60)
    tempo_confirm = st.slider("Tempo Confirm. Estáticas (s)", 0.5, 5.0, 1.2)

with col2:
    st.header("Câmera")
    FRAME_WINDOW = st.image([])
    run = st.toggle("Ativar Reconhecimento", value=True)

if run and model:
    frame_buffer = deque(maxlen=10)

    cols = [f"pos_{i}" for i in range(63)]
    cols += [f"d_total_{i}" for i in range(63)]
    cols += [f"step_{s}_{i}" for s in range(9) for i in range(63)]
    cols += [f"dist_p{p}" for p in range(21)]

    while True:
        success, frame = camera.read()
        if not success:
            st.error("Não foi possível acessar a câmera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        letra_detectada_agora = "..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
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

                if len(frame_buffer) == 10:
                    features = extract_idea2_features(frame_buffer)
                    input_df = pd.DataFrame([features], columns=cols)

                    probs = model.predict_proba(input_df)
                    max_prob = np.max(probs)

                    if max_prob > threshold:
                        letra_detectada_agora = model.predict(input_df)[0]

                        # --- LETRAS DINÂMICAS ---
                        if letra_detectada_agora in LETRAS_DINAMICAS:
                            tempo_desde_ultimo = (
                                time.time()
                                - st.session_state.ultimo_disparo_dinamico
                            )

                            if tempo_desde_ultimo > COOLDOWN_DINAMICO:
                                st.session_state.palavra += (
                                    letra_detectada_agora
                                )
                                st.session_state.ultimo_disparo_dinamico = (
                                    time.time()
                                )
                                st.session_state.letra_atual = ""
                                st.toast(
                                    f"Letra '{letra_detectada_agora}'"
                                    " adicionada!"
                                )

                        # --- LETRAS ESTÁTICAS ---
                        else:
                            if (
                                letra_detectada_agora
                                == st.session_state.letra_atual
                            ):
                                tempo_decorrido = (
                                    time.time() - st.session_state.tempo_inicio
                                )

                                cv2.rectangle(
                                    frame,
                                    (50, 400),
                                    (450, 430),
                                    (255, 255, 255),
                                    2,
                                )
                                progresso = int(
                                    (tempo_decorrido / tempo_confirm) * 400
                                )
                                cv2.rectangle(
                                    frame,
                                    (50, 400),
                                    (50 + min(progresso, 400), 430),
                                    (0, 255, 0),
                                    -1,
                                )

                                if tempo_decorrido >= tempo_confirm:
                                    st.session_state.palavra += (
                                        letra_detectada_agora
                                    )
                                    st.session_state.letra_atual = ""
                                    st.session_state.tempo_inicio = time.time()
                                    st.toast(
                                        f"Letra '{letra_detectada_agora}'"
                                        " adicionada!"
                                    )
                            else:
                                st.session_state.letra_atual = (
                                    letra_detectada_agora
                                )
                                st.session_state.tempo_inicio = time.time()
                    else:
                        st.session_state.letra_atual = ""

        placeholder_letra.metric("Detectando:", letra_detectada_agora)
        placeholder_texto.info(f"### {st.session_state.palavra}")

        FRAME_WINDOW.image(frame, channels="BGR")
        time.sleep(0.01)