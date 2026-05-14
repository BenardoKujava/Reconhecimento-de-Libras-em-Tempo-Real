import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import warnings

# Warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Libras IA - TCC", layout="wide")

@st.cache_resource
def get_cap():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

# Câmera
camera = get_cap()

# Estados
if 'palavra' not in st.session_state:
    st.session_state.palavra = ""
if 'letra_atual' not in st.session_state:
    st.session_state.letra_atual = ""
if 'tempo_inicio' not in st.session_state:
    st.session_state.tempo_inicio = 0

# Model
@st.cache_resource
def load_model():
    with open('models/libras_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Mediapipe
@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0)

hands = load_mediapipe()
mp_drawing = mp.solutions.drawing_utils

# Interface
st.title("Reconhecimento de Libras em Tempo Real")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Painel de Controle")
    
    placeholder_letra = st.empty()
    
    st.subheader("Texto Formado:")
    placeholder_texto = st.empty()
    
    st.divider()
    
    # Botões
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
    threshold = st.slider("Sensibilidade (Confiança)", 0.0, 1.0, 0.80)
    tempo_confirm = st.slider("Tempo de Confirm. (seg)", 0.5, 5.0, 1.5)

with col2:
    st.header("Câmera")
    FRAME_WINDOW = st.image([]) 

    run = st.toggle('Ativar Reconhecimento', value=True)

# Looping do processamento
if run:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Não foi possível acessar a câmera. Verifique se ela não está sendo usada por outro app.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        letra_detectada_agora = "..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])
                
                input_data = np.array(coords).reshape(1, -1)
                probs = model.predict_proba(input_data)
                max_prob = np.max(probs)

                if max_prob > threshold:
                    letra_detectada_agora = model.predict(input_data)[0]
                    
                    if letra_detectada_agora != "Nada":
                        if letra_detectada_agora == st.session_state.letra_atual:
                            tempo_decorrido = time.time() - st.session_state.tempo_inicio
                            
                            cv2.rectangle(frame, (50, 400), (450, 430), (255, 255, 255), 2)
                            progresso = int((tempo_decorrido / tempo_confirm) * 400)
                            cv2.rectangle(frame, (50, 400), (50 + min(progresso, 400), 430), (0, 255, 0), -1)
                            
                            if tempo_decorrido >= tempo_confirm:
                                st.session_state.palavra += letra_detectada_agora
                                st.session_state.letra_atual = ""
                                st.session_state.tempo_inicio = time.time()
                                st.toast(f"Letra '{letra_detectada_agora}' adicionada!")
                        else:
                            st.session_state.letra_atual = letra_detectada_agora
                            st.session_state.tempo_inicio = time.time()
                else:
                    st.session_state.letra_atual = ""

        # Atualizador da interface
        placeholder_letra.metric("Detectando:", letra_detectada_agora)
        placeholder_texto.info(f"### {st.session_state.palavra}")
        
        # Mostra o vídeo
        FRAME_WINDOW.image(frame, channels="BGR")
        
        time.sleep(0.01)
else:
    st.info("Câmera pausada. Ative o botão acima para retomar.")