# Sistema de Reconhecimento em Tempo Real do Alfabeto em Libras (Estático e Dinâmico)

Este repositório contém a implementação prática do Trabalho de Conclusão de Curso (TCC) focado no **reconhecimento automático em tempo real do alfabeto dactilológico da Língua Brasileira de Sinais (Libras)**, abrangendo tanto as letras **estáticas** (A, B, C...) quanto as **dinâmicas** (H, J, K, X, Y, Z).

O sistema utiliza técnicas avançadas de engenharia de características temporais combinadas com visão computacional leve, permitindo alta acurácia com baixo custo computacional em webcams convencionais e CPUs.

---

## Diferencial Técnico e Arquitetura    

Diferente de abordagens tradicionais baseadas em redes neurais sequenciais pesadas (como LSTMs ou GRUs), o projeto adota uma **modelagem temporal eficiente com Random Forest Classifier**, baseada em uma janela deslizante de 10 quadros ($f_0, f_1, \dots, f_9$).

Para cada amostra, é gerado um vetor de **714 características temporais**:

1. **Configuração Final da Mão ($f_9$):** $63$ coordenadas organizadas e normalizadas $(x, y, z)$.
2. **Deslocamento Global ($\Delta_{\text{total}} = f_9 - f_0$):** $63$ valores do vetor de deslocamento total.
3. **Variações Quadro a Quadro ($\Delta_k = f_{k+1} - f_k$):** $9 \times 63 = 567$ variações temporais intermediárias.
4. **Distância Acumulada por Landmark ($\sum \text{dist}_k$):** $21$ valores representando a energia/velocidade total de movimento de cada ponto da mão.

### Lógica de Reconhecimento Híbrida
* **Sinais Estáticos (A, B, C...):** Utilizam barra de tempo progressiva de confirmação para evitar adições acidentais.
* **Sinais Dinâmicos (H, J, K, X, Y, Z):** Utilizam gatilho imediato com alta confiança e trava temporária de disparo (*cooldown* de 1.5s), contornando a desaceleração natural do gesto no final do movimento.

---

## Tecnologias Utilizadas

* **Linguagem:** Python 3.9+
* **Visão Computacional:** OpenCV, MediaPipe Hands (extratro de 21 landmarks tridimensionais)
* **Aprendizado de Máquina:** Scikit-Learn (`RandomForestClassifier`)
* **Interface do Usuário:** Streamlit
* **Manipulação de Dados:** Pandas, NumPy
* **Persistência de Modelo:** Pickle / Joblib

---

## Estrutura de Arquivos

```text
.
├── app.py                      # Aplicação principal (Interface Streamlit + Visão Computacional)
├── collect_dynamic_idea2.py    # Script unificado para coleta de amostras (estáticas e dinâmicas)
├── train_idea2_model.py        # Script para treinamento e validação do Random Forest
├── data/
│   └── landmarks_idea2.csv     # Dataset unificado com as 714 características por amostra
├── models/
│   └── libras_model.pkl        # Modelo Random Forest treinado e serializado
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação do repositório
```

---

## Como Executar o Projeto

### 1. Pré-requisitos e Instalação

Clone este repositório e instale as dependências necessárias:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Executar a Aplicação em Tempo Real

Para iniciar a interface web e utilizar o sistema via webcam:

```bash
streamlit run app.py
```

---

## Fluxo de Coleta e Treinamento

Caso queira expandir o dataset ou recalibrar o modelo:

### Coletar Novas Amostras:
```bash
python collect_dynamic_idea2.py
```
* **Para Letras Dinâmicas:** Pressione e segure a tecla correspondente enquanto realiza o gesto completo no ar.
* **Para Letras Estáticas:** Mantenha a mão parada no gesto por 1 segundo e pressione a tecla correspondente.

### Treinar o Modelo:
```bash
python train_idea2_model.py
```
O script exibirá a acurácia global, o relatório de classificação por letra e salvará o novo arquivo `models/libras_model.pkl`.

---

## Contexto Acadêmico

* **Trabalho de Conclusão de Curso (TCC)**
* **Tema:** Reconhecimento do Alfabeto Dactilológico Completo em Libras em Tempo Real
* **Padrão de Formatação:** Sociedade Brasileira de Computação (SBC)