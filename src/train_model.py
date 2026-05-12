import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Carregamento de dados

cols = []
for i in range(21):
    cols.extend([f'x{i}', f'y{i}', f'z{i}'])
cols.append('label')

csv_path = 'data/landmarks.csv'

if not os.path.exists(csv_path):
    print(f"Erro: O arquivo {csv_path} não foi encontrado. Realize a coleta primeiro.")
    exit()

df = pd.read_csv(csv_path)

df = df[df['label'] != 'label']

# Preparação de dados

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treinando modelo com {len(X_train)} amostras...")

# Random Forest

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*30)
print(f"ACURÁCIA DO MODELO: {accuracy * 100:.2f}%")
print("="*30)

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvando o model

model_path = 'models/libras_model.pkl'

if not os.path.exists('models'):
    os.makedirs('models')
    
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\nSucesso! Modelo salvo em: {model_path}")