import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

csv_path = "data/landmarks_idea2.csv"
model_path = "models/libras_model.pkl"

if not os.path.exists(csv_path):
    print(f"Erro: {csv_path} não encontrado.")
    exit()

df = pd.read_csv(csv_path)
df = df[df["label"] != "label"].copy()

X = df.drop("label", axis=1)
y = df["label"]

print(
    f"Treinando modelo do Projeto com {len(df)} amostras (714 features"
    " temporais)..."
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n==============================")
print(f"ACURÁCIA DO MODELO: {acc * 100:.2f}%")
print(f"==============================\n")
print(classification_report(y_test, y_pred))

if not os.path.exists("models"):
    os.makedirs("models")

with open(model_path, "wb") as f:
    pickle.dump(clf, f)

print(f"Modelo salvo em: {model_path}")