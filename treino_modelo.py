import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Carrega os dados
df = pd.read_csv('dados/dados_treinamento.csv')

# Converte a última coluna para string
df[df.columns[-1]] = df[df.columns[-1]].astype(str)

# Converte coordenadas para float e remove dados inválidos
for col in df.columns[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

# Separa coordenadas (X) e rótulos (y)
X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1].astype(str)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treina o modelo
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# Avalia
print(f"Acurácia: {modelo.score(X_test, y_test):.2f}")

# Salva o modelo
with open('modelo_libras.pkl', 'wb') as f:
    pickle.dump(modelo, f)
