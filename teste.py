import pandas as pd

df = pd.read_csv('dados/dados_treinamento.csv')
df = df.dropna()
print(df.head())
print(df.dtypes)
print(f"Número de amostras após limpeza: {len(df)}")