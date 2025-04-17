import pandas as pd
import os

colunas = [f'x{i}' if i % 2 == 0 else f'y{i//2}' for i in range(42)]
colunas.append('label')

pastas = 'dados'
arquivos = [f for f in os.listdir(pastas) if f.endswith('.csv')]

df_final = pd.DataFrame(columns=colunas)

for arquivo in arquivos:
    caminho = os.path.join(pastas, arquivo)
    df = pd.read_csv(caminho, header=None, usecols=range(43), dtype=str)
    df.columns = colunas
    df_final = pd.concat([df_final, df], ignore_index=True)

df_final.to_csv(os.path.join(pastas, 'dados_treinamento.csv'), index=False)
print("Todos os dados foram unificados com sucesso!")
