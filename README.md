
# Projeto: Leitura de Libras com Visão Computacional

Este projeto utiliza **Python**, **OpenCV**, **MediaPipe** e **scikit-learn** para reconhecer sinais da Língua Brasileira de Sinais (LIBRAS) capturados pela webcam. Além do reconhecimento em tempo real, o sistema inclui um **modo de jogo simples** onde o usuário tenta acertar letras exibidas na tela com os sinais corretos.

## 📁 Estrutura do Projeto

```
📦 Projeto/
├── dados/                    # Contém arquivos CSV com dados coletados
├── coleta_dados.py          # Script para capturar dados de um gesto específico
├── juntar_dados.py          # Une todos os arquivos CSV em um só para treinamento
├── treino_modelo.py         # Treina o modelo de machine learning
├── reconhecimento_tempo_real.py # Executa o reconhecimento em tempo real com webcam
├── modelo_libras.pkl        # Arquivo gerado com o modelo treinado
├── label_encoder.pkl        # Arquivo com os rótulos codificados
└── README.md                # Documentação do projeto
```

## 🧠 Tecnologias Utilizadas

- Python 3.10+
- OpenCV
- MediaPipe
- Scikit-learn
- Pandas
- NumPy

## 🧪 Instalação das Dependências

Você pode instalar todas as bibliotecas necessárias com o seguinte comando:

```bash
pip install pandas numpy opencv-python mediapipe scikit-learn
```

## 🚀 Como Executar o Projeto

### 🎮 Modo de Jogo (jogo_libras.py)

- Exibe uma **letra-alvo** que o usuário deve representar com as mãos.
- Uma **caixa verde** na tela indica onde fazer o gesto.
- Se o gesto for reconhecido corretamente:
  - A pontuação é incrementada.
  - Uma confirmação **visual no canto superior direito** aparece com a mensagem "Acertou!" em verde.
  - Uma nova letra-alvo é gerada.

### ▶️ Como executar

1. Certifique-se de que o arquivo `modelo_libras.pkl` (modelo treinado) esteja no mesmo diretório.
2. Execute o modo jogo com:

```bash
python jogo_libras.py
```

3. Pressione **Q** para sair.

---
## 💪 Treinando o modelo

### 1. Coleta de Dados
Execute o script abaixo e pressione:
- `c` para iniciar a coleta.
- `q` para sair.

```bash
python coleta_dados.py
```

Os dados serão salvos no diretório `dados/`.

### 2. Juntar os Dados
Após coletar várias letras (como `dados_letra_A.csv`, `dados_letra_B.csv`, etc.), execute:

```bash
python juntar_dados.py
```

Isso criará um arquivo único chamado `dados_treinamento.csv`.

### 3. Treinar o Modelo
Para treinar o classificador com KNN:

```bash
python treino_modelo.py
```

Esse processo gerará os arquivos `modelo_libras.pkl` e `label_encoder.pkl`.

### 4. Reconhecimento em Tempo Real

```bash
python reconhecimento_tempo_real.py
```

Uma caixa verde será exibida na tela indicando a área onde o gesto deve ser feito. O modelo tentará reconhecer a letra correspondente ao gesto da mão em tempo real.

## ✅ Status

✅ Modo de jogo  
✅ Coleta de dados  
✅ Treinamento de modelo  
✅ Reconhecimento em tempo real  

## 📌 Observações

- Certifique-se de que a mão esteja bem visível na área indicada da webcam.
- O projeto reconhece apenas as letras que foram treinadas.


