import cv2
import mediapipe as mp
import pickle
import random

# ===== Inicialização =====

with open('modelo_libras.pkl', 'rb') as f:
    modelo = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

letras = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
letra_alvo = random.choice(letras)
pontuacao = 0
acertou = False
contador_confirma = 0

# ===== Funções utilitárias =====

def extrair_pontos(hand_landmarks):
    """Extrai coordenadas x e y dos 21 pontos da mão."""
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]

def mostrar_texto_superior(frame, texto, cor=(0, 255, 0)):
    """Exibe texto no canto superior direito."""
    altura, largura = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    x1, y1 = largura - tw - 40, 20
    x2, y2 = largura - 20, y1 + th + 20
    cv2.rectangle(frame, (x1, y1), (x2, y2), (144, 238, 144), -1)
    cv2.putText(frame, texto, (x1 + 10, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, cor, 3)

def atualizar_letra_alvo():
    """Escolhe uma nova letra alvo aleatória."""
    return random.choice(letras)

# ===== Loop principal =====

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    altura, largura = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    # Verifica se é hora de atualizar a letra alvo
    if acertou:
        contador_confirma += 1
        if contador_confirma > 30:
            acertou = False
            contador_confirma = 0
            letra_alvo = atualizar_letra_alvo()

    # Processa landmarks
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            pontos = extrair_pontos(hand_landmarks)

            if len(pontos) == 42:
                pred = modelo.predict([pontos])[0]
                cv2.putText(frame, f"Letra detectada: {pred}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if pred == letra_alvo and not acertou:
                    pontuacao += 1
                    acertou = True

    # Desenha área de referência
    cv2.rectangle(frame, (100, 180), (300, 390), (0, 255, 0), 2)

    # Mostra instruções e pontuação
    cv2.putText(frame, f"Sinalize: {letra_alvo}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.putText(frame, f"Pontos: {pontuacao}", (10, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if acertou:
        mostrar_texto_superior(frame, "Acertou!", cor=(0, 100, 0))

    cv2.imshow("Jogo de Libras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== Finalização =====

cap.release()
cv2.destroyAllWindows()
