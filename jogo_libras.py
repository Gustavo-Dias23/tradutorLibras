import cv2
import mediapipe as mp
import pickle
import random
import numpy as np

# Carrega modelo treinado
with open('modelo_libras.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Inicializa câmera
cap = cv2.VideoCapture(0)

letras = list("ABCDEFGHIJKLMNNOPQRSTUVWXYZ")
letra_alvo = random.choice(letras)
pontuacao = 0
acertou = False
contador_confirma = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)

    if acertou:
        contador_confirma += 1
        if contador_confirma > 30:
            acertou = False
            contador_confirma = 0
            letra_alvo = random.choice(letras)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y])

            if len(pontos) == 42:
                pred = modelo.predict([pontos])[0]
                cv2.putText(frame, f"Letra detectada: {pred}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if pred == letra_alvo and not acertou:
                    pontuacao += 1
                    acertou = True

    # Desenha retângulo de referência
    cv2.rectangle(frame, (100, 180), (300, 390), (0, 255, 0), 2)

    # Mostra letra alvo
    cv2.putText(frame, f"Sinalize: {letra_alvo}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Mostra pontuação no canto inferior esquerdo
    altura, largura = frame.shape[:2]
    cv2.putText(frame, f"Pontos: {pontuacao}", (10, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Retângulo e texto de acerto no canto superior direito
    if acertou:
        texto = "Acertou!"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x1, y1 = largura - tw - 40, 20
        x2, y2 = largura - 20, 20 + th + 20
        cv2.rectangle(frame, (x1, y1), (x2, y2), (144, 238, 144), -1)
        cv2.putText(frame, texto, (x1 + 10, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 100, 0), 3)

    cv2.imshow("Jogo de Libras", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()