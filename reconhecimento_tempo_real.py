import cv2
import mediapipe as mp
import pickle
import numpy as np

with open('modelo_libras.pkl', 'rb') as f:
    modelo = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(rgb)
    cv2.rectangle(frame, (100, 180), (300, 390), (0, 255, 0), 2)

    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            pontos = []
            for lm in hand_landmarks.landmark:
                pontos.extend([lm.x, lm.y])
            
            if len(pontos) == 42:
                pred = modelo.predict([pontos])[0]
                cv2.putText(frame, f'Letra: {pred}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Reconhecimento em tempo real", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
