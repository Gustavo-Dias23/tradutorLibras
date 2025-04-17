import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Nome do arquivo de saída
arquivo_csv = 'dados/dados_letra_L.csv'

# Abre a webcam
cap = cv2.VideoCapture(0)

coletando = False
contador = 0
total_amostras = 100  # Quantas amostras deseja coletar

with open(arquivo_csv, mode='w', newline='') as f:
    escritor_csv = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)
        cv2.rectangle(frame, (100, 180), (300, 390), (0, 255, 0), 2)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if coletando:
                    dados = []
                    for ponto in hand_landmarks.landmark:
                        dados.extend([ponto.x, ponto.y])
                    dados.append('L')  # rótulo da letra
                    escritor_csv.writerow(dados)
                    contador += 1
                    time.sleep(0.1)  # pequeno delay entre coletas

        cv2.putText(frame, f'Amostras coletadas: {contador}/{total_amostras}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Coletando dados - Letra", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            coletando = True
        elif key == ord('q') or contador >= total_amostras:
            break

cap.release()
cv2.destroyAllWindows()
