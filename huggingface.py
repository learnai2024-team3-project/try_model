from transformers import pipeline
import os
from PIL import Image

model_name = "RavenOnur/Sign-Language"
pipe = pipeline("image-classification", model=model_name)

import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 啟動相機
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("忽略空幀")
            continue

        # 翻轉影像水平，這樣鏡像效果
        image = cv2.flip(image, 1)
        # 將影像從 BGR 轉換到 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 處理影像並偵測手部
        results = hands.process(image_rgb)

        # 繪製手部標誌
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 獲取邊界框座標
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                # 將相對座標轉換為絕對座標
                h, w, _ = image.shape
                x_min = int(x_min * w) - 30
                x_max = int(x_max * w) + 30
                y_min = int(y_min * h) - 30
                y_max = int(y_max * h) + 30

                # 繪製矩形
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (243, 6, 242), 2)
        
        image_pil = Image.fromarray(image)

        prediction = pipe(image_pil)
        print(prediction)
        most_probable = prediction[0]['label']
        
        cv2.putText(image, most_probable, (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)    
        cv2.imshow('MediaPipe Hands', image)
            
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
