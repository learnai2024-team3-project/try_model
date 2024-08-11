import cv2
import mediapipe as mp
import numpy as np
import time
import json
import tensorflow as tf
import os
from mediapipe_functions import *
import pandas as pd
import streamlit as st

SRC_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = "1st"
INTEFFACE_ARGS_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/inference_args.json"
MODEL_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/model.tflite"
CHAR_MAP_PATH = SRC_PATH + "/Model/character_to_prediction_index.json"
TEST_VIDEO_PATH = SRC_PATH + "/Test/"

cap = cv2.VideoCapture(TEST_VIDEO_PATH + "goodbye.mp4")
# cap = cv2.VideoCapture(0)

final_landmarks=[]
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image, results = mediapipe_detection(frame, holistic)
        draw(image, results)
        landmarks = extract_coordinates(results)
        final_landmarks.extend(landmarks)
    
df1 = pd.DataFrame(final_landmarks, columns=['x','y','z'])

ROWS_PER_FRAME = 543
def load_relevant_data_subset(df):
    data_columns = ['x', 'y', 'z']
    data = df
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

test_df = load_relevant_data_subset(df1)
test_df = tf.convert_to_tensor(test_df)

interpreter = tf.lite.Interpreter(MODEL_PATH)
interpreter.allocate_tensors()
prediction_fn = interpreter.get_signature_runner("serving_default")

output = prediction_fn(inputs=test_df)

sign = np.argmax(output["outputs"])
sign_json = pd.read_json(CHAR_MAP_PATH, typ='series')
sign_df = pd.DataFrame(sign_json)
pred = sign_df.iloc[sign]

st.write(pred)
top_indices = np.argsort(output['outputs'])[::-1][:5]
top_values = output['outputs'][top_indices]
    
output_df = sign_df.iloc[top_indices]
output_df['Value'] = top_values

output_df.rename(columns = {0:'Index'}, inplace = True)
output_df.drop(['Index'],1, inplace=True)
st.write(output_df)
