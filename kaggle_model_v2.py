# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# import json
# import tensorflow as tf
# import os
# from mediapipe_functions import *
# import pandas as pd
# import streamlit as st

# SRC_PATH = os.path.dirname(os.path.abspath(__file__))
# BASE_PATH = "asl"
# INTEFFACE_ARGS_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/inference_args.json"
# MODEL_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/model.tflite"
# CHAR_MAP_PATH = SRC_PATH + "/Model/character_to_prediction_index.json"
# TEST_VIDEO_PATH = SRC_PATH + "/Test/"

# cap = cv2.VideoCapture(TEST_VIDEO_PATH + "goodbye.mp4")
# # cap = cv2.VideoCapture(0)

# final_landmarks=[]
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         image, results = mediapipe_detection(frame, holistic)
#         landmarks = extract_coordinates(results)
#         final_landmarks.extend(landmarks)
    
# df1 = pd.DataFrame(final_landmarks, columns=['x','y','z'])

# ROWS_PER_FRAME = 543
# def load_relevant_data_subset(df):
#     data_columns = ['x', 'y', 'z']
#     data = df
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)

# test_df = load_relevant_data_subset(df1)
# test_df = tf.convert_to_tensor(test_df)

# interpreter = tf.lite.Interpreter(MODEL_PATH)
# interpreter.allocate_tensors()
# prediction_fn = interpreter.get_signature_runner("serving_default")
# print("Input shape:", test_df.shape)

# input_details = interpreter.get_input_details()
# print("Model expects input shape:", input_details[0]['shape'])

# output = prediction_fn(inputs=test_df)

# sign = np.argmax(output["outputs"])
# sign_json = pd.read_json(CHAR_MAP_PATH, typ='series')
# sign_df = pd.DataFrame(sign_json)
# pred = sign_df.iloc[sign]

# st.write(pred)
# top_indices = np.argsort(output['outputs'])[::-1][:5]
# top_values = output['outputs'][top_indices]
    
# output_df = sign_df.iloc[top_indices]
# output_df['Value'] = top_values

# output_df.rename(columns = {0:'Index'}, inplace = True)
# output_df.drop(['Index'],1, inplace=True)
# st.write(output_df)

import cv2
import mediapipe as mp
import numpy as np
import time
import json
import tensorflow as tf
import os
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
prevTime = 0

SRC_PATH = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = "asl"
INTEFFACE_ARGS_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/inference_args.json"
MODEL_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/model.tflite"
CHAR_MAP_PATH = SRC_PATH + "/Model/character_to_prediction_index.json"
TEST_VIDEO_PATH = SRC_PATH + "/Test/"

MAX_FRAMES = 10

# 读取 JSON 文件中的特征
with open(INTEFFACE_ARGS_PATH, 'r') as f:
    inference_args = json.load(f)

selected_columns = inference_args["selected_columns"]

selected_columns_temp = inference_args.get("selected_columns", [])

# 加载 TensorFlow Lite 模型和字符映射
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

with open(CHAR_MAP_PATH, "r") as f:
    character_map = json.load(f)
rev_character_map = {j: i for i, j in character_map.items()}

# 确保模型中包含所需的签名
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "outputs"

found_signatures = list(interpreter.get_signature_list().keys())

if REQUIRED_SIGNATURE not in found_signatures:
    raise Exception('Required input signature not found.')

# 准备模型进行推理
interpreter.allocate_tensors()

# 获取推理函数
prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)

# 打印 selected_columns 的长度
print(len(selected_columns))  # 应该输出390
res_point_size = len(selected_columns)

# cap = cv2.VideoCapture(TEST_VIDEO_PATH + "asl.mp4")
cap = cv2.VideoCapture(0)
prediction_str = ""

def set_column_data(colum_name, idx, x, y, z):
    name_x = f"x_{colum_name}_{idx}"
    name_y = f"y_{colum_name}_{idx}"
    name_z = f"z_{colum_name}_{idx}"
    
    for i, column in enumerate(column_lut):
        if column[0] == name_x:
            column_lut[i][1] = x
            #print(f"set {column}")
        elif column[0] == name_y:
            column_lut[i][1] = y
            #print(f"set {column}")
        elif column[0] == name_z:
            column_lut[i][1] = z
            #print(f"set {column}")
            
# 直接使用 mp_drawing.DrawingSpec 来定义绘制风格
face_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
face_connection_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
hand_connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

pose_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
pose_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)

top_predictions = None
frames = []

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh, \
     mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image)
        hand_results = hands.process(image)
        pose_results = pose.process(image)

        res_point = []
        
        column_lut = []
        for column in selected_columns:
            column_lut.append([column, np.nan])

        # 填充脸部数据
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_landmark_style,
                    connection_drawing_spec=face_connection_style)
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    set_column_data("face", idx,
                                    landmark.x,
                                    landmark.y,
                                    landmark.z)
                
        # 填充左手数据
        # 填充右手数据
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                if hand_label == "Left":
                    hand_name = "left_hand"
                else:
                    hand_name = "right_hand"
            
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    set_column_data(hand_name, idx,
                                    landmark.x,
                                    landmark.y,
                                    landmark.z)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_style,
                    connection_drawing_spec=hand_connection_style)
        
        # 填充姿态数据
        if pose_results.pose_landmarks:
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                set_column_data("pose", idx,
                                    landmark.x,
                                    landmark.y,
                                    landmark.z)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)
                
        #print(column_lut)
        
        for i, column in enumerate(column_lut):
            res_point.append(column_lut[i][1])

        print(len(res_point))
        
        # 确保res_point的长度为res_point_size
        if len(res_point) == res_point_size:
            frames.append(res_point)
            
        if len(frames) == MAX_FRAMES:
            frames = np.array(frames).reshape(MAX_FRAMES, res_point_size).astype(np.float32)
            test_df = tf.convert_to_tensor(frames)
            output = prediction_fn(inputs=test_df)

            if len(output[REQUIRED_OUTPUT]) > 0:
                top_indices = np.argsort(output[REQUIRED_OUTPUT], axis=1)[:, -5:][0][::-1]
                top_values = output[REQUIRED_OUTPUT][0][top_indices]

                prediction_str = "-".join([rev_character_map.get(s, "") for s in top_indices])
                top_predictions = [(rev_character_map.get(s, ""), top_values[i]) for i, s in enumerate(top_indices)]
            else:
                prediction_str = ""
                top_predictions = None
                
            frames = []
            
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        if top_predictions != None:
            for i, (pred, prob) in enumerate(top_predictions):
                cv2.putText(image, f"Top {i+1}: {pred} - {prob:.2f}", (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        cv2.putText(image, f"Prediction: {prediction_str} {len(frames)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 10, 255), 2)
        
        cv2.imshow('MediaPipe Hands, Pose, and Face', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()