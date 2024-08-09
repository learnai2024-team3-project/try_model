import cv2
import mediapipe as mp
import numpy as np
import time
import json
import tensorflow as tf
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
prevTime = 0

SRC_PATH = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = "1st"
INTEFFACE_ARGS_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/inference_args.json"
MODEL_PATH = SRC_PATH + "/Model/" + BASE_PATH + "/model.tflite"
CHAR_MAP_PATH = SRC_PATH + "/Model/character_to_prediction_index.json"

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


with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh, \
     mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    column_lut = []
    for column in selected_columns:
        column_lut.append([column, 0])
    
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
            res_point = np.array(res_point).reshape(1, len(res_point)).astype(np.float32)
            # 执行预测
            output = prediction_fn(inputs=res_point)
            max_indices = np.argmax(output[REQUIRED_OUTPUT], axis=1)
            prediction_str = "-".join([rev_character_map.get(s, "") for s in max_indices])
            #prediction_str = rev_character_map.get(max_indices[0], "")
        else:
            prediction_str = "Data no enough"
            
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(image, f"Prediction: {prediction_str}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Hands, Pose, and Face', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
