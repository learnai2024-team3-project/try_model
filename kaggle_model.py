import cv2
import mediapipe as mp
import numpy as np
import time
import json
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
prevTime = 0

# 加载 TensorFlow Lite 模型和字符映射
interpreter = tf.lite.Interpreter("./Model/model.tflite")

with open("./Model/character_to_prediction_index.json", "r") as f:
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
prediction_fn = interpreter.get_signature_runner("serving_default")

# 定义用于ASL拼写识别的keyXYZ
keyXYZ = [
    "x_right_hand_0", "y_right_hand_0", "z_right_hand_0",
    "x_right_hand_1", "y_right_hand_1", "z_right_hand_1",
    "x_right_hand_2", "y_right_hand_2", "z_right_hand_2",
    "x_right_hand_3", "y_right_hand_3", "z_right_hand_3",
    "x_right_hand_4", "y_right_hand_4", "z_right_hand_4",
    "x_right_hand_5", "y_right_hand_5", "z_right_hand_5",
    "x_right_hand_6", "y_right_hand_6", "z_right_hand_6",
    "x_right_hand_7", "y_right_hand_7", "z_right_hand_7",
    "x_right_hand_8", "y_right_hand_8", "z_right_hand_8",
    "x_right_hand_9", "y_right_hand_9", "z_right_hand_9",
    "x_right_hand_10", "y_right_hand_10", "z_right_hand_10",
    "x_right_hand_11", "y_right_hand_11", "z_right_hand_11",
    "x_right_hand_12", "y_right_hand_12", "z_right_hand_12",
    "x_right_hand_13", "y_right_hand_13", "z_right_hand_13",
    "x_right_hand_14", "y_right_hand_14", "z_right_hand_14",
    "x_right_hand_15", "y_right_hand_15", "z_right_hand_15",
    "x_right_hand_16", "y_right_hand_16", "z_right_hand_16",
    "x_right_hand_17", "y_right_hand_17", "z_right_hand_17",
    "x_right_hand_18", "y_right_hand_18", "z_right_hand_18",
    "x_right_hand_19", "y_right_hand_19", "z_right_hand_19",
    "x_right_hand_20", "y_right_hand_20", "z_right_hand_20",
    "x_left_hand_0", "y_left_hand_0", "z_left_hand_0",
    "x_left_hand_1", "y_left_hand_1", "z_left_hand_1",
    "x_left_hand_2", "y_left_hand_2", "z_left_hand_2",
    "x_left_hand_3", "y_left_hand_3", "z_left_hand_3",
    "x_left_hand_4", "y_left_hand_4", "z_left_hand_4",
    "x_left_hand_5", "y_left_hand_5", "z_left_hand_5",
    "x_left_hand_6", "y_left_hand_6", "z_left_hand_6",
    "x_left_hand_7", "y_left_hand_7", "z_left_hand_7",
    "x_left_hand_8", "y_left_hand_8", "z_left_hand_8",
    "x_left_hand_9", "y_left_hand_9", "z_left_hand_9",
    "x_left_hand_10", "y_left_hand_10", "z_left_hand_10",
    "x_left_hand_11", "y_left_hand_11", "z_left_hand_11",
    "x_left_hand_12", "y_left_hand_12", "z_left_hand_12",
    "x_left_hand_13", "y_left_hand_13", "z_left_hand_13",
    "x_left_hand_14", "y_left_hand_14", "z_left_hand_14",
    "x_left_hand_15", "y_left_hand_15", "z_left_hand_15",
    "x_left_hand_16", "y_left_hand_16", "z_left_hand_16",
    "x_left_hand_17", "y_left_hand_17", "z_left_hand_17",
    "x_left_hand_18", "y_left_hand_18", "z_left_hand_18",
    "x_left_hand_19", "y_left_hand_19", "z_left_hand_19",
    "x_left_hand_20", "y_left_hand_20", "z_left_hand_20",
    "x_pose_13", "y_pose_13", "z_pose_13",
    "x_pose_15", "y_pose_15", "z_pose_15",
    "x_pose_17", "y_pose_17", "z_pose_17",
    "x_pose_19", "y_pose_19", "z_pose_19",
    "x_pose_21", "y_pose_21", "z_pose_21",
    "x_pose_14", "y_pose_14", "z_pose_14",
    "x_pose_16", "y_pose_16", "z_pose_16",
    "x_pose_18", "y_pose_18", "z_pose_18",
    "x_pose_20", "y_pose_20", "z_pose_20",
    "x_pose_22", "y_pose_22", "z_pose_22"
]

# 确保总共156个数据点
print(len(keyXYZ))  # 打印keyXYZ的长度
print(len(keyXYZ) * 3)  # 打印keyXYZ的总数据点数

# cap = cv2.VideoCapture("ABC.mp4")
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, \
     mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image)
        pose_results = pose.process(image)

        res_point = []  # 清空 res_point
        hand_points = []  # 用于存储手部关键点以绘制
        pose_points = []  # 用于存储姿势关键点以绘制

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    res_point.append(landmark.x)
                    res_point.append(landmark.y)
                    res_point.append(landmark.z)
                    hand_points.append((landmark.x, landmark.y))

        if pose_results.pose_landmarks:
            for idx in [13, 15, 17, 19, 21, 14, 16, 18, 20, 22]:  # 获取与手部相关的姿势标志点
                landmark = pose_results.pose_landmarks.landmark[idx]
                res_point.append(landmark.x)
                res_point.append(landmark.y)
                res_point.append(landmark.z)
                pose_points.append((landmark.x, landmark.y))
                
        prediction_str = ""

        # 确保res_point的长度为156
        if len(res_point) == 156:
            res_point = np.array(res_point).reshape(1, len(res_point)).astype(np.float32)
            
            # 执行预测
            output = prediction_fn(inputs=res_point)
            prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
            print(f"Prediction: {prediction_str}")

            cv2.putText(image, f"Prediction: {prediction_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制手部关键点（绿色）
        h, w, _ = image.shape
        for point in hand_points:
            x = int(point[0] * w)
            y = int(point[1] * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        # 绘制姿势关键点（红色）
        for point in pose_points:
            x = int(point[0] * w)
            y = int(point[1] * h)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

        # 将识别结构绘制在右上角
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), (255, 255, 255), -1)
        alpha = 0.6  # 透明度
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.putText(image, f"Prediction: {prediction_str}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        cv2.imshow('MediaPipe Hands and Pose', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()