import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# ── EAR 计算函数 ──────────────────────────────
def calculate_ear(landmarks, eye_indices, w, h):
    # 取6个关键点坐标
    p = []
    for i in eye_indices:
        lm = landmarks[i]
        p.append((lm.x * w, lm.y * h))

    # 计算两条垂直距离
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    # 计算水平距离
    h1 = np.linalg.norm(np.array(p[0]) - np.array(p[3]))

    return (v1 + v2) / (2.0 * h1)

# MediaPipe 468个关键点中，眼部对应的索引
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21   # 低于此值判定为闭眼
BLINK_FRAMES  = 2      # 连续几帧闭眼算一次眨眼

# ── 初始化 MediaPipe ──────────────────────────
model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── 状态变量 ──────────────────────────────────
blink_count   = 0
closed_frames = 0

cap = cv2.VideoCapture(0)
print("✅ EAR检测启动！按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_img)

    if results.face_landmarks:
        lms = results.face_landmarks[0]

        left_ear  = calculate_ear(lms, LEFT_EYE,  w, h)
        right_ear = calculate_ear(lms, RIGHT_EYE, w, h)
        ear       = (left_ear + right_ear) / 2.0

        # 眨眼计数逻辑
        if ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if closed_frames >= BLINK_FRAMES:
                blink_count += 1
            closed_frames = 0

        # 判断状态
        if ear < EAR_THRESHOLD:
            status = "CLOSED"
            color  = (0, 0, 255)    # 红色
        else:
            status = "OPEN"
            color  = (0, 255, 0)    # 绿色

        # 显示信息
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Eye: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("EAR Blink Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()