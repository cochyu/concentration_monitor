import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# ── 计算函数 ──────────────────────────────────
def calc_ear(landmarks, indices, w, h):
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    h1 = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h1)

def calc_mar(landmarks, indices, w, h):
    p = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[7]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[6]))
    v3 = np.linalg.norm(np.array(p[3]) - np.array(p[5]))
    h1 = np.linalg.norm(np.array(p[0]) - np.array(p[4]))
    return (v1 + v2 + v3) / (2.0 * h1)

def calc_head_pose(landmarks, w, h):
    # 用鼻尖、下巴、左右眼角、左右嘴角估算头部角度
    nose   = np.array([landmarks[1].x * w,   landmarks[1].y * h])
    chin   = np.array([landmarks[152].x * w, landmarks[152].y * h])
    l_eye  = np.array([landmarks[33].x * w,  landmarks[33].y * h])
    r_eye  = np.array([landmarks[263].x * w, landmarks[263].y * h])

    # Pitch（低头/抬头）：鼻子和下巴的垂直比例
    face_h  = np.linalg.norm(chin - nose)
    pitch   = (nose[1] - l_eye[1]) / (face_h + 1e-6)

    # Yaw（左右偏头）：鼻尖到左右眼的水平距离差
    dist_l  = nose[0] - l_eye[0]
    dist_r  = r_eye[0] - nose[0]
    yaw     = (dist_l - dist_r) / (dist_l + dist_r + 1e-6)

    return pitch, yaw

# ── 关键点索引 ────────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 181]
# 简化嘴巴索引（8个点）
MOUTH_8   = [61, 37, 0, 267, 291, 314, 17, 84]

# ── 阈值设定 ──────────────────────────────────
EAR_THRESH   = 0.21
MAR_THRESH   = 0.6   # 嘴巴张大到这个值判定为哈欠
PITCH_THRESH = 0.15  # 低头阈值
BLINK_FRAMES = 2
YAWN_FRAMES  = 15    # 哈欠持续帧数

# ── 初始化 ────────────────────────────────────
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
blink_count    = 0
yawn_count     = 0
closed_frames  = 0
yawn_frames    = 0
nod_start      = None   # 低头开始时间

cap = cv2.VideoCapture(0)
print("✅ 全特征检测启动！按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_img)

    if results.face_landmarks:
        lms = results.face_landmarks[0]

        # 计算三个特征值
        ear        = (calc_ear(lms, LEFT_EYE, fw, fh) +
                      calc_ear(lms, RIGHT_EYE, fw, fh)) / 2.0
        mar        = calc_mar(lms, MOUTH_8, fw, fh)
        pitch, yaw = calc_head_pose(lms, fw, fh)

        # 眨眼计数
        if ear < EAR_THRESH:
            closed_frames += 1
        else:
            if closed_frames >= BLINK_FRAMES:
                blink_count += 1
            closed_frames = 0

        # 哈欠计数
        if mar > MAR_THRESH:
            yawn_frames += 1
        else:
            if yawn_frames >= YAWN_FRAMES:
                yawn_count += 1
            yawn_frames = 0

        # 疲劳判断
        fatigue = (blink_count > 15 or yawn_count >= 2
                   or closed_frames > 30)

        # 显示特征值
        cv2.putText(frame, f"EAR:{ear:.2f}  MAR:{mar:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        cv2.putText(frame, f"Pitch:{pitch:.2f}  Yaw:{yaw:.2f}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        cv2.putText(frame, f"Blinks:{blink_count}  Yawns:{yawn_count}",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 1)

        # 状态提示
        if fatigue:
            cv2.putText(frame, "⚠ FATIGUE DETECTED",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
        elif pitch < -PITCH_THRESH:
            cv2.putText(frame, "HEAD DOWN",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "FOCUSED",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Feature Extraction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()