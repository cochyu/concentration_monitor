import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import time
from collections import deque

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
    nose  = np.array([landmarks[1].x * w,   landmarks[1].y * h])
    chin  = np.array([landmarks[152].x * w, landmarks[152].y * h])
    l_eye = np.array([landmarks[33].x * w,  landmarks[33].y * h])
    r_eye = np.array([landmarks[263].x * w, landmarks[263].y * h])
    face_h = np.linalg.norm(chin - nose)
    pitch  = (nose[1] - l_eye[1]) / (face_h + 1e-6)
    dist_l = nose[0] - l_eye[0]
    dist_r = r_eye[0] - nose[0]
    yaw    = (dist_l - dist_r) / (dist_l + dist_r + 1e-6)
    return pitch, yaw

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, -1)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r),
                   (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

# ── 关键点索引 ────────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_8   = [61, 37, 0, 267, 291, 314, 17, 84]

# ── 加载模型 ──────────────────────────────────
model  = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")
print("✅ 模型加载成功")

# ── 初始化MediaPipe ───────────────────────────
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── 状态配置 ──────────────────────────────────
LABELS  = {0: "Focused", 1: "Distracted", 2: "Fatigue"}
COLORS  = {0: (0,200,80), 1: (0,140,255), 2: (0,0,220)}
SMOOTHING = 10  # 平滑窗口，避免结果跳动

# ── 统计变量 ──────────────────────────────────
pred_buffer    = deque(maxlen=SMOOTHING)
focused_time   = 0.0
distract_time  = 0.0
fatigue_time   = 0.0
alert_msg      = ""
alert_until    = 0
last_time      = time.time()
start_time     = time.time()

cap = cv2.VideoCapture(0)
fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("✅ 系统启动！按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    now      = time.time()
    dt       = now - last_time
    last_time = now

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_img)

    current_label = 0  # 默认专注

    if results.face_landmarks:
        lms = results.face_landmarks[0]

        ear        = (calc_ear(lms, LEFT_EYE, fw, fh) +
                      calc_ear(lms, RIGHT_EYE, fw, fh)) / 2.0
        mar        = calc_mar(lms, MOUTH_8, fw, fh)
        pitch, yaw = calc_head_pose(lms, fw, fh)

        # SVM预测
        feat   = scaler.transform([[ear, mar, pitch, yaw]])
        pred   = model.predict(feat)[0]
        pred_buffer.append(pred)

        # 平滑：取最近N帧的众数
        current_label = max(set(pred_buffer), key=list(pred_buffer).count)

        # 累计时间
        if current_label == 0:
            focused_time  += dt
        elif current_label == 1:
            distract_time += dt
        else:
            fatigue_time  += dt

        # 警报逻辑
        if fatigue_time > 10 and now > alert_until:
            alert_msg   = "Please rest!"
            alert_until = now + 5
        if distract_time > 15 and now > alert_until:
            alert_msg   = "Stay focused!"
            alert_until = now + 5

    else:
        current_label = 1  # 没检测到脸算分心
        distract_time += dt

    # ── 绘制UI ────────────────────────────────
    label  = LABELS[current_label]
    color  = COLORS[current_label]
    total  = max(focused_time + distract_time + fatigue_time, 1)

    # 顶部状态栏
    draw_rounded_rect(frame, 10, 10, 220, 44, 8, (30,30,30), 0.7)
    cv2.putText(frame, label, (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 右侧统计面板
    draw_rounded_rect(frame, fw-200, 10, 188, 120, 8, (30,30,30), 0.7)
    elapsed = int(now - start_time)
    cv2.putText(frame, f"Time: {elapsed//60:02d}:{elapsed%60:02d}",
                (fw-188, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(frame, f"Focused  {focused_time/total*100:.0f}%",
                (fw-188, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,80), 1)
    cv2.putText(frame, f"Distract {distract_time/total*100:.0f}%",
                (fw-188, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 1)
    cv2.putText(frame, f"Fatigue  {fatigue_time/total*100:.0f}%",
                (fw-188, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 1)
    cv2.putText(frame, f"EAR:{(calc_ear(results.face_landmarks[0], LEFT_EYE, fw, fh) if results.face_landmarks else 0):.2f}" ,
                (fw-188, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # 警报横幅
    if now < alert_until:
        draw_rounded_rect(frame, fw//2-160, fh-70, 320, 50, 10, (0,0,180), 0.85)
        cv2.putText(frame, alert_msg, (fw//2-140, fh-36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    cv2.imshow("Concentration Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 退出统计 ──────────────────────────────────
cap.release()
cv2.destroyAllWindows()
total = max(focused_time + distract_time + fatigue_time, 1)
print(f"\n===== 本次会话报告 =====")
print(f"总时长:  {int(total)//60:02d}:{int(total)%60:02d}")
print(f"专注:    {focused_time/total*100:.1f}%")
print(f"分心:    {distract_time/total*100:.1f}%")
print(f"疲劳:    {fatigue_time/total*100:.1f}%")