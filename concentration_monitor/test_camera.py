import cv2

print("正在打开摄像头...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 摄像头打开失败！")
else:
    print("✅ 摄像头打开成功！按 Q 退出")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # 镜像翻转
    
    # 在画面左上角显示文字
    cv2.putText(frame, 'Press Q to quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow('Test Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("摄像头已关闭")