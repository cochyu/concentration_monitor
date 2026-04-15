# concentration_monitor
## 2026.04.15   1.0.0
工具及环境：python 3.10.6   VSCord 1.115.0（python插件）
    Claude提供代码指导
具体操作步骤：
### 第一阶段：环境搭建
1.创建并激活虚拟环境
'''
cd Desktop
mkdir concentration_monitor
cd concentration_monitor
# 在桌面创建项目文件夹
python -m venv venv
# 创建虚拟环境
venv\Scripts\activate
# 激活虚拟环境
'''
2.安装核心依赖库
'''
pip install opencv-python mediapipe numpy scikit-learn pillow
# 安装好后可用下面的代码检测是否完成
python -c "import cv2, mediapipe, sklearn; print('所有库安装成功！')"
'''
### 第二阶段：基础测试
1.打开摄像头
'''
python test_camera.py
# 英文键盘Q键关闭
'''
2.人脸关键点检测
'''
python test_mediapipe.py
# 这是整个项目的心脏。MediaPipe 能从摄像头画面里实时找到脸上的 468个关键点，我们的眼部、嘴部、头部姿态特征全部从这里提取
# 成功的话能看到：脸上布满绿色网格点
'''
### 第三阶段：人脸检测+特征提取
关键算法：EAR = Eye Aspect Ratio（眼睛纵横比）
原理：眼睛睁开时，EAR 值约为 0.25~0.35
    眼睛闭合时，EAR 值接近 0.0
    连续多帧 EAR < 0.21 → 判定为眨眼或闭眼疲劳
<img width="680" height="320" alt="ear_eye_diagram" src="https://github.com/user-attachments/assets/9daf178e-6eac-499e-98e6-e52753b84741" />
'''
python test_ear.py
# 结果：画面左上角会实时显示三个数值，眨眼Blinks增加
'''
