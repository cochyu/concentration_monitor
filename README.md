# concentration_monitor

> **版本：** 1.0.0  
> **更新日期：** 2026.04.15

本项目是一个基于计算机视觉的专注度/疲劳监测系统，通过分析人脸关键点实现实时检测。

---

## 🛠️ 工具及环境
* **Python**: 3.10.6
* **IDE**: VS Code 1.115.0 (含 Python 插件)
* **AI 辅助**: 由 Claude 提供代码架构指导

---

## 🚀 具体操作步骤

### 第一阶段：环境搭建

1. **创建并激活虚拟环境**
   ```bash
   # 进入桌面并创建项目文件夹
   cd Desktop
   mkdir concentration_monitor
   cd concentration_monitor

   # 创建虚拟环境
   python -m venv venv

   # 激活虚拟环境 (Windows)
   venv\Scripts\activate
   ```

2. **安装核心依赖库**
   ```bash
   pip install opencv-python mediapipe numpy scikit-learn pillow

   # 安装完成后校验
   python -c "import cv2, mediapipe, sklearn; print('✅ 所有库安装成功！')"
   ```

---

### 第二阶段：基础测试

1. **打开摄像头测试**
   ```bash
   python test_camera.py
   ```
   > **操作说明**：确保摄像头正常调用，英文键盘按下 `Q` 键可退出。

2. **人脸关键点检测**
   ```bash
   python test_mediapipe.py
   ```
   * **核心逻辑**：本项目的心脏。MediaPipe 能实时定位脸部 **468 个关键点**。
   * **特征提取**：眼部、嘴部、头部姿态特征均由此处提取。
   * **预期结果**：画面中人脸布满**绿色网格点**。

---

### 第三阶段：人脸检测 + 特征提取

#### 💡 关键算法：EAR (Eye Aspect Ratio)
**算法原理：**
* **眼睛睁开**：EAR 值约为 `0.25 ~ 0.35`
* **眼睛闭合**：EAR 值接近 `0.0`
* **疲劳判定**：当连续多帧 `EAR < 0.21` 时，判定为眨眼或闭眼疲劳。



<img width="680" alt="ear_eye_diagram" src="https://github.com/user-attachments/assets/9daf178e-6eac-499e-98e6-e52753b84741" />

**运行测试：**
```bash
python test_ear.py
```
* **结果反馈**：画面左上角实时显示数值变化，眨眼时 `Blinks` 计数增加。

#### MAR+头部姿态
 ```bash
   python test_mediapipe.py
   ```
**测试方法：**
* **正常看摄像头**：显示FOCUSED
* **故意张大嘴巴保持几秒**：Yawns 数字增加
* **低下头**：显示 HEAD DOWN
* **快速眨眼15次**：触发 FATIGUE DETECTED

### 第四阶段：训练AI模型

1. **采集训练数据**
   ```bash
   python collect_data.py
   ```

**训练方法：**
| 按键 | 动作   | 怎么配合                          |
|------|--------|-----------------------------------|
| F    | 专注   | 正常坐好看摄像头，按 F 开始录    |
| D    | 分心   | 低头/转头，按 D 开始录           |
| T    | 疲劳   | 故意眯眼/张嘴，按 T 开始录       |
| S    | 停止   | 换状态前先按 S 停止              |
| Q    | 保存退出 | 采集够了按 Q                     |
 > **提示**：尽量多录制采集，每种状态至少采集200条。


