# 🎭 人脸情绪识别与脸型分析系统

基于计算机视觉和深度学习的人脸分析系统，能够实时识别人物情绪状态和脸型特征。

## ✨ 功能特点

### 情绪识别 (8种情绪)
- 😄 **开心** - 检测笑容程度和面部动作
- 😢 **悲伤** - 识别嘴角下垂、眉毛上扬
- 😠 **愤怒** - 检测眉毛压低、眼神变化
- 😲 **惊讶** - 识别睁大的眼睛和张开的嘴巴
- 😐 **平静** - 中性表情识别
- 😨 **恐惧** - 检测恐惧相关的面部特征
- 🤢 **厌恶** - 识别厌恶表情
- 😕 **困惑** - 检测眉毛不对称等特征

### 脸型分析 (8种脸型)
- 👤 **椭圆形** - 理想脸型，线条柔和
- 🔴 **圆形** - 脸长和脸宽相近
- ⬜ **方形** - 棱角分明，下颌线条硬朗
- 📏 **长方形** - 脸长明显大于脸宽
- 💗 **心形** - 额头宽，下巴尖细
- 💎 **菱形** - 颧骨最宽，额头和下巴较窄
- 🔺 **三角形** - 下巴宽，额头较窄
- 📐 **长形** - 脸长比例特别大

### 系统特性
- 📹 实时摄像头分析
- 📸 拍照分析模式
- 🔄 实时自动分析模式
- 📊 详细的情绪分布可视化
- 📏 面部测量数据分析
- 💡 个性化发型、眼镜、妆容建议
- 🌐 美观的Web界面

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 摄像头设备
- 现代浏览器 (Chrome/Firefox/Edge)

### 安装步骤

1. **克隆/下载项目**
```bash
cd face_analyzer
```

2. **创建虚拟环境** (推荐)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **启动应用**
```bash
python app.py
```

5. **访问应用**
打开浏览器访问: http://localhost:5000

## 📖 使用说明

### 基本使用流程

1. **启动摄像头**
   - 点击"开启摄像头"按钮
   - 允许浏览器访问摄像头权限

2. **进行分析**
   - **拍照模式**: 点击"拍照分析"按钮
   - **实时模式**: 打开"实时分析模式"开关

3. **查看结果**
   - 情绪分析结果 (主要情绪 + 8种情绪分布)
   - 脸型分析结果 (脸型类型 + 特征描述)
   - 面部测量数据
   - 个性化建议 (发型/眼镜/妆容)

### 快捷键
- `空格键` - 拍照分析

## 📁 项目结构

```
face_analyzer/
├── app.py                 # 主应用程序 (Flask)
├── face_detector.py       # 人脸检测模块
├── emotion_analyzer.py   # 情绪分析模块
├── face_shape_analyzer.py # 脸型分析模块
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明文档
├── templates/
│   └── index.html        # Web界面模板
├── static/
│   ├── css/
│   │   └── style.css     # 样式表
│   └── js/
│       └── main.js       # 前端JavaScript
└── models/               # 模型文件目录
```

## 🔧 技术栈

### 后端
- **Flask** - Web框架
- **OpenCV** - 图像处理
- **MediaPipe** - 人脸检测与关键点提取 (Google)
- **NumPy** - 数值计算
- **TensorFlow** - 深度学习 (预留扩展)

### 前端
- **原生JavaScript** - 交互逻辑
- **CSS3** - 样式和动画
- **HTML5** - 媒体API (摄像头)

### 算法
- **MediaPipe FaceMesh** - 468个人脸关键点检测
- **FACS (面部动作编码系统)** - 情绪分析基础
- **几何比例分析** - 脸型判定算法

## 📊 算法说明

### 情绪识别算法

基于 **MediaPipe FaceMesh** 提取的468个面部关键点，通过以下指标分析情绪：

1. **嘴唇形状** - 嘴角上扬/下垂程度
2. **眼睛状态** - 眼睛睁开程度和形状
3. **眉毛位置** - 眉毛高度和倾斜度
4. **面部动作单元 (AUs)** - FACS编码系统

情绪强度分级：微弱、轻微、中等、明显、强烈

### 脸型分析算法

基于面部关键点的几何测量：

1. **脸长/脸宽比** - 基础脸型判断
2. **额头宽度** - 判断额头发际线
3. **颧骨宽度** - 面部最宽处
4. **下颌宽度** - 下颌角度和宽度
5. **面部曲率** - 轮廓圆润度
6. **下颌角度** - 判断棱角程度

## 🛠️ 高级配置

### 调整检测灵敏度

在 `face_detector.py` 中修改：

```python
face_detector = FaceDetector(min_detection_confidence=0.5)
```

- 值越小，检测越灵敏但可能有误检
- 值越大，检测更严格但更精确

### 自定义情绪权重

在 `emotion_analyzer.py` 中修改：

```python
self.emotion_weights = {
    Emotion.HAPPY: 1.0,
    Emotion.SAD: 1.0,
    # ...
}
```

### 修改服务器端口

在 `app.py` 末尾修改：

```python
app.run(host='0.0.0.0', port=8080, debug=True)  # 改为8080端口
```

## 🐛 故障排除

### 摄像头无法访问
- 确保摄像头未被其他应用占用
- 检查浏览器权限设置
- 尝试使用HTTPS或localhost环境

### 检测结果不准确
- 确保光线充足且均匀
- 面部正对摄像头
- 避免遮挡面部 (眼镜、口罩等)
- 调整检测灵敏度参数

### 安装依赖失败
```bash
# 单独安装MediaPipe
pip install mediapipe

# 如果OpenCV安装失败
pip install opencv-python-headless
```

## 📝 API接口

### POST /analyze
分析上传的图像

**请求体:**
```json
{
    "image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

**响应:**
```json
{
    "success": true,
    "emotion": {
        "primary": "开心",
        "confidence": 0.85,
        "all_scores": {...},
        "intensity": "明显",
        "action_units": {...}
    },
    "face_shape": {
        "type": "椭圆形",
        "confidence": 0.72,
        "description": "...",
        "key_features": [...],
        "measurements": {...},
        "advice": {...}
    },
    "annotated_image": "data:image/jpeg;base64,..."
}
```

### GET /api/emotions
获取支持的情绪类型列表

### GET /api/face_shapes
获取支持的脸型类型列表

### GET /health
健康检查

## 🔒 隐私说明

- 所有图像处理在本地完成，不会上传到服务器
- 摄像头数据仅在浏览器中处理
- 支持离线使用 (除首次加载外)

## 📄 许可证

MIT License

## 🙏 致谢

- [MediaPipe](https://mediapipe.dev/) - Google的人脸检测框架
- [Flask](https://flask.palletsprojects.com/) - Python Web框架
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 联系方式

如有问题或建议，欢迎反馈。

---

**人脸情绪识别与脸型分析系统** © 2024
