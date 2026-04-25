"""
人脸情绪识别与脸型分析系统 - 主应用程序

功能:
1. 实时人脸检测
2. 情绪识别 (开心、悲伤、愤怒、惊讶、平静、恐惧、厌恶、困惑)
3. 脸型分析 (椭圆形、圆形、方形、长方形、心形、菱形、三角形、长形)

技术栈:
- Flask: Web框架
- OpenCV: 图像处理
- MediaPipe: 人脸检测与关键点提取
- NumPy: 数值计算
"""

import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask import send_from_directory
from PIL import Image
import io
import json
from datetime import datetime

from face_detector import FaceDetector
from emotion_analyzer import EmotionAnalyzer, Emotion
from face_shape_analyzer import FaceShapeAnalyzer, FaceShape


app = Flask(__name__)

# 初始化检测器和分析器
face_detector = FaceDetector(min_detection_confidence=0.5)
emotion_analyzer = EmotionAnalyzer()
face_shape_analyzer = FaceShapeAnalyzer()

# 全局变量存储当前分析结果
current_analysis = {
    'emotion': None,
    'face_shape': None,
    'timestamp': None,
}


def base64_to_image(base64_string):
    """将Base64字符串转换为OpenCV图像"""
    try:
        # 移除data URL前缀
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]

        # 解码Base64
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img
    except Exception as e:
        print(f"Base64转图像错误: {e}")
        return None


def image_to_base64(image):
    """将OpenCV图像转换为Base64字符串"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_base64}"
    except Exception as e:
        print(f"图像转Base64错误: {e}")
        return None


def analyze_frame(image):
    """
    分析单帧图像

    Args:
        image: OpenCV图像 (BGR格式)

    Returns:
        分析结果字典
    """
    result = {
        'success': False,
        'emotion': None,
        'face_shape': None,
        'annotated_image': None,
        'error': None,
    }

    try:
        # 1. 检测人脸
        has_face, detections = face_detector.detect_face(image)

        if not has_face:
            result['error'] = '未检测到人脸'
            result['annotated_image'] = image_to_base64(image)
            return result

        # 2. 获取人脸关键点
        has_landmarks, landmarks = face_detector.get_face_landmarks(image)

        if not has_landmarks:
            result['error'] = '无法获取面部关键点'
            result['annotated_image'] = image_to_base64(
                face_detector.draw_face_detection(image, detections)
            )
            return result

        # 3. 情绪分析
        emotion, emotion_confidence, emotion_scores = emotion_analyzer.analyze_emotion(landmarks)
        emotion_details = emotion_analyzer.get_emotion_details(landmarks)

        # 4. 脸型分析
        face_shape, shape_confidence, shape_details = face_shape_analyzer.analyze_face_shape(landmarks)
        shape_advice = face_shape_analyzer.get_face_shape_advice(face_shape)

        # 5. 绘制检测结果
        annotated_image = face_detector.draw_face_detection(image, detections, draw_landmarks=True)

        # 在图像上添加文字信息
        ih, iw = annotated_image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(iw, ih) / 800
        thickness = max(1, int(min(iw, ih) / 400))

        # 情绪信息
        emotion_text = f"情绪: {emotion.value} ({emotion_confidence*100:.1f}%)"
        cv2.putText(annotated_image, emotion_text, (10, 30),
                   font, font_scale, (0, 255, 0), thickness)

        # 脸型信息
        shape_text = f"脸型: {face_shape.value} ({shape_confidence*100:.1f}%)"
        cv2.putText(annotated_image, shape_text, (10, 60),
                   font, font_scale, (255, 165, 0), thickness)

        # 组装结果
        result['success'] = True
        result['emotion'] = {
            'primary': emotion.value,
            'confidence': round(emotion_confidence, 3),
            'all_scores': {k: round(v, 3) for k, v in emotion_scores.items()},
            'intensity': emotion_details['intensity'],
            'action_units': emotion_details['action_units'],
        }
        result['face_shape'] = {
            'type': face_shape.value,
            'confidence': round(shape_confidence, 3),
            'description': shape_details['description'],
            'key_features': shape_details['key_features'],
            'measurements': shape_details['measurements'],
            'advice': shape_advice,
        }
        result['annotated_image'] = image_to_base64(annotated_image)
        result['timestamp'] = datetime.now().isoformat()

        return result

    except Exception as e:
        result['error'] = f'分析错误: {str(e)}'
        result['annotated_image'] = image_to_base64(image)
        return result


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    分析上传的图像

    接收Base64编码的图像，返回分析结果
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': '未提供图像数据'}), 400

        # 转换Base64到图像
        image = base64_to_image(data['image'])

        if image is None:
            return jsonify({'success': False, 'error': '图像解码失败'}), 400

        # 分析图像
        result = analyze_frame(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500


@app.route('/analyze_frame', methods=['POST'])
def analyze_frame_endpoint():
    """
    分析视频帧 (用于实时分析)
    """
    try:
        data = request.get_json()

        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': '未提供帧数据'}), 400

        # 转换Base64到图像
        image = base64_to_image(data['frame'])

        if image is None:
            return jsonify({'success': False, 'error': '帧解码失败'}), 400

        # 分析图像
        result = analyze_frame(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': f'服务器错误: {str(e)}'}), 500


@app.route('/video_feed')
def video_feed():
    """
    视频流 (用于直接摄像头访问)
    """
    def generate():
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 分析帧
            result = analyze_frame(frame)

            # 使用标注后的图像
            if result['annotated_image']:
                # 转换Base64回图像用于流传输
                annotated = base64_to_image(result['annotated_image'])
                if annotated is not None:
                    _, buffer = cv2.imencode('.jpg', annotated)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/static/<path:path>')
def send_static(path):
    """静态文件服务"""
    return send_from_directory('static', path)


@app.route('/api/face_shapes')
def get_face_shapes():
    """获取所有脸型类型信息"""
    shapes = [shape.value for shape in FaceShape]
    return jsonify({'face_shapes': shapes})


@app.route('/api/emotions')
def get_emotions():
    """获取所有情绪类型信息"""
    emotions = [emotion.value for emotion in Emotion]
    return jsonify({'emotions': emotions})


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': True,
    })


def create_app():
    """应用工厂函数"""
    return app


if __name__ == '__main__':
    print("=" * 60)
    print("人脸情绪识别与脸型分析系统")
    print("=" * 60)
    print("功能:")
    print("  - 实时人脸检测")
    print("  - 情绪识别 (8种情绪)")
    print("  - 脸型分析 (8种脸型)")
    print("=" * 60)
    print("启动服务器...")
    print("请在浏览器中访问: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
