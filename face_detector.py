"""
人脸检测模块 - 使用MediaPipe进行人脸检测和关键点提取
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict


class FaceDetector:
    """人脸检测器类，用于检测人脸位置、提取面部关键点"""

    def __init__(self, min_detection_confidence: float = 0.5):
        """
        初始化人脸检测器

        Args:
            min_detection_confidence: 最小检测置信度 (0-1)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化人脸检测模型
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )

        # 初始化人脸关键点检测模型 (468个关键点)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def detect_face(self, image: np.ndarray) -> Tuple[bool, Optional[List]]:
        """
        检测图像中的人脸

        Args:
            image: BGR格式的图像数组

        Returns:
            (是否检测到人脸, 人脸检测结果列表)
        """
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 执行人脸检测
        results = self.face_detection.process(rgb_image)

        if results.detections:
            return True, results.detections
        return False, None

    def get_face_landmarks(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        获取人脸关键点坐标 (468个点)

        Args:
            image: BGR格式的图像数组

        Returns:
            (是否成功, 关键点坐标数组 shape: (468, 3))
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            return True, landmarks
        return False, None

    def get_face_bbox(self, image: np.ndarray, detection) -> Tuple[int, int, int, int]:
        """
        从检测结果中获取人脸边界框坐标

        Args:
            image: 原始图像
            detection: MediaPipe检测结果

        Returns:
            (x, y, width, height) - 人脸框坐标
        """
        ih, iw = image.shape[:2]
        bboxC = detection.location_data.relative_bounding_box

        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # 确保坐标在图像范围内
        x = max(0, x)
        y = max(0, y)

        return x, y, w, h

    def get_face_region(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据关键点获取面部不同区域

        Args:
            image: 原始图像
            landmarks: 关键点坐标数组

        Returns:
            包含各面部区域关键点的字典
        """
        ih, iw = image.shape[:2]

        # 将归一化坐标转换为像素坐标
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= iw
        pixel_landmarks[:, 1] *= ih

        # 定义面部区域的关键点索引 (MediaPipe FaceMesh)
        regions = {
            'face_outline': pixel_landmarks[list(range(0, 17)) + list(range(26, 22, -1))],  # 脸部轮廓
            'left_eyebrow': pixel_landmarks[70:80],  # 左眉毛
            'right_eyebrow': pixel_landmarks[105:115],  # 右眉毛
            'left_eye': pixel_landmarks[33:42],  # 左眼
            'right_eye': pixel_landmarks[362:371],  # 右眼
            'nose': pixel_landmarks[168:195],  # 鼻子
            'upper_lip': pixel_landmarks[61:68],  # 上嘴唇
            'lower_lip': pixel_landmarks[146:153],  # 下嘴唇
            'jawline': pixel_landmarks[0:17],  # 下巴轮廓
        }

        return regions

    def draw_face_detection(self, image: np.ndarray, detections: List,
                          draw_landmarks: bool = True) -> np.ndarray:
        """
        在图像上绘制人脸检测框和关键点

        Args:
            image: 原始图像
            detections: 人脸检测结果
            draw_landmarks: 是否绘制关键点

        Returns:
            绘制后的图像
        """
        annotated_image = image.copy()

        for detection in detections:
            # 绘制边界框
            x, y, w, h = self.get_face_bbox(image, detection)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制置信度
            score = detection.score[0] if detection.score else 0
            cv2.putText(annotated_image, f"Face: {score:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制面部关键点
        if draw_landmarks:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mesh_results = self.face_mesh.process(rgb_image)

            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # 绘制关键点
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        face_landmarks,
                        self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

        return annotated_image

    def get_face_center(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        计算人脸中心点坐标

        Args:
            landmarks: 关键点坐标数组

        Returns:
            (center_x, center_y)
        """
        center = np.mean(landmarks, axis=0)
        return center[0], center[1]

    def calculate_face_size(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        计算人脸的宽度和高度

        Args:
            landmarks: 关键点坐标数组

        Returns:
            (width, height) - 人脸的归一化宽度和高度
        """
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]

        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)

        return width, height

    def release(self):
        """释放资源"""
        self.face_detection.close()
        self.face_mesh.close()
