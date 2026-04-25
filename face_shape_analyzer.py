"""
脸型分析模块 - 根据面部轮廓特征点判断脸型类型
"""

import numpy as np
from typing import Dict, Tuple, List
from enum import Enum
from scipy.spatial import distance


class FaceShape(Enum):
    """脸型类型枚举"""
    OVAL = "椭圆形"
    ROUND = "圆形"
    SQUARE = "方形"
    RECTANGLE = "长方形"
    HEART = "心形"
    DIAMOND = "菱形"
    TRIANGLE = "三角形"
    OBLONG = "长形"


class FaceShapeAnalyzer:
    """脸型分析器 - 基于面部关键点分析脸型"""

    def __init__(self):
        """初始化脸型分析器"""
        # 脸型特征定义
        self.shape_characteristics = {
            FaceShape.OVAL: {
                'description': '脸部线条柔和，额头和下巴宽度相近，脸部轮廓流畅圆润',
                'features': ['额头略宽于下巴', '脸部线条柔和', '下颌曲线流畅'],
            },
            FaceShape.ROUND: {
                'description': '脸部长度和宽度相近，脸部轮廓圆润，没有明显棱角',
                'features': ['脸宽和脸长接近', '脸颊饱满', '下颌线条圆润'],
            },
            FaceShape.SQUARE: {
                'description': '额头、颧骨和下巴宽度相近，下颌线条明显，棱角分明',
                'features': ['额头、颧骨、下巴宽度接近', '下颌线条硬朗', '面部棱角分明'],
            },
            FaceShape.RECTANGLE: {
                'description': '脸长明显大于脸宽，额头和下巴宽度相近，面部轮廓较直',
                'features': ['脸长明显大于脸宽', '额头和下巴宽度相近', '面部线条较直'],
            },
            FaceShape.HEART: {
                'description': '额头较宽，颧骨突出，下巴尖细，呈倒三角形状',
                'features': ['额头宽', '颧骨突出', '下巴尖细'],
            },
            FaceShape.DIAMOND: {
                'description': '颧骨最宽，额头和下巴较窄，脸部轮廓呈菱形',
                'features': ['颧骨是脸部最宽处', '额头较窄', '下巴尖细'],
            },
            FaceShape.TRIANGLE: {
                'description': '下巴较宽，额头较窄，脸部轮廓呈正三角形状',
                'features': ['下颌宽于额头', '额头较窄', '下颌线条明显'],
            },
            FaceShape.OBLONG: {
                'description': '脸长明显大于脸宽，额头和下巴宽度相近但比颧骨宽',
                'features': ['脸长明显大于脸宽', '额头和下巴比颧骨宽', '面部线条较长'],
            },
        }

    def analyze_face_shape(self, landmarks: np.ndarray) -> Tuple[FaceShape, float, Dict[str, any]]:
        """
        分析脸型

        Args:
            landmarks: 面部关键点数组 (468, 3) - 归一化坐标

        Returns:
            (脸型类型, 置信度, 详细分析信息)
        """
        # 提取关键测量点
        measurements = self._extract_measurements(landmarks)

        # 计算各种脸型评分
        scores = self._calculate_shape_scores(measurements)

        # 确定最可能的脸型
        max_shape = max(scores, key=scores.get)
        confidence = scores[max_shape]

        # 获取详细分析
        details = self._get_shape_details(max_shape, measurements, scores)

        return max_shape, confidence, details

    def _extract_measurements(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        提取面部关键测量数据

        定义关键点索引:
        - 下巴: 152
        - 额头中心: 10
        - 左脸边缘: 234
        - 右脸边缘: 454
        - 左颧骨: 345
        - 右颧骨: 115
        - 左下颌角: 58
        - 右下颌角: 288
        - 左太阳穴: 105
        - 右太阳穴: 334
        """
        measurements = {}

        # 脸部轮廓点 (使用脸部边缘的17个点 + 额头区域)
        face_outline_indices = list(range(0, 17)) + [21, 22, 251, 289, 305, 351, 419, 280]
        face_outline = landmarks[face_outline_indices]

        # 1. 脸长: 从额头到下巴的距离
        forehead = landmarks[10]  # 额头中心
        chin = landmarks[152]     # 下巴尖端
        measurements['face_length'] = np.linalg.norm(forehead[:2] - chin[:2])

        # 2. 脸宽: 左右脸颊最宽处的距离
        left_face = landmarks[234]  # 左脸边缘
        right_face = landmarks[454]  # 右脸边缘
        measurements['face_width'] = np.linalg.norm(left_face[:2] - right_face[:2])

        # 3. 额头宽度
        left_forehead = landmarks[105]  # 左太阳穴
        right_forehead = landmarks[334]  # 右太阳穴
        measurements['forehead_width'] = np.linalg.norm(left_forehead[:2] - right_forehead[:2])

        # 4. 颧骨宽度
        left_cheek = landmarks[345]  # 左颧骨
        right_cheek = landmarks[115]  # 右颧骨
        measurements['cheekbone_width'] = np.linalg.norm(left_cheek[:2] - right_cheek[:2])

        # 5. 下颌宽度
        left_jaw = landmarks[58]   # 左下颌角
        right_jaw = landmarks[288]  # 右下颌角
        measurements['jaw_width'] = np.linalg.norm(left_jaw[:2] - right_jaw[:2])

        # 6. 下巴宽度 (下巴底部宽度)
        left_chin = landmarks[148]
        right_chin = landmarks[377]
        measurements['chin_width'] = np.linalg.norm(left_chin[:2] - right_chin[:2])

        # 7. 计算脸部长宽比
        measurements['length_width_ratio'] = (
            measurements['face_length'] / measurements['face_width']
            if measurements['face_width'] > 0 else 0
        )

        # 8. 计算各部位比例
        measurements['forehead_to_jaw_ratio'] = (
            measurements['forehead_width'] / measurements['jaw_width']
            if measurements['jaw_width'] > 0 else 0
        )

        measurements['cheekbone_to_jaw_ratio'] = (
            measurements['cheekbone_width'] / measurements['jaw_width']
            if measurements['jaw_width'] > 0 else 0
        )

        # 9. 计算下颌角度
        measurements['jaw_angle'] = self._calculate_jaw_angle(landmarks)

        # 10. 计算脸部曲线度
        measurements['face_curvature'] = self._calculate_face_curvature(face_outline)

        # 11. 计算前额倾斜度
        measurements['forehead_slope'] = self._calculate_forehead_slope(landmarks)

        return measurements

    def _calculate_jaw_angle(self, landmarks: np.ndarray) -> float:
        """
        计算下颌角度

        使用下巴尖和两个下颌角形成的角度
        """
        chin = landmarks[152][:2]
        left_jaw = landmarks[58][:2]
        right_jaw = landmarks[288][:2]

        # 计算向量
        v1 = left_jaw - chin
        v2 = right_jaw - chin

        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle)

    def _calculate_face_curvature(self, face_outline: np.ndarray) -> float:
        """
        计算脸部轮廓的曲线程度

        通过计算轮廓点与椭圆拟合的偏离程度
        """
        # 使用脸部轮廓点计算
        points = face_outline[:, :2]

        # 计算轮廓的紧凑度
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        std_distance = np.std(distances)
        mean_distance = np.mean(distances)

        # 曲率越高，标准差相对于均值越小 (越接近圆形)
        curvature = 1.0 - (std_distance / (mean_distance + 1e-10))

        return curvature

    def _calculate_forehead_slope(self, landmarks: np.ndarray) -> float:
        """
        计算前额倾斜度
        """
        forehead_top = landmarks[10][:2]
        forehead_left = landmarks[105][:2]
        forehead_right = landmarks[334][:2]

        # 计算两侧倾斜度
        left_slope = abs(forehead_top[1] - forehead_left[1])
        right_slope = abs(forehead_top[1] - forehead_right[1])

        return (left_slope + right_slope) / 2

    def _calculate_shape_scores(self, m: Dict[str, float]) -> Dict[FaceShape, float]:
        """
        根据测量数据计算各种脸型的匹配分数
        """
        scores = {}

        ratio = m['length_width_ratio']
        jaw_angle = m['jaw_angle']
        curvature = m['face_curvature']
        forehead_jaw_ratio = m['forehead_to_jaw_ratio']
        cheekbone_jaw_ratio = m['cheekbone_to_jaw_ratio']

        # 1. 椭圆形 (Oval)
        # 特征: 长宽比约1.5，下颌角度适中，曲线度适中
        oval_score = (
            self._gaussian_score(ratio, 1.5, 0.2) * 0.3 +
            self._gaussian_score(jaw_angle, 110, 15) * 0.3 +
            curvature * 0.2 +
            self._gaussian_score(forehead_jaw_ratio, 1.0, 0.15) * 0.2
        )
        scores[FaceShape.OVAL] = oval_score

        # 2. 圆形 (Round)
        # 特征: 长宽比接近1，高曲线度，下颌角度大
        round_score = (
            self._gaussian_score(ratio, 1.0, 0.15) * 0.4 +
            curvature * 0.3 +
            self._gaussian_score(jaw_angle, 130, 15) * 0.3
        )
        scores[FaceShape.ROUND] = round_score

        # 3. 方形 (Square)
        # 特征: 下颌角度接近90度，曲线度低，额头和下颌宽度相近
        square_score = (
            self._gaussian_score(jaw_angle, 95, 10) * 0.4 +
            (1 - curvature) * 0.3 +
            self._gaussian_score(forehead_jaw_ratio, 1.0, 0.1) * 0.3
        )
        scores[FaceShape.SQUARE] = square_score

        # 4. 长方形 (Rectangle)
        # 特征: 长宽比大，下颌角度适中，额头和下颌宽度相近
        rectangle_score = (
            self._gaussian_score(ratio, 1.8, 0.2) * 0.4 +
            self._gaussian_score(forehead_jaw_ratio, 1.0, 0.1) * 0.3 +
            (1 - curvature) * 0.3
        )
        scores[FaceShape.RECTANGLE] = rectangle_score

        # 5. 心形 (Heart)
        # 特征: 额头明显宽于下颌，下颌角度大，下巴尖细
        heart_score = (
            self._gaussian_score(forehead_jaw_ratio, 1.3, 0.2) * 0.4 +
            self._gaussian_score(jaw_angle, 125, 15) * 0.3 +
            self._gaussian_ratio_score(cheekbone_jaw_ratio, 1.2, 0.15) * 0.3
        )
        scores[FaceShape.HEART] = heart_score

        # 6. 菱形 (Diamond)
        # 特征: 颧骨最宽，额头和下颌较窄
        diamond_score = (
            self._gaussian_score(cheekbone_jaw_ratio, 1.3, 0.2) * 0.4 +
            self._gaussian_score(forehead_jaw_ratio, 0.85, 0.15) * 0.3 +
            curvature * 0.3
        )
        scores[FaceShape.DIAMOND] = diamond_score

        # 7. 三角形 (Triangle)
        # 特征: 下颌宽于额头，下颌角度小
        triangle_score = (
            self._gaussian_score(forehead_jaw_ratio, 0.8, 0.1) * 0.4 +
            self._gaussian_score(jaw_angle, 95, 10) * 0.3 +
            (1 - curvature) * 0.3
        )
        scores[FaceShape.TRIANGLE] = triangle_score

        # 8. 长形 (Oblong)
        # 特征: 长宽比很大，曲线度低
        oblong_score = (
            self._gaussian_score(ratio, 2.0, 0.25) * 0.5 +
            (1 - curvature) * 0.3 +
            self._gaussian_score(forehead_jaw_ratio, 1.1, 0.15) * 0.2
        )
        scores[FaceShape.OBLONG] = oblong_score

        # 归一化分数
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _gaussian_score(self, value: float, target: float, sigma: float) -> float:
        """
        使用高斯函数计算匹配分数
        """
        return np.exp(-0.5 * ((value - target) / sigma) ** 2)

    def _gaussian_ratio_score(self, value: float, target: float, sigma: float) -> float:
        """
        使用高斯函数计算比率匹配分数 (仅考虑目标以上的值)
        """
        if value < target:
            return np.exp(-0.5 * ((value - target) / sigma) ** 2)
        return 1.0

    def _get_shape_details(self, shape: FaceShape, measurements: Dict[str, float],
                          scores: Dict[FaceShape, float]) -> Dict[str, any]:
        """
        获取脸型详细分析信息
        """
        info = self.shape_characteristics[shape]

        # 按分数排序所有脸型
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'shape_name': shape.value,
            'description': info['description'],
            'key_features': info['features'],
            'measurements': {
                'face_length': round(measurements['face_length'] * 100, 2),
                'face_width': round(measurements['face_width'] * 100, 2),
                'length_width_ratio': round(measurements['length_width_ratio'], 2),
                'jaw_angle': round(measurements['jaw_angle'], 1),
                'face_curvature': round(measurements['face_curvature'], 3),
            },
            'all_shape_scores': {k.value: round(v, 3) for k, v in sorted_scores[:4]},
        }

    def get_face_shape_advice(self, shape: FaceShape) -> Dict[str, List[str]]:
        """
        根据脸型提供建议

        Args:
            shape: 脸型类型

        Returns:
            包含发型、眼镜、妆容建议的字典
        """
        advice = {
            FaceShape.OVAL: {
                'hairstyle': ['几乎所有发型都适合', '中长发最能展现优势', '避免完全遮住额头的发型'],
                'glasses': ['几乎所有款式都适合', '猫眼款式尤其好看', '圆形镜框也很适合'],
                'makeup': ['强调自然轮廓', '腮红打在颧骨上方', '唇妆可以选择任何风格'],
            },
            FaceShape.ROUND: {
                'hairstyle': ['侧分长发可以拉长脸型', '头顶蓬松的发型', '避免齐刘海和过短的发型'],
                'glasses': ['方形或矩形镜框', '棱角分明的款式', '避免圆形镜框'],
                'makeup': ['修容在脸颊两侧', '眉形略带角度', '唇妆选择饱满的颜色'],
            },
            FaceShape.SQUARE: {
                'hairstyle': ['柔和的波浪卷发', '侧分长刘海', '避免整齐的直发'],
                'glasses': ['圆形或椭圆形镜框', '猫眼款式', '避免过方的镜框'],
                'makeup': ['在下巴处打阴影', '眉形略带弧度', '腮红斜扫'],
            },
            FaceShape.RECTANGLE: {
                'hairstyle': ['带有刘海的发型', '两侧有卷度的发型', '避免完全露额头'],
                'glasses': ['大尺寸圆形镜框', '装饰性强的款式', '避免过窄的镜框'],
                'makeup': ['额头和下巴打阴影', '横向腮红', '柔和的眉形'],
            },
            FaceShape.HEART: {
                'hairstyle': ['下巴长度的波波头', '刘海遮住部分额头', '避免顶部过高的发型'],
                'glasses': ['下缘较宽的款式', '浅色或透明镜框', '避免上缘过宽的款式'],
                'makeup': ['额头打阴影', '下巴使用高光', '腮红在脸颊中央'],
            },
            FaceShape.DIAMOND: {
                'hairstyle': ['下巴长度的发型', '刘海可以遮盖颧骨', '避免过于蓬松的发型'],
                'glasses': ['椭圆形镜框', '轻质金属框架', '避免过小或过大的镜框'],
                'makeup': ['颧骨下方打阴影', '额头和下巴使用高光', '唇妆突出丰满感'],
            },
            FaceShape.TRIANGLE: {
                'hairstyle': ['顶部蓬松的发型', '有刘海的发型', '下巴长度的发型平衡下颌'],
                'glasses': ['上缘较宽的款式', '猫眼款式', '装饰性集中在上方'],
                'makeup': ['下颌打阴影', '额头使用高光', '腮红斜向上扫'],
            },
            FaceShape.OBLONG: {
                'hairstyle': ['带有刘海的发型', '横向蓬松的发型', '避免顶部过高的发型'],
                'glasses': ['大尺寸圆形或方形镜框', '装饰性边框', '避免过小或过窄的镜框'],
                'makeup': ['横向腮红', '眉毛平直', '额头和下巴打阴影'],
            },
        }

        return advice.get(shape, {})
