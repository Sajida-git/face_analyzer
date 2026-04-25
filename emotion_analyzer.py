"""
情绪分析模块 - 根据面部特征点识别情绪状态
"""

import numpy as np
from typing import Dict, Tuple, List
from enum import Enum


class Emotion(Enum):
    """情绪类型枚举"""
    HAPPY = "开心"
    SAD = "悲伤"
    ANGRY = "愤怒"
    SURPRISED = "惊讶"
    NEUTRAL = "平静"
    FEAR = "恐惧"
    DISGUST = "厌恶"
    CONFUSED = "困惑"


class EmotionAnalyzer:
    """情绪分析器 - 基于面部关键点分析情绪"""

    def __init__(self):
        """初始化情绪分析器"""
        # 情绪权重映射
        self.emotion_weights = {
            Emotion.HAPPY: 1.0,
            Emotion.SAD: 1.0,
            Emotion.ANGRY: 1.0,
            Emotion.SURPRISED: 1.0,
            Emotion.NEUTRAL: 1.0,
            Emotion.FEAR: 1.0,
            Emotion.DISGUST: 1.0,
            Emotion.CONFUSED: 1.0,
        }

    def analyze_emotion(self, landmarks: np.ndarray) -> Tuple[Emotion, float, Dict[str, float]]:
        """
        分析情绪

        Args:
            landmarks: 面部关键点数组 (468, 3)

        Returns:
            (主要情绪, 置信度, 所有情绪分数字典)
        """
        # 计算各项情绪指标
        scores = {}

        # 1. 检测笑容 (Happy)
        scores['happy'] = self._detect_smile(landmarks)

        # 2. 检测悲伤表情
        scores['sad'] = self._detect_sadness(landmarks)

        # 3. 检测愤怒表情
        scores['angry'] = self._detect_anger(landmarks)

        # 4. 检测惊讶表情
        scores['surprised'] = self._detect_surprise(landmarks)

        # 5. 检测恐惧表情
        scores['fear'] = self._detect_fear(landmarks)

        # 6. 检测厌恶表情
        scores['disgust'] = self._detect_disgust(landmarks)

        # 7. 检测平静/中性表情
        scores['neutral'] = self._detect_neutral(landmarks, scores)

        # 8. 检测困惑表情
        scores['confused'] = self._detect_confused(landmarks)

        # 归一化分数
        total_score = sum(scores.values())
        if total_score > 0:
            normalized_scores = {k: v / total_score for k, v in scores.items()}
        else:
            normalized_scores = scores

        # 确定主要情绪
        max_emotion = max(normalized_scores, key=normalized_scores.get)
        max_score = normalized_scores[max_emotion]

        # 将字符串映射到Emotion枚举
        emotion_map = {
            'happy': Emotion.HAPPY,
            'sad': Emotion.SAD,
            'angry': Emotion.ANGRY,
            'surprised': Emotion.SURPRISED,
            'fear': Emotion.FEAR,
            'disgust': Emotion.DISGUST,
            'neutral': Emotion.NEUTRAL,
            'confused': Emotion.CONFUSED,
        }

        return emotion_map[max_emotion], max_score, normalized_scores

    def _detect_smile(self, landmarks: np.ndarray) -> float:
        """
        检测笑容程度

        通过嘴角上扬程度、嘴唇张开程度等指标判断
        """
        # 嘴唇关键点索引
        upper_lip_top = landmarks[61]  # 上唇中央
        lower_lip_bottom = landmarks[146]  # 下唇中央
        left_corner = landmarks[291]  # 左嘴角
        right_corner = landmarks[61]   # 右嘴角 (索引需修正)

        # 正确的嘴角索引
        left_corner = landmarks[291]   # 左嘴角
        right_corner = landmarks[61]   # 右嘴角

        # 计算嘴角上扬程度
        mouth_center_y = (landmarks[13][1] + landmarks[14][1]) / 2  # 嘴唇中央y坐标
        left_corner_y = landmarks[291][1]
        right_corner_y = landmarks[61][1]

        # 嘴角相对于嘴唇中心的高度差
        left_lift = mouth_center_y - left_corner_y
        right_lift = mouth_center_y - right_corner_y

        # 嘴角上扬程度 (正值表示上扬)
        smile_lift = (left_lift + right_lift) / 2

        # 计算嘴唇张开程度
        mouth_open = abs(landmarks[14][1] - landmarks[13][1])

        # 综合笑容评分
        smile_score = max(0, smile_lift * 10) + mouth_open * 5

        # 归一化到0-1范围
        smile_score = min(1.0, smile_score * 2)

        return max(0, smile_score)

    def _detect_sadness(self, landmarks: np.ndarray) -> float:
        """
        检测悲伤表情

        特征：嘴角下垂、眉毛内角上扬、眼睛微眯
        """
        # 嘴角下垂程度
        mouth_center_y = (landmarks[13][1] + landmarks[14][1]) / 2
        left_corner_y = landmarks[291][1]
        right_corner_y = landmarks[61][1]

        # 嘴角下垂 (负值表示下垂)
        corner_droop = (left_corner_y + right_corner_y) / 2 - mouth_center_y

        # 眉毛内角上扬 (皱眉)
        left_brow_inner = landmarks[105]
        left_brow_outer = landmarks[334]
        right_brow_inner = landmarks[70]
        right_brow_outer = landmarks[69]

        # 眉毛倾斜程度
        left_brow_tilt = left_brow_inner[1] - left_brow_outer[1]
        right_brow_tilt = right_brow_inner[1] - right_brow_outer[1]

        # 悲伤评分
        sadness_score = max(0, corner_droop * 8) + max(0, left_brow_tilt + right_brow_tilt) * 3

        return min(1.0, sadness_score)

    def _detect_anger(self, landmarks: np.ndarray) -> float:
        """
        检测愤怒表情

        特征：眉毛压低并内聚、嘴唇紧闭或张开、眼睛瞪大
        """
        # 眉毛压低程度
        left_brow_y = (landmarks[105][1] + landmarks[107][1]) / 2
        right_brow_y = (landmarks[70][1] + landmarks[72][1]) / 2

        # 眼睛位置
        left_eye_y = (landmarks[33][1] + landmarks[133][1]) / 2
        right_eye_y = (landmarks[362][1] + landmarks[263][1]) / 2

        # 眉毛与眼睛的距离 (愤怒时眉毛压低)
        brow_eye_distance = ((left_eye_y - left_brow_y) + (right_eye_y - right_brow_y)) / 2

        # 正常距离约为0.05-0.08，愤怒时更小
        anger_from_brows = max(0, 0.08 - brow_eye_distance) * 15

        # 嘴唇紧闭程度
        lip_distance = abs(landmarks[13][1] - landmarks[14][1])
        lip_tightness = max(0, 0.02 - lip_distance) * 20

        # 愤怒评分
        anger_score = anger_from_brows * 0.6 + lip_tightness * 0.4

        return min(1.0, anger_score)

    def _detect_surprise(self, landmarks: np.ndarray) -> float:
        """
        检测惊讶表情

        特征：眉毛上扬、眼睛睁大、嘴巴张开
        """
        # 眉毛上扬程度
        left_brow_y = landmarks[105][1]
        right_brow_y = landmarks[70][1]
        brow_avg_y = (left_brow_y + right_brow_y) / 2

        # 眼睛睁开程度
        left_eye_open = abs(landmarks[159][1] - landmarks[145][1])
        right_eye_open = abs(landmarks[386][1] - landmarks[374][1])
        avg_eye_open = (left_eye_open + right_eye_open) / 2

        # 嘴巴张开程度
        mouth_open = abs(landmarks[14][1] - landmarks[13][1])

        # 惊讶评分 (综合眉毛位置、眼睛睁开程度和嘴巴张开程度)
        surprise_score = avg_eye_open * 3 + mouth_open * 8

        return min(1.0, surprise_score)

    def _detect_fear(self, landmarks: np.ndarray) -> float:
        """
        检测恐惧表情

        特征：眉毛上扬并内聚、眼睛睁大、嘴巴微张
        """
        # 眉毛上扬和内聚
        left_brow_inner = landmarks[105][1]
        left_brow_outer = landmarks[334][1]
        right_brow_inner = landmarks[70][1]
        right_brow_outer = landmarks[69][1]

        brow_raise = (left_brow_outer + right_brow_outer - left_brow_inner - right_brow_inner) / 2

        # 眼睛睁大
        left_eye_open = abs(landmarks[159][1] - landmarks[145][1])
        right_eye_open = abs(landmarks[386][1] - landmarks[374][1])

        # 嘴巴微张
        mouth_open = abs(landmarks[14][1] - landmarks[13][1])

        # 恐惧评分
        fear_score = max(0, brow_raise * 5) + (left_eye_open + right_eye_open) * 2 + mouth_open * 3

        return min(1.0, fear_score)

    def _detect_disgust(self, landmarks: np.ndarray) -> float:
        """
        检测厌恶表情

        特征：鼻子皱起、上唇上扬、眉毛压低
        """
        # 鼻子区域
        nose_tip = landmarks[4]
        upper_lip = landmarks[13]

        # 上唇上扬程度
        lip_raise = nose_tip[1] - upper_lip[1]

        # 眉毛压低
        left_brow_y = landmarks[105][1]
        right_brow_y = landmarks[70][1]

        # 厌恶评分
        disgust_score = max(0, lip_raise * 8)

        return min(1.0, disgust_score)

    def _detect_neutral(self, landmarks: np.ndarray, other_scores: Dict[str, float]) -> float:
        """
        检测中性/平静表情

        当其他情绪分数都很低时，认为是中性表情
        """
        # 如果所有其他情绪分数都很低，则中性分数高
        max_other = max(other_scores.values()) if other_scores else 0

        # 中性评分与最大其他情绪分数成反比
        neutral_score = max(0, 1.0 - max_other * 2)

        return neutral_score

    def _detect_confused(self, landmarks: np.ndarray) -> float:
        """
        检测困惑表情

        特征：眉毛不对称、单侧眉头皱起
        """
        # 眉毛不对称性
        left_brow_positions = [landmarks[i][1] for i in [105, 107, 109]]
        right_brow_positions = [landmarks[i][1] for i in [70, 72, 74]]

        left_brow_avg = np.mean(left_brow_positions)
        right_brow_avg = np.mean(right_brow_positions)

        # 眉毛不对称程度
        brow_asymmetry = abs(left_brow_avg - right_brow_avg)

        # 困惑评分
        confused_score = brow_asymmetry * 5

        return min(1.0, confused_score)

    def get_emotion_details(self, landmarks: np.ndarray) -> Dict[str, any]:
        """
        获取详细的情绪分析结果

        Args:
            landmarks: 面部关键点数组

        Returns:
            包含详细情绪信息的字典
        """
        emotion, confidence, all_scores = self.analyze_emotion(landmarks)

        # 获取面部动作单元 (Action Units) 信息
        action_units = self._extract_action_units(landmarks)

        return {
            'primary_emotion': emotion.value,
            'confidence': confidence,
            'all_scores': all_scores,
            'action_units': action_units,
            'intensity': self._calculate_emotion_intensity(all_scores),
        }

    def _extract_action_units(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        提取面部动作单元 (Facial Action Units)

        基于FACS (Facial Action Coding System)
        """
        aus = {}

        # AU1: 内眉上扬
        aus['AU1'] = self._calculate_au1(landmarks)

        # AU2: 外眉上扬
        aus['AU2'] = self._calculate_au2(landmarks)

        # AU4: 眉毛压低
        aus['AU4'] = self._calculate_au4(landmarks)

        # AU6: 脸颊上扬 (笑容相关)
        aus['AU6'] = self._calculate_au6(landmarks)

        # AU12: 嘴角上扬
        aus['AU12'] = self._calculate_au12(landmarks)

        # AU25: 嘴唇分开
        aus['AU25'] = self._calculate_au25(landmarks)

        # AU26: 下巴下降 (嘴巴张开)
        aus['AU26'] = self._calculate_au26(landmarks)

        return aus

    def _calculate_au1(self, landmarks: np.ndarray) -> float:
        """AU1: 内眉上扬"""
        left_inner = landmarks[105][1]
        left_outer = landmarks[334][1]
        return max(0, left_inner - left_outer) * 10

    def _calculate_au2(self, landmarks: np.ndarray) -> float:
        """AU2: 外眉上扬"""
        left_outer = landmarks[334][1]
        right_outer = landmarks[69][1]
        brow_avg = (left_outer + right_outer) / 2
        return max(0, 0.3 - brow_avg) * 5

    def _calculate_au4(self, landmarks: np.ndarray) -> float:
        """AU4: 眉毛压低"""
        left_brow = landmarks[105][1]
        right_brow = landmarks[70][1]
        eye_level = (landmarks[33][1] + landmarks[362][1]) / 2
        return max(0, eye_level - (left_brow + right_brow) / 2) * 10

    def _calculate_au6(self, landmarks: np.ndarray) -> float:
        """AU6: 脸颊上扬"""
        cheek_lift = landmarks[123][1] - landmarks[147][1]
        return max(0, cheek_lift) * 8

    def _calculate_au12(self, landmarks: np.ndarray) -> float:
        """AU12: 嘴角上扬"""
        mouth_center = (landmarks[13][1] + landmarks[14][1]) / 2
        left_corner = landmarks[291][1]
        right_corner = landmarks[61][1]
        return (mouth_center - (left_corner + right_corner) / 2) * 10

    def _calculate_au25(self, landmarks: np.ndarray) -> float:
        """AU25: 嘴唇分开"""
        return abs(landmarks[14][1] - landmarks[13][1]) * 20

    def _calculate_au26(self, landmarks: np.ndarray) -> float:
        """AU26: 下巴下降"""
        return abs(landmarks[152][1] - landmarks[13][1]) * 5

    def _calculate_emotion_intensity(self, scores: Dict[str, float]) -> str:
        """
        计算情绪强度等级
        """
        max_score = max(scores.values())

        if max_score < 0.3:
            return "微弱"
        elif max_score < 0.5:
            return "轻微"
        elif max_score < 0.7:
            return "中等"
        elif max_score < 0.85:
            return "明显"
        else:
            return "强烈"
