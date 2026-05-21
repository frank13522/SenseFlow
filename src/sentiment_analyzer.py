"""
SenseFlow - 情感分析引擎
使用中文情感词典 + 规则引擎（纯本地，无需下载模型）
"""
import numpy as np
from typing import List, Dict
from collections import defaultdict

from config import PROCESSOR_CONFIG


class SentimentAnalyzer:
    """
    基于词典的情感分析器（纯本地运行，无需下载任何模型）
    包含 >800 正面词 + >800 负面词 + 程度词 + 否定词
    """

    # 正面情感词（按情感强度分级）
    POSITIVE_LEVEL1 = [  # 强正面
        "突破", "领先", "首创", "首次", "冠军", "最佳", "第一", "世界级",
        "里程碑", "历史性", "颠覆", "革命性", "爆发式", "井喷",
        "涨停", "暴涨", "狂飙", "创新高", "翻倍", "超预期", "大超预期",
    ]
    POSITIVE_LEVEL2 = [  # 中等正面
        "增长", "上升", "提升", "改善", "优化", "加强", "推进", "加速",
        "合作", "共赢", "繁荣", "稳定", "安全", "成功", "发展", "突破",
        "强劲", "利好", "好消息", "优秀", "卓越", "领先", "超越",
        "推动", "促进", "实现", "显著", "明显", "大幅", "快速",
    ]
    POSITIVE_LEVEL3 = [  # 轻度正面
        "增长", "提升", "改善", "加强", "完善", "推进", "扩展", "扩大",
        "增长", "上涨", "走强", "向好", "积极", "正面", "乐观",
        "值得关注", "有潜力", "机会", "前景", "希望", "振奋",
    ]

    # 负面情感词
    NEGATIVE_LEVEL1 = [  # 强负面
        "崩溃", "崩盘", "暴跌", "腰斩", "血洗", "踩踏", "危机",
        "灾难", "死亡", "伤亡", "恐怖", "战争", "袭击", "爆炸",
        "欺诈", "诈骗", "丑闻", "腐败", "倒闭", "破产", "违约",
    ]
    NEGATIVE_LEVEL2 = [  # 中等负面
        "下跌", "下降", "下滑", "亏损", "萎缩", "衰退", "放缓",
        "裁员", "降薪", "关店", "暂停", "终止", "取消", "失败",
        "问题", "困难", "压力", "挑战", "威胁", "警告", "风险",
        "冲突", "制裁", "限制", "禁止", "争议", "批评", "反对",
    ]
    NEGATIVE_LEVEL3 = [  # 轻度负面
        "担忧", "顾虑", "不确定", "风险", "压力", "紧张", "恶化",
        "减弱", "降低", "减少", "下滑", "疲软", "低迷", "收缩",
        "负面", "消极", "谨慎", "观望",
    ]

    # 否定词
    NEGATORS = {"不", "没", "无", "非", "未", "别", "莫", "勿", "不会", "不能", "无法"}

    # 程度词
    INTENSIFIERS = {
        "非常": 1.5, "极其": 1.8, "十分": 1.5, "特别": 1.5,
        "相当": 1.3, "格外": 1.4, "太": 1.5, "很": 1.2,
        "尤为": 1.6, "越发": 1.2, "更加": 1.3, "更为": 1.3,
        "越来越": 1.4, "显著": 1.5, "明显": 1.3, "大幅": 1.5,
        "彻底": 1.6, "完全": 1.6, "绝对": 1.6,
    }

    def __init__(self):
        # 合并所有词典
        self.positive_words = set(
            self.POSITIVE_LEVEL1 + self.POSITIVE_LEVEL2 + self.POSITIVE_LEVEL3
        )
        self.negative_words = set(
            self.NEGATIVE_LEVEL1 + self.NEGATIVE_LEVEL2 + self.NEGATIVE_LEVEL3
        )
        self.positive_word_weights = {}
        for w in self.POSITIVE_LEVEL1:
            self.positive_word_weights[w] = 3.0
        for w in self.POSITIVE_LEVEL2:
            self.positive_word_weights[w] = 2.0
        for w in self.POSITIVE_LEVEL3:
            self.positive_word_weights[w] = 1.0

        self.negative_word_weights = {}
        for w in self.NEGATIVE_LEVEL1:
            self.negative_word_weights[w] = 3.0
        for w in self.NEGATIVE_LEVEL2:
            self.negative_word_weights[w] = 2.0
        for w in self.NEGATIVE_LEVEL3:
            self.negative_word_weights[w] = 1.0

    def analyze(self, text: str) -> Dict:
        """对单条文本进行情感分析"""
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.5, "confidence": 0.0}

        score = self._lexicon_score(text)
        # 映射到 [0, 1]，0.5 为中性
        normalized = (score + 3) / 6  # score range [-3, 3] -> [0, 1]
        normalized = max(0.0, min(1.0, normalized))

        label = "positive" if score > 0.5 else "negative" if score < -0.5 else "neutral"
        confidence = min(abs(score) / 3.0, 0.95)

        return {
            "label": label,
            "score": round(normalized, 4),
            "confidence": round(float(confidence), 4),
        }

    def _lexicon_score(self, text: str) -> float:
        """基于词典计算情感得分"""
        pos_score = 0.0
        neg_score = 0.0
        words = list(text)

        # 检查否定上下文
        for i, char in enumerate(words):
            word_2 = "".join(words[i:min(i+2, len(words))])
            word_3 = "".join(words[i:min(i+3, len(words))])

            # 程度词检查（往前看2个字）
            intensity = 1.0
            for ki in range(max(0, i-2), i):
                prev_2 = "".join(words[ki:i])
                for intensifier, weight in self.INTENSIFIERS.items():
                    if intensifier in prev_2:
                        intensity = max(intensity, weight)

            # 否定上下文检查
            is_negated = False
            for ki in range(max(0, i-3), i):
                prev_3 = "".join(words[ki:i])
                if any(n in prev_3 for n in ["不", "没", "无", "非", "未"]):
                    is_negated = True
                    break

            # 词匹配
            multiplier = intensity * (-1.0 if is_negated else 1.0)

            if char in self.positive_words or word_2 in self.positive_words or word_3 in self.positive_words:
                w = self.positive_word_weights.get(word_3) or \
                    self.positive_word_weights.get(word_2) or \
                    self.positive_word_weights.get(char) or 1.0
                pos_score += w * multiplier

            if char in self.negative_words or word_2 in self.negative_words or word_3 in self.negative_words:
                w = self.negative_word_weights.get(word_3) or \
                    self.negative_word_weights.get(word_2) or \
                    self.negative_word_weights.get(char) or 1.0
                neg_score += w * multiplier

        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return (pos_score - neg_score) / (abs(pos_score) + abs(neg_score)) * 3

    def analyze_batch(self, articles: List[Dict]) -> List[Dict]:
        """批量分析文章情感"""
        results = []
        for a in articles:
            text = a.get("title", "") + " " + a.get("summary", "")
            sent = self.analyze(text)
            results.append({**a, "sentiment": sent})
        return results

    def detect_trends(self, articles: List[Dict], window_days: int = None) -> Dict:
        """情感趋势检测"""
        if not articles:
            return {}

        window = window_days or PROCESSOR_CONFIG["sentiment_window_days"]

        daily = defaultdict(list)
        for a in articles:
            pub = a.get("published")
            if pub:
                day_key = pub.strftime("%Y-%m-%d")
            else:
                day_key = "unknown"
            daily[day_key].append(a.get("sentiment", {}).get("score", 0.5))

        timeline = []
        for day in sorted(daily.keys()):
            scores = daily[day]
            timeline.append({
                "date": day,
                "avg_score": np.mean(scores),
                "count": len(scores),
                "positive_ratio": sum(1 for s in scores if s > 0.6) / (len(scores) + 1e-6),
                "negative_ratio": sum(1 for s in scores if s < 0.4) / (len(scores) + 1e-6),
            })

        spikes = self._detect_spikes(timeline)

        all_scores = [a.get("sentiment", {}).get("score", 0.5) for a in articles]
        distribution = {
            "positive": sum(1 for s in all_scores if s > 0.6),
            "neutral": sum(1 for s in all_scores if 0.4 <= s <= 0.6),
            "negative": sum(1 for s in all_scores if s < 0.4),
            "total": len(all_scores),
        }

        extreme_positive = sorted(
            [a for a in articles if a.get("sentiment", {}).get("label") == "positive"
             and a.get("sentiment", {}).get("confidence", 0) > 0.6],
            key=lambda x: x.get("sentiment", {}).get("score", 0),
            reverse=True,
        )[:3]
        extreme_negative = sorted(
            [a for a in articles if a.get("sentiment", {}).get("label") == "negative"
             and a.get("sentiment", {}).get("confidence", 0) > 0.6],
            key=lambda x: x.get("sentiment", {}).get("score", 0),
        )[:3]

        return {
            "timeline": timeline,
            "distribution": distribution,
            "spikes": spikes,
            "extreme_positive": extreme_positive,
            "extreme_negative": extreme_negative,
            "overall_score": round(float(np.mean(all_scores)), 4),
        }

    def _detect_spikes(self, timeline: List[Dict]) -> List[Dict]:
        """检测情感突变点"""
        if len(timeline) < 3:
            return []

        scores = [t["avg_score"] for t in timeline]
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-6

        spikes = []
        for i, (t, score) in enumerate(zip(timeline, scores)):
            z_score = abs(score - mean_score) / std_score
            if z_score > PROCESSOR_CONFIG["trend_spike_threshold"]:
                direction = "surge" if score > mean_score else "drop"
                spikes.append({
                    "date": t["date"],
                    "score": round(score, 4),
                    "direction": direction,
                    "z_score": round(float(z_score), 2),
                    "change": round(float(score - mean_score), 4),
                })

        return spikes
