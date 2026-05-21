"""
SenseFlow - 报告生成器
AI 驱动的新闻摘要与洞察报告
"""
import numpy as np
from typing import List, Dict
from datetime import datetime
from collections import Counter
import re


class ReportGenerator:
    """自动生成新闻分析报告"""

    def __init__(self):
        pass

    def generate_daily_briefing(self, articles: List[Dict], topics: List[Dict],
                                 trends: Dict) -> Dict:
        """生成每日简报"""
        if not articles:
            return {}

        # 概览
        overview = {
            "total_articles": len(articles),
            "categories": self._count_by_key(articles, "category"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        # 今日关键词
        keywords = self._extract_top_keywords(articles, top_k=15)

        # 热点话题摘要
        topic_summaries = []
        for t in topics[:5]:
            topic_articles = t["articles"]
            summary_text = self._summarize_articles(topic_articles)
            topic_summaries.append({
                "title": t["topic"],
                "count": t["count"],
                "summary": summary_text,
                "sample_articles": [
                    {"title": a["title"], "url": a["url"]}
                    for a in topic_articles[:2]
                ],
            })

        # 情感洞察
        sentiment_insight = self._generate_sentiment_insight(trends)

        # 风险预警
        risk_alerts = self._detect_risk_articles(articles)

        # 投资建议（基于情感分析）
        investment_signals = self._generate_investment_signals(articles, trends)

        return {
            "overview": overview,
            "keywords": keywords,
            "topics": topic_summaries,
            "sentiment": sentiment_insight,
            "risks": risk_alerts,
            "signals": investment_signals,
        }

    def _count_by_key(self, articles: List[Dict], key: str) -> Dict:
        """按字段统计"""
        counter = Counter(a.get(key, "unknown") for a in articles)
        return dict(counter)

    def _extract_top_keywords(self, articles: List[Dict], top_k=15) -> List[Dict]:
        """提取高频词"""
        all_text = " ".join(a.get("title", "") + " " + a.get("summary", "") for a in articles)
        # 简单词频统计
        import jieba
        words = jieba.cut(all_text)
        stopwords = set([
            "的", "了", "在", "是", "和", "就", "不", "都", "一", "一个", "上", "也",
            "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "这", "那", "它", "他", "她", "们", "这个", "一个", "可能", "已经",
            "一个", "一些", "这个", "这个", "根据", "通过", "关于", "对于",
        ])
        freq = {}
        for w in words:
            if len(w) < 2 or w in stopwords or w.isdigit() or not re.search(r"[\u4e00-\u9fff]", w):
                continue
            freq[w] = freq.get(w, 0) + 1
        return [{"word": k, "count": v} for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    def _summarize_articles(self, articles: List[Dict]) -> str:
        """生成话题摘要"""
        titles = [a.get("title", "") for a in articles[:5]]
        if not titles:
            return ""
        # 取最长的共同子串作为主题
        combined = " | ".join(titles[:3])
        if len(combined) > 120:
            combined = combined[:120] + "..."
        return combined

    def _generate_sentiment_insight(self, trends: Dict) -> str:
        """生成情感洞察文字"""
        dist = trends.get("distribution", {})
        total = dist.get("total", 1)
        pos_r = dist.get("positive", 0) / total * 100
        neg_r = dist.get("negative", 0) / total * 100

        overall = trends.get("overall_score", 0.5)
        if overall > 0.6:
            tone = "偏正面"
        elif overall < 0.4:
            tone = "偏负面"
        else:
            tone = "中性"

        spikes = trends.get("spikes", [])
        spike_text = ""
        if spikes:
            last = spikes[-1]
            spike_text = f"注意：{last['date']} 出现情感突变（{last['direction']=='surge' and '正面' or '负面'}飙升）"

        return (
            f"今日新闻整体{tone}，正面占比约 {pos_r:.0f}%，负面约 {neg_r:.0f}%。"
            f"综合情感指数 {overall:.2f}（满分1.0）。{spike_text}"
        )

    def _detect_risk_articles(self, articles: List[Dict]) -> List[Dict]:
        """检测风险文章（负面情感 + 高置信度）"""
        risk_keywords = [
            "危机", "崩盘", "暴跌", "战争", "死亡", "恐怖", "事故",
            "欺诈", "丑闻", "制裁", "禁令", "下架", "亏损", "破产",
            "裁员", "违约", "诈骗", "爆炸", "袭击", "灾害",
        ]
        risk_articles = []
        for a in articles:
            sent = a.get("sentiment", {})
            if sent.get("label") == "negative" and sent.get("confidence", 0) > 0.65:
                text = a.get("title", "") + a.get("summary", "")
                if any(kw in text for kw in risk_keywords):
                    risk_articles.append({
                        "title": a.get("title", "")[:80],
                        "url": a.get("url", ""),
                        "sentiment": sent,
                    })
        return risk_articles[:5]

    def _generate_investment_signals(self, articles: List[Dict], trends: Dict) -> List[str]:
        """基于新闻情感生成投资信号"""
        signals = []
        dist = trends.get("distribution", {})
        total = dist.get("total", 1)
        pos_r = dist.get("positive", 0) / total
        neg_r = dist.get("negative", 0) / total

        if pos_r > 0.6:
            signals.append({
                "type": "bullish",
                "icon": "📈",
                "text": "市场情绪乐观，正面新闻占主导，短期偏多",
            })
        if neg_r > 0.4:
            signals.append({
                "type": "bearish",
                "icon": "📉",
                "text": "负面情绪升温，风险事件需警惕，建议谨慎",
            })
        if pos_r > 0.5 and neg_r < 0.2:
            signals.append({
                "type": "strong_bullish",
                "icon": "🚀",
                "text": "市场信心强劲，积极信号明显",
            })
        if neg_r > 0.5:
            signals.append({
                "type": "risk_alert",
                "icon": "⚠️",
                "text": "风险偏好收缩，建议做好对冲准备",
            })

        if not signals:
            signals.append({
                "type": "neutral",
                "icon": "➡️",
                "text": "多空情绪均衡，等待方向明确",
            })

        return signals

    def format_markdown_report(self, briefing: Dict) -> str:
        """输出 Markdown 格式报告"""
        lines = []
        lines.append(f"# 📰 SenseFlow 每日新闻简报")
        lines.append(f"\n**生成时间：** {briefing['overview']['generated_at']}")
        lines.append(f"**覆盖文章：** {briefing['overview']['total_articles']} 篇\n")

        lines.append("## 🔥 今日热点话题\n")
        for i, t in enumerate(briefing["topics"], 1):
            lines.append(f"**{i}. {t['title']}** ({t['count']} 篇)")
            lines.append(f"> {t['summary']}")
            lines.append("")

        lines.append("## 💡 关键词云\n")
        words = [f"`{k['word']}`({k['count']})" for k in briefing["keywords"][:12]]
        lines.append(" ".join(words))
        lines.append("")

        lines.append("## 😊 情感洞察\n")
        lines.append(briefing["sentiment"])
        lines.append("")

        if briefing["risks"]:
            lines.append("## ⚠️ 风险预警\n")
            for r in briefing["risks"]:
                lines.append(f"- [{r['title']}]({r['url']})")
            lines.append("")

        lines.append("## 🎯 投资信号\n")
        for s in briefing["signals"]:
            lines.append(f"{s['icon']} **{s['type'].upper()}**: {s['text']}")
        lines.append("")

        return "\n".join(lines)
