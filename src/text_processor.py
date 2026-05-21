"""
SenseFlow - 文本处理与语义聚类
使用 TF-IDF + 余弦相似度（纯本地，无需下载模型）
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict
import jieba
import re

from config import PROCESSOR_CONFIG


def jieba_tokenizer(text: str) -> List[str]:
    """结巴分词"""
    words = jieba.cut(text)
    return [w for w in words if len(w) >= 2 and not w.isdigit()]


class TextProcessor:
    """TF-IDF 向量化 + 层次聚类"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=jieba_tokenizer,
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
        )
        self.embeddings = None
        self.texts = []

    def embed(self, texts: List[str]) -> np.ndarray:
        """TF-IDF 向量化"""
        clean = [self._clean_text(t) for t in texts]
        self.texts = clean
        embeddings = self.vectorizer.fit_transform(clean).toarray()
        self.embeddings = embeddings
        return embeddings

    def _clean_text(self, text: str) -> str:
        """中文预处理"""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[\r\n\t]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:1500]

    def cluster_articles(self, articles: List[Dict]) -> List[Dict]:
        """对新闻文章进行话题聚类"""
        if len(articles) < 3:
            return [{"topic": "综合新闻", "articles": articles, "count": len(articles), "score": 0.0}]

        texts = [a.get("title", "") + ". " + a.get("summary", "") for a in articles]
        embeddings = self.embed(texts)

        n_clusters = min(len(articles), PROCESSOR_CONFIG["top_n_topics"] * 2)
        if len(articles) < 10:
            n_clusters = max(1, len(articles) // 3)

        # 用余弦距离做层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="complete",
        )
        labels = clustering.fit_predict(embeddings)

        topics = {}
        for i, label in enumerate(labels):
            if label not in topics:
                topics[label] = []
            topics[label].append((i, embeddings[i]))

        # 生成话题标题
        result = []
        for label, items in topics.items():
            indices = [idx for idx, _ in items]
            topic_articles = [articles[idx] for idx in indices]
            representative_title = self._get_representative(indices, texts)
            score = self._compute_topic_score(indices)

            result.append({
                "topic": representative_title,
                "articles": topic_articles,
                "count": len(topic_articles),
                "score": score,
            })

        result.sort(key=lambda x: (x["count"], x["score"]), reverse=True)
        return result[:PROCESSOR_CONFIG["top_n_topics"]]

    def _get_representative(self, indices: List[int], texts: List[str]) -> str:
        """提取话题代表标题（TF-IDF 权重最高的文章）"""
        if not indices or self.embeddings is None:
            return "综合新闻"
        vecs = self.embeddings[indices]
        # 取每个词的 TF-IDF 权重和最高的文章
        scores = vecs.sum(axis=1)
        best_idx = indices[int(np.argmax(scores))]
        title = texts[best_idx]
        return title[:60] + ("..." if len(title) > 60 else "")

    def _compute_topic_score(self, indices: List[int]) -> float:
        """计算话题内聚度分数"""
        if len(indices) < 2 or self.embeddings is None:
            return 0.0
        vecs = self.embeddings[indices]
        center = vecs.mean(axis=0)
        avg_dist = float(np.mean([np.linalg.norm(v - center) for v in vecs]))
        coherence = 1.0 / (1.0 + avg_dist)
        return coherence * len(indices)


class KeywordExtractor:
    """关键词自动提取（TF-IDF 词频）"""

    STOPWORDS = set([
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
        "看", "好", "自己", "这", "那", "它", "他", "她", "们", "这个", "那个",
        "什么", "怎么", "为什么", "如何", "可以", "已经", "可能", "应该", "因为",
        "所以", "但是", "如果", "虽然", "或者", "而且", "以及", "一个", "一些",
        "一定", "一样", "一起", "一直", "已经", "正在", "目前", "现在", "今天",
        "根据", "通过", "关于", "对于", "作为", "因此", "此外", "另外", "其中",
        "之后", "之前", "以后", "以来", "开始", "进行", "完成", "实现", "包括",
        "属于", "位于", "成为", "存在", "使用", "提供", "需要", "能够", "希望",
        "相关", "这种", "各种", "其他", "另外", "最新", "更", "还", "又", "再",
        "只是", "只有", "只要", "甚至",
    ])

    def extract(self, articles: List[Dict], top_k=20) -> List[Dict]:
        """从文章列表提取关键词"""
        all_text = " ".join([
            a.get("title", "") + " " + a.get("summary", "")
            for a in articles
        ])
        words = jieba.cut(all_text)
        freq = {}
        for w in words:
            if len(w) < 2 or w in self.STOPWORDS or w.isdigit():
                continue
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [{"keyword": k, "count": v} for k, v in sorted_words[:top_k]]
