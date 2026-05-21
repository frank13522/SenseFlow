"""
SenseFlow - 多模态新闻智能聚合与深度分析系统
配置管理
"""

# 新闻源配置
RSS_SOURCES = {
    "中文科技": [
        "https://www.36kr.com/feed",
        "https://feeds.feedburner.com/36kr",
    ],
    "中文综合": [
        "https://news.cnblogs.com/NewsRss/NewsRss.aspx",
    ],
    "英文AI": [
        "http://feeds.feedburner.com/TechCrunch/startups",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
    "英文综合": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ],
}

# API新闻源
API_SOURCES = {
    "newsapi": {
        "key": "",  # 用户填入自己的API Key
        "endpoints": {
            "top_headlines": "https://newsapi.org/v2/top-headlines",
            "everything": "https://newsapi.org/v2/everything",
        }
    }
}

# 关键词追踪（用于趋势检测）
TRACK_KEYWORDS = [
    "人工智能", "AI", "大模型", "GPT", "深度学习", "机器学习",
    "自动驾驶", "机器人", "元宇宙", "Web3", "区块链",
    "新能源", "半导体", "芯片", "量子计算",
    "美国大选", "中美关系", "俄乌战争", "台海",
    "气候变化", "碳中和", "碳达峰",
    "经济", "通胀", "美联储", "央行", "人民币",
]

# ============================================================
# 国内镜像配置（解决 HuggingFace 被墙问题）
# ============================================================
import os
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_MIRROR

# 模型配置
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "device": "cpu",
    "batch_size": 8,
}

SENTIMENT_CONFIG = {
    "model_name": "uer/roberta-base-finetuned-chinanews-chinese",
    "device": "cpu",
    "cache_dir": "./cache",
}

# 处理配置
PROCESSOR_CONFIG = {
    "min_articles_per_topic": 3,      # 最少文章数才形成话题
    "similarity_threshold": 0.65,        # 相似度阈值
    "top_n_topics": 10,               # 显示前N个话题
    "sentiment_window_days": 7,        # 情感趋势时间窗口
    "trend_spike_threshold": 1.5,      # 趋势飙升阈值（相对均值倍数）
}

# 展示配置
DISPLAY_CONFIG = {
    "articles_per_page": 20,
    "max_article_preview_chars": 300,
    "dark_mode": True,
    "accent_color": "#4F8EF7",
    "font_family": "Noto Sans SC",
}

# 缓存配置
CACHE_CONFIG = {
    "enabled": True,
    "ttl_minutes": 15,  # 缓存有效期
    "cache_dir": "./cache",
}
