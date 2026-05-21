# SenseFlow - 多模态新闻智能聚合与深度分析系统

> **用深度学习解决信息过载问题 — 从海量新闻中，自动发现信号，忽略噪音。**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 项目概述

**SenseFlow** 是一个端到端的新闻智能分析系统，使用现代深度学习技术，自动完成：

| 模块 | 技术 | 说明 |
|------|------|------|
| 🔍 新闻聚合 | RSS解析 + 并发抓取 | 多语言、多源、自动去重 |
| 🧠 语义聚类 | Sentence-Transformers + 层次聚类 | 自动发现热点话题 |
| 😊 情感分析 | 中文RoBERTa + 词典备用 | 新闻级情感判断 |
| 📈 趋势检测 | 移动窗口 + Z-score | 自动识别情绪拐点 |
| 📋 AI报告 | 规则生成 | 自动输出每日简报 |

---

## 🔧 快速开始

### 1. 安装依赖

```bash
cd SenseFlow
pip install -r requirements.txt
```

### 2. 运行分析

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

### 3. 也可直接运行核心模块（无需界面）

```bash
python -c "
from src.news_fetcher import NewsFetcher
from src.text_processor import TextProcessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.report_generator import ReportGenerator

fetcher = NewsFetcher()
articles = fetcher.fetch_all()

processor = TextProcessor()
topics = processor.cluster_articles(articles)

analyzer = SentimentAnalyzer()
articles = analyzer.analyze_batch(articles)
trends = analyzer.detect_trends(articles)

report = ReportGenerator()
briefing = report.generate_daily_briefing(articles, topics, trends)
print(briefing['sentiment'])
"
```

---

## 🏗️ 项目架构

```
SenseFlow/
├── app.py                      # Streamlit 主界面 (精美Dashboard)
├── requirements.txt            # Python 依赖
├── config.py                   # 配置（新闻源/模型参数）
├── README.md
│
└── src/
    ├── news_fetcher.py         # 新闻获取（RSS + 并发）
    ├── text_processor.py       # 文本嵌入 + 语义聚类
    ├── sentiment_analyzer.py    # 深度学习情感分析
    └── report_generator.py     # AI 报告生成
```

---

## 🧠 核心技术详解

### 1. 语义 Embedding → 话题聚类

使用 **Sentence-Transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) 将每条新闻标题+摘要编码为 384 维向量，然后用**层次聚类 (Agglomerative Clustering)** 按语义相似度自动分组：

```
文本 → 清洗 → Embedding(384维) → 层次聚类 → 话题分组
```

### 2. 深度学习情感分析

**模型链路（带自动降级）：**

```
输入文本
  → RoBERTa-Chinese (uer/roberta-base-finetuned-chinanews-chinese)
  → 输出: {正面/负面/中性, 分数[-1,1], 置信度}
  → (如模型不可用) → 词典情感分析（备用）
```

**为什么选中文新闻专用模型：**
- 通用模型对中文新闻的讽刺、反讽判断不准
- 专用模型在中文财经/科技新闻上表现更好
- 自动降级确保任何环境下都能运行

### 3. 趋势检测算法

```python
# 1. 按时间窗口聚合情感均值
daily_sentiment = [mean(scores) for each_day]

# 2. 计算基准（均值 + 标准差）
baseline = mean(daily_sentiment)
std = std(daily_sentiment)

# 3. Z-score 拐点检测
for day in timeline:
    z = |score - baseline| / std
    if z > 1.5:  # 超过1.5倍标准差
        alert(f"情感突变: {day.date}")
```

---

## 🎨 界面预览

```
┌─────────────────────────────────────────────────────┐
│  🧠 SenseFlow — 新闻智能分析系统                     │
├─────────────────────────────────────────────────────┤
│  📊 今日概览                                        │
│  ┌──────┬──────┬──────┬──────┬──────┐              │
│  │文章总数│ 正面  │ 中性  │ 负面  │整体倾向│              │
│  │ 156篇 │ 78篇  │ 52篇  │ 26篇  │ 中性  │              │
│  └──────┴──────┴──────┴──────┴──────┘              │
│                                                     │
│  📈 情感趋势图  [折线图]                             │
│  📰 报道数量  [柱状图]                               │
│                                                     │
│  😊 情感分布 [饼图]  │  🔥 关键词 [条形图]            │
│                                                     │
│  🔥 热点话题聚类                                     │
│  [话题1: AI突破] [话题2: 市场动态] [话题3: 国际]     │
│                                                     │
│  ⚠️ 风险预警  │  🎯 市场信号                        │
└─────────────────────────────────────────────────────┘
```

---

## 📡 添加自定义新闻源

编辑 `config.py`：

```python
RSS_SOURCES = {
    "我的分类": [
        "https://your-feed-url.com/rss",
    ],
}
```

---

## 🔬 核心模型说明

| 模型 | 用途 | 大小 | 是否必须 |
|------|------|------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 文本向量化 | ~500MB | ✅ (自动下载) |
| `uer/roberta-base-finetuned-chinanews-chinese` | 中文情感 | ~400MB | ⚠️ (可选，降级) |
| `nlptown/bert-base-multilingual-uncased-sentiment` | 英文情感 | ~600MB | ⚠️ (可选，降级) |

**首次运行会自动下载模型**（需要网络连接）。如下载失败，系统自动降级到词典分析，保证可用。

---

## ⚙️ 系统要求

- Python 3.9+
- RAM: 4GB+ (加载模型后 ~2GB)
- 磁盘: 2GB+ (模型缓存)
- 网络: 访问RSS源（国内可能需要代理）
- GPU: 可选（默认CPU运行，CPU也能跑）

---

## 🐛 常见问题

**Q: Streamlit 界面显示中文乱码？**
```bash
# 设置环境变量后再运行
set PYTHONIOENCODING=utf-8
streamlit run app.py
```

**Q: 模型下载失败（网络问题）？**
- 系统会自动使用词典备用方案，情感分析仍可用
- 或手动设置代理：`set HTTP_PROXY=http://127.0.0.1:7890`

**Q: RSS抓取不到内容？**
- 部分网站有反爬，可尝试添加 User-Agent 或使用 NewsAPI

---

## 🚀 进阶扩展方向

- [ ] 接入 NewsAPI / 腾讯新闻 API 获取更全数据
- [ ] 使用 LLM (ChatGLM / Qwen) 生成新闻摘要
- [ ] 添加知识图谱可视化
- [ ] 实时推送（Telegram / 微信）
- [ ] 历史趋势回测

---

## 📄 License

MIT License - 可自由使用、修改、分发

---

> *"信息爆炸的时代，真正的智慧不是获取更多信息，而是从噪音中提取信号。"*
