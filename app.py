"""
SenseFlow - 多模态新闻智能聚合与深度分析系统
主界面 - Streamlit Dashboard
=======================================
解决现实问题：信息过载时代，从噪音中找到真正重要的信号。
核心技术：Embedding聚类 + 情感分析 + 趋势检测 + AI摘要
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
from datetime import datetime

# ============================================================
# 必须在导入模型前设置 HuggingFace 镜像（国内网络）
# ============================================================
import config as _cfg  # noqa: F401  (config.py 内部自动设置 HF_ENDPOINT)
from src.report_generator import ReportGenerator

# 设置中文字体支持
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "WenQuanYi Micro Hei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 页面配置
st.set_page_config(
    page_title="SenseFlow - 新闻智能分析",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 自定义 CSS =====
st.markdown("""
<style>
/* 全局字体 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

* {
    font-family: 'Noto Sans SC', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
}

/* 顶部品牌栏 */
.main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 1.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(79, 142, 247, 0.3);
}

.main-header h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #00d2ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-header p {
    color: rgba(255,255,255,0.7);
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}

/* 指标卡片 */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(79,142,247,0.15);
}
.metric-number {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4F8EF7, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.3rem;
}

/* 文章卡片 */
.article-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #4F8EF7;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    transition: all 0.2s;
}
.article-card:hover {
    border-left-color: #7b2ff7;
    box-shadow: 0 4px 16px rgba(79,142,247,0.12);
}
.article-title {
    font-size: 1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.4rem;
    line-height: 1.5;
}
.article-meta {
    font-size: 0.78rem;
    color: #999;
}
.article-summary {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* 情感标签 */
.sentiment-positive {
    background: linear-gradient(135deg, #00c853, #69f0ae);
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.sentiment-negative {
    background: linear-gradient(135deg, #ff1744, #ff6e40);
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.sentiment-neutral {
    background: linear-gradient(135deg, #78909c, #b0bec5);
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* 话题分组 */
.topic-group {
    background: linear-gradient(135deg, #f8f9ff, #eef1ff);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(79,142,247,0.1);
}

/* 侧边栏 */
.css-1d391kg {
    background: #f8f9ff;
}

/* 加载动画 */
.loading-spinner {
    text-align: center;
    padding: 3rem;
    color: #4F8EF7;
}

/* 图表容器 */
.chart-container {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ===== 核心逻辑 =====
@st.cache_data(ttl=900)
def run_analysis():
    """执行完整分析流程（无外部模型依赖，纯本地运行）"""
    sys.path.insert(0, os.path.dirname(__file__))

    from src.news_fetcher import NewsFetcher
    from src.text_processor import TextProcessor, KeywordExtractor
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.report_generator import ReportGenerator

    with st.spinner("正在抓取新闻..."):
        fetcher = NewsFetcher(timeout=12)
        articles = fetcher.fetch_all()

    if not articles:
        return None

    with st.spinner("正在进行语义聚类（TF-IDF）..."):
        processor = TextProcessor()
        topics = processor.cluster_articles(articles)

    with st.spinner("正在进行情感分析（词典引擎）..."):
        analyzer = SentimentAnalyzer()
        articles_analyzed = analyzer.analyze_batch(articles)
        trends = analyzer.detect_trends(articles_analyzed)

    with st.spinner("正在生成分析报告..."):
        kw_extractor = KeywordExtractor()
        keywords = kw_extractor.extract(articles)
        report_gen = ReportGenerator()
        briefing = report_gen.generate_daily_briefing(articles_analyzed, topics, trends)

    return {
        "articles": articles_analyzed,
        "topics": topics,
        "trends": trends,
        "keywords": keywords,
        "briefing": briefing,
    }




# ===== 侧边栏 =====
with st.sidebar:
    st.markdown("## 🧠 SenseFlow")
    st.markdown("新闻智能分析系统")
    st.divider()

    st.markdown("### ⚙️ 设置")
    auto_refresh = st.checkbox("自动刷新 (每15分钟)", value=False)

    st.markdown("### 📊 数据源")
    st.caption("✓ 中文科技新闻")
    st.caption("✓ 中文综合新闻")
    st.caption("✓ 英文AI/科技")
    st.caption("✓ 英文综合")

    st.divider()
    st.markdown("### ℹ️ 关于")
    st.caption("SenseFlow v1.0")
    st.caption("TF-IDF 语义聚类 + 词典情感分析")

    if st.button("🔄 立即刷新数据", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ===== 主界面 =====
# 顶部品牌栏
st.markdown("""
<div class="main-header">
    <h1>🧠 SenseFlow</h1>
    <p>多模态新闻智能聚合 · 深度情感分析 · 热点趋势检测 · AI 驱动洞察</p>
</div>
""", unsafe_allow_html=True)

# 分析按钮
col_main = st.columns([1])
with col_main[0]:
    if st.button("🚀 开始智能分析", type="primary", use_container_width=True):
        with st.spinner("正在执行深度分析，请稍候..."):
            result = run_analysis()
            if result:
                st.session_state["result"] = result
                st.success("分析完成！")
            else:
                st.error("未能获取新闻数据，请检查网络后重试。")

# ===== 结果展示 =====
if "result" in st.session_state:
    result = st.session_state["result"]
    articles = result["articles"]
    topics = result["topics"]
    trends = result["trends"]
    keywords = result["keywords"]
    briefing = result["briefing"]

    # ===== 概览指标 =====
    st.markdown("## 📊 今日概览")

    dist = trends.get("distribution", {})
    total = dist.get("total", 1)
    pos_n = dist.get("positive", 0)
    neg_n = dist.get("negative", 0)
    neu_n = dist.get("neutral", 0)
    overall = trends.get("overall_score", 0.5)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📰 文章总数", f"{total}", delta=f"{total} 篇")
    with col2:
        st.metric("😊 正面", f"{pos_n}篇", delta=f"{pos_n/total*100:.1f}%")
    with col3:
        st.metric("😐 中性", f"{neu_n}篇", delta=f"{neu_n/total*100:.1f}%")
    with col4:
        st.metric("😞 负面", f"{neg_n}篇", delta=f"{neg_n/total*100:.1f}%")
    with col5:
        tone = "乐观" if overall > 0.55 else "谨慎" if overall < 0.45 else "中性"
        st.metric("🎯 整体倾向", tone, delta=f"指数 {overall:.2f}")

    st.divider()

    # ===== 情感趋势图 =====
    if trends.get("timeline"):
        st.markdown("## 📈 情感趋势分析")

        timeline_data = trends["timeline"]
        dates = [t["date"] for t in timeline_data]
        scores = [t["avg_score"] for t in timeline_data]
        counts = [t["count"] for t in timeline_data]

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.65, 0.35],
            subplot_titles=("📈 情感指数趋势 (1.0=正面, 0.0=负面)", "📰 报道数量"),
            vertical_spacing=0.12,
        )

        # 情感曲线
        fig.add_trace(
            go.Scatter(
                x=dates, y=scores,
                mode="lines+markers",
                name="情感指数",
                line=dict(color="#4F8EF7", width=3),
                marker=dict(size=8, color="#4F8EF7"),
                fill="tozeroy",
                fillcolor="rgba(79,142,247,0.1)",
                text=[f"{s:.3f}" for s in scores],
                hovertemplate="日期: %{x}<br>指数: %{text}<extra></extra>",
            ),
            row=1, col=1,
        )
        # 正面阈值线
        fig.add_hline(y=0.6, line_dash="dash", line_color="#00c853", opacity=0.5, row=1, col=1)
        # 负面阈值线
        fig.add_hline(y=0.4, line_dash="dash", line_color="#ff1744", opacity=0.5, row=1, col=1)

        # 文章数量柱状图
        fig.add_trace(
            go.Bar(
                x=dates, y=counts,
                name="文章数",
                marker_color="rgba(123,47,247,0.5)",
                text=counts,
                textposition="outside",
            ),
            row=2, col=1,
        )

        fig.update_layout(
            height=500,
            showlegend=False,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, max(counts) * 1.3 if counts else 10], row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

    # ===== 情感分布 =====
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 😊 情感分布")
        if dist.get("total", 0) > 0:
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=["😊 正面", "😐 中性", "😞 负面"],
                    values=[pos_n, neu_n, neg_n],
                    marker=dict(colors=["#00c853", "#78909c", "#ff1744"]),
                    textinfo="label+percent",
                    textfont_size=13,
                    hole=0.55,
                )
            ])
            fig_pie.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="white",
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("### 🔥 关键词热度")
        if keywords:
            kw_text = " ".join([f"{k['keyword']} " * min(k['count'], 3) for k in keywords[:20]])
            # 词频柱状图
            kw_df = pd.DataFrame(keywords[:15])
            fig_bar = px.bar(
                kw_df,
                x="count",
                y="keyword",
                orientation="h",
                title="",
                color="count",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_bar.update_coloraxes(showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ===== 热点话题 =====
    st.markdown("## 🔥 热点话题聚类")

    tabs = st.tabs([f"话题 {i+1}: {t['topic'][:20]}" for i, t in enumerate(topics[:6])])

    for ti, tab in enumerate(tabs):
        with tab:
            t = topics[ti]
            # 话题情感
            t_articles = t["articles"]
            t_sentiments = [a.get("sentiment", {}).get("score", 0.5) for a in t_articles]
            avg_sent = sum(t_sentiments) / len(t_sentiments) if t_sentiments else 0.5

            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                st.markdown(f"**{t['topic']}**")
                st.caption(f"共 {t['count']} 篇相关文章")

                for a in t_articles[:5]:
                    sent = a.get("sentiment", {})
                    label = sent.get("label", "neutral")
                    score = sent.get("score", 0.5)
                    conf = sent.get("confidence", 0)

                    if label == "positive":
                        sent_html = f'<span class="sentiment-positive">正面 {score:.0%}</span>'
                    elif label == "negative":
                        sent_html = f'<span class="sentiment-negative">负面 {score:.0%}</span>'
                    else:
                        sent_html = f'<span class="sentiment-neutral">中性</span>'

                    pub_time = a.get("published", "")
                    time_str = pub_time.strftime("%m-%d %H:%M") if pub_time else ""

                    st.markdown(f"""
                    <div class="article-card">
                        <div class="article-title">{a['title']}</div>
                        <div class="article-meta">
                            {a.get('category','')} &nbsp;|&nbsp; {time_str} &nbsp;|&nbsp; {sent_html}
                        </div>
                        {f'<div class="article-summary">{a.get("summary","")[:200]}...</div>' if a.get("summary") else ""}
                    </div>
                    """, unsafe_allow_html=True)

            with col_t2:
                # 话题情感指标
                st.markdown("**话题情感**")
                sent_display = "😊 正面主导" if avg_sent > 0.6 else "😞 负面主导" if avg_sent < 0.4 else "😐 中性"
                st.info(sent_display)
                st.caption(f"情感指数: {avg_sent:.2f}")

                if t_sentiments:
                    fig_topic = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_sent,
                        gauge=dict(
                            axis=dict(range=[0, 1], ticks=""),
                            bar=dict(color="#4F8EF7"),
                            bgcolor="white",
                            borderwidth=0,
                            steps=[
                                dict(range=[0, 0.4], color="#ffcdd2"),
                                dict(range=[0.4, 0.6], color="#eceff1"),
                                dict(range=[0.6, 1.0], color="#c8e6c9"),
                            ],
                        ),
                        number=dict(suffix="", valueformat=".2f"),
                    ))
                    fig_topic.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_topic, use_container_width=True)

    st.divider()

    # ===== 风险预警 =====
    if briefing.get("risks"):
        st.markdown("### ⚠️ 风险预警")

        for r in briefing["risks"]:
            with st.expander(f"⚠️ {r['title']}", expanded=False):
                st.markdown(f"**情感分析：** {r['sentiment'].get('label','')} (置信度 {r['sentiment'].get('confidence',0):.0%})")
                st.markdown(f"**链接：** [{r['url']}]({r['url']})")

    # ===== 投资信号 =====
    st.markdown("### 🎯 市场信号")
    signals = briefing.get("signals", [])

    cols = st.columns(len(signals) if signals else 1)
    for si, sig in enumerate(signals):
        with cols[si % len(cols)]:
            icon_map = {"bullish": "📈", "bearish": "📉", "strong_bullish": "🚀",
                       "risk_alert": "⚠️", "neutral": "➡️"}
            color_map = {
                "bullish": "success", "bearish": "error",
                "strong_bullish": "success", "risk_alert": "warning", "neutral": "info"
            }
            st.markdown(f"**{icon_map.get(sig['type'], '➡️')} {sig['text']}**")

    st.divider()

    # ===== AI 每日简报 =====
    st.markdown("## 📋 AI 每日简报")
    report = briefing.get("sentiment", "")
    st.info(f"**情感洞察：** {report}")

    # 极端情绪文章
    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
        st.markdown("### 🟢 最正面文章")
        ep = trends.get("extreme_positive", [])
        if ep:
            for a in ep[:3]:
                st.markdown(f"- **{a['title'][:60]}** ({a.get('sentiment',{}).get('score',0):.2f})")
        else:
            st.caption("暂无明显正面文章")

    with col_ext2:
        st.markdown("### 🔴 最负面文章")
        en = trends.get("extreme_negative", [])
        if en:
            for a in en[:3]:
                st.markdown(f"- **{a['title'][:60]}** ({a.get('sentiment',{}).get('score',0):.2f})")
        else:
            st.caption("暂无明显负面文章")

    # ===== Markdown 报告下载 =====
    report_gen = ReportGenerator()
    md_report = report_gen.format_markdown_report(briefing)
    st.download_button(
        "📥 下载 Markdown 报告",
        data=md_report,
        file_name=f"senseflow_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

else:
    # 空状态展示
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#666;">
        <h2 style="color:#4F8EF7;">🔍 点击上方按钮开始智能分析</h2>
        <p style="font-size:1.1rem; max-width:600px; margin:1rem auto;">
            SenseFlow 将自动抓取多源新闻，进行深度情感分析、话题聚类和趋势检测，
            让你从海量信息中快速把握真正重要的信号。
        </p>
        <div style="display:flex; gap:2rem; justify-content:center; margin-top:2rem; flex-wrap:wrap;">
            <div class="metric-card">
                <div class="metric-number">📰</div>
                <div class="metric-label">多源聚合</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">🧠</div>
                <div class="metric-label">深度学习</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">😊</div>
                <div class="metric-label">情感分析</div>
            </div>
            <div class="metric-card">
                <div class="metric-number">📈</div>
                <div class="metric-label">趋势检测</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; font-size:0.85rem;">
        SenseFlow v1.0 · 基于深度学习的新闻智能分析系统 · 使用 Streamlit + Transformers 构建
    </div>
    """, unsafe_allow_html=True)
