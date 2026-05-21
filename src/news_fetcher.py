"""
SenseFlow - 新闻数据获取层
RSS解析 + API抓取
"""
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import RSS_SOURCES, API_SOURCES, TRACK_KEYWORDS


class NewsFetcher:
    """统一新闻获取接口"""

    def __init__(self, timeout=15, max_workers=4):
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; SenseFlow/1.0)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        })
        self._cache = {}
        self._cache_time = {}

    def fetch_all(self) -> List[Dict]:
        """并发抓取所有RSS源"""
        tasks = []
        for category, feeds in RSS_SOURCES.items():
            for feed_url in feeds:
                tasks.append((category, feed_url))

        articles = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_feed, url, cat): (cat, url)
                for cat, url in tasks
            }
            for future in as_completed(futures):
                cat, url = futures[future]
                try:
                    result = future.result()
                    articles.extend(result)
                except Exception as e:
                    print(f"[WARN] Feed fetch failed: {url} -> {e}")

        # 去重
        seen = set()
        unique = []
        for a in articles:
            fid = a.get("feed_id", a["url"])
            if fid not in seen:
                seen.add(fid)
                unique.append(a)

        print(f"[OK] 共获取 {len(unique)} 篇去重新闻")
        return unique

    def _fetch_feed(self, url: str, category: str) -> List[Dict]:
        """解析单个RSS源"""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"[WARN] Failed to fetch {url}: {e}")
            return []

        articles = []
        for entry in feed.entries[:50]:  # 每源最多50条
            article = self._parse_entry(entry, category, url)
            if article:
                articles.append(article)

        print(f"[OK] {category} / {url} -> {len(articles)} articles")
        return articles

    def _parse_entry(self, entry, category: str, feed_url: str) -> Optional[Dict]:
        """解析单条RSS条目"""
        try:
            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            summary_raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
            published = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)

            # 清理HTML标签
            soup = BeautifulSoup(summary_raw, "lxml")
            summary = soup.get_text(separator=" ", strip=True)
            if len(summary) > 500:
                summary = summary[:500] + "..."

            # 时间解析
            pub_date = None
            if published:
                try:
                    from time import mktime
                    pub_date = datetime.fromtimestamp(mktime(published))
                except Exception:
                    pub_date = datetime.now()

            return {
                "title": title.strip(),
                "url": link.strip(),
                "summary": summary.strip(),
                "published": pub_date,
                "category": category,
                "feed_id": f"{feed_url}::{link.strip()}",
                "keywords": self._extract_keywords(title + " " + summary),
            }
        except Exception as e:
            return None

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        found = []
        for kw in TRACK_KEYWORDS:
            if kw in text:
                found.append(kw)
        return found[:10]

    def fetch_by_keyword(self, keyword: str) -> List[Dict]:
        """关键词搜索（NewsAPI）"""
        api_cfg = API_SOURCES.get("newsapi", {})
        key = api_cfg.get("key", "")
        if not key:
            return []

        params = {
            "q": keyword,
            "apiKey": key,
            "language": "zh",
            "sortBy": "publishedAt",
            "pageSize": 20,
        }
        try:
            resp = requests.get(
                api_cfg["endpoints"]["everything"],
                params=params,
                timeout=self.timeout
            )
            data = resp.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "summary": item.get("description", "") or "",
                    "published": item.get("publishedAt", ""),
                    "category": keyword,
                    "source": item.get("source", {}).get("name", ""),
                    "feed_id": item.get("url", ""),
                    "keywords": [keyword],
                })
            return articles
        except Exception as e:
            print(f"[WARN] NewsAPI fetch failed: {e}")
            return []
