# -*- coding: utf-8 -*-
"""Semantic Scholar API で文献検索 (API キー不要)

Usage:
  python scripts/search_papers.py "Fourier Neural Operator guided wave"
  python scripts/search_papers.py "multi-defect SHM deep learning" --limit 20
  python scripts/search_papers.py --all   # 全トピック一括検索
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
import sys

API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

# 我々のプロジェクトに関連するトピック
TOPICS = [
    {
        "name": "FNO / Neural Operator for GW surrogate",
        "queries": [
            "Fourier Neural Operator guided wave Lamb wave surrogate",
            "neural operator wave propagation composite structure",
            "deep learning surrogate Lamb wave simulation",
        ],
    },
    {
        "name": "Multi-defect SHM",
        "queries": [
            "multi defect structural health monitoring guided wave",
            "multiple damage detection composite Lamb wave deep learning",
            "simultaneous defect characterization ultrasonic",
        ],
    },
    {
        "name": "Data augmentation for SHM / NDT",
        "queries": [
            "data augmentation structural health monitoring limited data",
            "physics-informed data augmentation guided wave",
            "GAN augmentation SHM ultrasonic",
            "transfer learning guided wave damage detection",
        ],
    },
    {
        "name": "DL surrogate for FEM simulation",
        "queries": [
            "deep learning surrogate finite element wave propagation",
            "graph neural network surrogate FEM structural",
            "physics-informed neural network Lamb wave damage",
        ],
    },
    {
        "name": "GW SHM benchmark dataset",
        "queries": [
            "benchmark dataset guided wave structural health monitoring",
            "open data ultrasonic guided wave CFRP",
        ],
    },
]

FIELDS = "title,authors,year,venue,citationCount,externalIds,abstract"


def search_semantic_scholar(query, limit=10, year_range="2021-2026"):
    """Semantic Scholar API で論文検索"""
    params = {
        "query": query,
        "limit": str(limit),
        "fields": FIELDS,
        "year": year_range,
    }
    url = API_BASE + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Payload2026-Research/1.0")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [])
    except Exception as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        return []


def format_paper(paper, idx):
    """論文を整形表示"""
    title = paper.get("title", "N/A")
    year = paper.get("year", "?")
    venue = paper.get("venue", "")
    citations = paper.get("citationCount", 0)
    authors = paper.get("authors", [])
    author_str = ", ".join(a.get("name", "") for a in authors[:4])
    if len(authors) > 4:
        author_str += " et al."

    # DOI or arXiv link
    ext_ids = paper.get("externalIds", {})
    doi = ext_ids.get("DOI", "")
    arxiv = ext_ids.get("ArXiv", "")
    link = ""
    if doi:
        link = f"https://doi.org/{doi}"
    elif arxiv:
        link = f"https://arxiv.org/abs/{arxiv}"

    abstract = paper.get("abstract", "")
    if abstract and len(abstract) > 200:
        abstract = abstract[:200] + "..."

    lines = []
    lines.append(f"  [{idx}] {title} ({year})")
    lines.append(f"      Authors: {author_str}")
    if venue:
        lines.append(f"      Venue: {venue}")
    lines.append(f"      Citations: {citations}")
    if link:
        lines.append(f"      URL: {link}")
    if abstract:
        lines.append(f"      Abstract: {abstract}")
    return "\n".join(lines)


def search_all_topics(limit_per_query=5):
    """全トピックを一括検索"""
    seen_titles = set()
    results_by_topic = {}

    for topic in TOPICS:
        print(f"\n{'=' * 70}")
        print(f"  {topic['name']}")
        print(f"{'=' * 70}")

        topic_papers = []
        for query in topic["queries"]:
            print(f"\n  Query: \"{query}\"")
            papers = search_semantic_scholar(query, limit=limit_per_query)
            time.sleep(3.5)  # API rate limit (100 req/5min, conservative)

            for p in papers:
                title = p.get("title", "")
                if title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    topic_papers.append(p)

        # Sort by citations (descending)
        topic_papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)

        print(f"\n  --- Top results ({len(topic_papers)} unique papers) ---")
        for i, p in enumerate(topic_papers[:10], 1):
            print(format_paper(p, i))
            print()

        results_by_topic[topic["name"]] = topic_papers

    return results_by_topic


def search_single(query, limit=10):
    """単一クエリで検索"""
    print(f"Searching: \"{query}\" (limit={limit})")
    papers = search_semantic_scholar(query, limit=limit)
    papers.sort(key=lambda p: p.get("citationCount", 0), reverse=True)

    for i, p in enumerate(papers, 1):
        print(format_paper(p, i))
        print()

    return papers


def main():
    parser = argparse.ArgumentParser(description="Semantic Scholar 文献検索")
    parser.add_argument("query", nargs="?", default=None, help="検索クエリ")
    parser.add_argument("--limit", type=int, default=10, help="結果数")
    parser.add_argument("--all", action="store_true", help="全トピック一括検索")
    parser.add_argument("--json", type=str, default=None, help="結果をJSONに保存")
    args = parser.parse_args()

    if args.all:
        results = search_all_topics(limit_per_query=args.limit)
        if args.json:
            # Flatten for JSON export
            flat = {}
            for topic, papers in results.items():
                flat[topic] = [
                    {
                        "title": p.get("title"),
                        "year": p.get("year"),
                        "venue": p.get("venue"),
                        "citations": p.get("citationCount"),
                        "authors": [a.get("name") for a in p.get("authors", [])],
                        "doi": p.get("externalIds", {}).get("DOI"),
                        "arxiv": p.get("externalIds", {}).get("ArXiv"),
                        "abstract": (p.get("abstract") or "")[:300],
                    }
                    for p in papers[:10]
                ]
            with open(args.json, "w") as f:
                json.dump(flat, f, indent=2, ensure_ascii=False)
            print(f"\n  Results saved: {args.json}")
    elif args.query:
        search_single(args.query, limit=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
