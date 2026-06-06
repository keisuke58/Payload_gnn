# -*- coding: utf-8 -*-
"""
文献 PDF ダウンローダー

Semantic Scholar API で DOI/arXiv を取得し、
arXiv or Unpaywall 経由で PDF をダウンロード。

Usage:
  python scripts/download_papers.py                    # 全重要論文
  python scripts/download_papers.py --query "FNO wave"  # 検索してDL
"""

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import sys

PDF_DIR = "papers"

# プロジェクトに関連する重要論文リスト
# arXiv ID があるものは直接ダウンロード可能
PAPERS = [
    # === FNO / Neural Operator ===
    {
        "id": "li2021fno",
        "title": "Fourier Neural Operator for Parametric PDEs",
        "arxiv": "2010.08895",
        "priority": "high",
    },
    {
        "id": "lehmann2024mifno",
        "title": "MIFNO for 3D elastodynamics (HEMEW3D)",
        "arxiv": "2404.10115",
        "priority": "high",
    },
    {
        "id": "li2023gino",
        "title": "GINO: Geometry-Informed Neural Operator",
        "arxiv": "2309.00583",
        "priority": "high",
    },
    {
        "id": "wen2022ufno",
        "title": "U-FNO: Enhanced FNO with U-net skip connections",
        "arxiv": "2109.03697",
        "priority": "medium",
    },
    {
        "id": "rai2023deeponet_lamb",
        "title": "DeepONet for Lamb Wave in Composite Laminates",
        "doi": "10.1016/j.compstruct.2023.117207",
        "priority": "high",
    },
    # === Multi-Defect SHM ===
    {
        "id": "zhao2023gnn_shm",
        "title": "GNN-Based Damage Detection in SHM",
        "doi": "10.3390/s23208525",
        "priority": "high",
    },
    {
        "id": "rautela2022multi_damage",
        "title": "Multiple Damage in Composites Using Lamb Waves + ML",
        "doi": "10.1016/j.compstruct.2021.114947",
        "priority": "medium",
    },
    # === Data Augmentation ===
    {
        "id": "liao2023physics_aug",
        "title": "Physics-Informed Data Augmentation for GW",
        "doi": "10.1177/14759217221124744",
        "priority": "medium",
    },
    {
        "id": "sawant2023fewshot",
        "title": "Few-Shot Learning for GW Damage Detection",
        "doi": "10.1016/j.ndteint.2023.102844",
        "priority": "high",
    },
    # === DL Surrogate ===
    {
        "id": "pfaff2021meshgraphnets",
        "title": "MeshGraphNets (DeepMind)",
        "arxiv": "2010.03409",
        "priority": "medium",
    },
    {
        "id": "shukla2023pinn_lamb",
        "title": "PINN for Lamb Wave + Damage Detection",
        "arxiv": "2005.03596",
        "priority": "medium",
    },
    # === PI-FNO / Wave ===
    {
        "id": "song2023pifno",
        "title": "PI-FNO for Wave Equations",
        "arxiv": "2308.01823",
        "priority": "medium",
    },
    {
        "id": "wang2023neural_op_longtime",
        "title": "Neural Operator for Long-Time Integration",
        "arxiv": "2304.07116",
        "priority": "low",
    },
    # === Review ===
    {
        "id": "sony2023data_aug_review",
        "title": "Data Augmentation for DL-Based SHM: A Review",
        "doi": "10.3390/s23063214",
        "priority": "low",
    },
]

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER = "https://api.semanticscholar.org/graph/v1/paper"


def download_file(url, dest, timeout=30):
    """Download a file with proper headers."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent",
                   "Mozilla/5.0 (Payload2026-Research/1.0)")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 5000:
                # Too small, probably an error page
                return False
            with open(dest, 'wb') as f:
                f.write(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"    Download failed: {e}")
        return False


def download_arxiv(arxiv_id, dest):
    """Download PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"    arXiv: {url}")
    return download_file(url, dest, timeout=60)


def download_unpaywall(doi, dest):
    """Try to get open-access PDF via Unpaywall API."""
    email = "keisuke58@keio.jp"
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        req = urllib.request.Request(api_url)
        req.add_header("User-Agent", "Payload2026-Research/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        best_oa = data.get("best_oa_location", {})
        if best_oa and best_oa.get("url_for_pdf"):
            pdf_url = best_oa["url_for_pdf"]
            print(f"    Unpaywall: {pdf_url}")
            return download_file(pdf_url, dest, timeout=60)
    except Exception as e:
        print(f"    Unpaywall lookup failed: {e}")
    return False


def search_semantic_scholar(query, limit=3):
    """Search Semantic Scholar for paper metadata."""
    params = {
        "query": query,
        "limit": str(limit),
        "fields": "title,externalIds,openAccessPdf,year",
    }
    url = S2_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Payload2026-Research/1.0")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("data", [])
    except Exception as e:
        print(f"    S2 search failed: {e}")
        return []


def download_paper(paper, pdf_dir):
    """Try to download a single paper PDF."""
    paper_id = paper["id"]
    dest = os.path.join(pdf_dir, f"{paper_id}.pdf")

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  [{paper_id}] Already downloaded ({size_mb:.1f} MB)")
        return True

    print(f"  [{paper_id}] {paper['title']}")

    # Method 1: arXiv direct
    if "arxiv" in paper:
        if download_arxiv(paper["arxiv"], dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"    OK ({size_mb:.1f} MB)")
            return True

    # Method 2: DOI → Unpaywall
    if "doi" in paper:
        if download_unpaywall(paper["doi"], dest):
            size_mb = os.path.getsize(dest) / 1e6
            print(f"    OK ({size_mb:.1f} MB)")
            return True

    # Method 3: Search Semantic Scholar for open access PDF
    search_query = paper.get("search", paper["title"])
    results = search_semantic_scholar(search_query)
    time.sleep(1.5)

    for r in results:
        # Check for open access PDF
        oa_pdf = r.get("openAccessPdf", {})
        if oa_pdf and oa_pdf.get("url"):
            print(f"    S2 OA: {oa_pdf['url']}")
            if download_file(oa_pdf["url"], dest, timeout=60):
                size_mb = os.path.getsize(dest) / 1e6
                print(f"    OK ({size_mb:.1f} MB)")
                return True

        # Try arXiv from external IDs
        ext_ids = r.get("externalIds", {})
        arxiv_id = ext_ids.get("ArXiv")
        if arxiv_id:
            if download_arxiv(arxiv_id, dest):
                size_mb = os.path.getsize(dest) / 1e6
                print(f"    OK ({size_mb:.1f} MB)")
                return True

        # Try DOI → Unpaywall
        doi = ext_ids.get("DOI")
        if doi:
            if download_unpaywall(doi, dest):
                size_mb = os.path.getsize(dest) / 1e6
                print(f"    OK ({size_mb:.1f} MB)")
                return True

    print(f"    FAILED (no open-access PDF found)")
    return False


def main():
    parser = argparse.ArgumentParser(description="文献 PDF ダウンローダー")
    parser.add_argument("--dir", default=PDF_DIR,
                        help="PDF 保存ディレクトリ")
    parser.add_argument("--priority", default="all",
                        choices=["high", "medium", "low", "all"],
                        help="優先度フィルタ")
    parser.add_argument("--query", default=None,
                        help="Semantic Scholar で検索してDL")
    parser.add_argument("--limit", type=int, default=5,
                        help="検索結果の上限")
    args = parser.parse_args()

    pdf_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.dir)
    os.makedirs(pdf_dir, exist_ok=True)

    if args.query:
        # Ad-hoc search & download
        print(f"Searching: \"{args.query}\"")
        results = search_semantic_scholar(args.query, limit=args.limit)
        for i, r in enumerate(results):
            title = r.get("title", "unknown")
            safe_title = re.sub(r'[^\w\s-]', '', title)[:60].strip().replace(' ', '_')
            paper = {
                "id": f"search_{i:02d}_{safe_title}",
                "title": title,
            }
            ext_ids = r.get("externalIds", {})
            if ext_ids.get("ArXiv"):
                paper["arxiv"] = ext_ids["ArXiv"]
            if ext_ids.get("DOI"):
                paper["doi"] = ext_ids["DOI"]
            oa = r.get("openAccessPdf", {})
            if oa and oa.get("url"):
                paper["oa_url"] = oa["url"]
            download_paper(paper, pdf_dir)
            time.sleep(2)
        return

    # Download from curated list
    priorities = {"high": 1, "medium": 2, "low": 3}
    if args.priority != "all":
        target_pri = priorities[args.priority]
        papers = [p for p in PAPERS if priorities[p["priority"]] <= target_pri]
    else:
        papers = PAPERS

    print(f"=== 文献 PDF ダウンロード ===")
    print(f"  対象: {len(papers)} 論文")
    print(f"  保存先: {pdf_dir}/")
    print()

    success, failed = 0, 0
    for paper in papers:
        ok = download_paper(paper, pdf_dir)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(2)  # rate limit

    print(f"\n=== 完了 ===")
    print(f"  成功: {success}")
    print(f"  失敗: {failed}")
    print(f"  保存先: {pdf_dir}/")


if __name__ == "__main__":
    main()
