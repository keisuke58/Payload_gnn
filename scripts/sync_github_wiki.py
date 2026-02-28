#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wiki_repo の内容を GitHub Wiki に同期する。
- wiki_repo/*.md を .wiki リポジトリにコピー
- 相対画像パスを raw.githubusercontent.com の絶対URLに変換
- .wiki に push

Usage:
  python scripts/sync_github_wiki.py [--dry-run]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_REPO = os.path.join(PROJECT_ROOT, "wiki_repo")
RAW_BASE = "https://raw.githubusercontent.com/keisuke58/Payload_gnn/main"


def convert_image_paths(content: str) -> str:
    """相対画像パスを raw.githubusercontent.com 絶対URLに変換"""
    def repl(m):
        alt, path = m.group(1), m.group(2).strip().replace("\\", "/")
        if path.startswith("http://") or path.startswith("https://"):
            return m.group(0)
        if path.startswith("../"):
            path = path[3:]
        if path.startswith("images/"):
            return f"![{alt}]({RAW_BASE}/wiki_repo/images/{path[7:]})"
        if path.startswith("figures/"):
            return f"![{alt}]({RAW_BASE}/figures/{path[9:]})"
        return f"![{alt}]({RAW_BASE}/wiki_repo/images/{path})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, content)


def main():
    parser = argparse.ArgumentParser(description="Sync wiki_repo to GitHub Wiki")
    parser.add_argument("--dry-run", action="store_true", help="Push せずに内容のみ表示")
    args = parser.parse_args()

    # CI (GITHUB_TOKEN) の場合は HTTPS、ローカルは SSH
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        wiki_url = f"https://x-access-token:{token}@github.com/keisuke58/Payload_gnn.wiki.git"
    else:
        wiki_url = "git@github.com:keisuke58/Payload_gnn.wiki.git"
    with tempfile.TemporaryDirectory() as tmp:
        wiki_dir = os.path.join(tmp, "wiki")
        print(f"Cloning {wiki_url} ...")
        subprocess.run(
            ["git", "clone", wiki_url, wiki_dir],
            check=True,
            capture_output=True,
        )

        # wiki_repo の .md をコピー＆変換
        count = 0
        for name in sorted(os.listdir(WIKI_REPO)):
            if not name.endswith(".md"):
                continue
            src = os.path.join(WIKI_REPO, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(wiki_dir, name)
            with open(src, "r", encoding="utf-8") as f:
                content = f.read()
            content = convert_image_paths(content)
            with open(dst, "w", encoding="utf-8") as f:
                f.write(content)
            count += 1
            print(f"  Synced: {name}")

        if count == 0:
            print("No .md files found in wiki_repo")
            sys.exit(1)

        # 画像は raw.githubusercontent.com の main リポジトリ参照に変換済み（コピー不要）

        # git add, commit, push
        os.chdir(wiki_dir)
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("No changes to push.")
            return
        print("\nChanges:")
        print(status.stdout)

        if args.dry_run:
            print("\n[DRY-RUN] Would commit and push. Run without --dry-run to apply.")
            return

        # メインリポジトリの git config を継承（未設定時はフォールバック）
        email = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "config", "user.email"],
            capture_output=True,
            text=True,
        ).stdout.strip() or "noreply@github.com"
        name = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "config", "user.name"],
            capture_output=True,
            text=True,
        ).stdout.strip() or "Wiki Sync"
        subprocess.run(["git", "config", "user.email", email], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", name], check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "docs(wiki): sync from wiki_repo"],
            check=True,
            capture_output=True,
        )
        print("\nPushing to GitHub Wiki ...")
        subprocess.run(["git", "push", "origin", "master"], check=True)
        print("Done. Wiki: https://github.com/keisuke58/Payload_gnn/wiki")


if __name__ == "__main__":
    main()
