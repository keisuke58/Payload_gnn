#!/bin/bash
# wiki_repo をリモートへ完璧に push するスクリプト
# Usage: ./scripts/wiki_push.sh [commit_message]
#   -n, --dry-run: コミット・push せず変更内容のみ表示

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DRY_RUN=false
COMMIT_MSG=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      COMMIT_MSG="$1"
      shift
      ;;
  esac
done

# デフォルトコミットメッセージ
if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="docs(wiki): update wiki_repo $(date +%Y-%m-%d)"
fi

echo "========================================================"
echo "Wiki Push — wiki_repo → origin/main"
echo "========================================================"
echo ""

# 1. リモート確認
if ! git remote get-url origin &>/dev/null; then
  echo "Error: No remote 'origin' configured."
  exit 1
fi
echo "Remote: $(git remote get-url origin)"
echo ""

# 2. wiki_repo の変更状況
echo "[1/4] Checking wiki_repo status..."
WIKI_MODIFIED=$(git status --porcelain wiki_repo 2>/dev/null | grep -E '^[ MARC]' || true)
WIKI_UNTRACKED=$(git status --porcelain wiki_repo 2>/dev/null | grep -E '^\?\?' || true)

if [[ -z "$WIKI_MODIFIED" && -z "$WIKI_UNTRACKED" ]]; then
  echo "No changes in wiki_repo. Nothing to push."
  exit 0
fi

echo "Changes to be committed:"
git status wiki_repo
echo ""

# 3. 画像参照の簡易チェック（ローカル相対パスが壊れていないか）
echo "[2/4] Validating wiki image references..."
BROKEN=0
for f in wiki_repo/*.md; do
  [[ -f "$f" ]] || continue
  # wiki_repo/images/ への参照をチェック（存在しない画像があれば警告）
  while IFS= read -r line; do
    if [[ "$line" =~ \!\[.*\]\(([^\)]+)\) ]]; then
      path="${BASH_REMATCH[1]}"
      # raw.githubusercontent.com はリモート参照なのでスキップ
      [[ "$path" =~ ^https ]] && continue
      # 相対パス: wiki_repo/images/... または images/...
      if [[ "$path" =~ ^wiki_repo/images/ ]]; then
        if [[ ! -f "$ROOT/$path" ]]; then
          echo "  Warning: Missing image: $path (in $f)"
          BROKEN=$((BROKEN + 1))
        fi
      elif [[ "$path" =~ ^images/ ]]; then
        if [[ ! -f "$ROOT/wiki_repo/$path" ]]; then
          echo "  Warning: Missing image: wiki_repo/$path (in $f)"
          BROKEN=$((BROKEN + 1))
        fi
      fi
    fi
  done < "$f"
done
if [[ $BROKEN -gt 0 ]]; then
  if [[ "$YES" == true ]]; then
    echo "  Found $BROKEN potentially missing image(s). Continuing (--yes)."
  elif [[ -t 0 ]]; then
    echo "  Found $BROKEN potentially missing image(s). Push anyway? (Ctrl+C to abort)"
    read -r -n 1 -p "  Continue? [y/N] " ans
    echo
    [[ "${ans,,}" != "y" ]] && exit 1
  else
    echo "  Error: $BROKEN missing image(s). Use -y to force, or fix paths."
    exit 1
  fi
fi
echo "  OK"
echo ""

# 4. Stage & Commit
echo "[3/4] Staging wiki_repo..."
git add wiki_repo/
echo ""

if [[ "$DRY_RUN" == true ]]; then
  echo "[DRY-RUN] Would commit with: $COMMIT_MSG"
  echo "[DRY-RUN] Would push to origin main"
  git diff --cached --stat
  echo ""
  echo "To actually push, run without -n:"
  echo "  ./scripts/wiki_push.sh \"$COMMIT_MSG\""
  exit 0
fi

echo "[4/4] Committing..."
git commit -m "$COMMIT_MSG"
echo ""

echo "Pushing to origin main..."
git push origin main
echo ""
echo "========================================================"
echo "Done. Wiki pushed successfully."
echo "========================================================"
