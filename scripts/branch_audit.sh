#!/usr/bin/env bash
# Reproduce docs/INTEGRATION.md: branch inventory, containment and merge cost.
#
#   ./scripts/branch_audit.sh                    # audit origin/*
#   BRANCHES="main friday-dev" ./scripts/branch_audit.sh
#
# Read-only. Never touches the worktree, never pushes.

set -uo pipefail

REMOTE="${REMOTE:-origin}"
BRANCHES="${BRANCHES:-main arena/01a0e981-friday friday-dev arena/aee71f50-friday arena/ad9dff8f-friday arena/95e7eb54-friday}"
NEWEST="${NEWEST:-arena/ad9dff8f-friday}"

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
row() { printf '  %-32s %6s %9s %7s %8s\n' "$@"; }
ref_of() { echo "refs/remotes/$REMOTE/$1"; }
exists() { git rev-parse --quiet --verify "$(ref_of "$1")" >/dev/null 2>&1; }

git rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repository" >&2; exit 1; }
git fetch --quiet "$REMOTE" '+refs/heads/*:refs/remotes/'"$REMOTE"'/*' 2>/dev/null || true

say "INVENTORY ($REMOTE/*)"
row "branch" "files" "py-lines" "tests" "commits"
for b in $BRANCHES; do
  exists "$b" || { echo "  (missing: $b)" >&2; continue; }
  ref="$(ref_of "$b")"
  files=$(git ls-tree -r --name-only "$ref" | wc -l)
  loc=$(
    git ls-tree -r --name-only "$ref" | grep '\.py$' | while IFS= read -r f; do
      git show "$ref:$f" 2>/dev/null | wc -l
    done | awk '{s+=$1} END {print s+0}'
  )
  tests=$(git ls-tree -r --name-only "$ref" | grep -c '^tests/.*\.py$')
  ahead=$(git rev-list --count "$REMOTE/main..$ref" 2>/dev/null)
  row "$b" "$files" "$loc" "$tests" "$ahead"
done

say "ANCESTRY"
for b in $BRANCHES; do
  exists "$b" || continue
  ref="$(ref_of "$b")"
  mb=$(git merge-base "$REMOTE/main" "$ref" 2>/dev/null | cut -c1-12)
  if exists "$NEWEST" && git merge-base --is-ancestor "$ref" "$(ref_of "$NEWEST")" 2>/dev/null; then
    rel="contained in $NEWEST"
  elif exists "$NEWEST" && git merge-base --is-ancestor "$(ref_of "$NEWEST")" "$ref" 2>/dev/null; then
    rel="contains $NEWEST"
  else
    rel="sibling"
  fi
  printf '  %-32s merge-base=%-14s %s\n' "$b" "${mb:--}" "$rel"
done

say "CONTAINMENT vs $NEWEST"
if exists "$NEWEST"; then
  newest="$(ref_of "$NEWEST")"
  for b in $BRANCHES; do
    [ "$b" = "$NEWEST" ] && continue
    exists "$b" || continue
    ref="$(ref_of "$b")"
    shared=$(comm -12 <(git ls-tree -r --name-only "$newest" | sort) <(git ls-tree -r --name-only "$ref" | sort) | wc -l)
    only_new=$(comm -23 <(git ls-tree -r --name-only "$newest" | sort) <(git ls-tree -r --name-only "$ref" | sort) | wc -l)
    only_b=$(comm -13 <(git ls-tree -r --name-only "$newest" | sort) <(git ls-tree -r --name-only "$ref" | sort) | wc -l)
    verdict="orthogonal"
    [ "$only_b" -le 2 ] && verdict="superseded by $NEWEST"
    [ "$only_b" -eq 0 ] && verdict="fully contained"
    printf '  %-32s shared=%-4s only-newest=%-4s only-branch=%-4s  %s\n' \
      "$b" "$shared" "$only_new" "$only_b" "$verdict"
  done
fi

say "MERGE COST ($NEWEST + $(git rev-parse --abbrev-ref HEAD))"
if exists "$NEWEST"; then
  tmp=$(mktemp)
  if git merge-tree --write-tree --name-only "$(ref_of "$NEWEST")" "$(git rev-parse --abbrev-ref HEAD)" >"$tmp" 2>&1; then
    echo "  clean merge — no conflicts"
  else
    n=$(grep -c '^CONFLICT' "$tmp")
    echo "  $n conflict(s):"
    grep '^CONFLICT' "$tmp" | sed 's/^/    /'
  fi
  rm -f "$tmp"
fi

say "DONE"
echo "  Interpretation: docs/INTEGRATION.md"
