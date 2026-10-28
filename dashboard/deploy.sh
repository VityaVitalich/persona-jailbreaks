#!/usr/bin/env bash
# Rebuild data.json from the results in this repo and publish the dashboard to the gh-pages branch.
# usage: bash dashboard/deploy.sh            (run from anywhere inside the repo)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
REMOTE="${PJB_REMOTE:-$(git -C "$REPO" remote get-url origin)}"
python3 "$HERE/build_data.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cp "$HERE/index.html" "$HERE/style.css" "$HERE/data.json" "$TMP/"; touch "$TMP/.nojekyll"
# refuse to publish anything that looks like a secret or a prompt/response field
if grep -E -q 'sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|gh[opsu]_[A-Za-z0-9]{30,}|olp_[A-Za-z0-9]{16,}|"(response|final_response|response_text|completion)"' "$TMP"/*; then
  echo "refusing to deploy: secret-like string or response field in the build" >&2; exit 1; fi
cd "$TMP"
git init -q -b gh-pages
git add index.html style.css data.json .nojekyll
git diff --cached --stat
git -c user.name="$(git -C "$REPO" config user.name)" -c user.email="$(git -C "$REPO" config user.email)" \
  commit -q -m "Deploy dashboard ($(date -u +%Y-%m-%dT%H:%MZ))"
[ "$(git branch --show-current)" = gh-pages ] || { echo "not on gh-pages" >&2; exit 1; }
git push -f "$REMOTE" gh-pages:gh-pages
echo "deployed; Pages: https://vityavitalich.github.io/persona-jailbreaks/"
