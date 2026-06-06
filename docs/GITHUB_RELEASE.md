# GitHub release checklist

## 1. Create repository

On GitHub: **New repository** → name `fire-danger-threshold-audit` → public → no README (this folder already has one).

## 2. Push from local machine

```bash
cd large-fire-threshold-audit-review
git init
git add .
git status   # verify no .env, no large .pth shards, no E:\ paths in tracked files
git commit -m "Initial public release v1.0.0 for manuscript reproducibility"
git branch -M main
git remote add origin https://github.com/anonymous2026738/large-fire-threshold-audit-review.git
git push -u origin main
```

Repository: https://github.com/anonymous2026738/large-fire-threshold-audit-review

## 3. Pre-push privacy scan

```bash
git grep -i "E:\\\\FireEqual" || echo "OK: no Windows absolute paths"
git grep -iE "api_key|token|password|secret" -- ':!zenodo_metadata.json' || echo "OK"
```

## 4. GitHub release tag

```bash
git tag -a v1.0.0 -m "Manuscript reproducibility release"
git push origin v1.0.0
```

Then enable Zenodo GitHub integration (see `ZENODO_UPLOAD_INSTRUCTIONS.md`).
