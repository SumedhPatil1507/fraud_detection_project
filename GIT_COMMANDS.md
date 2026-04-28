# 📦 Git Commands to Update GitHub

## Quick Push (All Changes)

```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "feat: Add enterprise features - RBAC, encryption, SAR, shadow mode, optimizer, savings tracker"

# Push to main branch
git push origin main
```

---

## Detailed Workflow

### 1. Check Status

```bash
# See what files changed
git status

# See detailed changes
git diff
```

### 2. Stage Changes Selectively

```bash
# Stage new enterprise modules
git add src/encryption.py
git add src/rbac.py
git add src/sar.py
git add src/shadow.py
git add src/retrain.py
git add src/validation.py
git add src/optimizer.py
git add src/savings_tracker.py
git add src/rate_limit.py

# Stage updated files
git add api.py
git add requirements.txt
git add README.md

# Stage new tests
git add tests/test_validation.py
git add tests/test_rbac.py
git add tests/test_optimizer.py

# Stage documentation
git add DEPLOYMENT.md
git add GIT_COMMANDS.md
```

### 3. Commit with Conventional Commits

```bash
# Feature commit
git commit -m "feat: Add enterprise security features

- Implement RBAC with admin/analyst/viewer roles
- Add Fernet encryption for PII at rest
- Add data validation layer with business rules
- Add API rate limiting per role
- Update API with RBAC middleware"

# Add more commits for different features
git commit -m "feat: Add financial intelligence features

- Implement automated SAR generation
- Add dynamic cost-benefit optimizer
- Add fraud savings tracker with ROI projections
- Update API with SAR and savings endpoints"

git commit -m "feat: Add MLOps features

- Implement shadow mode deployment
- Add automated retraining pipeline
- Add drift-triggered retraining
- Add model versioning and webhooks"

git commit -m "docs: Update README and add deployment guide

- Comprehensive README with all premium features
- Add DEPLOYMENT.md with production setup
- Add environment variable documentation
- Add security checklist"

git commit -m "test: Add comprehensive test suite

- Add validation tests
- Add RBAC permission tests
- Add optimizer tests
- Update CI/CD pipeline"
```

### 4. Push to GitHub

```bash
# Push to main branch
git push origin main

# Or push to a feature branch first
git checkout -b feature/enterprise-upgrade
git push origin feature/enterprise-upgrade
# Then create PR on GitHub
```

---

## Create a New Release

```bash
# Tag the release
git tag -a v2.0.0 -m "Release v2.0.0 - Enterprise Edition

Features:
- RBAC with role-based permissions
- Data encryption at rest
- Automated SAR generation
- Shadow mode deployment
- Dynamic cost optimizer
- Fraud savings tracker
- Automated retraining pipeline
- Comprehensive test suite
- Production deployment guide"

# Push tags
git push origin v2.0.0

# Or push all tags
git push --tags
```

---

## Branch Strategy

### Feature Branch Workflow

```bash
# Create feature branch
git checkout -b feature/rbac-implementation
git add src/rbac.py tests/test_rbac.py
git commit -m "feat: Implement RBAC system"
git push origin feature/rbac-implementation

# Create PR on GitHub, then merge

# Update main
git checkout main
git pull origin main
```

### Hotfix Workflow

```bash
# Create hotfix branch
git checkout -b hotfix/security-patch
# Make fixes
git commit -m "fix: Security vulnerability in API auth"
git push origin hotfix/security-patch
# Merge to main immediately
```

---

## Undo Changes (If Needed)

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo specific file
git checkout -- filename.py

# Revert a pushed commit
git revert <commit-hash>
git push origin main
```

---

## View History

```bash
# View commit history
git log --oneline --graph --all

# View changes in last commit
git show HEAD

# View changes in specific file
git log -p filename.py
```

---

## Sync with Remote

```bash
# Fetch latest changes
git fetch origin

# Pull and merge
git pull origin main

# Pull and rebase
git pull --rebase origin main
```

---

## Clean Up

```bash
# Remove untracked files (dry run)
git clean -n

# Remove untracked files
git clean -f

# Remove ignored files too
git clean -fx
```

---

## GitHub Actions

After pushing, GitHub Actions will automatically:
1. Run pytest tests
2. Check code quality
3. Build Docker images
4. Deploy to Streamlit Cloud (if configured)

Check status at: `https://github.com/SumedhPatil1507/fraud_detection_project/actions`

---

## Quick Reference

```bash
# One-liner to commit and push everything
git add . && git commit -m "feat: Enterprise upgrade" && git push origin main

# Check remote URL
git remote -v

# Change remote URL
git remote set-url origin https://github.com/SumedhPatil1507/fraud_detection_project.git

# View branches
git branch -a

# Delete local branch
git branch -d branch-name

# Delete remote branch
git push origin --delete branch-name
```

---

## Best Practices

1. **Commit often** — Small, focused commits are better
2. **Write clear messages** — Use conventional commits format
3. **Test before pushing** — Run `pytest` locally first
4. **Pull before push** — Avoid merge conflicts
5. **Use branches** — Don't commit directly to main for large features
6. **Tag releases** — Use semantic versioning (v2.0.0, v2.1.0, etc.)
7. **Review diffs** — Use `git diff` before committing
8. **Keep history clean** — Squash commits if needed

---

## Troubleshooting

**Merge conflict:**
```bash
git status  # See conflicted files
# Edit files to resolve conflicts
git add .
git commit -m "fix: Resolve merge conflicts"
git push origin main
```

**Accidentally committed secrets:**
```bash
# Remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (dangerous!)
git push origin --force --all
```

**Large files:**
```bash
# Use Git LFS for model files
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "chore: Add Git LFS for model files"
```
