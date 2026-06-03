---
type: entity
tags: [git, github, repository]
code_refs: [README.md, .gitignore, .dockerignore, fly.toml]
updated: 2026-06-03
---

# GitHub repository

The local SenseCV checkout is a Git repository prepared for GitHub as a
source-only project.

## Remote

```text
GitHub account: IsaqueEfraimIFCE
Repository URL: https://github.com/IsaqueEfraimIFCE/sensecv.git
Remote name: origin
Branch: master
Tracking: origin/master
```

This is the account/repository reported by:

```powershell
git remote -v
git branch -vv
```

## Commit history

Latest known commits before the current wiki/code documentation pass:

```text
30f6bec Document SenseCV DroNet run
8d7272f Prepare SenseCV for GitHub
```

`8d7272f` added the GitHub-ready project surface: `README.md`, `.gitignore`,
`.dockerignore`, `Dockerfile`, `fly.toml`, `requirements.txt`, and the wiki
updates for SenseCV naming, zip import, and Fly deployment.

`30f6bec` documented the fixed 3 FPS DroNet run over the
`SenseCV-02-06-2026-IFCE-Gimbal` dataset.

## Source-only policy

The repository intentionally keeps large and local data out of Git:

- Raw dataset roots such as `SenseCV-*`, `2026_*`, `Supermercado dataset/`,
  `SenseCV dataset/`, and `SenseCV dataset (lateral)/`.
- Generated DroNet outputs such as `dronet_samsung_random_samples/` and
  `dronet_sensecv_02062026_3fps/`.
- Uploaded datasets, runtime exports, local history, logs, virtual
  environments, zip files, and MP4 files.

The Fly Docker build is also source-only. `.dockerignore` starts from `*` and
then includes only the app code and deploy files required by the container.
Datasets for deployed instances should be imported through `/api/upload-zip`
or placed on the Fly volume at `/data/clips`.

## Related pages

- [[deployment-operations]] for Fly.io app, volume, and health checks.
- [[api-routes]] for the upload/import and live DroNet endpoints.
- [[dronet-live-classification]] for the current live model panel/API work.
