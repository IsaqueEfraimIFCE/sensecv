---
type: entity
tags: [git, github, repository]
code_refs: [README.md, .gitignore, .dockerignore, fly.toml]
updated: 2026-06-09
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

Current published history on `origin/master`:

```text
8d3c6ad Publish SenseCV project source
bcf1859 Load Fly example dataset
f04a646 Fix Fly no-dataset startup
8949c0c Document latest Fly deployment
d99ab4f Document GitHub state and live DroNet API
30f6bec Document SenseCV DroNet run
8d7272f Prepare SenseCV for GitHub
```

`8d7272f` added the GitHub-ready project surface: `README.md`, `.gitignore`,
`.dockerignore`, `Dockerfile`, `fly.toml`, `requirements.txt`, and the wiki
updates for SenseCV naming, zip import, and Fly deployment.

`30f6bec` documented the fixed 3 FPS DroNet run over the
`SenseCV-02-06-2026-IFCE-Gimbal` dataset.

`bcf1859` is the remote commit that loaded the Fly example dataset and was
already present on GitHub when the June 9 publish operation began.

`8d3c6ad` is the current GitHub `master` head. It publishes the local
source-only project tree on top of the existing remote history without a force
push. It moves the source layout under `src/sensecv/`, moves deploy files under
`deploy/`, moves the wiki under `docs/wiki/`, adds `.env.example`, adds data
and log `.gitkeep` placeholders, and vendors the DroNet source layer under
`third_party/dronet/`.

During the June 9 session, a first local root commit was created before the
remote was configured:

```text
99dd97a Commit SenseCV project source
```

That commit is preserved locally as branch `local-root-snapshot`, but it is not
the branch published to GitHub. The published branch is `master`, tracking
`origin/master`.

## June 9 publish notes

- The checkout initially had no `origin` remote, which is why the local commit
  did not appear on GitHub.
- `origin` was set to
  `https://github.com/IsaqueEfraimIFCE/sensecv.git`.
- A direct push of the unrelated local root commit was rejected because GitHub
  already had commits through `bcf1859`.
- The publish was redone by creating a new commit on top of `origin/master` so
  GitHub history was preserved and the push could fast-forward:
  `bcf1859..8d3c6ad`.
- Remote verification after push reported:
  `8d3c6ad4ece95653f3dbd97d78ac4d6301bc9042 refs/heads/master`.
- Local `master` now points to `8d3c6ad` and tracks `origin/master`.
- The commit author was supplied per-command as
  `Isaque Efraim <IsaqueEfraimIFCE@users.noreply.github.com>` because no Git
  user identity was configured locally.

## Source-only policy

The repository intentionally keeps large and local data out of Git:

- Raw dataset roots such as `SenseCV-*`, `2026_*`, `Supermercado dataset/`,
  `SenseCV dataset/`, and `SenseCV dataset (lateral)/`.
- Generated DroNet outputs such as `dronet_samsung_random_samples/` and
  `dronet_sensecv_02062026_3fps/`.
- Uploaded datasets, runtime exports, local history, logs, virtual
  environments, zip files, and MP4 files.
- `.env` is ignored; `.env.example` is committed as the shareable template.

Known ignored local state after the June 9 publish includes `.env`,
`data/history.json`, local datasets under `data/datasets/` and
`data/uploaded_datasets/`, runtime exports under `data/exports/`, and Python
`__pycache__/` folders. These are expected to remain outside Git.

The Fly Docker build is also source-only. `.dockerignore` starts from `*` and
then includes only the app code and deploy files required by the container.
Datasets for deployed instances should be imported through `/api/upload-zip`
or placed on the Fly volume at `/data/clips`.

## Related pages

- [[deployment-operations]] for Fly.io app, volume, and health checks.
- [[api-routes]] for the upload/import and live DroNet endpoints.
- [[dronet-live-classification]] for the current live model panel/API work.
