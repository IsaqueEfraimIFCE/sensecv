---
type: concept
tags: [labeling, taxonomy]
code_refs: [app.py, templates/index.html]
updated: 2026-05-26
---

# Classification taxonomy

Every export is labeled by a small decision tree of chips in the
[[viewer-frontend]]. The chosen values compose both the **export folder name**
and fields in [[history-json]].

## The tree
```
occurrence
├─ sem_obstaculos                         → name: <n>_sem_obstaculos
└─ obstaculo
   ├─ obs_pos ∈ {centro, direita, esquerda}
   └─ response
      ├─ parada                            → name: <n>_obstaculo_<pos>_parada
      └─ desvio
         └─ desvio_dir ∈ {direita, esquerda}
                                           → name: <n>_obstaculo_<pos>_desvio_<dir>
```

## Folder-name composition
`api_crop()` joins the active parts with `_` (after sanitizing the name with
`[^\w\-]→_`):
- `sem_obstaculos`: `[name, "sem_obstaculos"]`
- `obstaculo`: `[name, "obstaculo", obs_pos, response, (desvio_dir if desvio)]`

The frontend builds the **same** string live (`getProposedFolderName()`) to show
a preview and flag collisions before you export. Empty name → rejected (400);
existing name → rejected (409).

## Semantics
- **occurrence** — was there an obstacle in the path? (no / yes)
- **obs_pos** — where the obstacle sat relative to the walker (center/right/left).
- **response** — what the walker did: stopped (`parada`) or went around (`desvio`).
- **desvio_dir** — which side they passed on (only when `desvio`).

## `number`
The leading name is normally a zero-padded sequential integer
(`get_next_number()` → `001`, `002`, …). If the name is all digits it's stored as
`number`; otherwise `number = 0`. See [[history-json]].

This label set is the dataset's target vocabulary — keep it stable, and record
any additions here and in the [[log]].

