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
â”œâ”€ sem_obstaculos                         â†’ name: <n>_sem_obstaculos
â””â”€ obstaculo
   â”œâ”€ obs_pos âˆˆ {centro, direita, esquerda}
   â””â”€ response
      â”œâ”€ parada                            â†’ name: <n>_obstaculo_<pos>_parada
      â””â”€ desvio
         â””â”€ desvio_dir âˆˆ {direita, esquerda}
                                           â†’ name: <n>_obstaculo_<pos>_desvio_<dir>
```

## Folder-name composition
`api_crop()` joins the active parts with `_` (after sanitizing the name with
`[^\w\-]â†’_`):
- `sem_obstaculos`: `[name, "sem_obstaculos"]`
- `obstaculo`: `[name, "obstaculo", obs_pos, response, (desvio_dir if desvio)]`

The frontend builds the **same** string live (`getProposedFolderName()`) to show
a preview and flag collisions before you export. Empty name â†’ rejected (400);
existing name â†’ rejected (409).

## Semantics
- **occurrence** â€” was there an obstacle in the path? (no / yes)
- **obs_pos** â€” where the obstacle sat relative to the walker (center/right/left).
- **response** â€” what the walker did: stopped (`parada`) or went around (`desvio`).
- **desvio_dir** â€” which side they passed on (only when `desvio`).

## `number`
The leading name is normally a zero-padded sequential integer
(`get_next_number()` â†’ `001`, `002`, â€¦). If the name is all digits it's stored as
`number`; otherwise `number = 0`. See [[history-json]].

This label set is the dataset's target vocabulary â€” keep it stable, and record
any additions here and in the [[log]].

