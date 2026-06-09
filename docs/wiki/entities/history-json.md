---
type: entity
tags: [data-model, ledger]
code_refs: [app.py]
updated: 2026-05-30
---

# history.json - Export ledger

JSON array at the project root. One record is appended per successful manual
export by `api_crop()` in [[app-backend]]. It drives the UI history list, the
exported marks in the clip dropdown, and learned walking suggestions.

It is no longer treated as an append-only source of truth for files. On
`load_history()`, records whose `folder` no longer exists under `exports/` are
pruned and the file is rewritten. This keeps history aligned when old exports
are deleted.

## Record shape
```jsonc
{
  "number":      1,
  "folder":      "01_obstaculo_centro_parada",
  "source_clip": "2026_01_16-12_44_51",
  "source_idx":  0,
  "start":       14.23,
  "end":         25.62,
  "duration":    11.39,
  "occurrence":  "obstaculo",
  "obs_pos":     "centro",
  "response":    "parada",
  "desvio_dir":  null,
  "exported_at": "2026-05-26T13:01:00"
}
```

## Derived values
- **Next number** is not `max(history.number) + 1`. It is the first missing
  positive integer from the leading numeric prefixes of actual folders in
  `exports/`. Empty/no numbered exports means `1`, displayed as `01`.
- **Collision check** compares the proposed folder against actual
  `EXPORT_FOLDERS`, with backend `name_exists()` as the final authority.
- **Learned windows** still use this ledger: exact source-clip exports are
  training/ground-truth for walking suggestions.

## Cross-refs
Field meanings: [[classification-taxonomy]]. How records are produced:
[[export-pipeline]]. Export folder schema: [[exports-output]].

