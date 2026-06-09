# DroNet LLM Wiki Agent Instructions

This workspace uses the LLM Wiki pattern. Treat raw project files as sources and `wiki/` as the maintained knowledge base.

## Read Order

1. `wiki/index.md`
2. `wiki/maintenance-schema.md`
3. Pages relevant to the current task
4. Raw source files only as needed for verification

## Rules

- Do not edit raw sources, model weights, generated media, or result CSVs unless explicitly asked.
- Keep `wiki/index.md` current when adding or renaming pages.
- Append every ingest, query-to-page, lint pass, or substantial wiki maintenance event to `wiki/log.md`.
- Use Obsidian-style links for wiki pages.
- Preserve uncertainty in `wiki/open-questions.md`.

