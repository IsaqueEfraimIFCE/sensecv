# SenseCV Wiki â€” Schema & Maintenance Guide

This directory is an **LLM-maintained wiki** for the SenseCV project. It is a
persistent, interlinked knowledge base that sits *between* the raw codebase and
whoever is asking questions about it. A human curates and asks; the LLM writes
and maintains every page here.

> If you are an LLM agent reading this: this file is your operating manual.
> Read it fully before editing any wiki page.

---

## The three layers

1. **Raw sources** â€” the actual project, *outside* this `wiki/` directory:
   `app.py`, `templates/index.html`, `history.json`, the timestamped clip
   folders (`2026_01_16-*/`), and `exports/`. These are the source of truth.
   The wiki **describes** them; it never duplicates code verbatim except for
   short, illustrative excerpts. When code and wiki disagree, the code wins â€”
   fix the wiki.

2. **The wiki** â€” this directory. LLM-generated markdown: an overview, entity
   pages (files, data structures, routes), and concept pages (algorithms,
   workflows). The LLM owns this layer entirely.

3. **The schema** â€” this file. Conventions and workflows. Co-evolve it as the
   project and our working style change.

---

## Directory layout

```
wiki/
  CLAUDE.md            â† you are here (schema)
  index.md             â† catalog of every page (content-oriented)
  log.md               â† append-only timeline (chronological)
  overview.md          â† project synthesis / entry point
  entities/            â† concrete things: files, data models, routes
  concepts/            â† ideas, algorithms, workflows that span files
```

## Page conventions

- **Filenames**: kebab-case, no spaces (`velocity-estimation.md`).
- **Links**: Obsidian wikilinks `[[velocity-estimation]]` (filename, no path,
  no extension â€” Obsidian resolves it anywhere in the vault). Link liberally;
  a link to a page that doesn't exist yet marks a page worth writing.
- **Frontmatter**: every page starts with YAML for Dataview:
  ```yaml
  ---
  type: entity | concept | meta
  tags: [sensor, frontend, ...]
  code_refs: [app.py, templates/index.html]   # files this page documents
  updated: 2026-05-26
  ---
  ```
- **Code references**: cite real symbols and line-anchored locations as
  `` `app.py â€º suggest_crop()` `` so a reader can jump to the source. Do **not**
  paste large code blocks â€” summarize behavior and show only the essential
  lines.
- **One idea per page.** If a page sprawls, split it and cross-link.

---

## Workflows

### Ingest (a code change or new source landed)
1. Read the changed file(s) in the raw layer.
2. Discuss the takeaways with the human.
3. Update the affected entity/concept pages (a single change often touches
   several â€” e.g. a new route touches [[api-routes]] **and** the concept page
   for what it does).
4. Update [[index]] if pages were added/renamed.
5. Append a dated entry to [[log]].

### Query (a question about the project)
1. Read [[index]] to locate relevant pages, then drill in.
2. Synthesize an answer **with citations** to wiki pages and code symbols.
3. If the answer is durable (a comparison, a discovered connection, a gotcha),
   **file it back** as a new page or a section â€” don't let it vanish into chat.

### Lint (periodic health check)
Scan for: contradictions between pages, claims the code no longer supports,
orphan pages (no inbound links), concepts mentioned but lacking a page, missing
cross-references, and TODO/`[[unwritten-page]]` stubs. Propose fixes and new
questions to investigate.

---

## Project context (for fast onboarding)

SenseCV is a **local Flask tool for annotating recorded walking clips** that
pair phone video with IMU sensor streams (accelerometer + gyroscope). The
working directory name â€” *"Supermercado Telefrango (Sem GPS)"* â€” signals the
domain: building an indoor, **GPS-free** navigation/obstacle dataset (a
supermarket walk-through). The operator reviews each clip, lets the tool suggest
the usable segment (phone held vertical, person walking), crops it, classifies
the obstacle situation, and exports a trimmed video + time-rebased sensor data.

Start at [[overview]].

---

## Log entry format

Append-only, one entry per operation, consistent prefix so the log stays
greppable (`grep "^## \[" log.md | tail -5`):

```
## [2026-05-26] ingest | <what changed>
- bullet of what was updated
- pages touched: [[a]], [[b]]
```


