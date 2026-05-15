# Custom Changes on Top of `rohitg00/agentmemory`

This document records the project-level changes we added on top of the upstream `agentmemory` repository for a Markdown-first research workflow.

It focuses on code and product behavior changes, not user-imported data files.

## Goals

The original repo is a general-purpose memory server and viewer. This customized version extends it into a research workbench that can:

- import and query many Markdown research notes
- answer questions directly over those notes
- render Markdown, tables, and LaTeX properly in the viewer
- expose higher-level research endpoints
- build a deterministic paper-oriented knowledge graph from imported paper memories

## High-Level Changes

### 1. Research workbench APIs

Added or expanded API endpoints in [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:1):

- `/agentmemory/chat`
  - multi-turn chat over retrieved memories plus graph context
- `/agentmemory/documents`
  - list imported Markdown documents
- `/agentmemory/documents/read`
  - read a specific imported document
- `/agentmemory/documents/summary`
  - summarize a specific document with the configured LLM
- `/agentmemory/research-map`
  - synthesize a research map from retrieved notes
- `/agentmemory/experiments`
  - extract an experiment tracker from notes
- `/agentmemory/suggestions`
  - generate reverse questions / next-step prompts
- `/agentmemory/index/status`
  - report model, embedding, document, memory, and graph status
- `/agentmemory/graph/rebuild-from-memories`
  - rebuild graph from latest memories, now also runs the deterministic paper graph builder
- `/agentmemory/graph/build-papers`
  - build a paper-oriented graph directly from imported paper memories

### 2. Better retrieval for paper titles and Chinese queries

Improved ask/chat retrieval in [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:317):

- added lexical fallback when semantic retrieval misses
- added normalization for Chinese natural-language questions
- boosted matches against:
  - imported Markdown filenames
  - `Imported ReadPaper Markdown: ...` lines
  - arXiv identifiers
- improved graph-context fallback from matched memories

This fixes cases like:

- `Qwen3 技術報告在講什麼？`

where the upstream retrieval path could miss the correct imported paper memory.

### 3. Deterministic paper graph builder

Added a paper-specific graph builder in [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:372).

Instead of depending only on LLM graph extraction, this builder parses imported paper memories and creates stable graph structure:

- `file` nodes
- `paper` nodes
- `source` nodes
- `documents` edges
- `references` edges
- `hosted_at` edges

This is especially useful when imported Markdown entries are short and the LLM extractor would otherwise produce sparse or inconsistent graph output.

### 4. Stronger graph query behavior

Updated [src/functions/graph.ts](/home/lai/github/agentmemory/src/functions/graph.ts:1):

- query normalization for natural-language search
- alias-aware graph lookup
- better matching against node names and string properties

This makes native graph queries work better for paper titles, arXiv ids, and mixed Chinese/English queries.

### 5. Viewer turned into a research UI

Extended [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1) from the original viewer into a research workbench with tabs for:

- `Chat`
- `Documents`
- `Research Map`
- `Experiments`
- `Questions`
- `Indexing`

The viewer now supports:

- chat over imported notes
- document preview and LLM summary
- research synthesis workflows
- graph/index control actions
- paper graph build button

### 6. Markdown rendering in the viewer

Added a lightweight client-side Markdown renderer in [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1391).

It now renders:

- headings
- paragraphs
- lists
- blockquotes
- inline code / code blocks
- links
- GitHub-style Markdown tables

This fixes the earlier issue where documents were shown as raw text full of `#` and `|`.

### 7. LaTeX rendering with local KaTeX

Added local KaTeX support:

- [package.json](/home/lai/github/agentmemory/package.json:1)
- [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:7)
- [src/viewer/server.ts](/home/lai/github/agentmemory/src/viewer/server.ts:74)
- [src/auth.ts](/home/lai/github/agentmemory/src/auth.ts:1)

Behavior:

- renders `$...$`, `$$...$$`, `\(...\)`, `\[...\]`
- serves KaTeX assets locally from `/vendor/katex/*`
- avoids CDN dependency
- updates CSP to allow same-origin script/style loading for these assets

### 8. Chat layout fix

Adjusted the chat UI in [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:490) so the answer area is not visually cut off by the input box.

Notable change:

- sticky bottom composer via `.ask-composer`

### 9. Gemini embedding configuration improvements

Updated [src/providers/embedding/gemini.ts](/home/lai/github/agentmemory/src/providers/embedding/gemini.ts:1) to honor environment-driven Gemini embedding settings, including:

- `GEMINI_EMBEDDING_MODEL`
- `GEMINI_EMBEDDING_DIMENSIONS`

This was used to support:

- `gemini-embedding-2`
- `768` dimensions

### 10. Extended graph type system

Updated [src/types.ts](/home/lai/github/agentmemory/src/types.ts:338) to add custom graph types used by the paper workflow.

Added node types:

- `paper`
- `source`

Added edge types:

- `documents`
- `references`
- `hosted_at`

## Files Changed

Core code changes were made in:

- [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:1)
- [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1)
- [src/functions/graph.ts](/home/lai/github/agentmemory/src/functions/graph.ts:1)
- [src/providers/embedding/gemini.ts](/home/lai/github/agentmemory/src/providers/embedding/gemini.ts:1)
- [src/viewer/server.ts](/home/lai/github/agentmemory/src/viewer/server.ts:1)
- [src/auth.ts](/home/lai/github/agentmemory/src/auth.ts:1)
- [src/types.ts](/home/lai/github/agentmemory/src/types.ts:1)
- [package.json](/home/lai/github/agentmemory/package.json:1)

## Environment Assumptions Used in This Customization

Typical local settings used with this customized version:

```env
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.1-flash-lite
EMBEDDING_PROVIDER=gemini
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSIONS=768
GRAPH_EXTRACTION_ENABLED=true
CONSOLIDATION_ENABLED=true
AGENTMEMORY_AUTO_COMPRESS=true
AGENTMEMORY_INJECT_CONTEXT=true
```

## Recommended Usage

### Build the paper graph

Use either:

- Viewer -> `Indexing` -> `Build Papers Graph`

or:

```bash
curl -X POST http://127.0.0.1:3111/agentmemory/graph/build-papers
```

### Rebuild the full graph from memories

Use either:

- Viewer -> `Indexing` -> `Rebuild KG from memories`

or:

```bash
curl -X POST http://127.0.0.1:3111/agentmemory/graph/rebuild-from-memories \
  -H 'Content-Type: application/json' \
  -d '{"limit":200}'
```

### Ask questions over imported Markdown

Use:

- Viewer -> `Chat`

Example queries:

- `Qwen3 技術報告在講什麼？`
- `哪些文件跟 ToM 或信任有關？`
- `整理目前 benchmark 的結論`

## Known Notes

- The repo may contain local imported data directories and exported zip files that are not part of the code customization itself.
- The deterministic paper graph builder complements the original LLM graph extractor; it does not replace all upstream graph logic.
- Viewer behavior may require a hard refresh after frontend changes because the browser can cache the old HTML/JS.

## Why This Matters

The upstream repo is strong as a memory substrate. These changes make it much better for:

- literature review
- paper note ingestion
- question answering over Markdown corpora
- paper-centric graph exploration
- research workflow orchestration from within the built-in viewer
