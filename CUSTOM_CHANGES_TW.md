# 相對原版 `rohitg00/agentmemory` 的客製化改動說明

這份文件整理了我們在 upstream `agentmemory` 基礎上做的客製化修改，目標是把它從通用 memory server 擴充成一個以 Markdown 研究資料為核心的工作台。

這份說明聚焦在：

- 程式碼改動
- 功能行為改動
- 新增 API / Viewer 能力

不包含使用者本地自行匯入的雜項資料檔。

## 這次客製化的目標

原版 repo 的強項是記憶擷取、檢索與圖譜基礎設施；這次改動主要是為了支援研究筆記 / paper workflow：

- 匯入大量 Markdown 文件
- 直接對 Markdown 語料問答
- 在 viewer 內正確顯示 Markdown、表格、LaTeX
- 做研究地圖、實驗追蹤、反向提問
- 針對 paper 記憶建立穩定的知識圖譜

## 主要改動總覽

### 1. 新增研究工作台 API

在 [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:1) 中新增或擴充了以下 API：

- `/agentmemory/chat`
  - 以 memory + graph context 做多輪問答
- `/agentmemory/documents`
  - 列出匯入的 Markdown 文件
- `/agentmemory/documents/read`
  - 讀取指定 Markdown 文件
- `/agentmemory/documents/summary`
  - 用 LLM 對單篇文件做摘要
- `/agentmemory/research-map`
  - 生成研究地圖
- `/agentmemory/experiments`
  - 抽取實驗追蹤表
- `/agentmemory/suggestions`
  - 生成反向問題 / 下一步提問
- `/agentmemory/index/status`
  - 回報 model、embedding、documents、memories、graph 狀態
- `/agentmemory/graph/rebuild-from-memories`
  - 從 memories 重建圖譜，現在也會一起跑 paper graph builder
- `/agentmemory/graph/build-papers`
  - 直接從 imported paper memories 建立 paper-oriented graph

### 2. 強化問答檢索，改善 paper title / 中文 query 命中率

在 [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:317) 補了 ask/chat retrieval 的 fallback 與 lexical scoring。

改善點包括：

- semantic retrieval 命中差時會啟用 lexical fallback
- 支援中文自然語句正規化
- 對以下訊號提高權重：
  - 匯入 Markdown 檔名
  - `Imported ReadPaper Markdown: ...` 這一行
  - arXiv id
- graph context 查不到時，會從命中的 memories 合成可用的 graph context

這讓像：

- `Qwen3 技術報告在講什麼？`

這類 query 可以穩定打到正確 paper memory，而不是漂到不相關文件。

### 3. 新增 deterministic paper graph builder

在 [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:372) 新增 paper-specific graph builder。

這一層不是只靠 LLM 做 graph extraction，而是直接從 imported paper memories 解析並建立穩定結構：

- `file` node
- `paper` node
- `source` node
- `documents` edge
- `references` edge
- `hosted_at` edge

這樣做的原因是：很多 imported paper memory 其實內容很短，若只依賴 LLM graph extraction，圖譜會很稀疏，甚至查不到 paper 本身。

### 4. 強化 native graph query

在 [src/functions/graph.ts](/home/lai/github/agentmemory/src/functions/graph.ts:1) 改善 graph query 行為：

- query normalization
- alias-aware matching
- 對 node name / string properties 的比對更寬容

這讓 graph query 對下列輸入更穩定：

- paper 標題
- arXiv 編號
- 中英混合 query

### 5. 將 viewer 擴充成研究工作台

在 [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1) 大幅擴充 viewer，加入以下 tab：

- `Chat`
- `Documents`
- `Research Map`
- `Experiments`
- `Questions`
- `Indexing`

viewer 現在可以：

- 直接對匯入文件問答
- 瀏覽文件與摘要
- 做研究整理與實驗抽取
- 觸發 graph rebuild / paper graph build

### 6. Markdown 顯示改成真正渲染

在 [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1391) 加了輕量 Markdown renderer。

目前支援：

- headings
- paragraphs
- lists
- blockquotes
- inline code / code blocks
- links
- GitHub-style Markdown tables

這解決了原本文件預覽滿版都是 `#`、`|` 原文的問題。

### 7. LaTeX 顯示改用本地 KaTeX

相關改動在：

- [package.json](/home/lai/github/agentmemory/package.json:1)
- [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:7)
- [src/viewer/server.ts](/home/lai/github/agentmemory/src/viewer/server.ts:74)
- [src/auth.ts](/home/lai/github/agentmemory/src/auth.ts:1)

目前支援：

- `$...$`
- `$$...$$`
- `\(...\)`
- `\[...\]`

KaTeX 資產由本地 `/vendor/katex/*` 提供，不依賴外部 CDN。

### 8. 修正 Chat 版面被輸入框擋住

在 [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:490) 調整了 chat layout。

重點：

- 新增 sticky bottom composer：`.ask-composer`

這讓回答區不會再被輸入框視覺上截斷。

### 9. 改善 Gemini embedding 設定支援

在 [src/providers/embedding/gemini.ts](/home/lai/github/agentmemory/src/providers/embedding/gemini.ts:1) 改成可讀取環境變數：

- `GEMINI_EMBEDDING_MODEL`
- `GEMINI_EMBEDDING_DIMENSIONS`

這次實際使用的是：

- `gemini-embedding-2`
- `768` dimensions

### 10. 擴充 graph type system

在 [src/types.ts](/home/lai/github/agentmemory/src/types.ts:338) 補了 paper workflow 需要的 graph 類型。

新增 node types：

- `paper`
- `source`

新增 edge types：

- `documents`
- `references`
- `hosted_at`

## 改動到的主要檔案

- [src/triggers/api.ts](/home/lai/github/agentmemory/src/triggers/api.ts:1)
- [src/viewer/index.html](/home/lai/github/agentmemory/src/viewer/index.html:1)
- [src/functions/graph.ts](/home/lai/github/agentmemory/src/functions/graph.ts:1)
- [src/providers/embedding/gemini.ts](/home/lai/github/agentmemory/src/providers/embedding/gemini.ts:1)
- [src/viewer/server.ts](/home/lai/github/agentmemory/src/viewer/server.ts:1)
- [src/auth.ts](/home/lai/github/agentmemory/src/auth.ts:1)
- [src/types.ts](/home/lai/github/agentmemory/src/types.ts:1)
- [package.json](/home/lai/github/agentmemory/package.json:1)

## 這份客製化常用的環境設定

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

## 建議使用方式

### 建立 paper graph

可以用：

- Viewer -> `Indexing` -> `Build Papers Graph`

或：

```bash
curl -X POST http://127.0.0.1:3111/agentmemory/graph/build-papers
```

### 重建完整 graph

可以用：

- Viewer -> `Indexing` -> `Rebuild KG from memories`

或：

```bash
curl -X POST http://127.0.0.1:3111/agentmemory/graph/rebuild-from-memories \
  -H 'Content-Type: application/json' \
  -d '{"limit":200}'
```

### 直接對 imported Markdown 問答

使用：

- Viewer -> `Chat`

例如：

- `Qwen3 技術報告在講什麼？`
- `哪些文件跟 ToM 或信任有關？`
- `整理目前 benchmark 的結論`

## 補充說明

- repo 裡可能有本地匯入資料與範例文件，但這份文件主要記錄的是程式碼與功能改動。
- deterministic paper graph builder 是補強 upstream graph extraction，不是完全取代原本圖譜邏輯。
- viewer 改版後如果畫面沒變，通常需要硬重新整理，因為瀏覽器可能 cache 舊版 HTML / JS。

## 這批改動的意義

原版 `agentmemory` 很適合做記憶基礎設施；這批客製化讓它更適合：

- 文獻閱讀
- Markdown 研究資料管理
- paper 問答
- paper-centric graph exploration
- 在內建 viewer 內直接做研究工作流
