import type { ISdk, ApiRequest } from "iii-sdk";
import { existsSync } from "node:fs";
import { readdir, readFile, stat } from "node:fs/promises";
import { basename, join, relative, resolve } from "node:path";
import type { Session, CompressedObservation, HookPayload, Memory, GraphNode, GraphEdge } from "../types.js";
import { KV, generateId } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import { getLatestHealth } from "../health/monitor.js";
import type { MetricsStore } from "../eval/metrics-store.js";
import type { ResilientProvider } from "../providers/resilient.js";
import { VERSION } from "../version.js";
import { timingSafeCompare } from "../auth.js";
import { renderViewerDocument } from "../viewer/document.js";
import { MAX_FILES_UPPER_BOUND } from "../functions/replay.js";
import {
  isGraphExtractionEnabled,
  isConsolidationEnabled,
  isAutoCompressEnabled,
  isContextInjectionEnabled,
  detectEmbeddingProvider,
  detectLlmProviderKind,
  getEnvVar,
} from "../config.js";

type Response = {
  status_code: number;
  headers?: Record<string, string>;
  body: unknown;
};

type AskContextItem = {
  id: string;
  title: string;
  content: string;
  score?: number;
  timestamp?: string;
};

type GraphContextNode = {
  id?: string;
  name?: string;
  type?: string;
  properties?: Record<string, unknown>;
};

type GraphContextEdge = {
  id?: string;
  type?: string;
  sourceNodeId?: string;
  targetNodeId?: string;
  weight?: number;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type MarkdownDoc = {
  path: string;
  title: string;
  size: number;
  updatedAt: string;
};

function parseOptionalInt(raw: unknown): number | undefined {
  if (raw === undefined || raw === null || raw === "") return undefined;
  const n = typeof raw === "number" ? raw : parseInt(String(raw), 10);
  return Number.isFinite(n) ? n : undefined;
}

function checkAuth(
  req: ApiRequest,
  secret: string | undefined,
): Response | null {
  if (!secret) return null;
  const auth = req.headers?.["authorization"] || req.headers?.["Authorization"];
  if (
    typeof auth !== "string" ||
    !timingSafeCompare(auth, `Bearer ${secret}`)
  ) {
    return { status_code: 401, body: { error: "unauthorized" } };
  }
  return null;
}

function requireConfiguredSecret(
  secret: string | undefined,
  feature: string,
): Response | null {
  if (secret) return null;
  return {
    status_code: 503,
    body: { error: `${feature} requires AGENTMEMORY_SECRET` },
  };
}

function flagDisabledResponse(opts: {
  error: string;
  flag: string;
  enableHow: string;
  docsHref: string;
}): Response {
  return {
    status_code: 503,
    body: opts,
  };
}

function graphDisabledResponse(): Response {
  return flagDisabledResponse({
    error: "Knowledge graph not enabled",
    flag: "GRAPH_EXTRACTION_ENABLED",
    enableHow: "Set GRAPH_EXTRACTION_ENABLED=true and restart. Requires an LLM provider key.",
    docsHref: "https://github.com/rohitg00/agentmemory#knowledge-graph",
  });
}

function consolidationDisabledResponse(): Response {
  return flagDisabledResponse({
    error: "Consolidation pipeline not enabled",
    flag: "CONSOLIDATION_ENABLED",
    enableHow: "Set CONSOLIDATION_ENABLED=true and restart. Requires an LLM provider key.",
    docsHref: "https://github.com/rohitg00/agentmemory#consolidation",
  });
}

function asNonEmptyString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function parseOptionalFiniteNumber(value: unknown): number | undefined | null {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return undefined;
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function parseOptionalPositiveInt(value: unknown): number | undefined | null {
  const parsed = parseOptionalFiniteNumber(value);
  if (parsed === undefined || parsed === null) return parsed;
  if (!Number.isInteger(parsed) || parsed < 1) return null;
  return parsed;
}

function clipText(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : `${text.slice(0, maxChars)}\n[truncated]`;
}

function renderAskSystemPrompt(): string {
  return [
    "You answer questions using agentmemory retrieval context.",
    "Ground your answer in the provided memories and graph context.",
    "If the context is thin, say what is missing instead of inventing details.",
    "Use the user's language unless they ask otherwise.",
    "When useful, cite sources inline as [M1], [M2], or [G1].",
  ].join("\n");
}

function renderAskUserPrompt(opts: {
  question: string;
  memories: AskContextItem[];
  graphNodes: Array<{ id?: string; name?: string; type?: string; properties?: Record<string, unknown> }>;
  graphEdges: Array<{ id?: string; type?: string; sourceNodeId?: string; targetNodeId?: string; weight?: number }>;
}): string {
  const memoryBlock =
    opts.memories.length > 0
      ? opts.memories
          .map(
            (m, i) =>
              `[M${i + 1}] ${m.title}\n` +
              `id: ${m.id}${m.score !== undefined ? ` score: ${m.score}` : ""}\n` +
              clipText(m.content, 1800),
          )
          .join("\n\n")
      : "No retrieved memories.";

  const graphBlock =
    opts.graphNodes.length > 0
      ? opts.graphNodes
          .slice(0, 30)
          .map((n, i) => {
            const props = n.properties ? JSON.stringify(n.properties).slice(0, 500) : "{}";
            return `[G${i + 1}] ${n.name ?? n.id} (${n.type ?? "node"}) ${props}`;
          })
          .join("\n")
      : "No matching graph nodes.";

  const edgeBlock =
    opts.graphEdges.length > 0
      ? opts.graphEdges
          .slice(0, 30)
          .map(
            (e) =>
              `- ${e.sourceNodeId} -[${e.type ?? "related"}${e.weight !== undefined ? `:${e.weight}` : ""}]-> ${e.targetNodeId}`,
          )
          .join("\n")
      : "No matching graph edges.";

  return [
    `Question:\n${opts.question}`,
    "",
    "Retrieved memories:",
    memoryBlock,
    "",
    "Knowledge graph nodes:",
    graphBlock,
    "",
    "Knowledge graph edges:",
    edgeBlock,
    "",
    "Answer directly. Include concise citations to memory or graph labels when they support claims.",
  ].join("\n");
}

function renderChatUserPrompt(opts: {
  messages: ChatMessage[];
  memories: AskContextItem[];
  graphNodes: Array<{ id?: string; name?: string; type?: string; properties?: Record<string, unknown> }>;
  graphEdges: Array<{ id?: string; type?: string; sourceNodeId?: string; targetNodeId?: string; weight?: number }>;
}): string {
  const transcript = opts.messages
    .slice(-12)
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n\n");
  const lastQuestion =
    [...opts.messages].reverse().find((m) => m.role === "user")?.content || "";
  return [
    "Conversation:",
    transcript,
    "",
    renderAskUserPrompt({
      question: lastQuestion,
      memories: opts.memories,
      graphNodes: opts.graphNodes,
      graphEdges: opts.graphEdges,
    }),
    "",
    "Reply as the assistant in the conversation. Keep continuity with earlier turns.",
  ].join("\n");
}

const IMPORTED_MARKDOWN_DIR = resolve(process.cwd(), "imported_markdown");

async function walkMarkdownDocs(dir: string): Promise<MarkdownDoc[]> {
  if (!existsSync(dir)) return [];
  const out: MarkdownDoc[] = [];
  async function walk(current: string): Promise<void> {
    const entries = await readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const abs = join(current, entry.name);
      if (entry.isDirectory()) {
        await walk(abs);
        continue;
      }
      if (!/\.(md|markdown)$/i.test(entry.name)) continue;
      const st = await stat(abs);
      const rel = relative(dir, abs);
      out.push({
        path: rel,
        title: basename(entry.name).replace(/\.(md|markdown)$/i, ""),
        size: st.size,
        updatedAt: st.mtime.toISOString(),
      });
    }
  }
  await walk(dir);
  return out.sort((a, b) => a.path.localeCompare(b.path));
}

function resolveImportedMarkdownPath(relPath: string): string | null {
  const abs = resolve(IMPORTED_MARKDOWN_DIR, relPath);
  if (
    abs !== IMPORTED_MARKDOWN_DIR &&
    !abs.startsWith(`${IMPORTED_MARKDOWN_DIR}/`)
  ) {
    return null;
  }
  if (!/\.(md|markdown)$/i.test(abs)) return null;
  return abs;
}

function memoryToAskContext(memory: Memory, score?: number): AskContextItem {
  return {
    id: memory.id,
    title: memory.title,
    content: memory.content,
    score,
    timestamp: memory.updatedAt || memory.createdAt,
  };
}

function normalizeSearchText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[“”‘’"'`]/g, "")
    .replace(/[？?！!。.,，:：;；()（）[\]{}<>]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildSearchNeedles(question: string): { phrases: string[]; terms: string[] } {
  const normalized = normalizeSearchText(question);
  const stripped = normalized
    .replace(/請(?:簡短|簡單|詳細)?回答/g, " ")
    .replace(/(?:是什麼|在講什麼|講什麼|內容是什麼|內容|介紹一下|摘要一下)/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const phrases = Array.from(
    new Set([normalized, stripped].filter((part) => part.length >= 2)),
  );
  const rawTerms = stripped
    .split(/[^0-9a-z\u4e00-\u9fff._-]+/i)
    .map((term) => term.trim())
    .filter((term) => term.length >= 2);
  const cjkChunks = rawTerms.flatMap((term) => {
    if (!/[\u4e00-\u9fff]/.test(term) || term.length <= 4) return [term];
    const chunks: string[] = [term];
    for (let size = 2; size <= Math.min(6, term.length); size += 1) {
      for (let i = 0; i <= term.length - size; i += 1) {
        chunks.push(term.slice(i, i + size));
      }
    }
    return chunks;
  });
  const terms = Array.from(new Set(cjkChunks)).sort((a, b) => b.length - a.length);
  return { phrases, terms };
}

function scoreMemoryLexicalMatch(
  memory: Memory | AskContextItem,
  needles: { phrases: string[]; terms: string[] },
): number {
  const title = normalizeSearchText(memory.title || "");
  const content = normalizeSearchText(memory.content || "");
  const fileText = "files" in memory
    ? normalizeSearchText((memory.files || []).join(" "))
    : "";
  const conceptText = "concepts" in memory
    ? normalizeSearchText((memory.concepts || []).join(" "))
    : "";
  const filenameMatch = content.match(/imported readpaper markdown:\s+([^\n]+)/i);
  const importedFile = normalizeSearchText(filenameMatch?.[1] || "");
  let score = 0;

  for (const phrase of needles.phrases) {
    if (title.includes(phrase)) score += 140;
    if (importedFile.includes(phrase)) score += 220;
    if (fileText.includes(phrase)) score += 180;
    if (content.includes(phrase)) score += 90;
    if (conceptText.includes(phrase)) score += 40;
  }

  for (const term of needles.terms) {
    if (title.includes(term)) score += 30 + Math.min(term.length, 12);
    if (importedFile.includes(term)) score += 60 + Math.min(term.length, 18);
    if (fileText.includes(term)) score += 40 + Math.min(term.length, 12);
    if (content.includes(term)) score += 15 + Math.min(term.length, 10);
    if (conceptText.includes(term)) score += 8;
  }

  return score;
}

type ParsedPaperMemory = {
  fileName: string;
  filePath?: string;
  paperTitle: string;
  arxivId?: string;
  sourceUrl?: string;
  sourceObservationIds: string[];
  aliases: string[];
  memoryId: string;
};

function derivePaperTitleFromFileName(fileName: string): string {
  return fileName
    .replace(/\.(md|markdown)$/i, "")
    .replace(/^\d+_/, "")
    .replace(/^\d+\s+/, "")
    .replace(/^\d{4}\.\d{4,5}v\d+_?/i, "")
    .replace(/_/g, " ")
    .trim();
}

function parseImportedPaperMemory(memory: Memory): ParsedPaperMemory | null {
  const explicitFile = memory.content.match(/Imported ReadPaper Markdown:\s+([^\n]+)/i)?.[1]?.trim();
  const filePath = (memory.files || []).find((file) =>
    /readpaper_md/i.test(file) || /\.(md|markdown)$/i.test(file),
  );
  const fileName = explicitFile || (filePath ? basename(filePath) : "");
  const headingTitle = memory.content.match(/^#\s+(.+)$/m)?.[1]?.trim();
  const arxivId = memory.content.match(/\b\d{4}\.\d{4,5}v\d+\b/i)?.[0];
  const sourceUrl = memory.content.match(/source:\s*(https?:\/\/\S+)/i)?.[1]?.trim();
  const derivedTitle = fileName ? derivePaperTitleFromFileName(fileName) : "";
  const paperTitle = headingTitle || derivedTitle || memory.title.trim();
  if (!fileName || !paperTitle) return null;

  const aliasCandidates = [
    paperTitle,
    derivedTitle,
    fileName.replace(/\.(md|markdown)$/i, ""),
    arxivId || "",
    memory.title,
  ].map((value) => value.trim()).filter(Boolean);

  return {
    fileName,
    filePath,
    paperTitle,
    arxivId: arxivId || undefined,
    sourceUrl: sourceUrl || undefined,
    sourceObservationIds: memory.sourceObservationIds?.length ? memory.sourceObservationIds : [memory.id],
    aliases: Array.from(new Set(aliasCandidates)),
    memoryId: memory.id,
  };
}

async function upsertGraphNode(
  kv: StateKV,
  draft: Pick<GraphNode, "type" | "name" | "properties" | "sourceObservationIds"> & Partial<GraphNode>,
): Promise<GraphNode> {
  const existingNodes = await kv.list<GraphNode>(KV.graphNodes);
  const existing = existingNodes.find((node) => node.type === draft.type && node.name === draft.name);
  const now = new Date().toISOString();
  if (existing) {
    const merged: GraphNode = {
      ...existing,
      properties: { ...existing.properties, ...draft.properties },
      sourceObservationIds: Array.from(new Set([
        ...(existing.sourceObservationIds || []),
        ...(draft.sourceObservationIds || []),
      ])),
      aliases: Array.from(new Set([...(existing.aliases || []), ...(draft.aliases || [])])),
      updatedAt: now,
      stale: false,
    };
    await kv.set(KV.graphNodes, existing.id, merged);
    return merged;
  }

  const created: GraphNode = {
    id: draft.id || generateId("gn"),
    type: draft.type,
    name: draft.name,
    properties: draft.properties || {},
    sourceObservationIds: draft.sourceObservationIds || [],
    aliases: draft.aliases || [],
    createdAt: draft.createdAt || now,
    updatedAt: now,
    stale: false,
  };
  await kv.set(KV.graphNodes, created.id, created);
  return created;
}

async function upsertGraphEdge(
  kv: StateKV,
  draft: Pick<GraphEdge, "type" | "sourceNodeId" | "targetNodeId" | "weight" | "sourceObservationIds">,
): Promise<GraphEdge> {
  const existingEdges = await kv.list<GraphEdge>(KV.graphEdges);
  const existing = existingEdges.find((edge) =>
    edge.type === draft.type &&
    edge.sourceNodeId === draft.sourceNodeId &&
    edge.targetNodeId === draft.targetNodeId,
  );
  const now = new Date().toISOString();
  if (existing) {
    const merged: GraphEdge = {
      ...existing,
      weight: Math.max(existing.weight || 0, draft.weight || 0),
      sourceObservationIds: Array.from(new Set([
        ...(existing.sourceObservationIds || []),
        ...(draft.sourceObservationIds || []),
      ])),
      stale: false,
    };
    await kv.set(KV.graphEdges, existing.id, merged);
    return merged;
  }

  const created: GraphEdge = {
    id: generateId("ge"),
    type: draft.type,
    sourceNodeId: draft.sourceNodeId,
    targetNodeId: draft.targetNodeId,
    weight: draft.weight,
    sourceObservationIds: draft.sourceObservationIds,
    createdAt: now,
    isLatest: true,
    stale: false,
  };
  await kv.set(KV.graphEdges, created.id, created);
  return created;
}

async function buildPaperGraphFromMemories(
  kv: StateKV,
  memories: Memory[],
): Promise<{
  papers: number;
  fileNodes: number;
  paperNodes: number;
  sourceNodes: number;
  edges: number;
}> {
  let papers = 0;
  let fileNodes = 0;
  let paperNodes = 0;
  let sourceNodes = 0;
  let edges = 0;

  for (const memory of memories) {
    const parsed = parseImportedPaperMemory(memory);
    if (!parsed) continue;
    papers += 1;

    const fileNode = await upsertGraphNode(kv, {
      type: "file",
      name: parsed.fileName,
      aliases: parsed.filePath ? [parsed.filePath] : [],
      properties: {
        path: parsed.filePath || parsed.fileName,
        memoryId: parsed.memoryId,
        imported: "readpaper",
      },
      sourceObservationIds: parsed.sourceObservationIds,
    });
    fileNodes += 1;

    const paperNode = await upsertGraphNode(kv, {
      type: "paper",
      name: parsed.paperTitle,
      aliases: parsed.aliases,
      properties: {
        arxiv_id: parsed.arxivId || null,
        source: parsed.sourceUrl || null,
        memoryId: parsed.memoryId,
        fileName: parsed.fileName,
      },
      sourceObservationIds: parsed.sourceObservationIds,
    });
    paperNodes += 1;

    await upsertGraphEdge(kv, {
      type: "documents",
      sourceNodeId: fileNode.id,
      targetNodeId: paperNode.id,
      weight: 1,
      sourceObservationIds: parsed.sourceObservationIds,
    });
    edges += 1;

    if (parsed.arxivId) {
      const arxivNode = await upsertGraphNode(kv, {
        type: "source",
        name: `arXiv:${parsed.arxivId}`,
        aliases: [parsed.arxivId, `https://arxiv.org/abs/${parsed.arxivId}`],
        properties: {
          provider: "arxiv",
          arxiv_id: parsed.arxivId,
          url: `https://arxiv.org/abs/${parsed.arxivId}`,
        },
        sourceObservationIds: parsed.sourceObservationIds,
      });
      sourceNodes += 1;
      await upsertGraphEdge(kv, {
        type: "references",
        sourceNodeId: paperNode.id,
        targetNodeId: arxivNode.id,
        weight: 0.95,
        sourceObservationIds: parsed.sourceObservationIds,
      });
      edges += 1;
    }

    if (parsed.sourceUrl) {
      const sourceNode = await upsertGraphNode(kv, {
        type: "source",
        name: parsed.sourceUrl,
        aliases: [],
        properties: {
          provider: "web",
          url: parsed.sourceUrl,
        },
        sourceObservationIds: parsed.sourceObservationIds,
      });
      sourceNodes += 1;
      await upsertGraphEdge(kv, {
        type: "hosted_at",
        sourceNodeId: paperNode.id,
        targetNodeId: sourceNode.id,
        weight: 0.85,
        sourceObservationIds: parsed.sourceObservationIds,
      });
      edges += 1;
    }
  }

  return { papers, fileNodes, paperNodes, sourceNodes, edges };
}

function memoriesToObservations(memories: Memory[]): CompressedObservation[] {
  return memories
    .filter((m) => m.isLatest !== false && m.title && m.content)
    .map((m) => ({
      id: m.id,
      sessionId: m.sessionIds[0] ?? "memory",
      timestamp: m.updatedAt || m.createdAt,
      type: "decision",
      title: m.title,
      facts: [m.content],
      narrative: m.content,
      concepts: m.concepts || [],
      files: m.files || [],
      importance: m.strength || 7,
    }));
}

async function retrieveAskContext(
  sdk: ISdk,
  kv: StateKV,
  question: string,
  limit: number,
): Promise<AskContextItem[]> {
  const compact = await sdk.trigger({
    function_id: "mem::smart-search",
    payload: { query: question, limit },
  }) as {
    results?: Array<{ obsId: string; sessionId?: string; score?: number; title?: string; timestamp?: string }>;
  };

  const expandIds = (compact.results || []).map((r) => ({
    obsId: r.obsId,
    sessionId: r.sessionId,
  }));
  const expanded = expandIds.length > 0
    ? await sdk.trigger({
        function_id: "mem::smart-search",
        payload: { expandIds },
      }) as {
        results?: Array<{
          obsId: string;
          observation?: CompressedObservation;
        }>;
      }
    : { results: [] };

  const scoreById = new Map(
    (compact.results || []).map((r) => [r.obsId, r.score] as const),
  );
  const expandedMemories: AskContextItem[] = (expanded.results || [])
    .map((item) => item.observation)
    .filter((o): o is CompressedObservation => Boolean(o))
    .map((o) => ({
      id: o.id,
      title: o.title,
      content: o.narrative || o.facts?.join("\n") || "",
      score: scoreById.get(o.id),
      timestamp: o.timestamp,
    }));
  const expandedIds = new Set(expandedMemories.map((m) => m.id));
  const remembered = await Promise.all(
    (compact.results || [])
      .filter((r) => !expandedIds.has(r.obsId))
      .map(async (r): Promise<AskContextItem | null> => {
        const memory = await kv.get<Memory>(KV.memories, r.obsId).catch(() => null);
        return memory ? memoryToAskContext(memory, r.score) : null;
      }),
  );
  const initial = [
    ...expandedMemories,
    ...remembered.filter((m): m is AskContextItem => Boolean(m)),
  ];
  const needles = buildSearchNeedles(question);
  const scoredInitial = initial
    .map((item) => ({ item, lexicalScore: scoreMemoryLexicalMatch(item, needles) }))
    .sort((a, b) => b.lexicalScore - a.lexicalScore);
  const hasStrongLexicalMatch = (scoredInitial[0]?.lexicalScore || 0) >= 120;
  const existingIds = new Set(initial.map((m) => m.id));
  const lexicalFallback = (await kv.list<Memory>(KV.memories))
    .filter((m) => m.isLatest !== false && !existingIds.has(m.id))
    .map((m) => ({ memory: m, lexicalScore: scoreMemoryLexicalMatch(m, needles) }))
    .filter((entry) => entry.lexicalScore > 0)
    .sort((a, b) => b.lexicalScore - a.lexicalScore)
    .slice(0, limit)
    .map((entry) => memoryToAskContext(entry.memory, entry.lexicalScore));
  const rankedInitial = scoredInitial.map((entry) => ({
    ...entry.item,
    score: entry.item.score ?? entry.lexicalScore,
  }));
  const merged = [
    ...(hasStrongLexicalMatch ? rankedInitial : lexicalFallback),
    ...(hasStrongLexicalMatch ? lexicalFallback : rankedInitial),
  ];
  const deduped: AskContextItem[] = [];
  const seen = new Set<string>();
  for (const item of merged) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    deduped.push(item);
    if (deduped.length >= limit) break;
  }
  return deduped;
}

async function retrieveGraphContext(
  sdk: ISdk,
  kv: StateKV,
  question: string,
  includeGraph: boolean,
  memories: AskContextItem[] = [],
): Promise<{
  nodes: GraphContextNode[];
  edges: GraphContextEdge[];
}> {
  if (!includeGraph || !isGraphExtractionEnabled()) return { nodes: [], edges: [] };
  try {
    const graph = await sdk.trigger({
      function_id: "mem::graph-query",
      payload: { query: question, maxDepth: 2 },
    }) as { nodes?: unknown[]; edges?: unknown[] };
    const direct = {
      nodes: Array.isArray(graph.nodes)
        ? graph.nodes as GraphContextNode[]
        : [],
      edges: Array.isArray(graph.edges)
        ? graph.edges as GraphContextEdge[]
        : [],
    };
    if (direct.nodes.length > 0) return direct;
  } catch {
    // Fall through to local fuzzy fallback below.
  }

  const allNodes = (await kv.list<GraphNode>(KV.graphNodes).catch(() => []))
    .filter((n) => !n.stale);
  const allEdges = (await kv.list<GraphEdge>(KV.graphEdges).catch(() => []))
    .filter((e) => !e.stale);
  if (allNodes.length === 0) return { nodes: [], edges: [] };

  const needles = buildSearchNeedles(question);
  const fileHints = Array.from(new Set(
    memories.flatMap((m) => {
      const arxivMatches = Array.from(m.content.matchAll(/\b\d{4}\.\d{4,5}v\d+\b/gi)).map((x) => x[0].toLowerCase());
      const fileMatches = Array.from(m.content.matchAll(/imported_markdown\/[^\s)\]]+/gi)).map((x) => basename(x[0]).toLowerCase());
      return [...arxivMatches, ...fileMatches];
    }),
  ));

  const matchingNodes = allNodes.filter((node) => {
    const haystacks = [
      normalizeSearchText(node.name),
      ...Object.values(node.properties || {})
        .filter((v): v is string => typeof v === "string")
        .map((v) => normalizeSearchText(v)),
    ];
    const joined = haystacks.join(" ");
    if (needles.phrases.some((phrase) => joined.includes(phrase))) return true;
    if (needles.terms.some((term) => joined.includes(term))) return true;
    if (fileHints.some((hint) => joined.includes(normalizeSearchText(hint)))) return true;
    return false;
  }).slice(0, 24);

  const nodeIds = new Set(matchingNodes.map((n) => n.id));
  const relatedEdges = allEdges.filter(
    (e) => nodeIds.has(e.sourceNodeId) || nodeIds.has(e.targetNodeId),
  ).slice(0, 48);
  if (matchingNodes.length > 0) return { nodes: matchingNodes, edges: relatedEdges };
  return synthesizeGraphContextFromMemories(memories);
}

function synthesizeGraphContextFromMemories(
  memories: AskContextItem[],
): { nodes: GraphContextNode[]; edges: GraphContextEdge[] } {
  const nodes: GraphContextNode[] = [];
  const edges: GraphContextEdge[] = [];
  const seen = new Set<string>();
  for (const memory of memories.slice(0, 8)) {
    const sourceMatch = memory.content.match(/source:\s*(https?:\/\/\S+)/i);
    const titleMatch = memory.content.match(/^#\s+(.+)$/m);
    const fileMatch = memory.content.match(/Imported ReadPaper Markdown:\s+([^\n]+)/i);
    const arxivMatch = memory.content.match(/\b\d{4}\.\d{4,5}v\d+\b/i);

    const fileName = fileMatch?.[1]?.trim();
    const paperTitle = titleMatch?.[1]?.trim();
    const paperId = arxivMatch?.[0]?.toLowerCase();

    if (fileName) {
      const fileNodeId = `mem-file:${fileName}`;
      if (!seen.has(fileNodeId)) {
        seen.add(fileNodeId);
        nodes.push({
          id: fileNodeId,
          name: fileName,
          type: "file",
          properties: { memoryId: memory.id },
        });
      }
      if (paperTitle) {
        const paperNodeId = `mem-paper:${paperId || paperTitle}`;
        if (!seen.has(paperNodeId)) {
          seen.add(paperNodeId);
          nodes.push({
            id: paperNodeId,
            name: paperTitle,
            type: "paper",
            properties: {
              arxiv_id: paperId,
              source: sourceMatch?.[1] || null,
              memoryId: memory.id,
            },
          });
        }
        edges.push({
          id: `edge:${fileNodeId}:${paperNodeId}`,
          type: "documents",
          sourceNodeId: fileNodeId,
          targetNodeId: paperNodeId,
        });
      }
    }
  }
  return { nodes, edges };
}

function requireSummarizer(
  provider: ResilientProvider | { circuitState?: unknown } | undefined,
): (ResilientProvider & { summarize: (system: string, user: string) => Promise<string> }) | Response {
  if (!provider || !("summarize" in provider)) {
    return {
      status_code: 503,
      body: {
        error: "LLM provider not available",
        enableHow: "Set an LLM provider key such as GEMINI_API_KEY and restart.",
      },
    };
  }
  return provider as ResilientProvider & { summarize: (system: string, user: string) => Promise<string> };
}

function isResponse(value: unknown): value is Response {
  return Boolean(value && typeof value === "object" && "status_code" in value);
}

async function latestMemories(kv: StateKV, limit = 40): Promise<Memory[]> {
  const memories = await kv.list<Memory>(KV.memories);
  return memories
    .filter((m) => m.isLatest !== false)
    .sort((a, b) => (b.updatedAt || b.createdAt || "").localeCompare(a.updatedAt || a.createdAt || ""))
    .slice(0, limit);
}

function renderMemoryDigest(memories: Memory[], maxChars = 1600): string {
  if (memories.length === 0) return "No memories available.";
  return memories
    .map((m, i) => [
      `[M${i + 1}] ${m.title}`,
      `id: ${m.id}`,
      `files: ${(m.files || []).slice(0, 6).join(", ") || "none"}`,
      clipText(m.content, maxChars),
    ].join("\n"))
    .join("\n\n");
}

export function registerApiTriggers(
  sdk: ISdk,
  kv: StateKV,
  secret?: string,
  metricsStore?: MetricsStore,
  provider?: ResilientProvider | { circuitState?: unknown },
): void {
  sdk.registerFunction(
    "middleware::api-auth",
    async (input: {
      request?: { headers?: Record<string, string | undefined> };
    }) => {
      if (!secret) return { action: "continue" };
      const headers = input?.request?.headers || {};
      const auth = headers["authorization"] || headers["Authorization"];
      if (
        typeof auth !== "string" ||
        !timingSafeCompare(auth, `Bearer ${secret}`)
      ) {
        return {
          action: "respond",
          response: { status_code: 401, body: { error: "unauthorized" } },
        };
      }
      return { action: "continue" };
    },
  );

  sdk.registerFunction("api::liveness",
    async (): Promise<Response> => ({
      status_code: 200,
      body: { status: "ok", service: "agentmemory" },
    }),
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::liveness",
    config: { api_path: "/agentmemory/livez", http_method: "GET" },
  });

  sdk.registerFunction("api::config-flags",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const providerKind = detectLlmProviderKind();
      const embeddingProvider = detectEmbeddingProvider() ? "embeddings" : "none";
      const flags = [
        {
          key: "GRAPH_EXTRACTION_ENABLED",
          label: "Knowledge graph extraction",
          enabled: isGraphExtractionEnabled(),
          default: false,
          affects: ["Graph", "Dashboard"],
          needsLlm: true,
          description: "Extracts entities and relations from observations into a knowledge graph.",
          enableHow: "Set GRAPH_EXTRACTION_ENABLED=true and provide an LLM key, then restart.",
          docsHref: "https://github.com/rohitg00/agentmemory#knowledge-graph",
        },
        {
          key: "CONSOLIDATION_ENABLED",
          label: "Memory consolidation",
          enabled: isConsolidationEnabled(),
          default: false,
          affects: ["Dashboard", "Memories", "Crystals"],
          needsLlm: true,
          description: "Periodically summarizes sessions into semantic facts + procedures.",
          enableHow: "Set CONSOLIDATION_ENABLED=true and provide an LLM key, then restart.",
          docsHref: "https://github.com/rohitg00/agentmemory#consolidation",
        },
        {
          key: "AGENTMEMORY_AUTO_COMPRESS",
          label: "LLM-powered observation compression",
          enabled: isAutoCompressEnabled(),
          default: false,
          affects: ["Memories", "Timeline"],
          needsLlm: true,
          description: "Every observation is compressed by the LLM for richer summaries (costs tokens). OFF uses zero-LLM synthetic compression.",
          enableHow: "Set AGENTMEMORY_AUTO_COMPRESS=true and provide an LLM key.",
          docsHref: "https://github.com/rohitg00/agentmemory/issues/138",
        },
        {
          key: "AGENTMEMORY_INJECT_CONTEXT",
          label: "In-conversation context injection",
          enabled: isContextInjectionEnabled(),
          default: false,
          affects: ["Hooks"],
          needsLlm: false,
          description: "Hooks write recalled context into Claude Code's conversation. OFF captures in the background without injecting.",
          enableHow: "Set AGENTMEMORY_INJECT_CONTEXT=true and restart.",
          docsHref: "https://github.com/rohitg00/agentmemory/issues/143",
        },
      ];
      return {
        status_code: 200,
        body: {
          version: VERSION,
          provider: providerKind,
          embeddingProvider,
          flags,
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::config-flags",
    config: {
      api_path: "/agentmemory/config/flags",
      http_method: "GET",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::health", 
    async (req: ApiRequest): Promise<Response> => {
      const health = await getLatestHealth(kv);
      const functionMetrics = metricsStore ? await metricsStore.getAll() : [];
      const circuitBreaker =
        provider && "circuitState" in provider ? provider.circuitState : null;

      const status = health?.status || "healthy";
      const statusCode = status === "critical" ? 503 : 200;

      return {
        status_code: statusCode,
        body: {
          status,
          service: "agentmemory",
          version: VERSION,
          health: health || null,
          functionMetrics,
          circuitBreaker,
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::health",
    config: {
      api_path: "/agentmemory/health",
      http_method: "GET",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::observe",
    async (req: ApiRequest<HookPayload>): Promise<Response> => {
      const body = (req.body ?? {}) as Record<string, unknown>;
      const hookType = asNonEmptyString(body.hookType);
      const sessionId = asNonEmptyString(body.sessionId);
      const project = asNonEmptyString(body.project);
      const cwd = asNonEmptyString(body.cwd);
      const timestamp = asNonEmptyString(body.timestamp);
      if (!hookType || !sessionId || !project || !cwd || !timestamp) {
        return {
          status_code: 400,
          body: {
            error:
              "hookType, sessionId, project, cwd, and timestamp are required strings",
          },
        };
      }
      const payload: HookPayload = {
        hookType: hookType as HookPayload["hookType"],
        sessionId,
        project,
        cwd,
        timestamp,
        data: body.data,
      };
      const result = await sdk.trigger({ function_id: "mem::observe", payload });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::observe",
    config: {
      api_path: "/agentmemory/observe",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::context",
    async (
      req: ApiRequest<{ sessionId: string; project: string; budget?: number }>,
    ): Promise<Response> => {
      const body = (req.body ?? {}) as Record<string, unknown>;
      const sessionId = asNonEmptyString(body.sessionId);
      const project = asNonEmptyString(body.project);
      if (!sessionId || !project) {
        return {
          status_code: 400,
          body: { error: "sessionId and project are required strings" },
        };
      }
      const budget = parseOptionalPositiveInt(body.budget);
      if (budget === null) {
        return {
          status_code: 400,
          body: { error: "budget must be a positive integer" },
        };
      }
      const payload: { sessionId: string; project: string; budget?: number } = {
        sessionId,
        project,
      };
      if (budget !== undefined) payload.budget = budget;
      const result = await sdk.trigger({ function_id: "mem::context", payload });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::context",
    config: {
      api_path: "/agentmemory/context",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::search", 
    async (
      req: ApiRequest<{
        query: string;
        limit?: number;
        project?: string;
        cwd?: string;
        format?: string;
        token_budget?: number;
      }>,
    ): Promise<Response> => {
      const body = (req.body ?? {}) as Record<string, unknown>;
      if (typeof body.query !== "string" || !body.query.trim()) {
        return { status_code: 400, body: { error: "query is required and must be a non-empty string" } };
      }
      if (
        body.limit !== undefined &&
        (!Number.isInteger(body.limit) || (body.limit as number) < 1)
      ) {
        return { status_code: 400, body: { error: "limit must be a positive integer" } };
      }
      if (body.project !== undefined && typeof body.project !== "string") {
        return { status_code: 400, body: { error: "project must be a string" } };
      }
      if (body.cwd !== undefined && typeof body.cwd !== "string") {
        return { status_code: 400, body: { error: "cwd must be a string" } };
      }
      if (
        body.format !== undefined &&
        (typeof body.format !== "string" ||
          !["full", "compact", "narrative"].includes(body.format.trim().toLowerCase()))
      ) {
        return {
          status_code: 400,
          body: { error: "format must be one of: full, compact, narrative" },
        };
      }
      if (
        body.token_budget !== undefined &&
        (!Number.isInteger(body.token_budget) || (body.token_budget as number) < 1)
      ) {
        return {
          status_code: 400,
          body: { error: "token_budget must be a positive integer" },
        };
      }
      const payload = {
        query: body.query.trim(),
        limit: body.limit as number | undefined,
        project: body.project as string | undefined,
        cwd: body.cwd as string | undefined,
        format:
          typeof body.format === "string"
            ? body.format.trim().toLowerCase()
            : undefined,
        token_budget: body.token_budget as number | undefined,
      };
      const result = await sdk.trigger({ function_id: "mem::search", payload: payload });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::search",
    config: {
      api_path: "/agentmemory/search",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::compress-file", 
    async (req: ApiRequest<{ filePath: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const body = (req.body ?? {}) as Record<string, unknown>;
      const filePath = asNonEmptyString(body.filePath);
      if (!filePath) {
        return {
          status_code: 400,
          body: { error: "filePath is required and must be a non-empty string" },
        };
      }
      const result = await sdk.trigger({
        function_id: "mem::compress-file",
        payload: { filePath },
      });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::compress-file",
    config: { api_path: "/agentmemory/compress-file", http_method: "POST" },
  });

  sdk.registerFunction("api::replay::load",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const sessionId = asNonEmptyString(req.query_params?.["sessionId"]);
      if (!sessionId) {
        return { status_code: 400, body: { error: "sessionId is required" } };
      }
      const result = await sdk.trigger({
        function_id: "mem::replay::load",
        payload: { sessionId },
      });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::replay::load",
    config: { api_path: "/agentmemory/replay/load", http_method: "GET" },
  });

  sdk.registerFunction("api::replay::sessions",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const sessions = await kv.list<Session>(KV.sessions);
      sessions.sort((a, b) =>
        (b.startedAt || "").localeCompare(a.startedAt || ""),
      );
      return { status_code: 200, body: { success: true, sessions } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::replay::sessions",
    config: { api_path: "/agentmemory/replay/sessions", http_method: "GET" },
  });

  sdk.registerFunction("api::replay::import",
    async (
      req: ApiRequest<{ path?: string; maxFiles?: number }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const body = (req.body ?? {}) as Record<string, unknown>;
      const payload: { path?: string; maxFiles?: number } = {};
      if (body.path !== undefined) {
        if (typeof body.path !== "string" || body.path.trim().length === 0) {
          return {
            status_code: 400,
            body: { error: "path must be a non-empty string" },
          };
        }
        payload.path = body.path.trim();
      }
      if (body.maxFiles !== undefined) {
        const n = body.maxFiles as number;
        if (
          !Number.isInteger(n) ||
          n < 1 ||
          n > MAX_FILES_UPPER_BOUND
        ) {
          return {
            status_code: 400,
            body: {
              error: `maxFiles must be an integer between 1 and ${MAX_FILES_UPPER_BOUND}`,
            },
          };
        }
        payload.maxFiles = n;
      }
      const result = await sdk.trigger({
        function_id: "mem::replay::import-jsonl",
        payload,
      });
      return { status_code: 202, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::replay::import",
    config: { api_path: "/agentmemory/replay/import-jsonl", http_method: "POST" },
  });

  sdk.registerFunction("api::session::start",
    async (
      req: ApiRequest<{ sessionId: string; project: string; cwd: string }>,
    ): Promise<Response> => {
      const body = (req.body ?? {}) as Record<string, unknown>;
      const sessionId = asNonEmptyString(body.sessionId);
      const project = asNonEmptyString(body.project);
      const cwd = asNonEmptyString(body.cwd);
      if (!sessionId || !project || !cwd) {
        return {
          status_code: 400,
          body: {
            error: "sessionId, project, and cwd are required non-empty strings",
          },
        };
      }
      const session: Session = {
        id: sessionId,
        project,
        cwd,
        startedAt: new Date().toISOString(),
        status: "active",
        observationCount: 0,
      };
      await kv.set(KV.sessions, sessionId, session);
      const contextResult = await sdk.trigger<
        { sessionId: string; project: string },
        { context: string }
      >({ function_id: "mem::context", payload: { sessionId, project } });
      return {
        status_code: 200,
        body: { session, context: contextResult.context },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::session::start",
    config: {
      api_path: "/agentmemory/session/start",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::session::end",
    async (req: ApiRequest<{ sessionId: string }>): Promise<Response> => {
      const sessionId = asNonEmptyString((req.body as Record<string, unknown>)?.sessionId);
      if (!sessionId) {
        return {
          status_code: 400,
          body: { error: "sessionId is required and must be a non-empty string" },
        };
      }
      await kv.update(KV.sessions, sessionId, [
        { type: "set", path: "endedAt", value: new Date().toISOString() },
        { type: "set", path: "status", value: "completed" },
      ]);
      return { status_code: 200, body: { success: true } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::session::end",
    config: {
      api_path: "/agentmemory/session/end",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::summarize", 
    async (req: ApiRequest<{ sessionId: string }>): Promise<Response> => {
      const sessionId = asNonEmptyString((req.body as Record<string, unknown>)?.sessionId);
      if (!sessionId) {
        return { status_code: 400, body: { error: "sessionId is required" } };
      }
      const result = await sdk.trigger({
        function_id: "mem::summarize",
        payload: { sessionId },
      });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::summarize",
    config: {
      api_path: "/agentmemory/summarize",
      http_method: "POST",
      middleware_function_ids: ["middleware::api-auth"],
    },
  });

  sdk.registerFunction("api::sessions", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const sessions = await kv.list<Session>(KV.sessions);
      return { status_code: 200, body: { sessions } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::sessions",
    config: { api_path: "/agentmemory/sessions", http_method: "GET" },
  });

  sdk.registerFunction("api::observations", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const sessionId = asNonEmptyString(req.query_params?.["sessionId"]);
      if (!sessionId)
        return { status_code: 400, body: { error: "sessionId required" } };
      const observations = await kv.list<CompressedObservation>(
        KV.observations(sessionId),
      );
      return { status_code: 200, body: { observations } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::observations",
    config: { api_path: "/agentmemory/observations", http_method: "GET" },
  });

  sdk.registerFunction("api::file-context", 
    async (
      req: ApiRequest<{ sessionId: string; files: string[] }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::file-context", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::file-context",
    config: { api_path: "/agentmemory/file-context", http_method: "POST" },
  });

  sdk.registerFunction("api::enrich", 
    async (
      req: ApiRequest<{
        sessionId: string;
        files: string[];
        terms?: string[];
        toolName?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (
        !req.body?.sessionId ||
        typeof req.body.sessionId !== "string" ||
        !Array.isArray(req.body?.files) ||
        req.body.files.length === 0 ||
        !req.body.files.every((f: unknown) => typeof f === "string")
      ) {
        return {
          status_code: 400,
          body: {
            error: "sessionId (string) and files (string[]) are required",
          },
        };
      }
      if (
        req.body.terms !== undefined &&
        (!Array.isArray(req.body.terms) ||
          !req.body.terms.every((t: unknown) => typeof t === "string"))
      ) {
        return {
          status_code: 400,
          body: { error: "terms must be an array of strings" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::enrich", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::enrich",
    config: { api_path: "/agentmemory/enrich", http_method: "POST" },
  });

  sdk.registerFunction("api::remember", 
    async (
      req: ApiRequest<{
        content: string;
        type?: string;
        concepts?: string[];
        files?: string[];
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (
        !req.body?.content ||
        typeof req.body.content !== "string" ||
        !req.body.content.trim()
      ) {
        return { status_code: 400, body: { error: "content is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::remember", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::remember",
    config: { api_path: "/agentmemory/remember", http_method: "POST" },
  });

  sdk.registerFunction("api::forget", 
    async (
      req: ApiRequest<{
        sessionId?: string;
        observationIds?: string[];
        memoryId?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.sessionId && !req.body?.memoryId) {
        return {
          status_code: 400,
          body: { error: "sessionId or memoryId is required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::forget", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::forget",
    config: { api_path: "/agentmemory/forget", http_method: "POST" },
  });

  sdk.registerFunction("api::consolidate", 
    async (
      req: ApiRequest<{ project?: string; minObservations?: number }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::consolidate", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::consolidate",
    config: { api_path: "/agentmemory/consolidate", http_method: "POST" },
  });

  sdk.registerFunction("api::patterns", 
    async (req: ApiRequest<{ project?: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::patterns", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::patterns",
    config: { api_path: "/agentmemory/patterns", http_method: "POST" },
  });

  sdk.registerFunction("api::generate-rules", 
    async (req: ApiRequest<{ project?: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::generate-rules", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::generate-rules",
    config: { api_path: "/agentmemory/generate-rules", http_method: "POST" },
  });

  sdk.registerFunction("api::migrate", 
    async (req: ApiRequest<{ dbPath: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.dbPath || typeof req.body.dbPath !== "string") {
        return { status_code: 400, body: { error: "dbPath is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::migrate", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::migrate",
    config: { api_path: "/agentmemory/migrate", http_method: "POST" },
  });

  sdk.registerFunction("api::evict", 
    async (req: ApiRequest<{ dryRun?: boolean }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const dryRun =
        req.query_params?.["dryRun"] === "true" || req.body?.dryRun === true;
      const result = await sdk.trigger({ function_id: "mem::evict", payload: { dryRun } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::evict",
    config: { api_path: "/agentmemory/evict", http_method: "POST" },
  });

  sdk.registerFunction("api::smart-search", 
    async (
      req: ApiRequest<{ query?: string; expandIds?: string[]; limit?: number }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (
        !req.body?.query &&
        (!req.body?.expandIds || req.body.expandIds.length === 0)
      ) {
        return {
          status_code: 400,
          body: { error: "query or expandIds is required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::smart-search", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::smart-search",
    config: { api_path: "/agentmemory/smart-search", http_method: "POST" },
  });

  sdk.registerFunction(
    "api::ask",
    async (
      req: ApiRequest<{
        question?: string;
        query?: string;
        limit?: number;
        includeGraph?: boolean;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const question = asNonEmptyString(req.body?.question ?? req.body?.query);
      if (!question) {
        return { status_code: 400, body: { error: "question is required" } };
      }
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;

      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) {
        return { status_code: 400, body: { error: "limit must be a positive integer" } };
      }
      const limit = Math.min(parsedLimit ?? 8, 20);

      const memories = await retrieveAskContext(sdk, kv, question, limit);
      const graph = await retrieveGraphContext(sdk, kv, question, req.body?.includeGraph !== false, memories);
      const answer = await summarizer.summarize(
        renderAskSystemPrompt(),
        renderAskUserPrompt({
          question,
          memories,
          graphNodes: graph.nodes,
          graphEdges: graph.edges,
        }),
      );

      return {
        status_code: 200,
        body: {
          answer,
          memories,
          graph,
          model: summarizer.name,
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::ask",
    config: { api_path: "/agentmemory/ask", http_method: "POST" },
  });

  sdk.registerFunction(
    "api::chat",
    async (
      req: ApiRequest<{
        messages?: ChatMessage[];
        question?: string;
        limit?: number;
        includeGraph?: boolean;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;
      const messages = Array.isArray(req.body?.messages)
        ? req.body.messages
            .filter((m): m is ChatMessage =>
              Boolean(m) &&
              (m.role === "user" || m.role === "assistant") &&
              typeof m.content === "string" &&
              m.content.trim().length > 0,
            )
            .map((m) => ({ role: m.role, content: m.content.trim() }))
        : [];
      const directQuestion = asNonEmptyString(req.body?.question);
      if (directQuestion) messages.push({ role: "user", content: directQuestion });
      const question = [...messages].reverse().find((m) => m.role === "user")?.content;
      if (!question) {
        return { status_code: 400, body: { error: "messages or question is required" } };
      }
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) {
        return { status_code: 400, body: { error: "limit must be a positive integer" } };
      }
      const limit = Math.min(parsedLimit ?? 8, 20);
      const memories = await retrieveAskContext(sdk, kv, question, limit);
      const graph = await retrieveGraphContext(sdk, kv, question, req.body?.includeGraph !== false, memories);
      const answer = await summarizer.summarize(
        renderAskSystemPrompt(),
        renderChatUserPrompt({
          messages,
          memories,
          graphNodes: graph.nodes,
          graphEdges: graph.edges,
        }),
      );
      return {
        status_code: 200,
        body: { answer, memories, graph, model: summarizer.name },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::chat",
    config: { api_path: "/agentmemory/chat", http_method: "POST" },
  });

  sdk.registerFunction("api::documents",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const docs = await walkMarkdownDocs(IMPORTED_MARKDOWN_DIR);
      const memories = await kv.list<Memory>(KV.memories);
      const memoryCountByPath = new Map<string, number>();
      for (const memory of memories) {
        for (const file of memory.files || []) {
          const normalized = file.replace(/^imported_markdown\//, "");
          memoryCountByPath.set(normalized, (memoryCountByPath.get(normalized) || 0) + 1);
        }
      }
      return {
        status_code: 200,
        body: {
          root: IMPORTED_MARKDOWN_DIR,
          documents: docs.map((doc) => ({
            ...doc,
            memoryCount: memoryCountByPath.get(doc.path) || 0,
          })),
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::documents",
    config: { api_path: "/agentmemory/documents", http_method: "GET" },
  });

  sdk.registerFunction("api::document-read",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const relPath = asNonEmptyString(req.query_params?.["path"]);
      if (!relPath) return { status_code: 400, body: { error: "path is required" } };
      const abs = resolveImportedMarkdownPath(relPath);
      if (!abs || !existsSync(abs)) return { status_code: 404, body: { error: "document not found" } };
      const content = await readFile(abs, "utf8");
      return { status_code: 200, body: { path: relPath, content } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::document-read",
    config: { api_path: "/agentmemory/documents/read", http_method: "GET" },
  });

  sdk.registerFunction("api::document-summary",
    async (req: ApiRequest<{ path?: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;
      const relPath = asNonEmptyString(req.body?.path);
      if (!relPath) return { status_code: 400, body: { error: "path is required" } };
      const abs = resolveImportedMarkdownPath(relPath);
      if (!abs || !existsSync(abs)) return { status_code: 404, body: { error: "document not found" } };
      const content = await readFile(abs, "utf8");
      const summary = await summarizer.summarize(
        "You summarize research markdown for a user building a memory-assisted research system. Use Traditional Chinese.",
        [
          `File: ${relPath}`,
          "",
          clipText(content, 24000),
          "",
          "Return concise markdown with: 核心主張, 方法/證據, 重要細節, 限制, 可做的下一步實驗.",
        ].join("\n"),
      );
      return { status_code: 200, body: { path: relPath, summary, model: summarizer.name } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::document-summary",
    config: { api_path: "/agentmemory/documents/summary", http_method: "POST" },
  });

  sdk.registerFunction("api::graph-rebuild-from-memories",
    async (req: ApiRequest<{ limit?: number }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!isGraphExtractionEnabled()) return graphDisabledResponse();
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) {
        return { status_code: 400, body: { error: "limit must be a positive integer" } };
      }
      const memories = await latestMemories(kv, Math.min(parsedLimit ?? 200, 1000));
      const observations = memoriesToObservations(memories);
      let totalCreated = 0;
      const batches = [];
      for (let i = 0; i < observations.length; i += 8) {
        const batch = observations.slice(i, i + 8);
        const result = await sdk.trigger({
          function_id: "mem::graph-extract",
          payload: { observations: batch },
        }) as { nodesCreated?: number; edgesCreated?: number; nodes?: unknown[]; edges?: unknown[] };
        totalCreated += (result.nodesCreated || 0) + (result.edgesCreated || 0);
        batches.push(result);
      }
      const paperGraph = await buildPaperGraphFromMemories(kv, memories);
      const stats = await sdk.trigger({ function_id: "mem::graph-stats", payload: {} }).catch(() => null);
      return {
        status_code: 200,
        body: {
          success: true,
          memories: memories.length,
          observations: observations.length,
          batches: batches.length,
          created: totalCreated,
          deterministicPaperGraph: paperGraph,
          stats,
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::graph-rebuild-from-memories",
    config: { api_path: "/agentmemory/graph/rebuild-from-memories", http_method: "POST" },
  });

  sdk.registerFunction("api::graph-build-papers",
    async (req: ApiRequest<{ limit?: number }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) {
        return { status_code: 400, body: { error: "limit must be a positive integer" } };
      }
      const memories = await latestMemories(kv, Math.min(parsedLimit ?? 1000, 2000));
      const result = await buildPaperGraphFromMemories(kv, memories);
      const stats = await sdk.trigger({ function_id: "mem::graph-stats", payload: {} }).catch(() => null);
      return {
        status_code: 200,
        body: {
          success: true,
          scannedMemories: memories.length,
          ...result,
          stats,
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::graph-build-papers",
    config: { api_path: "/agentmemory/graph/build-papers", http_method: "POST" },
  });

  sdk.registerFunction("api::index-status",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const docs = await walkMarkdownDocs(IMPORTED_MARKDOWN_DIR);
      const memories = await kv.list<Memory>(KV.memories);
      let graphStats: unknown = null;
      try {
        graphStats = await sdk.trigger({ function_id: "mem::graph-stats", payload: {} });
      } catch {
        graphStats = null;
      }
      return {
        status_code: 200,
        body: {
          version: VERSION,
          llmProvider: detectLlmProviderKind(),
          model: getEnvVar("GEMINI_MODEL") || getEnvVar("ANTHROPIC_MODEL") || getEnvVar("OPENAI_MODEL") || provider?.name || "unknown",
          embeddingProvider: detectEmbeddingProvider() ? "embeddings" : "none",
          embeddingModel: getEnvVar("GEMINI_EMBEDDING_MODEL") || getEnvVar("OPENAI_EMBEDDING_MODEL") || getEnvVar("VOYAGE_EMBEDDING_MODEL") || "default",
          embeddingDimensions: getEnvVar("GEMINI_EMBEDDING_DIMENSIONS") || null,
          importedMarkdownRoot: IMPORTED_MARKDOWN_DIR,
          documents: docs.length,
          memories: memories.length,
          latestMemories: memories.filter((m) => m.isLatest !== false).length,
          graphStats,
          flags: {
            graphExtraction: isGraphExtractionEnabled(),
            consolidation: isConsolidationEnabled(),
            autoCompress: isAutoCompressEnabled(),
            injectContext: isContextInjectionEnabled(),
          },
        },
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::index-status",
    config: { api_path: "/agentmemory/index/status", http_method: "GET" },
  });

  sdk.registerFunction("api::research-map",
    async (req: ApiRequest<{ query?: string; limit?: number }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;
      const query = asNonEmptyString(req.body?.query) || "research map";
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) return { status_code: 400, body: { error: "limit must be a positive integer" } };
      const memories = query === "research map"
        ? (await latestMemories(kv, Math.min(parsedLimit ?? 50, 120))).map((m) => memoryToAskContext(m))
        : await retrieveAskContext(sdk, kv, query, Math.min(parsedLimit ?? 20, 60));
      const map = await summarizer.summarize(
        "You build research maps from memory snippets. Use Traditional Chinese and concise markdown.",
        [
          `Focus query: ${query}`,
          "",
          "Memory snippets:",
          memories.map((m, i) => `[M${i + 1}] ${m.title}\n${clipText(m.content, 1400)}`).join("\n\n") || "No memories.",
          "",
          "Return: 主題群集, 已知結論, 缺口/矛盾, 建議閱讀順序, 下一步研究路線.",
        ].join("\n"),
      );
      return { status_code: 200, body: { map, memories, model: summarizer.name } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::research-map",
    config: { api_path: "/agentmemory/research-map", http_method: "POST" },
  });

  sdk.registerFunction("api::experiments",
    async (req: ApiRequest<{ query?: string; limit?: number }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;
      const query = asNonEmptyString(req.body?.query) || "experiment benchmark result dataset metric ablation";
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) return { status_code: 400, body: { error: "limit must be a positive integer" } };
      const memories = await retrieveAskContext(sdk, kv, query, Math.min(parsedLimit ?? 24, 60));
      const report = await summarizer.summarize(
        "You extract experiment tracking information from research notes. Use Traditional Chinese markdown.",
        [
          "Retrieved notes:",
          memories.map((m, i) => `[M${i + 1}] ${m.title}\n${clipText(m.content, 1500)}`).join("\n\n") || "No memories.",
          "",
          "Return a compact experiment tracker table with: 實驗/假設, 資料集, 方法/設定, 指標, 結果, 狀態, 下一步. Add open risks below.",
        ].join("\n"),
      );
      return { status_code: 200, body: { report, memories, model: summarizer.name } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::experiments",
    config: { api_path: "/agentmemory/experiments", http_method: "POST" },
  });

  sdk.registerFunction("api::suggestions",
    async (req: ApiRequest<{ query?: string; limit?: number }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const summarizer = requireSummarizer(provider);
      if (isResponse(summarizer)) return summarizer;
      const query = asNonEmptyString(req.body?.query) || "open questions assumptions next steps";
      const parsedLimit = parseOptionalPositiveInt(req.body?.limit);
      if (parsedLimit === null) return { status_code: 400, body: { error: "limit must be a positive integer" } };
      const memories = await retrieveAskContext(sdk, kv, query, Math.min(parsedLimit ?? 20, 60));
      const suggestions = await summarizer.summarize(
        "You ask high-leverage reverse questions based on a user's research memory. Use Traditional Chinese markdown.",
        [
          "Memory context:",
          memories.map((m, i) => `[M${i + 1}] ${m.title}\n${clipText(m.content, 1400)}`).join("\n\n") || renderMemoryDigest(await latestMemories(kv, 25), 1000),
          "",
          "Generate 10 sharp questions the user should answer next. Group by: 論文主張, 實驗設計, 資料/評估, 實作風險, 寫作/投稿.",
        ].join("\n"),
      );
      return { status_code: 200, body: { suggestions, memories, model: summarizer.name } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::suggestions",
    config: { api_path: "/agentmemory/suggestions", http_method: "POST" },
  });

  sdk.registerFunction("api::timeline", 
    async (
      req: ApiRequest<{
        anchor: string;
        project?: string;
        before?: number;
        after?: number;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.anchor) {
        return { status_code: 400, body: { error: "anchor is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::timeline", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::timeline",
    config: { api_path: "/agentmemory/timeline", http_method: "POST" },
  });

  sdk.registerFunction("api::profile", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const project = req.query_params["project"] as string;
      if (!project) {
        return {
          status_code: 400,
          body: { error: "project query param is required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::profile", payload: { project } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::profile",
    config: { api_path: "/agentmemory/profile", http_method: "GET" },
  });

  sdk.registerFunction("api::export", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::export", payload: {} });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::export",
    config: { api_path: "/agentmemory/export", http_method: "GET" },
  });

  sdk.registerFunction("api::import", 
    async (
      req: ApiRequest<{
        exportData: unknown;
        strategy?: "merge" | "replace" | "skip";
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.exportData) {
        return { status_code: 400, body: { error: "exportData is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::import", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::import",
    config: { api_path: "/agentmemory/import", http_method: "POST" },
  });

  sdk.registerFunction("api::relations", 
    async (
      req: ApiRequest<{ sourceId: string; targetId: string; type: string }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.sourceId || !req.body?.targetId || !req.body?.type) {
        return {
          status_code: 400,
          body: { error: "sourceId, targetId, and type are required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::relate", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::relations",
    config: { api_path: "/agentmemory/relations", http_method: "POST" },
  });

  sdk.registerFunction("api::evolve", 
    async (
      req: ApiRequest<{
        memoryId: string;
        newContent: string;
        newTitle?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.memoryId || !req.body?.newContent) {
        return {
          status_code: 400,
          body: { error: "memoryId and newContent are required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::evolve", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::evolve",
    config: { api_path: "/agentmemory/evolve", http_method: "POST" },
  });

  sdk.registerFunction("api::auto-forget", 
    async (req: ApiRequest<{ dryRun?: boolean }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const dryRun =
        req.query_params?.["dryRun"] === "true" || req.body?.dryRun === true;
      const result = await sdk.trigger({ function_id: "mem::auto-forget", payload: { dryRun } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::auto-forget",
    config: { api_path: "/agentmemory/auto-forget", http_method: "POST" },
  });

  sdk.registerFunction("api::claude-bridge-read", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::claude-bridge-read", payload: {} });
        return { status_code: 200, body: result };
      } catch {
        return {
          status_code: 404,
          body: { error: "Claude bridge not enabled" },
        };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::claude-bridge-read",
    config: { api_path: "/agentmemory/claude-bridge/read", http_method: "GET" },
  });

  sdk.registerFunction("api::claude-bridge-sync", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::claude-bridge-sync", payload: {} });
        return { status_code: 200, body: result };
      } catch {
        return {
          status_code: 404,
          body: { error: "Claude bridge not enabled" },
        };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::claude-bridge-sync",
    config: {
      api_path: "/agentmemory/claude-bridge/sync",
      http_method: "POST",
    },
  });

  sdk.registerFunction("api::graph-query", 
    async (
      req: ApiRequest<{
        startNodeId?: string;
        nodeType?: string;
        maxDepth?: number;
        query?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::graph-query", payload: req.body || {} });
        return { status_code: 200, body: result };
      } catch {
        return graphDisabledResponse();
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::graph-query",
    config: { api_path: "/agentmemory/graph/query", http_method: "POST" },
  });

  sdk.registerFunction("api::graph-stats", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::graph-stats", payload: {} });
        return { status_code: 200, body: result };
      } catch {
        return graphDisabledResponse();
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::graph-stats",
    config: { api_path: "/agentmemory/graph/stats", http_method: "GET" },
  });

  sdk.registerFunction("api::graph-extract", 
    async (req: ApiRequest<{ observations: unknown[] }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (
        !Array.isArray(req.body?.observations) ||
        req.body.observations.length === 0
      ) {
        return {
          status_code: 400,
          body: { error: "observations array is required" },
        };
      }
      try {
        const result = await sdk.trigger({ function_id: "mem::graph-extract", payload: req.body });
        return { status_code: 200, body: result };
      } catch {
        return graphDisabledResponse();
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::graph-extract",
    config: { api_path: "/agentmemory/graph/extract", http_method: "POST" },
  });

  sdk.registerFunction("api::consolidate-pipeline", 
    async (req: ApiRequest<{ tier?: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::consolidate-pipeline", payload: req.body || {},
         });
        return { status_code: 200, body: result };
      } catch {
        return consolidationDisabledResponse();
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::consolidate-pipeline",
    config: {
      api_path: "/agentmemory/consolidate-pipeline",
      http_method: "POST",
    },
  });

  sdk.registerFunction("api::team-share", 
    async (
      req: ApiRequest<{ itemId: string; itemType: string; project?: string }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.itemId || !req.body?.itemType) {
        return {
          status_code: 400,
          body: { error: "itemId and itemType are required" },
        };
      }
      try {
        const result = await sdk.trigger({ function_id: "mem::team-share", payload: req.body });
        return { status_code: 201, body: result };
      } catch {
        return { status_code: 404, body: { error: "Team memory not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::team-share",
    config: { api_path: "/agentmemory/team/share", http_method: "POST" },
  });

  sdk.registerFunction("api::team-feed", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const parsedLimit = parseOptionalInt(req.query_params?.["limit"]);
        const limit = parsedLimit ?? 20;
        const result = await sdk.trigger({ function_id: "mem::team-feed", payload: { limit } });
        return { status_code: 200, body: result };
      } catch {
        return { status_code: 404, body: { error: "Team memory not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::team-feed",
    config: { api_path: "/agentmemory/team/feed", http_method: "GET" },
  });

  sdk.registerFunction("api::team-profile", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::team-profile", payload: {} });
        return { status_code: 200, body: result };
      } catch {
        return { status_code: 404, body: { error: "Team memory not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::team-profile",
    config: { api_path: "/agentmemory/team/profile", http_method: "GET" },
  });

  sdk.registerFunction("api::audit",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const parsedLimit = parseOptionalInt(req.query_params?.["limit"]);
      const entries = await sdk.trigger({ function_id: "mem::audit-query", payload: {
        operation: req.query_params?.["operation"],
        limit: parsedLimit ?? 50,
      } });
      return { status_code: 200, body: { entries, success: true } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::audit",
    config: { api_path: "/agentmemory/audit", http_method: "GET" },
  });

  sdk.registerFunction("api::governance-delete", 
    async (
      req: ApiRequest<{ memoryIds: string[]; reason?: string }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.memoryIds || !Array.isArray(req.body.memoryIds)) {
        return {
          status_code: 400,
          body: { error: "memoryIds array is required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::governance-delete", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::governance-delete",
    config: {
      api_path: "/agentmemory/governance/memories",
      http_method: "DELETE",
    },
  });

  sdk.registerFunction("api::governance-bulk", 
    async (
      req: ApiRequest<{
        type?: string[];
        dateFrom?: string;
        dateTo?: string;
        qualityBelow?: number;
        dryRun?: boolean;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::governance-bulk", payload: req.body || {} });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::governance-bulk",
    config: {
      api_path: "/agentmemory/governance/bulk-delete",
      http_method: "POST",
    },
  });

  sdk.registerFunction("api::snapshots", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::snapshot-list", payload: {} });
        return { status_code: 200, body: result };
      } catch {
        return { status_code: 404, body: { error: "Snapshots not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::snapshots",
    config: { api_path: "/agentmemory/snapshots", http_method: "GET" },
  });

  sdk.registerFunction("api::snapshot-create", 
    async (req: ApiRequest<{ message?: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::snapshot-create", payload: req.body || {},
         });
        return { status_code: 201, body: result };
      } catch {
        return { status_code: 404, body: { error: "Snapshots not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::snapshot-create",
    config: { api_path: "/agentmemory/snapshot/create", http_method: "POST" },
  });

  sdk.registerFunction("api::snapshot-restore", 
    async (req: ApiRequest<{ commitHash: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.commitHash) {
        return { status_code: 400, body: { error: "commitHash is required" } };
      }
      try {
        const result = await sdk.trigger({ function_id: "mem::snapshot-restore", payload: req.body });
        return { status_code: 200, body: result };
      } catch {
        return { status_code: 404, body: { error: "Snapshots not enabled" } };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::snapshot-restore",
    config: { api_path: "/agentmemory/snapshot/restore", http_method: "POST" },
  });

  sdk.registerFunction("api::memories",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const memories = await kv.list<import("../types.js").Memory>(KV.memories);
      const latest = req.query_params?.["latest"] === "true";
      const filtered = latest ? memories.filter((m) => m.isLatest) : memories;
      return { status_code: 200, body: { memories: filtered } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::memories",
    config: { api_path: "/agentmemory/memories", http_method: "GET" },
  });

  sdk.registerFunction("api::memory-by-id",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const id = req.path_params?.["id"];
      if (!id || typeof id !== "string") {
        return { status_code: 400, body: { error: "id path parameter is required" } };
      }
      const memory = await kv.get<import("../types.js").Memory>(KV.memories, id);
      if (!memory) {
        return { status_code: 404, body: { error: `memory not found: ${id}` } };
      }
      return { status_code: 200, body: { memory } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::memory-by-id",
    config: { api_path: "/agentmemory/memories/:id", http_method: "GET" },
  });

  sdk.registerFunction("api::semantic-list",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const semantic = await kv.list<import("../types.js").SemanticMemory>(KV.semantic);
      return { status_code: 200, body: { semantic } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::semantic-list",
    config: { api_path: "/agentmemory/semantic", http_method: "GET" },
  });

  sdk.registerFunction("api::procedural-list",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const procedural = await kv.list<import("../types.js").ProceduralMemory>(KV.procedural);
      return { status_code: 200, body: { procedural } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::procedural-list",
    config: { api_path: "/agentmemory/procedural", http_method: "GET" },
  });

  sdk.registerFunction("api::relations-list",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const relations = await kv.list<import("../types.js").MemoryRelation>(KV.relations);
      return { status_code: 200, body: { relations } };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::relations-list",
    config: { api_path: "/agentmemory/relations", http_method: "GET" },
  });

  sdk.registerFunction("api::vision-search",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const body = (req.body ?? {}) as Record<string, unknown>;
      const queryText = asNonEmptyString(body["queryText"]);
      const queryImageRef = asNonEmptyString(body["queryImageRef"]);
      const queryImageBase64 = asNonEmptyString(body["queryImageBase64"]);
      const sessionId = asNonEmptyString(body["sessionId"]);
      if (!queryText && !queryImageRef && !queryImageBase64) {
        return {
          status_code: 400,
          body: { error: "queryText, queryImageRef, or queryImageBase64 required" },
        };
      }
      const topKParsed = parseOptionalPositiveInt(body["topK"]);
      if (topKParsed === null) {
        return { status_code: 400, body: { error: "topK must be a positive integer" } };
      }
      const payload: Record<string, unknown> = {};
      if (queryText) payload["queryText"] = queryText;
      if (queryImageRef) payload["queryImageRef"] = queryImageRef;
      if (queryImageBase64) payload["queryImageBase64"] = queryImageBase64;
      if (sessionId) payload["sessionId"] = sessionId;
      if (topKParsed !== undefined) payload["topK"] = Math.min(50, topKParsed);
      const result = await sdk.trigger({ function_id: "mem::vision-search", payload });
      const resp = result as { success?: boolean; error?: string };
      if (resp?.success === false) {
        return { status_code: resp.error?.includes("disabled") ? 503 : 400, body: resp };
      }
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::vision-search",
    config: { api_path: "/agentmemory/vision-search", http_method: "POST" },
  });

  sdk.registerFunction("api::vision-embed",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const body = (req.body ?? {}) as Record<string, unknown>;
      const imageRef = asNonEmptyString(body["imageRef"]);
      const sessionId = asNonEmptyString(body["sessionId"]);
      const observationId = asNonEmptyString(body["observationId"]);
      if (!imageRef) {
        return { status_code: 400, body: { error: "imageRef is required" } };
      }
      const payload: Record<string, unknown> = { imageRef };
      if (sessionId) payload["sessionId"] = sessionId;
      if (observationId) payload["observationId"] = observationId;
      const result = await sdk.trigger({ function_id: "mem::vision-embed", payload });
      const resp = result as { success?: boolean; error?: string };
      if (resp?.success === false) {
        return { status_code: resp.error?.includes("disabled") ? 503 : 400, body: resp };
      }
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::vision-embed",
    config: { api_path: "/agentmemory/vision-embed", http_method: "POST" },
  });

  sdk.registerFunction("api::slot-list", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const result = await sdk.trigger({ function_id: "mem::slot-list", payload: {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-list",
    config: { api_path: "/agentmemory/slots", http_method: "GET" },
  });

  sdk.registerFunction("api::slot-get", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const label = asNonEmptyString(req.query_params?.["label"]);
    if (!label) return { status_code: 400, body: { error: "label query param required" } };
    const result = await sdk.trigger({ function_id: "mem::slot-get", payload: { label } });
    const resp = result as { success?: boolean; error?: string };
    if (resp?.success === false) {
      return { status_code: resp.error?.includes("not found") ? 404 : 400, body: resp };
    }
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-get",
    config: { api_path: "/agentmemory/slot", http_method: "GET" },
  });

  sdk.registerFunction("api::slot-create", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const body = (req.body ?? {}) as Record<string, unknown>;
    const label = asNonEmptyString(body["label"]);
    if (!label) return { status_code: 400, body: { error: "label required" } };
    // Reject malformed inputs instead of silently dropping them.
    if (body["content"] !== undefined && typeof body["content"] !== "string") {
      return { status_code: 400, body: { error: "content must be a string" } };
    }
    if (body["description"] !== undefined && typeof body["description"] !== "string") {
      return { status_code: 400, body: { error: "description must be a string" } };
    }
    if (body["pinned"] !== undefined && typeof body["pinned"] !== "boolean") {
      return { status_code: 400, body: { error: "pinned must be a boolean" } };
    }
    if (
      body["scope"] !== undefined &&
      body["scope"] !== "project" &&
      body["scope"] !== "global"
    ) {
      return { status_code: 400, body: { error: "scope must be 'project' or 'global'" } };
    }
    const sizeLimit = parseOptionalPositiveInt(body["sizeLimit"]);
    if (sizeLimit === null) {
      return { status_code: 400, body: { error: "sizeLimit must be a positive integer" } };
    }
    if (sizeLimit !== undefined && sizeLimit > 20000) {
      return { status_code: 400, body: { error: "sizeLimit must be <= 20000" } };
    }
    const payload: Record<string, unknown> = { label };
    if (typeof body["content"] === "string") payload["content"] = body["content"];
    if (typeof body["description"] === "string") payload["description"] = body["description"];
    if (sizeLimit !== undefined) payload["sizeLimit"] = sizeLimit;
    if (typeof body["pinned"] === "boolean") payload["pinned"] = body["pinned"];
    if (body["scope"] === "project" || body["scope"] === "global") payload["scope"] = body["scope"];
    const result = await sdk.trigger({ function_id: "mem::slot-create", payload });
    const resp = result as { success?: boolean; error?: string };
    if (resp?.success === false) {
      return { status_code: resp.error?.includes("exists") ? 409 : 400, body: resp };
    }
    return { status_code: 201, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-create",
    config: { api_path: "/agentmemory/slot", http_method: "POST" },
  });

  sdk.registerFunction("api::slot-append", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const body = (req.body ?? {}) as Record<string, unknown>;
    const label = asNonEmptyString(body["label"]);
    const text = typeof body["text"] === "string" ? body["text"] : null;
    if (!label || !text) return { status_code: 400, body: { error: "label and text required" } };
    const result = await sdk.trigger({ function_id: "mem::slot-append", payload: { label, text } });
    const resp = result as { success?: boolean; error?: string };
    if (resp?.success === false) {
      const notFound = resp.error?.includes("not found");
      const overLimit = resp.error?.includes("exceed");
      return { status_code: notFound ? 404 : overLimit ? 413 : 400, body: resp };
    }
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-append",
    config: { api_path: "/agentmemory/slot/append", http_method: "POST" },
  });

  sdk.registerFunction("api::slot-replace", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const body = (req.body ?? {}) as Record<string, unknown>;
    const label = asNonEmptyString(body["label"]);
    const content = body["content"];
    if (!label || typeof content !== "string") {
      return { status_code: 400, body: { error: "label and content (string) required" } };
    }
    const result = await sdk.trigger({ function_id: "mem::slot-replace", payload: { label, content } });
    const resp = result as { success?: boolean; error?: string };
    if (resp?.success === false) {
      const notFound = resp.error?.includes("not found");
      const overLimit = resp.error?.includes("exceed");
      return { status_code: notFound ? 404 : overLimit ? 413 : 400, body: resp };
    }
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-replace",
    config: { api_path: "/agentmemory/slot/replace", http_method: "POST" },
  });

  sdk.registerFunction("api::slot-delete", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const label = asNonEmptyString(req.query_params?.["label"]);
    if (!label) return { status_code: 400, body: { error: "label query param required" } };
    const result = await sdk.trigger({ function_id: "mem::slot-delete", payload: { label } });
    const resp = result as { success?: boolean; error?: string };
    if (resp?.success === false) {
      return { status_code: resp.error?.includes("not found") ? 404 : 400, body: resp };
    }
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-delete",
    config: { api_path: "/agentmemory/slot", http_method: "DELETE" },
  });

  sdk.registerFunction("api::slot-reflect", async (req: ApiRequest): Promise<Response> => {
    const authErr = checkAuth(req, secret);
    if (authErr) return authErr;
    const body = (req.body ?? {}) as Record<string, unknown>;
    const sessionId = asNonEmptyString(body["sessionId"]);
    if (!sessionId) return { status_code: 400, body: { error: "sessionId required" } };
    const maxObservations = parseOptionalPositiveInt(body["maxObservations"]);
    if (maxObservations === null) return { status_code: 400, body: { error: "maxObservations must be a positive integer" } };
    const payload: Record<string, unknown> = { sessionId };
    if (maxObservations !== undefined) payload["maxObservations"] = maxObservations;
    const result = await sdk.trigger({ function_id: "mem::slot-reflect", payload });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({
    type: "http",
    function_id: "api::slot-reflect",
    config: { api_path: "/agentmemory/slot/reflect", http_method: "POST" },
  });

  sdk.registerFunction("api::action-create",
    async (
      req: ApiRequest<{
        title: string;
        description?: string;
        priority?: number;
        createdBy?: string;
        project?: string;
        tags?: string[];
        parentId?: string;
        edges?: Array<{ type: string; targetActionId: string }>;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.title) {
        return { status_code: 400, body: { error: "title is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::action-create", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::action-create",
    config: { api_path: "/agentmemory/actions", http_method: "POST" },
  });

  sdk.registerFunction("api::action-update", 
    async (
      req: ApiRequest<{
        actionId: string;
        status?: string;
        title?: string;
        description?: string;
        priority?: number;
        result?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.actionId) {
        return { status_code: 400, body: { error: "actionId is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::action-update", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::action-update",
    config: { api_path: "/agentmemory/actions/update", http_method: "POST" },
  });

  sdk.registerFunction("api::action-list", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::action-list", payload: {
        status: req.query_params?.["status"],
        project: req.query_params?.["project"],
        parentId: req.query_params?.["parentId"],
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::action-list",
    config: { api_path: "/agentmemory/actions", http_method: "GET" },
  });

  sdk.registerFunction("api::action-get", 
    async (req: ApiRequest<{ actionId: string }>): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const actionId = req.query_params?.["actionId"] as string;
      if (!actionId) {
        return { status_code: 400, body: { error: "actionId required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::action-get", payload: { actionId } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::action-get",
    config: { api_path: "/agentmemory/actions/get", http_method: "GET" },
  });

  sdk.registerFunction("api::action-edge", 
    async (
      req: ApiRequest<{
        sourceActionId: string;
        targetActionId: string;
        type: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.sourceActionId || !req.body?.targetActionId || !req.body?.type) {
        return { status_code: 400, body: { error: "sourceActionId, targetActionId, and type are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::action-edge-create", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::action-edge",
    config: { api_path: "/agentmemory/actions/edges", http_method: "POST" },
  });

  sdk.registerFunction("api::frontier", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const parsedLimit = parseOptionalInt(req.query_params?.["limit"]);
      const result = await sdk.trigger({ function_id: "mem::frontier", payload: {
        project: req.query_params?.["project"],
        agentId: req.query_params?.["agentId"],
        limit: parsedLimit,
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::frontier",
    config: { api_path: "/agentmemory/frontier", http_method: "GET" },
  });

  sdk.registerFunction("api::next", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::next", payload: {
        project: req.query_params?.["project"],
        agentId: req.query_params?.["agentId"],
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::next",
    config: { api_path: "/agentmemory/next", http_method: "GET" },
  });

  sdk.registerFunction("api::lease-acquire", 
    async (
      req: ApiRequest<{ actionId: string; agentId: string; ttlMs?: number }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.actionId || !req.body?.agentId) {
        return { status_code: 400, body: { error: "actionId and agentId are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::lease-acquire", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::lease-acquire",
    config: { api_path: "/agentmemory/leases/acquire", http_method: "POST" },
  });

  sdk.registerFunction("api::lease-release", 
    async (
      req: ApiRequest<{ actionId: string; agentId: string; result?: string }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.actionId || !req.body?.agentId) {
        return { status_code: 400, body: { error: "actionId and agentId are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::lease-release", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::lease-release",
    config: { api_path: "/agentmemory/leases/release", http_method: "POST" },
  });

  sdk.registerFunction("api::lease-renew", 
    async (
      req: ApiRequest<{ actionId: string; agentId: string; ttlMs?: number }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.actionId || !req.body?.agentId) {
        return { status_code: 400, body: { error: "actionId and agentId are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::lease-renew", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::lease-renew",
    config: { api_path: "/agentmemory/leases/renew", http_method: "POST" },
  });

  sdk.registerFunction("api::routine-create",
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.name || !req.body?.steps) {
        return {
          status_code: 400,
          body: { error: "name and steps are required" },
        };
      }
      const result = await sdk.trigger({ function_id: "mem::routine-create", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::routine-create",
    config: { api_path: "/agentmemory/routines", http_method: "POST" },
  });

  sdk.registerFunction("api::routine-list", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::routine-list", payload: {
        frozen: req.query_params?.["frozen"] === "true" ? true : undefined,
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::routine-list",
    config: { api_path: "/agentmemory/routines", http_method: "GET" },
  });

  sdk.registerFunction("api::routine-run", 
    async (
      req: ApiRequest<{ routineId: string; project?: string; initiatedBy?: string }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.routineId) {
        return { status_code: 400, body: { error: "routineId is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::routine-run", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::routine-run",
    config: { api_path: "/agentmemory/routines/run", http_method: "POST" },
  });

  sdk.registerFunction("api::routine-status", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const runId = req.query_params?.["runId"] as string;
      if (!runId) {
        return { status_code: 400, body: { error: "runId query param required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::routine-status", payload: { runId } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::routine-status",
    config: { api_path: "/agentmemory/routines/status", http_method: "GET" },
  });

  sdk.registerFunction("api::signal-send", 
    async (
      req: ApiRequest<{
        from: string;
        to?: string;
        content: string;
        type?: string;
        replyTo?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.from || !req.body?.content) {
        return { status_code: 400, body: { error: "from and content are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::signal-send", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::signal-send",
    config: { api_path: "/agentmemory/signals/send", http_method: "POST" },
  });

  sdk.registerFunction("api::signal-read", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const agentId = req.query_params?.["agentId"] as string;
      if (!agentId) {
        return { status_code: 400, body: { error: "agentId query param required" } };
      }
      const parsedLimit = parseOptionalInt(req.query_params?.["limit"]);
      const result = await sdk.trigger({ function_id: "mem::signal-read", payload: {
        agentId,
        unreadOnly: req.query_params?.["unreadOnly"] === "true",
        threadId: req.query_params?.["threadId"],
        limit: parsedLimit,
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::signal-read",
    config: { api_path: "/agentmemory/signals", http_method: "GET" },
  });

  sdk.registerFunction("api::checkpoint-create", 
    async (
      req: ApiRequest<{
        name: string;
        description?: string;
        type?: string;
        linkedActionIds?: string[];
        expiresInMs?: number;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.name) {
        return { status_code: 400, body: { error: "name is required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::checkpoint-create", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::checkpoint-create",
    config: { api_path: "/agentmemory/checkpoints", http_method: "POST" },
  });

  sdk.registerFunction("api::checkpoint-resolve", 
    async (
      req: ApiRequest<{
        checkpointId: string;
        status: string;
        resolvedBy?: string;
        result?: unknown;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.checkpointId || !req.body?.status) {
        return { status_code: 400, body: { error: "checkpointId and status are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::checkpoint-resolve", payload: req.body });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::checkpoint-resolve",
    config: { api_path: "/agentmemory/checkpoints/resolve", http_method: "POST" },
  });

  sdk.registerFunction("api::checkpoint-list", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::checkpoint-list", payload: {
        status: req.query_params?.["status"],
        type: req.query_params?.["type"],
      } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::checkpoint-list",
    config: { api_path: "/agentmemory/checkpoints", http_method: "GET" },
  });

  sdk.registerFunction("api::mesh-register", 
    async (
      req: ApiRequest<{ url: string; name: string; sharedScopes?: string[] }>,
    ): Promise<Response> => {
      const secretErr = requireConfiguredSecret(secret, "mesh");
      if (secretErr) return secretErr;
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      if (!req.body?.url || !req.body?.name) {
        return { status_code: 400, body: { error: "url and name are required" } };
      }
      const result = await sdk.trigger({ function_id: "mem::mesh-register", payload: req.body });
      return { status_code: 201, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::mesh-register",
    config: { api_path: "/agentmemory/mesh/peers", http_method: "POST" },
  });

  sdk.registerFunction("api::mesh-list", 
    async (req: ApiRequest): Promise<Response> => {
      const secretErr = requireConfiguredSecret(secret, "mesh");
      if (secretErr) return secretErr;
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::mesh-list", payload: {} });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::mesh-list",
    config: { api_path: "/agentmemory/mesh/peers", http_method: "GET" },
  });

  sdk.registerFunction("api::mesh-sync", 
    async (
      req: ApiRequest<{ peerId?: string; direction?: string }>,
    ): Promise<Response> => {
      const secretErr = requireConfiguredSecret(secret, "mesh");
      if (secretErr) return secretErr;
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::mesh-sync", payload: req.body || {} });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::mesh-sync",
    config: { api_path: "/agentmemory/mesh/sync", http_method: "POST" },
  });

  sdk.registerFunction("api::mesh-receive", 
    async (req: ApiRequest): Promise<Response> => {
      const secretErr = requireConfiguredSecret(secret, "mesh");
      if (secretErr) return secretErr;
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const result = await sdk.trigger({ function_id: "mem::mesh-receive", payload: req.body || {} });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::mesh-receive",
    config: { api_path: "/agentmemory/mesh/receive", http_method: "POST" },
  });

  sdk.registerFunction("api::mesh-export", 
    async (req: ApiRequest): Promise<Response> => {
      const secretErr = requireConfiguredSecret(secret, "mesh");
      if (secretErr) return secretErr;
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const since = req.query_params?.["since"] as string;
      if (since) {
        const parsed = new Date(since).getTime();
        if (Number.isNaN(parsed)) {
          return { status_code: 400, body: { error: "Invalid 'since' date format" } };
        }
      }
      const project = req.query_params?.["project"] as string | undefined;
      const sinceTime = since ? new Date(since).getTime() : 0;
      const df = <T>(items: T[], field: "updatedAt" | "createdAt") =>
        items.filter((i) => new Date((i as Record<string, unknown>)[field] as string).getTime() > sinceTime);
      const memories = await kv.list<import("../types.js").Memory>(KV.memories);
      let actions = await kv.list<import("../types.js").Action>(KV.actions);
      if (project) {
        actions = actions.filter((a) => a.project === project);
      }
      const body: Record<string, unknown> = {
        memories: df(memories, "updatedAt"),
        actions: df(actions, "updatedAt"),
      };
      if (!project) {
        const semantic = await kv.list<import("../types.js").SemanticMemory>(KV.semantic);
        const procedural = await kv.list<import("../types.js").ProceduralMemory>(KV.procedural);
        const relations = await kv.list<import("../types.js").MemoryRelation>(KV.relations);
        const graphNodes = await kv.list<import("../types.js").GraphNode>(KV.graphNodes);
        const graphEdges = await kv.list<import("../types.js").GraphEdge>(KV.graphEdges);
        body.semantic = df(semantic, "updatedAt");
        body.procedural = df(procedural, "updatedAt");
        body.relations = df(relations, "createdAt");
        body.graphNodes = graphNodes.filter(
          (n) => new Date(n.updatedAt || n.createdAt).getTime() > sinceTime,
        );
        body.graphEdges = df(graphEdges, "createdAt");
      }
      return { status_code: 200, body };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::mesh-export",
    config: { api_path: "/agentmemory/mesh/export", http_method: "GET" },
  });

  sdk.registerFunction("api::flow-compress", 
    async (
      req: ApiRequest<{
        runId?: string;
        actionIds?: string[];
        project?: string;
      }>,
    ): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      try {
        const result = await sdk.trigger({ function_id: "mem::flow-compress", payload: req.body || {} });
        return { status_code: 200, body: result };
      } catch {
        return {
          status_code: 404,
          body: { error: "Flow compression requires a provider" },
        };
      }
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::flow-compress",
    config: { api_path: "/agentmemory/flow/compress", http_method: "POST" },
  });

  sdk.registerFunction("api::branch-detect", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const cwd = (req.query_params?.["cwd"] as string) || process.cwd();
      const result = await sdk.trigger({ function_id: "mem::detect-worktree", payload: { cwd } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::branch-detect",
    config: { api_path: "/agentmemory/branch/detect", http_method: "GET" },
  });

  sdk.registerFunction("api::branch-worktrees", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const cwd = (req.query_params?.["cwd"] as string) || process.cwd();
      const result = await sdk.trigger({ function_id: "mem::list-worktrees", payload: { cwd } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::branch-worktrees",
    config: { api_path: "/agentmemory/branch/worktrees", http_method: "GET" },
  });

  sdk.registerFunction("api::branch-sessions", 
    async (req: ApiRequest): Promise<Response> => {
      const authErr = checkAuth(req, secret);
      if (authErr) return authErr;
      const cwd = (req.query_params?.["cwd"] as string) || process.cwd();
      const result = await sdk.trigger({ function_id: "mem::branch-sessions", payload: { cwd } });
      return { status_code: 200, body: result };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::branch-sessions",
    config: { api_path: "/agentmemory/branch/sessions", http_method: "GET" },
  });

  sdk.registerFunction("api::viewer", 
    async (req: ApiRequest): Promise<Response> => {
      const denied = checkAuth(req, secret);
      if (denied) return denied;
      const rendered = renderViewerDocument();
      if (rendered.found) {
        return {
          status_code: 200,
          headers: {
            "Content-Type": "text/html",
            "Content-Security-Policy": rendered.csp,
          },
          body: rendered.html,
        };
      }
      return {
        status_code: 404,
        headers: {
          "Content-Type": "text/html",
        },
        body: "<!DOCTYPE html><html><body><h1>agentmemory</h1><p>viewer not found</p></body></html>",
      };
    },
  );
  sdk.registerTrigger({
    type: "http",
    function_id: "api::viewer",
    config: { api_path: "/agentmemory/viewer", http_method: "GET" },
  });

  sdk.registerFunction("api::sentinel-create",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.name) return { status_code: 400, body: { error: "name is required" } };
    const result = await sdk.trigger({ function_id: "mem::sentinel-create", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sentinel-create", config: { api_path: "/agentmemory/sentinels", http_method: "POST" } });

  sdk.registerFunction("api::sentinel-trigger",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.sentinelId) return { status_code: 400, body: { error: "sentinelId is required" } };
    const result = await sdk.trigger({ function_id: "mem::sentinel-trigger", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sentinel-trigger", config: { api_path: "/agentmemory/sentinels/trigger", http_method: "POST" } });

  sdk.registerFunction("api::sentinel-check",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const result = await sdk.trigger({ function_id: "mem::sentinel-check", payload: {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sentinel-check", config: { api_path: "/agentmemory/sentinels/check", http_method: "POST" } });

  sdk.registerFunction("api::sentinel-cancel",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.sentinelId) return { status_code: 400, body: { error: "sentinelId is required" } };
    const result = await sdk.trigger({ function_id: "mem::sentinel-cancel", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sentinel-cancel", config: { api_path: "/agentmemory/sentinels/cancel", http_method: "POST" } });

  sdk.registerFunction("api::sentinel-list",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const result = await sdk.trigger({ function_id: "mem::sentinel-list", payload: { status: params.status, type: params.type } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sentinel-list", config: { api_path: "/agentmemory/sentinels", http_method: "GET" } });

  sdk.registerFunction("api::sketch-create",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.title) return { status_code: 400, body: { error: "title is required" } };
    const result = await sdk.trigger({ function_id: "mem::sketch-create", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-create", config: { api_path: "/agentmemory/sketches", http_method: "POST" } });

  sdk.registerFunction("api::sketch-add",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.sketchId || !body?.title) return { status_code: 400, body: { error: "sketchId and title are required" } };
    const result = await sdk.trigger({ function_id: "mem::sketch-add", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-add", config: { api_path: "/agentmemory/sketches/add", http_method: "POST" } });

  sdk.registerFunction("api::sketch-promote",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.sketchId) return { status_code: 400, body: { error: "sketchId is required" } };
    const result = await sdk.trigger({ function_id: "mem::sketch-promote", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-promote", config: { api_path: "/agentmemory/sketches/promote", http_method: "POST" } });

  sdk.registerFunction("api::sketch-discard",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.sketchId) return { status_code: 400, body: { error: "sketchId is required" } };
    const result = await sdk.trigger({ function_id: "mem::sketch-discard", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-discard", config: { api_path: "/agentmemory/sketches/discard", http_method: "POST" } });

  sdk.registerFunction("api::sketch-list",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const result = await sdk.trigger({ function_id: "mem::sketch-list", payload: { status: params.status, project: params.project } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-list", config: { api_path: "/agentmemory/sketches", http_method: "GET" } });

  sdk.registerFunction("api::sketch-gc",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const result = await sdk.trigger({ function_id: "mem::sketch-gc", payload: {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::sketch-gc", config: { api_path: "/agentmemory/sketches/gc", http_method: "POST" } });

  sdk.registerFunction("api::crystallize",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.actionIds) return { status_code: 400, body: { error: "actionIds is required" } };
    const result = await sdk.trigger({ function_id: "mem::crystallize", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::crystallize", config: { api_path: "/agentmemory/crystals/create", http_method: "POST" } });

  sdk.registerFunction("api::crystal-list",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const limit = parseOptionalPositiveInt(params.limit);
    if (limit === null) {
      return {
        status_code: 400,
        body: { error: "invalid numeric parameter: limit" },
      };
    }
    const result = await sdk.trigger({ function_id: "mem::crystal-list", payload: { project: params.project, sessionId: params.sessionId, limit } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::crystal-list", config: { api_path: "/agentmemory/crystals", http_method: "GET" } });

  sdk.registerFunction("api::auto-crystallize",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    const result = await sdk.trigger({ function_id: "mem::auto-crystallize", payload: body || {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::auto-crystallize", config: { api_path: "/agentmemory/crystals/auto", http_method: "POST" } });

  sdk.registerFunction("api::diagnose",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    const result = await sdk.trigger({ function_id: "mem::diagnose", payload: body || {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::diagnose", config: { api_path: "/agentmemory/diagnostics", http_method: "POST" } });

  sdk.registerFunction("api::heal",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    const result = await sdk.trigger({ function_id: "mem::heal", payload: body || {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::heal", config: { api_path: "/agentmemory/diagnostics/heal", http_method: "POST" } });

  sdk.registerFunction("api::facet-tag",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.targetId || !body?.dimension || !body?.value) return { status_code: 400, body: { error: "targetId, dimension, and value are required" } };
    const result = await sdk.trigger({ function_id: "mem::facet-tag", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::facet-tag", config: { api_path: "/agentmemory/facets", http_method: "POST" } });

  sdk.registerFunction("api::facet-untag",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.targetId || !body?.dimension) return { status_code: 400, body: { error: "targetId and dimension are required" } };
    const result = await sdk.trigger({ function_id: "mem::facet-untag", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::facet-untag", config: { api_path: "/agentmemory/facets/remove", http_method: "POST" } });

  sdk.registerFunction("api::facet-query",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    const result = await sdk.trigger({ function_id: "mem::facet-query", payload: body || {} });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::facet-query", config: { api_path: "/agentmemory/facets/query", http_method: "POST" } });

  sdk.registerFunction("api::facet-get",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    if (!params.targetId) return { status_code: 400, body: { error: "targetId query param is required" } };
    const result = await sdk.trigger({ function_id: "mem::facet-get", payload: { targetId: params.targetId } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::facet-get", config: { api_path: "/agentmemory/facets", http_method: "GET" } });

  sdk.registerFunction("api::facet-stats",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const result = await sdk.trigger({ function_id: "mem::facet-stats", payload: { targetType: params.targetType } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::facet-stats", config: { api_path: "/agentmemory/facets/stats", http_method: "GET" } });

  sdk.registerFunction("api::verify",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.id || typeof body.id !== "string") return { status_code: 400, body: { error: "id is required" } };
    const result = await sdk.trigger({ function_id: "mem::verify", payload: { id: body.id } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::verify", config: { api_path: "/agentmemory/verify", http_method: "POST" } });

  sdk.registerFunction("api::cascade-update",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.supersededMemoryId || typeof body.supersededMemoryId !== "string") {
      return { status_code: 400, body: { error: "supersededMemoryId is required" } };
    }
    const result = await sdk.trigger({ function_id: "mem::cascade-update", payload: { supersededMemoryId: body.supersededMemoryId } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::cascade-update", config: { api_path: "/agentmemory/cascade-update", http_method: "POST" } });

  sdk.registerFunction("api::lesson-save",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.content || typeof body.content !== "string") return { status_code: 400, body: { error: "content is required" } };
    const tags = typeof body.tags === "string" ? (body.tags as string).split(",").map((t: string) => t.trim()).filter(Boolean) : Array.isArray(body.tags) ? body.tags : [];
    const result = (await sdk.trigger({
      function_id: "mem::lesson-save",
      payload: {
        content: body.content,
        context: body.context || "",
        confidence: typeof body.confidence === "number" ? body.confidence : undefined,
        project: typeof body.project === "string" ? body.project : undefined,
        tags,
        source: "manual",
      },
    })) as { action?: string };
    const statusCode = result?.action === "created" ? 201 : 200;
    return { status_code: statusCode, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::lesson-save", config: { api_path: "/agentmemory/lessons", http_method: "POST" } });

  sdk.registerFunction("api::lesson-list",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const minConfidence = parseOptionalFiniteNumber(params.minConfidence);
    if (minConfidence === null) {
      return {
        status_code: 400,
        body: { error: "invalid numeric parameter: minConfidence" },
      };
    }
    const limit = parseOptionalPositiveInt(params.limit);
    if (limit === null) {
      return {
        status_code: 400,
        body: { error: "invalid numeric parameter: limit" },
      };
    }
    const result = await sdk.trigger({ function_id: "mem::lesson-list", payload: {
      project: params.project,
      source: params.source,
      minConfidence,
      limit,
    } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::lesson-list", config: { api_path: "/agentmemory/lessons", http_method: "GET" } });

  sdk.registerFunction("api::lesson-search",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.query || typeof body.query !== "string") return { status_code: 400, body: { error: "query is required" } };
    const result = await sdk.trigger({ function_id: "mem::lesson-recall", payload: body });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::lesson-search", config: { api_path: "/agentmemory/lessons/search", http_method: "POST" } });

  sdk.registerFunction("api::lesson-strengthen",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.lessonId || typeof body.lessonId !== "string") return { status_code: 400, body: { error: "lessonId is required" } };
    const result = await sdk.trigger({ function_id: "mem::lesson-strengthen", payload: { lessonId: body.lessonId } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::lesson-strengthen", config: { api_path: "/agentmemory/lessons/strengthen", http_method: "POST" } });

  sdk.registerFunction("api::obsidian-export", async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = (req.body as Record<string, unknown>) || {};
    const vaultDir = asNonEmptyString(body.vaultDir);
    if (!vaultDir) {
      return {
        status_code: 400,
        body: { error: "vaultDir must be a non-empty string" },
      };
    }
    const types = typeof body.types === "string" ? body.types.split(",").map((t: string) => t.trim()).filter(Boolean) : undefined;
    const result = await sdk.trigger({ function_id: "mem::obsidian-export", payload: { vaultDir, types } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::obsidian-export", config: { api_path: "/agentmemory/obsidian/export", http_method: "POST" } });

  sdk.registerFunction("api::reflect",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = (req.body as Record<string, unknown>) || {};
    const result = await sdk.trigger({ function_id: "mem::reflect", payload: {
      project: typeof body.project === "string" ? body.project : undefined,
      maxClusters: typeof body.maxClusters === "number" ? body.maxClusters : undefined,
    } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::reflect", config: { api_path: "/agentmemory/reflect", http_method: "POST" } });

  sdk.registerFunction("api::insight-list",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const params = req.query_params || {};
    const minConfidence = parseOptionalFiniteNumber(params.minConfidence);
    if (minConfidence === null) {
      return {
        status_code: 400,
        body: { error: "invalid numeric parameter: minConfidence" },
      };
    }
    const limit = parseOptionalPositiveInt(params.limit);
    if (limit === null) {
      return {
        status_code: 400,
        body: { error: "invalid numeric parameter: limit" },
      };
    }
    const result = await sdk.trigger({ function_id: "mem::insight-list", payload: {
      project: params.project,
      minConfidence,
      limit,
    } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::insight-list", config: { api_path: "/agentmemory/insights", http_method: "GET" } });

  sdk.registerFunction("api::insight-search",  async (req: ApiRequest) => {
    const denied = checkAuth(req, secret);
    if (denied) return denied;
    const body = req.body as Record<string, unknown>;
    if (!body?.query || typeof body.query !== "string") return { status_code: 400, body: { error: "query is required" } };
    const result = await sdk.trigger({ function_id: "mem::insight-search", payload: {
      query: body.query,
      project: typeof body.project === "string" ? body.project : undefined,
      minConfidence: typeof body.minConfidence === "number" ? body.minConfidence : undefined,
      limit: typeof body.limit === "number" ? body.limit : undefined,
    } });
    return { status_code: 200, body: result };
  });
  sdk.registerTrigger({ type: "http", function_id: "api::insight-search", config: { api_path: "/agentmemory/insights/search", http_method: "POST" } });
}
