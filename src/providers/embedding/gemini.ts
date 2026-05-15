import type { EmbeddingProvider } from "../../types.js";
import { getEnvVar } from "../../config.js";

const BATCH_LIMIT = 100;
const DEFAULT_MODEL = "gemini-embedding-001";
const DEFAULT_DIMENSIONS = 768;

function normalizeModelName(model: string): string {
  return model.startsWith("models/") ? model : `models/${model}`;
}

export class GeminiEmbeddingProvider implements EmbeddingProvider {
  readonly name = "gemini";
  readonly dimensions: number;
  private apiKey: string;
  private model: string;
  private apiBase: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || getEnvVar("GEMINI_API_KEY") || "";
    if (!this.apiKey) throw new Error("GEMINI_API_KEY is required");
    this.model = normalizeModelName(
      getEnvVar("GEMINI_EMBEDDING_MODEL") || DEFAULT_MODEL,
    );
    this.apiBase =
      `https://generativelanguage.googleapis.com/v1beta/${this.model}:batchEmbedContents`;
    this.dimensions = parseInt(
      getEnvVar("GEMINI_EMBEDDING_DIMENSIONS") || String(DEFAULT_DIMENSIONS),
      10,
    ) || DEFAULT_DIMENSIONS;
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const results: Float32Array[] = [];

    for (let i = 0; i < texts.length; i += BATCH_LIMIT) {
      const chunk = texts.slice(i, i + BATCH_LIMIT);
      const response = await fetch(`${this.apiBase}?key=${this.apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requests: chunk.map((t) => ({
            model: this.model,
            content: { parts: [{ text: t }] },
            outputDimensionality: this.dimensions,
          })),
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(
          `Gemini embedding failed for ${this.model} (${response.status}): ${err}`,
        );
      }

      const data = (await response.json()) as {
        embeddings: Array<{ values: number[] }>;
      };

      for (const emb of data.embeddings) {
        results.push(l2Normalize(new Float32Array(emb.values)));
      }
    }

    return results;
  }
}

let zeroNormWarned = false;

function l2Normalize(vec: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]! * vec[i]!;
  const norm = Math.sqrt(sum);
  if (norm === 0) {
    if (!zeroNormWarned) {
      zeroNormWarned = true;
      process.stderr.write(
        `[agentmemory] warn: Gemini embedding provider returned a zero-norm ` +
          `embedding (length=${vec.length}); leaving it un-normalized. ` +
          `Subsequent zero-norm vectors will not be reported.\n`,
      );
    }
    return vec;
  }
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / norm;
  return vec;
}
