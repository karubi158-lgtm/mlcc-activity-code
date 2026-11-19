import "dotenv/config";
import express from "express";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

// ---------------------------------------------------------------------------
// In-memory LRU Response Cache + In-Flight Deduplication (prevents 429 bursts)
// ---------------------------------------------------------------------------
const MAX_CACHE_ENTRIES = 250;
const CACHE_TTL_MS = 30 * 60 * 1000;

const responseCache = new Map<string, { value: unknown; expiresAt: number }>();
const inFlightRequests = new Map<string, Promise<unknown>>();

function makeCacheKey(prefix: string, ...parts: string[]): string {
  const hash = crypto.createHash("sha256");
  hash.update(prefix);
  for (const part of parts) {
    hash.update("\0");
    hash.update(part);
  }
  return hash.digest("hex");
}

function getCached<T>(key: string): T | null {
  const entry = responseCache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    responseCache.delete(key);
    return null;
  }
  // Refresh LRU recency
  responseCache.delete(key);
  responseCache.set(key, entry);
  return entry.value as T;
}

function setCached(key: string, value: unknown): void {
  if (responseCache.size >= MAX_CACHE_ENTRIES) {
    const oldestKey = responseCache.keys().next().value;
    if (oldestKey) responseCache.delete(oldestKey);
  }
  responseCache.set(key, { value, expiresAt: Date.now() + CACHE_TTL_MS });
}

function getGenAIClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface ParsedApiError {
  status: number;
  message: string;
  retryAfterSeconds?: number;
}

function parseGenAiError(err: unknown): ParsedApiError {
  const raw = err instanceof Error ? err.message : String(err);
  let status = 500;
  let retryAfterSeconds: number | undefined;

  if (raw.includes("429") || raw.includes("RESOURCE_EXHAUSTED") || raw.toLowerCase().includes("quota")) {
    status = 429;
    const retryMatch = raw.match(/retry in\s*([\d.]+)s/i) || raw.match(/"retryDelay"\s*:\s*"(\d+)s"/i);
    if (retryMatch) {
      retryAfterSeconds = Math.max(2, Math.ceil(Number(retryMatch[1])));
    } else {
      retryAfterSeconds = 5;
    }
    return {
      status: 429,
      retryAfterSeconds,
      message: `Cloud Vision AI rate limit reached. Please wait ${retryAfterSeconds}s and try again, or attach a billing-enabled API key in Settings > Secrets.`,
    };
  }

  if (raw.includes("403") || raw.includes("PERMISSION_DENIED") || raw.includes("API_KEY_INVALID")) {
    return {
      status: 403,
      message: "Gemini API key permission error. Please verify your key in Settings > Secrets.",
    };
  }

  // Strip raw JSON wrappers if present
  try {
    const maybeJson = JSON.parse(raw);
    if (maybeJson?.error?.message) {
      return { status: Number(maybeJson.error.code) || 500, message: maybeJson.error.message };
    }
  } catch {
    // plain text
  }

  return { status, message: raw.slice(0, 260) };
}

/**
 * Attempts high-throughput Flash-Lite first, then falls back to Gemini 4 Hydrogen,
 * with a short bounded retry on 429 rate-limit responses.
 */
async function generateJsonWithResilience(
  ai: GoogleGenAI,
  buildRequest: (model: string, thinkingLevel: ThinkingLevel) => Parameters<GoogleGenAI["models"]["generateContent"]>[0]
) {
  const attempts: Array<{ model: string; level: ThinkingLevel; waitBeforeMs?: number }> = [
    { model: "gemini-3.1-flash-lite", level: ThinkingLevel.MINIMAL },
    { model: "gemini-4-hydrogen", level: ThinkingLevel.LOW },
    { model: "gemini-3.1-flash-lite", level: ThinkingLevel.MINIMAL, waitBeforeMs: 3600 },
  ];

  let lastErr: unknown;
  for (const attempt of attempts) {
    if (attempt.waitBeforeMs) {
      await sleep(attempt.waitBeforeMs);
    }
    try {
      return await ai.models.generateContent(buildRequest(attempt.model, attempt.level));
    } catch (err) {
      lastErr = err;
      const parsed = parseGenAiError(err);
      const isNotFound = String(err).includes("404") || String(err).includes("NOT_FOUND");
      if (parsed.status === 429 || isNotFound) {
        continue;
      }
      throw err;
    }
  }
  throw lastErr;
}

async function callGcpVisionRestIfConfigured(base64Data: string) {
  const gcpKey = process.env.GOOGLE_CLOUD_VISION_API_KEY;
  if (!gcpKey) return null;

  try {
    const resp = await fetch(
      `https://vision.googleapis.com/v1/images:annotate?key=${encodeURIComponent(gcpKey)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          requests: [
            {
              image: { content: base64Data },
              features: [{ type: "DOCUMENT_TEXT_DETECTION", maxResults: 1 }],
            },
          ],
        }),
      }
    );
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      responses?: Array<{
        fullTextAnnotation?: {
          text?: string;
          pages?: Array<{
            blocks?: Array<{
              paragraphs?: Array<{
                words?: Array<{
                  symbols?: Array<{ text?: string; confidence?: number }>;
                }>;
              }>;
            }>;
          }>;
        };
      }>;
    };
    const annotation = data?.responses?.[0]?.fullTextAnnotation;
    const text = annotation?.text?.trim();
    if (!text) return null;

    const tokens: Array<{ char: string; confidence: number; kind: string }> = [];
    for (const page of annotation?.pages ?? []) {
      for (const block of page.blocks ?? []) {
        for (const para of block.paragraphs ?? []) {
          for (const word of para.words ?? []) {
            for (const sym of word.symbols ?? []) {
              const ch = sym.text ?? "";
              if (!ch.trim()) continue;
              tokens.push({
                char: ch,
                confidence: Math.round((sym.confidence ?? 0.92) * 1000) / 1000,
                kind: /\d/.test(ch) ? "digit" : /[a-zA-Z]/.test(ch) ? "letter" : "symbol",
              });
            }
          }
        }
      }
    }
    const meanConf =
      tokens.length > 0
        ? tokens.reduce((acc, t) => acc + t.confidence, 0) / tokens.length
        : 0.93;

    return {
      detectedText: text,
      confidence: Math.round(meanConf * 1000) / 1000,
      tokens,
      provider: "GOOGLE_CLOUD_VISION_REST_API",
    };
  } catch {
    return null;
  }
}

function safeEvaluateMath(expr: string): string | null {
  const cleaned = expr.replace(/×/g, "*").replace(/÷/g, "/").replace(/=/g, "").trim();
  if (!/^[\d+\-*/().\s]+$/.test(cleaned)) return null;
  if (!/[+\-*/]/.test(cleaned)) return null;
  try {
    const fn = new Function(`"use strict"; return (${cleaned});`);
    const val = fn();
    if (typeof val === "number" && Number.isFinite(val)) {
      return String(Math.round(val * 10000) / 10000);
    }
    return null;
  } catch {
    return null;
  }
}

async function createServer() {
  const app = express();
  app.use(express.json({ limit: "12mb" }));

  // Health check
  app.get("/api/health", (_req, res) => {
    res.json({ status: "healthy", service: "Digit AIpro Full-Stack Backend" });
  });

  // ---------------------------------------------------------------------------
  // Module I: Vision OCR Fallback Endpoint (Digits, Letters, Words, Equations)
  // ---------------------------------------------------------------------------
  app.post("/api/vision-ocr", async (req, res) => {
    try {
      const { imageBase64, mode = "auto" } = req.body as {
        imageBase64?: string;
        mode?: "auto" | "digit" | "letter" | "word" | "math";
      };

      if (!imageBase64 || typeof imageBase64 !== "string") {
        return res.status(400).json({ error: "Missing imageBase64 payload." });
      }

      const match = imageBase64.match(/^data:(image\/[a-zA-Z0-9+.-]+);base64,(.+)$/);
      const mimeType = match ? match[1] : "image/png";
      const rawBase64 = match ? match[2] : imageBase64;

      if (!rawBase64 || rawBase64.length < 32) {
        return res.status(400).json({ error: "Empty or invalid image data." });
      }

      const cacheKey = makeCacheKey("vision-ocr", mode, rawBase64);
      const cached = getCached<Record<string, unknown>>(cacheKey);
      if (cached) {
        return res.json({ ...cached, cached: true });
      }

      const runTask = async () => {
        // 1. Optional direct GCP Cloud Vision API if GOOGLE_CLOUD_VISION_API_KEY is set
        const gcpResult = await callGcpVisionRestIfConfigured(rawBase64);
        if (gcpResult && gcpResult.detectedText) {
          const mathVal = safeEvaluateMath(gcpResult.detectedText);
          const payload = {
            success: true,
            detectedText: gcpResult.detectedText,
            contentType: mathVal
              ? "math_expression"
              : gcpResult.detectedText.length === 1 && /\d/.test(gcpResult.detectedText)
                ? "digit"
                : gcpResult.detectedText.length === 1 && /[a-zA-Z]/.test(gcpResult.detectedText)
                  ? "letter"
                  : /\s/.test(gcpResult.detectedText)
                    ? "sentence"
                    : /^\d+$/.test(gcpResult.detectedText)
                      ? "number"
                      : "word",
            confidence: gcpResult.confidence,
            tokens: gcpResult.tokens,
            mathEvaluated: mathVal,
            explanation: "Recognized via Google Cloud Vision DOCUMENT_TEXT_DETECTION.",
            provider: gcpResult.provider,
          };
          setCached(cacheKey, payload);
          return payload;
        }

        // 2. Server-side Gemini Vision OCR Engine with Flash-Lite + Hydrogen Fallback
        const ai = getGenAIClient();
        if (!ai) {
          const err = new Error("Vision OCR service unavailable: GEMINI_API_KEY is not configured on the server.");
          (err as Error & { statusCode?: number }).statusCode = 503;
          throw err;
        }

        const modeHint =
          mode === "digit"
            ? "Focus carefully on recognizing the single handwritten digit (0-9) or single character."
            : mode === "letter"
              ? "Focus on recognizing the handwritten letter (A-Z or a-z)."
              : mode === "math"
                ? "Transcribe the handwritten mathematical expression accurately (digits and operators +, -, *, /, =)."
                : "Transcribe all handwritten digits, letters, words, sentences, documents, or math expressions accurately.";

        const response = await generateJsonWithResilience(ai, (model, thinkingLevel) => ({
          model,
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType,
                  data: rawBase64,
                },
              },
              {
                text:
                  `You are a high-precision handwriting OCR engine (Cloud Vision DOCUMENT_TEXT_DETECTION fallback). ` +
                  `${modeHint} ` +
                  `The image may have white strokes on a dark navy/black background OR dark ink on a light/paper background. ` +
                  `Return ONLY the exact characters/words visible in the image. If blank, return "" for detectedText and confidence 0.0.`,
              },
            ],
          },
          config: {
            thinkingConfig: { thinkingLevel },
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                detectedText: {
                  type: Type.STRING,
                  description: "Exact transcribed handwritten digit, letter, word, number, or expression.",
                },
                contentType: {
                  type: Type.STRING,
                  description: "One of: digit, letter, number, word, math_expression, sentence, document",
                },
                confidence: {
                  type: Type.NUMBER,
                  description: "Confidence score from 0.0 to 1.0",
                },
                tokens: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      char: { type: Type.STRING },
                      confidence: { type: Type.NUMBER },
                      kind: { type: Type.STRING },
                    },
                  },
                  description: "Per-character breakdown in reading order.",
                },
                explanation: {
                  type: Type.STRING,
                  description: "Brief 1-sentence note describing what was recognized.",
                },
              },
            },
          },
        }));

        const parsed = JSON.parse((response.text ?? "{}").trim()) as {
          detectedText?: string;
          contentType?: string;
          confidence?: number;
          tokens?: Array<{ char: string; confidence: number; kind: string }>;
          explanation?: string;
        };

        const detectedText = (parsed.detectedText ?? "").trim();
        const mathEvaluated = detectedText ? safeEvaluateMath(detectedText) : null;
        const confidence = Math.max(0, Math.min(1, Number(parsed.confidence ?? (detectedText ? 0.94 : 0))));
        const tokens =
          Array.isArray(parsed.tokens) && parsed.tokens.length > 0
            ? parsed.tokens
            : detectedText
                .split("")
                .filter((ch) => ch.trim().length > 0)
                .map((ch) => ({
                  char: ch,
                  confidence: Math.round(confidence * 1000) / 1000,
                  kind: /\d/.test(ch) ? "digit" : /[a-zA-Z]/.test(ch) ? "letter" : "symbol",
                }));

        const payload = {
          success: true,
          detectedText,
          contentType: mathEvaluated ? "math_expression" : parsed.contentType || "word",
          confidence: Math.round(confidence * 1000) / 1000,
          tokens,
          mathEvaluated,
          explanation:
            parsed.explanation ||
            (detectedText
              ? `Vision OCR extracted "${detectedText}" with ${(confidence * 100).toFixed(1)}% confidence.`
              : "No recognizable handwriting found on canvas."),
          provider: "CLOUD_VISION_MULTIMODAL_OCR",
        };
        if (detectedText) {
          setCached(cacheKey, payload);
        }
        return payload;
      };

      let pending = inFlightRequests.get(cacheKey);
      if (!pending) {
        pending = runTask().finally(() => inFlightRequests.delete(cacheKey));
        inFlightRequests.set(cacheKey, pending);
      }

      const result = await pending;
      return res.json(result);
    } catch (err) {
      console.error("Error in /api/vision-ocr:", err);
      const statusOverride = (err as Error & { statusCode?: number }).statusCode;
      if (statusOverride) {
        return res.status(statusOverride).json({ error: (err as Error).message });
      }
      const parsed = parseGenAiError(err);
      return res.status(parsed.status).json({
        error: parsed.message,
        retryAfterSeconds: parsed.retryAfterSeconds,
      });
    }
  });

  // ---------------------------------------------------------------------------
  // Module I: Cloud Translation Endpoint
  // ---------------------------------------------------------------------------
  app.post("/api/translate", async (req, res) => {
    try {
      const { text, targetLanguage = "es" } = req.body as {
        text?: string;
        targetLanguage?: string;
      };
      if (!text || !text.trim()) {
        return res.status(400).json({ error: "Text is required for translation." });
      }

      const trimmed = text.trim();
      const cacheKey = makeCacheKey("translate", targetLanguage, trimmed);
      const cached = getCached<Record<string, unknown>>(cacheKey);
      if (cached) {
        return res.json({ ...cached, cached: true });
      }

      const ai = getGenAIClient();
      if (!ai) {
        return res.status(503).json({ error: "Translation service unavailable: GEMINI_API_KEY missing." });
      }

      const response = await generateJsonWithResilience(ai, (model, thinkingLevel) => ({
        model,
        contents: `Translate the following text accurately into language code "${targetLanguage}":\n\n${trimmed}`,
        config: {
          thinkingConfig: { thinkingLevel },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              translatedText: { type: Type.STRING },
              detectedSourceLanguage: { type: Type.STRING },
            },
          },
        },
      }));

      const parsed = JSON.parse((response.text ?? "{}").trim()) as {
        translatedText?: string;
        detectedSourceLanguage?: string;
      };

      const payload = {
        success: true,
        inputText: trimmed,
        translatedText: parsed.translatedText ?? trimmed,
        detectedSourceLanguage: parsed.detectedSourceLanguage ?? "en",
        targetLanguage,
      };
      setCached(cacheKey, payload);
      return res.json(payload);
    } catch (err) {
      console.error("Error in /api/translate:", err);
      const parsed = parseGenAiError(err);
      return res.status(parsed.status).json({
        error: parsed.message,
        retryAfterSeconds: parsed.retryAfterSeconds,
      });
    }
  });

  // ---------------------------------------------------------------------------
  // Module I: Cloud Natural Language Sentiment Endpoint
  // ---------------------------------------------------------------------------
  app.post("/api/sentiment", async (req, res) => {
    try {
      const { text } = req.body as { text?: string };
      if (!text || !text.trim()) {
        return res.status(400).json({ error: "Feedback text is required." });
      }

      const trimmed = text.trim();
      const cacheKey = makeCacheKey("sentiment", trimmed);
      const cached = getCached<Record<string, unknown>>(cacheKey);
      if (cached) {
        return res.json({ ...cached, cached: true });
      }

      const ai = getGenAIClient();
      if (!ai) {
        return res.status(503).json({ error: "Sentiment service unavailable: GEMINI_API_KEY missing." });
      }

      const response = await generateJsonWithResilience(ai, (model, thinkingLevel) => ({
        model,
        contents: `Perform Google Cloud Natural Language style sentiment and entity analysis on:\n\n"${trimmed}"`,
        config: {
          thinkingConfig: { thinkingLevel },
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.NUMBER, description: "Polarity from -1.0 (very negative) to +1.0 (very positive)" },
              magnitude: { type: Type.NUMBER, description: "Emotional intensity >= 0.0" },
              category: { type: Type.STRING, description: "Positive, Neutral, or Negative" },
              entities: { type: Type.ARRAY, items: { type: Type.STRING } },
            },
          },
        },
      }));

      const parsed = JSON.parse((response.text ?? "{}").trim()) as {
        score?: number;
        magnitude?: number;
        category?: string;
        entities?: string[];
      };

      const score = Math.max(-1, Math.min(1, Math.round((parsed.score ?? 0) * 100) / 100));
      const magnitude = Math.max(0, Math.round((parsed.magnitude ?? 0.5) * 100) / 100);
      const category =
        parsed.category === "Positive" || parsed.category === "Negative" || parsed.category === "Neutral"
          ? parsed.category
          : score >= 0.25
            ? "Positive"
            : score <= -0.25
              ? "Negative"
              : "Neutral";

      const payload = {
        success: true,
        score,
        magnitude,
        category,
        entities: parsed.entities ?? [],
      };
      setCached(cacheKey, payload);
      return res.json(payload);
    } catch (err) {
      console.error("Error in /api/sentiment:", err);
      const parsed = parseGenAiError(err);
      return res.status(parsed.status).json({
        error: parsed.message,
        retryAfterSeconds: parsed.retryAfterSeconds,
      });
    }
  });

  // Guaranteed valid JSON download for the Colab / Jupyter notebook
  app.get(["/Digit_AIpro_Trained_Model.ipynb", "/api/download-notebook"], (_req, res) => {
    const publicPath = path.join(__dirname, "public", "Digit_AIpro_Trained_Model.ipynb");
    const distPath = path.join(__dirname, "dist", "Digit_AIpro_Trained_Model.ipynb");
    const targetPath = fs.existsSync(publicPath) ? publicPath : fs.existsSync(distPath) ? distPath : null;
    if (!targetPath) {
      return res.status(404).json({ error: "Notebook file not found on server." });
    }
    res.setHeader("Content-Type", "application/x-ipynb+json; charset=utf-8");
    res.setHeader("Content-Disposition", 'attachment; filename="Digit_AIpro_Trained_Model.ipynb"');
    return res.sendFile(targetPath);
  });

  // Mount Vite dev middleware or serve production static files
  if (process.env.NODE_ENV !== "production") {
    const { createServer: createViteServer } = await import("vite");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, "dist");
    app.use(express.static(distDir));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Digit AIpro server listening on http://0.0.0.0:${PORT}`);
  });
}

createServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
