// Phase 0 · Lesson 04 — APIs and keys (TypeScript port, Hugging Face).
// Reads HF_TOKEN from env, parses a minimal .env file, then makes one
// chat-completions call with global fetch to the free Inference API.
// Set MOCK=1 to skip the network entirely.
// Refs: https://huggingface.co/docs/api-inference
//       https://nodejs.org/api/process.html#processenv
//       https://nodejs.org/api/globals.html#fetch (Node 18+ ships fetch)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";


type ChatRequest = {
  model: string;
  max_tokens: number;
  messages: { role: "user" | "assistant"; content: string }[];
};

type ChatResponse = {
  choices: { message: { content: string } }[];
  usage: { prompt_tokens: number; completion_tokens: number };
};

// .env loader. Same shape every framework follows; we skip a dep to stay
// portable. KEY=VALUE per line, # comments, optional surrounding quotes.
function loadDotenv(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {};
  }
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function mergeEnv(): NodeJS.ProcessEnv {
  // process.env wins so users can override the file without editing it.
  const fromFile = loadDotenv(resolve(process.cwd(), ".env"));
  return { ...fromFile, ...process.env };
}

// Fixture matches the real chat-completions response shape, so the
// surrounding code is identical whether MOCK=1 or not.
const MOCK_RESPONSE: ChatResponse = {
  choices: [
    {
      message: {
        content: "A neural network is a stack of differentiable functions that learns patterns by adjusting weights against a loss signal.",
      },
    },
  ],
  usage: { prompt_tokens: 12, completion_tokens: 28 },
};

async function callChat(apiKey: string, request: ChatRequest): Promise<ChatResponse> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    return MOCK_RESPONSE;
  }

  const resp = await fetch("https://router.huggingface.co/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${apiKey}`,
    },
    body: JSON.stringify(request),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`huggingface ${resp.status}: ${body.slice(0, 200)}`);
  }
  return (await resp.json()) as ChatResponse;
}

async function main(): Promise<number> {
  const env = mergeEnv();
  const model = (env.LLM_MODEL ?? "").trim() || "Qwen/Qwen2.5-7B-Instruct";
  const apiKey = env.HF_TOKEN ?? "mock";
  const usingMock = process.env.MOCK === "1" || apiKey === "mock";

  process.stdout.write("=== API Calls ===\n\n");
  process.stdout.write(
    usingMock
      ? "Mode: MOCK (no network). Unset MOCK and export HF_TOKEN for a live call.\n\n"
      : "Mode: LIVE.\n\n",
  );

  const request: ChatRequest = {
    model,
    max_tokens: 256,
    messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
  };

  try {
    const response = await callChat(apiKey, request);
    const text = response.choices[0]?.message.content ?? "";
    process.stdout.write(`response: ${text}\n`);
    process.stdout.write(
      `tokens: ${response.usage.prompt_tokens} in, ${response.usage.completion_tokens} out\n`,
    );
    return 0;
  } catch (err) {
    process.stderr.write(`request failed: ${(err as Error).message}\n`);
    return 1;
  }
}

main().then((code) => process.exit(code));
