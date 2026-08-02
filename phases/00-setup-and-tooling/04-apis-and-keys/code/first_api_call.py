# Phase 0 · Lesson 04 — APIs and keys (Hugging Face port).
# Same pattern as every AI API: key + endpoint + JSON request/response.
# Uses the free Serverless Inference API (no card required, rate-limited).
# Set MOCK=1 to skip the network entirely (for CI / keyless demos).
# Refs: https://huggingface.co/docs/api-inference
#       https://huggingface.co/docs/api-inference/tasks/chat-completion
# See docs/en.md in this lesson directory.

import json
import os
import urllib.request

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_URL = "https://router.huggingface.co/v1/chat/completions"

MOCK_RESPONSE = {
    "choices": [{"message": {"role": "assistant", "content": "A neural network is a stack of differentiable functions that learns patterns by adjusting weights against a loss signal."}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 28},
}


def _messages():
    return [{"role": "user", "content": "What is a neural network in one sentence?"}]


def call_with_sdk():
    if os.environ.get("MOCK") == "1":
        print(f"SDK response: {MOCK_RESPONSE['choices'][0]['message']['content']}")
        print("Tokens used: 12 in, 28 out (mock)")
        return

    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Set HF_TOKEN environment variable first (https://huggingface.co/settings/tokens)")
        return

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("Install the SDK: uv pip install huggingface_hub")
        return

    client = InferenceClient(token=api_key)
    completion = client.chat_completion(
        model=MODEL,
        max_tokens=256,
        messages=_messages(),
    )
    print(f"SDK response: {completion.choices[0].message.content}")
    print(f"Tokens used: {completion.usage.prompt_tokens} in, {completion.usage.completion_tokens} out")


def call_raw_http():
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Set HF_TOKEN environment variable first (https://huggingface.co/settings/tokens)")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "ai-engineering-from-scratch/0.1",
    }
    body = json.dumps({
        "model": MODEL,
        "max_tokens": 256,
        "messages": _messages(),
    }).encode()

    req = urllib.request.Request(HF_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Raw HTTP response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['prompt_tokens']} in, {result['usage']['completion_tokens']} out")


if __name__ == "__main__":
    print("=== API Calls ===\n")
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()
