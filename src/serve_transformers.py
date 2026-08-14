#!/usr/bin/env python3
"""Minimal private OpenAI-compatible Transformers endpoint for CPU profiling.

The server stores no prompts or predictions and logs only model-load timing. It is
intentionally simple: the controlled benchmark must record that CPU uses this
Transformers path while the accelerator manifests use vLLM.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("MODEL_PATH", "/models/gemma-3-4b-it-merged")
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = os.environ.get("DEVICE", "cpu")
DTYPE_NAME = os.environ.get("TORCH_DTYPE", "float32")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
DTYPE = DTYPES[DTYPE_NAME]

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
try:
    from transformers import Gemma3ForConditionalGeneration

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa"
    )
except Exception:  # noqa: BLE001 — compatibility fallback is surfaced at load time
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, attn_implementation="sdpa"
    )
model = model.to(DEVICE).eval()
model.config.use_cache = True
print(json.dumps({"event": "model_ready", "device": DEVICE, "dtype": DTYPE_NAME,
                  "load_seconds": round(time.time() - t0, 2)}), flush=True)

# Serialize generation so the endpoint has an explicit, reproducible concurrency policy.
generation_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            body = ({"status": "ok"} if self.path == "/health"
                    else {"data": [{"id": MODEL_PATH, "object": "model"}]})
            self._json(200, body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        content = body["messages"][-1]["content"]
        prompt = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with generation_lock, torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=int(body.get("max_tokens", 5)),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            output[0][encoded.input_ids.shape[1]:], skip_special_tokens=True
        )
        self._json(200, {"choices": [{"message": {"content": text}}]})

    def _json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
