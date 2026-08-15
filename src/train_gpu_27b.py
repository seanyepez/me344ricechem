"""QLoRA fine-tune Gemma 3 27B on the frozen RiceChem split using one A100.

Recorded variant: 4-bit NF4 quantized base
(bitsandbytes), LoRA r=32 α=64 on the same seven projection families, full-sequence
LM loss, MAX_LEN 768, 2 epochs, per-epoch reshuffle (seed 42+epoch), AdamW 2e-4,
dynamic per-batch padding (length-sorted buckets) for throughput. After training +
adapter save, the SAME process (weights already resident) serves a minimal
OpenAI-compatible endpoint so an authorized client can stream the frozen test
manifest over a port-forward; test pairs never land on cluster storage. POST /shutdown ends the
job cleanly and requires a bearer token whenever the server binds beyond loopback.
"""

import gzip
import hmac
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BitsAndBytesConfig

DATA = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3-27b-it")
OUT = Path(os.environ.get("OUTPUT_DIR", "/outputs/ricechem-lora-adapter-27b-gpu"))
BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
SHUTDOWN_TOKEN = os.environ.get("SHUTDOWN_TOKEN")

# Loopback is safe for direct execution. Container/Kubernetes callers may opt into a
# pod-facing bind, but must then protect the process-control endpoint with a token.
if BIND_HOST not in {"127.0.0.1", "localhost", "::1"} and not SHUTDOWN_TOKEN:
    raise RuntimeError(
        "SHUTDOWN_TOKEN is required when BIND_HOST is not a loopback address"
    )

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
MAX_LEN = int(os.environ.get("MAX_LEN", "768"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "2e-4"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "25"))
SEED = int(os.environ.get("SEED", "42"))
torch.manual_seed(SEED)

t_start = time.time()
print(f"CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
try:
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
except Exception as e:  # noqa: BLE001
    print(f"conditional-generation load failed ({e}); trying AutoModelForCausalLM", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    )
model.config.use_cache = False
lora = LoraConfig(
    r=32, lora_alpha=64.0, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
t_model = time.time()
print(f"model loaded in {t_model - t_start:.1f}s", flush=True)


def read_split(name):
    chunks = sorted(DATA.glob(f"{name}.jsonl.gz.*"))
    if chunks:
        blob = b"".join(p.read_bytes() for p in chunks)
        text = gzip.decompress(blob).decode()
    else:
        # ConfigMap-style gz chunks absent: fall back to a plain JSONL split file.
        plain = DATA / f"{name}.jsonl"
        if not plain.exists():
            raise FileNotFoundError(
                f"no {name}.jsonl.gz.* chunks and no {plain} in {DATA}; "
                "mount the prepared cm_chunks/ directory or the plain split files"
            )
        text = plain.read_text()
    return [json.loads(l) for l in text.splitlines() if l]


def to_text(row):
    return (
        f"<start_of_turn>user\n{row['prompt_user']}<end_of_turn>\n"
        f"<start_of_turn>model\n{row['target']}<end_of_turn>\n"
    )


train_rows = read_split("train")
enc_all = tok([to_text(r) for r in train_rows], truncation=False)["input_ids"]
train_kept = [(r, ids) for r, ids in zip(train_rows, enc_all) if len(ids) <= MAX_LEN]
print(f"train: {len(train_kept)} kept, {len(train_rows)-len(train_kept)} dropped over-length", flush=True)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
# Optional linear warmup to LR over WARMUP_STEPS optimizer steps, then constant.
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "0"))
sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP_STEPS))
         if WARMUP_STEPS > 0 else None)
model.train()
step = 0
real_tokens = 0
t_train0 = time.time()
for epoch in range(EPOCHS):
    order = list(range(len(train_kept)))
    random.Random(SEED + epoch).shuffle(order)
    # length-sorted buckets of 64, shuffled bucket order — dynamic padding with mixing
    chunks = [sorted(order[i:i + 64], key=lambda j: len(train_kept[j][1]))
              for i in range(0, len(order), 64)]
    batches = []
    for ch in chunks:
        for i in range(0, len(ch) - BATCH_SIZE + 1, BATCH_SIZE):
            batches.append(ch[i:i + BATCH_SIZE])
    random.Random(142 + epoch).shuffle(batches)
    for bi, batch_idx in enumerate(batches):
        texts = [to_text(train_kept[j][0]) for j in batch_idx]
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ids = enc.input_ids.cuda()
        mask = enc.attention_mask.cuda()
        labels = ids.clone()
        labels[mask == 0] = -100
        real_tokens += int(mask.sum().item())
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        (out.loss / GRAD_ACCUM).backward()
        if (bi + 1) % GRAD_ACCUM == 0:
            opt.step()
            if sched is not None:
                sched.step()
            opt.zero_grad()
            step += 1
            if step % LOG_EVERY == 0:
                el = time.time() - t_train0
                print(f"epoch {epoch} step {step} loss {out.loss.item():.4f} "
                      f"({el:.0f}s, {real_tokens/el:.0f} real tok/s)", flush=True)
t_train1 = time.time()

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUT))
print(f"adapter saved to {OUT}", flush=True)

timing = {
    "lane": "gpu-a100-27b-qlora",
    "seed": SEED,
    "devices": 1,
    "device_name": torch.cuda.get_device_name(0),
    "model_load_secs": round(t_model - t_start, 1),
    "train_secs": round(t_train1 - t_train0, 1),
    "steps": step,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "max_len": MAX_LEN,
    "epochs": EPOCHS,
    "lr": LR,
    "warmup_steps": WARMUP_STEPS,
    "quant": "nf4-double",
    "train_examples": len(train_kept),
    "real_tokens_trained": real_tokens,
    "real_tokens_per_sec": round(real_tokens / (t_train1 - t_train0), 1),
    "total_wall_secs": round(time.time() - t_start, 1),
}
print("TIMING " + json.dumps(timing), flush=True)

# ---- serve mode: stream eval through the resident model, then shut down ----
model.eval()
model.config.use_cache = True
shutdown = threading.Event()


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/shutdown":
            if SHUTDOWN_TOKEN:
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {SHUTDOWN_TOKEN}"
                if not hmac.compare_digest(supplied, expected):
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"forbidden")
                    return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"bye")
            shutdown.set()
            return
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        content = body["messages"][0]["content"]
        use_base = str(body.get("model", "")).lower().startswith("base")
        prompt = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            if use_base:
                with model.disable_adapter():
                    out_ids = model.generate(
                        **enc, max_new_tokens=int(body.get("max_tokens", 5)),
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    )
            else:
                out_ids = model.generate(
                    **enc, max_new_tokens=int(body.get("max_tokens", 5)),
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
        text = tok.decode(out_ids[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        resp = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass


srv = ThreadingHTTPServer((BIND_HOST, PORT), H)
print(f"SERVE_MODE ready on {BIND_HOST}:{PORT}", flush=True)
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()
shutdown.wait()
time.sleep(2)
srv.shutdown()
print("serve mode ended; job complete", flush=True)
