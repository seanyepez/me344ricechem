"""LoRA fine-tune Gemma 3 4B on the frozen RiceChem split using one A100.

The recorded run uses identical prompt text across lanes, LoRA rank 32 / alpha
64 on seven projection families, completion-only loss, MAX_LEN 768, three
epochs, deterministic per-epoch reshuffling, and AdamW. Input is read from
ConfigMap-compatible gzip chunks; the adapter and a final TIMING receipt are
written to the configured output directory.
"""

import gzip
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

DATA = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3-4b-it")
OUT = Path(os.environ.get("OUTPUT_DIR", "/outputs/ricechem-lora-adapter-4b-gpu"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "768"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "2e-4"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))

t_start = time.time()
print(f"CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_PATH)

try:
    from transformers import Gemma3ForConditionalGeneration
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
except Exception as e:
    print(f"conditional-generation load failed ({e}); trying AutoModelForCausalLM", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
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
model = model.cuda()
t_model = time.time()
print(f"model loaded in {t_model - t_start:.1f}s", flush=True)


def read_split(name):
    blob = b"".join(p.read_bytes() for p in sorted(DATA.glob(f"{name}.jsonl.gz.*")))
    return [json.loads(l) for l in gzip.decompress(blob).decode().splitlines() if l]


def to_text(row):
    return (
        f"<start_of_turn>user\n{row['prompt_user']}<end_of_turn>\n"
        f"<start_of_turn>model\n{row['target']}<end_of_turn>\n"
    )


t0 = time.time()
train_rows = read_split("train")
valid_rows = read_split("valid")
lens = [len(x) for x in tok([to_text(r) for r in train_rows], truncation=False)["input_ids"]]
train_kept = [r for r, L in zip(train_rows, lens) if L <= MAX_LEN]
train_dropped = len(train_rows) - len(train_kept)
print(f"train: {len(train_kept)} kept, {train_dropped} dropped over-length", flush=True)


COMPLETION_ONLY = os.environ.get("COMPLETION_ONLY", "1") == "1"


class Pairs(Dataset):
    def __init__(self, rows):
        self.enc = tok(
            [to_text(r) for r in rows], max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        # completion-only loss: mask the prompt so gradient lands on the verdict
        # tokens only; full-sequence loss dilutes a one-token verdict signal.
        prefixes = [
            f"<start_of_turn>user\n{r['prompt_user']}<end_of_turn>\n<start_of_turn>model\n"
            for r in rows
        ]
        self.plens = [len(ids) for ids in tok(prefixes)["input_ids"]]

    def __len__(self):
        return self.enc.input_ids.shape[0]

    def __getitem__(self, i):
        ids = self.enc.input_ids[i]
        mask = self.enc.attention_mask[i]
        labels = ids.clone()
        labels[mask == 0] = -100
        if COMPLETION_ONLY:
            labels[: self.plens[i]] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


real_tokens = 0
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
model.train()
step = 0
t_train0 = time.time()
for epoch in range(EPOCHS):
    order = list(range(len(train_kept)))
    random.Random(42 + epoch).shuffle(order)
    ds = Pairs([train_kept[i] for i in order])
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    for i, batch in enumerate(dl):
        batch = {k: v.cuda() for k, v in batch.items()}
        real_tokens += int(batch["attention_mask"].sum().item())
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        loss.backward()
        if (i + 1) % GRAD_ACCUM == 0:
            opt.step()
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
    "lane": "gpu-a100",
    "completion_only_loss": COMPLETION_ONLY,
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
    "train_examples": len(train_kept),
    "train_dropped_overlong": train_dropped,
    "real_tokens_trained": real_tokens,
    "real_tokens_per_sec": round(real_tokens / (t_train1 - t_train0), 1),
    "total_wall_secs": round(time.time() - t_start, 1),
}
print("TIMING " + json.dumps(timing), flush=True)
