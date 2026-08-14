"""LoRA fine-tune Gemma 3 4B on the frozen RiceChem split using TPU v5e-8.

Adapted from an authorized ME344 Lab 2 reference implementation: same model
load, LoRA wrap (rank 32 / alpha 64), rematerialization, and
adapter export. Changes: (1) dataset = RiceChem rubric-entailment pairs read from
ConfigMap-mounted gzip JSONL (no buckets — data never lands in shared storage);
(2) MAX_LEN 768, batch/epochs from env; (3) per-epoch reshuffle (seed 42+epoch);
(4) wall-clock + token instrumentation emitted as a final TIMING json line for the
hardware-economics table. This recorded TPU variant uses full-sequence LM loss.
"""

import dataclasses
import gzip
import json
import logging
import os
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import qwix
from flax import nnx
from orbax import checkpoint as ocp
from safetensors.numpy import save_file
from transformers import AutoConfig, AutoTokenizer

from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.models.safetensors_saver import join_path
from tunix.rl import reshard
from tunix.sft import peft_trainer, utils as sft_utils

DATA = Path(os.environ.get("DATA_DIR", "/data"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/gemma-3-4b-it"))
CKPT_PATH = Path(os.environ.get("CHECKPOINT_DIR", "/outputs/ricechem-lora-ckpt-4b"))
ADAPTER_PATH = Path(os.environ.get("OUTPUT_DIR", "/outputs/ricechem-lora-adapter-4b"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_LEN = int(os.environ.get("MAX_LEN", "768"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LORA_RANK = 32
LORA_ALPHA = 64.0
LR = float(os.environ.get("LR", "1e-3"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "150"))

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logging.getLogger("orbax").setLevel(logging.WARNING)
log = logging.getLogger("ricechem-sft").info

t_start = time.time()
log("TPU devices: %d | %s", jax.device_count(), [d.coords for d in jax.devices()])
MESH = [(jax.device_count(), 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

config = gemma3_lib.ModelConfig.gemma3_4b_it()
hf_config = AutoConfig.from_pretrained(str(MODEL_PATH))
hf_vocab_size = getattr(getattr(hf_config, "text_config", hf_config), "vocab_size")
if hf_vocab_size != config.num_embed:
    log("Overriding Tunix num_embed: %d -> %d", config.num_embed, hf_vocab_size)
    config = dataclasses.replace(config, num_embed=hf_vocab_size)
list(MODEL_PATH.iterdir())
with jax.set_mesh(mesh):
    base_model = params_safetensors_lib.create_model_from_safe_tensors(str(MODEL_PATH), config, mesh)
t_model = time.time()
log("Base model loaded in %.1fs", t_model - t_start)

lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
)
B, T = jax.device_count(), 8
trace_input = dict(
    last_tokens=jnp.ones((B, T), dtype=jnp.int32),
    positions=jnp.tile(jnp.arange(T)[None, :], (B, 1)),
    cache=None,
    attention_mask=jnp.tril(jnp.ones((B, T, T), dtype=jnp.bool_)),
)
with jax.set_mesh(mesh):
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, rngs=nnx.Rngs(0), **trace_input)
del base_model
lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
n_lora = sum(p.size for p in jax.tree.leaves(nnx.state(lora_model, nnx.LoRAParam)))
assert n_lora > 0, "qwix matched zero modules"
log("LoRA injected — %.1fM trainable params.", n_lora / 1e6)

remat_cfg = dataclasses.replace(config, remat_config=gemma3_lib.RematConfig.DECODER)
lora_model.config = remat_cfg
for layer in lora_model.layers:
    layer.config = remat_cfg

hf_tok = AutoTokenizer.from_pretrained(str(MODEL_PATH))


def read_split(name):
    blob = b"".join(
        p.read_bytes() for p in sorted(DATA.glob(f"{name}.jsonl.gz.*"))
    )
    rows = [json.loads(l) for l in gzip.decompress(blob).decode().splitlines() if l]
    return rows


def to_text(row):
    return (
        f"<start_of_turn>user\n{row['prompt_user']}<end_of_turn>\n"
        f"<start_of_turn>model\n{row['target']}<end_of_turn>\n"
    )


t0 = time.time()
train_rows = read_split("train")
log("Rows: train=%d (no in-training eval: final-step adapter is the declared "
    "selection rule, and a second compiled eval program OOMs v5e-8 HBM at bs16xlen768)",
    len(train_rows))

# Drop examples that would truncate (the target token must survive).
def keep_fits(rows, name):
    texts = [to_text(r) for r in rows]
    lens = [len(ids) for ids in hf_tok(texts, truncation=False)["input_ids"]]
    kept = [r for r, L in zip(rows, lens) if L <= MAX_LEN]
    dropped = len(rows) - len(kept)
    log("%s: %d kept, %d dropped as over-length (max_len=%d)", name, len(kept), dropped, MAX_LEN)
    return kept, dropped


train_rows, train_dropped = keep_fits(train_rows, "train")
valid_dropped = 0


def tokenize(rows):
    texts = [to_text(r) for r in rows]
    enc = hf_tok(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="np")
    return enc.input_ids.astype(np.int32), enc.attention_mask.astype(bool)


_TRIL = np.tril(np.ones((MAX_LEN, MAX_LEN), dtype=bool))


def to_batches(rows):
    # Keep batches in host memory; pre-materializing attention masks on device
    # consumes HBM before the training program starts. JAX transfers per step.
    tokens, mask = tokenize(rows)
    out = []
    for i in range(0, len(tokens) - BATCH_SIZE + 1, BATCH_SIZE):
        tok, m = tokens[i:i + BATCH_SIZE], mask[i:i + BATCH_SIZE]
        positions = np.maximum(np.cumsum(m, axis=1) - 1, 0).astype(np.int32)
        attn = _TRIL[None, :, :] & m[:, None, :]
        out.append({
            "input_tokens": tok,
            "input_mask": m,
            "positions": positions,
            "attention_mask": attn,
        })
    return out


train_ds = []
real_tokens = 0
for epoch in range(EPOCHS):
    order = list(range(len(train_rows)))
    random.Random(42 + epoch).shuffle(order)
    shuffled = [train_rows[i] for i in order]
    batches = to_batches(shuffled)
    for b in batches:
        real_tokens += int(np.asarray(b["input_mask"]).sum())
    train_ds.extend(batches)
val_ds = []
t_data = time.time()
log("Batches: train=%d (over %d epochs) val=disabled shape=%s (%.1fs)",
    len(train_ds), EPOCHS, train_ds[0]["input_tokens"].shape, t_data - t0)

MAX_STEPS = len(train_ds)
cfg = dict(
    eval_every_n_steps=10**9,
    max_steps=MAX_STEPS,
    checkpoint_root_directory=str(CKPT_PATH),
    checkpointing_options=ocp.CheckpointManagerOptions(save_interval_steps=300, max_to_keep=2),
)
training_config = peft_trainer.TrainingConfig(**cfg)
trainer = peft_trainer.PeftTrainer(lora_model, optax.adamw(LR), training_config)
log("Starting train: %d steps (bs=%d, len=%d, %d epochs). First step is XLA compile…",
    MAX_STEPS, BATCH_SIZE, MAX_LEN, EPOCHS)
t_train0 = time.time()
with jax.set_mesh(mesh):
    trainer.train(train_ds, val_ds)
t_train1 = time.time()
log("Training complete — checkpoint at %s", CKPT_PATH)

lora_layers = {}
for path, value in nnx.iter_graph(lora_model):
    if isinstance(value, nnx.LoRAParam):
        path_str = join_path(path[:-1])
        if path_str in lora_layers:
            assert "lora_b" in str(path[-1])
            lora_layers[path_str].append(np.asarray(value.value))
        else:
            assert "lora_a" in str(path[-1])
            lora_layers[path_str] = [np.asarray(value.value)]


def to_peft(name, t):
    arr = np.asarray(t)
    return arr.reshape(-1, LORA_RANK).T if name == "lora_A" else arr.reshape(LORA_RANK, -1).T


adapter = {}
for key, (lora_a, lora_b) in gemma3_params._extract_gemma3_lora_layers(lora_layers).items():
    hf = gemma3_params._gemma3_state_key_to_safetensors_key(key).removesuffix(".weight")
    adapter[f"{hf}.lora_A.weight"] = np.ascontiguousarray(to_peft("lora_A", lora_a))
    adapter[f"{hf}.lora_B.weight"] = np.ascontiguousarray(to_peft("lora_B", lora_b))

ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
save_file(adapter, str(ADAPTER_PATH / "adapter_model.safetensors"))
(ADAPTER_PATH / "adapter_config.json").write_text(json.dumps({
    "peft_type": "LORA", "r": LORA_RANK, "lora_alpha": LORA_ALPHA,
    "base_model_name_or_path": "google/gemma-3-4b-it",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}, indent=2))
log("Adapter (%d tensors) written to %s", len(adapter), ADAPTER_PATH)

timing = {
    "lane": "tpu-v5e",
    "devices": jax.device_count(),
    "model_load_secs": round(t_model - t_start, 1),
    "data_prep_secs": round(t_data - t0, 1),
    "train_secs": round(t_train1 - t_train0, 1),
    "steps": MAX_STEPS,
    "batch_size": BATCH_SIZE,
    "max_len": MAX_LEN,
    "epochs": EPOCHS,
    "lr": LR,
    "train_examples": len(train_rows),
    "train_dropped_overlong": train_dropped,
    "valid_dropped_overlong": valid_dropped,
    "real_tokens_trained": real_tokens,
    "padded_tokens_trained": MAX_STEPS * BATCH_SIZE * MAX_LEN,
    "real_tokens_per_sec": round(real_tokens / (t_train1 - t_train0), 1),
    "total_wall_secs": round(time.time() - t_start, 1),
}
print("TIMING " + json.dumps(timing), flush=True)
