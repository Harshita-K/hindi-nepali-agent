# Training Algorithm — Full Walkthrough with Numbers

This document explains, mechanically and step by step, exactly what happens
during pretraining — from raw data to a trained checkpoint — with the real
numbers from the actual Hindi (Model H) and Nepali (Model L) runs. It's a
companion to `report/phase2/report.md` (which also covers the *design
reasoning* behind each choice); this file is the "what exactly runs, in
what order, with what numbers" reference.

The algorithm and code are **identical** for both languages — only the
data and tokenizer differ. Numbers below are cited per-language wherever
they differ.

---

## 1. Dataset

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Train docs | 1,340,887 | 1,855,143 |
| Val docs | 13,682 | 18,930 |
| Test docs | 13,684 | 18,931 |
| Vocabulary size | 8,000 | 8,000 |
| Test-set tokens (measured) | 6,630,400 | 7,754,752 |
| Test-set bytes (UTF-8, measured) | 63,021,969 | 85,985,088 |
| Train-set tokens (measured) | **675,033,768** | not captured in this session — see `nepali/data/bin/train_meta.json` for the exact figure |

Source: `{hindi,nepali}/data/splits/{train,val,test}.jsonl`, Phase 1's
cleaned, deduplicated, ≥20%-manually-collected corpora. Each language has
its own SentencePiece BPE tokenizer (Phase 1, vocab size 8,000 each, no
pretrained tokenizer).

---

## 2. Step 1 — Tokenize to a flat binary (`common/train/data.py::build_bin`)

For each split (train, val, test):
1. Read each JSON line, extract `"text"`.
2. Encode with the language's SentencePiece model → list of token ids.
3. Append the tokenizer's EOS id after every document (document boundary
   marker — the only one that exists in the resulting stream).
4. Append all ids to a running buffer; flush to disk as `uint16` every 2M
   tokens (bounds peak memory regardless of corpus size).
5. Result: one flat binary file per split (`train.bin`, `val.bin`,
   `test.bin`) — just a long 1-D array of token ids, nothing else.

This runs once per split, not once per epoch — training never re-tokenizes
text.

## 3. Step 2 — Sample a batch (`common/train/data.py::get_batch`)

Called every micro-batch during training (and during validation/eval):

1. Memory-map the binary file (`np.memmap`) — pages are pulled from disk
   only as touched, so the whole file is never loaded into RAM at once.
2. Draw `batch_size` (32) independent random integers `i` uniformly from
   `[0, total_tokens - block_size - 1)` — a fresh, unrelated-to-anything-
   before draw every single call.
3. For each `i`: `x = data[i : i+512]` (input), `y = data[i+1 : i+513]`
   (target — `x` shifted one token later, i.e. the token that actually
   followed each position in the real corpus).
4. Stack the 32 pairs into two `(32, 512)` tensors, move to GPU.

This is **random sampling with replacement**, not epoch-based shuffling —
see §7 for why, and what that means for how much of the corpus gets seen.

## 4. Step 3 — Forward pass (`common/model/transformer.py::GPT`)

Input `x`: `(32, 512)` token ids.

1. **Embed**: token embedding `(8000 × 512)` + learned positional
   embedding `(512 × 512)`, summed → `(32, 512, 512)`. Embedding dropout
   (0.1) applied.
2. **7 × Transformer block**, each:
   - Pre-norm causal self-attention: `LayerNorm → Q,K,V projections (512×512
     each) → reshape to 8 heads × 64 dims → scaled dot-product attention
     with additive causal mask (`softmax(QKᵀ/√64 + M)`) → concat heads →
     output projection (512×512) → residual add.
   - Pre-norm feed-forward: `LayerNorm → Linear(512→2048) → GELU →
     Linear(2048→512)` → residual add.
3. **Final LayerNorm**, then **output head**: `Linear(512→8000)`, weight
   tied to the input token-embedding matrix → logits `(32, 512, 8000)`.

## 5. Step 4 — Loss

Cross-entropy between logits at position *t* and the true token at
position *t+1*, averaged over all `32 × 512` positions in the batch —
the causal language-modeling objective.

## 6. Step 5 — Optimizer step (`common/train/optim.py`)

1. Gradients accumulate over 4 micro-batches (`grad_accum_steps`) before
   each optimizer step — so one step actually sees `4 × 32 = 128` windows
   = 65,536 tokens.
2. `scaler.unscale_` → `clip_grad_norm_(max_norm=1.0)` → `AdamW.step()`
   (β=(0.9, 0.95), weight decay 0.1 applied only to ≥2D weight matrices,
   not biases/LayerNorm) → LR scheduler step (linear warmup over 200
   steps, then cosine decay to 10% of peak 3e-4) → zero gradients.
3. Mixed precision (AMP) active on GPU, no-ops on CPU.

## 7. Step 6 — Checkpoint (`common/train/checkpoint.py`)

Every 200 steps: save model weights + optimizer state + LR-scheduler state
+ current step + tokens seen to `latest.pt` (overwritten, not
accumulated), plus `best.pt` whenever validation loss improves. Re-running
the same training command auto-resumes from `latest.pt` — no manual
bookkeeping needed, since random-sampling (§3) means "resume from step N"
is fully defined by the step count alone (no shuffle-order state to
restore).

## 8. Full run: the numbers

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Total steps | 8,000 | 8,000 |
| Tokens/step | 65,536 | 65,536 |
| **Total tokens processed** | **524,288,000** | **524,288,000** |
| Corpus coverage (tokens processed ÷ corpus size) | ≈0.78× | not computed (train-token count not captured this session) |
| Restarts (Colab disconnects) | 1 | **4** |
| Wall-clock training time | 2,695.8s (~45 min, T4 GPU) | **15,223.5s (≈4.23 hours, T4 GPU, across 5 segments)** |
| Final train loss | 3.5647 | 3.7242 |
| Final val loss | 3.4717 | 3.6646 |
| Final val perplexity | 32.19 | 39.04 |

Both `train_log.csv` files are exact (Hindi: `hindi/model/train_log.csv`;
Nepali provided directly). Nepali's log shows restarts at steps
2,820/3,820/4,820/6,820, each correctly resuming from the prior 200-step
checkpoint boundary (2,800/3,800/4,800/6,800) — direct evidence the
checkpoint/resume mechanism (§7) worked correctly on every occurrence, not
just the one RAM-pressure incident originally known about.

---

## 9. Evaluation results

Evaluation runs the model over the **test** split only (never touched
during training) — see `common/eval/intrinsic.py`, `generate.py`,
`attention.py` for the exact procedures.

### 9.1 Intrinsic metrics (deterministic full pass over test set)

| Metric | Hindi | Nepali |
|---|---|---|
| Test tokens | 6,630,400 | 7,754,752 |
| Cross-entropy (nats/token) | 3.4646 | 3.6971 |
| **Perplexity** | **31.96** | **40.33** |
| **Bits-per-byte** | **0.5259** | **0.4810** |

Nepali has higher token-level perplexity (expected — it's the
lower-resource language) but *lower* bits-per-byte than Hindi. This isn't
a contradiction: Nepali's tokenizer has higher fertility (~4.2
chars/token vs. Hindi's ~3.7, per the Phase 1 tokenizer reports), so each
Nepali token already covers more raw bytes — a higher per-token cost still
divides out to fewer bits per byte. This is exactly why BPB, not
perplexity, is the metric used for the cross-tokenizer H-vs-L comparison.

### 9.2 Generation quality (greedy + temperature 0.5/1.0/1.5)

**Note**: ROUGE-L was initially broken for both languages — `rouge_score`'s
default tokenizer strips all non-ASCII characters, silently zeroing every
Devanagari score (verified: identical text scored 0.0). Fixed in
`common/eval/metrics.py` with a Unicode-aware tokenizer; numbers below are
from the corrected re-run. Greedy is deterministic and reproduced
identically; the temperature conditions used fresh random samples (no
fixed seed), so their exact values differ slightly from any numbers seen
in earlier, pre-fix runs — expected sampling variance.

**Hindi** (80 test examples, 32-token prefix → 64 generated tokens):

| Condition | BLEU-4 | chrF++ | ROUGE-L | Repetition (4-gram) | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|---|
| Greedy | 1.76 | 13.88 | 0.255 | 0.603 | 0.132 | 0.270 |
| Temp 0.5 | 1.49 | 14.91 | 0.247 | 0.222 | 0.223 | 0.536 |
| Temp 1.0 | 0.86 | 16.52 | 0.252 | 0.003 | 0.463 | 0.915 |
| Temp 1.5 | 0.12 | 14.26 | 0.233 | 0.0 | 0.676 | 0.990 |

**Nepali** (83 test examples, same setup):

| Condition | BLEU-4 | chrF++ | ROUGE-L | Repetition (4-gram) | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|---|
| Greedy | 1.75 | 14.60 | 0.212 | 0.557 | 0.176 | 0.277 |
| Temp 0.5 | 1.91 | 16.90 | 0.242 | 0.169 | 0.336 | 0.605 |
| Temp 1.0 | 0.84 | 18.30 | 0.233 | 0.002 | 0.598 | 0.937 |
| Temp 1.5 | 0.10 | 15.97 | 0.228 | 0.0 | 0.754 | 0.991 |

Both languages show the identical qualitative pattern: greedy decoding
degenerates into verbatim phrase loops (repetition 60.3%/55.7%); repetition
falls and diversity rises monotonically with temperature; but — a genuine
finding, not the initially-expected result — **BLEU-4 and ROUGE-L are
highest for greedy and *decline* with temperature in both languages**,
because greedy's safe, generic phrasing happens to share more surface
words with some plausible text even while looping, and word-overlap
metrics can't tell that apart from genuine on-topic content. chrF++ is the
one reference-based metric that behaves sensibly, peaking at temp 1.0 for
both languages (16.52 Hindi, 18.30 Nepali — Nepali's overall highest value
of any condition). See `report/phase2/report.md` §3.2/§4.2 for the full
discussion of why this makes BLEU/ROUGE-L actively misleading here, not
merely "uninformative."

### 9.3 Attention analysis (200 test sentences, 128-token window)

| | Hindi | Nepali |
|---|---|---|
| Local/positional heads | Clearest in layers 3-4 (entropy as low as 0.75 nats, distance as low as 1.7-7.6 tokens) | Even more extreme: layer 4 head 2 entropy = **0.19** nats (near-deterministic attention) |
| Late-layer (layer 6) pattern | Attention-sink on heads 0-1 (heavy weight on token 0 from all queries); heads 2-3 more genuinely local/diagonal | Attention-sink visible on **all 4** plotted heads, more pronounced than Hindi |
| Mean distance, layer 6 | 15.6-39.1 tokens | 24.6-37.4 tokens (similar range, more uniformly high) |

Both models learn the same qualitative structure (specialized local heads
in the middle layers, a first-token attention sink late) despite training
on unrelated data with independent tokenizers/vocabularies — evidence this
is a property of the architecture/objective rather than an artifact of one
particular corpus.

---

## 10. Status

Everything is complete: architecture, pretraining (both languages, exact
logs), generation quality (both languages, ROUGE-L fix applied), attention
analysis (both languages), and the full H-vs-L comparison — see
`report/phase2/report.md` §5 for the full resource-level writeup.
