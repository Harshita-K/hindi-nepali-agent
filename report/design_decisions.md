# Design Decisions — Phase 1 & Phase 2

A single reference for every non-obvious choice made across data collection,
tokenizer construction, architecture, and pretraining — what was chosen,
what the alternatives were, and why the alternative was passed over. Full
narrative context and results live in `report/phase1/report.md` and
`report/phase2/report.md`; this document exists to make the *decisions*
themselves easy to find and defend on their own, independent of the results
that came out of them.

---

## Phase 1 — Data & Tokenizer

### Why Hindi and Nepali

**Decision**: Hindi as Model H (higher-resource), Nepali as Model L
(lower-resource, from the assignment's fixed allowed list: Assamese,
Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali, Sindhi).

**Why**: Hindi has, by a wide margin, the largest publicly available text
volume of any non-English Indian language — large web-crawl splits, and a
commercially mature, sitemap-indexed news ecosystem. That made the 500M-token
target and 20% manual-collection floor achievable without extraordinary
effort, which was deliberate: Model H is meant to be the *control*, so any
modeling problem surfaced in Phase 2/3 can be attributed to architecture or
training choices, not a starved corpus.

Nepali was picked from the allowed low-resource list because it sits in a
genuinely useful middle ground: present in every multilingual corpus used
(FineWeb2, IndicCorp, Sangraha, Wikipedia) but only thinly, and with an
active-but-fragmented news industry where several major outlets explicitly
block AI crawlers. That makes it a real stress-test of the manual-collection
requirement — scarcer than Hindi, but not so scarce that manual collection
was a dead end (which some other options on the allowed list would have
been, given how little they're represented in these corpora at all).

### Why these four downloaded sources

**Decision**: FineWeb2, IndicCorp, Sangraha, and official Wikipedia dumps —
same four for both languages.

**Why**: Using three overlapping-but-distinct web-scale corpora rather than
just one deliberately reduces the risk that the downloaded portion ends up
dominated by a single crawler's particular biases or gaps. FineWeb2 is
included for its quality-filtering pipeline (avoids the boilerplate/spam
problems raw Common Crawl has); IndicCorp for being purpose-built for Indic
NLP (cleaner in-domain text than a generic crawl); Sangraha for drawing from
different underlying sources than the other two (real diversity, not
redundant coverage); Wikipedia for a fundamentally different register
(encyclopedic/structured vs. news/web), balancing the corpus rather than
making it purely news-flavored.

### Why a 20% manual-collection floor, and why it wasn't strictly enforced

**Decision**: target ≥20% of final tokens from manual collection (own
scraping/OCR), but build with `enforce_ratio=False` for both languages —
prioritizing the ~500M-token *total* target over strictly guaranteeing the
ratio.

**Why**: The assignment requires manual collection specifically so the
corpus isn't just "whatever a web crawl happened to have," and 20% is a
substantial floor without being so high that reaching 500M tokens becomes
infeasible for the harder language. Not enforcing it as a hard cap (the code
supports `enforce_ratio=True`, which would have capped downloaded tokens to
whatever ratio the current manual pool supported, producing a *smaller*
corpus if needed) was a considered trade-off: manual collection was still
actively growing via ongoing scraping at build time, so trading a small,
honestly-reported shortfall (Nepali landed at 19.89%, not enforced-19.89%)
for hitting the token target was judged better than a smaller corpus that
technically clears the floor.

### Cleaning pipeline choices

**Unicode normalization (NFC) + `ftfy` mojibake repair, before anything
else.** Scraped HTML and some downloaded sources carry inconsistent
encodings; NFC also ensures visually-identical characters hash identically
during dedup, which matters directly for the next step.

**Script filter at 60% Devanagari fraction (line-level, not just
document-level).** Chosen as a practical middle threshold: strict enough to
drop non-Devanagari boilerplate (English ads, embedded code, nav text), but
not so strict that legitimate mixed-script sentences — common and natural in
Hindi/Nepali news, e.g. an English proper noun inside a Devanagari sentence
— get discarded. Applying it per-line rather than only per-document means a
mostly-clean document doesn't get thrown out over a few contaminated lines.

**Minimum length filter at 20 characters post-cleaning.** Removes
near-empty scraps (a lone headline fragment) that survive script filtering
but carry no real content.

**Exact-dedup via SHA-256 content hash, manual-priority tie-break.** Docs
are sorted so every manual document is processed before any downloaded
document, *before* the dedup pass runs; since dedup keeps the first copy of
each hash it sees, this deterministically attributes any manual/downloaded
exact-duplicate (a real case — our own scraper and a web crawl both picking
up the same article) to manual, rather than leaving the outcome to arbitrary
file-load order.

**Doc-level train/val/test splits, not sentence- or paragraph-level.**
Guarantees no partial-document leakage between train and eval sets — a
document's sentences can't end up split across two splits.

### Why SentencePiece BPE, and why 8,000 vocabulary

**Decision**: an independent SentencePiece BPE tokenizer per language,
8,000-token vocabulary, `character_coverage=0.9995`.

**Why SentencePiece over a raw/library BPE implementation**: SentencePiece
doesn't require pre-tokenization by whitespace, which matters specifically
for Devanagari text, where word boundaries and whitespace don't always align
as cleanly as they do in English (compound words, particle attachment).
Building a whitespace-pretokenized BPE by hand would have to reinvent
exactly this handling.

**Why BPE over WordPiece or Unigram**: BPE gives a good balance of
vocabulary coverage and sequence-length efficiency for morphologically rich
Indic scripts — its frequency-driven merge process naturally keeps common
whole words as single pieces (`▁के`, `▁छ`) while still being able to
decompose rarer or morphologically complex words, without needing a
likelihood-based vocabulary search (Unigram) or WordPiece's likelihood-ratio
merge criterion, which add complexity without a clear benefit for this
project's scale.

**Why 8,000 vocabulary, not the originally planned 24K/32K**: this was a
deliberate downward revision once the ~25M-parameter target for Phase 2 was
locked in. The embedding table (`vocab_size × d_model`) is a much larger
fraction of total parameters at 25M scale than it would be for a larger
model, so a smaller vocabulary buys proportionally more of the parameter
budget back for the actual Transformer stack. The visible cost is more
aggressive subword splitting on longer/rarer words (e.g. मुज़फ्फरनगर → 4
pieces instead of 1-2), but both languages' unknown-token rates stayed well
under 0.3% even at this smaller size — the vocabulary still covers the
corpus well, so the trade was judged worth it. `character_coverage=0.9995`
was set high (rather than SentencePiece's more common 0.9995-for-Latin-script
default assumption) specifically because Devanagari's glyph inventory
(including rarer conjuncts and diacritics) is larger than what a lower
coverage setting would reliably capture.

---

## Phase 2 — Architecture & Pretraining

### Why build the Transformer from raw primitives

**Decision**: `nn.Linear` / `nn.Embedding` / `nn.LayerNorm` / `nn.Dropout`
only — no `nn.Transformer*`, no HuggingFace model classes, no pre-built
attention block.

**Why**: This is the assignment's hard constraint, but the reasoning behind
the constraint is worth stating: it forces every tensor shape and design
choice in the forward pass to be something actually built and understood,
not inherited from a library default — which is exactly what gets probed at
evaluation/viva. Shared *code* (one `common/model/transformer.py`) is still
used for both languages, since the constraint is about not using pre-built
neural network modules, not about writing the forward pass twice — Model H
and Model L each get their own weights, tokenizer, and config from that
shared code.

### Positional embeddings: learned absolute, not sinusoidal or RoPE

**Decision**: a learned absolute positional embedding table
(`max_seq_len × d_model`), summed with token embeddings.

**Why**: At a ~25M-parameter budget with a fixed, known maximum training
sequence length (512 tokens), a learned table is simpler to implement,
verify, and reason about than sinusoidal or relative/RoPE schemes, and costs
very little (≈262K parameters, about 1% of the model). The trade-off,
accepted deliberately: it hard-caps the model at `max_seq_len` tokens — no
extrapolation to longer sequences the way sinusoidal/RoPE allow. Since
neither Model H nor Model L needs to process sequences longer than 512
tokens anywhere in this project, that limitation doesn't cost anything in
practice here.

### Multi-head attention: four separate projections, not fused QKV

**Decision**: independent `nn.Linear` layers for `W_Q`, `W_K`, `W_V`, `W_O`,
rather than one fused QKV weight matrix.

**Why**: Prioritizes readability/explainability (each projection's role is
visually distinct in the code) over the minor efficiency gain of fusing
them into one matmul — a reasonable trade at this model scale, and again
directly relevant to being able to explain every tensor operation.

### Why 8 heads / `d_k=64`

**Decision**: `n_heads=8`, giving `d_k = d_model / n_heads = 512/8 = 64`.

**Why**: 64 is the same per-head dimension GPT-2 uses — a well-tested size
chosen as a sensible default rather than something exhaustively tuned,
since architecture search itself was out of scope given the time available.

### Pre-norm, not post-norm

**Decision**: LayerNorm applied *before* each sublayer (attention, FFN),
not after.

**Why**: Pre-norm keeps the residual stream unnormalized end-to-end, which
is empirically much more stable for deeper stacks — post-norm GPT-style
models are known to be far more sensitive to learning-rate/warmup schedule
and easier to destabilize. On a tight compute/time budget where a failed
run is expensive to diagnose and re-launch on Colab, that extra stability
margin was judged worth taking over post-norm's (slightly) more common use
in the original Transformer paper.

### Activation function: GELU

**Decision**: GELU in the feed-forward network (`Linear → GELU → Linear`),
not ReLU or a gated variant like SwiGLU.

**Why**: GELU is the standard choice for GPT-style pretraining and gives
smoother gradients than ReLU (no hard zero cutoff), which is generally
preferred for Transformer training stability. SwiGLU (used in some more
recent LLMs) requires a different FFN parameter shape (an extra gating
projection) for a similar total parameter count, adding implementation
complexity without a clear necessity at this ~25M-parameter, ~500M-token
scale — the project's constraints explicitly frame this as *not* an
architecture bake-off, so a well-precedented standard choice was preferred
over chasing a marginal, harder-to-verify gain.

### FFN expansion ratio: 4x

**Decision**: FFN inner dimension = `4 × d_model = 2048`.

**Why**: The standard GPT expansion ratio — gives the feed-forward sublayer
meaningfully more representational capacity than the attention sublayer
alone, which is the well-established reason Transformers use an expansion
> 1x here rather than keeping the FFN at `d_model` width throughout.

### Tied input/output embeddings

**Decision**: `lm_head.weight = token_emb.weight`.

**Why**: Saves `vocab_size × d_model` = 4,096,000 parameters — about 13.4%
of what the model would otherwise be (30.5M untied vs. 26.4M tied) — and
ties the input and output token representations, standard practice for LMs
at this scale. Given the ~25M-parameter target, that's a meaningful chunk
of budget freed up for the actual Transformer stack rather than a
second full-size embedding-shaped matrix.

### Depth vs. width: 7 layers, `d_model=512`

**Decision**: narrower-but-deeper (`d_model=512`, 7 layers) over a
wider-shallower alternative at the same total parameter count.

**Why**: With an 8,000-token vocabulary, the embedding table is a
near-fixed ~4.1-4.4M-parameter cost (with positional embeddings) regardless
of depth/width choice, leaving roughly 22M for the Transformer stack itself
— two knobs available: go wider with fewer layers, or narrower with more.
Depth generally helps a decoder-only LM compose hierarchical/longer-range
structure more effectively per parameter than raw width does at this small
scale, and 7 layers still keeps per-step compute modest enough to train
~500M tokens within a realistic Colab session length. `d_model=512` and
`n_heads=8` were kept fixed as reasonable, well-precedented defaults rather
than exhaustively searched.

### Optimizer: AdamW, decoupled weight decay on ≥2D parameters only

**Decision**: AdamW, `betas=(0.9, 0.95)`, weight decay 0.1 applied only to
parameters with ≥2 dimensions (weight matrices) — biases and LayerNorm
gain/bias are excluded.

**Why**: AdamW with this beta configuration is the standard choice for
Transformer LM pretraining, well-precedented across GPT-style models at
this scale. Excluding 1D parameters from weight decay is standard practice
too: decaying a scale/shift parameter (a LayerNorm gain, or any bias) toward
zero has no principled justification the way decaying a weight matrix's
magnitude does, and empirically can hurt normalization behavior.

### LR schedule: linear warmup + cosine decay

**Decision**: 200-step linear warmup, then cosine decay to 10% of peak LR
(`min_lr_ratio=0.1`); peak LR `3e-4`.

**Why**: Warmup avoids instability while weights are still near their
random initialization and Adam's second-moment estimates are still
unreliable — a large LR applied immediately risks an early divergence that
would waste an entire Colab session. Cosine decay anneals smoothly rather
than dropping the LR abruptly at a fixed schedule boundary. `3e-4` peak LR
is a common, well-tested default for small-scale Transformer LM
pretraining, not searched from scratch.

### Gradient clipping at 1.0

**Decision**: clip gradient norm to a maximum of 1.0 every step.

**Why**: Cheap insurance against occasional large gradient spikes
destabilizing training — a small, standard safeguard with negligible cost
that specifically protects against wasting a long unattended Colab run to
a rare bad batch.

### Batch size, gradient accumulation, and block size → 65,536 tokens/step

**Decision**: `batch_size=32 × grad_accum_steps=4 × block_size=512 =
65,536` tokens per optimizer step.

**Why**: This combination was chosen to fit comfortably within a single
Colab GPU's memory (T4) while keeping the effective batch size (in tokens)
large enough for stable gradient estimates — grad accumulation lets the
*effective* batch exceed what fits in memory at once, without changing the
per-step token budget's relationship to the target token count.

### Training length: 8,000 steps (~524M tokens)

**Decision**: `max_steps=8000` at 65,536 tokens/step ≈ 524,288,000 tokens.

**Why**: Directly hits the assignment's ~500M-token-per-model pretraining
target, with the small overshoot simply being a clean consequence of using
round step/batch numbers rather than solving for an exact token count.

### Mixed precision (AMP), CUDA-only

**Decision**: enabled when running on CUDA, automatically disabled on CPU.

**Why**: Speeds up training and reduces memory use on GPU with negligible
effect on convergence at this scale — a standard, low-risk optimization
that matters concretely for fitting the ~500M-token run inside a realistic
Colab session length.

### Checkpointing: every 200 steps, `latest.pt` overwritten + `best.pt` kept

**Decision**: save `{model weights, optimizer state, scheduler state, step,
tokens_seen, config}` every 200 steps to `latest.pt` (overwritten each
time, not accumulated), plus a separate `best.pt` whenever validation loss
improves; re-running the same command auto-resumes from `latest.pt`.

**Why**: Mandatory per the assignment specifically because Colab sessions
terminate unexpectedly — this was not a hypothetical risk: it happened
during Hindi's actual run (once) and Nepali's (four times, each correctly
recovered). Overwriting `latest.pt` rather than accumulating checkpoints
bounds disk usage regardless of run length; saving `best.pt` separately
means the best model is preserved even if training later degrades or is
cut short past its peak.

### Random-sampling-with-replacement, not epoch-based shuffling

**Decision**: pretraining batches are random fixed-length windows drawn
with replacement from the tokenized corpus, not a shuffled-epoch traversal.

**Why**: The assignment specifies a token *budget*, not an epoch count,
so there's no natural "epoch" to shuffle over in the first place. This
also makes checkpoint-resume trivial — resuming is fully defined by the
step counter alone, with no shuffle-order or corpus-position state that
would otherwise need to be saved and restored — and keeps memory bounded
via `numpy.memmap` regardless of corpus size (Hindi's train split alone is
~675M tokens before subsampling).

### Evaluation metric choices

**Perplexity *and* bits-per-byte, not perplexity alone.** Perplexity is
measured in tokens, and Hindi/Nepali have independently-trained tokenizers
with different fertility (chars/token) — so a perplexity number alone isn't
directly comparable between the two models. BPB normalizes by raw UTF-8
bytes instead, making it the metric actually suited to the required H-vs-L
comparison. (This mattered concretely: Nepali came out *worse* on
perplexity but *better* on BPB — exactly the divergence BPB exists to
catch, and evidence perplexity alone would have been misleading here.)

**Greedy + three temperatures (0.5, 1.0, 1.5) for generation, not just
one setting.** Greedy alone would only show the degeneration failure mode;
temperature sampling shows how repetition/diversity trade off as randomness
increases, and comparing metrics *across* conditions is what let the
project identify a real methodological finding — BLEU/ROUGE-L reward
greedy's repetitive output, while only chrF++ correctly favors temp=1.0.

**BLEU-4, chrF++, and ROUGE-L together, not just one.** Reported together
specifically so their disagreement could be examined rather than assumed
away: BLEU-4 is the most brittle for Hindi/Nepali (whole-word n-gram
matching penalized hard by rich inflectional morphology), chrF++ is
character-level and more morphology-tolerant, and ROUGE-L (LCS-based) sits
between them but shares BLEU's word-level fragility. Reporting only one
would have hidden this trade-off.

**Repetition-rate and Distinct-1/2 alongside the reference-based
metrics.** These reference-*free* diagnostics turned out to be more
trustworthy than BLEU/ROUGE-L for judging actual generation quality here,
precisely because they don't have the single-reference-comparison problem
that makes BLEU/ROUGE-L reward repetitive "safe" text — a finding the
project could only make by having both kinds of metric to compare.

**Attention entropy and mean attention distance, on a 128-token window.**
Entropy captures how peaked vs. diffuse a head's attention is (a low-entropy
head is behaving like a local/positional head); mean distance captures how
far a head reaches on average — together they distinguish a true
long-range content-based head from an attention-sink head that technically
has a large mean distance but for a structurally different reason (a fixed
early-token anchor, not broad scanning). 128 tokens (not the full 512token
context) was chosen to keep per-sentence evaluation fast across 200
sentences while still being long enough not to artificially cap the
distance metric.
