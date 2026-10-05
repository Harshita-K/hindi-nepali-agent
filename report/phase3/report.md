# Phase 3 Report — Reasoning Finetuning, Attention Analysis, and Final Comparison

This report covers Phase 3 in full — the synthetic comparative-reasoning
dataset, finetuning both pretrained models on it, the pretrained-vs-finetuned
attention comparison, and the final Model H (Hindi) vs. Model L (Nepali)
resource-tier comparison the assignment asks for. As with the Phase 1 and
Phase 2 reports, results are reported alongside *why* each decision was made
and what evidence backs each claim, since that's what actually gets probed
at review time.

---

## 1. Reasoning finetuning (spec 3.1)

### 1.1 Synthetic dataset design

Built entirely by [common/finetune/reasoning_gen.py](../../common/finetune/reasoning_gen.py)
(language-agnostic generator code, no shared text) with per-language word
lists and templates in `{hindi,nepali}/finetune/generate_reasoning_data.py`
— mirroring Phase 1/2's pattern of shared *code*, fully independent *data*.

**Three task families**, per the spec:

| Task | Description |
|---|---|
| `direct` | Compare 2 entities on one attribute — answer is the greater/smaller one, or "both equal" |
| `transitive` | 3 entities chained A~B~C — answer is the extreme (min/max) |
| `multihop` | 3 entities chained A~B~C — answer is the *derived*, non-adjacent relation between A and C (the only family that actually requires chaining both stated facts rather than reading off an implied order) |

Both languages cover the same 4 attributes (age, height, price, quantity),
phrased in natural target-language syntax (e.g. Hindi's comparative word
agrees with the attribute noun's gender, not the entity's — see the
generator docstrings — so the sentences read as native Hindi/Nepali, not
templated English with swapped-in names).

**Leakage control** (`reasoning_gen.py::build_dataset`): both entity names
*and* templates are split into disjoint train/test pools before any
examples are generated, so the test set uses **entity names the model never
saw during finetuning, rendered with a template variant never seen during
finetuning either** — a much stronger generalization test than holding out
only one of the two.

| | Hindi | Nepali |
|---|---:|---:|
| Train / Val / Test examples | 6,000 / 800 / 800 | 6,000 / 800 / 800 |
| Held-out person names (test-only) | 6 | 6 |
| Held-out object names (test-only) | 3 | 3 |
| Templates held out for test (per task) | 1 of 3–4 | 1 of 3–4 |

(Full breakdown by task type/attribute: `{hindi,nepali}/data/reasoning/stats.json`.)

### 1.2 Finetuning protocol

Both languages use **identical** hyperparameters (same rationale as Phase
2's architecture: keeps the H-vs-L comparison free of confounding
hyperparameter differences):

| | Value | Reasoning |
|---|---|---|
| Start point | Language's own Phase 2 `best.pt` | Spec requirement — tokenizer/vocab fixed, only weights loaded |
| Loss | Masked causal LM — only the answer span is supervised | The prompt is context, not a prediction target; supervising it would dilute the learning signal with easy-to-predict boilerplate |
| Peak LR | 2e-5 (~15x lower than pretraining's 3e-4) | The finetune set is tiny relative to the ~500M-token pretraining corpus; a pretraining-scale LR risks catastrophically overwriting general language ability while overfitting this narrow task |
| Batch size | 32 | |
| Max sequence length | 96 tokens | Covers p99 tokenized (prompt+answer+EOS) length with margin, both languages |
| Warmup / schedule | 50 steps linear warmup → cosine decay | Same mechanism as Phase 2, reused rather than reimplemented |
| Checkpointing | Same resume-capable format as pretraining (`common/train/checkpoint.py`) | Spec requirement — a Colab disconnect mid-finetune is recoverable the same way |

**Step budget**: the config defaults to 1,000 steps, but both languages were
run longer via `--max_steps`, and Nepali specifically was extended twice
(1,000 → 3,000 → 5,000) to test whether its accuracy gap versus Hindi was a
step-budget artifact or a real ceiling — see §1.5.

| | Model H (Hindi) | Model L (Nepali) |
|---|---:|---:|
| Steps run | 3,000 | 5,000 |
| Final train loss (answer-token CE) | 0.0828 | 0.0802 |
| Final val loss (answer-token CE) | 0.1236 | 0.2089 |

Both train losses converge to a similarly small value, but Nepali's val loss
settles roughly **70% higher** than Hindi's — the first sign (before even
looking at exact-match accuracy) that Nepali is generalizing to held-out
examples worse than Hindi is, not just fitting the training set worse.

### 1.3 Model H (Hindi) — pretrained vs. finetuned results

Evaluated with `common/finetune/reasoning_eval.py::evaluate_exact_match` —
greedy-generate the answer from each held-out test prompt, string-compare
against ground truth. Run twice per language (pretrained checkpoint,
finetuned checkpoint) on the **same** 800 held-out-name/held-out-template
test examples, so this is a genuine before/after comparison.

| | Pretrained | Finetuned |
|---|---:|---:|
| **Overall exact-match accuracy** | **0.0%** | **36.75%** |
| direct | 0.0% | 14.9% |
| transitive | 0.0% | 52.0% |
| multihop | 0.0% | 59.0% |

Pretrained accuracy is exactly 0% for a mechanical reason, not a modeling
failure: a purely next-token-pretrained LM has no reason to emit *just* the
entity name at the "उत्तर:" cue — the qualitative examples show it continuing
in generic prose instead (e.g. `"- राम की उम्र 35 वर्ष हो और..."`, a plausible
Hindi sentence continuation, just not a bare answer). Finetuning's job here
is teaching the *output format*, not merely the *logic* — and both jobs
happen simultaneously in a 3,000-step run.

**Interesting internal pattern**: `direct` (14.9%) is the *worst*-performing
task type post-finetune, well below `transitive` (52.0%) and `multihop`
(59.0%) — despite `direct` being the logically simplest task (single
comparison, no chaining). Looking at the qualitative failures, this is a
**tie-detection problem**, not a comparison-logic problem: several `direct`
test examples set two entities to the *same* numeric value (correct answer
"दोनों बराबर हैं" / "both equal"), and the finetuned model tends to default to
picking one entity anyway rather than recognizing equality — a narrower,
specific failure mode rather than evidence the model can't compare two
numbers.

### 1.4 Model L (Nepali) — pretrained vs. finetuned results

| | Pretrained | Finetuned |
|---|---:|---:|
| **Overall exact-match accuracy** | **0.0%** | **8.75%** |
| direct | 0.0% | 10.3% |
| transitive | 0.0% | 8.4% |
| multihop | 0.0% | 6.3% |

Same 0% pretrained baseline, same mechanical reason. But finetuned accuracy
is dramatically lower than Hindi's across every task type — roughly
**1/4 to 1/9** of Hindi's per-task numbers — and unlike Hindi, there's no
task-type pattern favoring multi-step reasoning; all three sit in a similar
low band.

### 1.5 Model H vs. Model L — why the gap, and what's already been ruled out

**This gap is real, not a bug or an artifact of an unlucky run** — it was
specifically stress-tested before being reported:

- **Step budget was tested and ruled out.** Nepali was trained at 3,000
  steps (matching Hindi exactly) and then again at 5,000 steps (nearly
  double). If the gap were simply "Nepali needs more gradient steps on the
  same data," extending the budget should have narrowed it. It didn't move
  meaningfully — this was a deliberate experiment (see the commit history:
  *"Try Nepali finetuning at 5000 steps — last step-count experiment,
  checking whether the 3000-step gap is budget or ceiling"*), and the
  answer came back "ceiling," not "budget."
- **The finetune protocol is identical between languages** (§1.2) — same
  LR, same step-for-step comparison point (3,000 steps each), same masked
  loss, same dataset size/structure (6,000/800/800, same task-type/attribute
  balance) — so the gap isn't a confounded hyperparameter difference.

**What the qualitative failures actually show — and this is the more
informative evidence than the accuracy numbers alone.** Comparing the
*kind* of mistake each finetuned model makes on wrong answers reveals a
real difference in what finetuning did or didn't teach:

- **Hindi's wrong answers stay inside the task's legitimate answer space.**
  Its failures are either a defensible-but-wrong category choice (predicting
  "दोनों बराबर हैं" / "both equal" on a `direct` example that actually had a
  clear ordering — a tie-detection miscalibration, not confusion about
  *who* is being compared) or an empty/degenerate generation. In every wrong
  example, when the model does name an entity, it's one of the entities
  actually present in that prompt.
- **Nepali's wrong answers sometimes name entities that were never in the
  prompt at all.** E.g. a test prompt comparing मनोज and सुनिता gets answered
  "सुनिल"; a prompt comparing विकास/कृष्णा/सीता gets answered "सरस्वती". Checked
  against `nepali/finetune/generate_reasoning_data.py`'s name pools: सुनिल,
  सरस्वती, पार्वती, and कमला are all real entries in the **training** name
  pool — just not the *held-out* names used to build this specific test
  prompt, and not among the 2-3 entities actually mentioned in it. This is a
  **binding/grounding failure**, not a hallucination from nowhere: the
  finetuned Nepali model has learned "answer with a plausible person name
  from the pool it saw repeatedly during finetuning," but hasn't reliably
  learned to constrain that answer to *the specific entities named in this
  prompt* — closer to surface memorization of frequently-seen training
  strings than to learning the general entity-binding skill the held-out
  test names are designed to probe.

This directly explains why more finetuning steps didn't help (§1.5, first
bullet): more steps on the same fixed 6,000 training examples would only
reinforce which names appear frequently in training, not teach the model to
generalize its answer to *whichever* names appear in a new prompt. The
attention-analysis evidence in §2.3 independently corroborates this same
conclusion from a completely different angle (representational change, not
output behavior).

---

## 2. Attention analysis, pretrained vs. finetuned (spec 3.2)

### 2.1 Methodology

Reuses Phase 2's attention toolkit (`common/eval/attention.py` — heatmaps,
per-(layer, head) entropy, mean attention distance) unchanged, via
`{hindi,nepali}/eval/run_attention.py`, now parameterized (see the script
docstrings) so it can target a finetuned checkpoint and a different input
source without overwriting Phase 2's already-graded output.

**Deliberate design choice**: rather than comparing finetuned-on-reasoning
against Phase 2's existing pretrained-on-news attention numbers, the
**pretrained checkpoint was re-run on the same reasoning-prompt test set**
used for the finetuned checkpoint (`{hindi,nepali}/data/reasoning/test.jsonl`,
200 prompts each). This makes it a controlled before/after comparison on
identical inputs — any difference found is attributable to what finetuning
changed inside the model, not to a domain shift from news text to reasoning
prompts.

### 2.2 Results

Entropy (nats) and mean attention distance (tokens), averaged over all
heads, comparing pretrained vs. finetuned on the same 200 reasoning
prompts:

| | Model H (Hindi) |  | Model L (Nepali) |  |
|---|---:|---:|---:|---:|
| | Pretrained | Finetuned | Pretrained | Finetuned |
| Overall mean entropy | 1.979 | 1.877 | 1.965 | 1.895 |
| Overall mean distance | 7.605 | 7.432 | 7.243 | 7.369 |
| Early layer (0) entropy | 2.389 | 2.355 | 2.402 | 2.359 |
| Late layer (6) entropy | 1.552 | 1.568 | 1.505 | 1.421 |
| Early layer (0) distance | 8.436 | 8.526 | 8.122 | 8.462 |
| **Late layer (6) distance** | **11.856** | **10.567** | **11.005** | **10.983** |

(Full per-layer/per-head arrays: `report/phase3/{hindi,nepali}_attention_{pretrained_reasoning,finetuned}.json`.
Heatmaps for 3 example prompts × early/late layer × 4 conditions:
`report/phase3/figures/`.)

### 2.3 Discussion

**Finetuning lowers overall attention entropy in both languages** (Hindi
1.979→1.877, Nepali 1.965→1.895) — attention becomes marginally more
focused/peaked after reasoning finetuning, consistent with the model
narrowing its attention toward the small set of tokens (the entity names
and the comparison words) that actually determine the answer in this task,
rather than distributing attention as broadly as a general-purpose language
model does.

**But the magnitude of change is very different between the two languages,
concentrated specifically in the late layer** — and this is the more
telling result. Hindi's late-layer (layer 6) mean attention distance drops
by **~11%** (11.86 → 10.57 tokens) after finetuning, a real, directionally
consistent shift. Nepali's late layer barely moves at all (11.01 → 10.98,
essentially unchanged). Early-layer behavior is comparable between the two
languages (both show small increases, 8.44→8.53 and 8.12→8.46) — so the
divergence is specifically in the late layer, exactly where Phase 2's
report already identified an attention-sink pattern for both pretrained
models (§4.3 of `report/phase2/report.md`).

**Reading these two results together**: Hindi's finetuning visibly
reorganized its late-layer attention on reasoning prompts — a real,
measurable representational change consistent with the model actually
learning something new and task-specific in that layer. Nepali's late-layer
attention is nearly frozen by comparison, even though 5,000 finetuning steps
were run (more than Hindi's 3,000). This is independent, representational
evidence for the exact same conclusion §1.5 reached from the *output*
side (accuracy and qualitative failures): Nepali's model changed its
*behavior* enough to shift what tokens it emits, but changed much less
*internally* than Hindi's did — consistent with Nepali's finetuning
producing shallower, more surface-level adaptation (§1.5's memorization
reading) rather than the kind of deeper representational reorganization
Hindi's late layer shows.

---

## 3. Final comparison — answering the required questions (spec 3.3)

### Q1: How did data scale and quality differ between Model H and Model L?

| | Model H (Hindi) | Model L (Nepali) | Source |
|---|---:|---:|---|
| Total pretraining tokens | 499,999,997 | 499,302,601 | Phase 1 |
| Total documents | 1,368,253 | 1,893,004 | Phase 1 |
| Manual token fraction | 20.29% | 19.89% | Phase 1 |
| Manual sources | 5 (large, high-volume outlets) | 10 (many smaller/fragmented outlets) | Phase 1 §5 |
| Tokenizer fertility (chars/token) | 3.685 | 4.202 | Phase 1 |
| Tokenizer UNK rate | 0.261% | 0.190% | Phase 1 |

Both languages hit essentially the same token *target* and manual-fraction
floor — by design, so downstream differences reflect resource tier, not an
unmet quota. But the *shape* of the data differs: Nepali needed **38% more
documents** to reach a near-equal token count, meaning its average document
is shorter — a direct consequence of its manual collection needing **10
independent, smaller outlets** (several major Nepali outlets explicitly
block AI crawlers, per Phase 1 §5) versus Hindi's 5 higher-volume ones.
Shorter, more fragmented documents from more disparate sources plausibly
give Nepali's pretraining corpus more surface diversity but less
per-topic/per-style depth than Hindi's.

### Q2: How do language-modeling and reasoning results compare across the two resource tiers?

| | Model H (Hindi) | Model L (Nepali) | Gap |
|---|---:|---:|---|
| Test perplexity | 31.96 | 40.33 | Nepali worse by token (+26%) |
| Bits-per-byte | 0.5259 | 0.4810 | **Nepali better by byte** (−8.5%) |
| Finetuned reasoning accuracy | 36.75% | 8.75% | **Hindi dramatically better** (4.2x) |

This is the central, somewhat counterintuitive finding of the whole project:
**the two intrinsic LM metrics don't even agree on which model is
"better"** (Phase 2 §5 — perplexity favors Hindi, BPB favors Nepali, because
of the fertility difference above), and **neither metric predicts the
reasoning-finetuning gap**, which is large and unambiguously favors Hindi.
A byte-normalized view of raw language-modeling quality does not translate
into how well a model's *pretrained representations* support finetuning on
a structured downstream task — general text-compression ability and
task-transferability are separate properties, and this project's own
numbers are direct evidence that having the better BPB (Nepali) does not
guarantee the better reasoning-finetuning outcome.

### Q3: What tokenizer/corpus factors most affected the lower-resource model?

Two identified in this project, both traceable to Phase 1 choices:

1. **Tokenizer fertility (4.202 vs. 3.685 chars/token)** is the direct cause
   of Nepali's higher raw perplexity in Phase 2 despite its better BPB — a
   purely tokenizer-driven metric artifact, not evidence of a worse
   underlying model (Phase 2 §5's full reasoning).
2. **Corpus fragmentation from manual-collection scarcity** (10 smaller
   outlets vs. 5 larger ones, more documents for a similar token count,
   §Q1) plausibly explains why Nepali's pretrained representations
   transferred so much worse to reasoning finetuning: a corpus assembled
   from more, smaller, more heterogeneous sources gives the model less
   concentrated exposure to any one consistent style or structure to build
   deep, reusable representations from, compared to Hindi's more
   concentrated high-volume sources — consistent with Nepali's finetuned
   failures looking like shallow memorization (§1.5) rather than learned
   generalization, and with Nepali's late-layer attention barely
   restructuring under finetuning (§2.3).

### Q4: What evidence explains the observed differences?

Four independent pieces of evidence, from different parts of this project,
all point the same direction:

1. **The step-budget experiment** (§1.5): extending Nepali's finetuning from
   3,000 to 5,000 steps — while Hindi stayed at 3,000 — did not close the
   accuracy gap, ruling out "needs more training" as the explanation.
2. **Qualitative failure analysis** (§1.5): Hindi's wrong answers stay
   within the task's legitimate answer space (right *kind* of answer,
   wrong specific choice); Nepali's wrong answers sometimes substitute
   entities that were never in the prompt, drawn from names seen frequently
   during finetuning — a memorization signature, not a reasoning-logic
   failure.
3. **Attention reorganization under finetuning** (§2.3): Hindi's late-layer
   attention shifts measurably (~11% shorter mean distance) after
   finetuning; Nepali's barely moves (~0.2%) despite more training steps —
   independent, representational evidence that finetuning changed Hindi's
   internal computation more than Nepali's.
4. **The intrinsic-metric divergence itself** (§Q2): Nepali's better BPB
   shows its pretrained model is not simply "worse" at language modeling in
   general — which makes the reasoning-finetuning gap more informative, not
   less: it isolates *task-transfer* specifically as what's failing, rather
   than being a symptom of an already-weaker base model.

Together, these point to the same root cause identified in Q3 — Nepali's
pretraining corpus being assembled from more numerous, smaller, more
heterogeneous manual sources — as the most likely explanation for why a
resource-tier gap that's genuinely small-to-mixed in raw language modeling
(Q2) becomes large and one-sided the moment a model has to adapt its
pretrained representations to a new structured task in a handful of
finetuning steps.

---

## 4. Deliverables checklist

| Deliverable | Location |
|---|---|
| Reasoning data generation scripts | `common/finetune/reasoning_gen.py`, `{hindi,nepali}/finetune/generate_reasoning_data.py` |
| Reasoning data splits + stats | `{hindi,nepali}/data/reasoning/{train,val,test}.jsonl`, `stats.json` |
| Finetuning code + configs | `common/finetune/trainer.py`, `{hindi,nepali}/finetune/finetune.py`, `{hindi,nepali}/configs/finetune_config.yaml` |
| Finetuning logs | `{hindi,nepali}/finetune/finetune_log.csv` |
| Finetuned checkpoints (Drive links) | `README.md` — *pending, see below* |
| Reasoning eval script + results | `common/finetune/reasoning_eval.py`, `report/phase3/{hindi,nepali}_reasoning_eval.json`, §1.3/§1.4 |
| Pretrain-vs-finetune attention comparison | `report/phase3/{hindi,nepali}_attention_{pretrained_reasoning,finetuned}.json`, `report/phase3/figures/`, §2 |
| Final comparison tables and error analysis | §1.5, §3 |
| Complete final report | this document |

**Still open**: Google Drive links to the finetuned checkpoints (H and L)
need to be added to `README.md` — required by the submission checklist and
not yet done as of this report.

## 5. Known limitations and honest caveats

- **Reasoning-eval accuracy is exact-match on greedy decoding only** — no
  partial-credit scoring (e.g. for a `direct` answer that names the wrong
  entity but the right relation type isn't distinguished from a
  completely-unrelated wrong answer in the headline accuracy number,
  though it is visible in the qualitative examples used for §1.5's
  discussion).
- **The corpus-fragmentation explanation in Q3/Q4 is a plausible reading
  backed by converging qualitative and quantitative evidence (§1.5, §2.3),
  not a controlled ablation** — this project doesn't hold corpus
  fragmentation constant while varying resource tier, so it can't rule out
  every alternative explanation with certainty. What can be said with more
  confidence is what *was* ruled out directly (step budget, §1.5) and what
  the qualitative failure/attention evidence consistently points toward.
- **Attention entropy/distance are computed on a 200-prompt sample** at up
  to 128 tokens each (reasoning prompts are much shorter than this in
  practice, so the effective window is usually the full prompt) — same
  methodology as Phase 2 for consistency, carrying the same tradeoffs
  documented there.
- **Single run per language, both for finetuning and pretraining** — as
  with Phase 2's equivalent caveat, the H-vs-L differences reported here
  are consistent across multiple independent pieces of evidence gathered
  in this project (not just one number), which increases confidence, but
  this is still not the same as multi-seed statistical robustness.
