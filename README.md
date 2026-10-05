# Language Models and Agents — Individual Project

Two independent decoder-only Transformer LMs trained from scratch:

| | Model H (higher-resource) | Model L (lower-resource) |
|---|---|---|
| Language | Hindi | Nepali |
| Directory | [`hindi/`](hindi/) | [`nepali/`](nepali/) |

No shared data, tokenizer, vocabulary, or weights between the two models.

## Quick start: try the models in a browser (Gradio link)

The fastest way to run the project after cloning is the interactive demo
in `common/eval/interact.py`. It loads the pretrained and/or finetuned
checkpoints for both languages and serves a small web UI. Passing
`--share` prints a public `https://xxxx.gradio.live` link that works from
any browser. The link stays up only while the command is running (72 hours
at most).

The checkpoints are too large for git. Download them from the Google Drive
folder linked under [Large artifacts](#large-artifacts-rawprocessed-corpora-splits-checkpoints).
Only the four `best.pt` files are needed:

| Flag | File in the Drive folder |
|---|---|
| `--hindi_pretrained` | `hindi/output/best.pt` |
| `--hindi_finetuned` | `hindi/finetune_output/best.pt` |
| `--nepali_pretrained` | `nepali/output/best.pt` |
| `--nepali_finetuned` | `nepali/finetune_output/best.pt` |

You can leave out any of the four flags. The UI then offers only the
checkpoints you passed.

### Option A: Google Colab (recommended, no local setup)

```python
!git clone https://github.com/Harshita-K/hindi-nepali-agent.git LMA
%cd LMA
!pip install -q sentencepiece pyyaml gradio

# Add the shared Drive folder to your own Drive ("Add shortcut to Drive")
# so the paths below exist, then mount it:
from google.colab import drive
drive.mount('/content/drive')

!python -m common.eval.interact \
    --hindi_pretrained  /content/drive/MyDrive/LMA/hindi/output/best.pt \
    --hindi_finetuned   /content/drive/MyDrive/LMA/hindi/finetune_output/best.pt \
    --nepali_pretrained /content/drive/MyDrive/LMA/nepali/output/best.pt \
    --nepali_finetuned  /content/drive/MyDrive/LMA/nepali/finetune_output/best.pt \
    --share
```

Change the `/content/drive/MyDrive/LMA/...` paths if the folder sits
somewhere else in your Drive.

### Option B: local machine

```bash
git clone https://github.com/Harshita-K/hindi-nepali-agent.git LMA
cd LMA
pip install torch sentencepiece pyyaml gradio

# Download the four best.pt files from the Drive folder into ckpts/,
# keeping the same sub-paths (e.g. ckpts/hindi/output/best.pt), then:
python -m common.eval.interact \
    --hindi_pretrained  ckpts/hindi/output/best.pt \
    --hindi_finetuned   ckpts/hindi/finetune_output/best.pt \
    --nepali_pretrained ckpts/nepali/output/best.pt \
    --nepali_finetuned  ckpts/nepali/finetune_output/best.pt \
    --share
```

Run the command from the repository root. The models are small (~25M
parameters), so a CPU is enough; a GPU is used automatically if one is
available. Without `--share`, the UI is served only at
`http://127.0.0.1:7860`.

Using the UI: pick a language and a checkpoint, then enter a prompt, or
click one of the built-in examples. The finetuned checkpoints answer best
when the prompt ends with the answer cue (`उत्तर:` for Hindi, `जवाफ:` for
Nepali). Temperature 0 gives greedy decoding.

## Repository layout

```
common/            shared utility code (text cleaning, scraping, PDF/OCR, stats, URL discovery) used by both languages
hindi/
  data/
    scraping/      collection scripts (Wikipedia dump, public corpora, manual news scraping, manual PDF/OCR)
    raw/           raw per-source JSONL (gitignored -- large; goes to Drive)
    preprocessing/ cleaning, dedup, train/val/test split + stats generation
    processed/     cleaned+deduped+sampled corpus (gitignored)
    splits/        train/val/test JSONL (gitignored; large)
  tokenizer/
    train_tokenizer.py
    vocab/         trained SentencePiece .model/.vocab files (small -- committed)
  model/ train/ eval/   # Phase 2+
  configs/
    data_config.yaml
nepali/             mirrors hindi/
report/
  phase1/           corpus stats, tokenizer reports, phase 1 write-up
```

## Large artifacts (raw/processed corpora, splits, checkpoints)

Not committed to git. Uploaded to institutional OneDrive/SharePoint storage
(shareable link, not Google Drive — a substitution made due to a genuine
personal Drive/OneDrive storage-quota shortfall for the ~47GB combined
volume; see `report/phase1/report.md` section 10 for the full reasoning):

- **All raw sources, processed corpora, and splits (Hindi + Nepali):**
  https://iiithydresearch-my.sharepoint.com/personal/ahana_talukdar_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fahana%5Ftalukdar%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FLMA%5FHarshita&viewid=645125c6%2Dfd29%2D494e%2D9af6%2Ddc9d91243e02&ga=1
- **Pretrained checkpoints, optimizer/scheduler state, and training logs
  (Model H Hindi + Model L Nepali, Phase 2):**
  https://drive.google.com/drive/folders/1Upzy3quO_er26MlOW6MHpBm-aZrJd0Ua?usp=share_link
- **Finetuned reasoning checkpoints, optimizer/scheduler state, and
  finetuning logs (Model H Hindi + Model L Nepali, Phase 3):**
  https://drive.google.com/drive/folders/1Upzy3quO_er26MlOW6MHpBm-aZrJd0Ua?usp=share_link
  (same folder as the Phase 2 link above — finetuned checkpoints live under
  `hindi/finetune_output/` and `nepali/finetune_output/` alongside the
  pretrained `hindi/output/` and `nepali/output/`)


## Reproduction steps (Phase 1)

For each language (`hindi` / `nepali`), from the repo root:

```bash
pip install -r requirements.txt

# 1. Collect downloaded sources (public corpora -- slow, run in background)
python hindi/data/scraping/download_wikipedia.py
python hindi/data/scraping/download_public_corpus.py --dataset fineweb2 --limit 600000
python hindi/data/scraping/download_public_corpus.py --dataset indiccorp --limit 500000
python hindi/data/scraping/download_public_corpus.py --dataset sangraha --limit 600000
# nepali/ mirrors the same three commands

# 2. Collect manual sources (>=20% of tokens must be manual).
#    Each of these is a standalone, resumable, per-domain bulk scraper --
#    safe to Ctrl-C and re-run; already-scraped URLs are skipped.
#    Hindi:
python hindi/data/scraping/bulk_scrape.py --patrika-days 500        # Patrika + Jagran
python hindi/data/scraping/bulk_scrape_indiatv.py --days 500
python hindi/data/scraping/bulk_scrape_abplive.py --days 500
python hindi/data/scraping/scrape_archive_org.py --max-books 500
python hindi/data/scraping/scrape_wikisource.py
#    Nepali:
python nepali/data/scraping/bulk_scrape_onlinekhabar.py --months-back 28
python nepali/data/scraping/bulk_scrape_ekantipur.py --max-subsitemaps 140
python nepali/data/scraping/bulk_scrape_dcnepal.py --max-urls 200000
python nepali/data/scraping/bulk_scrape.py --setopati-id-start 6000     # Setopati
python nepali/data/scraping/bulk_scrape_khabarhub.py
python nepali/data/scraping/bulk_scrape_reportersnepal.py
python nepali/data/scraping/bulk_scrape_rajdhani.py
python nepali/data/scraping/bulk_scrape_nepalpage.py
python nepali/data/scraping/bulk_scrape_bizmandu.py
python nepali/data/scraping/bulk_scrape_ratopati.py                     # slow: site-mandated 20s crawl-delay
python nepali/data/scraping/scrape_archive_org.py --max-books 500
#    Optional supplementary downloaded datasets (Nepali only, used to
#    supplement the smaller native downloaded pool -- see report/phase1/report.md
#    section 5 for exact sources/licenses):
#      Kaggle: ashokpant/nepali-news-dataset-large, disisbig/nepali-wikipedia-articles
#      (requires `pip install kaggle` + ~/.kaggle/kaggle.json credentials)

# 3. Clean, dedup (manual wins ties against downloaded on exact duplicates), sample to target
python <lang>/data/preprocessing/build_corpus.py

# 4. Split + generate corpus statistics report
python <lang>/data/preprocessing/make_splits.py

# 5. Train the tokenizer + generate fertility/UNK report
python <lang>/tokenizer/train_tokenizer.py
```

Outputs land in `report/phase1/{lang}_corpus_stats.json` and
`report/phase1/{lang}_tokenizer_report.json`. See `report/phase1/report.md`
for the full write-up (dataset statistics, cleaning steps, tokenizer
comparison, manual-collection description).

## Phase 2 (Colab): architecture, configs, and pretraining

Implementation lives in `common/model/` (from-scratch decoder-only
Transformer: `config.py`, `transformer.py`, `sanity_checks.py`,
`count_params.py`) and `common/train/` (data binarization, checkpointing,
optimizer/schedule, `Trainer`), with thin per-language CLI entry points in
`hindi/train/` and `nepali/train/`. See `hindi/configs/model_config.yaml`
and `train_config.yaml` (Nepali mirrors them) for the exact architecture
(~25M params/model, learned absolute positional embeddings, 7-layer
pre-norm Transformer, tied embeddings) and pretraining hyperparameters
(~500M-token budget at 8000 steps).

Run in a fresh Colab notebook (T4 GPU runtime recommended):

```python
# 1. Get the code onto the Colab VM (either clone your repo, or upload a zip)
!git clone https://github.com/Harshita-K/hindi-nepali-agent.git LMA
%cd LMA
!pip install -q -r requirements.txt

# 2. Mount Drive for persistent checkpoint storage (Colab's local disk is
#    wiped on disconnect -- checkpoints MUST live somewhere durable).
from google.colab import drive
drive.mount('/content/drive')

# 3. Pull the Hindi/Nepali splits + tokenizer (see the OneDrive links above)
#    into hindi/data/splits/ and nepali/data/splits/, and confirm
#    hindi/tokenizer/vocab/ and nepali/tokenizer/vocab/ are present
#    (both are small and already committed to git).

# 4. Tokenize each language's splits into training binaries (run once)
!python hindi/train/prepare_data.py
!python nepali/train/prepare_data.py

# 5. Sanity-check the architecture before committing GPU time
!python -m common.model.sanity_checks --config hindi/configs/model_config.yaml
!python -m common.model.count_params --config hindi/configs/model_config.yaml
!python -m common.model.count_params --config nepali/configs/model_config.yaml

# 6. Pretrain. Re-running the same command after a disconnect auto-resumes
#    from out_dir/latest.pt -- no flags needed.
!python hindi/train/train.py  --out_dir /content/drive/MyDrive/lma_ckpts/hindi
!python nepali/train/train.py --out_dir /content/drive/MyDrive/lma_ckpts/nepali
```

Each `out_dir` accumulates `latest.pt` (resumable checkpoint, overwritten
each `save_interval`), `best.pt` (lowest val loss so far), and
`train_log.csv` (step, tokens seen, train/val loss, val perplexity, lr --
used for the Phase 2 loss-curve plots). Copy these to wherever you host
large artifacts (see the OneDrive link above) and link them in this README
before the Phase 2 deadline.

## Phase 2 (Colab): evaluation

Once a checkpoint exists (`best.pt` recommended -- lowest val loss), Phase
2.3's evaluation lives in `common/eval/` (`generate.py`, `intrinsic.py`,
`metrics.py`, `attention.py`, `plot_loss_curve.py`) with per-language CLI
entry points in `hindi/eval/` and `nepali/eval/`:

```python
CKPT_H = "/content/drive/MyDrive/lma_ckpts/hindi/best.pt"
CKPT_L = "/content/drive/MyDrive/lma_ckpts/nepali/best.pt"

# also needs the eval-only libs (BLEU/chrF/ROUGE-L)
!pip install -q sacrebleu rouge-score

# intrinsic metrics: perplexity + bits-per-byte on the held-out test split
!python hindi/eval/run_intrinsic.py  --ckpt {CKPT_H}
!python nepali/eval/run_intrinsic.py --ckpt {CKPT_L}

# generation quality: greedy + temp 0.5/1.0/1.5, BLEU-4/chrF++/ROUGE-L, diversity
!python hindi/eval/run_generation.py  --ckpt {CKPT_H}
!python nepali/eval/run_generation.py --ckpt {CKPT_L}

# attention analysis: heatmaps (early + late layer) + entropy/mean-distance summaries
!python hindi/eval/run_attention.py  --ckpt {CKPT_H}
!python nepali/eval/run_attention.py --ckpt {CKPT_L}

# loss curves from each run's train_log.csv
!python -m common.eval.plot_loss_curve --log /content/drive/MyDrive/lma_ckpts/hindi/train_log.csv  --out report/phase2/figures/hindi_loss_curve.png  --title "Model H (Hindi)"
!python -m common.eval.plot_loss_curve --log /content/drive/MyDrive/lma_ckpts/nepali/train_log.csv --out report/phase2/figures/nepali_loss_curve.png --title "Model L (Nepali)"
```

Outputs land in `report/phase2/`: `{lang}_intrinsic.json` (PPL/BPB),
`{lang}_generation.json` + `{lang}_generation_samples.txt` (metrics +
qualitative examples per decoding condition), `{lang}_attention.json` +
`figures/{lang}_attn_*.png` (heatmaps), and `figures/{lang}_loss_curve.png`.
Copy the whole `report/phase2/` folder back out of Colab (e.g. `zip -r
report_phase2.zip report/phase2 && files.download(...)`, or just `git add`
it from a synced clone) so it ends up committed on the `phase-2` branch --
per the assignment, only what's in the repo gets graded.

## Phase 3 (Colab): reasoning finetuning, evaluation, and attention comparison

Implementation lives in `common/finetune/` (`reasoning_gen.py` for the
synthetic-data generator, `data.py` for masked-loss batching, `trainer.py`
for the finetuning loop, `reasoning_eval.py` for exact-match evaluation),
with per-language entry points in `hindi/finetune/` and `nepali/finetune/`.
See `hindi/configs/finetune_config.yaml` and `nepali/configs/finetune_config.yaml`
for the exact hyperparameters (identical across languages, by design).

```python
CKPT_H_PRETRAINED = "/content/drive/MyDrive/LMA/hindi/output/best.pt"
CKPT_L_PRETRAINED = "/content/drive/MyDrive/LMA/nepali/output/best.pt"

# 1. Generate the synthetic comparative-reasoning dataset (once; deterministic/reproducible)
!python hindi/finetune/generate_reasoning_data.py
!python nepali/finetune/generate_reasoning_data.py

# 2. Finetune each language's own pretrained checkpoint. Re-running the same
#    command after a disconnect auto-resumes from out_dir/latest.pt.
!python hindi/finetune/finetune.py \
    --pretrained_ckpt {CKPT_H_PRETRAINED} \
    --out_dir /content/drive/MyDrive/LMA/hindi/finetune_output --max_steps 3000
!python nepali/finetune/finetune.py \
    --pretrained_ckpt {CKPT_L_PRETRAINED} \
    --out_dir /content/drive/MyDrive/LMA/nepali/finetune_output --max_steps 5000

# also needs the eval-only lib
!pip install -q sentencepiece

CKPT_H_FINETUNED = "/content/drive/MyDrive/LMA/hindi/finetune_output/best.pt"
CKPT_L_FINETUNED = "/content/drive/MyDrive/LMA/nepali/finetune_output/best.pt"

# 3. Exact-match reasoning eval: pretrained vs. finetuned, same held-out test set
!python hindi/eval/run_reasoning.py  --pretrained_ckpt {CKPT_H_PRETRAINED} --finetuned_ckpt {CKPT_H_FINETUNED}
!python nepali/eval/run_reasoning.py --pretrained_ckpt {CKPT_L_PRETRAINED} --finetuned_ckpt {CKPT_L_FINETUNED}

# 4. Attention analysis on comparative-reasoning prompts, pretrained vs.
#    finetuned (writes to report/phase3/ -- never touches the already-graded
#    report/phase2/ files; see the script docstrings for the flag meanings)
!python hindi/eval/run_attention.py \
    --ckpt {CKPT_H_PRETRAINED} \
    --text_jsonl hindi/data/reasoning/test.jsonl --text_field prompt \
    --fig_dir report/phase3/figures --report_path report/phase3/hindi_attention_pretrained_reasoning.json \
    --prefix hindi_pretrained_reasoning --label "Model H (Hindi, pretrained)"
!python hindi/eval/run_attention.py \
    --ckpt {CKPT_H_FINETUNED} \
    --text_jsonl hindi/data/reasoning/test.jsonl --text_field prompt \
    --fig_dir report/phase3/figures --report_path report/phase3/hindi_attention_finetuned.json \
    --prefix hindi_finetuned --label "Model H (Hindi, finetuned)"
!python nepali/eval/run_attention.py \
    --ckpt {CKPT_L_PRETRAINED} \
    --text_jsonl nepali/data/reasoning/test.jsonl --text_field prompt \
    --fig_dir report/phase3/figures --report_path report/phase3/nepali_attention_pretrained_reasoning.json \
    --prefix nepali_pretrained_reasoning --label "Model L (Nepali, pretrained)"
!python nepali/eval/run_attention.py \
    --ckpt {CKPT_L_FINETUNED} \
    --text_jsonl nepali/data/reasoning/test.jsonl --text_field prompt \
    --fig_dir report/phase3/figures --report_path report/phase3/nepali_attention_finetuned.json \
    --prefix nepali_finetuned --label "Model L (Nepali, finetuned)"
```

Outputs land in `report/phase3/`: `{lang}_reasoning_eval.json`
(pretrained-vs-finetuned exact-match accuracy, overall + by task type/
attribute, plus qualitative fixed/still-wrong examples) and
`{lang}_attention_{pretrained_reasoning,finetuned}.json` +
`figures/{lang}_{pretrained_reasoning,finetuned}_attn_*.png` (heatmaps).
Full write-up and analysis: `report/phase3/report.md`.

## Manual source compliance note

Scraping sources are restricted to sites whose `robots.txt` does not
explicitly disallow `ClaudeBot` / `anthropic-ai` / `Claude-Web` by name (or
the equivalent Cloudflare `Content-Signal: ai-train=no` convention).

**Excluded** (explicit AI-crawler block found): `amarujala.com`,
`bhaskar.com`, `bbc.com` (Hindi *and* Nepali), `hindi.news18.com`,
`nepalpress.com`, `zeenews.india.com`, `baahrakhari.com`, `himalkhabar.com`,
`samacharpatra.com`, `karobardaily.com`, `sahityapost.com`.

**Used** (Hindi): `patrika.com`, `indiatv.in`, `abplive.com`, `jagran.com`,
`archive.org`, Hindi Wikisource.

**Used** (Nepali): `onlinekhabar.com`, `ekantipur.com`, `dcnepal.com`,
`setopati.com`, `khabarhub.com`, `reportersnepal.com`, `rajdhanidaily.com`,
`nepalpage.com`, `bizmandu.com`, `ratopati.com`, `archive.org`.

See `report/phase1/report.md` section 5 for the full per-site methodology
(sitemap type, archive depth) and additional sites that were investigated
but found unusable (dead/unreliable, non-news content, no discoverable
sitemap) rather than excluded on compliance grounds.

## Requirements notes

- `tesseract` + `poppler` system binaries are required for OCR (`brew install tesseract tesseract-lang poppler` on macOS). Install the `hin` and `nep` Tesseract language packs.
- The `datasets` library (HF) is used only to pull raw text corpora (FineWeb2, IndicCorp, Sangraha) — no pretrained models or tokenizers are used anywhere, per the assignment's hard constraints.
- `kaggle` (optional) is used only for two supplementary downloaded Nepali datasets — requires a personal API token at `~/.kaggle/kaggle.json`, not committed to the repo.
- All bulk scrapers are resumable — safe to Ctrl-C and re-run; already-collected URLs are skipped on the next run.
