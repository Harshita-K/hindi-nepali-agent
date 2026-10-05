# Phase 1 Report — Data Collection and Tokenizer Construction

This report doesn't just state the final numbers — it explains the reasoning
behind every major decision made while building the two corpora, since the
"why" is often more informative than the "what" for a project like this.

## 1. Language selection

### Why Hindi for Model H

Hindi was chosen as the higher-resource language because it has, by a wide
margin, the largest publicly available text volume of any non-English Indian
language. Concretely, that abundance shows up at every stage of this
project:

- **Downloaded corpora**: full Wikipedia dumps, and large multilingual web
  crawls (FineWeb2, IndicCorp, Sangraha) all have substantial Hindi splits —
  large enough that hitting the 500M-token *downloaded* portion of the
  target was never in doubt.
- **Manual collection**: Hindi has a dense, commercially mature online news
  ecosystem — outlets like Patrika, India TV, and ABP Live publish hundreds
  of articles a day and expose years of sitemap-indexed archives, making
  large-scale compliant scraping straightforward.

Choosing a well-resourced language for Model H was deliberate: the
assignment's real difficulty is meant to live in Model L (the low-resource
language), so Model H should be the "control" — a case where hitting the
500M-token target and the 20% manual floor is achievable without heroics,
so that any modeling/pretraining problems in Phase 2 can be attributed to
architecture or training choices rather than a starved corpus.

### Why Nepali for Model L

Nepali was picked from the assignment's allowed lower-resource list
(Assamese, Bhojpuri, Bodo, Dogri, Konkani, Maithili, Manipuri, Mizo, Nepali,
Sindhi) for a few concrete reasons:

- It has a real, if smaller, presence in every major multilingual corpus
  used here (FineWeb2, IndicCorp, Sangraha, Wikipedia) — enough to make a
  competitive downloaded pool feasible, unlike some options on the list
  which are barely represented in these corpora at all.
- It has an active, fragmented online news industry — dozens of outlets,
  none dominant — which turned out to be the right kind of "hard": genuinely
  scarcer than Hindi, but not so scarce that manual collection was
  impossible. This made Nepali a meaningful stress-test of the
  manual-collection requirement rather than either a non-issue (as it would
  be for Hindi) or a dead end (as it might be for a language with almost no
  organized news media online).
- Practically, several of Nepali's largest outlets (Baahrakhari, Himal
  Khabar, Samachar Patra, Karobar Daily, SahityaPost, BBC Nepali) explicitly
  disallow AI crawlers in `robots.txt`, which had to be discovered and
  respected rather than worked around. That constraint is exactly the kind
  of realistic, compliance-driven obstacle a lower-resource-language data
  pipeline should surface.

## 2. Overall pipeline design and why it's structured this way

Every language directory follows the same `raw/ → processed/ → splits/`
staging, for reasons that came directly out of how the assignment is graded
and how the actual data collection unfolded:

- **`raw/` is never modified or deleted once written.** Each source (one
  scraper, one downloaded dataset) writes to its own file
  (`patrika_bulk.jsonl`, `fineweb2.jsonl`, etc.), tagged with
  `source_type: manual` or `downloaded` and a `source` name. This keeps
  every source's provenance auditable indefinitely and means the expensive
  scraping/downloading work never has to be repeated — `build_corpus.py` can
  be re-run as many times as needed (and was, repeatedly, as new sources
  came online) purely by reading `raw/`, with zero risk to already-collected
  data.
- **`processed/corpus.jsonl` is the single cleaned, deduplicated, sampled
  output.** It's a build artifact, fully reproducible from `raw/` plus the
  cleaning/sampling code — so it's safe to regenerate at any time and never
  needs manual editing.
- **`splits/{train,val,test}.jsonl` are doc-level splits of `processed/`.**
  Splitting at the document level (not the sentence or paragraph level)
  guarantees no partial-document leakage between train and eval sets.

This staged design is also why every scraper was written to be
**idempotent and resumable**: each one tracks already-collected URLs (by
reading its own output file) and skips them on re-run. That mattered in
practice — scraping ran across many sessions, was paused and resumed
repeatedly (including deliberately, to free disk headroom or to prioritize a
corpus rebuild), and a design that couldn't tolerate interruption would have
made the whole project fragile.

## 3. Downloaded-source selection — why these four

Both languages draw from the same four downloaded sources:

- **FineWeb2** — a large, deduplicated, quality-filtered multilingual web
  crawl. Chosen as the primary bulk downloaded source because its filtering
  pipeline is specifically designed to avoid the boilerplate/spam problems
  raw Common Crawl dumps have.
- **IndicCorp** — an Indic-language-specific corpus built explicitly for
  Indian-language NLP, giving cleaner in-domain text than a generic
  multilingual crawl would.
- **Sangraha** — an AI4Bharat corpus targeting Indian languages
  specifically; included because it draws from different underlying sources
  than FineWeb2/IndicCorp, adding real diversity rather than redundant
  coverage of the same web pages.
- **Wikipedia** (official dumps) — included for its high editorial quality
  and because it's a fundamentally different register of text (encyclopedic,
  structured) than news/web text, which helps balance the corpus rather than
  making it purely news-flavored.

Using three overlapping-but-distinct web-scale corpora rather than just one
was a deliberate choice to reduce the risk that the downloaded 80% ends up
dominated by one crawler's particular biases or gaps.

## 4. Manual-collection methodology

### The compliance-first approach

Before writing a single scraper, every candidate site's `robots.txt` was
checked for one thing specifically: does it name `ClaudeBot`, `anthropic-ai`,
`Claude-Web`, or the newer `Content-Signal: ai-train=no` convention as
disallowed? If yes, the site was excluded outright — not routed around with
a different user-agent, which would defeat the purpose of checking in the
first place. This is why the exclusion lists below exist and are as long as
they are: compliance was checked *before* investing in a scraper, not
discovered after the fact.

A secondary, more mundane filter also applied: some sites that were
technically permissive turned out not to be usable for other reasons —
unreliable connectivity, no real historical archive, or content that wasn't
actually news/prose text. Those are listed separately below as
"investigated but not viable," to keep the distinction between "we're not
allowed to" and "it didn't actually work" clear.

### Why per-site discovery method varies

Each manual source uses whichever URL-discovery method actually fit that
site's infrastructure, rather than a single uniform approach — because
forcing one pattern onto every site would have either missed most of a
site's archive or been unreliable:

- **WordPress `wp-sitemap.xml` sites** (DC Nepal, Khabarhub, Reporters
  Nepal, Nepal Page, and the Yoast-SEO variant used by Rajdhani Daily and
  Bizmandu) — these sites auto-generate a paginated sitemap index covering
  their *entire* post history, so this was always the first thing checked
  and, where available, the easiest and deepest source of URLs.
- **ID-range crawling** (Setopati, Ratopati) — used where a site routes
  articles purely by a numeric ID in the URL path regardless of the slug
  text (confirmed empirically by requesting a real ID with a dummy slug and
  getting a 200). This lets a wide numeric range be sampled directly without
  needing any sitemap at all, which matters because neither of these two
  sites exposes one.
- **Date-based sitemaps** (Patrika, ABP Live) — used where a site exposes a
  daily sitemap file keyed by date (`google-sitemap-{date}.xml`,
  `news-{DD-MM-YYYY}.xml`). Iterating one URL per calendar day going back as
  far as the archive stays dense gives deep, systematic coverage.
- **A custom two-level sitemap index** (Ekantipur) — this site's sitemap
  didn't match any standard convention, so its structure had to be
  reverse-engineered directly (`/sitemap_index/newsdetail/{n}`, confirmed by
  probing which page numbers still returned content — real coverage stopped
  around page 133 of 501 listed). It also required extra handling because
  each story is published as two parallel URLs (Nepali and an English
  translation); only the Nepali ones were kept.
- **Paginated date-archive crawling** (OnlineKhabar) — this was a *re-scrape*
  specifically to recover previously corrupted content (an encoding bug in
  an earlier pass had mojibake'd non-ASCII text), so it re-walks the site's
  own month-by-month archive listing pages rather than a sitemap.

### Rate limiting: why 0.8s and where it differs

All scrapers default to a self-imposed 0.8-second minimum interval between
requests to the same domain (not applied globally — different domains run
fully in parallel, since a per-domain lock is what's throttled). This value
was chosen as a middle ground: polite enough not to risk the target site
rate-limiting or blocking the collection IP (a real risk that was explicitly
weighed and deliberately avoided rather than tested), while still permitting
meaningful throughput. When asked later whether to speed collection up
further, the decision was to add *more parallel domains* rather than lower
this interval — a strictly safer way to increase total throughput, since
adding a new domain carries zero added risk to the domains already running,
whereas shortening the interval on an existing domain does. The one
exception is **Ratopati**, whose `robots.txt` itself specifies
`Crawl-delay: 20` — a site-mandated value, so it was honored as-is rather
than overridden with the project's own default.

### Sites used (Hindi)

- **Patrika** (patrika.com) — daily Google-News sitemaps, ~500 days of
  history. `robots.txt` explicitly *allows* AI crawlers by name, making it
  an unambiguous first choice.
- **India TV** (indiatv.in) — daily sitemap crawl, ~500 days of history.
- **ABP Live** (abplive.com) — daily sitemap, dense back through mid-2024.
  Found specifically to fill a gap after the original three Hindi sources
  were judged sufficient in volume but not yet at the 100M-manual-token
  target; its `robots.txt` carries no bot-specific restriction.
- **Jagran** (jagran.com) — flat `sitemap.xml`; a small contribution, kept
  because it was already compliant and cost nothing extra to include.

Books/PDFs: Hindi-language books from **archive.org** (42 documents) and
**Hindi Wikisource** (4,267 documents) — both chosen because they're
public-domain by construction, so there's no licensing ambiguity, and
archive.org's search API allows filtering by language directly.

**Excluded** (explicit AI-crawler block in `robots.txt`): Amar Ujala,
Dainik Bhaskar, BBC Hindi, News18 Hindi, Zee News, and any site carrying the
Cloudflare-managed `Content-Signal: ai-train=no` block pattern (the same
pattern seen on several excluded Nepali sites below).

### Sites used (Nepali)

Ten independent manual sources were ultimately built for Nepali —
substantially more than Hindi's four — precisely because Nepali's
per-site archives are shallower and its major outlets are more likely to
block AI crawlers, so reaching a comparable manual-token count required
spreading collection across many more (smaller) sources rather than relying
on two or three large ones:

- **DC Nepal** (dcnepal.com) — WordPress sitemap, expanded partway through
  the project from an initial small `--max-urls` cap to 200,000 specifically
  to push into the site's older archive once the easy recent content was
  exhausted.
- **Ekantipur** (ekantipur.com) — the single largest Nepali source found
  (~65K articles across ~19 months); investigated and built specifically
  because it's one of the largest national dailies and, unlike several
  comparably large outlets, carries no AI-crawler restriction.
- **OnlineKhabar** (onlinekhabar.com) — an original source, later re-run as
  a 28-month *recovery* pass after discovering an encoding bug had corrupted
  earlier-collected non-ASCII text.
- **Setopati** (setopati.com) — ID-range crawl.
- **Khabarhub** (khabarhub.com) — WordPress sitemap.
- **Reporters Nepal** (reportersnepal.com) — WordPress sitemap, archive back
  to 2017.
- **Rajdhani Daily** (rajdhanidaily.com) — Yoast SEO sitemap, archive back
  to 2019.
- **Nepal Page** (nepalpage.com) — WordPress sitemap, archive back to 2020.
- **Bizmandu** (bizmandu.com) — a business/economy-focused outlet,
  deliberately added for topical diversity (the other nine sources are
  general news) and because its archive reaches back to 2013, the deepest
  of any Nepali source used.
- **Ratopati** (ratopati.com) — ID-range crawl, run at the site's own
  mandated 20-second crawl delay.

Books/PDFs: Nepali-language books from **archive.org** (43 documents).

**Excluded** (explicit AI-crawler block or `ai-train=no` signal):
Baahrakhari, Himal Khabar, Samachar Patra, Karobar Daily, SahityaPost, BBC
Nepali, NepalPress.

**Investigated but found not viable** (compliant, but unusable for other
reasons — kept distinct from the exclusion list above because these weren't
policy decisions):
- **Gorkhapatra** — repeatedly unreachable (connection timeouts across
  multiple attempts on different days), and the one sitemap that did
  respond turned out to contain government job/service postings rather than
  news articles.
- **Nagarik News** — permissive `robots.txt`, but its `sitemap.xml`
  consistently failed with a malformed/truncated server response across
  retries. Its content was obtained anyway, but through a third-party
  GitHub-hosted corpus — which correctly counts as *downloaded*, not
  manual, since collecting it wasn't our own scraping effort.
- **News24 Nepal** — site suspended (account-suspension page).
- **Ekagajpatra** — turned out to be a document-verification/certification
  portal, not a news site, despite a plausible-sounding name.
- **Open Data Nepal** — a legitimate open-data portal, but hosts structured
  government datasets (tables, statistics), not prose text, so it's
  structurally unsuitable for a language-model corpus regardless of its
  licensing or compliance status.
- **Ujyaalo Online / Nepal Samaya** — permissive `robots.txt`, but neither
  exposes a discoverable sitemap, and both would have needed a bespoke
  page-scraping approach not attempted here given time constraints.

A tesseract-ocr wordlist (`nep.wordlist`) was also considered as a possible
Nepali text source and explicitly rejected: it's ~33,500 isolated words with
no sentence structure, useful for OCR dictionaries but actively harmful to
inject into a corpus meant to teach coherent language generation.

## 5. Supplementary downloaded datasets (Nepali)

Three third-party datasets were added specifically to grow Nepali's
downloaded pool without duplicating effort already covered by
FineWeb2/IndicCorp/Sangraha/Wikipedia:

- **Kaggle 20-category Nepali news dataset** (Shahi & Pant, 2018, GPL-2.0,
  6,927 articles across 20 topical categories) — a citable, academically
  published dataset, chosen for its clean labeling and known provenance.
- **A second Kaggle Nepali Wikipedia snapshot** (CC-BY-SA-4.0, 19,013
  articles) — added because it's a different snapshot (different
  article-count and coverage) than the official Wikipedia dump already in
  use, so it contributes real additional coverage rather than duplicating
  it; any exact-duplicate overlap is caught by the standard dedup pass
  regardless.
- **Nagarik News corpus** and a small **Nepali sports-news** dataset,
  both sourced from public GitHub repositories — added as smaller
  supplements. Neither repository has a LICENSE file, which is flagged
  explicitly here as a compliance consideration worth a second look before
  final submission, rather than silently treated as unrestricted.

All three are tagged `source_type: downloaded`, not `manual` — per the
assignment's actual distinction (see Section 6), that tag depends on *who*
did the collection work, not on what the content is. A dataset of scraped
news articles that a third party already collected, cleaned, and packaged
is downloaded even though its *content* looks identical to what our own
scrapers produce.

## 6. Manual vs. downloaded: what the distinction actually means here

This came up directly during the project and is worth stating explicitly:
**`manual` means we did the collection work ourselves** — our own scraper
hit a live site, or we OCR'd/transcribed the text ourselves. **`downloaded`
means someone else already collected, cleaned, and packaged it**, regardless
of whether the underlying content is news, encyclopedic text, or anything
else. This is why the Kaggle/GitHub datasets above — even though they're
literally news articles, the same *kind* of content our scrapers produce —
count as downloaded: the collection effort behind them wasn't ours.

## 7. Dataset statistics

| | Hindi (Model H) | Nepali (Model L) |
|---|---:|---:|
| Total documents | 1,368,253 | 1,893,004 |
| Total tokens (whitespace proxy) | 499,999,997 | 499,302,601 |
| Target tokens | 500,000,000 | 500,000,000 |
| Manual token fraction | 20.29% | 19.89% |
| Manual tokens | 101,450,874 | 99,302,601 |
| Downloaded tokens | 398,549,123 | 400,000,000 |
| Train / Val / Test doc counts | 1,340,887 / 13,682 / 13,684 | 1,855,143 / 18,930 / 18,931 |

**Why Nepali's downloaded figure is exactly 400,000,000:** rather than
letting the sampler fill downloaded tokens up to whatever gap remained after
manual (the default policy), Nepali's sampling was explicitly configured
with a direct downloaded-token cap of 400M. This was a deliberate choice
made once it was clear Nepali's raw downloaded supply (hundreds of millions
of tokens available) was never the constraint — manual collection was — so
capping downloaded at a clean, specific number and letting manual fill in
whatever it had reached gave a more interpretable, intentional split than
an emergent one.

**Why the manual-fraction floor (20%) isn't strictly enforced in either
final corpus:** the sampling code supports a hard-floor mode
(`enforce_ratio=True`) that guarantees ≥20% manual by capping downloaded
tokens to whatever ratio the current manual pool supports — even if that
means a smaller total corpus. Both final builds instead used
`enforce_ratio=False`, prioritizing the ~500M total-token target while
manual collection was still actively growing via ongoing scraping. The
practical effect: Hindi's manual collection had already grown enough by the
time of the final build that it *cleared 20% anyway* (20.29%) even without
the floor being enforced. Nepali landed at 19.89% — just under the floor —
which is reported honestly here rather than masked, along with the reason
(manual collection continuing to grow in parallel, not yet caught up to
where Hindi's had).

**Shortfall justification (Nepali):** 499,302,601 tokens against a
500,000,000 target is a shortfall of 697,399 tokens (0.14%) — effectively
negligible, and a direct consequence of the 400M downloaded cap plus
whatever manual had reached at build time, rather than any inability to
reach 500M (downloaded supply alone was far larger than needed).

## 8. Cleaning steps applied

1. **Unicode normalization (NFC) + mojibake repair (`ftfy`).** Needed
   because scraped HTML and some downloaded sources carry inconsistent
   encodings; NFC normalization also ensures visually-identical characters
   hash identically during dedup.
2. **Script filtering** — keep only lines with ≥60% Devanagari characters.
   Chosen as a practical threshold to drop non-Devanagari boilerplate
   (English ads, embedded code, navigation text) without being so strict
   that legitimate mixed-script sentences (a common, natural pattern in
   Hindi/Nepali news, e.g. an English proper noun inside a Devanagari
   sentence) get discarded.
3. **Whitespace/punctuation normalization.**
4. **Exact-duplicate removal via content hashing** (SHA-256 over
   normalized, lowercased, whitespace-collapsed text).
   - **Manual/downloaded tie-break, by explicit request:** when the same
     content hash appears in both a manual and a downloaded source (a real
     scenario — e.g. a news article picked up by both our own scraper and a
     web crawl like FineWeb2), the document is attributed to **manual**
     rather than whichever source happened to load first. This was
     implemented by sorting all documents so every manual document is
     processed before any downloaded document, prior to the dedup pass —
     since the dedup logic keeps the *first* copy of each hash it sees,
     this guarantees manual wins any such collision. Without this ordering,
     the outcome would have been decided arbitrarily by filename sort order,
     which isn't a defensible or even a *stable* policy across reruns as new
     source files get added.
5. **Minimum length filter** — drop documents under 20 characters
   post-cleaning, removing near-empty scraps that survive script filtering
   but carry no real content (e.g. a lone headline fragment).

## 9. Tokenizer

| | Hindi | Nepali |
|---|---:|---:|
| Algorithm | SentencePiece BPE | SentencePiece BPE |
| Vocabulary size | 8,000 | 8,000 |
| Avg chars/token | 3.685 | 4.202 |
| Unknown-token rate | 0.261% | 0.190% |

**Why SentencePiece BPE:** BPE gives a good balance of vocabulary coverage
and sequence-length efficiency for morphologically rich Indic scripts, and
SentencePiece specifically doesn't require pre-tokenization by whitespace
(unlike a raw BPE implementation), which matters for Devanagari text where
word boundaries and whitespace don't always align as cleanly as in English.

**Why 8,000 vocab for both languages (not 32K/24K):** a smaller vocabulary
was chosen deliberately over the initially-planned larger ones. The
practical tradeoff, visible directly in the tokenization examples below, is
more aggressive subword splitting (e.g. मुज़फ्फरनगर splits into
`▁मुज़`+`फ्फर`+`नगर` rather than fewer, larger pieces) in exchange for a much
smaller embedding table — proportionally more significant for a ~25M
parameter model (per the assignment's Phase 2 target) than it would be for a
larger model, where the embedding table is a small fraction of total
parameters regardless of vocab size. Both unknown-token rates stayed very
low (well under 0.3%) even at this smaller size, indicating the vocabulary
still covers the corpus well despite being a quarter to a third the size
originally planned.

**Tokenization examples:**

Hindi — *"तस्लीम बेनकाब मुज़फ्फरनगर। हालहि में हुई अवैध शराब के सेवन से मौतों के कारण शराब"*
→ `▁तस्` `लीम` `▁बेन` `काब` `▁मु` `ज़` `फ्फर` `नगर` `।` `▁हाल` `हि` `▁में` `▁हुई` `▁अवैध` `▁शराब` `▁के` `▁सेवन` `▁से` `▁मौत` `ों` `▁के` `▁कारण` `▁शराब`

Hindi — *"ये हो सकते हैं कारण वे समस्याएं जिनसे कान में दर्द महसूस हो सकता है, उनमें कान म"*
→ `▁ये` `▁हो` `▁सकते` `▁हैं` `▁कारण` `▁वे` `▁समस्याएं` `▁जिन` `से` `▁कान` `▁में` `▁दर्द` `▁महसूस` `▁हो` `▁सकता` `▁है` `,` `▁उनमें` `▁कान` `▁म`

Nepali — *"काठमाडौं, भदौ १५ । नेपाली बजारमा ओपो मोबाइलको नयाँ उत्पादन ओपो एफ नाइन आएको छ ।"*
→ `▁काठमाडौं` `,` `▁भदौ` `▁१५` `▁।` `▁नेपाली` `▁बजारमा` `▁ओ` `पो` `▁मोबाइ` `लको` `▁नयाँ` `▁उत्पादन` `▁ओ` `पो` `▁एफ` `▁न` `ाइन` `▁आएको` `▁छ` `▁।`

Nepali — *"मंसिर १० गते मतदान भएको डोल्पामा पनि मतगणनास्थल निर्माण लगायतका सबै तयारी भइसकेक"*
→ `▁मंसिर` `▁१०` `▁गते` `▁मतदान` `▁भएको` `▁डोल्पा` `मा` `▁पनि` `▁मतगणना` `स्थल` `▁निर्माण` `▁लगायतका` `▁सबै` `▁तयारी` `▁भइ` `सक` `ेक`

Note common, high-frequency words (`▁हो`, `▁के`, `▁छ`, `▁पनि`) still get
single whole-word pieces even at 8K vocab — the splitting pressure falls
mostly on longer, less frequent, or morphologically complex words, which is
the expected and desirable behavior of a frequency-driven BPE vocabulary.

**Token-frequency statistics** (computed by encoding the val split with the
trained model and counting how often each vocab piece id appears):

| | Hindi | Nepali |
|---|---:|---:|
| Vocab pieces used at least once | 7,868 / 8,000 (98.35%) | 7,899 / 8,000 (98.74%) |
| Unused vocab pieces | 132 (1.65%) | 101 (1.26%) |
| Used-piece frequency: min / median / mean / max | 1 / 34 / 121.94 / 27,324 | 1 / 36 / 99.73 / 22,364 |
| Most frequent piece | `▁के` (2.85% of all tokens) | `▁।` (2.84% of all tokens) |

The frequency distribution is heavily right-skewed, as expected for
natural-language BPE vocabularies: a handful of function words/punctuation
(postpositions like `▁के`/`▁में`, the Devanagari sentence-ending `।`) account
for a disproportionate share of tokens (median usage count of ~34-36 is
three orders of magnitude below the top piece's count in the tens of
thousands), while the long tail of vocab pieces covers rarer morphemes and
word fragments. Under 2% of the vocabulary went completely unused on the
val sample for either language, indicating the trained vocabulary is a good
fit for the corpus rather than allocating meaningful capacity to pieces the
data doesn't actually contain. Full top-20 piece breakdowns are in
`report/phase1/{hindi,nepali}_tokenizer_report.json` under
`token_frequency_stats`.

## Data access

The dataset and artifacts for this project are hosted at the following SharePoint link:

- [LMA_Harshita](https://iiithydresearch-my.sharepoint.com/:f:/g/personal/ahana_talukdar_research_iiit_ac_in/IgCtOEM_eA4BT7UtVBxYB2QqAa_mfO3IMLDeeR9dzQCzYOo?e=T9FLRd)


