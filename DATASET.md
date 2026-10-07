# DATASET.md - dataset cards

> **No synthetic, LLM-generated, invented or randomly-labelled data is used for training or evaluation.**
> The only synthetic text in the repository is in `tests/conftest.py` (a few benign sentences used to exercise code paths in unit tests, clearly labelled as fixtures).
> Raw and processed datasets are **not committed** (`data/raw/`, `data/processed/` are git-ignored) because the sources are Reddit / Twitter / YouTube-derived. Re-create them with `python scripts/download_data.py && python scripts/prepare_data.py`.
> Anything not confirmed from the source files, card, or paper is marked **Not verified.**

## Why four separate datasets?

No public dataset I could find provides sentiment **and** emotion **and** mental-health/distress labels on the same texts, in Hindi/Hinglish. I searched Hugging Face (queries: mental health, depression, suicide, distress, stress/dreaddit, anxiety, hinglish, code-mixed, Indic sentiment/emotion, self-harm) and the authors' sites. The result:

| Task | Dataset | Population | Languages with labels |
|---|---|---|---|
| **Risk / distress** (stress language) | Dreaddit | English Reddit posts in stress-related communities | English |
| **Sentiment** | Tweet Sentiment Multilingual (en, "hi" configs) + Hinglish YouTube comments | tweets / YouTube comments | English, Romanized Hindi / Hinglish |
| **Emotion** | BRIGHTER (SemEval-2025 Task 11, Track A; eng + hin) | sentences (see below) | English, Hindi (Devanagari) |

These are **different populations**. Labels are never merged into one "unified clinical target"; each head has its own loss and its own evaluation. Consequences that matter:

* There is **no labelled mental-health data in Hindi, Hinglish or any Indic language**, so risk performance outside English is *unmeasured*. The app flags such outputs as unvalidated.
* Per-language comparisons across tasks are not apples-to-apples (different texts, domains, annotators).
* I found **no suitable open dataset with suicide-risk or severity levels** that is real, multilingual, openly licensed and not just a reupload of Kaggle aggregates (e.g. `ourafla/Mental-Health_Text-Classification_Dataset` is a derived re-labelling of three other corpora; `PrevenIA/spanish-suicide-intent` is machine-translated). They were rejected. Hence the system has **no** severity or suicide-risk output.

---

## Dataset A - Dreaddit (risk / distress task)

| Field | Value |
|---|---|
| Exact name | Dreaddit: A Reddit Dataset for Stress Analysis in Social Media |
| Source / URL | Authors' site: <http://www.cs.columbia.edu/~eturcan/data/dreaddit.zip> (paper: <https://arxiv.org/abs/1911.00133>) |
| Authors / publisher | Elsbeth Turcan, Kathleen McKeown (Columbia University); Tenth International Workshop on Health Text Mining and Information Analysis (LOUHI), 2019 |
| License | **Not verified.** The distributed zip contains only two CSV files and no license/README. (The arXiv listing's license applies to the paper, not the data.) Treat as research-use only and do not redistribute. Reddit's terms also apply. |
| Retrieval date | 2026-10-07 |
| Files / checksums | `dreaddit-train.csv`, `dreaddit-test.csv`; SHA-256 recorded in `data/raw/MANIFEST.json` at download time |
| Language | English |
| Examples (raw) | train 2,838 · test 715 (official split). Paper abstract: ~3.5K segments from ~3K posts. |
| Examples (used) | train 2,391 · val 425 · test 715 (after de-duplication and a post-grouped validation split) |
| Text type | Real Reddit post segments (not synthetic, not translated) |
| Source population | Reddit posts. Subreddits present in the train file: ptsd, relationships, anxiety, domesticviolence, assistance, survivorsofabuse, homeless, almosthomeless, stress, food_pantry. Paper: "five different categories of Reddit communities". Demographics: **Not verified** (Reddit skews young, Western, English-speaking). |
| Label definition | `label` 1 = segment annotated as expressing **stress**, 0 = not. Crowd-annotated on Amazon Mechanical Turk (paper). `confidence` column = annotator agreement (not used for training). Per-annotator details: **Not verified.** |
| Label distribution (train used) | stress 1,251 · non_stress 1,140 (ratio 1.10 → no class weighting; see below) |
| What the label is **not** | It is **not** a clinical label. It is not depression, anxiety, PTSD, or suicide-risk status - it is whether crowd workers judged a text segment to express stress. |
| Preprocessing | Unicode NFKC, control-char removal, URLs→`<url>`, e-mails→`<email>`, phone numbers→`<phone>`, @mentions→`@user`, whitespace collapse. Case preserved. |
| Exclusions | exact in-split duplicates (16); train rows that are exact/near duplicates (TF-IDF cosine ≥ 0.9 after normalisation) of val/test rows (5); val rows duplicating test (1). Test is never trimmed except exact in-split duplicates. |
| Splits | official train/test kept. Validation = 15 % of train, **split by `post_id`** so segments of one post never straddle splits (train/test share no `post_id`; verified). The post id is dropped from processed data; only the coarse subreddit is kept. |
| Privacy limitations | Public Reddit text about sensitive life events; usernames are not in the CSV but text may contain self-disclosures that could be re-identified by search. Identifiers are masked at preprocessing. Do not attempt to re-identify authors. |
| Known biases | Stress as perceived by crowd workers, not by the author or a clinician; topic/community confound (e.g. PTSD/abuse communities vs. food-pantry/financial); English-only; short segments. |
| Intended use (per paper) | Research on stress detection in social media. |
| Inappropriate uses | Individual-level assessment, diagnosis, triage, surveillance of users, any clinical decision. |

## Dataset B1 - Tweet Sentiment Multilingual (sentiment task, English + Romanized Hindi)

| Field | Value |
|---|---|
| Exact name | `cardiffnlp/tweet_sentiment_multilingual` (UMSAB, XLM-T) |
| Source / URL | <https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual>, revision `14b13edfbc4046892f6011d114c29c0f83170589`; only `data/english/*.jsonl` and `data/hindi/*.jsonl` are used |
| Authors / publisher | Francesco Barbieri, Luis Espinosa Anke, Jose Camacho-Collados (Cardiff NLP). *XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond*, LREC 2022 (<https://aclanthology.org/2022.lrec-1.27>) |
| License | Per dataset card: Creative Commons Attribution 3.0 Unported (link to a SemEval Google-group post), **and** use must comply with Twitter's Terms of Service / Developer Agreement. Verify before any redistribution. |
| Retrieval date | 2026-10-07 |
| Languages | Card lists 8 languages; **only `english` and `hindi` are used.** |
| **Important finding** | The `hindi` config is **not Devanagari**. I measured the script: 1,838/1,839 train, and 100 % of validation and test tweets are Latin script. It is **Romanized Hindi / Hinglish-style, code-mixed**, and contains many **plain English** tweets. I therefore tag it `hi-Latn` and do **not** claim Devanagari sentiment support. (Card does not describe this.) |
| Examples (card) | per language: train 1,838 · val 323 · test 869. Files contain 1,839 / 324 / 870 lines (off by one, likely header/trailing line; counts used are measured from the files). |
| Examples (used) | en-Latn: train 1,833 · val 322 · test 870. hi-Latn (tweets only): see combined table below. |
| Text type | Real tweets, **not** synthetic. Whether any language was machine-translated: **Not verified** (counts are identical per language, which is a reason to check; not confirmed either way). Underlying source datasets: "extended\|other-tweet-datasets" - **Not verified.** |
| Labels | 0 negative · 1 neutral · 2 positive (card). Balanced by construction in this release (613/613/613 in train). Annotation procedure: **Not verified** here (see paper). |
| Privacy limitations | Tweets contain raw @handles in the raw files (not all anonymised); masked to `@user` at preprocessing. Twitter ToS restricts redistribution of tweet text → not committed. |
| Known biases | Twitter population; topic/time of collection not documented here; the "hindi" slice is domain-noisy as described above. |

## Dataset B2 - Hinglish YouTube comments sentiment (sentiment task, Hinglish)

| Field | Value |
|---|---|
| Exact name | `shae2977/hinglish-youtube-sentiments-dataset` |
| Source / URL | <https://huggingface.co/datasets/shae2977/hinglish-youtube-sentiments-dataset>, revision `84abda5e30d082a66c251fc46513cb89ce866cbb` |
| Authors / publisher | Hugging Face user `shae2977` (individual; no paper). **Not peer-reviewed.** |
| License | CC-BY-4.0 (dataset card) |
| Retrieval date | 2026-10-07 |
| Language | Hinglish (Romanized Hindi mixed with English), Latin script (card). Tag: `hi-Latn`. |
| Examples | card: 3,190 rows (neg 1,427 · neu 770 · pos 993). Columns: `video_id`, `comment`, `likes`, `sentiment`. 23 distinct videos. After removing 9 duplicate comments etc.: 3,174 used. |
| Text type | Real YouTube comments, scraped from entertainment videos (Bollywood, music, comedy, cooking); filtered to Roman-script Hinglish. Not synthetic, not translated. |
| Labels | Negative / Neutral / Positive, manually annotated in Label Studio (card). Number of annotators and inter-annotator agreement: **Not verified** (not stated; likely a single annotator - treat labels as noisy). |
| Splits | No official split. **Group split by `video_id`** (70/15/15, seeded) so comments from one video never straddle splits; with only 23 videos, split sizes and results are coarse and noisy. |
| Privacy limitations | Public comments; no usernames in the file; comment text may still contain personal info → masked at preprocessing. |
| Known biases | Entertainment domain (not mental-health related), single-annotator, 23 videos, likely popular-video skew. Sentiment model performance here says little about distress-related text. |

**Combined sentiment table (used):** train 5,854 · val 1,121 · test 2,213 (en-Latn 870 test; hi-Latn 1,343 test = tweets-hindi + youtube-hinglish). Train label counts: negative 2,381 · neutral 1,691 · positive 1,782.

## Dataset C - BRIGHTER emotion (emotion task, English + Hindi)

| Field | Value |
|---|---|
| Exact name | BRIGHTER - `brighter-dataset/BRIGHTER-emotion-categories`, configs `eng`, `hin` (SemEval-2025 Task 11, Track A) |
| Source / URL | <https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories>, revision `419566ac8f46f951b51584be36c13e5363008144`; papers <https://arxiv.org/abs/2502.11926> (ACL 2025), <https://arxiv.org/abs/2503.07269> |
| Authors / publisher | Shamsuddeen Hassan Muhammad, Nedjma Ousidhoum et al. (48 authors) |
| License | CC-BY-4.0 (dataset card) |
| Retrieval date | 2026-10-07 |
| Languages used | English (`eng`), Hindi (`hin`, **Devanagari**, verified from the files) |
| Examples (raw) | eng: 2,764 train · 230 dev · 5,528 test. hin: 2,556 train · 200 dev · 2,020 test |
| Examples (used) | en-Latn: train 2,379 · val 475 · test 2,759. hi-Deva: train 2,243 · val 405 · test 1,010 (see de-duplication) |
| Text type | Human-annotated by "fluent speakers" (paper abstract). Per-language text sources (e.g. which platforms/texts were used for English vs Hindi), and whether any text was translated: **Not verified** (not in the abstract/card). |
| Labels | Multi-label, 6 binary columns: anger, disgust, fear, joy, sadness, surprise; all-zero = no emotion. |
| **Important finding 1** | The English files have **no `disgust` annotation at all** (column entirely null). It is treated as **missing (-1), masked in the loss and in all metrics** - not as "absent". English `disgust` is therefore never evaluated, and the model's English disgust output is unvalidated. |
| **Important finding 2** | The dev/test files list every text **twice** (Track A and Track C copies, identical labels). These exact duplicates were collapsed (3,989 rows). Also 10 corrupted `#name?` rows were dropped. The official test sets are labelled (not blind). |
| Label rates (train) | en: anger .12 · fear .58 · joy .24 · sadness .32 · surprise .31 (no-emotion 8.7 %). hi: anger .17 · disgust .10 · fear .15 · joy .17 · sadness .17 · surprise .11 (no-emotion 23.5 %). English and Hindi have **very different label distributions**, so a pooled emotion F1 mixes two annotation regimes. |
| Splits | official train/dev/test kept; dev is tiny, so validation = dev + a seeded 12 % of train. |
| Privacy limitations | Source texts and their provenance: **Not verified.** Handled as potentially sensitive. |
| Known biases | Annotation regimes differ by language; label sets differ (English lacks disgust). |

## Dataset D - language / script identification

No separate labelled language-ID dataset was used. Script is derived deterministically from Unicode blocks. For Latin-script text, a small char-n-gram classifier (English vs Romanized Hindi) is trained from labels *derived from the sources above* (YouTube Hinglish train → `hi-Latn`; English tweets/BRIGHTER-eng/Dreaddit train → `en-Latn`). This has a **source/style confound** (one domain for Hinglish). See `reports/metrics/langid.json` for held-out accuracy and a noisy probe. `lingua-language-detector` (open-source library) is used for Devanagari Hindi-vs-Marathi and for clearly non-English Latin-script languages. Languages other than en/hi are detected at best by script and are **not evaluated**.

## De-duplication & leakage controls

* exact duplicates within each split removed (normalised key keeps Devanagari combining marks - an earlier `\W`-based key wrongly collapsed Hindi sentences; fixed and unit-tested);
* near-duplicates (char 4–5-gram TF-IDF cosine ≥ 0.9, punctuation/case-normalised) removed from the lower-priority split (test > val > train);
* group-wise splits where metadata exists (Dreaddit `post_id`, YouTube `video_id`);
* no author metadata exists in any dataset, so same-author leakage **cannot be ruled out**; translated-duplicate and benchmark-contamination checks across datasets were **not** performed (and MuRIL's pre-training data may include text overlapping with public benchmarks - unknowable here).

## Class imbalance (measured → decision)

| Task | Finding | Strategy |
|---|---|---|
| sentiment | train ratio max/min = 1.41 | none (below the 1.5 threshold) |
| risk | ratio 1.10 | none - and no oversampling of the sensitive class |
| emotion | positive rates 10–58 % per label | `pos_weight = clip(√(neg/pos), 1, 5)` per label, plus per-label decision thresholds tuned on validation F1 |

Details: `reports/metrics/class_balance.json`.
