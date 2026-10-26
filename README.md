# Multilingual Mental-Health Risk & Sentiment Analyzer

A multi-task NLP **research / screening** system. For a piece of text it returns **sentiment**, **emotion**, a **stress-related distress-language signal** (with a calibrated model probability), **language + script**, and token-level **model attribution** - from three model families (TF-IDF + LogReg baseline, multi-task BiLSTM, multi-task MuRIL transformer) behind a FastAPI service and a research-dashboard UI.

> **This tool is an NLP research and screening system. It is not a medical diagnostic tool and should not be used to make clinical decisions.**
> It does not diagnose depression, anxiety, PTSD or any disorder, cannot tell whether a person has a mental illness, and makes no claim of clinical validity for suicide risk. Model probabilities are statistical model outputs, **not clinical probabilities**.

## What the "risk" output really is (read this first)

* The risk head is trained on **Dreaddit**: English Reddit post segments from stress-related communities, labelled **stress / non-stress** by crowd workers. So the output is a **stress-language signal**, wording: *"language associated with elevated distress"*.
* **No dataset here has suicide-risk or severity levels**, so the system has no such output and the "high-risk" safety message is wired in but can never fire (`HIGH_RISK_LABELS` is empty - unit-tested).
* Risk is **evaluated on English only.** There is no labelled mental-health data in Hindi/Hinglish in this project. For other languages the risk output is an *unvalidated zero-shot transfer*; the API sets `validated_for_language.risk=false` and the UI says so.
* The Hindi/Hinglish examples from the brief (e.g. *"Yaar kuch bhi karne ka mann nahi karta"*) **will run through the model**, but I cannot tell you how accurate the result is, because no labelled data exists to measure it.

## Datasets (details, licenses, caveats: [`DATASET.md`](DATASET.md))

| Task | Dataset | Languages with labels |
|---|---|---|
| Risk (stress language) | Dreaddit (Turcan & McKeown, 2019) | English |
| Sentiment (3-class) | Tweet Sentiment Multilingual (Barbieri et al., 2022) en + "hi" · Hinglish YouTube comments (community dataset) | English, **Romanized** Hindi / Hinglish |
| Emotion (6-label, multi-label) | BRIGHTER / SemEval-2025 Task 11 | English, Hindi (Devanagari) |

Three different populations; labels are never merged. Things I found by inspecting the data rather than trusting the cards:
1. The tweet dataset's `hindi` config is **Latin-script (Romanized/code-mixed) and partly plain English**, not Devanagari → tagged `hi-Latn`; no Devanagari sentiment data exists in the project.
2. BRIGHTER English has **no `disgust` annotation** → masked (not treated as 0) in loss and metrics.
3. BRIGHTER dev/test list every text **twice** (Track A/C copies) → deduplicated; 10 corrupted `#name?` rows dropped.
4. A first version of my dedup key stripped Devanagari vowel signs and wrongly collapsed Hindi sentences; a near-duplicate check missed punctuation variants. Both were found (by a unit test / inspection), fixed, and **all models were retrained from scratch on the corrected splits**.

## Architecture

```
text → clean/PII-mask → tokenizer (word-level | MuRIL WordPiece)
     → encoder (BiLSTM | MuRIL) → masked mean+max pooling
     → shared layer (Linear-GELU-Dropout)
         ├── sentiment head  (softmax, 3)    CE loss
         ├── emotion head    (sigmoid, 6)    masked BCE (+pos_weight)
         └── risk head       (softmax, 2)    CE loss
L_total = λ_sent·L_sent + λ_emo·L_emo + λ_risk·L_risk      (λ = 1,1,1 by default, `--loss-weights`)
```
Batches are task-homogeneous (each comes from one dataset). **Why share a trunk?** Sentiment and emotion are both affect-related and plausibly share lexical/semantic features; stress-in-Reddit-posts is a different construct from a different population, so it has its own head/loss. **Whether sharing helps here was not ablated** (no single-task control runs) - treat it as a design rationale, not a result. Probabilities are temperature-scaled on the validation split; emotion decision thresholds are tuned per label on validation.

Transformer: `google/muril-base-cased` (Apache-2.0; 237.6 M params; trained on 17 Indian languages incl. transliterated text). Chosen over XLM-R-base/mBERT for Indic + transliteration coverage; all three tokenizers and loading were verified, and CPU step time was benchmarked (MuRIL ≈ XLM-R, distil-mBERT ≈ 2.5× faster). I did **not** run a head-to-head of the encoders. License of weights per the model card; language coverage per the MuRIL paper (not independently verified).

## Results

All numbers are measured on held-out **test** splits by `scripts/evaluate.py` and rendered from `reports/metrics/*.json` by `scripts/results_markdown.py`. Single training run (seed 13). Bootstrap CIs (over test examples only; training-seed variance is *not* captured) are given for the headline metrics; the Hinglish split rests on 23 videos and the risk test set has 715 segments.

### Model comparison (test splits)

| Model | Params | Sentiment macro-F1 | Emotion macro-F1 | Risk F1 (stress) | Risk macro-F1 | Risk PR-AUC | Risk ROC-AUC | Risk ECE (cal.) | Risk ECE (raw) | Latency ms (CPU) |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 1,159,030 | 0.489 | 0.612 | 0.760 | 0.740 | 0.843 | 0.826 | 0.039 | 0.060 | 3.9 |
| BiLSTM | 3,182,675 | 0.475 | 0.561 | 0.742 | 0.735 | 0.821 | 0.818 | 0.043 | 0.209 | 3.9 |
| Multilingual Transformer (MuRIL) | 237,952,523 | 0.611 | 0.690 | 0.807 | 0.790 | 0.830 | 0.848 | 0.033 | 0.125 | 89.2 |

### Macro-F1 by language / script (test)

| Task | Language (n) | baseline | bilstm | transformer |
|---|---|---|---|---|
| sentiment | en-Latn (870) | 0.509 | 0.491 | 0.696 |
| sentiment | hi-Latn (1343) | 0.478 | 0.466 | 0.557 |
| emotion | en-Latn (2759) | 0.525 | 0.503 | 0.644 |
| emotion | hi-Deva (1010) | 0.715 | 0.635 | 0.782 |
| risk | en-Latn (715) | 0.740 | 0.735 | 0.790 |

### Sentiment macro-F1 by source (test)

| Source | baseline | bilstm | transformer |
|---|---|---|---|
| tweets-english | 0.509 | 0.491 | 0.696 |
| tweets-hindi | 0.488 | 0.446 | 0.552 |
| youtube-hinglish | 0.464 | 0.496 | 0.558 |

### Emotion per-class F1 (test; English / Hindi)

| Model | Language | anger | disgust | fear | joy | sadness | surprise |
|---|---|---|---|---|---|---|---|
| baseline | en-Latn | 0.260 | n/a | 0.733 | 0.482 | 0.556 | 0.593 |
| baseline | hi-Deva | 0.691 | 0.789 | 0.773 | 0.689 | 0.581 | 0.768 |
| bilstm | en-Latn | 0.266 | n/a | 0.721 | 0.463 | 0.521 | 0.544 |
| bilstm | hi-Deva | 0.620 | 0.673 | 0.733 | 0.642 | 0.536 | 0.608 |
| transformer | en-Latn | 0.486 | n/a | 0.773 | 0.664 | 0.647 | 0.652 |
| transformer | hi-Deva | 0.656 | 0.757 | 0.816 | 0.835 | 0.757 | 0.874 |

### Calibration (temperature scaling fitted on validation)

| Model | Task | T | ECE raw → calibrated | Brier raw → calibrated |
|---|---|---|---|---|
| baseline | sentiment | 0.91 | 0.047 → 0.041 | 0.614 → 0.614 |
| baseline | emotion | 1.08 | 0.024 → 0.013 | 0.139 → 0.139 |
| baseline | risk | 1.37 | 0.060 → 0.039 | 0.348 → 0.342 |
| bilstm | sentiment | 8.21 | 0.392 → 0.033 | 0.884 → 0.627 |
| bilstm | emotion | 1.63 | 0.088 → 0.022 | 0.167 → 0.159 |
| bilstm | risk | 5.39 | 0.209 → 0.043 | 0.461 → 0.351 |
| transformer | sentiment | 1.81 | 0.094 → 0.052 | 0.528 → 0.516 |
| transformer | emotion | 1.16 | 0.030 → 0.009 | 0.126 → 0.125 |
| transformer | risk | 2.11 | 0.125 → 0.033 | 0.344 → 0.312 |


### Bootstrap 95% CIs (test examples resampled; training seed NOT varied)

| Model | Sentiment macro-F1 | Risk F1 | Risk PR-AUC |
|---|---|---|---|
| baseline | 0.488 [0.467, 0.511] | 0.760 [0.727, 0.794] | 0.843 [0.807, 0.875] |
| bilstm | 0.475 [0.455, 0.495] | 0.741 [0.706, 0.775] | 0.822 [0.782, 0.862] |
| transformer | 0.611 [0.590, 0.632] | 0.807 [0.774, 0.835] | 0.832 [0.785, 0.875] |

Paired differences vs. TF-IDF baseline (same resamples):

| Model | Δ Sentiment macro-F1 | Δ Risk F1 | Δ Risk PR-AUC |
|---|---|---|---|
| bilstm | -0.013 [-0.037, +0.010] | -0.019 [-0.048, +0.010] | -0.021 [-0.044, +0.000] |
| transformer | +0.123 [+0.097, +0.147] ✔ | +0.047 [+0.018, +0.079] ✔ | -0.010 [-0.045, +0.026] |

✔ = interval excludes 0. Latency = median single-text time for all three heads, CPU, 4 cores, measured with nothing else running.

![F1 by language](reports/figures/f1_by_language.png)
![Reliability diagram](reports/figures/reliability_risk.png)

Per-example errors (anonymised IDs, no text): [`reports/error_analysis.csv`](reports/error_analysis.csv) · summaries: `reports/metrics/error_summary.json` · language-ID: `reports/metrics/langid.json`.

## Key findings & honest interpretation

1. **The transformer clearly beats the baseline on sentiment (+0.12 macro-F1, CI excludes 0) and emotion (+0.08, both languages) and on risk F1 (+0.047, CI excludes 0).** But on risk **PR-AUC the transformer (0.832) is *not* better than TF-IDF (0.843)**: the difference is inside noise (Δ −0.010, CI [−0.045, +0.026]). The risk-F1 gain is mostly a recall gain at the fixed 0.5 threshold (recall 0.843 vs precision 0.774), i.e. a different operating point, not clearly better ranking. For the sensitive task, a 238 M-parameter model buys little over a 1 M-parameter bag-of-n-grams model except a modestly better ROC-AUC (0.848 vs 0.826) and 23× the latency (89 ms vs 4 ms).
2. **The BiLSTM adds no value over TF-IDF** on any task (differences vs baseline are negative and not significant; emotion is lower in both languages). With randomly-initialised embeddings and 2-6 K examples per task validation F1 plateaus around epoch 6 (≈0.59) while training loss keeps falling (2.5 → 0.41 over 12 epochs), i.e. it memorises rather than generalises (best epoch 11, only marginally above epoch 6). This is a negative result, reported as such; it also means the "shared-trunk BiLSTM" is not evidence that multi-task sharing helps (no single-task ablation was run).
3. **Calibration matters, mostly for the neural models.** Raw BiLSTM risk probabilities were badly over-confident (ECE 0.209 → 0.043 after temperature scaling, T = 5.4); the transformer improved 0.125 → 0.033 (T = 2.1). ECE on 715 examples is itself noisy, and Brier improvements are small - temperature scaling fixes confidence scale, not discrimination. Calibrated here ≠ clinically meaningful.
4. **Language gap (sentiment, transformer):** English 0.696 vs Hinglish/Romanized-Hindi 0.557 (−0.14); tweets-hindi 0.552 and YouTube-Hinglish 0.558. The two Hinglish sources are noisy (partly plain-English tweets; single-annotator comments). TF-IDF and BiLSTM are poor everywhere (≈0.47-0.51), so the multilingual pre-training is what carries Hinglish.
5. **Do not read "Hindi emotion (0.782) > English emotion (0.644)" as "Hindi is easier."** The two come from different annotation campaigns with very different label distributions (English `fear` is positive in 58 % of training rows, Hindi 15 %; English has no disgust labels; English `anger` F1 is only 0.49).
6. **Risk errors (transformer):** 91 false positives, 58 false negatives of 715 (precision 0.774, recall 0.843): about 1 in 4 flagged segments is a false alarm and ~16 % of stress-labelled segments are missed. Almost the entire risk test set is long text (693/715 ≥ 40 tokens), so the short-text behaviour of the risk head is untested. For sentiment, short texts (<8 tokens) err more (44 % vs 37 % medium).
7. **Sentiment ≠ stress.** On Dreaddit test segments, 83 % of truly-stressed segments get *negative* predicted sentiment vs 39 % of non-stress ones: correlated, but 17 % of stress segments are not negative. (Exploratory: the sentiment head was never validated on Reddit text.) Neutral is the weakest sentiment class (F1 0.53).
8. **Code-mixed/Hinglish:** evaluated as-is (original text, no normalisation) for sentiment only; no emotion or risk labels exist for Hinglish, so those heads are unmeasured there.

## Interpretability

`POST /predict {"explain": true}` (or the UI toggle) returns **Integrated Gradients** attributions over input embeddings for the stress-class score and the predicted sentiment. It is labelled *"Model attribution - not clinical reasoning"*. Attributions are approximate, can be unstable, and say what influenced this model's score, nothing about the author.

## Safety design

* Always-visible disclaimer; wording describes *language signals*, never "you have X".
* Elevated-signal notice: *"This result indicates language associated with elevated distress. This tool is not a medical diagnostic system."*
* Resources shown with an elevated signal are two **international directories only** (Find a Helpline; IASP crisis centres) - both fetched and confirmed on 2026-10-07. **No phone numbers are shipped** (they are country-specific and go stale); the user picks their own country. Re-verify the links before deployment.
* Unvalidated-language warnings (see above).

## Privacy

Submitted text is processed in memory and never logged, printed, or stored. `src/privacy.py` logs only whitelisted metadata (request ID, timestamp, route, status, latency, model version, language tag, risk label); any other field is dropped (unit-tested with a marker string across `/predict`, explain and batch). Validation errors don't echo input. Batch CSVs are read into memory and discarded (nothing is written to disk; results CSV is built in the browser); size/row limits apply; spreadsheet-formula injection in echoed IDs is neutralised. `data/`, checkpoints, uploads and logs are git-ignored. Identifiers (emails, phones, URLs, @mentions) are masked before text reaches a model. No user text is sent to any external service; the optional Sarvam adapter only runs if you call it and set `SARVAM_API_KEY`.

## Run it yourself (clone and go)

Tested with Python 3.13 on Linux. Python 3.10+ should work, but only 3.13 was used for the reference runs. You need about 10 GB of free disk and an internet connection for the first dataset download. A GPU is optional.

The repository does **not** ship trained models or raw datasets. Model files are too large for git, and the datasets come from Reddit, Twitter and YouTube, so they are not redistributed (see [`DATASET.md`](DATASET.md)). After cloning you download the public datasets and train the models yourself, which the steps below do.

### 1. Clone and install
```bash
git clone https://github.com/faizdevx/Multilingual-Mental-Health-Risk-.git
cd Multilingual-Mental-Health-Risk-
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt      # exact pins used for the experiments: requirements-lock.txt
```
`requirements.txt` leaves `torch` unpinned so you get the build that matches your machine (CPU or CUDA). If `pip install torch` picks the wrong build, follow the selector on pytorch.org.

### 2. Check the install (about a minute, no downloads, no GPU)
```bash
python -m pytest -q                       # unit tests
python scripts/train_lstm.py --smoke      # 2-epoch pipeline check on 64 examples
python scripts/train_transformer.py --smoke
```
Smoke runs use tiny subsets and a random encoder. They only prove the pipeline runs; they are not real models.

### 3. Get the data and build the splits
```bash
python scripts/download_data.py           # public datasets -> data/raw (git-ignored)
python scripts/prepare_data.py            # cleaning, dedup, splits -> data/processed
python scripts/freeze_splits.py --check   # verifies your splits match the published hashes in splits/
```
The first download pulls Dreaddit, the UMSAB tweets, the Hinglish YouTube corpus and BRIGHTER from their public sources. If the `--check` step reports a mismatch, your package versions probably differ from `requirements-lock.txt`.

### 4. Train the models
Pick what fits your hardware. Each step is independent of the others except the app, which loads whichever models exist.

| Model | Command | Time |
|---|---|---|
| TF-IDF + LogReg baseline | `python scripts/train_baseline.py` | minutes |
| Language-ID classifier | `python scripts/train_langid.py` | minutes |
| Multi-task BiLSTM | `python scripts/train_lstm.py` | about 15 min on 4 CPU cores |
| Multi-task MuRIL transformer | `python scripts/train_transformer.py` | hours on CPU, much faster on a GPU |

Minimum to see the app work: run the baseline and language-ID steps. The BiLSTM and transformer then appear as extra models. The first transformer run downloads `google/muril-base-cased` from Hugging Face (about 1 GB).

### 5. Start the app
```bash
uvicorn src.app.main:app_factory --factory --port 8000
```
Open http://127.0.0.1:8000. Models that have not been trained show as "not available" on the Models page, and API calls to them return an error instead of made-up output. Try the API directly:
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "I have been so stressed about exams lately", "model": "baseline"}'
```
Your text is processed in memory and is never logged (see [Privacy](#privacy)).

### 6. Reproduce the research experiments (optional)
```bash
python scripts/run_experiment.py configs/cpu/C_ST_sentiment_hingroberta.json --device cpu   # one config, 3 seeds
sh scripts/run_cpu_queue.sh                                                                 # the full CPU queue; resumable
python scripts/write_results_tables.py                                                      # rebuild the result tables from runs/
python scripts/evaluate.py                                                                  # metrics and calibration -> reports/
```
The full queue took days on a 4-core CPU. It saves a snapshot about every 10 minutes and resumes after an interruption, so you can stop and rerun it. Per-run configs, environments, histories and predictions from our runs are already committed under `runs/`, and the numbers in `reports/` can be traced to them. The paper is in `paper/research_paper.pdf`; rebuild it with `python paper/build_paper.py` (needs Playwright with Chromium).

### Common problems
* **`ModuleNotFoundError: src`**: run commands from the repository root.
* **Port 8000 busy**: use `--port 8001`.
* **Out of memory while training the transformer**: lower the batch size, e.g. `python scripts/train_transformer.py --batch-size 8`.
* **Dataset download fails**: rerun it; it needs access to huggingface.co and cs.columbia.edu. Behind a proxy, set `HTTPS_PROXY`.
* **Slow on CPU**: expected for MuRIL. Use Google Colab (`requirements-colab.txt`) or any machine with a CUDA GPU; the device is auto-detected.

## Quickstart (command reference)

```bash
pip install -r requirements.txt
python scripts/download_data.py && python scripts/prepare_data.py     # raw data -> data/raw, data/processed (git-ignored)
python scripts/train_baseline.py && python scripts/train_langid.py
python scripts/train_lstm.py                                          # ~15 min on 4 CPU cores
python scripts/train_transformer.py                                   # hours on CPU; use a GPU if you have one
python scripts/evaluate.py                                            # metrics, calibration, error analysis -> reports/
uvicorn src.app.main:app_factory --factory --port 8000                # http://127.0.0.1:8000
python -m pytest -q                                                   # no GPU, no downloads, no real training
python scripts/train_lstm.py --smoke                                  # 2-epoch pipeline check on a 64-example subset
python scripts/train_transformer.py --smoke                           # same, tiny random encoder (not a real model)
```
Smoke runs write to `checkpoints/smoke_*` and are not real models. Device is auto-detected (CUDA / MPS / CPU; `--device` to override). Configurable: `--epochs --batch-size --lr --max-len --patience --seed --limit --loss-weights`.

### API
```
GET  /            service info (JSON if Accept: application/json, else the dashboard)
GET  /health
POST /predict     {"text": "...", "model": "transformer|bilstm|baseline", "language": "auto|en-Latn|hi-Deva|hi-Latn", "explain": false}
POST /api/compare {"text": "..."}               (playground; BiLSTM vs transformer)
POST /api/batch   multipart CSV (text[,id,language]) → JSON, or format=csv
UI: / /analyze /playground /models /languages /errors /batch /health-ui /about
```
Response (abridged): `language, script, language_tag, sentiment, emotion, risk, risk_signal, risk_probability, risk_predictions[], model, inference_latency_ms, validated_for_language{}, safety{}`.

## Research upgrade: multi-seed, pre-registered experiments

**Conclusion and comparison with published research: [`reports/conclusion.md`](reports/conclusion.md)** · full write-up: [`reports/results_summary.md`](reports/results_summary.md) · audit: [`reports/research_audit.md`](reports/research_audit.md) · plan with verified literature: [`reports/experiment_plan.md`](reports/experiment_plan.md) · errors: [`reports/error_analysis.md`](reports/error_analysis.md) · reproducibility: [`reports/reproducibility.md`](reports/reproducibility.md) · paper outline: [`paper/outline.md`](paper/outline.md).

Test results, mean over 3 seeds, CPU fp32, paired bootstrap 95 % CIs:

| Finding | Result |
|---|---|
| **Hinglish sentiment** (major improvement 1): HingRoBERTa vs MuRIL | macro-F1 on `hi-Latn` **0.556 → 0.611, +0.056 [+0.032, +0.079]**; English unchanged (+0.020 [−0.002, +0.044]); English–Hinglish gap −27 % |
| **Stress detection** (major improvement 2): single-task vs multi-task MuRIL | PR-AUC **0.852 → 0.875, +0.023 [+0.008, +0.038]**; F1 unchanged; calibration worse (ECE 0.035 → 0.062) |
| Multi-task sharing | no measurable benefit on any task; negative transfer on stress ranking |
| Head+tail truncation for long posts (Sun et al., 2019) | did **not** help (validation macro-F1 −0.020 [−0.037, −0.003]) |
| Selective prediction (threshold chosen on validation) | stress macro-F1 0.783 → 0.828 while answering 83 % of inputs |
| Explanation faithfulness (IG deletion test) | faithful for English stress; only weakly faithful for Hinglish (58 % of examples) |
| Longer stress inputs (256 tokens, head or head+tail) | no gain on validation |

Not run (compute): LR grid, XLM-R control, transliteration/normalisation. Manual error taxonomy needs a human annotator. These are listed in `results_summary.md` §0. The served app still uses the original multi-task checkpoint; the numbers in the sections above are the original single-seed runs.

## Conclusion: what was improved, compared with published research

*Test sets, mean over 3 seeds, paired-bootstrap 95 % CIs. Published numbers were read from each paper's own result table and are context, not head-to-head comparisons, unless the test set is the same. Full version: [`reports/conclusion.md`](reports/conclusion.md).*

### What we improved

| # | Improvement | Before → after | Research basis |
|---|---|---|---|
| 1 | **Hinglish sentiment**: Hinglish-pre-trained encoder (HingRoBERTa) instead of MuRIL | `hi-Latn` macro-F1 **0.556 → 0.611** (+0.056 [+0.032, +0.079]); English unchanged; English–Hinglish gap −27 %; no test-set overlap with its pre-training corpus | Nayak & Joshi (2022); Khanuja et al. (2020) |
| 2 | **Stress detection**: dedicated single-task model instead of a shared multi-task model | PR-AUC **0.852 → 0.875** (+0.023 [+0.008, +0.038]); F1 unchanged; calibration worse (ECE 0.035 → 0.062) | Pruksachatkun et al. (2020); Liu et al. (2019) |
| 3 | **Reliability**: abstain on low-confidence inputs (threshold chosen on validation) | stress macro-F1 0.783 → **0.828** at 83 % coverage | Geifman & El-Yaniv (2017); Hendrycks & Gimpel (2017) |
| 4 | **Reproducibility**: 3 seeds, frozen splits, pinned packages, pre-registered test use | original run reproduced exactly | Dodge et al. (2020); Dror et al. (2018) |

Not improved (reported, not hidden): longer stress inputs (head+tail truncation, Sun et al. 2019: worse on validation); multi-task sharing (no gain on any task); emotion (below published baselines); explanations (faithful for English stress, weak for Hinglish).

### Comparison with published results

| Task | Published (source) | Ours | Reading |
|---|---|---|---|
| Stress, Dreaddit (same 715-post test set) | BERT-base F1 0.8065 (Turcan & McKeown 2019, Table 6) | F1 0.798 | **On par** with the original paper |
| Gain from Hinglish pre-training | +4.86 F1 over XLM-R on GLUECoS (Nayak & Joshi 2022) | +5.6 macro-F1 over MuRIL | **Effect replicated** on independent data |
| Romanized-Hindi tweets, UMSAB | XLM-Twitter 56.39 (Barbieri et al. 2022, Table 7) | HingRoBERTa 0.616 | About **5 points higher**; different training mix, so indicative only |
| English tweets, UMSAB | XLM-Twitter 70.63 (same table) | 0.704 | **On par** |
| Emotion, BRIGHTER English / Hindi | RoBERTa baseline 0.708 / 0.855 (SemEval-2025 Task 11, Table 2) | 0.639 / 0.770 | **Below** published baselines; future work |

### Agreement with the literature

| Prior claim | Our result |
|---|---|
| Pre-training data matters for code-mixed text (Nayak & Joshi 2022) | **Confirmed** |
| Multi-task benefit depends on the task pairing (Pruksachatkun et al. 2020) | **Consistent**: no positive transfer, and negative transfer on stress |
| Head+tail truncation helps long documents (Sun et al. 2019) | **Not reproduced** for Reddit posts |
| Seed variance is large (Dodge et al. 2020; Mosbach et al. 2021) | **Confirmed**: s.d. up to 0.044 |
| Explanations must be tested (Jacovi & Goldberg 2020; DeYoung et al. 2020) | Faithfulness turned out to be **language-dependent** |

### Research paper

Draft: [`paper/outline.md`](paper/outline.md). Working title: *"Closing the Romanized-Hindi Gap: Pre-training Data, Not Multi-Task Sharing, Drives Multilingual Sentiment and Stress-Language Classification."* All 35 references were verified against arXiv / ACL Anthology / Crossref ([`reports/experiment_plan.md`](reports/experiment_plan.md)).

## Repository layout
```
src/{config,preprocessing,langid,safety,privacy,explain,inference}.py
src/data/{download,prepare,datasets}.py      src/models/{baseline,multitask,tokenization,io}.py
src/training/{trainer,metrics,calibration,predict,run}.py      src/evaluation/{adapters,evaluate,report}.py
src/app/{main,reports}.py + templates/ static/      src/integrations/sarvam.py
scripts/  tests/  reports/  DATASET.md
```
Committed artefacts: `reports/` (metrics JSON, per-example predictions and `error_analysis.csv` with anonymised IDs and **no raw text**, figures). Not committed: data, checkpoints, logs.

## Limitations (please read)
* The transformer's best validation epoch was its last (3 of 3, time-capped on CPU, no early stop triggered), so it is probably **under-trained**; more epochs / tuning might change the comparison (and no hyper-parameter search was done for any model beyond TF-IDF's C).
* One training seed per model (bootstrap CIs cover test-sample noise only); small test sets (esp. Hinglish: 23 videos, single annotator).
* Risk = stress in English Reddit text; weak proxy for anything clinical; community/topic confound; performance on other populations unknown.
* Sentiment F1 is modest everywhere - three-class tweet sentiment with ~1.8 K examples per language is hard, and the Hinglish labels are noisy.
* Hindi (Devanagari) has emotion labels only; sentiment/risk on Devanagari input are zero-shot and unvalidated. Hinglish has sentiment labels only. Other Indic languages (Bengali, Tamil, Telugu, Kannada, Malayalam) are **not supported or evaluated**: script detection works, but nothing else is measured.
* Romanized-Hindi detection is a small classifier trained on one YouTube corpus (style confound; see `reports/metrics/langid.json`).
* **Code-mixed normalisation/transliteration was not implemented**, so no "original vs normalised input" comparison exists; code-mixed text is fed to the multilingual model as-is.
* Sarcasm cannot be identified from the available labels, so it is not analysed.
* Same-author leakage can't be excluded (no author metadata); no cross-dataset benchmark-contamination audit.
* The Sarvam adapter is unit-tested with mocked HTTP only - **never run against the live service**. TTS is not implemented.
* Dreaddit's license is **not verified**; tweets are subject to Twitter's terms. Check before any redistribution or commercial use.
