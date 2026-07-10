# audiolens

Audiolens fits a context-general Jacobian lens for an audio-input language
model, applies it to text or audio-token residuals, and keeps the resulting
J-lens readout distinct from downstream mood/prosody summaries. The first
target is `google/gemma-4-E2B-it`.

## Fresh-clone setup

Prerequisites are Git, Python 3.12, [uv](https://docs.astral.sh/uv/), a Modal
account with the Modal CLI authenticated, and a Hugging Face account with
access to `google/gemma-4-E2B-it`. From a fresh clone:

```bash
uv sync --frozen
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=<your-hugging-face-token>
```

The `huggingface` Modal secret supplies model/download and opt-in publication
authentication inside Modal. Do not put a token on an Audiolens command line.
The shared `audiolens-vol` Modal volume is created by the scripts. Raw fit
audio, lenses, checkpoints, and evaluation results are regenerated into that
volume and are not shipped in this repository; only the audited LibriSpeech
provenance manifest is committed.

The canonical multilingual anchor vocabulary is package data, so installed
callers do not need a repository-relative path:

```python
from audiolens import load_default_anchors

anchors, colors = load_default_anchors()
```

Audit it against the target tokenizer with:

```bash
uv run python scripts/anchor_report.py
```

## Canonical text J-lens and workspace-like evaluation

The canonical lens follows Anthropic's released Jacobian estimator with one
independent transport matrix per source layer. For each valid source position,
it sums the effects on current and future valid targets, then takes the mean
over valid source positions within each prompt and an equal mean over
successful prompts. It fits exactly 1,000
raw-text, 128-token sequences using Neuronpedia's reproducible WikiText-103
concatenation and rechunking recipe. The fit is text-only by design: broad
context averaging estimates a general disposition to verbalize, while the same
decoder-space lens can later be applied at audio positions.

Run the full canonical fit:

```bash
uv run modal run scripts/modal_fit_lens.py
```

The command prints content-addressed manifest and lens paths on
`audiolens-vol`, plus exact `modal volume get` commands. A lower
`--n-prompts` value is available for integration smoke tests, but those
artifacts are explicitly noncanonical.

Evaluate the completed 1,000-prompt artifact with the six prompt distributions
released alongside Anthropic's technical report:

```bash
uv run modal run scripts/modal_workspace_eval.py \
  --fit-manifest /vol/runs/<fit-manifest>.json \
  --lens /vol/lenses/<lens>.pt
modal volume get audiolens-vol eval/<workspace-report>.json eval/
uv run python scripts/sanity_check.py \
  --report eval/<workspace-report>.json
```

The evaluator preserves every item, layer, and concept rank; compares the
J-lens with logit-lens, transposed, permuted, and label-permutation controls;
and tests the preregistered Gemma range L13-L31 against early L0-L12 and motor
L32-L33 behavior. It never searches for a better band after seeing results.
The report ends in either `validated` or `no_band`; both are complete outcomes.
Only `validated` supports the narrow claim that this fixed range is
workspace-like for this model and lens.

### Observed canonical result

The source-bound run from commit `1798a21` completed all 1,000 prompts:

- fit config `527a832964d8c952d6d8f69e5dfcd5d599c02d4703865bf97c31ea5e93cdf7ae`;
- fp16 lens SHA-256
  `d70b506928da17596376b248890d70b642bf37c0bb58fc0fcb22897f51bc27c7`;
- evaluation config
  `42d1bf037a64f86c0d249cc38c03b8eaf9d790bd48cea6bcedc3ad90868642e8`;
  and
- report SHA-256
  `09978e63328a03518cdd9f585f91afb5465faf0731246fbb128bdcf68ed3a769`.

The preregistered result is **`no_band`**. Eight of nine criteria passed, but
the candidate-minus-logit AUC was negative on the poetry distribution
(`-0.02016`), violating the requirement that every distribution be
nonnegative. The equal-distribution candidate-band AUC was `0.50648` versus
`0.27278` for the logit lens, and the paired-bootstrap lower 95% bound for
their difference was `0.20521`; those aggregate results do not override the
failed per-distribution gate. No alternate band was searched.

A separate transfer-only smoke applied this exact lens across L0-L33 and all
audio positions in two fixed RAVDESS clips (83 and 82 positions). Its complete
layer-by-position top-token record is retained on `audiolens-vol` as
`eval/gemma-4-E2B-it-audio-transfer-cbb2b2c45a5a2abe125d20ce1aafe384d715d97e2df114685e35b86c701091ec.json`.
It demonstrates that the corrected text lens executes on real audio
residuals; it is not a workspace-band, mood, or artifact-promotion result.

This does **not** compute a formal J-space component. The report defines
J-space components as sparse nonnegative combinations of J-lens vectors, but
neither Anthropic's released package nor Neuronpedia's public J-lens serving
path includes that gradient-pursuit decomposition.

## Mixed WikiText and LibriSpeech fit

The reproducible mixed experiment refits the existing 400-prompt WikiText
recipe under a locked environment and adds 128 processor-validated
LibriSpeech utterances: 64 from `clean/train.100` and 64 from
`other/train.500`, with one 2–4 second, 16 kHz clip per speaker. The waveform
enters Gemma; the paired transcript in the committed manifest is provenance,
not a separate training example.

Gemma's audio tower runs once per sample. The fitter captures the exact
audio-conditioned inputs entering the language decoder, then replays the
decoder for JLens gradient batches. It saves audio-prefix lenses at 32, 64,
and 128 examples and merges the fp32 text400/audio128 means with
`JacobianLens.merge`. The mixed528 lens is therefore 24.24% audio by prompt
count, not by token count.

```bash
# Select/stage the fixed corpus, download the manifest, and audit it locally.
uv run modal run scripts/modal_fit_mixed_lens.py --stage-only
modal volume get audiolens-vol manifests/librispeech_audio_fit_128.jsonl manifests/
uv run python scripts/modal_fit_mixed_lens.py \
  --audit-manifest manifests/librispeech_audio_fit_128.jsonl

# Gate the expensive run, then fit text400 + audio32/64/128 + mixed528.
uv run modal run scripts/modal_fit_mixed_lens.py --validate-replay-only
uv run modal run scripts/modal_fit_mixed_lens.py --audio-limit 1
uv run modal run scripts/modal_fit_mixed_lens.py
```

The fit is content-addressed by ordered corpus hashes, exact model/data
revisions, the frozen `uv.lock`, attention backend, code digest, and estimator
settings. It remains a separate historical audio-local/mixed-corpus experiment
and cannot select the canonical J-lens or its workspace-like layer range.

## Opt-in Hugging Face publication

Publication is never automatic. To fit and then immediately publish the
default `text400,mixed528` runtime lenses:

```bash
uv run modal run scripts/modal_fit_mixed_lens.py \
  --publish-to-hf <namespace/repository>
```

To publish selected runtime lenses from an already completed run without
rerunning fitting:

```bash
uv run modal run scripts/modal_fit_mixed_lens.py \
  --publish-to-hf <namespace/repository> \
  --publish-run <run-tag> \
  --publish-lenses text400,mixed528 \
  --publish-license cc-by-sa-4.0 \
  --publish-private
```

`--publish-lenses` is a comma-separated list. `--publish-license` defaults to
`cc-by-sa-4.0`; `--publish-private` is optional. Each uploaded model repository
contains only:

- the selected, validated `.pt` runtime lenses;
- a generated Hugging Face model card (`README.md`); and
- a sanitized `audiolens-run.json` limited to those lenses.

The publisher verifies completion, runtime-lens kind, file size, and SHA-256.
It cannot include fit checkpoints, manifests, datasets, evaluation outputs,
absolute Modal volume paths, or Modal image IDs. Consumers should pin both the
Hugging Face revision and a trusted expected checksum:

```python
from audiolens.hub import download_lens

path = download_lens(
    "<namespace/repository>",
    "<run-tag>-mixed528.pt",
    revision="<commit-sha>",
    expected_sha256="<64-hex-sha256>",
)
```

## Optional held-out RAVDESS evaluation

RAVDESS is **not training data**. The optional evaluator downloads the
upstream speech archive directly from the Zenodo record, verifies its pinned
SHA-256, and keeps it on the Modal volume. No RAVDESS audio or generated
RAVDESS evaluation result is included in the release tree, wheel, or
Hugging Face bundle.

```bash
uv run modal run scripts/modal_audio_eval.py \
  --baseline-lens /vol/lenses/<run-tag>-text400.pt \
  --candidate-lens /vol/lenses/<run-tag>-mixed528.pt \
  --limit 3
uv run modal run scripts/modal_audio_eval.py \
  --baseline-lens /vol/lenses/<run-tag>-text400.pt \
  --candidate-lens /vol/lenses/<run-tag>-mixed528.pt
modal volume get audiolens-vol eval/ravdess-paired-<digest>.jsonl eval/
modal volume get audiolens-vol eval/ravdess-paired-<digest>.json eval/
uv run python scripts/analyze_audio_eval.py eval/ravdess-paired-<digest>.jsonl
```

Downloaded evaluation outputs remain local under the ignored `eval/`
directory. RAVDESS is CC BY-NC-SA 4.0 and remains subject to its upstream
terms. The evaluator is a held-out measurement only.

This first mixed fit changes both modality and corpus: without a matched
long-text control it cannot attribute a result uniquely to audio rather than
LibriSpeech content. A neutral or negative paired result is still a valid
experiment outcome and does not make the mixed lens a downstream default.

## Licenses and provenance

Audiolens source code is MIT licensed; see `LICENSE`. The fitted lens files
have a separate publisher-selected artifact license, conservatively
`cc-by-sa-4.0` by default because WikiText is a training input. That artifact
license does not replace upstream terms.

- Gemma `google/gemma-4-E2B-it`: upstream model card and Apache-2.0 license.
- Jacobian Lens (`anthropics/jacobian-lens`): Apache-2.0.
- WikiText-103 (`Salesforce/wikitext`): Wikipedia-derived CC BY-SA/GFDL
  provenance; the exact pinned revision and ordered prompt hash are recorded.
- LibriSpeech (`openslr/librispeech_asr`): CC BY 4.0; the exact pinned revision
  and committed manifest hash are recorded.
- RAVDESS: CC BY-NC-SA 4.0, held out and never redistributed here.
