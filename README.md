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

The fully source-bound run from commit `eafe7cc` completed all 1,000 prompts:

- fit config `7464e4104cde5f7d12110ed8b8f4fa90b0819edd92096a2e1bc8b8d826294568`;
- fp16 lens SHA-256
  `a0b0e6fb29c1eaf5c3d02e05a591e08cc9d43ef916020c5fcdb76c325b08529c`;
- evaluation config
  `6284af5fa0f5ae5753cbc8831e3cfed7be111412824dc16a1c4218f70fca393a`;
  and
- report SHA-256
  `16413a02d95aeada2e7a36a70fe100cc61c1d7138620ae0fdd1f0fdcd4791028`.

The preregistered result is **`no_band`**. Eight of nine criteria passed, but
the candidate-minus-logit AUC was negative on the poetry distribution
(`-0.02016`), violating the requirement that every distribution be
nonnegative. The equal-distribution candidate-band AUC was `0.50648` versus
`0.27278` for the logit lens, and the paired-bootstrap lower 95% bound for
their difference was `0.20521`; those aggregate results do not override the
failed per-distribution gate. No alternate band was searched.

A separate transfer-only smoke applied this exact lens on a Modal H100 across
L0-L33 and all audio positions in two fixed RAVDESS clips (83 and 82
positions). Its complete layer-by-position top-token record is retained on
`audiolens-vol` as
`eval/gemma-4-E2B-it-audio-transfer-196110a9f3b8533da8c68febcf1962b48b6ac5a7ece8a397f3bff8a088a2cceb.json`.
It demonstrates that the corrected text lens executes on real audio
residuals; it is not a workspace-band, mood, or artifact-promotion result.

This does **not** compute a formal J-space component. The report defines
J-space components as sparse nonnegative combinations of J-lens vectors, but
neither Anthropic's released package nor Neuronpedia's public J-lens serving
path includes that gradient-pursuit decomposition.

## Source-bound audio J-lens

The canonical audio fit is a separate, waveform-only application of Anthropic's
released estimator. It selects exactly 1,000 LibriSpeech utterances: 500 from
`clean/train.360` and 500 from `other/train.500`, alternating strata in fit
order with one 2–4 second, native-mono 16 kHz clip per globally unique speaker.
Selection uses SHA-256 speaker and utterance ranks over the complete pinned
metadata pool, not stream order. The exact source transcript is retained and
hashed as pair provenance but is never passed to `jlens.fit`.

Every accepted waveform is decoded, processed through Gemma's pinned
audio-to-decoder path, and admitted only when its audio span is contiguous and
has at least one stock-valid source position after `skip_first=16`. The fit then
uses the same stock prompt-level estimator as the text lens: one independent
fp32 running-sum matrix per L0–L33 source layer, target L34, and equal weight per
successful waveform. No audio/text merge, semantic evaluator, layer-band
selection, publication, or workspace claim is part of this run.

Run the source, selection, restore, replay, smoke-resume, and full-fit gates in
order:

```bash
uv run modal run scripts/modal_fit_audio_lens.py --rank-source-only
uv run modal run scripts/modal_fit_audio_lens.py --stage-corpus-only \
  --source-pool-sha256 <source-pool-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --selection-replay-only \
  --source-pool-sha256 <source-pool-sha256> \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --restore-source-only \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --preflight-only \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --replay-parity-only \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --smoke-only \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
uv run modal run scripts/modal_fit_audio_lens.py --fit \
  --ordered-corpus-sha256 <ordered-corpus-sha256>
```

The selection replay, source restore, all-1,000 processor replay, two-row
decoder replay, and separately persisted 10→20 resume each seal a
content-addressed gate bound to the immutable fit configuration. The production
fit will not allocate an H100 until all five gates validate. The full run
preserves an immutable 500-example prefix snapshot and emits only two per-layer
convergence diagnostics: identity-centered split-half cosine and
first-half-to-full relative L2. The final fp16 lens is validated
matrix-for-matrix against the 1,000-example fp32 running sums. Corpus, source
pool, attempt ledger, checkpoint, gate, stability report, and lens identities
are all content-bound in the completed run manifest on `audiolens-vol`.

This contract is schema v2. It intentionally rejects partial or completed v1
runs rather than reusing artifacts that predate checkpoint identity stamps and
durable execution gates.

### Observed source-bound audio fit

The complete H100 run passed all five execution gates and fitted all 1,000
waveforms:

- source pool: 252,702 pinned LibriSpeech rows (104,014 clean, 148,688 other);
- selected corpus: 500 clean + 500 other clips, 30,848 audited attempts,
  ordered-corpus SHA-256
  `1f194155b47db48726878c6026c507ca01b65d3e50736eab5dbea2c1bbc8c966`;
- implementation source SHA-256:
  `ccdb307a6ea9bf90aa1f8fae075dd9ec54b7ffb6c5738276cd896b6f96086c80`;
- fit config:
  `ee7cd4e42991fec5a00b4256ba466ff163ebd64fa22963334033066e7d531275`;
- final fp32 checkpoint: `n_done=next_idx=1000`, SHA-256
  `e764083944c3022b4c58f4ee58747691691cedab9c8fd2dd33f57e8006d8a724`;
- validated fp16 lens: 160,439,813 bytes, SHA-256
  `da0ccabf1ee14e4df060f97f31cf0132a0d3f6ed2cb45b6c77738693bc8f1aa9`;
- stability report:
  `b1ddd375af8029cc529ae5f6ab0e3ad7c9d2d179ef2097e8adff28791e27d6f7`.

Across L0–L33, the median identity-centered split-half cosine was `0.999894`
and the median first-half-to-full relative L2 was `0.02233`. The least stable
direction was L3 (`0.989104` cosine); the largest relative change was L0
(`0.08746`). Descriptively, first-half-to-full change decreased with depth:
mean relative L2 was `0.05540` over L0–L12, `0.02085` over L13–L22, and
`0.01243` over L23–L33.

These measurements establish a reproducible audio-conditioned J-lens. The
preregistered protocol does not define a convergence threshold, so these
descriptive diagnostics do not by themselves establish convergence, identify
a formal J-space component, or support a workspace-region claim.

## Fixed-band synthetic-speech evaluation

The audio evaluator attempts the same fixed L13–L31 hypothesis and frozen
publication-item concepts as the canonical text evaluator, without selecting a
new range after seeing audio results. It renders two pinned eSpeak 1.52.0 voices
for each of 259 eligible items (518 observations), calibrates one shortest
multilingual prompt per supported language with pinned
`openai/whisper-large-v3-turbo`, and only then permits the final audio lens to
score every L0–L33 layer. The final report is atomic: failed calibration or
scoring publishes no partial confirmatory scores.
This 259-item surface supersedes the initial 260-item feasibility target.
`multilingual/filipino-opposite-up` is excluded because eSpeak 1.52.0 has no
Filipino/Tagalog voice; `multilingual/irish-opposite-big` is excluded because
the pinned Whisper model rejects its `ga` language code. Both exclusions were
made before the source-final preregistration was sealed and before any
confirmatory scoring; neither was selected from lens results or calibration
error rates.

Run the nonconfirmatory smoke, seal the stimuli and calibration, and request the
confirmatory evaluation:

```bash
uv run modal run scripts/modal_audio_workspace_eval.py --smoke
uv run modal run scripts/modal_audio_workspace_eval.py --preregister
uv run modal run scripts/modal_audio_workspace_eval.py --evaluate \
  --preregistration /vol/audio-workspace-eval/preregistrations/<sha256>.json \
  --sha256 <sha256>
```

### Observed synthetic-stimulus result

The real H100 smoke completed both sacrificial observations across every
position, layer, candidate readout, and control. Its content-addressed report is
`audio-workspace-eval/smoke/3738d5e7d2e712cd4635f606ed59397359c589e817e44b3f865431c08cc1ac91.json`
on `audiolens-vol`.

The sealed full preregistration uses evaluator source SHA-256
`189dbf36fafa17994ced984aedddb5d000077601cb91a1e4ec20d6e4589b9e59`
and is retained as
`audio-workspace-eval/preregistrations/d2f0a3d82f10a3c5f18450d86290f2ce18fd724ed919bbde74211cf63cc0576a.json`.
Independent ASR calibration failed before the lens was loaded: macro character
error rate was `0.51363` against the preregistered maximum `0.35`, maximum cell
error was `2.52941` against `0.80`, and 13 of 68 cells exceeded the cell limit.
The failures covered Bengali, Hindi, Japanese, Korean, Russian, Serbian, and
Ukrainian stimuli. The evaluator therefore rejected the confirmatory request
with `confirmatory evaluation requires passed calibration` and emitted no
fixed-band comparison report. The scientifically valid result is
`inconclusive_synthetic_stimulus`: these eSpeak renderings are not sufficiently
intelligible to test whether the source-bound audio J-lens recovers the frozen
intermediate labels, so they provide neither positive nor negative evidence for
an audio workspace-like band.

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
