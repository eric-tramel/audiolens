# audiolens 🔍🎧

J-lens for audio-input LLMs: fit the Jacobian lens on text and genuine audio
soft-token inputs, then read the workspace over audio positions — mood from
tone, not just transcript.

First target: `google/gemma-4-E2B-it` (Apache 2.0, native audio input,
~2B effective, runs locally). Neuronpedia's base-model `gemma-4-e2b` lens is
the cross-check reference.

```bash
uv run modal run scripts/modal_fit_lens.py            # legacy WikiText-only fit
modal volume get audiolens-vol lenses/gemma-4-E2B-it_jacobian_lens.pt lenses/
uv run python scripts/sanity_check.py                 # text-side sanity battery
uv run python scripts/audio_readout.py data/ravdess/Actor_01/*.wav  # local audio smoke
```

## Mixed WikiText + audio fit

The reproducible mixed experiment refits the existing 400-prompt WikiText
recipe under a locked environment and adds 128 processor-validated
LibriSpeech utterances: 64 from `clean/train.100`, 64 from `other/train.500`,
one 2–4 second 16 kHz clip per speaker. The waveform enters Gemma; its paired
transcript is retained in the committed manifest for provenance only.

Gemma's audio tower runs once per sample. The fitter captures the exact
audio-conditioned inputs entering the language decoder, then replays only the
decoder for JLens's gradient batches. It saves audio prefix lenses at 32, 64,
and 128 examples and merges the fp32 text400/audio128 means with stock
`JacobianLens.merge`. The mixed528 lens is therefore 24.24% audio by prompt
count, not by token count.

```bash
# Select/stage the fixed corpus, then commit the downloaded manifest.
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
settings. It never overwrites the generic text-only lens.

RAVDESS stays completely held out. The current evaluator loads the new pinned
text400 baseline and mixed528 candidate together and applies both to the same
captured residuals from one forward per clip:

```bash
uv run modal run scripts/modal_audio_eval.py \
  --baseline-lens /vol/lenses/<run>-text400.pt \
  --candidate-lens /vol/lenses/<run>-mixed528.pt \
  --limit 3
uv run modal run scripts/modal_audio_eval.py \
  --baseline-lens /vol/lenses/<run>-text400.pt \
  --candidate-lens /vol/lenses/<run>-mixed528.pt
modal volume get audiolens-vol eval/ravdess-paired-<digest>.jsonl eval/
modal volume get audiolens-vol eval/ravdess-paired-<digest>.json eval/
uv run python scripts/analyze_audio_eval.py eval/ravdess-paired-<digest>.jsonl
```

The tracked `eval/ravdess_gemma-4-E2B-it.jsonl` is legacy evidence only: it
predates per-cluster token-count normalization, contains the disabled
curiosity cluster, and has no lens/anchor/code fingerprint. Never compare or
resume it as if it were produced by the paired evaluator.

This first mixed fit changes both modality and corpus: without a matched
long-text control it cannot attribute a result uniquely to audio rather than
LibriSpeech content. A neutral or negative paired result is still a valid
experiment outcome and does not promote the mixed lens to MoodMic's default.

Dataset attribution: LibriSpeech is CC BY 4.0; WikiText is CC BY-SA 4.0;
RAVDESS is CC BY-NC-SA 4.0. Raw training/evaluation audio is not committed.
