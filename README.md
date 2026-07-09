# audiolens 🔍🎧

J-lens for audio-input LLMs: fit the Jacobian lens on **text** (cheap, on
Modal), then read the workspace over **audio positions** — mood from tone,
not just transcript.

First target: `google/gemma-4-E2B-it` (Apache 2.0, native audio input,
~2B effective, runs locally). Neuronpedia's base-model `gemma-4-e2b` lens is
the cross-check reference.

```bash
uv run modal run scripts/modal_fit_lens.py            # fit on H100
modal volume get audiolens-vol lenses/gemma-4-E2B-it_jacobian_lens.pt lenses/
uv run python scripts/sanity_check.py                 # text-side sanity battery
uv run python scripts/audio_readout.py data/ravdess/Actor_01/*.wav  # local audio smoke
```

RAVDESS prosody experiment (fixed sentences, acted emotions — does the ring
track tone with words held constant?):

```bash
uv run modal run scripts/modal_audio_eval.py --limit 3   # in-cloud smoke first
uv run modal run scripts/modal_audio_eval.py             # full 1440-clip corpus
modal volume get audiolens-vol eval/ravdess_gemma-4-E2B-it.jsonl eval/
uv run python scripts/analyze_audio_eval.py eval/ravdess_gemma-4-E2B-it.jsonl
```

Results JSONL lands in `eval/` (tracked, for repeatability). Roadmap: text-side
sanity ✓ → audio-position readout ✓ (text-fit lens reads over audio tokens;
prosody moves the readout) → full RAVDESS statistics (this eval).
