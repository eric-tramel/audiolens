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
```

Roadmap: text-side sanity (top-k readout, correlation vs the base lens,
mood-anchor vetting on the Gemma tokenizer) → audio-position readout →
RAVDESS prosody experiment (fixed sentences, acted emotions: does the ring
track tone with words held constant?).
