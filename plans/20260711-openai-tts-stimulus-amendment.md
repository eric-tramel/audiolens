# OpenAI TTS Stimulus Amendment

Status: Implemented
Created: 2026-07-11
Amends: `plans/20260711-132354-audio-fixed-band-readout.md`

## Why

The eSpeak NG 1.52.0 stimulus attempt terminated `inconclusive_synthetic_stimulus`:
13 of 68 pinned-Whisper calibration cells exceeded the per-cell CER limit
(Bengali, Hindi, Japanese, Korean, Russian, Serbian, Ukrainian), macro CER
0.51363 against the 0.35 maximum. Listening confirmed these are not marginal
renderings: eSpeak has no reading frontend for CJK/Indic scripts (it announces
"Chinese letter" per kanji in English) and verbalizes quotation marks in some
Slavic languages. A quality probe of the same failing scripts through OpenAI
`gpt-4o-mini-tts`, round-tripped through Whisper and scored with the
evaluator's own CER code, produced intelligible speech in all seven languages;
residual nonzero CERs were metric artifacts (digit-vs-word spelling, Serbian
Cyrillic/Latin script, an API-only language-hint restriction absent from the
pinned HF Whisper).

## Decisions

- Replace eSpeak with OpenAI `gpt-4o-mini-tts`, voices `onyx` and `nova` as the
  two same-engine robustness variants. All 259 item identities, the fixed
  L13–L31 hypothesis, positions, controls, seeds, reducers, and thresholds are
  unchanged. Both prior item exclusions are retained so the item surface is
  identical across engine attempts.
- Provenance moves from regenerable-from-pinned-binary to sealed bytes:
  synthesis happens once, locally (`scripts/synthesize_audio_stimuli.py`);
  every API response is rewritten into a canonical mono 24 kHz PCM16 RIFF
  container (the API emits a streaming header with a placeholder data size),
  hashed, and bound into a content-addressed recipe under
  `audio-workspace-eval/source-stimuli/<recipe-sha256>/`. Modal staging
  verifies recipe seal, engine identity, coordinate identity, and per-file
  bytes before deterministic 24 kHz → 16 kHz `resample_poly` (up 2, down 3)
  normalization. Report validation independently re-derives every normalized
  WAV from the sealed source bytes (replacing the eSpeak re-synthesis check,
  which a nondeterministic API cannot support).
- Spoken input policy: double quotes are stripped and whitespace collapsed
  before synthesis (`tts_input`). The publication prompts end in a dangling
  opening quote with no spoken form that silences the engine. Scripts,
  references, and eligibility are unchanged; CER normalization already strips
  punctuation, so calibration references are unaffected. The spoken input and
  its hash are recorded per observation.
- Calibration metric-validity fixes, sealed before the new preregistration:
  per language, the calibration cell prefers the shortest non-number item
  (digit-vs-word ASR spellings measure the metric, not intelligibility), and
  Serbian is compared after Vuk–Gaj Cyrillic→Latin transliteration of both
  sides, because the pinned ASR emits Serbian in Latin script. Thresholds are
  unchanged (macro ≤ 0.35, cell ≤ 0.80, 68 cells).
- The nonconfirmatory smoke consumes two sealed non-publication WAVs from the
  same recipe instead of synthesizing live; `--smoke` and `--preregister` take
  `--tts-recipe <sha256>`.
- The synthesis script is added to the bound source relatives, so the
  generator of the sealed input is part of the source identity.

## Calibration ASR re-pin (same day)

The first OpenAI-stimulus preregistration attempt (recipe `7f5db5a2…`, sealed
preregistration `cd313b86…`) still terminated `inconclusive_synthetic_stimulus`
on exactly one cell: `bn/nova` at CER 12.88, where the pinned
`whisper-large-v3-turbo` fell into a token repetition loop emitting Tibetan
script. Macro CER passed at 0.2376 and all 66 non-Bengali cells were under the
limit. A Modal probe of ten fresh Bengali renders showed the turbo model
loops or romanizes short Bengali regardless of render — a property of the
distilled calibration model, not of the stimuli (the same clips round-trip
phonetically correctly through whisper-1).

The calibration ASR is therefore re-pinned to the full `openai/whisper-large-v3`
at revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, the model turbo was
distilled from. Under it no probe render loops. Because large-v3 still
romanizes some Bengali renders stochastically, the Bengali item's `onyx`
render was pre-screened: the sealed clip is the probe candidate that decodes
in Bengali script (CER 0.118); the `nova` render is unchanged (CER 0.529).
This pre-screen is stimulus preparation performed openly before sealing; the
recipe (`74c784de…`) seals the final bytes and the binding gate remains the
sealed calibration itself. Threshold values are unchanged.

## Non-Goals

Unchanged from the original plan: no formal J-space decomposition, no causal or
global-workspace claims, no natural-speech generalization, no alternate band
search, no refitting of either lens, and no edits to the canonical text
evaluator.
