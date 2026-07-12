# Generic Audio Lens

Status: Direction approved, not yet implemented
Created: 2026-07-11
Supersedes the speech-only audio lens as the forward direction; the completed
fixed-band experiment suite (see `20260711-132354-audio-fixed-band-readout.md`
and `20260711-openai-tts-stimulus-amendment.md`) is the frozen baseline.

## Direction

One audio lens over a broad acoustic distribution — speech plus environmental
and event audio (animals, water, weather, machines, human non-speech) — not a
family of per-domain lenses. Specialist lenses defeat the point of probing a
foundation model; the object of study is the model's general audio
understanding and the concepts it elicits in the candidate J-space of an
audio+text LLM. The bet is deliberately bitter-lesson shaped: keep the
estimator stock and simple, widen and scale the data.

## What carries over unchanged

- The fit construction: Anthropic's stock prompt-level Jacobian estimator,
  one fp32 running-sum matrix per source layer L0–L33 to target L34, equal
  weight per accepted waveform, waveform-only input.
- Caption/label discipline: every clip carries a caption or class label as
  hashed provenance only; text is never passed to the fit.
- The provenance machinery: deterministic complete-pool selection, source
  census, ordered-corpus identity, execution gates, durable checkpoints,
  immutable snapshots, fp16 distribution with matrix-equivalence checks,
  stability diagnostics.
- The evaluation philosophy: preregistered, fail-closed, controls
  (logit/transposed/permuted), no post-hoc band search.

## Corpus sketch (to be pinned)

- Speech stratum: retain the LibriSpeech recipe (unique speakers, 2–4 s,
  native 16 kHz mono).
- Sound-event strata: openly licensed effect/event libraries — FSD50K
  (CC Freesound, ~51k clips, tagged), ESC-50 (hold out for evaluation, not
  fit), plus curated open effect libraries as needed. Self-produced
  "captioned audio" (recorded or assembled clips with authored captions) is
  in scope for coverage gaps.
- Balance and dedup analogs to the speaker-uniqueness rule: per-uploader /
  per-recording uniqueness, per-class caps, deterministic hashed ranking over
  a complete pinned metadata pool.
- Licensing gate: CC-licensed sources only; provenance manifest committed as
  with LibriSpeech.

## Evaluation program

1. Behavioral gate first (cheap): does the model itself name held-out sound
   events ("What do you hear?") — the analog of the TTS intelligibility gate.
   No lens claims are meaningful for sounds the model cannot identify.
2. Sound-concept readout battery: held-out event clips (e.g., ESC-50) with
   class-name concept eligibility and the existing rank/AUC machinery, at the
   last audio position and response boundary, against all controls.
3. Comparability rerun: the frozen fixed-band synthetic-speech evaluation,
   re-executed with the generic lens, to measure what broadening the fit
   distribution does to the speech-side readout.
4. Eventually: the sparse J-space decomposition (out of scope until now),
   asking whether heard concepts and read concepts share components — the
   real cross-modal workspace question.

## Non-Goals

- Per-domain or per-kind specialist lenses.
- Mood/prosody claims (RAVDESS remains held-out evaluation material only).
- Reopening the completed fixed-band verdicts; they stand as the baseline.
