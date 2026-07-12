# Audio Fixed-Band Readout Experiment

Status: Ready to implement
Created: 2026-07-11T13:23:54Z
Source input: Use the completed audio J-lens to attempt the same fixed middle-layer working-space identification comparison previously run on text inputs.

## Objective

Run one preregistered, source-bound Modal experiment testing whether the exact completed audio J-lens recovers the canonical Anthropic evaluation intermediates from synthetic spoken versions of the same fixed prompt prefixes in L13–L31, relative to early, motor, logit, transposed, permuted, and label controls. Compare the result descriptively with the immutable common-item text evidence without changing the canonical text `no_band` verdict or claiming formal J-space, causal mediation, or a global workspace.

## Context

- Codebase findings:
  - `scripts/modal_workspace_eval.py` implements the sealed text experiment over L0–L12, L13–L31, and L32–L33. It preserves all item/layer/concept ranks and applies logit, transposed, permuted, and label-permutation controls. Its canonical result is `no_band`, not a validated text workspace band.
  - The text evaluator’s historical regional statistic minimizes rank over every layer in a region, giving the 19-layer candidate region more opportunities than the 13-layer early and two-layer motor regions. This experiment therefore uses an equal-per-layer regional mean as its primary localization statistic and reports the historical minimum only as secondary evidence.
  - `src/audiolens/audio_fitting.py::validate_completed_run` physically validates the exact audio corpus, five gates, checkpoints, stability report, and fp16 lens against the fp32 mean. The completed audio lens is an input artifact, not workspace evidence.
  - `src/audiolens/models/gemma4.py::prepare_audio` is intentionally fit-specific: audio-only framing, a 128-position limit, and stock-JLens valid-position rules. Evaluation needs a separate preparation path and must not weaken this fit contract.
  - `scripts/modal_audio_eval.py` demonstrates the efficient execution pattern—one multimodal forward followed by several lens readouts—but its RAVDESS labels and historical mixed lenses are not valid workspace comparators.
- History findings:
  - The historical mixed528 lens averaged text and audio Jacobians correctly but measured a different construct; its mood/prosody result was explicitly withdrawn as workspace evidence.
  - The canonical text run fixed L13–L31 before scoring and ended `no_band` because poetry candidate-minus-logit AUC was `-0.02016`, despite positive aggregate evidence. No alternate band was searched.
  - The exact audio run at commit `44ac8fa` completed 1,000 waveform prompts and produced lens SHA-256 `da0ccabf1ee14e4df060f97f31cf0132a0d3f6ed2cb45b6c77738693bc8f1aa9` under fit config `ee7cd4e42991fec5a00b4256ba466ff163ebd64fa22963334033066e7d531275`.
- Public prior art:
  - Anthropic separates J-lens fitting, J-space decomposition, and workspace-function evidence: https://transformer-circuits.pub/2026/workspace/index.html
  - The exact released evaluator semantics and fixed intermediates come from https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e and https://raw.githubusercontent.com/anthropics/jacobian-lens/581d398613e5602a5af361e1c34d3a92ea82ba8e/data/evaluations/README.md
  - Neuronpedia confirms the operational readout `residual @ J.T` followed by final normalization/unembedding, but its UI layer selection and filtered top-n output are exploratory rather than preregistered scientific defaults: https://github.com/hijohnnylin/neuronpedia
  - eSpeak NG 1.52.0 supports 35 of the 36 languages in the publication-prefix multilingual fixture; it has no Filipino/Tagalog voice: https://raw.githubusercontent.com/espeak-ng/espeak-ng/1.52.0/docs/languages.md
  - The independent calibration ASR is pinned to `openai/whisper-large-v3-turbo` revision `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`.
- Constraints:
  - All Gemma, processor, final-lens, and Whisper execution occurs on Modal GPUs. Local tests cover pure contracts and mocked orchestration only.
  - Confirmatory ranks may not be logged, checkpointed, committed, or exposed before the complete report validates. Infrastructure retry must restart scoring from item zero under the identical preregistration.
  - The experiment is conditional on one synthetic-speech engine, two variants of that engine, this model, this audio fit, and these fixture items. It cannot establish natural-speech generalization.

## Decisions

- Use spoken versions of the canonical fixture prefixes rather than a new FLEURS/language-ID task. This preserves the text experiment’s item identities, independently authored intermediates, layer hypothesis, and rank semantics.
- Use 260 publication-prefix items: association 50, multihop 50, multilingual 53, order-ops 55, and poetry 52. Exclude all 96 typo items because speech removes the orthographic manipulation. Exclude only `filipino-opposite-up` from multilingual before synthesis because the pinned engine has no Filipino/Tagalog voice. Recompute the descriptive text summary on exactly these 260 identities.
- Derive target-free spoken scripts mechanically from each canonical boundary: association uses its prompt; multihop, multilingual, and order-ops use the existing target-excluding prompt; poetry uses the prefix ending at the canonical newline boundary. Never add audio-specific labels or synonyms.
- Build eSpeak NG 1.52.0 from `https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.52.0.tar.gz`, SHA-256 `bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23`. Invoke it without a shell using literal argv `espeak-ng -s 180 -p 50 -a 100 -g 10 -v <code>+<variant> --stdout <script>` for variants `m1` and `f1`. Record the built binary hash and `--voices` output hash.
- Convert eSpeak’s mono PCM16 output to native mono 16 kHz with the locked SciPy/SoundFile environment: `scipy.signal.resample_poly` with `up=320`, `down=441`, then finite check, clip to `[-1,1]`, and SoundFile `PCM_16` output. Record source WAV, normalized WAV, decoded PCM, sample count, duration, argv, phoneme output, and all hashes.
- Freeze this multilingual eSpeak mapping: `arabic=ar`, `bengali=bn`, `bulgarian=bg`, `chinese=cmn`, `croatian=hr`, `czech=cs`, `danish=da`, `dutch=nl`, `estonian=et`, `finnish=fi`, `french=fr`, `german=de`, `greek=el`, `hebrew=he`, `hindi=hi`, `hungarian=hu`, `indonesian=id`, `irish=ga`, `italian=it`, `japanese=ja`, `korean=ko`, `norwegian=nb`, `persian=fa`, `polish=pl`, `portuguese=pt`, `romanian=ro`, `russian=ru`, `serbian=sr`, `slovak=sk`, `spanish=es`, `swedish=sv`, `thai=th`, `turkish=tr`, `ukrainian=uk`, `vietnamese=vi`. All non-multilingual fixtures use `en-us`.
- Treat `m1`/`f1` as same-engine robustness variants, not independent voices. Average them within item and retain each variant’s direction as a required point check.
- Before final-lens loading, calibrate TTS intelligibility with pinned Whisper on the shortest sealed confirmatory script per language and both variants, using the frozen language code, greedy decoding, and Unicode NFKC/casefold/punctuation-and-whitespace-normalized character error rate. The complete 70-cell calibration passes when macro CER is at most `0.35` and no cell exceeds `0.80`. Failure yields `inconclusive_synthetic_stimulus`; it does not permit item, language, or voice repair or exclusion.
- Add `prepare_audio_evaluation(processor, path, max_sequence_length=512)` without changing fit preparation. Use `add_generation_prompt=True`, require one contiguous audio-soft-token span, no truncation, complete sequence length at most 512, and exact recorded framing. Name the final audio position `last_processor_valid_audio_position`; do not call it semantically aligned. Define `response_position` as the final assistant-prefix token whose logits predict the first response token.
- Capture residuals L0–L34 in one batch-one deterministic forward. Apply lenses only at L0–L33; use L34 raw logits to define the actual unmodified next-token target and motor evidence.
- Score semantic intermediates at both `last_processor_valid_audio_position` and `response_position`. A candidate-region effect confined to the audio placeholder cannot validate. Motor evidence uses only the actual unmodified next-token argmax at `response_position`, never an expected answer.
- Use full-vocabulary ranks with the canonical strict-greater tie rule, tokenizer eligibility, allowed forms, k-grid `{1,2,5,10,20,50,100}`, and normalized log-k AUC. Do not reconstruct ranks from top-k output.
- The primary regional reducer is: allowed-form rank → per-item/per-layer pass@k curve and AUC → mean layer AUC inside each fixed region → mean two variants within item → mean items within distribution → equal mean of distributions. The historical min-over-region statistic is non-adjudicating secondary evidence.
- Construct the permuted-J control exactly as the canonical evaluator: source layer rotated by `+17 mod 34`, then output rows permuted by a single CPU `torch.randperm` at seed `2026070903`. Do not describe this as geometry- or spectrum-preserving. Transposed J remains `residual @ J`; candidate remains `residual @ J.T`.
- Use 10,000 paired original-item bootstraps at seed `2026070902`, with both variants bundled and the entire reducer recomputed. Use 10,000 same-distribution label-bundle permutations at seed `2026070901`, allowing canonical self-assignments and moving all concept labels for an item together across both variants, positions, and layers. Record arrays/digests and plus-one p-values.
- The label max-stat family is exactly candidate semantic AUC over four cells: `{all five distributions, four non-multilingual distributions} × {last audio position, response position}`. Each permutation replicate contributes the maximum of these four AUCs; the one-sided p-value for each observed cell uses that maximum-null. Motor and structural contrasts use paired bootstrap intervals, not the label null.
- Use deterministic inference: `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Torch import, batch size one, fixed item order, `model.eval()`, `torch.inference_mode()`, dropout/sampling disabled, eager attention, TF32 disabled, fixed seeds, and `torch.use_deterministic_algorithms(True)`. Abort rather than relax the policy if a kernel is nondeterministic.
- Report independent evidence fields—semantic-vs-logit, structural controls, fixed-region localization, response-boundary corroboration, motor transition, and common-item text comparison—but one narrow top-level synthetic-speech status.
- Top-level statuses are `validated_fixed_band_synthetic_speech_readout`, `no_fixed_band_synthetic_speech_readout`, `inconclusive_synthetic_stimulus`, and `invalid_protocol_or_artifact`. An interrupted/incomplete job remains unpublished/pending and is never a scientific status.

## Non-Goals

- Formal sparse nonnegative J-space decomposition, J-space coordinates/capacity, or gradient pursuit.
- Coordinate swaps, activation ablations, causal mediation, selective intervention, global-workspace, or consciousness claims.
- Natural-speech generalization, FLEURS, a new language-ID benchmark, matched-text reruns, or applying the text lens to audio.
- Refitting either canonical lens, comparing historical mixed528/RAVDESS artifacts, or selecting a different audio layer band.
- Broad evaluator-framework refactors, edits to `scripts/modal_workspace_eval.py`, publication, hub promotion, or model-family expansion.

## Implementation Plan

1. Add strict pure audio-evaluation contracts and statistics.
   - Files: new `src/audiolens/audio_workspace_eval.py`; new `tests/test_audio_workspace_eval.py`.
   - Change: define schema/version/kinds; immutable artifact/stimulus/preregistration/report identities; exact 260-item expected coordinates; TTS mapping/engine/config; overlap normalization; full-vocabulary rank reduction; layer-count-fair summaries; common-item text-summary recomputation from validated raw ranks; bundled bootstraps/permutations/max-stat; criteria/status precedence; canonical JSON/hash utilities; report recomputation and mutation rejection.
   - Change: physically validate the audio completed-run chain and lens bytes through `validate_completed_run`; validate the historical text report only through its current loader and bind it as external context. Do not edit or import private helpers from `scripts/modal_workspace_eval.py`; duplicate only leaf formulas and lock them with golden equivalence fixtures so its source digest remains unchanged.
   - Tests: exact fields/hashes; wrong bytes/profile/layers/dtype/source/config abort before model load; 260 coordinate reconstruction; typo/Filipino exclusions; overlap rejection using NFKC/casefold/punctuation/whitespace-normalized transcript content in addition to waveform/PCM hashes; rank ties/nonfinite values; region-width fairness and layer-duplication invariance; effective label shuffle; deterministic arrays/digests; threshold boundaries; every status and independent evidence field; strict report round-trip and resealed/unsealed mutation attacks.

2. Add evaluation-only Gemma preparation.
   - Files: `src/audiolens/models/base.py`, `src/audiolens/models/gemma4.py`, `src/audiolens/models/__init__.py`, `tests/test_models.py`.
   - Change: introduce an evaluation-prepared audio record containing opaque model inputs, exact input IDs, contiguous audio positions, `last_processor_valid_audio_position`, `response_position`, framing IDs, layout, and manifest fields. Add `prepare_audio_evaluation` using the existing audio-only message plus `add_generation_prompt=True`, 512-position no-truncation policy, explicit audio input validation, and no fit-valid-position restriction. Preserve `prepare_audio`, `GEMMA4_PROFILE.max_sequence_length`, and `GemmaPreparedAudioLensModel` byte-for-byte except imports/exports required by the new adjacent API.
   - Tests: exact batch-one inputs, required audio tensors, one contiguous span, assistant prefix, both positions, L34 availability, sequence boundary 512/513, no truncation, malformed/missing/nonfinite input rejection, and proof the existing fit API behavior remains unchanged.

3. Implement immutable stimulus staging and calibration.
   - Files: new `scripts/modal_audio_workspace_eval.py`; `tests/test_audio_workspace_eval.py`.
   - Change: build a Modal image that compiles the pinned eSpeak tarball after verifying its SHA; include locked SciPy/SoundFile/Whisper dependencies and exact package metadata. Fetch bounded fixture bytes using existing pinned URLs/hashes, validate the canonical text report and its eligibility/raw item identities, derive 260 scripts mechanically, synthesize both variants, normalize to 16 kHz PCM16, validate the processor layout without model weights, check fit-corpus overlap, and write one immutable content-addressed stimulus manifest plus WAVs.
   - Change: run the 70-cell independent Whisper calibration before final-lens load and seal its raw normalized transcripts, per-cell CERs, macro CER, model revision, inference config, and status. No score-bearing audio-lens output exists in this stage.
   - Tests: tarball/binary/voice-inventory identity; literal command argv without shell; all language mappings; deterministic path/order/hash; exact 520 WAVs and 70 calibration cells; transcript/PCM/waveform overlap; objective failures; CER normalization and thresholds; immutable-write/reuse behavior; no final-lens/model-scoring call before staging/calibration succeeds.

4. Implement deterministic, atomic H100 scoring.
   - Files: `scripts/modal_audio_workspace_eval.py`; `src/audiolens/audio_workspace_eval.py`; `tests/test_audio_workspace_eval.py`.
   - Change: accept only a preregistration path and SHA; rehash/revalidate all physical artifacts and source/runtime/config/stimulus/calibration identities before GPU scoring. Run one fixed-order, batch-one Gemma forward per 520 waveform observations; capture L0–L34 at both fixed positions; compute full-vocabulary candidate/logit/transposed/permuted ranks, same-distribution label-pool ranks, and response-position motor evidence without logging tokens/ranks/metrics.
   - Change: require exactly 520 item-variant records, each with two positions, L0–L33 lens/control records, L34 actual-output evidence, all eligible concept IDs, finite bounded ranks, and no duplicates/extras/missing coordinates. Recompute summaries, 10,000 bootstraps, 10,000 permutations, max-stat p-values, criteria, failed criteria, independent evidence fields, and top-level status solely from raw records.
   - Change: retain confirmatory data only in memory/job-local scratch. On failure, publish at most a score-free infrastructure receipt and leave preregistration pending. On success, validate the complete report, atomically write exactly one content-addressed report, then commit the volume. A completed report is reusable only after full validation; no partial resume or `limit` option exists.
   - Tests: planted positive/negative/control signals; audio-position-only signal cannot validate; non-multilingual aggregate cannot be driven by multilingual voice cues; point and confidence requirements; motor target uses L34 actual argmax; injected crash exposes no scores; same-SHA retry; config/source/runtime drift rejection; exact cardinality/completeness; no alternate choice accepted by schema.

5. Add a real-path nonconfirmatory smoke and independent validator.
   - Files: `scripts/modal_audio_workspace_eval.py`; `tests/test_audio_workspace_eval.py`.
   - Change: expose four explicit CLI modes: `--preregister`, `--smoke`, `--evaluate --preregistration <path> --sha256 <sha>`, and `--validate-report <path> --sha256 <sha>`. Smoke uses two non-publication fixture rows in a separate namespace, the real pinned Gemma processor/model on H100, a sacrificial deterministic matrix set rather than the final lens, both positions, L0–L34, all controls, duplicate identical inference, report construction, and atomic publish. It records only nonconfirmatory evidence.
   - Change: the independent Modal validator reloads the complete report, preregistration, stimuli, physical audio lens, completed audio run, and historical text report; reconstructs every expected coordinate and all metrics/statistics/adjudication; and reproduces the report SHA.
   - Tests: import-light deployment; fake end-to-end preregister/smoke/evaluate/validate orchestration; duplicate smoke inference equality; interruption behavior; exact CLI dispatch and no confirmatory debug/limit surface.

6. Execute the frozen experiment and analyze the result.
   - Files: no code changes during the confirmatory run.
   - Change: run preregistration and calibration, record their immutable path/SHA, run the real smoke, then run exactly one confirmatory H100 evaluation. Infrastructure retries may use only the exact same preregistration SHA and restart from item zero. Run the independent Modal validator after completion.
   - Change: compare the audio fair-region summaries with the mechanically recomputed 260-item text summaries while preserving the canonical text report’s `no_band` status. Interpret a positive result only as fixed-band intermediate readout on the exact eSpeak corpus; interpret a negative result as a complete failure of this fixed hypothesis, not absence of all audio workspace organization.
   - Tests: production evidence includes complete report identity, exact 520 records, calibration status, runtime/source/artifact identities, all criteria and failed criteria, and independently reproduced report hash.

## Validation

- `uv run pytest tests/test_audio_workspace_eval.py tests/test_models.py tests/test_workspace_eval.py -q` — all focused contract, adapter, orchestration, and canonical-regression tests pass.
- `uv run ruff check scripts/modal_audio_workspace_eval.py src/audiolens/audio_workspace_eval.py src/audiolens/models/base.py src/audiolens/models/gemma4.py tests/test_audio_workspace_eval.py tests/test_models.py` — zero diagnostics.
- `uv run python -m py_compile scripts/modal_audio_workspace_eval.py src/audiolens/audio_workspace_eval.py src/audiolens/models/base.py src/audiolens/models/gemma4.py` — successful compile/import.
- LSP diagnostics for each changed Python file — no errors.
- `uv run modal run scripts/modal_audio_workspace_eval.py --preregister` — emits an immutable preregistration/stimulus/calibration path and SHA, with 260 items, 520 WAV observations, 70 calibration cells, macro CER ≤0.35, and no cell CER >0.80; otherwise terminates `inconclusive_synthetic_stimulus` before final-lens load.
- `uv run modal run scripts/modal_audio_workspace_eval.py --smoke` — real H100 model/processor path completes on nonconfirmatory rows; duplicate inference produces identical IDs/ranks; both positions, L0–L34, controls, and atomic report path validate.
- `uv run modal run scripts/modal_audio_workspace_eval.py --evaluate --preregistration <path> --sha256 <sha>` — publishes either one complete 520-record scientific report or no score-bearing artifact.
- `uv run modal run scripts/modal_audio_workspace_eval.py --validate-report <path> --sha256 <sha>` — independently reproduces physical identities, metrics, criteria, status, and report SHA.

## Risks And Mitigations

- Synthetic speech may be unintelligible or encode engine-specific cues: bind exact engine/voices/audio, require the independent Whisper gate before final-lens loading, require a non-multilingual corroborating aggregate, and limit every claim to this synthetic corpus.
- The final audio placeholder may not be semantically aligned: name it honestly, score semantics at the assistant response boundary too, and require response-boundary corroboration.
- The candidate region has more layers: average per-layer AUC within each region and bootstrap the complete fair reducer; keep min-over-band non-adjudicating.
- Multilingual TTS quality varies and Filipino is unsupported: exclude the named Filipino row before synthesis, freeze mappings, gate all remaining language/variant cells collectively, and permit no repairs after outputs.
- Long spoken prompts are outside the 2–4-second fit distribution: use a frozen 512-position no-truncation envelope, report durations/fit-range strata descriptively, and make any overflow protocol-invalid rather than dropping items.
- Partial output could enable optional stopping: expose no score-bearing checkpoints/logs; restart exact-SHA infrastructure retries from zero; publish only after complete validation.
- Observational readout can be mistaken for workspace identification: enforce fixed status/claim flags rejecting J-space, causal, global-workspace, selectivity, and consciousness language.
- Editing the canonical text evaluator would invalidate its source-bound report: do not touch it; validate golden formula equivalence from the audio leaf implementation instead.

## Acceptance Criteria

- A content-addressed preregistration is sealed before final-audio-lens loading or confirmatory model scoring and binds exact physical artifacts, 260 scripts, 520 WAVs, 70 calibration cells, TTS/ASR/runtime/source identities, positions, regions, controls, seeds, reducers, thresholds, expected records, and allowed statuses.
- The independent Whisper calibration passes its frozen macro/per-cell CER rule; otherwise the terminal result is `inconclusive_synthetic_stimulus` and no lens scores exist.
- The real Modal smoke exercises the exact processor/model position path, L0–L34, deterministic duplicate inference, all controls, and atomic publication without using confirmatory items or the final lens.
- The final run uses the exact completed audio lens and sealed preregistration, deterministic batch-one inference, no metadata leakage, no truncation, and no partial score-bearing output.
- The report contains exactly 520 complete item-variant records, two declared positions, L0–L33 lens/control evidence, L34 actual-output evidence, all eligible labels, and no missing/duplicate/extra/nonfinite coordinates.
- Every summary, confidence interval, permutation statistic, criterion, failed criterion, common-item text summary, top-level status, and self-hash is reproducible solely from raw ranks and sealed identities.
- Every candidate-minus-control, candidate-minus-early, candidate-minus-motor, and motor-agreement comparison used for adjudication has its named paired lower 95% bound above zero; motor-minus-candidate JS has its paired upper 95% bound below zero. Bare point `>` never satisfies a criterion.
- The four-cell max-stat label family, same-distribution bundled shuffles, plus-one p-values, tie rules, and array digests exactly reproduce from seed `2026070901`.
- `validated_fixed_band_synthetic_speech_readout` is emitted only when: all protocol/calibration/completeness gates pass; five-distribution content-position candidate-minus-logit delta is at least 0.02 with lower 95% >0; its max-stat label p≤0.01; every distribution and both variants have nonnegative point deltas; candidate beats transposed/permuted with paired lower 95% >0; candidate beats early/motor with paired lower 95% >0; the four-distribution non-multilingual response-position candidate-minus-logit and candidate-minus-early/motor lower bounds are >0; and response-position motor actual-token agreement/divergence criteria pass with paired confidence. Any scientific failure after valid complete execution yields terminal `no_fixed_band_synthetic_speech_readout` without alternate search.
- A separate Modal invocation rehashes physical artifacts/stimuli/preregistration, reconstructs exact coordinates, recomputes the full report, and reproduces the canonical report SHA/status.
- Final reporting states the immutable text result remains `no_band`; the audio result is conditional on exact synthetic stimuli and artifacts; and no formal J-space, causal mediation, natural-speech workspace, global workspace, selectivity, consciousness, or universal layer boundary was established.

## Implementation Handoff

Implement `plans/20260711-132354-audio-fixed-band-readout.md` exactly in `../worktrees/audiolens-audio-workspace-eval`. Preserve `scripts/modal_workspace_eval.py` and all historical artifacts unchanged. Build the pure strict contract first, add the adjacent evaluation-only Gemma preparation, then implement pinned eSpeak staging/Whisper calibration and atomic deterministic Modal scoring. Use only nonconfirmatory rows and sacrificial matrices during smoke work. Run the final audio lens only after the preregistration and calibration artifacts are sealed, publish no partial scores, independently validate the complete report, and report the narrow fixed-band synthetic-speech outcome without alternate layer search or workspace/J-space overclaim.