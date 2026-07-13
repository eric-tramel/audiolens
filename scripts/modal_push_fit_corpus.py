"""Publish the sealed audio-lens fit corpus to a private Hugging Face dataset.

Reads the completed audio fit's 1,000-row corpus ledger and content-addressed
FLAC blobs from `audiolens-vol`, verifies every byte against the sealed
hashes, and pushes one config of the private homebase dataset. The dataset is
private: LibriSpeech redistribution terms are satisfied by CC-BY-4.0 anyway,
but nothing here is published.

    uv run modal run scripts/modal_push_fit_corpus.py \
        --repo eric-tramel/audiolens-audio-corpus
"""

from __future__ import annotations

import modal

VOL_MOUNT = "/vol"
FIT_CONFIG_SHA256 = "ee7cd4e42991fec5a00b4256ba466ff163ebd64fa22963334033066e7d531275"
RUN_PATH = f"{VOL_MOUNT}/audio-fit-runs/{FIT_CONFIG_SHA256}/run.json"
CONFIG_NAME = "librispeech-jlens-1000"
LENS_SHA256 = "da0ccabf1ee14e4df060f97f31cf0132a0d3f6ed2cb45b6c77738693bc8f1aa9"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("datasets", "soundfile", "huggingface_hub", "torch", "torchcodec")
)

app = modal.App("audiolens-push-fit-corpus", image=image)
vol = modal.Volume.from_name("audiolens-vol", create_if_missing=False)

CARD = f"""---
license: cc-by-4.0
language: [en]
pretty_name: AudioLens training audio corpus
---

# AudioLens training audio corpus (private)

Homebase for the audio material used to fit AudioLens Jacobian lenses for
`google/gemma-4-E2B-it`. Private working data; not for redistribution.

## Configs

- `{CONFIG_NAME}` (split `fit`): the exact sealed corpus of the completed
  waveform-only audio J-lens fit — 1,000 LibriSpeech utterances (500
  `clean/train.360`, 500 `other/train.500`), one clip per globally unique
  speaker, native 16 kHz mono FLAC, in sealed fit order. LibriSpeech is
  CC-BY-4.0 (Panayotov et al.). Provenance: fit config
  `{FIT_CONFIG_SHA256}`, fitted lens SHA-256 `{LENS_SHA256}`. The `caption`
  column is the source transcript; for the sealed lens it was provenance
  only and was never passed to `jlens.fit`.
- Future configs will add openly licensed non-speech sound/event audio with
  captions (FSD50K, WavCaps-derived material, self-produced captioned
  recordings) toward one generic audio lens. See
  `plans/20260711-generic-audio-lens.md` in the AudioLens repository.

## Schema

`audio` (16 kHz mono), `caption`, `modality`, `source_dataset`,
`source_split`, `source_id`, `pair_id`, `speaker_id`, `chapter_id`,
`selection_index`, `stratum_index`, `duration_seconds`, `audio_sha256`,
`decoded_pcm_sha256`, `transcript_sha256`.
"""


@app.function(
    timeout=2 * 60 * 60, volumes={VOL_MOUNT: vol}, secrets=[modal.Secret.from_name("huggingface")]
)
def push(repo: str) -> str:
    import hashlib
    import json
    import os
    import pathlib

    from datasets import Audio, Dataset, Features, Value
    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    run = json.loads(pathlib.Path(RUN_PATH).read_text())
    corpus = run["corpus"]
    rows_path = pathlib.Path(VOL_MOUNT) / corpus["rows_path"]
    rows_bytes = rows_path.read_bytes()
    if hashlib.sha256(rows_bytes).hexdigest() != corpus["rows_sha256"]:
        raise RuntimeError("corpus ledger bytes changed")
    rows = [json.loads(line) for line in rows_bytes.splitlines()]
    if len(rows) != 1_000:
        raise RuntimeError(f"expected 1,000 rows, found {len(rows)}")

    examples = []
    for row in rows:
        blob = pathlib.Path(VOL_MOUNT) / row["volume_path"]
        payload = blob.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["audio_sha256"]:
            raise RuntimeError(f"{row['source_id']}: sealed audio bytes changed")
        examples.append(
            {
                "audio": {"bytes": payload, "path": f"{row['source_id']}.flac"},
                "caption": row["transcript"],
                "modality": "speech",
                "source_dataset": row["dataset"],
                "source_split": row["split"],
                "source_id": row["source_id"],
                "pair_id": row["pair_id"],
                "speaker_id": int(row["speaker_id"]),
                "chapter_id": int(row["chapter_id"]),
                "selection_index": int(row["selection_index"]),
                "stratum_index": int(row["stratum_index"]),
                "duration_seconds": float(row["duration_seconds"]),
                "audio_sha256": row["audio_sha256"],
                "decoded_pcm_sha256": row["decoded_pcm_sha256"],
                "transcript_sha256": row["transcript_sha256"],
            }
        )
    features = Features(
        {
            "audio": Audio(sampling_rate=16_000),
            "caption": Value("string"),
            "modality": Value("string"),
            "source_dataset": Value("string"),
            "source_split": Value("string"),
            "source_id": Value("string"),
            "pair_id": Value("string"),
            "speaker_id": Value("int64"),
            "chapter_id": Value("int64"),
            "selection_index": Value("int64"),
            "stratum_index": Value("int64"),
            "duration_seconds": Value("float64"),
            "audio_sha256": Value("string"),
            "decoded_pcm_sha256": Value("string"),
            "transcript_sha256": Value("string"),
        }
    )
    dataset = Dataset.from_list(examples, features=features)
    dataset.push_to_hub(
        repo,
        config_name=CONFIG_NAME,
        split="fit",
        private=True,
        token=token,
        commit_message=f"Sealed audio-lens fit corpus (fit config {FIT_CONFIG_SHA256[:12]})",
    )
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message="Document homebase layout and sealed-corpus provenance",
    )
    info = api.dataset_info(repo)
    return json.dumps(
        {
            "repo": repo,
            "private": info.private,
            "rows": len(examples),
            "config": CONFIG_NAME,
        }
    )


@app.local_entrypoint()
def main(repo: str = "eric-tramel/audiolens-audio-corpus"):
    print(push.remote(repo))
