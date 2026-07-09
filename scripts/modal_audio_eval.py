"""Fresh paired RAVDESS evaluation for text400 and mixed528 lenses.

Both lenses read the same captured residuals from one Gemma forward per clip.
Outputs are content-addressed by model, code, lens, anchor, and scoring identity;
the legacy ``eval/ravdess_gemma-4-E2B-it.jsonl`` is never resumed or changed.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
MODEL_ID = "google/gemma-4-E2B-it"
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAVDESS_SHA256 = "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657"
RAVDESS_N_CLIPS = 1440
READ_LAYERS = [23, 29, 33]
TOPK = 10
SCHEMA_VERSION = 2


def _source_digest() -> str:
    relatives = (
        "uv.lock",
        "anchors/multilingual.yaml",
        "src/audiolens/__init__.py",
        "src/audiolens/fitting.py",
        "scripts/modal_audio_eval.py",
        "scripts/analyze_audio_eval.py",
    )
    if all((REPO_ROOT / relative).is_file() for relative in relatives):
        digest = hashlib.sha256()
        for relative in relatives:
            digest.update(relative.encode())
            digest.update((REPO_ROOT / relative).read_bytes())
        return digest.hexdigest()
    injected = os.environ.get("AUDIOLENS_EVAL_SOURCE_DIGEST")
    if injected:
        return injected
    raise RuntimeError("cannot determine evaluation source digest")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        injected = os.environ.get("AUDIOLENS_GIT_REVISION")
        if injected:
            return injected
        raise RuntimeError("cannot determine source Git revision") from None


GIT_REVISION = _git_revision()
SOURCE_DIGEST = _source_digest()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .uv_sync(
        uv_project_dir=str(REPO_ROOT), frozen=True, groups=["fit"], gpu="H100"
    )
    .env(
        {
            "HF_HOME": f"{VOL_MOUNT}/hf",
            "AUDIOLENS_GIT_REVISION": GIT_REVISION,
            "AUDIOLENS_EVAL_SOURCE_DIGEST": SOURCE_DIGEST,
        }
    )
    .add_local_python_source("audiolens")
    .add_local_dir(str(REPO_ROOT / "anchors"), remote_path="/root/anchors")
)

app = modal.App("audiolens-paired-audio-eval", image=image)
vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)


def _stage_ravdess() -> str:
    import hashlib
    import os
    import shutil
    import urllib.request
    import zipfile

    root = pathlib.Path(f"{VOL_MOUNT}/ravdess")
    zip_path = pathlib.Path(f"{VOL_MOUNT}/ravdess_speech.zip")
    if not zip_path.exists():
        tmp_zip = zip_path.with_suffix(".part")
        if tmp_zip.exists() or tmp_zip.is_symlink():
            tmp_zip.unlink()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(tmp_zip, flags, 0o600)
        try:
            with urllib.request.urlopen(RAVDESS_URL) as response, os.fdopen(
                descriptor, "wb"
            ) as target:
                descriptor = -1
                shutil.copyfileobj(response, target)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        tmp_zip.replace(zip_path)

    digest = hashlib.sha256()
    with open(zip_path, "rb") as archive_file:
        for chunk in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != RAVDESS_SHA256:
        zip_path.unlink()
        raise RuntimeError(f"RAVDESS zip sha256 mismatch: {digest.hexdigest()}")

    expected: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = []
        total_size = 0
        for info in archive.infolist():
            member = pathlib.PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if member.is_absolute() or ".." in member.parts or (mode & 0o170000) == 0o120000:
                raise RuntimeError(f"unsafe RAVDESS archive member {info.filename!r}")
            if info.file_size > 100 * 1024 * 1024:
                raise RuntimeError(f"oversized RAVDESS archive member {info.filename!r}")
            total_size += info.file_size
            if total_size > 1024 * 1024 * 1024:
                raise RuntimeError("RAVDESS archive exceeds 1 GiB uncompressed")
            members.append(info)
            if not info.is_dir() and member.suffix.lower() == ".wav":
                file_digest = hashlib.sha256()
                with archive.open(info) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                if member.name in expected:
                    raise RuntimeError(f"duplicate RAVDESS clip name {member.name}")
                expected[member.name] = file_digest.hexdigest()
        if len(expected) != RAVDESS_N_CLIPS:
            raise RuntimeError(
                f"verified RAVDESS archive has {len(expected)} clips, expected {RAVDESS_N_CLIPS}"
            )

        actual = {}
        if root.is_dir():
            for wav in root.rglob("*.wav"):
                file_digest = hashlib.sha256()
                with open(wav, "rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                actual[wav.name] = file_digest.hexdigest()
        if actual == expected:
            return str(root)

        if root.exists():
            shutil.rmtree(root)
        tmp_root = root.with_suffix(".extracting")
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        archive.extractall(tmp_root, members=members)
        tmp_root.replace(root)
    vol.commit()
    return str(root)


@app.function(
    gpu="H100",
    timeout=3 * 60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_eval(baseline_lens: str, candidate_lens: str, limit: int = 0) -> str:
    import importlib.metadata
    import json
    import os

    import torch
    import transformers

    import jlens
    from jlens.hooks import ActivationRecorder

    from audiolens import (
        ACTED_TO_CLUSTER,
        anchor_fingerprint,
        anchor_token_ids,
        load_anchors,
        mood_readout,
        parse_ravdess_name,
        resolve_audio_token_id,
    )
    from audiolens.fitting import (
        MODEL_REVISION,
        atomic_write_json,
        config_digest,
        paired_resume_prefix,
        sha256_file,
        validate_lens,
    )

    ravdess_root = _stage_ravdess()
    clips = sorted(pathlib.Path(ravdess_root).rglob("*.wav"))
    if len(clips) != RAVDESS_N_CLIPS:
        raise RuntimeError(f"staged RAVDESS has {len(clips)}, expected {RAVDESS_N_CLIPS}")

    anchor_words, _anchor_colors = load_anchors("/root/anchors/multilingual.yaml")
    if "curiosity" in anchor_words:
        raise RuntimeError("production evaluation anchors unexpectedly include curiosity")
    config = {
        "schema_version": SCHEMA_VERSION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "runtime": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "modal_environment": {
                key: os.environ[key]
                for key in ("MODAL_IMAGE_ID",)
                if key in os.environ
            },
        },
        "source": {"git_revision": GIT_REVISION, "digest": SOURCE_DIGEST},
        "attention_implementation": "eager",
        "ravdess_sha256": RAVDESS_SHA256,
        "n_clips": RAVDESS_N_CLIPS,
        "read_layers": READ_LAYERS,
        "topk": TOPK,
        "lenses": {
            "text400": {"path": baseline_lens, "sha256": sha256_file(baseline_lens)},
            "mixed528": {"path": candidate_lens, "sha256": sha256_file(candidate_lens)},
        },
        "anchors": {
            "fingerprint": anchor_fingerprint(anchor_words),
            "clusters": list(anchor_words),
            "acted_to_cluster": {
                acted: cluster
                for acted, cluster in ACTED_TO_CLUSTER.items()
                if cluster in anchor_words
            },
        },
        "scoring": "per-cluster-mean-anchor-token-probability-v1",
    }
    digest = config_digest(config)
    results_path = pathlib.Path(f"{VOL_MOUNT}/eval/ravdess-paired-{digest[:12]}.jsonl")
    metadata_path = results_path.with_suffix(".json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_sha256") != digest or metadata.get("config") != config:
            raise RuntimeError(f"paired eval config mismatch at {metadata_path}")
    else:
        metadata = {
            "config": config,
            "config_sha256": digest,
            "results": str(results_path),
            "completed": False,
        }
        atomic_write_json(metadata_path, metadata)

    done = paired_resume_prefix(results_path, config, [clip.name for clip in clips])
    todo = [clip for clip in clips if clip.name not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        if len(done) == RAVDESS_N_CLIPS and not metadata.get("completed"):
            metadata["completed"] = True
            metadata["n_records"] = len(done)
            atomic_write_json(metadata_path, metadata)
            vol.commit()
        return f"{results_path}: already complete ({len(done)} clips)"

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION
    )
    tokenizer = processor.tokenizer
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    model = jlens.from_hf(hf, tokenizer, force_bos=False)
    lenses = {
        "text400": jlens.JacobianLens.load(baseline_lens),
        "mixed528": jlens.JacobianLens.load(candidate_lens),
    }
    validate_lens(lenses["text400"], 400)
    validate_lens(lenses["mixed528"], 528)
    anchors = anchor_token_ids(tokenizer, anchor_words)
    audio_id = resolve_audio_token_id(hf.config, tokenizer)

    with open(results_path, "a", encoding="utf-8") as output:
        for index, wav in enumerate(todo):
            messages = [{"role": "user", "content": [{"type": "audio", "audio": str(wav)}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
            positions = (inputs["input_ids"][0] == audio_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise RuntimeError(f"{wav.name}: no audio soft tokens")
            with torch.no_grad(), ActivationRecorder(model.layers, at=READ_LAYERS) as recorder:
                hf.model(**inputs, use_cache=False)

            record = {
                "clip": wav.name,
                "meta": parse_ravdess_name(wav.stem),
                "n_audio_tokens": int(positions.numel()),
                "seq_len": int(inputs["input_ids"].shape[1]),
                "readouts": {},
            }
            for label, lens in lenses.items():
                layer_results = {}
                for layer in READ_LAYERS:
                    residual = recorder.activations[layer][0][positions].float()
                    logits = model.unembed(lens.transport(residual, layer)).float()
                    mass, top_ids = mood_readout(logits, anchors, topk=TOPK)
                    layer_results[str(layer)] = {
                        "anchor_mass": mass,
                        "topk_ids": top_ids,
                        "topk_toks": [tokenizer.decode([token]) for token in top_ids],
                    }
                record["readouts"][label] = {"layers": layer_results}
            output.write(json.dumps(record, sort_keys=True) + "\n")
            if (index + 1) % 100 == 0 or index == len(todo) - 1:
                output.flush()
                vol.commit()
                print(f"{index + 1}/{len(todo)} ({wav.name})")

    total = len(done) + len(todo)
    if not limit and total == RAVDESS_N_CLIPS:
        metadata["completed"] = True
        metadata["n_records"] = total
        atomic_write_json(metadata_path, metadata)
    vol.commit()
    return f"{results_path}: {total} paired clips; metadata {metadata_path}"


@app.local_entrypoint()
def main(baseline_lens: str, candidate_lens: str, limit: int = 0):
    if not baseline_lens or not candidate_lens:
        raise SystemExit("--baseline-lens and --candidate-lens are required")
    print(run_eval.remote(baseline_lens, candidate_lens, limit))
