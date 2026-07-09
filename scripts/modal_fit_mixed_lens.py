"""Fit text400, audio32/64/128, and mixed528 Jacobian lenses on Modal.

The fit is content-addressed and uses a committed 128-row LibriSpeech
manifest.  Expensive work is deliberately gated:

    uv run modal run scripts/modal_fit_mixed_lens.py --stage-only
    modal volume get audiolens-vol manifests/librispeech_audio_fit_128.jsonl manifests/
    uv run modal run scripts/modal_fit_mixed_lens.py --validate-replay-only
    uv run modal run scripts/modal_fit_mixed_lens.py --audio-limit 1
    uv run modal run scripts/modal_fit_mixed_lens.py
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
MANIFEST_NAME = "librispeech_audio_fit_128.jsonl"
BUNDLED_MANIFEST = f"/root/manifests/{MANIFEST_NAME}"
VOLUME_MANIFEST = f"{VOL_MOUNT}/manifests/{MANIFEST_NAME}"
GENERIC_LENS_SHA256 = "3a5c5169fa9e1cecf0d2a0561c01f2d099991decdea8740efcba1dba7d741d13"
DEFAULT_PUBLISH_LENSES = "text400,mixed528"
DEFAULT_ARTIFACT_LICENSE = "cc-by-sa-4.0"


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


def _source_digest() -> str:
    relatives = (
        "pyproject.toml",
        "uv.lock",
        "src/audiolens/fitting.py",
        "src/audiolens/models.py",
        "scripts/modal_fit_mixed_lens.py",
    )
    if all((REPO_ROOT / relative).is_file() for relative in relatives):
        digest = hashlib.sha256()
        for relative in relatives:
            digest.update(relative.encode())
            digest.update((REPO_ROOT / relative).read_bytes())
        return digest.hexdigest()
    injected = os.environ.get("AUDIOLENS_SOURCE_DIGEST")
    if injected:
        return injected
    raise RuntimeError("cannot determine source digest")


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
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "AUDIOLENS_GIT_REVISION": GIT_REVISION,
            "AUDIOLENS_SOURCE_DIGEST": SOURCE_DIGEST,
        }
    )
    .add_local_python_source("audiolens")
    .add_local_dir(str(REPO_ROOT / "manifests"), remote_path="/root/manifests")
)

app = modal.App("audiolens-mixed-fit", image=image)
vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)


def _model_profile():
    from audiolens.models import DEFAULT_MODEL_KEY, get_model_profile

    return get_model_profile(DEFAULT_MODEL_KEY)

def _fit_profile_config(profile) -> dict[str, object]:
    """Content-addressed execution identity contributed by a model profile."""

    return {
        "profile_key": profile.key,
        "profile_version": profile.version,
        "profile_slug": profile.slug,
        "adapter_source": profile.adapter_source,
        "model": {"id": profile.model_id, "revision": profile.model_revision},
    }

def _fit_geometry_config(profile) -> dict[str, object]:
    """Profile-driven estimator and serialization settings."""

    return {
        "source_layers": list(profile.source_layers),
        "target_layer": profile.target_layer,
        "skip_first": profile.skip_first,
        "max_seq_len": profile.max_sequence_length,
        "dim_batch": profile.dimension_batch_size,
        "model_dtype": "bfloat16",
        "artifact_dtype": "float16",
    }



def _fit_run_tag(profile, config_sha256: str) -> str:
    return f"{profile.slug}-mixed-{config_sha256[:12]}"



def _runtime_identity() -> dict:
    import importlib.metadata
    import os

    import torch

    packages = [
        "accelerate",
        "datasets",
        "huggingface-hub",
        "modal",
        "soundfile",
        "torch",
        "transformers",
    ]
    return {
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python": os.sys.version,
        "cuda": torch.version.cuda,
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "device": torch.cuda.get_device_name(0),
        "modal_environment": {
            key: os.environ[key]
            for key in ("MODAL_IMAGE_ID",)
            if key in os.environ
        },
    }


def _read_manifest(path: str):
    from audiolens.fitting import audit_manifest_rows, load_jsonl

    profile = _model_profile()
    rows = load_jsonl(path)
    audit_manifest_rows(rows, profile=profile)
    return rows


@app.function(
    cpu=4.0,
    memory=16_384,
    timeout=60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def stage_manifest() -> str:
    """Select and atomically stage the exact 64-clean/64-other fit corpus."""
    import io
    import os
    import pathlib
    import shutil

    import soundfile as sf
    from datasets import load_dataset

    from audiolens.fitting import (
        FIT_SEED,
        LIBRISPEECH_REVISION,
        AudioFitContractError,
        audit_manifest_rows,
        audit_audio_manifest_files,
        describe_audio_sample,
        write_canonical_jsonl,
    )
    from audiolens.models import load_audio_processor

    profile = _model_profile()
    processor = load_audio_processor(profile.key)

    manifest_path = pathlib.Path(VOLUME_MANIFEST)
    final_root = pathlib.Path(
        f"{VOL_MOUNT}/fit_data/librispeech-{LIBRISPEECH_REVISION[:12]}-seed{FIT_SEED}"
    )
    if manifest_path.is_file() and final_root.is_dir():
        rows = _read_manifest(str(manifest_path))
        audit_audio_manifest_files(rows, processor, profile=profile)
        return f"{manifest_path} already staged ({len(rows)} rows)"

    tmp_root = final_root.with_name(f"{final_root.name}.tmp.{os.getpid()}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    if final_root.exists():
        shutil.rmtree(final_root)
    tmp_root.mkdir(parents=True)

    rows: list[dict] = []
    speakers: set[str] = set()
    strata = (("clean", "train.100"), ("other", "train.500"))
    for config, split in strata:
        print(f"filling 10,000-row shuffle buffer for {config}/{split}...")
        selected = 0
        dataset = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=split,
            revision=LIBRISPEECH_REVISION,
            streaming=True,
        ).decode(False)
        dataset = dataset.shuffle(seed=FIT_SEED, buffer_size=10_000)
        for source in dataset:
            speaker = str(source["speaker_id"])
            if speaker in speakers or not str(source["text"]).strip():
                continue
            audio_spec = source["audio"]
            blob = audio_spec.get("bytes")
            if not blob:
                continue
            try:
                info = sf.info(io.BytesIO(blob))
            except sf.LibsndfileError:
                continue
            duration = info.frames / info.samplerate
            if info.samplerate != 16_000 or info.channels != 1 or not 2.0 <= duration <= 4.0:
                continue
            audio, sample_rate = sf.read(
                io.BytesIO(blob), dtype="float32", always_2d=False
            )
            if sample_rate != 16_000 or audio.ndim != 1:
                continue
            sample_id = str(source["id"])
            filename = sample_id.replace("/", "_") + ".flac"
            staged_path = tmp_root / filename
            staged_path.write_bytes(blob)
            try:
                descriptor = describe_audio_sample(
                    processor, staged_path, profile=profile
                )
            except AudioFitContractError:
                staged_path.unlink()
                continue
            rows.append(
                {
                    "dataset": "openslr/librispeech_asr",
                    "revision": LIBRISPEECH_REVISION,
                    "config": config,
                    "split": split,
                    "selection_index": len(rows),
                    "id": sample_id,
                    "speaker_id": source["speaker_id"],
                    "chapter_id": source["chapter_id"],
                    "transcript": str(source["text"]).strip(),
                    **descriptor.manifest_fields,
                    "volume_path": str(final_root / filename),
                }
            )
            speakers.add(speaker)
            selected += 1
            if selected % 16 == 0:
                print(f"{config}/{split}: selected {selected}/64")
            if selected == 64:
                break
        if selected != 64:
            raise RuntimeError(f"selected only {selected}/64 rows for {config}/{split}")

    audit_manifest_rows(rows, profile=profile)
    tmp_root.replace(final_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_jsonl(manifest_path, rows)
    vol.commit()
    return f"{manifest_path}: staged 128 rows at {final_root}"


@app.function(
    cpu=4.0,
    memory=16_384,
    timeout=60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def restore_staged_audio() -> str:
    """Reconstruct the committed manifest's exact FLACs on a fresh volume."""
    import os
    import pathlib
    import shutil

    from datasets import load_dataset

    from audiolens.fitting import (
        AudioFitContractError,
        LIBRISPEECH_REVISION,
        audit_audio_manifest_files,
        sha256_bytes,
        sha256_file,
        write_canonical_jsonl,
    )
    from audiolens.models import load_audio_processor

    profile = _model_profile()
    processor = load_audio_processor(profile.key)

    rows = _read_manifest(BUNDLED_MANIFEST)
    volume_manifest = pathlib.Path(VOLUME_MANIFEST)
    if volume_manifest.is_file() and sha256_file(volume_manifest) == sha256_file(
        BUNDLED_MANIFEST
    ):
        try:
            audit_audio_manifest_files(rows, processor, profile=profile)
            return f"{volume_manifest}: exact committed corpus already present"
        except AudioFitContractError:
            pass

    final_root = pathlib.Path(rows[0]["volume_path"]).parent
    if any(pathlib.Path(row["volume_path"]).parent != final_root for row in rows):
        raise AudioFitContractError("manifest volume paths do not share one corpus root")
    tmp_root = final_root.with_name(f"{final_root.name}.restore.{os.getpid()}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True)

    for config, split in (("clean", "train.100"), ("other", "train.500")):
        targets = {
            str(row["id"]): row
            for row in rows
            if row["config"] == config and row["split"] == split
        }
        remaining = set(targets)
        dataset = load_dataset(
            "openslr/librispeech_asr",
            config,
            split=split,
            revision=LIBRISPEECH_REVISION,
            streaming=True,
        ).decode(False)
        for source in dataset:
            sample_id = str(source["id"])
            if sample_id not in remaining:
                continue
            blob = source["audio"].get("bytes")
            row = targets[sample_id]
            if not blob or sha256_bytes(blob) != row["audio_sha256"]:
                raise AudioFitContractError(f"retrieved audio hash mismatch for {sample_id}")
            (tmp_root / pathlib.Path(row["volume_path"]).name).write_bytes(blob)
            remaining.remove(sample_id)
            if not remaining:
                break
        if remaining:
            raise AudioFitContractError(
                f"could not retrieve {len(remaining)} committed IDs from {config}/{split}"
            )

    if final_root.exists():
        shutil.rmtree(final_root)
    tmp_root.replace(final_root)
    audit_audio_manifest_files(rows, processor, profile=profile)
    volume_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_jsonl(volume_manifest, rows)
    vol.commit()
    return f"{volume_manifest}: restored exact 128-row committed corpus"


def _load_runtime():
    from audiolens.models import load_model_runtime

    profile = _model_profile()
    return load_model_runtime(profile.key, device_map="cuda")


def _selected_gradient_rows(target, sources, valid_positions, dimensions):
    import torch

    per_source: list[list[torch.Tensor]] = [[] for _ in sources]
    for index, dimension in enumerate(dimensions):
        cotangent = torch.zeros_like(target)
        cotangent[0, valid_positions, dimension] = 1.0
        grads = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=cotangent,
            retain_graph=index < len(dimensions) - 1,
        )
        for rows, grad in zip(per_source, grads, strict=True):
            rows.append(grad[0, valid_positions].float().mean(0).detach().cpu())
    return [torch.stack(rows) for rows in per_source]


def _bf16_batch_diagnostics(actual, expected) -> dict[str, float]:
    """Measure sparse kernel-shape rounding and global replay parity."""
    import torch

    actual_float = actual.float()
    expected_float = expected.float()
    close = torch.isclose(actual_float, expected_float, rtol=1e-2, atol=1e-2)
    mismatch_fraction = 1.0 - close.float().mean().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    ).item()
    relative_l2 = (
        (actual_float - expected_float).norm() / expected_float.norm()
    ).item()
    return {
        "mismatch_fraction": mismatch_fraction,
        "cosine": cosine,
        "relative_l2": relative_l2,
    }


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def validate_replay() -> str:
    """Prove prepared replay matches the official multimodal decoder path."""
    import torch

    from jlens.hooks import ActivationRecorder

    from audiolens.fitting import audit_audio_manifest_files

    rows = _read_manifest(BUNDLED_MANIFEST)
    runtime = _load_runtime()
    profile = runtime.profile
    processor = runtime.processor
    audio_model = runtime.audio_lens_model
    audit_audio_manifest_files(rows, processor, profile=profile)
    path = rows[0]["volume_path"]
    layers = [*profile.source_layers, profile.target_layer]
    with torch.no_grad(), ActivationRecorder(runtime.layers, at=layers) as full_recorder:
        replay_ids = audio_model.encode(path)
    full_activations = {
        layer: full_recorder.activations[layer].detach().clone()
        for layer in layers
    }
    prepared = runtime.prepare_audio(path)

    with torch.no_grad(), ActivationRecorder(
        runtime.layers, at=layers
    ) as replay_recorder:
        audio_model.forward(replay_ids)
    for layer in layers:
        torch.testing.assert_close(
            replay_recorder.activations[layer],
            full_activations[layer],
            rtol=1e-2,
            atol=1e-2,
        )
    full_logits = runtime.unembed(
        full_activations[profile.target_layer].float()
    )
    replay_logits = runtime.unembed(
        replay_recorder.activations[profile.target_layer].float()
    )
    torch.testing.assert_close(replay_logits, full_logits, rtol=1e-2, atol=5e-2)

    batch_size = profile.dimension_batch_size
    with torch.no_grad(), ActivationRecorder(
        runtime.layers, at=layers
    ) as batch_recorder:
        audio_model.forward(replay_ids.expand(batch_size, -1))
    batch_diagnostics = {}
    for layer in layers:
        batch_diagnostics[str(layer)] = _bf16_batch_diagnostics(
            batch_recorder.activations[layer][0:1],
            replay_recorder.activations[layer],
        )
    worst_mismatch = max(
        batch_diagnostics.items(), key=lambda item: item[1]["mismatch_fraction"]
    )
    worst_cosine = min(batch_diagnostics.items(), key=lambda item: item[1]["cosine"])
    worst_relative_l2 = max(
        batch_diagnostics.items(), key=lambda item: item[1]["relative_l2"]
    )
    if (
        worst_cosine[1]["cosine"] < 0.999
        or worst_relative_l2[1]["relative_l2"] > 0.05
    ):
        raise RuntimeError(
            f"batch{batch_size} replay drift exceeds global bf16 limits: "
            f"mismatch={worst_mismatch}, cosine={worst_cosine}, "
            f"relative_l2={worst_relative_l2}"
        )
    del batch_recorder, replay_recorder, full_activations

    sources = list(
        dict.fromkeys(
            (profile.source_layers[0], profile.read_layer, profile.read_layers[-1])
        )
    )
    record_at = [*sources, profile.target_layer]
    valid = prepared.layout.valid_mask.nonzero(as_tuple=True)[0].to(
        prepared.audio_positions.device
    )
    if valid.numel() == 0:
        raise RuntimeError("prepared audio has no fit-valid positions")
    dimensions = min(8, profile.d_model)
    dims = (
        torch.linspace(
            0,
            profile.d_model - 1,
            dimensions,
            device=valid.device,
        )
        .long()
        .tolist()
    )
    with torch.enable_grad(), ActivationRecorder(
        runtime.layers, at=record_at, start_graph_at=profile.source_layers[0]
    ) as recorder:
        runtime.forward_audio(prepared)
        full_rows = _selected_gradient_rows(
            recorder.activations[profile.target_layer],
            [recorder.activations[layer] for layer in sources],
            valid,
            dims,
        )
    with torch.enable_grad(), ActivationRecorder(
        runtime.layers, at=record_at, start_graph_at=profile.source_layers[0]
    ) as recorder:
        audio_model.forward(replay_ids)
        replay_rows = _selected_gradient_rows(
            recorder.activations[profile.target_layer],
            [recorder.activations[layer] for layer in sources],
            valid,
            dims,
        )
    for layer, actual, expected in zip(sources, replay_rows, full_rows, strict=True):
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item()
        relative_l2 = ((actual - expected).norm() / expected.norm()).item()
        if cosine < 0.999 or relative_l2 > 0.02:
            raise RuntimeError(
                f"L{layer} gradient mismatch: cosine={cosine:.6f}, rel_l2={relative_l2:.6f}"
            )
    return (
        "prepared replay matches full wrapper: all layers, logits, "
        f"batch{batch_size}, gradients; worst mismatch={worst_mismatch}, "
        f"cosine={worst_cosine}, relative_l2={worst_relative_l2}"
    )


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def smoke_audio_fit() -> str:
    import pathlib

    import jlens

    from audiolens.fitting import audit_audio_manifest_files

    rows = _read_manifest(BUNDLED_MANIFEST)
    runtime = _load_runtime()
    profile = runtime.profile
    audit_audio_manifest_files(rows, runtime.processor, profile=profile)
    checkpoint = pathlib.Path(
        f"{VOL_MOUNT}/smoke/{profile.slug}-audio-one.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    smoke_layer = profile.read_layer
    lens = jlens.fit(
        runtime.audio_lens_model,
        prompts=[rows[0]["volume_path"]],
        source_layers=[smoke_layer],
        target_layer=profile.target_layer,
        dim_batch=profile.dimension_batch_size,
        max_seq_len=profile.max_sequence_length,
        skip_first=profile.skip_first,
        checkpoint_path=str(checkpoint),
        checkpoint_every=1,
        resume=False,
    )
    if lens.n_prompts != 1:
        raise RuntimeError(f"one-sample smoke produced {lens.n_prompts} prompts")
    vol.commit()
    return repr(lens)


def _wikitext_prompts() -> list[str]:
    from datasets import load_dataset

    from audiolens.fitting import WIKITEXT_REVISION

    rows = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        revision=WIKITEXT_REVISION,
    )
    prompts: list[str] = []
    for row in rows:
        text = row["text"].strip()
        if len(text) >= 200:
            prompts.append(text[:2000])
        if len(prompts) == 400:
            return prompts
    raise RuntimeError(f"WikiText yielded only {len(prompts)}/400 prompts")


def _ensure_run_json(path: pathlib.Path, config: dict) -> dict:
    import json

    from audiolens.fitting import AudioFitContractError, atomic_write_json, config_digest

    digest = config_digest(config)
    if path.exists():
        run = json.loads(path.read_text())
        if run.get("config_sha256") != digest or run.get("config") != config:
            raise AudioFitContractError(f"run config mismatch at {path}")
        return run
    run = {"config": config, "config_sha256": digest, "outputs": {}}
    atomic_write_json(path, run)
    return run


def _lens_change(left, right) -> dict[str, dict[str, float]]:
    """Per-layer cosine and relative-L2 convergence diagnostics."""
    import torch

    if left.source_layers != right.source_layers:
        raise RuntimeError("cannot compare lenses with different source layers")
    diagnostics = {}
    for layer in left.source_layers:
        left_j = left.jacobians[layer].float()
        right_j = right.jacobians[layer].float()
        diagnostics[str(layer)] = {
            "cosine": torch.nn.functional.cosine_similarity(
                left_j.flatten(), right_j.flatten(), dim=0
            ).item(),
            "relative_l2": ((right_j - left_j).norm() / left_j.norm()).item(),
        }
    return diagnostics


@app.function(
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def fit_all() -> str:
    import json
    import pathlib

    import torch

    import jlens

    from audiolens.fitting import (
        JLENS_REVISION,
        LIBRISPEECH_REVISION,
        WIKITEXT_REVISION,
        atomic_write_json,
        audit_audio_manifest_files,
        canonical_json_bytes,
        config_digest,
        fit_checkpoint_metadata,
        lens_from_fit_checkpoint,
        sha256_bytes,
        sha256_file,
        validate_lens,
        validate_runtime_lens_file,
    )

    profile = _model_profile()
    source_layers = list(profile.source_layers)
    rows = _read_manifest(BUNDLED_MANIFEST)
    manifest_sha = sha256_file(BUNDLED_MANIFEST)
    if sha256_file(VOLUME_MANIFEST) != manifest_sha:
        raise RuntimeError("bundled and staged manifests differ")
    prompts = _wikitext_prompts()
    prompt_hashes = [sha256_bytes(prompt.encode()) for prompt in prompts]
    wikitext_sha = sha256_bytes(canonical_json_bytes(prompt_hashes))
    config = {
        "schema_version": 2,
        **_fit_profile_config(profile),
        "runtime": _runtime_identity(),
        "source": {"git_revision": GIT_REVISION, "digest": SOURCE_DIGEST},
        "jlens_revision": JLENS_REVISION,
        "attention_implementation": "eager",
        "wikitext": {
            "dataset": "Salesforce/wikitext",
            "revision": WIKITEXT_REVISION,
            "n_prompts": 400,
            "ordered_prompt_sha256": wikitext_sha,
        },
        "audio": {
            "dataset": "openslr/librispeech_asr",
            "revision": LIBRISPEECH_REVISION,
            "n_prompts": 128,
            "ordered_manifest_sha256": manifest_sha,
        },
        "fit": _fit_geometry_config(profile),
    }
    digest = config_digest(config)
    tag = _fit_run_tag(profile, digest)
    run_path = pathlib.Path(f"{VOL_MOUNT}/runs/{tag}.json")
    run = _ensure_run_json(run_path, config)
    checkpoint_dir = pathlib.Path(f"{VOL_MOUNT}/ckpt/{tag}")
    lens_dir = pathlib.Path(f"{VOL_MOUNT}/lenses")
    generic_lens = lens_dir / f"{profile.slug}_jacobian_lens.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lens_dir.mkdir(parents=True, exist_ok=True)
    if not generic_lens.is_file():
        raise RuntimeError(f"required generic lens is missing at {generic_lens}")
    generic_before = sha256_file(generic_lens)
    if generic_before != GENERIC_LENS_SHA256:
        raise RuntimeError(f"generic lens has unexpected SHA-256 {generic_before}")

    runtime = _load_runtime()
    audit_audio_manifest_files(rows, runtime.processor, profile=profile)
    text_checkpoint = checkpoint_dir / "text400.pt"
    text_lens = jlens.fit(
        runtime.text_lens_model,
        prompts=prompts,
        source_layers=source_layers,
        target_layer=profile.target_layer,
        dim_batch=profile.dimension_batch_size,
        max_seq_len=profile.max_sequence_length,
        skip_first=profile.skip_first,
        checkpoint_path=str(text_checkpoint),
        checkpoint_every=5,
        resume=True,
    )
    text_path = lens_dir / f"{tag}-text400.pt"
    validate_lens(text_lens, 400, profile=profile)
    text_lens.save(text_path)
    vol.commit()

    audio_checkpoint = checkpoint_dir / "audio128.pt"
    prefix_paths: dict[int, pathlib.Path] = {}
    prefix_lenses: dict[int, object] = {}
    audio_lens = None
    audio_prompts = [row["volume_path"] for row in rows]
    for prefix in (32, 64, 128):
        audio_lens = jlens.fit(
            runtime.audio_lens_model,
            prompts=audio_prompts[:prefix],
            source_layers=source_layers,
            target_layer=profile.target_layer,
            dim_batch=profile.dimension_batch_size,
            max_seq_len=profile.max_sequence_length,
            skip_first=profile.skip_first,
            checkpoint_path=str(audio_checkpoint),
            checkpoint_every=5,
            resume=True,
        )
        validate_lens(audio_lens, prefix, profile=profile)
        state = torch.load(audio_checkpoint, map_location="cpu", weights_only=True)
        if state["n_done"] != prefix or state["next_idx"] != prefix:
            raise RuntimeError(f"audio checkpoint did not reach exact prefix {prefix}")
        path = lens_dir / f"{tag}-audio{prefix}.pt"
        audio_lens.save(path)
        prefix_paths[prefix] = path
        prefix_lenses[prefix] = audio_lens
        vol.commit()
    if audio_lens is None:
        raise AssertionError("audio prefixes did not run")

    text_fp32 = lens_from_fit_checkpoint(text_checkpoint, 400, profile=profile)
    audio_fp32 = lens_from_fit_checkpoint(audio_checkpoint, 128, profile=profile)
    mixed = jlens.JacobianLens.merge([text_fp32, audio_fp32])
    validate_lens(mixed, 528, profile=profile)
    for layer in source_layers:
        expected = (
            400 * text_fp32.jacobians[layer] + 128 * audio_fp32.jacobians[layer]
        ) / 528
        torch.testing.assert_close(mixed.jacobians[layer], expected, rtol=0, atol=0)
    mixed_path = lens_dir / f"{tag}-mixed528.pt"
    mixed.save(mixed_path)

    outputs = {
        "text400": text_path,
        "audio32": prefix_paths[32],
        "audio64": prefix_paths[64],
        "audio128": prefix_paths[128],
        "mixed528": mixed_path,
        "text_checkpoint": text_checkpoint,
        "audio_checkpoint": audio_checkpoint,
    }
    runtime_lenses = {
        "text400": text_lens,
        "audio32": prefix_lenses[32],
        "audio64": prefix_lenses[64],
        "audio128": prefix_lenses[128],
        "mixed528": mixed,
    }
    checkpoint_metadata = {
        "text_checkpoint": fit_checkpoint_metadata(
            text_checkpoint, 400, profile=profile
        ),
        "audio_checkpoint": fit_checkpoint_metadata(
            audio_checkpoint, 128, profile=profile
        ),
    }
    run["outputs"] = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            **(
                {
                    "kind": "runtime_lens",
                    "dtype": "float16",
                    "n_prompts": runtime_lenses[name].n_prompts,
                    "d_model": runtime_lenses[name].d_model,
                    "source_layers": runtime_lenses[name].source_layers,
                }
                if name in runtime_lenses
                else checkpoint_metadata[name]
            ),
        }
        for name, path in outputs.items()
    }
    run["convergence"] = {
        "audio32_to_audio64": _lens_change(prefix_lenses[32], prefix_lenses[64]),
        "audio64_to_audio128": _lens_change(prefix_lenses[64], prefix_lenses[128]),
        "text400_to_mixed528": _lens_change(text_fp32, mixed),
    }
    run["completed"] = True
    atomic_write_json(run_path, run)
    for count, path in (
        (400, text_path),
        (32, prefix_paths[32]),
        (64, prefix_paths[64]),
        (128, prefix_paths[128]),
        (528, mixed_path),
    ):
        validate_runtime_lens_file(path, count, profile=profile)
    generic_after = sha256_file(generic_lens)
    if generic_after != generic_before:
        raise RuntimeError("generic lens changed during mixed fit")
    vol.commit()
    return json.dumps(
        {"tag": tag, "run": str(run_path), "outputs": run["outputs"]}, indent=2
    )


@app.function(
    cpu=2.0,
    memory=4096,
    timeout=30 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def publish_hf_run(
    run_tag: str,
    repo_id: str,
    selected_lenses: str = DEFAULT_PUBLISH_LENSES,
    artifact_license: str = DEFAULT_ARTIFACT_LICENSE,
    private: bool = False,
) -> str:
    """Publish selected runtime lenses from one completed volume run."""
    import json
    import pathlib
    import tempfile

    from audiolens.hub import prepare_hf_bundle, publish_hf_bundle

    if not run_tag or pathlib.PurePath(run_tag).name != run_tag:
        raise ValueError("run_tag must be one completed run name")
    run_path = pathlib.Path(f"{VOL_MOUNT}/runs/{run_tag}.json")
    with tempfile.TemporaryDirectory(prefix="audiolens-hf-") as temporary:
        bundle = prepare_hf_bundle(
            run_path,
            pathlib.Path(temporary) / "bundle",
            selected_lenses,
            artifact_license,
        )
        commit = publish_hf_bundle(bundle, repo_id, private=private)
    commit_url = getattr(commit, "commit_url", None) or getattr(commit, "url", None)
    return json.dumps(
        {
            "repo_id": repo_id,
            "run": run_tag,
            "lenses": selected_lenses.split(","),
            "artifact_license": artifact_license,
            "private": private,
            "commit": str(commit_url or commit),
        },
        indent=2,
    )


@app.local_entrypoint()
def main(
    stage_only: bool = False,
    validate_replay_only: bool = False,
    audio_limit: int = 128,
    audit_manifest: str = "",
    publish_to_hf: str = "",
    publish_run: str = "",
    publish_lenses: str = DEFAULT_PUBLISH_LENSES,
    publish_license: str = DEFAULT_ARTIFACT_LICENSE,
    publish_private: bool = False,
):
    publish_requested = bool(publish_to_hf)
    if publish_run and not publish_requested:
        raise SystemExit("--publish-run requires --publish-to-hf")
    if not publish_requested and (
        publish_private
        or publish_lenses != DEFAULT_PUBLISH_LENSES
        or publish_license != DEFAULT_ARTIFACT_LICENSE
    ):
        raise SystemExit("publication options require --publish-to-hf")
    if publish_requested:
        incompatible = []
        if stage_only:
            incompatible.append("--stage-only")
        if validate_replay_only:
            incompatible.append("--validate-replay-only")
        if audit_manifest:
            incompatible.append("--audit-manifest")
        if audio_limit != 128:
            incompatible.append("--audio-limit")
        if incompatible:
            raise SystemExit(
                f"--publish-to-hf is incompatible with {', '.join(incompatible)}"
            )
        if "/" not in publish_to_hf or publish_to_hf.strip() != publish_to_hf:
            raise SystemExit("--publish-to-hf must be a namespace/repository ID")
        names = [name.strip() for name in publish_lenses.split(",")]
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise SystemExit("--publish-lenses must contain unique comma-separated names")
        publish_lenses = ",".join(names)
        if publish_run:
            if pathlib.PurePath(publish_run).name != publish_run:
                raise SystemExit("--publish-run must be one completed run tag")
            print(
                publish_hf_run.remote(
                    publish_run,
                    publish_to_hf,
                    publish_lenses,
                    publish_license,
                    publish_private,
                )
            )
            return
    if audit_manifest:
        from audiolens.fitting import audit_manifest_rows, load_jsonl, sha256_file

        rows = load_jsonl(audit_manifest)
        audit_manifest_rows(rows, profile=_model_profile())
        print(f"{audit_manifest}: {len(rows)} rows, sha256={sha256_file(audit_manifest)}")
        return
    if stage_only:
        print(stage_manifest.remote())
        print(f"download: modal volume get audiolens-vol manifests/{MANIFEST_NAME} manifests/")
        return
    if validate_replay_only:
        print(restore_staged_audio.remote())
        print(validate_replay.remote())
        return
    if audio_limit == 1:
        print(restore_staged_audio.remote())
        print(smoke_audio_fit.remote())
        return
    if audio_limit != 128:
        raise SystemExit("--audio-limit supports only 1 (smoke) or 128 (full fit)")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True, cwd=REPO_ROOT
    )
    if status.strip():
        raise SystemExit("full fit requires a clean git worktree so source identity is durable")
    print(restore_staged_audio.remote())
    fit_result = fit_all.remote()
    print(fit_result)
    if publish_requested:
        import json

        run_tag = json.loads(fit_result)["tag"]
        print(
            publish_hf_run.remote(
                run_tag,
                publish_to_hf,
                publish_lenses,
                publish_license,
                publish_private,
            )
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Local structural audit for a downloaded audio-fit manifest."
    )
    parser.add_argument("--audit-manifest", required=True)
    arguments = parser.parse_args()
    from audiolens.fitting import audit_manifest_rows, load_jsonl, sha256_file

    manifest_rows = load_jsonl(arguments.audit_manifest)
    audit_manifest_rows(manifest_rows, profile=_model_profile())
    print(
        f"{arguments.audit_manifest}: {len(manifest_rows)} rows, "
        f"sha256={sha256_file(arguments.audit_manifest)}"
    )
