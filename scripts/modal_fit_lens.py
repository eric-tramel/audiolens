"""Fit the canonical text Jacobian lens on Modal.

The production artifact is the fp16 serialization of Anthropic's released
estimator fitted to exactly 1,000 ordered, Neuronpedia-style WikiText chunks.
``--n-prompts`` exists only for content-addressed smoke runs; those runs are
explicitly noncanonical.

    modal run scripts/modal_fit_lens.py
    modal run scripts/modal_fit_lens.py --n-prompts 1

The fit manifest, resumable fp32 checkpoint, and runtime lens all have paths
derived from the immutable fit-config digest.  A completed manifest is the
portable entry point for consumers; a bare lens file is not sufficient.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
MODEL_ID = "google/gemma-4-E2B-it"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_SPLIT = "train"
DATASET_TEXT_FIELD = "text"
CANONICAL_PROMPT_COUNT = 1_000
CHUNK_CHARS = 2_000
MIN_TAIL_CHARS = 200
CHECKPOINT_EVERY = 5
MODAL_FUNCTION_TIMEOUT_SECONDS = 24 * 60 * 60
D_MODEL = 1_536
FIT_MANIFEST_SCHEMA_VERSION = 1
FIT_MANIFEST_KIND = "canonical_text_jlens_fit"
CHUNK_ALGORITHM = "neuronpedia_concat_space_strip_emit_strict_gt_v1"


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


FIT_SOURCE_RELATIVES = (
    "pyproject.toml",
    "uv.lock",
    "src/audiolens/__init__.py",
    "src/audiolens/fitting.py",
    "src/audiolens/models/__init__.py",
    "src/audiolens/models/base.py",
    "src/audiolens/models/gemma4.py",
    "scripts/modal_fit_lens.py",
)


def _source_digest() -> str:
    relatives = FIT_SOURCE_RELATIVES
    if all((REPO_ROOT / relative).is_file() for relative in relatives):
        digest = hashlib.sha256()
        for relative in relatives:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update((REPO_ROOT / relative).read_bytes())
        return digest.hexdigest()
    injected = os.environ.get("AUDIOLENS_SOURCE_DIGEST")
    if injected:
        return injected
    raise RuntimeError("cannot determine source digest") from None


def _lock_digest() -> str:
    path = REPO_ROOT / "uv.lock"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    injected = os.environ.get("AUDIOLENS_LOCK_SHA256")
    if injected:
        return injected
    raise RuntimeError("cannot determine uv.lock digest")


GIT_REVISION = _git_revision()
SOURCE_DIGEST = _source_digest()
LOCK_SHA256 = _lock_digest()

_HAS_LOCAL_PROJECT = all(
    (REPO_ROOT / relative).is_file()
    for relative in FIT_SOURCE_RELATIVES
)

if _HAS_LOCAL_PROJECT:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .uv_sync(
            uv_project_dir=str(REPO_ROOT),
            frozen=True,
            groups=["fit"],
            gpu="H100",
        )
        .env(
            {
                "HF_HOME": f"{VOL_MOUNT}/hf",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "AUDIOLENS_GIT_REVISION": GIT_REVISION,
                "AUDIOLENS_SOURCE_DIGEST": SOURCE_DIGEST,
                "AUDIOLENS_LOCK_SHA256": LOCK_SHA256,
            }
        )
        .add_local_python_source("audiolens")
    )
    app = modal.App("audiolens-canonical-fit", image=image)
    vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)
else:
    # Evaluation images may bundle only this script.  Keep manifest validation
    # importable there without trying to rebuild a nonexistent uv project.
    image = None
    app = None
    vol = None


def _modal_fit_function(function):
    if app is None or vol is None:
        return function
    return app.function(
        gpu="H100",
        timeout=MODAL_FUNCTION_TIMEOUT_SECONDS,
        volumes={VOL_MOUNT: vol},
        secrets=[modal.Secret.from_name("huggingface")],
    )(function)


def _modal_local_entrypoint(function):
    if app is None:
        return function
    return app.local_entrypoint()(function)


def _commit_volume() -> None:
    global vol
    if vol is None:
        # Modal bundles the function module without the local uv project, so
        # image construction is intentionally skipped there.  Reacquire the
        # already-mounted named volume only when the fit actually commits.
        vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)
    vol.commit()


def _runtime_identity() -> dict[str, Any]:
    import importlib.metadata
    import platform

    import torch

    packages = (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "jlens",
        "modal",
        "torch",
        "transformers",
    )
    return {
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_function_timeout_seconds": MODAL_FUNCTION_TIMEOUT_SECONDS,
    }


def _load_model():
    """Load the one pinned eager-bf16 Gemma implementation used by the fit."""
    import torch
    import transformers

    from audiolens.fitting import MODEL_REVISION

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    return tokenizer, hf


def _wikitext_prompts(
    n_prompts: int,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    max_chars: int = CHUNK_CHARS,
    min_chars: int = MIN_TAIL_CHARS,
) -> list[str]:
    """Return exactly ``n_prompts`` using Neuronpedia's ordered rechunking.

    Rows are stripped, blanks and section headers are skipped, and every kept
    row is appended as ``" " + text``.  Chunks are emitted only while the
    buffer length is *strictly greater* than ``max_chars``; the exact suffix is
    carried forward.  A final stripped suffix is eligible only at
    ``min_chars`` or longer.  Returning fewer than requested is an error.

    ``rows`` is injectable so the byte-level corpus contract can be tested
    without network access.  Production always streams the pinned dataset.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}")
    if max_chars <= 0 or min_chars < 0:
        raise ValueError("chunk bounds must be max_chars > 0 and min_chars >= 0")

    if rows is None:
        from datasets import load_dataset

        from audiolens.fitting import WIKITEXT_REVISION

        rows = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            revision=WIKITEXT_REVISION,
            streaming=True,
            trust_remote_code=False,
        )

    prompts: list[str] = []
    buffer = ""
    for record in rows:
        text = str(record.get(DATASET_TEXT_FIELD, "")).strip()
        if not text or text.startswith("="):
            continue
        buffer += " " + text
        while len(buffer) > max_chars:
            prompts.append(buffer[:max_chars].strip())
            buffer = buffer[max_chars:]
            if len(prompts) >= n_prompts:
                return prompts

    tail = buffer.strip()
    if tail and len(tail) >= min_chars and len(prompts) < n_prompts:
        prompts.append(tail)
    if len(prompts) != n_prompts:
        raise RuntimeError(f"WikiText yielded only {len(prompts)}/{n_prompts} prompts")
    return prompts


def _ordered_prompt_sha256(prompts: list[str]) -> str:
    from audiolens.fitting import canonical_json_bytes, sha256_bytes

    prompt_hashes = [sha256_bytes(prompt.encode("utf-8")) for prompt in prompts]
    return sha256_bytes(canonical_json_bytes(prompt_hashes))


def _build_fit_config(
    prompts: list[str],
    *,
    requested_count: int,
    runtime_identity: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the complete immutable identity before allocating the model."""
    from audiolens.fitting import JLENS_REVISION, MODEL_REVISION, WIKITEXT_REVISION
    from audiolens.models import DEFAULT_MODEL_PROFILE

    profile = DEFAULT_MODEL_PROFILE

    if requested_count != len(prompts):
        raise ValueError(
            f"requested_count={requested_count} but received {len(prompts)} prompts"
        )
    source = dict(
        source_identity
        or {"git_revision": GIT_REVISION, "digest": SOURCE_DIGEST}
    )
    runtime = dict(runtime_identity or _runtime_identity())
    runtime["modal_function_timeout_seconds"] = MODAL_FUNCTION_TIMEOUT_SECONDS
    return {
        "schema_version": FIT_MANIFEST_SCHEMA_VERSION,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "text_field": DATASET_TEXT_FIELD,
            "revision": WIKITEXT_REVISION,
            "streaming": True,
            "trust_remote_code": False,
            "chunking": {
                "algorithm": CHUNK_ALGORITHM,
                "max_chars": CHUNK_CHARS,
                "min_tail_chars": MIN_TAIL_CHARS,
            },
            "requested_count": requested_count,
            "ordered_prompt_sha256": _ordered_prompt_sha256(prompts),
        },
        "jlens": {"revision": JLENS_REVISION},
        "prompt_policy": {
            "kind": "raw_text",
            "force_bos": True,
            "chat_template": False,
        },
        "fit": {
            "source_layers": list(profile.source_layers),
            "target_layer": profile.target_layer,
            "skip_first": profile.skip_first,
            "max_seq_len": profile.max_sequence_length,
            "dim_batch": profile.dimension_batch_size,
            "checkpoint_every": CHECKPOINT_EVERY,
            "resume": True,
            "compile": False,
            "d_model": D_MODEL,
            "model_dtype": "bfloat16",
            "accumulation_dtype": "float32",
            "artifact_dtype": "float16",
            "attention_backend": "eager",
        },
        "lock": {
            "uv_lock_sha256": LOCK_SHA256,
            "frozen": True,
            "dependency_group": "fit",
        },
        "source": source,
        "runtime": runtime,
    }


def _run_paths(config: Mapping[str, Any]) -> dict[str, str]:
    from audiolens.fitting import config_digest

    digest = config_digest(dict(config))
    tag = f"gemma-4-E2B-it-jlens-{digest}"
    return {
        "manifest": f"runs/{tag}.json",
        "checkpoint": f"ckpt/{tag}.pt",
        "lens": f"lenses/{tag}.pt",
    }
def _diagnostic_prefix_checkpoint(config: Mapping[str, Any]) -> str:
    checkpoint = _run_paths(config)["checkpoint"]
    return f"{checkpoint.removesuffix('.pt')}.prefix-500.pt"




def _ensure_fit_manifest_at(
    path: pathlib.Path,
    config: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    """Create or identity-check a pending/completed manifest atomically."""
    from audiolens.fitting import AudioFitContractError, atomic_write_json, config_digest

    digest = config_digest(config)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AudioFitContractError(f"invalid fit manifest at {path}") from exc
        if not isinstance(loaded, Mapping):
            raise AudioFitContractError(f"invalid fit manifest root at {path}")
        record = dict(loaded)
        if (
            record.get("schema_version") != FIT_MANIFEST_SCHEMA_VERSION
            or record.get("kind") != FIT_MANIFEST_KIND
            or record.get("fit_config_sha256") != digest
            or record.get("config") != config
            or record.get("paths") != paths
        ):
            raise AudioFitContractError(f"fit manifest identity mismatch at {path}")
        if record.get("status") not in {"pending", "complete"}:
            raise AudioFitContractError(f"invalid fit manifest status at {path}")
        return record

    record = {
        "schema_version": FIT_MANIFEST_SCHEMA_VERSION,
        "kind": FIT_MANIFEST_KIND,
        "status": "pending",
        "fit_config_sha256": digest,
        "config": config,
        "paths": paths,
        "canonical": False,
        "lens": None,
        "checkpoint": None,
        "stability": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, record)
    return record


def _initialize_fit_manifest(
    volume_root: str | pathlib.Path,
    config: dict[str, Any],
) -> tuple[pathlib.Path, dict[str, str], dict[str, Any]]:
    root = pathlib.Path(volume_root)
    paths = _run_paths(config)
    manifest_path = root / paths["manifest"]
    return manifest_path, paths, _ensure_fit_manifest_at(manifest_path, config, paths)


def _read_checkpoint_state(
    path: pathlib.Path,
    config: Mapping[str, Any],
    *,
    maximum_count: int,
    exact_count: int | None = None,
) -> dict[str, Any]:
    """Fail closed on stale/corrupt resume state before model allocation."""
    import torch

    from audiolens.fitting import AudioFitContractError, config_digest

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise AudioFitContractError(f"invalid fit checkpoint at {path}") from exc
    if not isinstance(loaded, Mapping):
        raise AudioFitContractError(f"invalid fit checkpoint root at {path}")
    state = dict(loaded)
    fit = config["fit"]
    for key in ("source_layers", "target_layer", "skip_first"):
        if state.get(key) != fit[key]:
            raise AudioFitContractError(f"checkpoint {key} does not match fit config")
    n_done, next_idx = state.get("n_done"), state.get("next_idx")
    if (
        not isinstance(n_done, int)
        or n_done != next_idx
        or n_done < 0
        or n_done > maximum_count
        or (exact_count is not None and n_done != exact_count)
    ):
        raise AudioFitContractError(
            f"checkpoint counts {(n_done, next_idx)} violate the exact-success contract"
        )
    stamped_digest = state.get("fit_config_sha256")
    if stamped_digest is not None and stamped_digest != config_digest(dict(config)):
        raise AudioFitContractError("checkpoint fit-config digest does not match")
    jacobian_sum = state.get("jacobian_sum")
    expected_layers = set(fit["source_layers"])
    if not isinstance(jacobian_sum, dict) or set(jacobian_sum) != expected_layers:
        raise AudioFitContractError("checkpoint Jacobian layers do not match")
    expected_shape = (fit["d_model"], fit["d_model"])
    if any(
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or tuple(value.shape) != expected_shape
        or not bool(torch.isfinite(value).all())
        for value in jacobian_sum.values()
    ):
        raise AudioFitContractError("checkpoint sums are not finite fp32 fit matrices")
    return state


def _stamp_checkpoint(path: pathlib.Path, config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from audiolens.fitting import config_digest

    state = _read_checkpoint_state(
        path,
        config,
        maximum_count=config["dataset"]["requested_count"],
    )
    state["fit_config_sha256"] = config_digest(config)
    state["ordered_prompt_sha256"] = config["dataset"]["ordered_prompt_sha256"]
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return state


def _copy_checkpoint_atomic(source: pathlib.Path, destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def _tensor_geometry(value) -> dict[str, Any]:
    import torch

    matrix = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw_sha = hashlib.sha256(matrix.numpy().tobytes(order="C")).hexdigest()
    return {
        "dtype": "float32",
        "shape": list(matrix.shape),
        "sha256_raw_le": raw_sha,
        "frobenius_norm": float(torch.linalg.vector_norm(matrix)),
        "rms": float(torch.sqrt(torch.mean(matrix.square()))),
    }


def _stability_diagnostics(
    prefix_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    source_layers: list[int],
) -> dict[str, Any]:
    """Describe fp32 first-half, disjoint second-half, and full means."""
    import torch

    if prefix_state["n_done"] != 500 or final_state["n_done"] != 1_000:
        raise ValueError("canonical stability requires exact 500/1000 checkpoints")
    layers: dict[str, Any] = {}
    for layer in source_layers:
        first = prefix_state["jacobian_sum"][layer].float() / 500
        full = final_state["jacobian_sum"][layer].float() / 1_000
        second = (final_state["jacobian_sum"][layer] - prefix_state["jacobian_sum"][layer]).float() / 500
        layers[str(layer)] = {
            "first_half": _tensor_geometry(first),
            "second_half": _tensor_geometry(second),
            "full": _tensor_geometry(full),
            "first_second_cosine": float(
                torch.nn.functional.cosine_similarity(first.flatten(), second.flatten(), dim=0)
            ),
            "first_second_relative_l2": float(
                torch.linalg.vector_norm(second - first) / torch.linalg.vector_norm(first)
            ),
            "first_full_relative_l2": float(
                torch.linalg.vector_norm(full - first) / torch.linalg.vector_norm(first)
            ),
        }
    return {
        "kind": "fp32_prefix_500_disjoint_halves",
        "first_half_count": 500,
        "second_half_count": 500,
        "full_count": 1_000,
        "production_artifacts": [],
        "layers": layers,
    }


def _atomic_save_runtime_lens(lens, path: pathlib.Path, expected_count: int, d_model: int) -> None:
    from dataclasses import replace

    import torch

    from audiolens.fitting import validate_runtime_lens_file
    from audiolens.models import DEFAULT_MODEL_PROFILE

    profile = replace(DEFAULT_MODEL_PROFILE, d_model=d_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    lens.save(str(tmp), dtype=torch.float16)
    validate_runtime_lens_file(tmp, expected_count, profile=profile)
    os.replace(tmp, path)
    validate_runtime_lens_file(path, expected_count, profile=profile)


def _checkpoint_manifest_metadata(
    path: pathlib.Path,
    relative_path: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from audiolens.fitting import sha256_file

    state = _read_checkpoint_state(
        path,
        config,
        maximum_count=config["dataset"]["requested_count"],
        exact_count=config["dataset"]["requested_count"],
    )
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": "float32",
        "kind": "running_sum",
        "n_done": state["n_done"],
        "next_idx": state["next_idx"],
        "d_model": config["fit"]["d_model"],
        "source_layers": config["fit"]["source_layers"],
        "target_layer": config["fit"]["target_layer"],
        "skip_first": config["fit"]["skip_first"],
    }


def _lens_manifest_metadata(
    path: pathlib.Path,
    relative_path: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from audiolens.fitting import sha256_file

    fit = config["fit"]
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": fit["artifact_dtype"],
        "n_prompts": config["dataset"]["requested_count"],
        "d_model": fit["d_model"],
        "shape": [fit["d_model"], fit["d_model"]],
        "source_layers": fit["source_layers"],
        "target_layer": fit["target_layer"],
        "skip_first": fit["skip_first"],
        "max_seq_len": fit["max_seq_len"],
        "dim_batch": fit["dim_batch"],
    }


def _validate_pinned_fit_config(config: Mapping[str, Any]) -> None:
    """Reject self-consistent manifests that do not describe this recipe."""
    from audiolens.fitting import (
        AudioFitContractError,
        JLENS_REVISION,
        MODEL_REVISION,
        WIKITEXT_REVISION,
    )
    from audiolens.models import DEFAULT_MODEL_PROFILE

    profile = DEFAULT_MODEL_PROFILE

    expected = {
        "schema_version",
        "model",
        "tokenizer",
        "dataset",
        "jlens",
        "prompt_policy",
        "fit",
        "lock",
        "source",
        "runtime",
    }
    if set(config) != expected or config.get("schema_version") != FIT_MANIFEST_SCHEMA_VERSION:
        raise AudioFitContractError("fit config schema is invalid")
    if config.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise AudioFitContractError("fit model identity is not pinned")
    if config.get("tokenizer") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise AudioFitContractError("fit tokenizer identity is not pinned")
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise AudioFitContractError("fit dataset config is invalid")
    requested = dataset.get("requested_count")
    prompt_sha = dataset.get("ordered_prompt_sha256")
    expected_dataset = {
        "id": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "text_field": DATASET_TEXT_FIELD,
        "revision": WIKITEXT_REVISION,
        "streaming": True,
        "trust_remote_code": False,
        "chunking": {
            "algorithm": CHUNK_ALGORITHM,
            "max_chars": CHUNK_CHARS,
            "min_tail_chars": MIN_TAIL_CHARS,
        },
        "requested_count": requested,
        "ordered_prompt_sha256": prompt_sha,
    }
    if (
        dataset != expected_dataset
        or not isinstance(requested, int)
        or requested <= 0
        or not isinstance(prompt_sha, str)
        or len(prompt_sha) != 64
    ):
        raise AudioFitContractError("fit dataset/corpus identity is not pinned")
    if config.get("jlens") != {"revision": JLENS_REVISION}:
        raise AudioFitContractError("fit JLens revision is not pinned")
    if config.get("prompt_policy") != {
        "kind": "raw_text",
        "force_bos": True,
        "chat_template": False,
    }:
        raise AudioFitContractError("fit raw-text/BOS policy is not pinned")
    if config.get("fit") != {
        "source_layers": list(profile.source_layers),
        "target_layer": profile.target_layer,
        "skip_first": profile.skip_first,
        "max_seq_len": profile.max_sequence_length,
        "dim_batch": profile.dimension_batch_size,
        "checkpoint_every": CHECKPOINT_EVERY,
        "resume": True,
        "compile": False,
        "d_model": D_MODEL,
        "model_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "artifact_dtype": "float16",
        "attention_backend": "eager",
    }:
        raise AudioFitContractError("fit estimator geometry/dtypes are not pinned")
    if config.get("lock") != {
        "uv_lock_sha256": LOCK_SHA256,
        "frozen": True,
        "dependency_group": "fit",
    }:
        raise AudioFitContractError("fit lock identity is not pinned")
    source = config.get("source")
    runtime = config.get("runtime")
    if (
        not isinstance(source, dict)
        or set(source) != {"git_revision", "digest"}
        or not isinstance(source["git_revision"], str)
        or len(source["git_revision"]) != 40
        or not isinstance(source["digest"], str)
        or len(source["digest"]) != 64
        or not isinstance(runtime, dict)
        or not runtime
        or runtime.get("modal_function_timeout_seconds")
        != MODAL_FUNCTION_TIMEOUT_SECONDS
    ):
        raise AudioFitContractError("fit source/runtime identity is invalid")


def _validate_completed_manifest(
    record: Mapping[str, Any],
    volume_root: str | pathlib.Path,
) -> dict[str, Any]:
    """Validate the immutable identity and bound runtime lens artifact."""
    from audiolens.fitting import (
        AudioFitContractError,
        config_digest,
        sha256_file,
        validate_runtime_lens_file,
    )
    from dataclasses import replace

    from audiolens.models import DEFAULT_MODEL_PROFILE

    config = record.get("config")
    if not isinstance(config, dict):
        raise AudioFitContractError("fit manifest has no immutable config")
    _validate_pinned_fit_config(config)
    expected_paths = _run_paths(config)
    if (
        record.get("schema_version") != FIT_MANIFEST_SCHEMA_VERSION
        or record.get("kind") != FIT_MANIFEST_KIND
        or record.get("status") != "complete"
        or record.get("fit_config_sha256") != config_digest(config)
        or record.get("paths") != expected_paths
    ):
        raise AudioFitContractError("completed fit manifest identity is invalid")
    requested = config["dataset"]["requested_count"]
    expected_canonical = requested == CANONICAL_PROMPT_COUNT
    if record.get("canonical") is not expected_canonical:
        raise AudioFitContractError("canonical flag does not match the successful count")
    lens = record.get("lens")
    if not isinstance(lens, dict):
        raise AudioFitContractError("completed fit manifest has no lens binding")
    root = pathlib.Path(volume_root)
    lens_path = root / expected_paths["lens"]
    if not lens_path.is_file():
        raise AudioFitContractError(f"bound lens artifact is missing at {lens_path}")
    try:
        expected_lens = _lens_manifest_metadata(
            lens_path,
            expected_paths["lens"],
            config,
        )
        if lens != expected_lens:
            raise AudioFitContractError(
                "lens metadata does not match its bound artifact"
            )
        profile = replace(
            DEFAULT_MODEL_PROFILE,
            d_model=config["fit"]["d_model"],
        )
        validate_runtime_lens_file(lens_path, requested, profile=profile)
        if sha256_file(lens_path) != lens["sha256"]:
            raise AudioFitContractError("lens SHA-256 changed during validation")
    except AudioFitContractError:
        raise
    except Exception as exc:
        raise AudioFitContractError(
            f"bound lens artifact is invalid at {lens_path}"
        ) from exc
    checkpoint = record.get("checkpoint")
    required_checkpoint_fields = {
        "relative_path",
        "sha256",
        "bytes",
        "dtype",
        "kind",
        "n_done",
        "next_idx",
        "d_model",
        "source_layers",
        "target_layer",
        "skip_first",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required_checkpoint_fields:
        raise AudioFitContractError("completed fit manifest checkpoint binding is invalid")
    expected_checkpoint = {
        "relative_path": expected_paths["checkpoint"],
        "dtype": "float32",
        "kind": "running_sum",
        "n_done": requested,
        "next_idx": requested,
        "d_model": config["fit"]["d_model"],
        "source_layers": config["fit"]["source_layers"],
        "target_layer": config["fit"]["target_layer"],
        "skip_first": config["fit"]["skip_first"],
    }
    if (
        any(checkpoint.get(key) != value for key, value in expected_checkpoint.items())
        or not isinstance(checkpoint.get("bytes"), int)
        or checkpoint["bytes"] <= 0
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint["sha256"])
    ):
        raise AudioFitContractError("fit manifest checkpoint metadata is invalid")
    if requested == CANONICAL_PROMPT_COUNT:
        stability = record.get("stability")
        if (
            not isinstance(stability, dict)
            or stability.get("kind") != "fp32_prefix_500_disjoint_halves"
            or stability.get("first_half_count") != 500
            or stability.get("second_half_count") != 500
            or stability.get("full_count") != 1_000
            or stability.get("production_artifacts") != []
            or set(stability.get("layers", {})) != {
                str(layer) for layer in config["fit"]["source_layers"]
            }
        ):
            raise AudioFitContractError("canonical stability diagnostics are invalid")
        import math

        geometry_keys = {
            "dtype",
            "shape",
            "sha256_raw_le",
            "frobenius_norm",
            "rms",
        }
        layer_keys = {
            "first_half",
            "second_half",
            "full",
            "first_second_cosine",
            "first_second_relative_l2",
            "first_full_relative_l2",
        }
        for layer_record in stability["layers"].values():
            if set(layer_record) != layer_keys:
                raise AudioFitContractError("canonical layer diagnostics are invalid")
            for name in ("first_half", "second_half", "full"):
                geometry = layer_record[name]
                raw_sha = (
                    geometry.get("sha256_raw_le")
                    if isinstance(geometry, dict)
                    else None
                )
                if (
                    not isinstance(geometry, dict)
                    or set(geometry) != geometry_keys
                    or geometry.get("dtype") != "float32"
                    or geometry.get("shape")
                    != [config["fit"]["d_model"], config["fit"]["d_model"]]
                    or not isinstance(raw_sha, str)
                    or len(raw_sha) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in raw_sha
                    )
                    or not math.isfinite(
                        geometry.get("frobenius_norm", float("nan"))
                    )
                    or not math.isfinite(geometry.get("rms", float("nan")))
                ):
                    raise AudioFitContractError(
                        "canonical matrix geometry diagnostics are invalid"
                    )
            if any(
                not math.isfinite(layer_record[name])
                for name in (
                    "first_second_cosine",
                    "first_second_relative_l2",
                    "first_full_relative_l2",
                )
            ):
                raise AudioFitContractError(
                    "canonical stability values are nonfinite"
                )
    elif record.get("stability") is not None:
        raise AudioFitContractError("noncanonical fit must not carry canonical diagnostics")
    return dict(record)


def load_completed_fit_manifest(
    path: str | pathlib.Path,
    *,
    volume_root: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Load a completed manifest and verify its bound fp16 lens.

    ``volume_root`` is the directory against which the manifest's portable
    relative paths are resolved.  For a normal ``<root>/runs/<manifest>``
    layout it is inferred automatically.
    """
    from audiolens.fitting import AudioFitContractError

    manifest_path = pathlib.Path(path)
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioFitContractError(f"invalid fit manifest at {manifest_path}") from exc
    if not isinstance(loaded, Mapping):
        raise AudioFitContractError(
            f"invalid fit manifest root at {manifest_path}"
        )
    record = dict(loaded)
    root = pathlib.Path(volume_root) if volume_root is not None else manifest_path.parent.parent
    validated = _validate_completed_manifest(record, root)
    expected_manifest = root / validated["paths"]["manifest"]
    if manifest_path.resolve() != expected_manifest.resolve():
        raise AudioFitContractError("fit manifest is not at its content-addressed path")
    return validated


def _cleanup_completed_prefix_snapshot(
    record: Mapping[str, Any],
    volume_root: str | pathlib.Path,
) -> bool:
    """Remove the private prefix snapshot only after canonical completion."""
    from audiolens.fitting import AudioFitContractError

    if record.get("status") != "complete" or record.get("canonical") is not True:
        return False
    stability = record.get("stability")
    if (
        not isinstance(stability, dict)
        or stability.get("kind") != "fp32_prefix_500_disjoint_halves"
    ):
        raise AudioFitContractError(
            "refusing to delete prefix snapshot before diagnostics complete"
        )
    config = record.get("config")
    if not isinstance(config, dict):
        raise AudioFitContractError("completed fit has no config for prefix cleanup")
    prefix_path = pathlib.Path(volume_root) / _diagnostic_prefix_checkpoint(config)
    if not prefix_path.exists():
        return False
    prefix_path.unlink()
    return True


def _persist_completed_manifest(
    manifest_path: pathlib.Path,
    record: dict[str, Any],
    volume_root: pathlib.Path,
) -> None:
    """Commit diagnostics first, then commit deletion of the temporary prefix."""
    from audiolens.fitting import atomic_write_json

    atomic_write_json(manifest_path, record)
    _commit_volume()
    if _cleanup_completed_prefix_snapshot(record, volume_root):
        _commit_volume()


def _complete_fit_manifest(
    manifest_path: pathlib.Path,
    record: dict[str, Any],
    volume_root: pathlib.Path,
) -> dict[str, Any]:
    import jlens

    config = record["config"]
    paths = record["paths"]
    requested = config["dataset"]["requested_count"]
    checkpoint_path = volume_root / paths["checkpoint"]
    final_state = _read_checkpoint_state(
        checkpoint_path,
        config,
        maximum_count=requested,
        exact_count=requested,
    )
    jacobians = {
        layer: value.float() / requested
        for layer, value in final_state["jacobian_sum"].items()
    }
    lens = jlens.JacobianLens(
        jacobians=jacobians,
        n_prompts=requested,
        d_model=config["fit"]["d_model"],
    )
    lens_path = volume_root / paths["lens"]
    _atomic_save_runtime_lens(lens, lens_path, requested, config["fit"]["d_model"])

    stability = None
    if requested == CANONICAL_PROMPT_COUNT:
        prefix_path = volume_root / _diagnostic_prefix_checkpoint(config)
        prefix_state = _read_checkpoint_state(
            prefix_path,
            config,
            maximum_count=500,
            exact_count=500,
        )
        stability = _stability_diagnostics(
            prefix_state,
            final_state,
            config["fit"]["source_layers"],
        )

    complete = copy.deepcopy(record)
    complete.update(
        {
            "status": "complete",
            "canonical": requested == CANONICAL_PROMPT_COUNT,
            "lens": _lens_manifest_metadata(lens_path, paths["lens"], config),
            "checkpoint": _checkpoint_manifest_metadata(
                checkpoint_path,
                paths["checkpoint"],
                config,
            ),
            "stability": stability,
        }
    )
    validated = _validate_completed_manifest(complete, volume_root)
    _persist_completed_manifest(manifest_path, validated, volume_root)
    return validated


def _fit_lens_impl(
    n_prompts: int,
    *,
    volume_root: str | pathlib.Path = VOL_MOUNT,
    rows: Iterable[Mapping[str, Any]] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, str] | None = None,
    model_loader=None,
) -> dict[str, Any]:
    """Prepare identity, resume the exact estimator, and bind the artifact."""
    import jlens

    from audiolens.fitting import AudioFitContractError

    prompts = _wikitext_prompts(n_prompts, rows=rows)
    config = _build_fit_config(
        prompts,
        requested_count=n_prompts,
        runtime_identity=runtime_identity,
        source_identity=source_identity,
    )
    root = pathlib.Path(volume_root)
    manifest_path, paths, record = _initialize_fit_manifest(root, config)
    if record["status"] == "complete":
        completed = load_completed_fit_manifest(manifest_path, volume_root=root)
        if _cleanup_completed_prefix_snapshot(completed, root):
            _commit_volume()
        return completed

    checkpoint_path = root / paths["checkpoint"]
    prefix_path = root / _diagnostic_prefix_checkpoint(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    (root / paths["lens"]).parent.mkdir(parents=True, exist_ok=True)

    current_count = 0
    if checkpoint_path.exists():
        current_state = _read_checkpoint_state(
            checkpoint_path,
            config,
            maximum_count=n_prompts,
        )
        current_count = current_state["n_done"]
    if n_prompts == CANONICAL_PROMPT_COUNT:
        if prefix_path.exists():
            _read_checkpoint_state(prefix_path, config, maximum_count=500, exact_count=500)
        elif current_count > 500:
            raise AudioFitContractError(
                "canonical resume passed 500 without the required fp32 prefix snapshot"
            )
        elif current_count == 500:
            _copy_checkpoint_atomic(checkpoint_path, prefix_path)

    if current_count == n_prompts:
        return _complete_fit_manifest(manifest_path, record, root)

    loader = _load_model if model_loader is None else model_loader
    tokenizer, hf = loader()
    model = jlens.from_hf(
        hf,
        tokenizer,
        layout=None,
        text_module=None,
        compile=False,
        force_bos=True,
    )
    fit = config["fit"]
    if model.n_layers != 35 or model.d_model != fit["d_model"]:
        raise AudioFitContractError(
            f"pinned model geometry changed to L={model.n_layers}, d={model.d_model}"
        )

    prefix_counts = (500, n_prompts) if n_prompts == CANONICAL_PROMPT_COUNT else (n_prompts,)
    for prefix_count in prefix_counts:
        if current_count >= prefix_count:
            continue
        lens = jlens.fit(
            model,
            prompts=prompts[:prefix_count],
            source_layers=fit["source_layers"],
            target_layer=fit["target_layer"],
            dim_batch=fit["dim_batch"],
            max_seq_len=fit["max_seq_len"],
            skip_first=fit["skip_first"],
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=fit["checkpoint_every"],
            resume=fit["resume"],
        )
        if lens.n_prompts != prefix_count:
            raise AudioFitContractError(
                f"fit returned {lens.n_prompts}/{prefix_count} successful prompts"
            )
        state = _stamp_checkpoint(checkpoint_path, config)
        if state["n_done"] != prefix_count or state["next_idx"] != prefix_count:
            raise AudioFitContractError(
                f"fit checkpoint did not reach exact prefix {prefix_count}"
            )
        current_count = prefix_count
        if prefix_count == 500 and n_prompts == CANONICAL_PROMPT_COUNT:
            _copy_checkpoint_atomic(checkpoint_path, prefix_path)
            _read_checkpoint_state(prefix_path, config, maximum_count=500, exact_count=500)
        _commit_volume()

    return _complete_fit_manifest(manifest_path, record, root)


@_modal_fit_function
def fit_lens(n_prompts: int = CANONICAL_PROMPT_COUNT) -> str:
    record = _fit_lens_impl(n_prompts)
    return json.dumps(
        {
            "manifest": record["paths"]["manifest"],
            "fit_config_sha256": record["fit_config_sha256"],
            "lens": record["lens"],
            "canonical": record["canonical"],
        },
        indent=2,
        sort_keys=True,
    )


@_modal_local_entrypoint
def main(n_prompts: int = CANONICAL_PROMPT_COUNT):
    result = fit_lens.remote(n_prompts=n_prompts)
    print(result)
    payload = json.loads(result)
    print(
        "download manifest: modal volume get audiolens-vol "
        f"{payload['manifest']} runs/"
    )
    print(
        "download lens: modal volume get audiolens-vol "
        f"{payload['lens']['relative_path']} lenses/"
    )
