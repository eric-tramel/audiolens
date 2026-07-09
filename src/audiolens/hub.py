"""Prepare, publish, and download minimal Hugging Face lens bundles."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import shutil
from collections.abc import Callable, Sequence
from typing import Any

DEFAULT_ARTIFACT_LICENSE = "cc-by-sa-4.0"
RUN_MANIFEST_NAME = "audiolens-run.json"
MODEL_CARD_NAME = "README.md"
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*\Z")


class HubBundleError(ValueError):
    """A run or bundle violates the public artifact contract."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lens_names(lenses: Sequence[str] | str) -> tuple[str, ...]:
    values = lenses.split(",") if isinstance(lenses, str) else list(lenses)
    if any(not isinstance(value, str) for value in values):
        raise HubBundleError("runtime lens names must be strings")
    names = tuple(value.strip() for value in values)
    if not names or any(not name for name in names):
        raise HubBundleError("at least one non-empty runtime lens name is required")
    if len(set(names)) != len(names):
        raise HubBundleError("selected runtime lens names must be unique")
    return names


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HubBundleError(f"cannot read JSON metadata at {path}: {error}") from error
    if not isinstance(value, dict):
        raise HubBundleError(f"{path}: expected a JSON object")
    return value


def _validate_output(
    name: str, spec: Any, *, source_path: pathlib.Path
) -> tuple[pathlib.Path, dict[str, Any]]:
    if not isinstance(spec, dict):
        raise HubBundleError(f"output {name!r} must be an object")
    if spec.get("kind") != "runtime_lens":
        raise HubBundleError(f"output {name!r} is not a runtime lens")
    declared_path = spec.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise HubBundleError(f"output {name!r} has no path")
    path = pathlib.Path(declared_path)
    if not path.is_absolute():
        path = source_path.parent / path
    if path.suffix != ".pt":
        raise HubBundleError(f"runtime lens {name!r} must be a .pt file")
    if not path.is_file():
        raise HubBundleError(f"runtime lens {name!r} does not exist at {path}")
    declared_bytes = spec.get("bytes")
    if type(declared_bytes) is not int or declared_bytes < 0:
        raise HubBundleError(f"runtime lens {name!r} has an invalid declared byte size")
    actual_bytes = path.stat().st_size
    if actual_bytes != declared_bytes:
        raise HubBundleError(
            f"runtime lens {name!r} byte-size mismatch: expected {declared_bytes}, "
            f"found {actual_bytes}"
        )
    expected_sha256 = spec.get("sha256")
    if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
        raise HubBundleError(f"runtime lens {name!r} has an invalid SHA-256")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise HubBundleError(
            f"runtime lens {name!r} SHA-256 mismatch: expected "
            f"{expected_sha256.lower()}, found {actual_sha256}"
        )
    return path, spec


def _remove_modal_environment(run: dict[str, Any]) -> None:
    runtime = run.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("modal_environment", None)
    config = run.get("config")
    if isinstance(config, dict):
        runtime = config.get("runtime")
        if isinstance(runtime, dict):
            runtime.pop("modal_environment", None)


def _assert_public_metadata(value: Any, *, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == "modal_environment":
                raise HubBundleError("sanitized metadata still contains modal_environment")
            _assert_public_metadata(child, key=child_key)
    elif isinstance(value, list):
        for child in value:
            _assert_public_metadata(child, key=key)
    elif isinstance(value, str):
        if "MODAL_IMAGE_ID" in value:
            raise HubBundleError("sanitized metadata still contains a Modal image ID")
        if pathlib.PurePosixPath(value).is_absolute():
            raise HubBundleError(f"sanitized metadata contains an absolute path in {key!r}")


def _model_card(
    manifest: dict[str, Any], names: tuple[str, ...], artifact_license: str
) -> str:
    outputs = manifest["outputs"]
    files = "\n".join(
        f"- `{outputs[name]['path']}` — `{name}`, {outputs[name]['bytes']} bytes, "
        f"SHA-256 `{outputs[name]['sha256']}`"
        for name in names
    )
    return f"""---
license: {artifact_license}
library_name: audiolens
base_model: google/gemma-4-E2B-it
datasets:
- Salesforce/wikitext
- openslr/librispeech_asr
tags:
- interpretability
- audio
- jacobian-lens
---

# Audiolens runtime lenses

These are selected runtime Jacobian lenses for `google/gemma-4-E2B-it`.
The base model and its use remain governed by the upstream Gemma model card
and Apache-2.0 license. Fitting and bundling code comes from
[audiolens](https://github.com/eric-tramel/audiolens) under the MIT License;
the Jacobian Lens implementation comes from
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) under
Apache-2.0. The lens files in this repository are offered under
`{artifact_license}`.

## Training data provenance

The text side uses the pinned `Salesforce/wikitext` WikiText-103 revision,
derived from Wikipedia with CC BY-SA/GFDL provenance. The audio side uses the
pinned `openslr/librispeech_asr` LibriSpeech revision (CC BY 4.0). Exact
revisions, ordered corpus hashes, source identity, and fit settings are in
`{RUN_MANIFEST_NAME}`. Dataset files and training checkpoints are not included.

RAVDESS is not training data. No RAVDESS audio, other evaluation dataset, or
evaluation result is included in this repository.

## Files

{files}
- `{RUN_MANIFEST_NAME}` — sanitized completed-run metadata for only the lenses above

Verify a lens against its pinned revision and expected SHA-256 before loading it.
"""


def prepare_hf_bundle(
    run_json: str | pathlib.Path,
    bundle_dir: str | pathlib.Path,
    lenses: Sequence[str] | str = ("text400", "mixed528"),
    artifact_license: str = DEFAULT_ARTIFACT_LICENSE,
) -> pathlib.Path:
    """Validate a completed fit run and build a minimal publishable directory.

    The destination must not already exist. Only selected ``runtime_lens``
    outputs, a generated model card, and sanitized run metadata are written.
    """
    if not isinstance(artifact_license, str) or not _LICENSE.fullmatch(artifact_license):
        raise HubBundleError(f"invalid artifact license identifier {artifact_license!r}")
    names = _lens_names(lenses)
    run_path = pathlib.Path(run_json)
    run = _read_json(run_path)
    if run.get("completed") is not True:
        raise HubBundleError(f"run is not complete: {run_path}")
    outputs = run.get("outputs")
    if not isinstance(outputs, dict):
        raise HubBundleError(f"{run_path}: outputs must be an object")

    selected: list[tuple[str, pathlib.Path, dict[str, Any]]] = []
    basenames: set[str] = set()
    for name in names:
        if name not in outputs:
            raise HubBundleError(f"completed run has no output named {name!r}")
        source, spec = _validate_output(name, outputs[name], source_path=run_path)
        if source.name in basenames:
            raise HubBundleError(f"selected outputs do not have unique filenames: {source.name}")
        basenames.add(source.name)
        selected.append((name, source, spec))

    destination = pathlib.Path(bundle_dir)
    if destination.exists():
        raise HubBundleError(f"bundle destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.preparing")
    if temporary.exists():
        raise HubBundleError(f"temporary bundle destination already exists: {temporary}")

    sanitized = copy.deepcopy(run)
    sanitized_outputs: dict[str, dict[str, Any]] = {}
    for name, source, spec in selected:
        public_spec = copy.deepcopy(spec)
        public_spec["path"] = source.name
        sanitized_outputs[name] = public_spec
    sanitized["outputs"] = sanitized_outputs
    sanitized["artifact_license"] = artifact_license
    _remove_modal_environment(sanitized)
    _assert_public_metadata(sanitized)

    try:
        temporary.mkdir()
        for _name, source, _spec in selected:
            shutil.copyfile(source, temporary / source.name)
        (temporary / RUN_MANIFEST_NAME).write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / MODEL_CARD_NAME).write_text(
            _model_card(sanitized, names, artifact_license), encoding="utf-8"
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _validate_bundle(bundle_dir: pathlib.Path) -> dict[str, Any]:
    if not bundle_dir.is_dir():
        raise HubBundleError(f"bundle directory does not exist: {bundle_dir}")
    manifest_path = bundle_dir / RUN_MANIFEST_NAME
    model_card = bundle_dir / MODEL_CARD_NAME
    if not model_card.is_file():
        raise HubBundleError(f"bundle has no {MODEL_CARD_NAME}")
    manifest = _read_json(manifest_path)
    if manifest.get("completed") is not True:
        raise HubBundleError("bundle run metadata is not complete")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise HubBundleError("bundle has no selected runtime lenses")
    expected = {RUN_MANIFEST_NAME, MODEL_CARD_NAME}
    output_files: set[str] = set()
    for name, spec in outputs.items():
        source, _ = _validate_output(name, spec, source_path=manifest_path)
        if source.parent != bundle_dir:
            raise HubBundleError(f"bundle output {name!r} escapes the bundle directory")
        if source.name in output_files:
            raise HubBundleError(
                f"bundle outputs do not have unique filenames: {source.name}"
            )
        output_files.add(source.name)
        expected.add(source.name)
    actual = {entry.name for entry in bundle_dir.iterdir()}
    if any(entry.is_symlink() or not entry.is_file() for entry in bundle_dir.iterdir()):
        raise HubBundleError("bundle may contain only regular top-level files")
    if actual != expected:
        extras = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise HubBundleError(f"bundle contents differ from manifest; extras={extras}, missing={missing}")
    _assert_public_metadata(manifest)
    return manifest


def publish_hf_bundle(
    bundle_dir: str | pathlib.Path,
    repo_id: str,
    *,
    private: bool = False,
    api: Any | None = None,
) -> Any:
    """Create a Hugging Face model repository if needed and upload a bundle.

    Authentication is delegated to ``huggingface_hub``'s standard environment
    and cache handling. No token argument is accepted or logged.
    """
    bundle = pathlib.Path(bundle_dir)
    _validate_bundle(bundle)
    if not isinstance(repo_id, str) or "/" not in repo_id or repo_id.strip() != repo_id:
        raise HubBundleError("repo_id must be a Hugging Face namespace/repository identifier")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(bundle),
        commit_message="Publish verified Audiolens runtime lenses",
    )


def download_lens(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    expected_sha256: str | None = None,
    verify_checksum: bool = True,
    cache_dir: str | pathlib.Path | None = None,
    download_fn: Callable[..., str] | None = None,
) -> pathlib.Path:
    """Download one runtime lens and optionally require its expected SHA-256.

    For a trust anchor, callers should pin both ``revision`` and
    ``expected_sha256``. Set ``verify_checksum=False`` only when integrity is
    deliberately handled elsewhere. Authentication uses Hugging Face's
    standard environment; this API intentionally has no token parameter.
    """
    if not isinstance(filename, str) or pathlib.PurePosixPath(filename).name != filename:
        raise HubBundleError("filename must be one top-level bundle filename")
    if not filename.endswith(".pt"):
        raise HubBundleError("filename must identify a .pt runtime lens")
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256):
            raise HubBundleError("expected_sha256 must contain 64 hexadecimal characters")
        expected_sha256 = expected_sha256.lower()
    elif verify_checksum:
        raise HubBundleError("expected_sha256 is required when verify_checksum is enabled")
    if download_fn is None:
        from huggingface_hub import hf_hub_download

        download_fn = hf_hub_download
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "filename": filename,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    downloaded = pathlib.Path(download_fn(**kwargs))
    if not downloaded.is_file():
        raise HubBundleError(f"downloaded lens is not a file: {downloaded}")
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(downloaded)
        if actual_sha256 != expected_sha256:
            raise HubBundleError(
                f"downloaded lens SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
    return downloaded
