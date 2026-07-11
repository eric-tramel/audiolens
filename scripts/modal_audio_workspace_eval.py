"""Frozen Modal orchestration for the audio fixed-band readout experiment.

The module is deliberately import-light. Model, lens, audio, SciPy, SoundFile,
and Whisper imports occur only inside remote execution paths. Confirmatory
records remain in memory until the complete report has been reconstructed and
validated; no partial score artifact is written.
"""

from __future__ import annotations
import contextlib

import hashlib
import inspect
import json
import os
import pathlib
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
VOLUME_NAME = "audiolens-vol"
APP_NAME = "audiolens-audio-workspace-eval"
MODEL_GPU = "H100"

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "70af34e20bd4b7a91f0de6b22675850c43922a03"
WHISPER_ID = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

TTS_ENGINE = "openai-tts"
TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_RESPONSE_FORMAT = "wav"
TTS_INPUT_POLICY = "strip_double_quotes_collapse_whitespace"
TTS_SYNTHESIS_POLICY = "sealed_source_bytes_nonreproducible_generation"
TTS_RECIPE_KIND = "audio_workspace_tts_recipe"
TTS_VARIANTS = ("onyx", "nova")
SOURCE_SAMPLE_RATE = 24_000
SAMPLE_RATE = 16_000
RESAMPLE_UP = 2
RESAMPLE_DOWN = 3

FINAL_AUDIO_FIT_CONFIG_SHA256 = "ee7cd4e42991fec5a00b4256ba466ff163ebd64fa22963334033066e7d531275"
FINAL_AUDIO_LENS_SHA256 = "da0ccabf1ee14e4df060f97f31cf0132a0d3f6ed2cb45b6c77738693bc8f1aa9"
FINAL_AUDIO_RUN_PATH = f"{VOL_MOUNT}/audio-fit-runs/{FINAL_AUDIO_FIT_CONFIG_SHA256}/run.json"
CANONICAL_TEXT_EVALUATION_SHA256 = (
    "6284af5fa0f5ae5753cbc8831e3cfed7be111412824dc16a1c4218f70fca393a"
)
CANONICAL_TEXT_REPORT_SHA256 = "16413a02d95aeada2e7a36a70fe100cc61c1d7138620ae0fdd1f0fdcd4791028"
CANONICAL_TEXT_REPORT_PATH = (
    f"{VOL_MOUNT}/eval/gemma-4-E2B-it-workspace-{CANONICAL_TEXT_EVALUATION_SHA256}.json"
)

ALL_LAYERS = tuple(range(34))
FINAL_MODEL_LAYER = 34
POSITIONS = ("last_processor_valid_audio_position", "response_position")
CONTROLS = ("candidate", "logit", "transposed", "permuted")
EARLY_LAYERS = tuple(range(13))
CANDIDATE_LAYERS = tuple(range(13, 32))
MOTOR_LAYERS = (32, 33)
KS = (1, 2, 5, 10, 20, 50, 100)
CONTROL_SEED = 2026070903
PERMUTATION_SEED = 2026070901
BOOTSTRAP_SEED = 2026070902
PERMUTATION_REPLICATES = 10_000
BOOTSTRAP_REPLICATES = 10_000
MAX_SEQUENCE_LENGTH = 512
EXPECTED_ITEM_COUNT = 259
EXPECTED_OBSERVATION_COUNT = 518
EXPECTED_CALIBRATION_CELLS = 68
CALIBRATION_MACRO_CER_MAX = 0.35
CALIBRATION_CELL_CER_MAX = 0.80
MAX_FIXTURE_BYTES = 64 * 1024
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_LENS_BYTES = 192 * 1024 * 1024

ARTIFACT_ROOT = f"{VOL_MOUNT}/audio-workspace-eval"
STIMULUS_ROOT = f"{ARTIFACT_ROOT}/stimuli"
CALIBRATION_ROOT = f"{ARTIFACT_ROOT}/calibrations"
PREREGISTRATION_ROOT = f"{ARTIFACT_ROOT}/preregistrations"
REPORT_ROOT = f"{ARTIFACT_ROOT}/reports"
SMOKE_ROOT = f"{ARTIFACT_ROOT}/smoke"
SOURCE_STIMULI_ROOT = f"{ARTIFACT_ROOT}/source-stimuli"
FIXTURE_CACHE_ROOT = f"{ARTIFACT_ROOT}/fixture-cache/{JLENS_REVISION}"

FIXTURE_SPECS = (
    {
        "distribution": "association",
        "filename": "lens-eval-association.json",
        "sha256": "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
        "n_bytes": 24_228,
        "raw_count": 102,
        "publication_count": 50,
        "selected_name_sha256": "107eddeee767b029528f200718167bed2f76fd25a0e08b54ea9289da92066e68",
    },
    {
        "distribution": "multihop",
        "filename": "lens-eval-multihop.json",
        "sha256": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
        "n_bytes": 21_869,
        "raw_count": 93,
        "publication_count": 50,
        "selected_name_sha256": "377d116630bffe505157e408906cc811860e771db509261a67f1c4188b51d033",
    },
    {
        "distribution": "multilingual",
        "filename": "lens-eval-multilingual.json",
        "sha256": "fa70b9bd89416a6d8d985a80dc628b109ae6fd3b25b9275c0fc5065d7ff4a0ef",
        "n_bytes": 24_284,
        "raw_count": 107,
        "publication_count": 54,
        "selected_name_sha256": "ed559e9071ff5b381febe7980447213577fc6fd4d212490241e54da7072849c8",
    },
    {
        "distribution": "order-ops",
        "filename": "lens-eval-order-ops.json",
        "sha256": "b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560",
        "n_bytes": 9_589,
        "raw_count": 55,
        "publication_count": 55,
        "selected_name_sha256": "73f146f5950d27f23abc778da086604d727d5f4b2aa7efd14658dd9fc9e5082d",
    },
    {
        "distribution": "poetry",
        "filename": "lens-eval-poetry.json",
        "sha256": "6aeb3415c5a5c3f3827c9efe63f006de02f5ef39a816bbac68e15e733aba60cc",
        "n_bytes": 21_533,
        "raw_count": 98,
        "publication_count": 52,
        "selected_name_sha256": "f1e324ec4d18814473f8ac7fea68464cb48106d5d2ae83eddb7db3ae6f68f0b9",
    },
)

LANGUAGE_CODES = {
    "arabic": "ar",
    "bengali": "bn",
    "bulgarian": "bg",
    "chinese": "cmn",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "norwegian": "nb",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "slovak": "sk",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "vietnamese": "vi",
}

LANGUAGE_TO_WHISPER = {
    **LANGUAGE_CODES,
    "chinese": "zh",
    "norwegian": "no",
}

SOURCE_RELATIVES = (
    "pyproject.toml",
    "uv.lock",
    "src/audiolens/__init__.py",
    "src/audiolens/audio_eval_model.py",
    "src/audiolens/audio_fitting.py",
    "src/audiolens/audio_workspace_eval.py",
    "src/audiolens/models/__init__.py",
    "src/audiolens/models/base.py",
    "src/audiolens/models/gemma4.py",
    "scripts/modal_audio_workspace_eval.py",
    "scripts/modal_workspace_eval.py",
    "scripts/synthesize_audio_stimuli.py",
)

CANONICAL_FIT_SOURCE_RELATIVES = (
    "pyproject.toml",
    "uv.lock",
    "src/audiolens/__init__.py",
    "src/audiolens/fitting.py",
    "src/audiolens/models/__init__.py",
    "src/audiolens/models/base.py",
    "src/audiolens/models/gemma4.py",
    "scripts/modal_fit_lens.py",
)
CANONICAL_WORKSPACE_SOURCE_RELATIVES = (
    *CANONICAL_FIT_SOURCE_RELATIVES,
    "scripts/modal_workspace_eval.py",
)


class ModalAudioWorkspaceEvalError(RuntimeError):
    """Fail-closed deployment, artifact, or deterministic-scoring violation."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded_bytes(
    path: str | pathlib.Path,
    *,
    label: str,
    maximum: int,
) -> bytes:
    candidate = pathlib.Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ModalAudioWorkspaceEvalError(f"missing or unsafe {label} at {candidate}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum:
            raise ModalAudioWorkspaceEvalError(f"{label} has invalid byte size")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            payload = source.read(maximum + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or len(payload) > maximum
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ino != before.st_ino
        ):
            raise ModalAudioWorkspaceEvalError(f"{label} changed while it was read")
        return payload
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _verified_local_copy(
    path: str | pathlib.Path,
    *,
    expected_sha256: str,
    label: str,
    maximum: int,
    suffix: str,
) -> Iterator[pathlib.Path]:
    payload = _read_bounded_bytes(path, label=label, maximum=maximum)
    if _sha256_bytes(payload) != expected_sha256:
        raise ModalAudioWorkspaceEvalError(f"{label} bytes changed")
    with tempfile.NamedTemporaryFile(suffix=suffix) as local:
        local.write(payload)
        local.flush()
        yield pathlib.Path(local.name)


def _relative_source_digest(relatives: Sequence[str], *, nul_separator: bool) -> str | None:
    if not all((REPO_ROOT / relative).is_file() for relative in relatives):
        return None
    digest = hashlib.sha256()
    for relative in relatives:
        digest.update(relative.encode("utf-8"))
        if nul_separator:
            digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
    return digest.hexdigest()


def _source_digest() -> str | None:
    return _relative_source_digest(
        SOURCE_RELATIVES,
        nul_separator=True,
    ) or (os.environ.get("AUDIOLENS_AUDIO_WORKSPACE_SOURCE_DIGEST") or None)


def _lock_digest() -> str | None:
    path = REPO_ROOT / "uv.lock"
    if path.is_file():
        return _sha256_file(path)
    return os.environ.get("AUDIOLENS_LOCK_SHA256") or None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("AUDIOLENS_GIT_REVISION") or None


SOURCE_DIGEST = _source_digest()
LOCK_SHA256 = _lock_digest()
GIT_REVISION = _git_revision()
CANONICAL_WORKSPACE_SOURCE_DIGEST = _relative_source_digest(
    CANONICAL_WORKSPACE_SOURCE_RELATIVES,
    nul_separator=False,
)
CANONICAL_FIT_SOURCE_DIGEST = _relative_source_digest(
    CANONICAL_FIT_SOURCE_RELATIVES,
    nul_separator=True,
)
_HAS_LOCAL_PROJECT = all((REPO_ROOT / relative).is_file() for relative in SOURCE_RELATIVES)
_DEPLOY_MODAL = _HAS_LOCAL_PROJECT and os.environ.get("AUDIOLENS_DISABLE_MODAL") != "1"

if _DEPLOY_MODAL:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "git",
            "libsndfile1",
        )
        .uv_sync(
            uv_project_dir=str(REPO_ROOT),
            frozen=True,
            groups=["fit"],
            gpu=MODEL_GPU,
        )
        .env(
            {
                "HF_HOME": f"{VOL_MOUNT}/hf",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "AUDIOLENS_AUDIO_WORKSPACE_SOURCE_DIGEST": SOURCE_DIGEST or "",
                "AUDIOLENS_LOCK_SHA256": LOCK_SHA256 or "",
                "AUDIOLENS_GIT_REVISION": GIT_REVISION or "",
                "AUDIOLENS_WORKSPACE_EVAL_SOURCE_DIGEST": (CANONICAL_WORKSPACE_SOURCE_DIGEST or ""),
                "AUDIOLENS_SOURCE_DIGEST": CANONICAL_FIT_SOURCE_DIGEST or "",
            }
        )
        .add_local_python_source("audiolens")
        .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
    )
    app = modal.App(APP_NAME, image=image)
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
else:
    image = None
    app = None
    vol = None


def _modal_gpu_function(*, timeout: int):
    def decorate(function):
        if app is None or vol is None:
            return function
        return app.function(
            gpu=MODEL_GPU,
            timeout=timeout,
            volumes={VOL_MOUNT: vol},
            secrets=[modal.Secret.from_name("huggingface")],
        )(function)

    return decorate


def _modal_cpu_function(*, timeout: int):
    def decorate(function):
        if app is None or vol is None:
            return function
        return app.function(
            cpu=4.0,
            memory=32_768,
            timeout=timeout,
            volumes={VOL_MOUNT: vol},
        )(function)

    return decorate


def _modal_local_entrypoint(function):
    if app is None:
        return function
    return app.local_entrypoint()(function)


def _commit_volume() -> None:
    global vol
    if vol is None:
        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    vol.commit()


def _require_source_identity() -> dict[str, str]:
    if SOURCE_DIGEST is None or len(SOURCE_DIGEST) != 64:
        raise ModalAudioWorkspaceEvalError("audio workspace source digest is unavailable")
    if LOCK_SHA256 is None or len(LOCK_SHA256) != 64:
        raise ModalAudioWorkspaceEvalError("uv.lock digest is unavailable")
    if GIT_REVISION is None or len(GIT_REVISION) != 40:
        raise ModalAudioWorkspaceEvalError("Git revision is unavailable")
    return {
        "git_revision": GIT_REVISION,
        "source_sha256": SOURCE_DIGEST,
        "lock_sha256": LOCK_SHA256,
    }


def _read_json(
    path: str | pathlib.Path, *, label: str, maximum: int = MAX_JSON_BYTES
) -> dict[str, Any]:
    candidate = pathlib.Path(path)
    payload = _read_bounded_bytes(candidate, label=label, maximum=maximum)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModalAudioWorkspaceEvalError(f"invalid {label} JSON at {candidate}") from exc
    if not isinstance(value, dict):
        raise ModalAudioWorkspaceEvalError(f"{label} must be a JSON object")
    return value


def _atomic_immutable_write(path: str | pathlib.Path, payload: bytes) -> pathlib.Path:
    """Atomically publish exact bytes at an identity-derived path."""
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        metadata = destination.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ModalAudioWorkspaceEvalError(
                f"immutable path is not a regular file: {destination}"
            )
        if destination.read_bytes() != payload:
            raise ModalAudioWorkspaceEvalError(f"immutable artifact collision at {destination}")
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            descriptor = -1
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
            os.fchmod(target.fileno(), 0o444)
        os.replace(temporary, destination)
        metadata = destination.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or destination.read_bytes() != payload
        ):
            raise ModalAudioWorkspaceEvalError(
                f"immutable artifact verification failed at {destination}"
            )
        try:
            directory = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        try:
            if directory >= 0:
                os.fsync(directory)
        finally:
            if directory >= 0:
                os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def _write_content_addressed_json(
    root: str | pathlib.Path, value: Mapping[str, Any]
) -> tuple[pathlib.Path, str]:
    payload = _canonical_json_bytes(dict(value)) + b"\n"
    digest = _sha256_bytes(payload)
    path = pathlib.Path(root) / f"{digest}.json"
    return _atomic_immutable_write(path, payload), digest


def _fetch_bounded(
    url: str, expected_bytes: int, expected_sha256: str, *, timeout: int = 60
) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read(expected_bytes + 1)
    if len(raw) != expected_bytes:
        raise ModalAudioWorkspaceEvalError(
            f"bounded fetch returned {len(raw)} bytes, expected {expected_bytes}"
        )
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise ModalAudioWorkspaceEvalError(f"bounded fetch SHA-256 mismatch: {actual}")
    return raw


def _fixture_url(spec: Mapping[str, Any]) -> str:
    return (
        "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
        f"{JLENS_REVISION}/data/evaluations/{spec['filename']}"
    )


def _decode_fixture(spec: Mapping[str, Any], raw: bytes) -> list[dict[str, Any]]:
    if len(raw) != spec["n_bytes"] or _sha256_bytes(raw) != spec["sha256"]:
        raise ModalAudioWorkspaceEvalError(f"{spec['distribution']}: pinned fixture bytes changed")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModalAudioWorkspaceEvalError(f"{spec['distribution']}: invalid fixture JSON") from exc
    if not isinstance(value, dict) or set(value) != {"items"}:
        raise ModalAudioWorkspaceEvalError(f"{spec['distribution']}: fixture schema changed")
    items = value["items"]
    if not isinstance(items, list) or len(items) != spec["raw_count"]:
        raise ModalAudioWorkspaceEvalError(f"{spec['distribution']}: raw item count changed")
    names: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModalAudioWorkspaceEvalError(f"{spec['distribution']}[{index}]: item changed")
        name = item.get("name")
        prompt = item.get("prompt")
        intermediates = item.get("intermediates")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(prompt, str)
            or not prompt
            or not isinstance(intermediates, list)
            or not intermediates
            or not all(isinstance(label, str) and label for label in intermediates)
        ):
            raise ModalAudioWorkspaceEvalError(
                f"{spec['distribution']}[{index}]: item content changed"
            )
        names.append(name)
    selected = items[: spec["publication_count"]]
    if _sha256_json([item["name"] for item in selected]) != spec["selected_name_sha256"]:
        raise ModalAudioWorkspaceEvalError(
            f"{spec['distribution']}: publication identities changed"
        )
    return [dict(item) for item in items]


def _fetch_fixtures(
    cache_root: str | pathlib.Path = FIXTURE_CACHE_ROOT,
) -> dict[str, bytes]:
    from audiolens.audio_workspace_eval import decode_publication_fixtures

    root = pathlib.Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    raw_by_distribution: dict[str, bytes] = {}
    for spec in FIXTURE_SPECS:
        path = root / str(spec["filename"])
        if path.exists():
            raw = path.read_bytes()
        else:
            raw = _fetch_bounded(
                _fixture_url(spec),
                int(spec["n_bytes"]),
                str(spec["sha256"]),
            )
            _decode_fixture(spec, raw)
            _atomic_immutable_write(path, raw)
        _decode_fixture(spec, raw)
        raw_by_distribution[str(spec["distribution"])] = raw
    decode_publication_fixtures(raw_by_distribution)
    return raw_by_distribution


def _derive_confirmatory_items(
    raw_fixtures: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    from audiolens.audio_workspace_eval import (
        build_spoken_items,
        decode_publication_fixtures,
    )

    selected = decode_publication_fixtures(raw_fixtures)
    items = build_spoken_items(selected)
    if len(items) != EXPECTED_ITEM_COUNT:
        raise ModalAudioWorkspaceEvalError(
            f"derived {len(items)} confirmatory items, expected {EXPECTED_ITEM_COUNT}"
        )
    return items


def _bind_items_to_text_report(
    items: Sequence[Mapping[str, Any]], text_report: Mapping[str, Any]
) -> None:
    eligibility = text_report.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise ModalAudioWorkspaceEvalError("canonical text report lacks eligibility")
    distributions = eligibility.get("distributions")
    if not isinstance(distributions, Mapping):
        raise ModalAudioWorkspaceEvalError("canonical text eligibility changed")
    expected: list[tuple[str, str]] = []
    for spec in FIXTURE_SPECS:
        distribution = str(spec["distribution"])
        text_distribution = distributions.get(distribution)
        if not isinstance(text_distribution, Mapping):
            raise ModalAudioWorkspaceEvalError(f"canonical text report lacks {distribution!r}")
        text_items = text_distribution.get("items")
        if not isinstance(text_items, list) or len(text_items) != spec["publication_count"]:
            raise ModalAudioWorkspaceEvalError(f"canonical text {distribution} identities changed")
        names = [row.get("name") for row in text_items if isinstance(row, Mapping)]
        if (
            len(names) != spec["publication_count"]
            or _sha256_json(names) != spec["selected_name_sha256"]
        ):
            raise ModalAudioWorkspaceEvalError(f"canonical text {distribution} name digest changed")
        expected.extend(
            (distribution, str(name))
            for name in names
            if not (
                distribution == "multilingual"
                and name in {"filipino-opposite-up", "irish-opposite-big"}
            )
        )
    actual = [(str(item["distribution"]), str(item["name"])) for item in items]
    if actual != expected:
        raise ModalAudioWorkspaceEvalError(
            "spoken item identities do not match the validated canonical text report"
        )


def _expected_tts_engine() -> dict[str, Any]:
    return {
        "engine": TTS_ENGINE,
        "endpoint": TTS_ENDPOINT,
        "model": TTS_MODEL,
        "response_format": TTS_RESPONSE_FORMAT,
        "input_policy": TTS_INPUT_POLICY,
        "synthesis_policy": TTS_SYNTHESIS_POLICY,
        "voices": list(TTS_VARIANTS),
    }


def _load_tts_recipe(
    recipe_sha256: str,
    *,
    root: str | pathlib.Path = SOURCE_STIMULI_ROOT,
) -> tuple[dict[str, Any], pathlib.Path]:
    """Load and seal-check the sealed local-synthesis recipe for the run."""
    from audiolens.audio_workspace_eval import validate_seal

    if not isinstance(recipe_sha256, str) or len(recipe_sha256) != 64:
        raise ModalAudioWorkspaceEvalError("TTS recipe identity must be a SHA-256")
    recipe_root = pathlib.Path(root) / recipe_sha256
    recipe = _read_json(recipe_root / "recipe.json", label="TTS recipe")
    try:
        value = validate_seal(recipe, "recipe_sha256", "TTS recipe")
    except Exception as exc:
        raise ModalAudioWorkspaceEvalError("TTS recipe seal is invalid") from exc
    if value.get("recipe_sha256") != recipe_sha256:
        raise ModalAudioWorkspaceEvalError("TTS recipe content digest mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != TTS_RECIPE_KIND
        or value.get("engine") != _expected_tts_engine()
        or not isinstance(value.get("synthesized_at"), str)
        or not isinstance(value.get("observations"), list)
        or len(value["observations"]) != EXPECTED_OBSERVATION_COUNT
        or not isinstance(value.get("smoke_observations"), list)
        or len(value["smoke_observations"]) != 2
    ):
        raise ModalAudioWorkspaceEvalError("TTS recipe frozen identity changed")
    return value, recipe_root


def _read_recipe_wav(recipe_root: pathlib.Path, entry: Mapping[str, Any], *, label: str) -> bytes:
    relative = entry.get("wav_relative_path")
    if (
        not isinstance(relative, str)
        or pathlib.PurePosixPath(relative).is_absolute()
        or ".." in pathlib.PurePosixPath(relative).parts
    ):
        raise ModalAudioWorkspaceEvalError(f"{label} recipe WAV path is invalid")
    payload = _read_bounded_bytes(
        recipe_root / relative,
        label=label,
        maximum=MAX_AUDIO_BYTES,
    )
    if _sha256_bytes(payload) != entry.get("source_wav_sha256") or len(payload) != entry.get(
        "n_bytes"
    ):
        raise ModalAudioWorkspaceEvalError(f"{label} sealed source WAV bytes changed")
    return payload


def _normalize_wav(source_wav: bytes) -> tuple[bytes, dict[str, Any]]:
    """Normalize sealed source mono PCM16 to deterministic native 16 kHz PCM16."""
    import io

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not source_wav or len(source_wav) > MAX_AUDIO_BYTES:
        raise ModalAudioWorkspaceEvalError("source WAV byte size is invalid")
    source_io = io.BytesIO(source_wav)
    info = sf.info(source_io)
    if (
        info.samplerate != SOURCE_SAMPLE_RATE
        or info.channels != 1
        or info.subtype != "PCM_16"
        or info.frames <= 0
    ):
        raise ModalAudioWorkspaceEvalError("source WAV is not mono 24000 Hz PCM16")
    source_io.seek(0)
    source_pcm, decoded_rate = sf.read(source_io, dtype="int16", always_2d=False)
    if decoded_rate != SOURCE_SAMPLE_RATE or source_pcm.ndim != 1:
        raise ModalAudioWorkspaceEvalError("decoded source WAV layout changed")
    normalized = resample_poly(
        source_pcm.astype(np.float64) / 32768.0,
        up=RESAMPLE_UP,
        down=RESAMPLE_DOWN,
    )
    if normalized.ndim != 1 or normalized.size <= 0 or not np.isfinite(normalized).all():
        raise ModalAudioWorkspaceEvalError("resampled audio is empty or nonfinite")
    normalized = np.clip(normalized, -1.0, 1.0)
    output = io.BytesIO()
    sf.write(output, normalized, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    normalized_wav = output.getvalue()
    output.seek(0)
    decoded, rate = sf.read(output, dtype="int16", always_2d=False)
    if rate != SAMPLE_RATE or decoded.ndim != 1 or decoded.size != normalized.size:
        raise ModalAudioWorkspaceEvalError("normalized WAV round trip changed")
    little = np.asarray(decoded, dtype="<i2")
    source_little = np.asarray(source_pcm, dtype="<i2")
    return normalized_wav, {
        "source_sample_rate": SOURCE_SAMPLE_RATE,
        "source_sample_count": int(source_pcm.size),
        "source_decoded_pcm_sha256": _sha256_bytes(source_little.tobytes(order="C")),
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(decoded.size),
        "duration_seconds": int(decoded.size) / SAMPLE_RATE,
        "decoded_pcm_sha256": _sha256_bytes(little.tobytes(order="C")),
    }


def _stage_stimuli(
    items: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    fit_manifest_sha256: str,
    source_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    *,
    tts_recipe_sha256: str,
    recipe_loader: Callable[[str], tuple[dict[str, Any], pathlib.Path]] = _load_tts_recipe,
    wav_reader: Callable[..., bytes] = _read_recipe_wav,
) -> tuple[dict[str, Any], pathlib.Path]:
    from audiolens.audio_workspace_eval import (
        audit_fit_overlap,
        build_stimulus_manifest,
        tts_input,
        validate_stimulus_manifest,
    )

    if len(items) != EXPECTED_ITEM_COUNT:
        raise ModalAudioWorkspaceEvalError(
            f"stimulus staging requires exactly {EXPECTED_ITEM_COUNT} items"
        )
    recipe, recipe_root = recipe_loader(tts_recipe_sha256)
    recipe_observations = recipe["observations"]
    observations: list[dict[str, Any]] = []
    wav_payloads: dict[str, bytes] = {}
    for coordinate_index, item in enumerate(items):
        if item["coordinate_index"] != coordinate_index:
            raise ModalAudioWorkspaceEvalError("stimulus coordinate index changed")
        for variant in TTS_VARIANTS:
            observation_index = len(observations)
            entry = recipe_observations[observation_index]
            spoken = tts_input(str(item["script"]))
            if not isinstance(entry, Mapping) or (
                entry.get("observation_index"),
                entry.get("coordinate_index"),
                entry.get("distribution"),
                entry.get("name"),
                entry.get("variant"),
                entry.get("language"),
                entry.get("script_sha256"),
                entry.get("tts_input"),
                entry.get("tts_input_sha256"),
            ) != (
                observation_index,
                coordinate_index,
                item["distribution"],
                item["name"],
                variant,
                item["language"],
                item["script_sha256"],
                spoken,
                _sha256_bytes(spoken.encode("utf-8")),
            ):
                raise ModalAudioWorkspaceEvalError(
                    "sealed TTS recipe does not match the canonical spoken items"
                )
            source_wav = wav_reader(recipe_root, entry, label="stimulus source WAV")
            normalized_wav, audio = _normalize_wav(source_wav)
            relative = f"wavs/{observation_index:03d}-{variant}.wav"
            observations.append(
                {
                    "observation_index": observation_index,
                    "coordinate_index": coordinate_index,
                    "distribution": item["distribution"],
                    "name": item["name"],
                    "variant": variant,
                    "language": item["language"],
                    "script_sha256": item["script_sha256"],
                    "tts_input": spoken,
                    "tts_input_sha256": _sha256_bytes(spoken.encode("utf-8")),
                    "wav_relative_path": relative,
                    "source_wav_sha256": _sha256_bytes(source_wav),
                    "source_pcm_sha256": audio["source_decoded_pcm_sha256"],
                    "normalized_wav_sha256": _sha256_bytes(normalized_wav),
                    "decoded_pcm_sha256": audio["decoded_pcm_sha256"],
                    "source_sample_rate": audio["source_sample_rate"],
                    "sample_rate": audio["sample_rate"],
                    "sample_count": audio["sample_count"],
                    "duration_seconds": audio["duration_seconds"],
                }
            )
            wav_payloads[relative] = normalized_wav
    if (
        len(observations) != EXPECTED_OBSERVATION_COUNT
        or len(wav_payloads) != EXPECTED_OBSERVATION_COUNT
    ):
        raise ModalAudioWorkspaceEvalError(
            f"stimulus staging did not create exactly {EXPECTED_OBSERVATION_COUNT} WAV observations"
        )
    overlap = audit_fit_overlap(
        observations,
        items,
        fit_rows,
        fit_manifest_sha256=fit_manifest_sha256,
    )
    manifest = build_stimulus_manifest(
        items=items,
        observations=observations,
        overlap_audit=overlap,
        tts_recipe_sha256=recipe["recipe_sha256"],
        source_identity=dict(source_identity),
        runtime_identity=runtime_identity,
    )
    validate_stimulus_manifest(manifest, expected_items=list(items))
    manifest_root = pathlib.Path(STIMULUS_ROOT) / manifest["stimulus_manifest_sha256"]
    for relative, wav in wav_payloads.items():
        _atomic_immutable_write(manifest_root / relative, wav)
    manifest_path = _atomic_immutable_write(
        manifest_root / "manifest.json",
        _canonical_json_bytes(manifest) + b"\n",
    )
    return manifest, manifest_path


def _calibration_coordinates(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    from audiolens.audio_workspace_eval import calibration_coordinates

    return calibration_coordinates(manifest["items"])


def _run_whisper_calibration(
    manifest: Mapping[str, Any],
    manifest_path: str | pathlib.Path,
    *,
    transcriber: Callable[[pathlib.Path, str], str] | None = None,
) -> dict[str, Any]:
    from audiolens.audio_workspace_eval import (
        build_calibration,
        character_error_rate,
        normalize_transcript_for_language,
        validate_calibration,
    )

    if transcriber is None:
        import soundfile as sf
        import torch
        import transformers

        processor = transformers.AutoProcessor.from_pretrained(
            WHISPER_ID, revision=WHISPER_REVISION
        )
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_ID,
            revision=WHISPER_REVISION,
            dtype=torch.float16,
            device_map="cuda",
            attn_implementation="eager",
        ).eval()

        def transcriber(path: pathlib.Path, language: str) -> str:
            audio, rate = sf.read(path, dtype="float32", always_2d=False)
            if rate != SAMPLE_RATE or audio.ndim != 1:
                raise ModalAudioWorkspaceEvalError("calibration WAV layout changed")
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            features = inputs.input_features.to(model.device, dtype=torch.float16)
            with torch.inference_mode():
                generated = model.generate(
                    input_features=features,
                    language=language,
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                )
            return str(processor.batch_decode(generated, skip_special_tokens=True)[0])

    root = pathlib.Path(manifest_path).parent
    observation_by_coordinate = {
        (
            row["distribution"],
            row["name"],
            row["variant"],
        ): row
        for row in manifest["observations"]
    }
    item_by_coordinate = {(item["distribution"], item["name"]): item for item in manifest["items"]}
    language_name_by_code = {code: language for language, code in LANGUAGE_CODES.items()}
    cells: list[dict[str, Any]] = []
    for coordinate in _calibration_coordinates(manifest):
        observation = observation_by_coordinate[
            (
                coordinate["distribution"],
                coordinate["name"],
                coordinate["variant"],
            )
        ]
        item = item_by_coordinate[(coordinate["distribution"], coordinate["name"])]
        wav = root / observation["wav_relative_path"]
        language_name = language_name_by_code[coordinate["language"]]
        whisper_code = LANGUAGE_TO_WHISPER[language_name]
        with _verified_local_copy(
            wav,
            expected_sha256=observation["normalized_wav_sha256"],
            label="calibration WAV",
            maximum=MAX_AUDIO_BYTES,
            suffix=".wav",
        ) as local_wav:
            transcript = transcriber(local_wav, whisper_code)
        if not isinstance(transcript, str):
            raise ModalAudioWorkspaceEvalError("Whisper transcriber returned nontext")
        reference = item["script"]
        cell_language = str(coordinate["language"])
        cells.append(
            {
                **coordinate,
                "reference": reference,
                "transcript": transcript,
                "normalized_reference": normalize_transcript_for_language(reference, cell_language),
                "normalized_transcript": normalize_transcript_for_language(
                    transcript, cell_language
                ),
                "cer": character_error_rate(reference, transcript, cell_language),
            }
        )
    calibration = build_calibration(
        stimulus_manifest=manifest,
        cells=cells,
    )
    validate_calibration(calibration, manifest)
    return calibration


def _runtime_identity(*, include_cuda: bool) -> dict[str, Any]:
    import importlib.metadata
    import platform

    packages = {
        package: importlib.metadata.version(package)
        for package in (
            "accelerate",
            "jlens",
            "modal",
            "numpy",
            "scipy",
            "soundfile",
            "torch",
            "transformers",
        )
    }
    modal_image_id = os.environ.get("MODAL_IMAGE_ID")
    if not isinstance(modal_image_id, str) or not modal_image_id:
        raise ModalAudioWorkspaceEvalError("MODAL_IMAGE_ID is unavailable for runtime binding")
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "modal_image_id": modal_image_id,
    }
    if include_cuda:
        import torch

        cuda = torch.version.cuda
        if not isinstance(cuda, str) or not cuda or not torch.cuda.is_available():
            raise ModalAudioWorkspaceEvalError("CUDA runtime identity is unavailable")
        identity["cuda"] = cuda
        identity["device"] = torch.cuda.get_device_name(0)
    return identity


def _load_canonical_text_report(path: str = CANONICAL_TEXT_REPORT_PATH) -> dict[str, Any]:
    import sys

    scripts_root = (
        "/root/scripts" if pathlib.Path("/root/scripts").is_dir() else str(REPO_ROOT / "scripts")
    )
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from modal_workspace_eval import load_completed_workspace_report

    report = load_completed_workspace_report(path)
    if report.get("evaluation_config_sha256") != CANONICAL_TEXT_EVALUATION_SHA256:
        raise ModalAudioWorkspaceEvalError("canonical text evaluation identity changed")
    if report.get("workspace_report_sha256") != CANONICAL_TEXT_REPORT_SHA256:
        raise ModalAudioWorkspaceEvalError("canonical text report identity changed")
    if report.get("adjudication", {}).get("status") != "no_band":
        raise ModalAudioWorkspaceEvalError("canonical text status is not no_band")
    return report


def _preview_audio_run(path: str = FINAL_AUDIO_RUN_PATH) -> dict[str, Any]:
    run = _read_json(path, label="completed audio run")
    if (
        run.get("status") != "complete"
        or run.get("fit_config_sha256") != FINAL_AUDIO_FIT_CONFIG_SHA256
    ):
        raise ModalAudioWorkspaceEvalError("completed audio run identity changed")
    lens = run.get("lens")
    if not isinstance(lens, Mapping) or lens.get("sha256") != FINAL_AUDIO_LENS_SHA256:
        raise ModalAudioWorkspaceEvalError("completed audio lens identity changed")
    return run


def _fit_corpus_rows(
    run: Mapping[str, Any],
    volume_root: str = VOL_MOUNT,
) -> list[dict[str, Any]]:
    corpus = run.get("corpus")
    if not isinstance(corpus, Mapping):
        raise ModalAudioWorkspaceEvalError("completed audio run lacks corpus artifact")
    relative = corpus.get("rows_path")
    if (
        not isinstance(relative, str)
        or pathlib.PurePosixPath(relative).is_absolute()
        or ".." in pathlib.PurePosixPath(relative).parts
    ):
        raise ModalAudioWorkspaceEvalError("completed audio corpus rows path is invalid")
    rows_sha256 = corpus.get("rows_sha256")
    if (
        not isinstance(rows_sha256, str)
        or len(rows_sha256) != 64
        or any(character not in "0123456789abcdef" for character in rows_sha256)
    ):
        raise ModalAudioWorkspaceEvalError("completed audio corpus rows identity is invalid")
    path = pathlib.Path(volume_root) / relative
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ModalAudioWorkspaceEvalError("completed audio corpus ledger is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ModalAudioWorkspaceEvalError(
            "completed audio corpus ledger must be a nonsymlink regular file"
        )
    if _sha256_file(path) != rows_sha256:
        raise ModalAudioWorkspaceEvalError("completed audio corpus ledger bytes changed")
    rows: list[dict[str, Any]] = []
    total = 0
    with open(path, "rb") as source:
        for line in source:
            total += len(line)
            if total > MAX_JSON_BYTES or len(line) > 1024 * 1024:
                raise ModalAudioWorkspaceEvalError("completed audio corpus ledger is oversized")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModalAudioWorkspaceEvalError(
                    "completed audio corpus row is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ModalAudioWorkspaceEvalError("completed audio corpus row is not an object")
            rows.append(value)
    if len(rows) != 1_000:
        raise ModalAudioWorkspaceEvalError("completed audio corpus does not have 1,000 rows")
    return rows


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModalAudioWorkspaceEvalError("tokenizer did not return input_ids") from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ModalAudioWorkspaceEvalError("tokenizer returned non-batch-one token IDs")
        ids = ids[0]
    if not isinstance(ids, (list, tuple)) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in ids
    ):
        raise ModalAudioWorkspaceEvalError("tokenizer returned invalid token IDs")
    return list(ids)


def _eligibility(items: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    from audiolens.audio_workspace_eval import allowed_forms, seal_mapping

    prepared: list[dict[str, Any]] = []
    for item in items:
        concepts: list[dict[str, Any]] = []
        for index, authored in enumerate(item["intermediates"]):
            forms = list(allowed_forms(str(item["distribution"]), str(authored)))
            token_ids: list[int] = []
            for form in forms:
                for rendered in (form, " " + form):
                    ids = _token_ids(tokenizer, rendered)
                    if len(ids) == 1 and ids[0] >= 0 and ids[0] not in token_ids:
                        token_ids.append(ids[0])
            if token_ids:
                concepts.append(
                    {
                        "concept_id": (f"{item['distribution']}/{item['name']}/{index}"),
                        "authored": str(authored),
                        "forms": forms,
                        "allowed_token_ids": token_ids,
                    }
                )
        concept_ids = [concept["concept_id"] for concept in concepts]
        prepared.append(
            {
                "distribution": item["distribution"],
                "name": item["name"],
                "included_in_metrics": bool(concepts),
                "eligible_concept_ids": concept_ids,
                "concepts": concepts,
            }
        )
    return seal_mapping({"items": prepared}, "eligibility_sha256")


def _tensor_identity(value: Any, *, name: str) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value) or value.layout != torch.strided or value.numel() == 0:
        raise ModalAudioWorkspaceEvalError(f"processor input {name} is not a nonempty dense tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": _sha256_bytes(payload),
    }


def _preparation_record(
    observation: Mapping[str, Any],
    prepared: Any,
) -> dict[str, Any]:
    framing = dict(prepared.manifest_fields)
    expected = {
        "audio_start": int(prepared.layout.audio_start),
        "n_audio_tokens": int(prepared.layout.n_audio_tokens),
        "audio_stop": int(prepared.layout.audio_stop),
        "sequence_length": int(prepared.layout.sequence_length),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "last_processor_valid_audio_position": int(prepared.last_processor_valid_audio_position),
        "response_position": int(prepared.response_position),
        "prefix_framing_ids": tuple(prepared.prefix_framing_ids),
        "suffix_framing_ids": tuple(prepared.suffix_framing_ids),
    }
    if framing != expected:
        raise ModalAudioWorkspaceEvalError("processor manifest framing changed")
    if (
        expected["audio_start"] < 0
        or expected["n_audio_tokens"] <= 0
        or expected["audio_stop"] != expected["audio_start"] + expected["n_audio_tokens"]
        or expected["last_processor_valid_audio_position"] != expected["audio_stop"] - 1
        or expected["response_position"] != expected["sequence_length"] - 1
        or expected["audio_stop"] > expected["response_position"]
        or expected["sequence_length"] > MAX_SEQUENCE_LENGTH
    ):
        raise ModalAudioWorkspaceEvalError("processor preparation geometry is invalid")
    if not isinstance(prepared.model_inputs, Mapping) or any(
        not isinstance(name, str) for name in prepared.model_inputs
    ):
        raise ModalAudioWorkspaceEvalError("processor model inputs are not a string-keyed mapping")
    return {
        "observation_index": int(observation["observation_index"]),
        "distribution": str(observation["distribution"]),
        "name": str(observation["name"]),
        "variant": str(observation["variant"]),
        "normalized_wav_sha256": str(observation["normalized_wav_sha256"]),
        "model_inputs": [
            _tensor_identity(prepared.model_inputs[name], name=str(name))
            for name in sorted(prepared.model_inputs)
        ],
        **{
            key: list(value) if isinstance(value, tuple) else value
            for key, value in expected.items()
        },
    }


def _processor_preparation_identity(
    manifest: Mapping[str, Any],
    manifest_path: str | pathlib.Path,
    processor: Any,
) -> dict[str, Any]:
    from audiolens.audio_eval_model import prepare_audio_evaluation

    root = pathlib.Path(manifest_path).parent
    records: list[dict[str, Any]] = []
    for observation in manifest["observations"]:
        wav = root / observation["wav_relative_path"]
        with _verified_local_copy(
            wav,
            expected_sha256=observation["normalized_wav_sha256"],
            label="processor preparation WAV",
            maximum=MAX_AUDIO_BYTES,
            suffix=".wav",
        ) as local_wav:
            prepared = prepare_audio_evaluation(
                processor,
                local_wav,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
            records.append(_preparation_record(observation, prepared))
    if len(records) != EXPECTED_OBSERVATION_COUNT:
        raise ModalAudioWorkspaceEvalError(
            f"processor preparation must cover exactly {EXPECTED_OBSERVATION_COUNT} observations"
        )
    body = {
        "kind": "audio_workspace_processor_preparation",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "observations": records,
    }
    return {**body, "preparation_sha256": _sha256_json(body)}


def _runtime_environment_from_preregistration(
    preregistration: Mapping[str, Any],
) -> Mapping[str, Any]:
    runtime = preregistration.get("runtime_identity")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "environment",
        "processor_preparation",
    }:
        raise ModalAudioWorkspaceEvalError("preregistered runtime identity schema changed")
    environment = runtime["environment"]
    preparation = runtime["processor_preparation"]
    if not isinstance(environment, Mapping) or not isinstance(preparation, Mapping):
        raise ModalAudioWorkspaceEvalError("preregistered runtime/preparation identity is invalid")
    body = dict(preparation)
    claimed = body.pop("preparation_sha256", None)
    if (
        claimed != _sha256_json(body)
        or body.get("kind") != "audio_workspace_processor_preparation"
        or body.get("model") != {"id": MODEL_ID, "revision": MODEL_REVISION}
        or body.get("max_sequence_length") != MAX_SEQUENCE_LENGTH
        or not isinstance(body.get("observations"), list)
        or len(body["observations"]) != EXPECTED_OBSERVATION_COUNT
    ):
        raise ModalAudioWorkspaceEvalError("preregistered processor preparation identity changed")
    return environment


def _require_runtime_environment(
    preregistration: Mapping[str, Any],
) -> None:
    expected = _runtime_environment_from_preregistration(preregistration)
    if expected != _runtime_identity(include_cuda=True):
        raise ModalAudioWorkspaceEvalError("preregistered runtime environment drifted")


def _build_preregistration(
    manifest: Mapping[str, Any],
    expected_items: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    text_report: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    from audiolens.audio_workspace_eval import build_preregistration

    return build_preregistration(
        stimulus_manifest=manifest,
        expected_items=expected_items,
        calibration=calibration,
        artifact_identity={
            "fit_config_sha256": FINAL_AUDIO_FIT_CONFIG_SHA256,
            "lens_sha256": FINAL_AUDIO_LENS_SHA256,
            "lens_dtype": "float16",
            "lens_layers": list(ALL_LAYERS),
            "completed_run_manifest_sha256": _sha256_file(FINAL_AUDIO_RUN_PATH),
        },
        historical_text_report=text_report,
        eligibility=eligibility,
        source_identity=source_identity,
        runtime_identity=runtime_identity,
    )


def _validate_physical_stimuli(
    preregistration: Mapping[str, Any],
    *,
    processor_loader: Callable[[], Any] | None = None,
    verify_normalization: bool = False,
) -> tuple[dict[str, Any], pathlib.Path, dict[str, Any]]:
    import io
    import numpy as np
    import soundfile as sf

    from audiolens.audio_workspace_eval import (
        validate_calibration,
        validate_preregistration,
        validate_stimulus_manifest,
    )

    stimulus_sha = preregistration["stimulus_manifest_sha256"]
    manifest_path = pathlib.Path(STIMULUS_ROOT) / stimulus_sha / "manifest.json"
    manifest = _read_json(manifest_path, label="stimulus manifest")
    expected_items = _derive_confirmatory_items(_fetch_fixtures())
    validate_stimulus_manifest(
        manifest,
        expected_items=expected_items,
    )
    preregistration_environment = _runtime_environment_from_preregistration(preregistration)
    stimulus_environment = {
        key: value
        for key, value in preregistration_environment.items()
        if key not in {"cuda", "device"}
    }
    if manifest.get(
        "runtime_identity"
    ) != stimulus_environment or stimulus_environment != _runtime_identity(include_cuda=False):
        raise ModalAudioWorkspaceEvalError(
            "stimulus runtime identity differs from preregistration or validator"
        )
    recipe, recipe_root = _load_tts_recipe(str(manifest.get("tts_recipe_sha256")))
    if manifest.get("source_identity") != preregistration["source_identity"]:
        raise ModalAudioWorkspaceEvalError("stimulus source identity changed")
    calibration_path = (
        pathlib.Path(CALIBRATION_ROOT) / f"{preregistration['calibration_sha256']}.json"
    )
    calibration = _read_json(calibration_path, label="Whisper calibration")
    validate_calibration(calibration, manifest)
    validate_preregistration(
        preregistration,
        stimulus_manifest=manifest,
        expected_items=expected_items,
        calibration=calibration,
    )
    for observation in manifest["observations"]:
        wav = manifest_path.parent / observation["wav_relative_path"]
        wav_payload = _read_bounded_bytes(
            wav,
            label="stimulus WAV",
            maximum=MAX_AUDIO_BYTES,
        )
        if _sha256_bytes(wav_payload) != observation["normalized_wav_sha256"]:
            raise ModalAudioWorkspaceEvalError("stimulus WAV changed")
        pcm, rate = sf.read(io.BytesIO(wav_payload), dtype="int16", always_2d=False)
        if rate != SAMPLE_RATE or pcm.ndim != 1:
            raise ModalAudioWorkspaceEvalError("stimulus WAV physical layout changed")
        decoded_sha = _sha256_bytes(np.asarray(pcm, dtype="<i2").tobytes(order="C"))
        if (
            decoded_sha != observation["decoded_pcm_sha256"]
            or int(pcm.size) != observation["sample_count"]
        ):
            raise ModalAudioWorkspaceEvalError("stimulus decoded PCM identity changed")
        if verify_normalization:
            entry = recipe["observations"][int(observation["observation_index"])]
            source_wav = _read_recipe_wav(recipe_root, entry, label="sealed source WAV")
            normalized_wav, audio = _normalize_wav(source_wav)
            if (
                _sha256_bytes(source_wav) != observation["source_wav_sha256"]
                or audio["source_decoded_pcm_sha256"] != observation["source_pcm_sha256"]
                or _sha256_bytes(normalized_wav) != observation["normalized_wav_sha256"]
                or audio["decoded_pcm_sha256"] != observation["decoded_pcm_sha256"]
                or audio["sample_count"] != observation["sample_count"]
            ):
                raise ModalAudioWorkspaceEvalError("independent normalization reproduction changed")
    if processor_loader is None:
        from audiolens.models import load_audio_processor

        processor = load_audio_processor()
    else:
        processor = processor_loader()
    preparation = _processor_preparation_identity(
        manifest,
        manifest_path,
        processor,
    )
    if preparation != preregistration["runtime_identity"]["processor_preparation"]:
        raise ModalAudioWorkspaceEvalError(
            "processor preparation no longer matches preregistration"
        )
    return manifest, manifest_path, calibration


def _validate_preregistration_file(path: str, expected_sha256: str) -> dict[str, Any]:
    from audiolens.audio_workspace_eval import validate_preregistration

    candidate = pathlib.Path(path)
    if (
        candidate.parent != pathlib.Path(PREREGISTRATION_ROOT)
        or candidate.suffix != ".json"
        or candidate.stem != expected_sha256
    ):
        raise ModalAudioWorkspaceEvalError(
            "preregistration must be its content-addressed frozen path"
        )
    record = _read_json(candidate, label="preregistration")
    if record.get("preregistration_sha256") != expected_sha256:
        raise ModalAudioWorkspaceEvalError("preregistration requested SHA-256 mismatch")
    validate_preregistration(record)
    if record.get("source_identity") != _require_source_identity():
        raise ModalAudioWorkspaceEvalError("preregistration source identity drifted")
    if record.get("calibration_status") != "passed":
        raise ModalAudioWorkspaceEvalError("confirmatory evaluation requires passed calibration")
    return record


def _deterministic_torch() -> Any:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ModalAudioWorkspaceEvalError(
            "CUBLAS_WORKSPACE_CONFIG must be set before Torch import"
        )
    import random

    import numpy as np
    import torch

    random.seed(CONTROL_SEED)
    np.random.seed(CONTROL_SEED)
    torch.manual_seed(CONTROL_SEED)
    torch.cuda.manual_seed_all(CONTROL_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def _control_permutations(torch: Any, d_model: int) -> tuple[dict[int, int], Any, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(CONTROL_SEED)
    output = torch.randperm(d_model, generator=generator)
    source = {layer: (layer + 17) % len(ALL_LAYERS) for layer in ALL_LAYERS}
    output_list = [int(value) for value in output.tolist()]
    return source, output, _sha256_json(output_list)


def _strict_group_rank(
    logits: Any,
    token_ids: Sequence[int],
    torch: Any,
    *,
    sorted_logits: Any | None = None,
    validated: bool = False,
) -> int:
    return _strict_group_ranks(
        logits,
        {"target": token_ids},
        torch,
        sorted_logits=sorted_logits,
        validated=validated,
    )["target"]


def _strict_group_ranks(
    logits: Any,
    groups: Mapping[str, Sequence[int]],
    torch: Any,
    *,
    sorted_logits: Any | None = None,
    validated: bool = False,
) -> dict[str, int]:
    """Rank many allowed-token groups with one exact full-vocabulary sort."""
    if logits.ndim != 1 or (not validated and not bool(torch.isfinite(logits).all())):
        raise ModalAudioWorkspaceEvalError("rank input shape changed")
    if not groups:
        return {}
    names: list[str] = []
    best_values: list[Any] = []
    for name, token_ids in groups.items():
        if not isinstance(name, str) or not name or not token_ids:
            raise ModalAudioWorkspaceEvalError("rank group identity changed")
        ids = torch.tensor(
            list(token_ids),
            dtype=torch.long,
            device=logits.device,
        )
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= logits.numel():
            raise ModalAudioWorkspaceEvalError("rank logits or token IDs are invalid")
        names.append(name)
        best_values.append(logits.index_select(0, ids).max())
    ordered = logits.sort().values if sorted_logits is None else sorted_logits
    if ordered.ndim != 1 or ordered.numel() != logits.numel() or ordered.device != logits.device:
        raise ModalAudioWorkspaceEvalError("sorted rank logits changed")
    best = torch.stack(best_values)
    greater = logits.numel() - torch.searchsorted(
        ordered,
        best,
        right=True,
    )
    ranks = (greater + 1).to(device="cpu").tolist()
    return {name: int(rank) for name, rank in zip(names, ranks, strict=True)}


def _js_divergence_nats(left: Any, right: Any, torch: Any) -> float:
    import math

    left_log = torch.log_softmax(left.float(), dim=-1)
    right_log = torch.log_softmax(right.float(), dim=-1)
    mixture = torch.logaddexp(left_log, right_log) - math.log(2.0)
    value = 0.5 * (
        (left_log.exp() * (left_log - mixture)).sum()
        + (right_log.exp() * (right_log - mixture)).sum()
    )
    if not bool(torch.isfinite(value)):
        raise ModalAudioWorkspaceEvalError("candidate/logit JS divergence is nonfinite")
    return float(value.clamp(min=0.0, max=math.log(2.0)).item())


def _move_prepared(prepared: Any, device: Any) -> Any:
    from dataclasses import replace

    inputs = prepared.model_inputs
    if hasattr(inputs, "to"):
        moved = inputs.to(device)
    elif isinstance(inputs, Mapping):
        moved = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
    else:
        raise ModalAudioWorkspaceEvalError("prepared model inputs cannot move to CUDA")
    audio_positions = prepared.audio_positions
    if hasattr(audio_positions, "to"):
        audio_positions = audio_positions.to(device)
    return replace(
        prepared,
        model_inputs=moved,
        input_ids=moved["input_ids"],
        audio_positions=audio_positions,
    )


def _score_one_observation(
    runtime: Any,
    prepared: Any,
    item_eligibility: Mapping[str, Any],
    distribution_pool: Sequence[Mapping[str, Any]],
    jacobians: Mapping[int, Any],
    permuted_jacobians: Mapping[int, Any],
    torch: Any,
) -> dict[str, Any]:
    from jlens.hooks import ActivationRecorder

    device = runtime.input_device
    prepared = _move_prepared(prepared, device)
    with (
        torch.inference_mode(),
        ActivationRecorder(runtime.layers, at=[*ALL_LAYERS, FINAL_MODEL_LAYER]) as recorder,
    ):
        runtime.forward_audio(prepared)
    position_values = {
        "last_processor_valid_audio_position": int(prepared.last_processor_valid_audio_position),
        "response_position": int(prepared.response_position),
    }
    final_residual = recorder.activations[FINAL_MODEL_LAYER][
        0, position_values["response_position"]
    ].float()
    final_logits = runtime.unembed(final_residual).float()
    if final_logits.ndim > 1:
        final_logits = final_logits.squeeze(0)
    if not bool(torch.isfinite(final_logits).all()):
        raise ModalAudioWorkspaceEvalError("L34 final logits are nonfinite")
    actual_id = int(final_logits.argmax().item())
    vocabulary_size = int(final_logits.numel())
    record: dict[str, Any] = {
        "vocabulary_size": vocabulary_size,
        "positions": {},
        "actual_output": {
            "layer": FINAL_MODEL_LAYER,
            "position": "response_position",
            "position_index": position_values["response_position"],
            "token_id": actual_id,
        },
    }
    pool_ids = {
        concept["concept_id"]: concept["allowed_token_ids"] for concept in distribution_pool
    }
    own_ids = {
        concept["concept_id"]: concept["allowed_token_ids"]
        for concept in item_eligibility["concepts"]
    }
    del final_logits
    for position_name, position in position_values.items():
        layers: dict[str, Any] = {}
        for layer in ALL_LAYERS:
            residual = recorder.activations[layer][0, position].float()
            jacobian = jacobians[layer]
            permuted = permuted_jacobians[layer]
            transports = {
                "candidate": residual @ jacobian.T,
                "logit": residual,
                "transposed": residual @ jacobian,
                "permuted": residual @ permuted.T,
            }
            logits = {
                name: runtime.unembed(value).float().squeeze(0)
                for name, value in transports.items()
            }
            if any(not bool(torch.isfinite(value).all()) for value in logits.values()):
                raise ModalAudioWorkspaceEvalError("candidate/control logits are nonfinite")
            sorted_logits = {control: value.sort().values for control, value in logits.items()}
            own_ranks = {
                control: _strict_group_ranks(
                    logits[control],
                    own_ids,
                    torch,
                    sorted_logits=sorted_logits[control],
                    validated=True,
                )
                for control in CONTROLS
            }
            label_pool = _strict_group_ranks(
                logits["candidate"],
                pool_ids,
                torch,
                sorted_logits=sorted_logits["candidate"],
                validated=True,
            )
            layer_record: dict[str, Any] = {
                "concept_ranks": own_ranks,
                "candidate_label_pool_ranks": label_pool,
            }
            if position_name == "response_position":
                layer_record["motor"] = {
                    "actual_token_id": actual_id,
                    "actual_token_ranks": {
                        control: _strict_group_rank(
                            logits[control],
                            [actual_id],
                            torch,
                            sorted_logits=sorted_logits[control],
                            validated=True,
                        )
                        for control in ("candidate", "logit")
                    },
                    "candidate_logit_top1_agreement": int(logits["candidate"].argmax().item())
                    == int(logits["logit"].argmax().item()),
                    "candidate_logit_js_nats": _js_divergence_nats(
                        logits["candidate"], logits["logit"], torch
                    ),
                }
            layers[str(layer)] = layer_record
            del transports, logits, sorted_logits
        record["positions"][position_name] = {
            "index": position,
            "layers": layers,
        }
    return record


def _load_validated_audio_lens(
    preregistration: Mapping[str, Any], torch: Any
) -> tuple[Any, dict[str, Any]]:
    import jlens

    from audiolens.audio_workspace_eval import validate_audio_artifact_chain

    run_manifest_bytes = _read_bounded_bytes(
        FINAL_AUDIO_RUN_PATH,
        label="completed audio run",
        maximum=MAX_JSON_BYTES,
    )
    try:
        run = json.loads(run_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModalAudioWorkspaceEvalError("completed audio run is invalid JSON") from exc
    if not isinstance(run, dict):
        raise ModalAudioWorkspaceEvalError("completed audio run must be a JSON object")
    expected_run_sha = preregistration["artifact_identity"]["completed_run_manifest_sha256"]
    if _sha256_bytes(run_manifest_bytes) != expected_run_sha:
        raise ModalAudioWorkspaceEvalError("completed audio run manifest bytes changed")
    validated_identity = validate_audio_artifact_chain(
        run,
        completed_run_manifest_bytes=run_manifest_bytes,
        volume_root=VOL_MOUNT,
        completed_run_manifest_sha256=preregistration["artifact_identity"][
            "completed_run_manifest_sha256"
        ],
    )
    if validated_identity != preregistration["artifact_identity"]:
        raise ModalAudioWorkspaceEvalError("preregistered physical audio artifact identity changed")
    lens_identity = run.get("lens")
    if not isinstance(lens_identity, Mapping):
        raise ModalAudioWorkspaceEvalError("validated audio run lacks lens")
    relative = lens_identity.get("relative_path")
    if not isinstance(relative, str):
        raise ModalAudioWorkspaceEvalError("validated audio lens path changed")
    lens_path = pathlib.Path(VOL_MOUNT) / relative
    lens_bytes = _read_bounded_bytes(
        lens_path,
        label="validated audio lens",
        maximum=MAX_LENS_BYTES,
    )
    if _sha256_bytes(lens_bytes) != FINAL_AUDIO_LENS_SHA256:
        raise ModalAudioWorkspaceEvalError("validated audio lens bytes changed")
    with tempfile.NamedTemporaryFile(suffix=".pt") as local_lens:
        local_lens.write(lens_bytes)
        local_lens.flush()
        lens = jlens.JacobianLens.load(local_lens.name)
    if (
        lens.n_prompts != 1_000
        or lens.d_model != 1_536
        or list(lens.source_layers) != list(ALL_LAYERS)
    ):
        raise ModalAudioWorkspaceEvalError("loaded audio lens tensor metadata changed")
    jacobians = {
        layer: lens.jacobians[layer].to(device="cuda", dtype=torch.float32) for layer in ALL_LAYERS
    }
    return jacobians, validated_identity


def _score_confirmatory(
    preregistration: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: pathlib.Path,
    *,
    scorer: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from audiolens.audio_workspace_eval import (
        control_identity,
        validate_score_records,
    )

    torch = _deterministic_torch()
    controls = control_identity()
    source_layers = {int(layer): int(source) for layer, source in controls["source_layers"].items()}
    output_permutation = torch.tensor(
        controls["output_basis"],
        dtype=torch.long,
        device="cpu",
    )
    if scorer is None:
        from audiolens.audio_eval_model import prepare_audio_evaluation
        from audiolens.models import (
            DEFAULT_MODEL_PROFILE,
            load_model_runtime,
        )

        jacobians, validated_artifact = _load_validated_audio_lens(preregistration, torch)
        output_permutation_cuda = output_permutation.to(device="cuda")
        permuted_jacobians = {
            layer: jacobians[source_layers[layer]]
            .index_select(0, output_permutation_cuda)
            .contiguous()
            for layer in ALL_LAYERS
        }
        runtime = load_model_runtime(
            DEFAULT_MODEL_PROFILE.key,
            device_map="cuda",
        )
        runtime.model.eval()
        if runtime.profile.model_id != MODEL_ID or runtime.profile.model_revision != MODEL_REVISION:
            raise ModalAudioWorkspaceEvalError("loaded Gemma profile identity changed")
    else:
        runtime = None
        jacobians = {}
        permuted_jacobians = {}
        validated_artifact = {"test_seam": True}
        prepare_audio_evaluation = None
    eligibility = preregistration["eligibility"]
    if (
        scorer is None
        and _eligibility(manifest["items"], runtime.processor.tokenizer) != eligibility
    ):
        raise ModalAudioWorkspaceEvalError(
            "preregistered eligibility differs from the pinned tokenizer"
        )
    item_eligibility = {(item["distribution"], item["name"]): item for item in eligibility["items"]}
    pools: dict[str, list[Mapping[str, Any]]] = {}
    for item in eligibility["items"]:
        pools.setdefault(item["distribution"], []).extend(item["concepts"])
    records: list[dict[str, Any]] = []
    for observation in manifest["observations"]:
        coordinate = (observation["distribution"], observation["name"])
        own = item_eligibility.get(coordinate)
        if own is None:
            raise ModalAudioWorkspaceEvalError("stimulus has no preregistered eligibility")
        wav = manifest_path.parent / observation["wav_relative_path"]
        if scorer is None:
            with _verified_local_copy(
                wav,
                expected_sha256=observation["normalized_wav_sha256"],
                label="stimulus WAV",
                maximum=MAX_AUDIO_BYTES,
                suffix=".wav",
            ) as local_wav:
                prepared = prepare_audio_evaluation(
                    runtime.processor,
                    local_wav,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                )
                expected_preparation = preregistration["runtime_identity"]["processor_preparation"][
                    "observations"
                ][observation["observation_index"]]
                if _preparation_record(observation, prepared) != expected_preparation:
                    raise ModalAudioWorkspaceEvalError(
                        "scoring preparation differs from preregistration"
                    )
                score = _score_one_observation(
                    runtime,
                    prepared,
                    own,
                    pools[observation["distribution"]],
                    jacobians,
                    permuted_jacobians,
                    torch,
                )
        else:
            score = scorer(
                observation,
                own,
                pools[observation["distribution"]],
            )
        records.append(
            {
                "distribution": observation["distribution"],
                "name": observation["name"],
                "variant": observation["variant"],
                "included_in_metrics": own["included_in_metrics"],
                "eligible_concept_ids": list(own["eligible_concept_ids"]),
                **score,
            }
        )
    validate_score_records(
        records,
        eligibility=eligibility,
        runtime_identity=preregistration["runtime_identity"],
    )
    return records, {
        "validated_audio_artifact": validated_artifact,
        "control_output_permutation_sha256": controls["output_basis_sha256"],
    }


def _build_complete_report(
    preregistration: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    text_report: Mapping[str, Any],
    physical: Mapping[str, Any],
) -> dict[str, Any]:
    del physical
    from audiolens.audio_workspace_eval import build_report

    return build_report(
        preregistration=preregistration,
        records=records,
        historical_text_report=text_report,
    )


def _validate_complete_report(
    report: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    historical_text_report: Mapping[str, Any],
) -> dict[str, Any]:
    from audiolens.audio_workspace_eval import validate_report

    return validate_report(
        report,
        preregistration=preregistration,
        historical_text_report=historical_text_report,
    )


def _preregister_impl(
    tts_recipe_sha256: str = "",
    *,
    fixture_loader: Callable[[], Mapping[str, bytes]] = _fetch_fixtures,
    stimulus_stager: Callable[..., tuple[dict[str, Any], pathlib.Path]] = _stage_stimuli,
    calibration_runner: Callable[..., dict[str, Any]] = _run_whisper_calibration,
    processor_loader: Callable[[], Any] | None = None,
    text_loader: Callable[[], dict[str, Any]] = _load_canonical_text_report,
    run_loader: Callable[[], dict[str, Any]] = _preview_audio_run,
) -> dict[str, Any]:
    # Historical text identities are physically validated before publication
    # rows are admitted to synthesis.
    text_report = text_loader()
    raw_fixtures = fixture_loader()
    items = _derive_confirmatory_items(raw_fixtures)
    _bind_items_to_text_report(items, text_report)

    # Reading the completed-run JSON and fit rows is metadata-only: neither the
    # final lens tensor nor Gemma weights are loaded before calibration seals.
    run = run_loader()
    fit_rows = _fit_corpus_rows(run)
    source_identity = _require_source_identity()
    stimulus_runtime_environment = _runtime_identity(include_cuda=False)
    manifest, manifest_path = stimulus_stager(
        items,
        fit_rows,
        _sha256_file(FINAL_AUDIO_RUN_PATH),
        source_identity,
        stimulus_runtime_environment,
        tts_recipe_sha256=tts_recipe_sha256,
    )
    calibration = calibration_runner(manifest, manifest_path)
    calibration_path = pathlib.Path(CALIBRATION_ROOT) / (
        f"{calibration['calibration_sha256']}.json"
    )
    _atomic_immutable_write(
        calibration_path,
        _canonical_json_bytes(calibration) + b"\n",
    )

    if processor_loader is None:
        from audiolens.models import load_audio_processor

        processor = load_audio_processor()
    else:
        processor = processor_loader()
    eligibility = _eligibility(items, processor.tokenizer)
    processor_preparation = _processor_preparation_identity(
        manifest,
        manifest_path,
        processor,
    )
    runtime_environment = _runtime_identity(include_cuda=True)
    runtime_identity = {
        "environment": runtime_environment,
        "processor_preparation": processor_preparation,
    }

    preregistration = _build_preregistration(
        manifest,
        items,
        calibration,
        eligibility,
        text_report,
        source_identity,
        runtime_identity,
    )
    digest = preregistration["preregistration_sha256"]
    path = pathlib.Path(PREREGISTRATION_ROOT) / f"{digest}.json"
    _atomic_immutable_write(
        path,
        _canonical_json_bytes(preregistration) + b"\n",
    )
    _commit_volume()
    return {
        "mode": "preregister",
        "status": (
            "pending" if calibration["status"] == "passed" else "inconclusive_synthetic_stimulus"
        ),
        "path": str(path),
        "sha256": digest,
        "stimulus_manifest_path": str(manifest_path),
        "stimulus_manifest_sha256": manifest["stimulus_manifest_sha256"],
        "item_count": EXPECTED_ITEM_COUNT,
        "observation_count": EXPECTED_OBSERVATION_COUNT,
        "calibration_cells": len(calibration["cells"]),
    }


def _evaluate_impl(
    preregistration_path: str,
    preregistration_sha256: str,
    *,
    scorer: Callable[..., dict[str, Any]] | None = None,
    report_builder: Callable[..., dict[str, Any]] = _build_complete_report,
    report_validator: Callable[..., dict[str, Any]] = _validate_complete_report,
    text_loader: Callable[[], dict[str, Any]] = _load_canonical_text_report,
) -> dict[str, Any]:
    preregistration = _validate_preregistration_file(
        preregistration_path,
        preregistration_sha256,
    )
    _require_runtime_environment(preregistration)
    manifest, manifest_path, _calibration = _validate_physical_stimuli(preregistration)
    text_report = text_loader()
    _bind_items_to_text_report(manifest["items"], text_report)
    records, physical = _score_confirmatory(
        preregistration,
        manifest,
        manifest_path,
        scorer=scorer,
    )
    report = report_builder(
        preregistration,
        records,
        text_report,
        physical,
    )
    validated = report_validator(report, preregistration, text_report)
    payload = _canonical_json_bytes(validated) + b"\n"
    if len(payload) > MAX_JSON_BYTES:
        raise ModalAudioWorkspaceEvalError(
            "complete report exceeds the frozen publication byte bound"
        )
    digest = validated["report_sha256"]
    report_path = pathlib.Path(REPORT_ROOT) / f"{digest}.json"
    _atomic_immutable_write(report_path, payload)
    _commit_volume()
    return {
        "mode": "evaluate",
        "status": validated["status"],
        "path": str(report_path),
        "sha256": digest,
        "records": EXPECTED_OBSERVATION_COUNT,
    }


def _sacrificial_jacobian(torch: Any, layer: int, d_model: int, device: Any) -> Any:
    diagonal = 0.75 + (layer + 1) / 100.0
    result = torch.eye(d_model, dtype=torch.float32, device=device) * diagonal
    indices = torch.arange(d_model, device=device)
    result[indices, (indices + layer + 1) % d_model] = 0.01
    return result


def _validate_smoke_records(
    records: Any,
    smoke_items: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    import math

    if not isinstance(records, list) or len(records) != 2:
        raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke must produce exactly two records")
    expected_names = [str(item["name"]) for item in smoke_items]
    validated: list[dict[str, Any]] = []
    for record, expected_name in zip(records, expected_names, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "namespace",
            "name",
            "score",
            "audio",
        }:
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke record schema changed")
        if record["namespace"] != "nonconfirmatory-smoke" or record["name"] != expected_name:
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke coordinate changed")
        score = record["score"]
        if not isinstance(score, Mapping) or set(score) != {
            "vocabulary_size",
            "positions",
            "actual_output",
        }:
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke score schema changed")
        vocabulary_size = score["vocabulary_size"]
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= max(KS)
        ):
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke vocabulary changed")
        positions = score["positions"]
        if not isinstance(positions, Mapping) or set(positions) != set(POSITIONS):
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke positions changed")
        position_indices: dict[str, int] = {}
        for position_name in POSITIONS:
            position = positions[position_name]
            if not isinstance(position, Mapping) or set(position) != {
                "index",
                "layers",
            }:
                raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke position schema changed")
            index = position["index"]
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke position index changed")
            position_indices[position_name] = index
            layers = position["layers"]
            if not isinstance(layers, Mapping) or set(layers) != {
                str(layer) for layer in ALL_LAYERS
            }:
                raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke must contain L0-L33")
            for layer in layers.values():
                expected_layer_keys = {
                    "concept_ranks",
                    "candidate_label_pool_ranks",
                }
                if position_name == "response_position":
                    expected_layer_keys.add("motor")
                if not isinstance(layer, Mapping) or set(layer) != expected_layer_keys:
                    raise ModalAudioWorkspaceEvalError(
                        "nonconfirmatory smoke layer evidence changed"
                    )
                concept_ranks = layer["concept_ranks"]
                if not isinstance(concept_ranks, Mapping) or set(concept_ranks) != set(CONTROLS):
                    raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke controls changed")
                rank_maps = [
                    *concept_ranks.values(),
                    layer["candidate_label_pool_ranks"],
                ]
                if any(
                    not isinstance(rank_map, Mapping)
                    or any(
                        isinstance(rank, bool)
                        or not isinstance(rank, int)
                        or not 1 <= rank <= vocabulary_size
                        for rank in rank_map.values()
                    )
                    for rank_map in rank_maps
                ):
                    raise ModalAudioWorkspaceEvalError(
                        "nonconfirmatory smoke full-vocabulary ranks changed"
                    )
                candidate_ranks = concept_ranks["candidate"]
                label_pool_ranks = layer["candidate_label_pool_ranks"]
                if not candidate_ranks or any(
                    label_pool_ranks.get(concept_id) != rank
                    for concept_id, rank in candidate_ranks.items()
                ):
                    raise ModalAudioWorkspaceEvalError(
                        "nonconfirmatory smoke candidate pool identity changed"
                    )
                if position_name == "response_position":
                    motor = layer["motor"]
                    if (
                        not isinstance(motor, Mapping)
                        or set(motor)
                        != {
                            "actual_token_id",
                            "actual_token_ranks",
                            "candidate_logit_top1_agreement",
                            "candidate_logit_js_nats",
                        }
                        or motor["actual_token_id"] != score["actual_output"].get("token_id")
                        or not isinstance(motor["actual_token_ranks"], Mapping)
                        or set(motor["actual_token_ranks"]) != {"candidate", "logit"}
                        or any(
                            isinstance(rank, bool)
                            or not isinstance(rank, int)
                            or not 1 <= rank <= vocabulary_size
                            for rank in motor["actual_token_ranks"].values()
                        )
                        or not isinstance(motor["candidate_logit_top1_agreement"], bool)
                        or isinstance(motor["candidate_logit_js_nats"], bool)
                        or not isinstance(motor["candidate_logit_js_nats"], (int, float))
                        or not math.isfinite(float(motor["candidate_logit_js_nats"]))
                        or not 0.0 <= float(motor["candidate_logit_js_nats"]) <= math.log(2.0)
                    ):
                        raise ModalAudioWorkspaceEvalError(
                            "nonconfirmatory smoke motor evidence changed"
                        )
        if not (
            position_indices["last_processor_valid_audio_position"]
            < position_indices["response_position"]
        ):
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke position geometry changed")
        actual = score["actual_output"]
        if (
            not isinstance(actual, Mapping)
            or set(actual) != {"layer", "position", "position_index", "token_id"}
            or actual["layer"] != FINAL_MODEL_LAYER
            or actual["position"] != "response_position"
            or actual["position_index"] != position_indices["response_position"]
            or isinstance(actual["token_id"], bool)
            or not isinstance(actual["token_id"], int)
            or not 0 <= actual["token_id"] < vocabulary_size
        ):
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke L34 evidence changed")
        audio = record["audio"]
        if (
            not isinstance(audio, Mapping)
            or audio.get("source_sample_rate") != SOURCE_SAMPLE_RATE
            or audio.get("sample_rate") != SAMPLE_RATE
            or isinstance(audio.get("sample_count"), bool)
            or not isinstance(audio.get("sample_count"), int)
            or audio["sample_count"] <= 0
            or any(
                not isinstance(audio.get(key), str) or len(audio[key]) != 64
                for key in (
                    "source_decoded_pcm_sha256",
                    "decoded_pcm_sha256",
                )
            )
        ):
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke audio evidence changed")
        validated.append(dict(record))
    return validated


def _smoke_impl(
    tts_recipe_sha256: str = "",
    *,
    fixture_loader: Callable[[], Mapping[str, bytes]] = _fetch_fixtures,
    inference: Callable[[Sequence[Mapping[str, Any]]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    raw_fixtures = fixture_loader()
    association_spec = next(spec for spec in FIXTURE_SPECS if spec["distribution"] == "association")
    source_rows = _decode_fixture(
        association_spec,
        raw_fixtures["association"],
    )[50:52]
    if len(source_rows) != 2:
        raise ModalAudioWorkspaceEvalError("nonpublication smoke rows are unavailable")
    smoke_items = [
        {
            "distribution": "smoke-association-nonpublication",
            "publication_index": 50 + index,
            "name": f"nonconfirmatory/{row['name']}",
            "script": str(row["prompt"]),
            "script_sha256": _sha256_bytes(str(row["prompt"]).encode("utf-8")),
            "language": "english",
            "voice_code": "en-us",
            "intermediates": list(row["intermediates"]),
        }
        for index, row in enumerate(source_rows)
    ]
    if inference is None:
        from audiolens.audio_workspace_eval import tts_input

        recipe, recipe_root = _load_tts_recipe(tts_recipe_sha256)
        engine_identity = {
            "recipe_sha256": recipe["recipe_sha256"],
            "synthesized_at": recipe["synthesized_at"],
            **recipe["engine"],
        }
        torch = _deterministic_torch()
        from audiolens.audio_eval_model import prepare_audio_evaluation
        from audiolens.models import (
            DEFAULT_MODEL_PROFILE,
            load_model_runtime,
        )

        runtime = load_model_runtime(DEFAULT_MODEL_PROFILE.key, device_map="cuda")
        runtime.model.eval()
        if runtime.profile.model_id != MODEL_ID or runtime.profile.model_revision != MODEL_REVISION:
            raise ModalAudioWorkspaceEvalError("smoke loaded an unexpected Gemma profile")
        smoke_runtime_identity = _runtime_identity(include_cuda=True)
        source_layers, output_permutation, permutation_digest = _control_permutations(torch, 1_536)
        output_permutation = output_permutation.to(runtime.input_device)
        jacobians = {
            layer: _sacrificial_jacobian(
                torch,
                layer,
                1_536,
                runtime.input_device,
            )
            for layer in ALL_LAYERS
        }
        permuted_jacobians = {
            layer: jacobians[source_layers[layer]].index_select(0, output_permutation).contiguous()
            for layer in ALL_LAYERS
        }
        smoke_eligibility = _eligibility(
            [
                {
                    **item,
                    "distribution": "association",
                }
                for item in smoke_items
            ],
            runtime.processor.tokenizer,
        )
        eligibility_by_name = {str(item["name"]): item for item in smoke_eligibility["items"]}
        distribution_pool = [
            concept for item in smoke_eligibility["items"] for concept in item["concepts"]
        ]
        if not distribution_pool:
            raise ModalAudioWorkspaceEvalError("nonconfirmatory smoke has no eligible concepts")
        with tempfile.TemporaryDirectory(prefix="audio-workspace-smoke-") as temporary:
            root = pathlib.Path(temporary)
            staged: list[tuple[dict[str, Any], pathlib.Path, dict[str, Any]]] = []
            for item, entry in zip(smoke_items, recipe["smoke_observations"], strict=True):
                spoken = tts_input(str(item["script"]))
                if not isinstance(entry, Mapping) or (
                    entry.get("publication_index"),
                    entry.get("name"),
                    entry.get("script_sha256"),
                    entry.get("tts_input"),
                    entry.get("variant"),
                    entry.get("language"),
                ) != (
                    item["publication_index"],
                    item["name"],
                    item["script_sha256"],
                    spoken,
                    TTS_VARIANTS[0],
                    "en-us",
                ):
                    raise ModalAudioWorkspaceEvalError(
                        "sealed TTS recipe smoke rows do not match the nonpublication items"
                    )
                source = _read_recipe_wav(recipe_root, entry, label="smoke source WAV")
                wav, audio = _normalize_wav(source)
                path = root / f"{item['publication_index']}.wav"
                path.write_bytes(wav)
                staged.append((item, path, audio))

            def run_once() -> list[dict[str, Any]]:
                records: list[dict[str, Any]] = []
                for item, path, audio in staged:
                    prepared = prepare_audio_evaluation(
                        runtime.processor,
                        path,
                        max_sequence_length=MAX_SEQUENCE_LENGTH,
                    )
                    score = _score_one_observation(
                        runtime,
                        prepared,
                        eligibility_by_name[item["name"]],
                        distribution_pool,
                        jacobians,
                        permuted_jacobians,
                        torch,
                    )
                    records.append(
                        {
                            "namespace": "nonconfirmatory-smoke",
                            "name": item["name"],
                            "score": score,
                            "audio": audio,
                        }
                    )
                return records

            first = run_once()
            second = run_once()
    else:
        first = inference(smoke_items)
        second = inference(smoke_items)
        permutation_digest = _sha256_json({"test_seam": "injected"})
        engine_identity = {"test_seam": "injected"}
        smoke_runtime_identity = {"test_seam": "injected"}
    if _canonical_json_bytes(first) != _canonical_json_bytes(second):
        raise ModalAudioWorkspaceEvalError("duplicate deterministic smoke inference changed")
    first = _validate_smoke_records(first, smoke_items)
    report = {
        "schema_version": 1,
        "kind": "audio_workspace_nonconfirmatory_smoke",
        "status": "complete",
        "namespace": "nonconfirmatory-smoke",
        "source": _require_source_identity(),
        "runtime": smoke_runtime_identity,
        "tts_engine": engine_identity,
        "items": smoke_items,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "uses_final_lens": False,
        "uses_confirmatory_items": False,
        "matrix_policy": "sacrificial_deterministic_diagonal_plus_shift",
        "control_output_permutation_sha256": permutation_digest,
        "records": first,
        "duplicate_inference_equal": True,
    }
    path, digest = _write_content_addressed_json(SMOKE_ROOT, report)
    _commit_volume()
    return {
        "mode": "smoke",
        "status": "complete",
        "path": str(path),
        "sha256": digest,
        "records": len(first),
    }


def _validate_report_impl(
    report_path: str,
    report_sha256: str,
    *,
    preregistration_loader: Callable[[str, str], dict[str, Any]] | None = None,
    text_loader: Callable[[], dict[str, Any]] | None = None,
    physical_validator: Callable[
        ...,
        tuple[dict[str, Any], pathlib.Path, dict[str, Any]],
    ]
    | None = None,
    completed_run_loader: Callable[[], dict[str, Any]] | None = None,
    completed_run_bytes: Callable[[], bytes] | None = None,
    artifact_validator: Callable[..., dict[str, Any]] | None = None,
    report_validator: Callable[..., dict[str, Any]] | None = None,
    report_builder: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate = pathlib.Path(report_path)
    if (
        candidate.parent != pathlib.Path(REPORT_ROOT)
        or candidate.suffix != ".json"
        or candidate.stem != report_sha256
    ):
        raise ModalAudioWorkspaceEvalError(
            "report must be its content-addressed frozen report path"
        )
    report = _read_json(candidate, label="audio workspace report")
    if report.get("report_sha256") != report_sha256:
        raise ModalAudioWorkspaceEvalError("report requested SHA-256 mismatch")
    preregistration_sha = report.get("preregistration_sha256")
    if not isinstance(preregistration_sha, str):
        raise ModalAudioWorkspaceEvalError("report lacks preregistration identity")
    if preregistration_loader is None:
        preregistration_loader = _validate_preregistration_file
    if text_loader is None:
        text_loader = _load_canonical_text_report
    if physical_validator is None:
        physical_validator = _validate_physical_stimuli
    if completed_run_loader is None:

        def load_completed_run() -> dict[str, Any]:
            return _read_json(
                FINAL_AUDIO_RUN_PATH,
                label="completed audio run",
            )

        completed_run_loader = load_completed_run
    if completed_run_bytes is None:

        def load_completed_run_bytes() -> bytes:
            return pathlib.Path(FINAL_AUDIO_RUN_PATH).read_bytes()

        completed_run_bytes = load_completed_run_bytes
    if artifact_validator is None:
        from audiolens.audio_workspace_eval import (
            validate_audio_artifact_chain,
        )

        artifact_validator = validate_audio_artifact_chain
    if report_validator is None:
        report_validator = _validate_complete_report
    if report_builder is None:
        report_builder = _build_complete_report

    preregistration_path = pathlib.Path(PREREGISTRATION_ROOT) / f"{preregistration_sha}.json"
    preregistration = preregistration_loader(
        str(preregistration_path),
        preregistration_sha,
    )
    text_report = text_loader()
    manifest, _manifest_path, _calibration = physical_validator(
        preregistration,
        verify_normalization=True,
    )
    _bind_items_to_text_report(manifest["items"], text_report)
    run = completed_run_loader()
    expected_run_sha = preregistration["artifact_identity"]["completed_run_manifest_sha256"]
    run_manifest_bytes = completed_run_bytes()
    if _sha256_bytes(run_manifest_bytes) != expected_run_sha:
        raise ModalAudioWorkspaceEvalError("completed audio run manifest bytes changed")
    artifact_identity = artifact_validator(
        run,
        completed_run_manifest_bytes=run_manifest_bytes,
        volume_root=VOL_MOUNT,
        completed_run_manifest_sha256=expected_run_sha,
    )
    if artifact_identity != preregistration["artifact_identity"]:
        raise ModalAudioWorkspaceEvalError("independently validated audio run identity changed")
    records = report.get("records")
    if not isinstance(records, list):
        raise ModalAudioWorkspaceEvalError(
            "complete report records are unavailable for reconstruction"
        )
    rebuilt = report_builder(
        preregistration,
        records,
        text_report,
        {
            "validated_audio_artifact": artifact_identity,
            "stimulus_manifest_sha256": manifest["stimulus_manifest_sha256"],
        },
    )
    validated = report_validator(
        report,
        preregistration,
        text_report,
    )
    if validated != report or rebuilt != validated:
        raise ModalAudioWorkspaceEvalError("independent report reconstruction changed")
    if rebuilt["report_sha256"] != report_sha256:
        raise ModalAudioWorkspaceEvalError("independent report SHA-256 reproduction failed")
    return {
        "mode": "validate-report",
        "status": rebuilt["status"],
        "path": str(candidate),
        "sha256": report_sha256,
        "records": len(rebuilt["records"]),
        "independently_reproduced": True,
    }


@_modal_gpu_function(timeout=24 * 60 * 60)
def preregister_experiment(tts_recipe: str) -> str:
    return json.dumps(_preregister_impl(tts_recipe), sort_keys=True)


@_modal_gpu_function(timeout=6 * 60 * 60)
def smoke_experiment(tts_recipe: str) -> str:
    return json.dumps(_smoke_impl(tts_recipe), sort_keys=True)


@_modal_gpu_function(timeout=24 * 60 * 60)
def evaluate_experiment(preregistration: str, sha256: str) -> str:
    return json.dumps(_evaluate_impl(preregistration, sha256), sort_keys=True)


@_modal_cpu_function(timeout=6 * 60 * 60)
def validate_report_experiment(report: str, sha256: str) -> str:
    return json.dumps(_validate_report_impl(report, sha256), sort_keys=True)


def _dispatch(
    *,
    preregister: bool,
    smoke: bool,
    evaluate: bool,
    validate_report: str,
    preregistration: str,
    sha256: str,
    tts_recipe: str,
    preregister_call: Callable[..., Any],
    smoke_call: Callable[..., Any],
    evaluate_call: Callable[..., Any],
    validate_call: Callable[..., Any],
) -> Any:
    modes = int(preregister) + int(smoke) + int(evaluate) + int(bool(validate_report))
    if modes != 1:
        raise SystemExit(
            "select exactly one of --preregister, --smoke, --evaluate, or --validate-report"
        )
    if preregister:
        if preregistration or sha256:
            raise SystemExit("--preregister accepts no artifact arguments")
        if not tts_recipe:
            raise SystemExit("--preregister requires --tts-recipe")
        return preregister_call(tts_recipe=tts_recipe)
    if smoke:
        if preregistration or sha256:
            raise SystemExit("--smoke accepts no artifact arguments")
        if not tts_recipe:
            raise SystemExit("--smoke requires --tts-recipe")
        return smoke_call(tts_recipe=tts_recipe)
    if tts_recipe:
        raise SystemExit("--tts-recipe applies only to --preregister and --smoke")
    if evaluate:
        if not preregistration or not sha256:
            raise SystemExit("--evaluate requires --preregistration and --sha256")
        return evaluate_call(preregistration=preregistration, sha256=sha256)
    if preregistration or not sha256:
        raise SystemExit("--validate-report requires its path and --sha256 only")
    return validate_call(report=validate_report, sha256=sha256)


@_modal_local_entrypoint
def main(
    preregister: bool = False,
    smoke: bool = False,
    evaluate: bool = False,
    validate_report: str = "",
    preregistration: str = "",
    sha256: str = "",
    tts_recipe: str = "",
):
    """Dispatch one and only one sealed deployment mode."""
    result = _dispatch(
        preregister=preregister,
        smoke=smoke,
        evaluate=evaluate,
        validate_report=validate_report,
        preregistration=preregistration,
        sha256=sha256,
        tts_recipe=tts_recipe,
        preregister_call=preregister_experiment.remote,
        smoke_call=smoke_experiment.remote,
        evaluate_call=evaluate_experiment.remote,
        validate_call=validate_report_experiment.remote,
    )
    print(result)


if __name__ == "__main__" and not inspect.isfunction(main):
    raise SystemExit("run this module through `modal run`, not as plain Python")
