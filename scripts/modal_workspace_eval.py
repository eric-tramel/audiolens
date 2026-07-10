"""Frozen H100 evaluation of the canonical Gemma J-lens.

The evaluator reproduces Anthropic's six released lens-quality distributions,
then adjudicates one preregistered transfer hypothesis: L13--L31 behaves more
like an intermediate workspace readout than L0--L12 or the L32--L33 motor
region.  It never searches for another band.  ``no_band`` is a complete,
scientifically valid outcome.

Run with explicit, completed fit artifacts on the shared Modal volume::

    uv run modal run scripts/modal_workspace_eval.py \
        --fit-manifest /vol/runs/<content-addressed-fit>.json \
        --lens /vol/lenses/<content-addressed-fit>.pt
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOL_MOUNT = "/vol"
MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "70af34e20bd4b7a91f0de6b22675850c43922a03"
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
D_MODEL = 1_536
WORKSPACE_REPORT_SCHEMA_VERSION = 1
WORKSPACE_REPORT_KIND = "workspace_jlens_evaluation"

ALL_LAYERS = tuple(range(34))
EARLY_LAYERS = tuple(range(13))
CANDIDATE_LAYERS = tuple(range(13, 32))
MOTOR_LAYERS = (32, 33)
REGIONS = {
    "early_l0_l12": EARLY_LAYERS,
    "candidate_l13_l31": CANDIDATE_LAYERS,
    "motor_l32_l33": MOTOR_LAYERS,
}
KS = (1, 2, 5, 10, 20, 50, 100)
VARIANTS = ("candidate", "logit", "transposed", "permuted")
PERMUTATION_REPLICATES = 1_000
PERMUTATION_SEED = 2026070901
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 2026070902
CONTROL_SEED = 2026070903
BATCH_SIZE = 8
FIT_MANIFEST_MAX_BYTES = 4 * 1024 * 1024
JS_DIVERGENCE_MAX_NATS = math.log(2.0)
JS_DIVERGENCE_FLOAT32_TOLERANCE = 1e-6
EXPECTED_RUNTIME_PACKAGES = {
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "huggingface-hub": "1.22.0",
    "jlens": "0.1.0",
    "modal": "1.5.1",
    "torch": "2.13.0",
    "transformers": "5.13.0",
}


class WorkspaceEvalContractError(RuntimeError):
    """A fail-closed benchmark, identity, scoring, or report violation."""


@dataclass(frozen=True)
class FixtureSpec:
    slug: str
    filename: str
    sha256: str
    n_bytes: int
    raw_count: int
    publication_count: int
    selected_name_sha256: str
    item_keys: frozenset[str]
    target_boundary: bool
    minimum_eligible_items: int
    minimum_eligible_concepts: int

    @property
    def url(self) -> str:
        return (
            "https://raw.githubusercontent.com/anthropics/jacobian-lens/"
            f"{JLENS_REVISION}/data/evaluations/{self.filename}"
        )


FIXTURES = (
    FixtureSpec(
        "association",
        "lens-eval-association.json",
        "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
        24_228,
        102,
        50,
        "107eddeee767b029528f200718167bed2f76fd25a0e08b54ea9289da92066e68",
        frozenset({"name", "prompt", "intermediates"}),
        False,
        50,
        50,
    ),
    FixtureSpec(
        "multihop",
        "lens-eval-multihop.json",
        "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
        21_869,
        93,
        50,
        "377d116630bffe505157e408906cc811860e771db509261a67f1c4188b51d033",
        frozenset({"name", "prompt", "target", "intermediates"}),
        True,
        48,
        56,
    ),
    FixtureSpec(
        "multilingual",
        "lens-eval-multilingual.json",
        "fa70b9bd89416a6d8d985a80dc628b109ae6fd3b25b9275c0fc5065d7ff4a0ef",
        24_284,
        107,
        54,
        "ed559e9071ff5b381febe7980447213577fc6fd4d212490241e54da7072849c8",
        frozenset({"name", "prompt", "target", "intermediates"}),
        True,
        54,
        216,
    ),
    FixtureSpec(
        "order-ops",
        "lens-eval-order-ops.json",
        "b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560",
        9_589,
        55,
        55,
        "73f146f5950d27f23abc778da086604d727d5f4b2aa7efd14658dd9fc9e5082d",
        frozenset({"name", "prompt", "target", "intermediates"}),
        True,
        55,
        109,
    ),
    FixtureSpec(
        "poetry",
        "lens-eval-poetry.json",
        "6aeb3415c5a5c3f3827c9efe63f006de02f5ef39a816bbac68e15e733aba60cc",
        21_533,
        98,
        52,
        "f1e324ec4d18814473f8ac7fea68464cb48106d5d2ae83eddb7db3ae6f68f0b9",
        frozenset({"name", "prompt", "intermediates"}),
        False,
        52,
        52,
    ),
    FixtureSpec(
        "typo",
        "lens-eval-typo.json",
        "9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa",
        14_646,
        96,
        96,
        "d9a1e6944adcde9b840b4b2e3d1b05451cf80743db514a128f62e7139d06ded0",
        frozenset({"name", "prompt", "intermediates"}),
        False,
        96,
        96,
    ),
)
FIXTURE_BY_SLUG = {fixture.slug: fixture for fixture in FIXTURES}

# The release documents number digit/word forms and operation symbol/word
# forms but does not publish an exhaustive table.  This frozen table is an
# explicit AudioLens evaluator policy; its canonical hash is report identity.
ORDER_OP_SYNONYMS: dict[str, tuple[str, ...]] = {
    "3": ("3", "three"),
    "4": ("4", "four"),
    "5": ("5", "five"),
    "6": ("6", "six"),
    "7": ("7", "seven"),
    "8": ("8", "eight"),
    "9": ("9", "nine"),
    "10": ("10", "ten"),
    "11": ("11", "eleven"),
    "12": ("12", "twelve"),
    "13": ("13", "thirteen"),
    "15": ("15", "fifteen"),
    "16": ("16", "sixteen"),
    "20": ("20", "twenty"),
    "24": ("24", "twenty-four", "twenty four"),
    "addition": ("addition", "+", "plus", "add"),
    "division": ("division", "/", "÷", "divide", "divided"),
    "mod": ("mod", "%", "modulo", "remainder"),
    "multiplication": ("multiplication", "*", "×", "times", "multiply"),
    "squared": ("squared", "²", "square"),
    "subtraction": ("subtraction", "-", "−", "minus", "subtract"),
}


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
WORKSPACE_SOURCE_RELATIVES = (
    *FIT_SOURCE_RELATIVES,
    "scripts/modal_workspace_eval.py",
)


def _source_digest() -> str:
    relatives = WORKSPACE_SOURCE_RELATIVES
    if all((REPO_ROOT / relative).is_file() for relative in relatives):
        digest = hashlib.sha256()
        for relative in relatives:
            digest.update(relative.encode())
            digest.update((REPO_ROOT / relative).read_bytes())
        return digest.hexdigest()
    injected = os.environ.get("AUDIOLENS_WORKSPACE_EVAL_SOURCE_DIGEST")
    if injected:
        return injected
    raise RuntimeError("cannot determine workspace-evaluation source digest")

def _fit_source_digest() -> str:
    """Identity needed when importing the fit validator in a Modal image."""
    relatives = FIT_SOURCE_RELATIVES
    if all((REPO_ROOT / relative).is_file() for relative in relatives):
        digest = hashlib.sha256()
        for relative in relatives:
            path = REPO_ROOT / relative
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()
    injected = os.environ.get("AUDIOLENS_SOURCE_DIGEST")
    if injected:
        return injected
    raise RuntimeError("cannot determine canonical-fit source digest")


def _lock_digest() -> str:
    path = REPO_ROOT / "uv.lock"
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    injected = os.environ.get("AUDIOLENS_LOCK_SHA256")
    if injected:
        return injected
    raise RuntimeError("cannot determine uv.lock digest")



_HAS_LOCAL_PROJECT = all(
    (REPO_ROOT / relative).is_file()
    for relative in WORKSPACE_SOURCE_RELATIVES
)
_DEPLOYMENT_ENABLED = (
    _HAS_LOCAL_PROJECT
    and os.environ.get("AUDIOLENS_REPORT_INSPECTOR_ONLY") != "1"
)


GIT_REVISION = _git_revision()
SOURCE_DIGEST = _source_digest()
FIT_SOURCE_DIGEST = _fit_source_digest()
LOCK_SHA256 = _lock_digest()

if _DEPLOYMENT_ENABLED:
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
                "AUDIOLENS_WORKSPACE_EVAL_SOURCE_DIGEST": SOURCE_DIGEST,
                "AUDIOLENS_SOURCE_DIGEST": FIT_SOURCE_DIGEST,
                "AUDIOLENS_LOCK_SHA256": LOCK_SHA256,
            }
        )
        .add_local_python_source("audiolens")
        .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/root/scripts")
    )
    app = modal.App("audiolens-workspace-eval", image=image)
    vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)
else:
    image = None
    app = None
    vol = None


def _modal_eval_function(function):
    if app is None or vol is None:
        return function
    return app.function(
        gpu="H100",
        timeout=4 * 60 * 60,
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
        vol = modal.Volume.from_name("audiolens-vol", create_if_missing=False)
    vol.commit()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fit_manifest_input(
    path: str | pathlib.Path,
    volume_root: str | pathlib.Path,
) -> pathlib.Path:
    """Reject caller-controlled paths before the fit loader reads any bytes."""
    import stat

    candidate = pathlib.Path(path)
    runs_root = pathlib.Path(volume_root) / "runs"
    if (
        not candidate.is_absolute()
        or candidate.parent != runs_root
        or candidate.suffix != ".json"
    ):
        raise WorkspaceEvalContractError(
            "fit manifest must be an explicit JSON file directly under /vol/runs"
        )
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise WorkspaceEvalContractError(
            f"fit manifest is missing at {candidate}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise WorkspaceEvalContractError(
            "fit manifest must be a nonsymlink regular file"
        )
    if metadata.st_size <= 0 or metadata.st_size > FIT_MANIFEST_MAX_BYTES:
        raise WorkspaceEvalContractError(
            f"fit manifest size must be 1..{FIT_MANIFEST_MAX_BYTES} bytes"
        )
    return candidate

def _load_source_bound_fit_manifest(
    path: str | pathlib.Path,
    volume_root: str | pathlib.Path,
    loader: Any,
) -> tuple[pathlib.Path, dict[str, Any]]:
    """Check bounded manifest provenance before loader touches the lens."""
    candidate = _validate_fit_manifest_input(path, volume_root)
    with open(candidate, "rb") as source:
        raw = source.read(FIT_MANIFEST_MAX_BYTES + 1)
    if len(raw) > FIT_MANIFEST_MAX_BYTES:
        raise WorkspaceEvalContractError("fit manifest grew beyond its size bound")
    try:
        preview = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceEvalContractError(
            f"fit manifest JSON is invalid at {candidate}"
        ) from exc
    if (
        not isinstance(preview, Mapping)
        or not isinstance(preview.get("config"), Mapping)
        or not isinstance(preview["config"].get("source"), Mapping)
        or preview["config"]["source"].get("digest") != FIT_SOURCE_DIGEST
    ):
        raise WorkspaceEvalContractError(
            "fit manifest source digest does not match bundled canonical fitter"
        )
    record = loader(candidate, volume_root=volume_root)
    return candidate, record


def _atomic_write_json(path: str | pathlib.Path, value: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value) + b"\n"
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _fixture_identity(spec: FixtureSpec) -> dict[str, Any]:
    return {
        "slug": spec.slug,
        "filename": spec.filename,
        "url": spec.url,
        "commit": JLENS_REVISION,
        "sha256": spec.sha256,
        "bytes": spec.n_bytes,
        "raw_count": spec.raw_count,
        "publication_count": spec.publication_count,
        "selection": "ordered_prefix_v1",
        "selected_name_encoding": "compact_json_utf8_ensure_ascii_false",
        "selected_name_sha256": spec.selected_name_sha256,
        "minimum_eligible_items": spec.minimum_eligible_items,
        "minimum_eligible_concepts": spec.minimum_eligible_concepts,
    }


def _valid_prompt_shape(prompt: Any) -> bool:
    if isinstance(prompt, str):
        return True
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and all(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message["role"] in {"user", "assistant"}
            and isinstance(message["content"], str)
            for message in prompt
        )
    )


def _decode_fixture(spec: FixtureSpec, raw: bytes) -> list[dict[str, Any]]:
    """Verify pinned bytes, exact schema, publication prefix, and item names."""
    if len(raw) != spec.n_bytes:
        raise WorkspaceEvalContractError(
            f"{spec.slug}: expected {spec.n_bytes} bytes, received {len(raw)}"
        )
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != spec.sha256:
        raise WorkspaceEvalContractError(
            f"{spec.slug}: byte SHA-256 mismatch {actual_sha}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceEvalContractError(f"{spec.slug}: invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise WorkspaceEvalContractError(f"{spec.slug}: top-level schema mismatch")
    items = payload["items"]
    if not isinstance(items, list) or len(items) != spec.raw_count:
        raise WorkspaceEvalContractError(
            f"{spec.slug}: expected {spec.raw_count} raw items"
        )
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != spec.item_keys:
            raise WorkspaceEvalContractError(
                f"{spec.slug}[{index}]: item schema mismatch"
            )
        name = item.get("name")
        prompt = item.get("prompt")
        concepts = item.get("intermediates")
        if not isinstance(name, str) or not name or name in seen:
            raise WorkspaceEvalContractError(f"{spec.slug}[{index}]: invalid name")
        seen.add(name)
        if not _valid_prompt_shape(prompt):
            raise WorkspaceEvalContractError(f"{spec.slug}/{name}: invalid prompt")
        if (
            not isinstance(concepts, list)
            or not concepts
            or not all(isinstance(concept, str) and concept for concept in concepts)
        ):
            raise WorkspaceEvalContractError(
                f"{spec.slug}/{name}: invalid intermediates"
            )
        if spec.target_boundary and (
            not isinstance(item.get("target"), str) or not item["target"]
        ):
            raise WorkspaceEvalContractError(f"{spec.slug}/{name}: invalid target")
    selected = items[: spec.publication_count]
    name_hash = _digest([item["name"] for item in selected])
    if name_hash != spec.selected_name_sha256:
        raise WorkspaceEvalContractError(
            f"{spec.slug}: ordered selected-name SHA-256 mismatch {name_hash}"
        )
    return selected


def _read_bounded_fixture(stream: Any, spec: FixtureSpec) -> bytes:
    raw = stream.read(spec.n_bytes + 1)
    if len(raw) > spec.n_bytes:
        raise WorkspaceEvalContractError(
            f"{spec.slug}: response exceeds pinned {spec.n_bytes} bytes"
        )
    return raw


def _fetch_fixtures(cache_root: str | pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    import urllib.request

    root = pathlib.Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    decoded: dict[str, list[dict[str, Any]]] = {}
    for spec in FIXTURES:
        path = root / spec.filename
        if not path.exists():
            temporary = path.with_name(f"{path.name}.part.{os.getpid()}")
            try:
                with urllib.request.urlopen(spec.url, timeout=60) as response:
                    raw = _read_bounded_fixture(response, spec)
                _decode_fixture(spec, raw)
                temporary.write_bytes(raw)
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        with open(path, "rb") as cached:
            raw = _read_bounded_fixture(cached, spec)
        decoded[spec.slug] = _decode_fixture(spec, raw)
    return decoded


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _render_contexts(
    tokenizer: Any,
    prompt: str | list[dict[str, str]],
    target: str | None,
) -> tuple[str, str]:
    """Render prompt and complete context with an open final continuation."""
    if isinstance(prompt, str):
        return prompt, prompt + (target or "")
    if not _valid_prompt_shape(prompt):
        raise WorkspaceEvalContractError("invalid multi-turn prompt")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise WorkspaceEvalContractError("tokenizer cannot render multi-turn prompt")
    messages = [dict(message) for message in prompt]
    if target is None:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        prompt_rendered = full_rendered = rendered
    elif messages[-1]["role"] == "assistant":
        prompt_rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        full_messages = [*messages[:-1], dict(messages[-1])]
        full_messages[-1]["content"] += target
        full_rendered = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        prompt_rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )
        full_rendered = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    if (
        not isinstance(prompt_rendered, str)
        or not isinstance(full_rendered, str)
        or not prompt_rendered
        or not full_rendered
        or not full_rendered.startswith(prompt_rendered)
    ):
        raise WorkspaceEvalContractError(
            "chat template did not preserve the declared prompt prefix"
        )
    if target is not None and not full_rendered.endswith(target):
        raise WorkspaceEvalContractError(
            "chat template did not preserve the declared target suffix"
        )
    return prompt_rendered, full_rendered


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        offsets = offsets[0]
    result_ids = [int(token) for token in ids]
    result_offsets = [(int(start), int(stop)) for start, stop in offsets]
    if len(result_ids) != len(result_offsets) or not result_ids:
        raise WorkspaceEvalContractError("tokenizer returned invalid offsets")
    return result_ids, result_offsets


def _force_bos(tokenizer: Any, ids: Sequence[int]) -> list[int]:
    bos = getattr(tokenizer, "bos_token_id", None)
    if not isinstance(bos, int) or bos < 0:
        raise WorkspaceEvalContractError("tokenizer has no valid BOS token")
    values = [int(token) for token in ids]
    return values if values and values[0] == bos else [bos, *values]


def _overlap_span(
    offsets: Sequence[tuple[int, int]], char_start: int, char_stop: int
) -> tuple[int, int]:
    overlap = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > char_start and start < char_stop
    ]
    if not overlap:
        raise WorkspaceEvalContractError("target has zero tokenizer tokens")
    expected = list(range(overlap[0], overlap[-1] + 1))
    if overlap != expected:
        raise WorkspaceEvalContractError("target token span is noncontiguous")
    return overlap[0], overlap[-1] + 1


def _decoded_token(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode([int(token_id)]))


def _build_readout(
    tokenizer: Any,
    distribution: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one no-leakage scored prefix from the complete declared context."""
    target = item.get("target")
    prompt_text, full_text = _render_contexts(tokenizer, item["prompt"], target)
    if target is not None:
        if distribution not in {"multihop", "multilingual", "order-ops"}:
            raise WorkspaceEvalContractError(
                f"{distribution}/{item['name']}: unexpected target field"
            )
        if not full_text.endswith(target):
            raise WorkspaceEvalContractError(
                f"{distribution}/{item['name']}: target is not a final continuation"
            )
        ids, offsets = _tokenize_with_offsets(tokenizer, full_text)
        target_char_start = len(full_text) - len(target)
        target_start, target_stop = _overlap_span(
            offsets, target_char_start, len(full_text)
        )
        if target_start == 0:
            raise WorkspaceEvalContractError(
                f"{distribution}/{item['name']}: target has no predecessor"
            )
        prefix_unbos = ids[:target_start]
        full_bos = _force_bos(tokenizer, ids)
        prefix_bos = _force_bos(tokenizer, prefix_unbos)
        bos_shift = len(full_bos) - len(ids)
        if len(prefix_bos) - len(prefix_unbos) != bos_shift:
            raise WorkspaceEvalContractError("prompt/full BOS policies differ")
        target_span = [target_start + bos_shift, target_stop + bos_shift]
        target_token_ids = full_bos[target_span[0] : target_span[1]]
        if not target_token_ids or len(prefix_bos) != target_span[0]:
            raise WorkspaceEvalContractError("target span leaked into scored prefix")
    else:
        ids, offsets = _tokenize_with_offsets(tokenizer, full_text)
        target_span = None
        target_token_ids = []
        if distribution == "poetry":
            newline = full_text.rfind("\n")
            if newline < 0:
                raise WorkspaceEvalContractError(
                    f"poetry/{item['name']}: prompt has no newline"
                )
            scored_unbos, newline_stop = _overlap_span(offsets, newline, newline + 1)
            if newline_stop != scored_unbos + 1:
                raise WorkspaceEvalContractError(
                    f"poetry/{item['name']}: newline is not one token"
                )
        elif distribution == "association":
            if not full_text.endswith("."):
                raise WorkspaceEvalContractError(
                    f"association/{item['name']}: prompt lacks final period"
                )
            scored_unbos = len(ids) - 1
            start, stop = offsets[scored_unbos]
            if not (start <= len(full_text) - 1 < stop):
                raise WorkspaceEvalContractError(
                    f"association/{item['name']}: final token is not the period"
                )
        elif distribution == "typo":
            scored_unbos = len(ids) - 1
        else:
            raise WorkspaceEvalContractError(
                f"{distribution}/{item['name']}: missing required target"
            )
        prefix_unbos = ids[: scored_unbos + 1]
        full_bos = _force_bos(tokenizer, ids)
        prefix_bos = _force_bos(tokenizer, prefix_unbos)
    scored_position = len(prefix_bos) - 1
    if scored_position < 1 or scored_position >= len(prefix_bos):
        raise WorkspaceEvalContractError(
            f"{distribution}/{item['name']}: invalid scored position"
        )
    if target_span is not None and len(prefix_bos) != target_span[0]:
        raise WorkspaceEvalContractError(
            f"{distribution}/{item['name']}: target token leaked into prefix"
        )
    return {
        "context_kind": distribution,
        "full_context_sha256": hashlib.sha256(full_text.encode()).hexdigest(),
        "full_context_input_ids": full_bos,
        "scored_prefix_input_ids": prefix_bos,
        "target_span": target_span,
        "target_token_ids": target_token_ids,
        "scored_position": scored_position,
        "predecessor_token_id": prefix_bos[-1],
        "decoded_predecessor": _decoded_token(tokenizer, prefix_bos[-1]),
    }


def _forms_for_concept(distribution: str, authored: str) -> tuple[str, ...]:
    if distribution != "order-ops":
        return (authored,)
    try:
        return ORDER_OP_SYNONYMS[authored]
    except KeyError as exc:
        raise WorkspaceEvalContractError(
            f"order-ops has unregistered intermediate {authored!r}"
        ) from exc


def _eligible_concept(
    tokenizer: Any,
    distribution: str,
    item_name: str,
    index: int,
    authored: str,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    accepted_ids: list[int] = []
    for form in _forms_for_concept(distribution, authored):
        for boundary, rendered in (("exact", form), ("leading", " " + form)):
            ids = _token_ids(tokenizer, rendered)
            if len(ids) == 1:
                token_id = ids[0]
                accepted.append(
                    {
                        "form": form,
                        "boundary": boundary,
                        "rendered": rendered,
                        "token_id": token_id,
                        "decoded": _decoded_token(tokenizer, token_id),
                    }
                )
                if token_id not in accepted_ids:
                    accepted_ids.append(token_id)
            else:
                exclusions.append(
                    {
                        "form": form,
                        "boundary": boundary,
                        "rendered": rendered,
                        "token_ids": ids,
                        "reason": "not_single_token",
                    }
                )
    concept_id = f"{item_name}:{index}"
    return {
        "concept_id": concept_id,
        "authored": authored,
        "forms": list(_forms_for_concept(distribution, authored)),
        "accepted": accepted,
        "allowed_token_ids": accepted_ids,
        "exclusions": exclusions,
        "eligible": bool(accepted_ids),
        "exclusion_reason": None if accepted_ids else "zero_single_token_forms",
    }


def _eligibility_rule() -> dict[str, Any]:
    return {
        "authored_concepts": "each intermediates entry is a distinct concept",
        "single_token_forms": ["exact", "single-leading-space"],
        "deduplication": "token_id_first_occurrence",
        "order_ops_synonym_policy": {
            key: list(forms) for key, forms in ORDER_OP_SYNONYMS.items()
        },
        "order_ops_synonym_policy_sha256": _digest(ORDER_OP_SYNONYMS),
        "item_denominator": "eligible concepts only; zero-concept items excluded",
        "base_model_competence_filter": False,
    }


def _preflight(
    tokenizer: Any,
    fixtures: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze eligibility and boundaries before loading the model or lens."""
    eligibility_distributions: dict[str, Any] = {}
    prepared: list[dict[str, Any]] = []
    for spec in FIXTURES:
        selected = fixtures.get(spec.slug)
        if not isinstance(selected, Sequence) or len(selected) != spec.publication_count:
            raise WorkspaceEvalContractError(f"{spec.slug}: selected fixture count changed")
        item_records: list[dict[str, Any]] = []
        eligible_items = 0
        eligible_concepts = 0
        for item in selected:
            boundary = _build_readout(tokenizer, spec.slug, item)
            concepts = [
                _eligible_concept(tokenizer, spec.slug, item["name"], index, authored)
                for index, authored in enumerate(item["intermediates"])
            ]
            active = [concept for concept in concepts if concept["eligible"]]
            eligible_concepts += len(active)
            if active:
                eligible_items += 1
            item_record = {
                "name": item["name"],
                "concepts": concepts,
                "eligible_concept_ids": [concept["concept_id"] for concept in active],
                "included_in_metrics": bool(active),
                "item_exclusion_reason": None if active else "zero_eligible_intermediates",
                "boundary": boundary,
            }
            item_records.append(item_record)
            prepared.append(
                {
                    "distribution": spec.slug,
                    "name": item["name"],
                    "boundary": boundary,
                    "concepts": active,
                    "included_in_metrics": bool(active),
                }
            )
        if eligible_items < spec.minimum_eligible_items:
            raise WorkspaceEvalContractError(
                f"{spec.slug}: eligible item coverage {eligible_items} below frozen "
                f"minimum {spec.minimum_eligible_items}"
            )
        if eligible_concepts < spec.minimum_eligible_concepts:
            raise WorkspaceEvalContractError(
                f"{spec.slug}: eligible concept coverage {eligible_concepts} below frozen "
                f"minimum {spec.minimum_eligible_concepts}"
            )
        eligibility_distributions[spec.slug] = {
            "selected_items": spec.publication_count,
            "eligible_items": eligible_items,
            "eligible_concepts": eligible_concepts,
            "minimum_eligible_items": spec.minimum_eligible_items,
            "minimum_eligible_concepts": spec.minimum_eligible_concepts,
            "items": item_records,
        }
    body = {
        "schema_version": 1,
        "rule": _eligibility_rule(),
        "distributions": eligibility_distributions,
    }
    manifest = {**body, "eligibility_sha256": _digest(body)}
    return manifest, prepared


def _batched_group_ranks(
    logits: Any,
    sorted_logits: Any,
    group_token_ids: Any,
    group_mask: Any,
    *,
    validated: bool = False,
):
    """Exact optimistic ranks for shared token groups across a logit batch."""
    import torch

    if (
        logits.ndim != 2
        or logits.shape[1] == 0
        or sorted_logits.shape != logits.shape
        or group_token_ids.ndim != 2
        or group_mask.shape != group_token_ids.shape
        or group_token_ids.numel() == 0
        or group_token_ids.device != logits.device
        or group_mask.device != logits.device
        or group_token_ids.dtype != torch.long
        or group_mask.dtype != torch.bool
    ):
        raise WorkspaceEvalContractError("batched rank tensors have invalid shape or IDs")
    if not validated and (
        not bool(group_mask.any(dim=1).all())
        or not bool(
            ((group_token_ids >= 0) & (group_token_ids < logits.shape[1])).all()
        )
        or not bool(torch.isfinite(sorted_logits[:, [0, -1]]).all())
    ):
        raise WorkspaceEvalContractError(
            "batched rank tensors contain invalid IDs or nonfinite logits"
        )
    gathered = logits[:, group_token_ids]
    best = gathered.masked_fill(~group_mask.unsqueeze(0), -torch.inf).amax(dim=-1)
    insertion = torch.searchsorted(
        sorted_logits.contiguous(), best.contiguous(), right=True
    )
    return 1 + logits.shape[1] - insertion




def _log_k_auc(pass_at_k: Mapping[int | str, float]) -> float:
    values = [float(pass_at_k.get(k, pass_at_k.get(str(k)))) for k in KS]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise WorkspaceEvalContractError("pass@k values must be finite probabilities")
    xs = [math.log(k) for k in KS]
    area = sum(
        (xs[index + 1] - xs[index]) * (values[index] + values[index + 1]) / 2.0
        for index in range(len(xs) - 1)
    )
    return area / (xs[-1] - xs[0])


def _item_variant_curve(
    item: Mapping[str, Any], variant: str, layers: Sequence[int]
) -> dict[int, float]:
    if not item.get("included_in_metrics"):
        raise WorkspaceEvalContractError("excluded item cannot enter a metric")
    concept_ids = item["eligible_concept_ids"]
    if not concept_ids:
        raise WorkspaceEvalContractError("metric item has no eligible concepts")
    minima: list[int] = []
    for concept_id in concept_ids:
        ranks = []
        for layer in layers:
            layer_record = item["layers"].get(str(layer))
            if not isinstance(layer_record, Mapping):
                raise WorkspaceEvalContractError(f"item is missing layer {layer}")
            rank = layer_record["concept_ranks"][variant].get(concept_id)
            if not isinstance(rank, int) or rank < 1:
                raise WorkspaceEvalContractError("concept rank is invalid")
            ranks.append(rank)
        minima.append(min(ranks))
    return {k: sum(rank <= k for rank in minima) / len(minima) for k in KS}


def _summarize_variant(
    items: Sequence[Mapping[str, Any]], variant: str, layers: Sequence[int]
) -> dict[str, Any]:
    active = [item for item in items if item.get("included_in_metrics")]
    if not active:
        raise WorkspaceEvalContractError("distribution has no eligible items")
    item_curves = [_item_variant_curve(item, variant, layers) for item in active]
    curve = {
        str(k): sum(item_curve[k] for item_curve in item_curves) / len(item_curves)
        for k in KS
    }
    return {"n_items": len(active), "pass_at_k": curve, "log_k_auc": _log_k_auc(curve)}


def _mean_distribution_summaries(
    summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(summaries) != set(FIXTURE_BY_SLUG):
        raise WorkspaceEvalContractError("aggregate requires every frozen distribution")
    curve = {
        str(k): sum(summary["pass_at_k"][str(k)] for summary in summaries.values())
        / len(summaries)
        for k in KS
    }
    return {
        "distribution_weighting": "equal",
        "pass_at_k": curve,
        "log_k_auc": _log_k_auc(curve),
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise WorkspaceEvalContractError("invalid quantile request")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise WorkspaceEvalContractError("quantile values are nonfinite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 < probability <= 1.0:
        raise WorkspaceEvalContractError("invalid nearest-rank percentile request")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise WorkspaceEvalContractError("percentile values are nonfinite")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _bootstrap_deltas(
    items_by_distribution: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    layers: Sequence[int] = CANDIDATE_LAYERS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if replicates <= 0:
        raise WorkspaceEvalContractError("bootstrap replicates must be positive")
    rng = random.Random(seed)
    per_distribution_pairs: dict[str, list[tuple[float, float]]] = {}
    for slug in FIXTURE_BY_SLUG:
        active = [item for item in items_by_distribution[slug] if item["included_in_metrics"]]
        if not active:
            raise WorkspaceEvalContractError(f"{slug}: bootstrap has no items")
        per_distribution_pairs[slug] = [
            (
                _log_k_auc(_item_variant_curve(item, "candidate", layers)),
                _log_k_auc(_item_variant_curve(item, "logit", layers)),
            )
            for item in active
        ]
    deltas: list[float] = []
    for _ in range(replicates):
        distribution_deltas = []
        for slug in FIXTURE_BY_SLUG:
            pairs = per_distribution_pairs[slug]
            sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            distribution_deltas.append(
                sum(candidate - logit for candidate, logit in sampled) / len(sampled)
            )
        deltas.append(sum(distribution_deltas) / len(distribution_deltas))
    return {
        "method": "paired_item_percentile_bootstrap_equal_distribution",
        "seed": seed,
        "replicates": replicates,
        "lower_95": _linear_quantile(deltas, 0.025),
        "median": _linear_quantile(deltas, 0.5),
        "upper_95": _linear_quantile(deltas, 0.975),
        "replicate_sha256": _digest(deltas),
    }


def _label_permutation_scores(
    items_by_distribution: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    layers: Sequence[int] = CANDIDATE_LAYERS,
    replicates: int = PERMUTATION_REPLICATES,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Within-distribution label-bundle permutation using frozen cross-ranks."""
    if replicates <= 0:
        raise WorkspaceEvalContractError("permutation replicates must be positive")
    rng = random.Random(seed)
    distributions: dict[str, tuple[list[Mapping[str, Any]], list[list[str]]]] = {}
    for slug in FIXTURE_BY_SLUG:
        active = [item for item in items_by_distribution[slug] if item["included_in_metrics"]]
        bundles = [list(item["eligible_concept_ids"]) for item in active]
        if not active or any(not bundle for bundle in bundles):
            raise WorkspaceEvalContractError(f"{slug}: invalid permutation labels")
        distributions[slug] = (active, bundles)
    scores: list[float] = []
    for _ in range(replicates):
        distribution_scores = []
        for slug in FIXTURE_BY_SLUG:
            active, bundles = distributions[slug]
            assigned = list(bundles)
            rng.shuffle(assigned)
            item_curves: list[dict[int, float]] = []
            for item, concept_ids in zip(active, assigned, strict=True):
                minima = []
                for concept_id in concept_ids:
                    ranks = [
                        item["layers"][str(layer)]["candidate_label_pool_ranks"][concept_id]
                        for layer in layers
                    ]
                    minima.append(min(ranks))
                item_curves.append(
                    {k: sum(rank <= k for rank in minima) / len(minima) for k in KS}
                )
            curve = {
                k: sum(item_curve[k] for item_curve in item_curves) / len(item_curves)
                for k in KS
            }
            distribution_scores.append(_log_k_auc(curve))
        scores.append(sum(distribution_scores) / len(distribution_scores))
    return {
        "method": "within_distribution_label_bundle_permutation_equal_distribution",
        "seed": seed,
        "replicates": replicates,
        "percentile": "nearest_rank_ceiling",
        "p99": _nearest_rank_percentile(scores, 0.99),
        "mean": sum(scores) / len(scores),
        "replicate_sha256": _digest(scores),
    }


def _motor_region_summary(
    items: Sequence[Mapping[str, Any]], layers: Sequence[int]
) -> dict[str, Any]:
    scored = list(items)
    if not scored:
        raise WorkspaceEvalContractError("motor summary has no items")
    agreement: dict[str, dict[str, float]] = {}
    for variant in ("candidate", "logit"):
        agreement[variant] = {
            str(k): sum(
                item["layers"][str(layer)]["motor"]["actual_final_rank"][variant] <= k
                for item in scored
                for layer in layers
            )
            / (len(scored) * len(layers))
            for k in KS
        }
    top1_agreement = sum(
        item["layers"][str(layer)]["motor"]["candidate_logit_top1_agreement"]
        for item in scored
        for layer in layers
    ) / (len(scored) * len(layers))
    divergence = sum(
        item["layers"][str(layer)]["motor"]["candidate_logit_js_nats"]
        for item in scored
        for layer in layers
    ) / (len(scored) * len(layers))
    return {
        "n_items": len(scored),
        "item_filter": "none",
        "next_token_agreement_at_k": agreement,
        "next_token_agreement_log_k_auc": {
            variant: _log_k_auc(curve) for variant, curve in agreement.items()
        },
        "candidate_logit_top1_agreement": top1_agreement,
        "candidate_logit_js_nats": divergence,
    }


def _build_summaries(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_distribution = {
        slug: [item for item in items if item["distribution"] == slug]
        for slug in FIXTURE_BY_SLUG
    }
    layer_sets = {"all_l0_l33": ALL_LAYERS, **REGIONS}
    distribution_summaries: dict[str, Any] = {}
    for slug, distribution_items in by_distribution.items():
        sets: dict[str, Any] = {}
        for set_name, layers in layer_sets.items():
            variants = {
                variant: _summarize_variant(distribution_items, variant, layers)
                for variant in VARIANTS
            }
            sets[set_name] = {
                "layers": list(layers),
                "variants": variants,
                "motor_metrics": _motor_region_summary(distribution_items, layers),
            }
        per_layer = {
            str(layer): {
                variant: _summarize_variant(distribution_items, variant, (layer,))
                for variant in VARIANTS
            }
            for layer in ALL_LAYERS
        }
        distribution_summaries[slug] = {
            "layer_sets": sets,
            "per_layer": per_layer,
        }
    aggregate_sets: dict[str, Any] = {}
    for set_name, layers in layer_sets.items():
        aggregate_variants: dict[str, Any] = {}
        for variant in VARIANTS:
            aggregate_variants[variant] = _mean_distribution_summaries(
                {
                    slug: distribution_summaries[slug]["layer_sets"][set_name][
                        "variants"
                    ][variant]
                    for slug in FIXTURE_BY_SLUG
                }
            )
        distribution_motor = {
            slug: distribution_summaries[slug]["layer_sets"][set_name]["motor_metrics"]
            for slug in FIXTURE_BY_SLUG
        }
        aggregate_agreement = {
            variant: {
                str(k): sum(
                    summary["next_token_agreement_at_k"][variant][str(k)]
                    for summary in distribution_motor.values()
                )
                / len(distribution_motor)
                for k in KS
            }
            for variant in ("candidate", "logit")
        }
        aggregate_motor = {
            "distribution_weighting": "equal",
            "item_filter": "none",
            "next_token_agreement_at_k": aggregate_agreement,
            "next_token_agreement_log_k_auc": {
                variant: _log_k_auc(curve)
                for variant, curve in aggregate_agreement.items()
            },
            "candidate_logit_top1_agreement": sum(
                summary["candidate_logit_top1_agreement"]
                for summary in distribution_motor.values()
            )
            / len(distribution_motor),
            "candidate_logit_js_nats": sum(
                summary["candidate_logit_js_nats"]
                for summary in distribution_motor.values()
            )
            / len(distribution_motor),
        }
        aggregate_sets[set_name] = {
            "layers": list(layers),
            "variants": aggregate_variants,
            "motor_metrics": aggregate_motor,
        }
    return {
        "distributions": distribution_summaries,
        "equal_distribution_aggregate": {"layer_sets": aggregate_sets},
    }


def _adjudication_evidence(
    summaries: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
) -> dict[str, Any]:
    distributions = summaries["distributions"]
    aggregate = summaries["equal_distribution_aggregate"]["layer_sets"]
    band = aggregate["candidate_l13_l31"]
    region_auc = {
        region: aggregate[region]["variants"]["candidate"]["log_k_auc"]
        for region in REGIONS
    }
    return {
        "bootstrap_candidate_minus_logit_auc": dict(bootstrap),
        "distribution_candidate_minus_logit_auc": {
            slug: distributions[slug]["layer_sets"]["candidate_l13_l31"][
                "variants"
            ]["candidate"]["log_k_auc"]
            - distributions[slug]["layer_sets"]["candidate_l13_l31"]["variants"][
                "logit"
            ]["log_k_auc"]
            for slug in FIXTURE_BY_SLUG
        },
        "aggregate_band_auc": {
            variant: band["variants"][variant]["log_k_auc"] for variant in VARIANTS
        },
        "label_permutation": {
            **dict(permutation),
            "observed_candidate_auc": band["variants"]["candidate"]["log_k_auc"],
        },
        "candidate_intermediate_auc_by_region": region_auc,
        "candidate_next_token_agreement_at_k_by_region": {
            region: aggregate[region]["motor_metrics"]["next_token_agreement_at_k"][
                "candidate"
            ]
            for region in REGIONS
        },
        "candidate_next_token_agreement_log_k_auc_by_region": {
            region: aggregate[region]["motor_metrics"][
                "next_token_agreement_log_k_auc"
            ]["candidate"]
            for region in REGIONS
        },
        "candidate_logit_js_nats_by_region": {
            region: aggregate[region]["motor_metrics"]["candidate_logit_js_nats"]
            for region in REGIONS
        },
    }


def _adjudicate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Apply only the frozen L13--L31 preregistered rules."""
    aggregate_auc = evidence["aggregate_band_auc"]
    region_auc = evidence["candidate_intermediate_auc_by_region"]
    agreement_auc = evidence["candidate_next_token_agreement_log_k_auc_by_region"]
    divergence = evidence["candidate_logit_js_nats_by_region"]
    permutation = evidence["label_permutation"]
    checks = {
        "bootstrap_lower_above_zero": evidence[
            "bootstrap_candidate_minus_logit_auc"
        ]["lower_95"]
        > 0.0,
        "nonnegative_every_distribution": all(
            delta >= 0.0
            for delta in evidence["distribution_candidate_minus_logit_auc"].values()
        ),
        "beats_transposed_control": aggregate_auc["candidate"]
        > aggregate_auc["transposed"],
        "beats_permuted_control": aggregate_auc["candidate"]
        > aggregate_auc["permuted"],
        "exceeds_label_permutation_p99": permutation["observed_candidate_auc"]
        > permutation["p99"],
        "band_exceeds_early_intermediate_auc": region_auc["candidate_l13_l31"]
        > region_auc["early_l0_l12"],
        "band_exceeds_motor_intermediate_auc": region_auc["candidate_l13_l31"]
        > region_auc["motor_l32_l33"],
        "motor_next_token_agreement_auc_exceeds_band": agreement_auc[
            "motor_l32_l33"
        ]
        > agreement_auc["candidate_l13_l31"],
        "motor_j_logit_divergence_below_band": divergence["motor_l32_l33"]
        < divergence["candidate_l13_l31"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "hypothesis": "fixed_l13_l31_workspace_like_transfer",
        "searched_alternate_bands": False,
        "criteria": checks,
        "failed_criteria": failed,
        "status": "validated" if not failed else "no_band",
        "evidence": dict(evidence),
    }


def _metric_config() -> dict[str, Any]:
    return {
        "rank": {
            "definition": "1_plus_count_strictly_greater_than_best_allowed_target",
            "ties": "optimistic",
            "nonfinite": "reject",
        },
        "reducer_order": [
            "minimum_allowed_token_rank_at_layer",
            "minimum_over_declared_layers",
            "fraction_intermediates_within_item",
            "equal_mean_over_items",
            "equal_mean_over_distributions",
        ],
        "ks": list(KS),
        "auc": "normalized_trapezoid_against_natural_log_k",
        "all_source_layers": list(ALL_LAYERS),
        "regions": {name: list(layers) for name, layers in REGIONS.items()},
        "variants": list(VARIANTS),
        "transposed_control": "residual_row_vector_matmul_J",
        "permuted_control": {
            "source_layer": "rotation_by_17_mod_34",
            "output_basis": "torch_cpu_randperm_rows",
            "seed": CONTROL_SEED,
        },
        "label_permutation": {
            "method": "within_distribution_label_bundle",
            "replicates": PERMUTATION_REPLICATES,
            "seed": PERMUTATION_SEED,
            "p99": "nearest_rank_ceiling",
        },
        "bootstrap": {
            "method": "paired_item_percentile_equal_distribution",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": [0.025, 0.975],
            "quantile": "linear_type7",
        },
        "motor": {
            "reference": "unmodified_model_actual_final_top1_token",
            "item_filter": "none",
            "agreement_ks": list(KS),
            "agreement_summary": "normalized_log_k_auc",
            "symmetric_divergence": "jensen_shannon_nats",
        },
        "batch_size": BATCH_SIZE,
        "adjudication": {
            "layers_are_frozen": True,
            "alternate_band_search": False,
            "motor_agreement_rule": "motor_log_k_auc_strictly_above_band",
            "strict_control_comparisons": True,
            "no_band_is_complete": True,
        },
    }


def _runtime_identity() -> dict[str, Any]:
    import importlib.metadata

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
        "packages": {package: importlib.metadata.version(package) for package in packages},
        "python": os.sys.version,
        "cuda": torch.version.cuda,
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "device": torch.cuda.get_device_name(0),
        "modal_environment": {
            key: os.environ[key] for key in ("MODAL_IMAGE_ID",) if key in os.environ
        },
    }


def _bind_fit_identity(
    manifest: Mapping[str, Any],
    fit_manifest_path: str | pathlib.Path,
    lens_path: str | pathlib.Path,
    volume_root: str | pathlib.Path,
) -> dict[str, Any]:
    root = pathlib.Path(volume_root).resolve()
    fit_manifest_path = pathlib.Path(fit_manifest_path).resolve()
    lens_path = pathlib.Path(lens_path).resolve()
    config = manifest.get("config")
    lens = manifest.get("lens")
    paths = manifest.get("paths")
    if (
        manifest.get("kind") != "canonical_text_jlens_fit"
        or manifest.get("status") != "complete"
        or manifest.get("canonical") is not True
        or not isinstance(config, Mapping)
        or not isinstance(lens, Mapping)
        or not isinstance(paths, Mapping)
    ):
        raise WorkspaceEvalContractError("fit manifest is not a completed canonical fit")
    expected_manifest = (root / paths["manifest"]).resolve()
    expected_lens = (root / paths["lens"]).resolve()
    if fit_manifest_path != expected_manifest:
        raise WorkspaceEvalContractError("supplied fit manifest path is not its bound path")
    if lens_path != expected_lens:
        raise WorkspaceEvalContractError("supplied lens path is not the manifest-bound lens")
    if config.get("source", {}).get("digest") != FIT_SOURCE_DIGEST:
        raise WorkspaceEvalContractError(
            "fit manifest source digest does not match bundled canonical fitter"
        )
    if not lens_path.is_file():
        raise WorkspaceEvalContractError("manifest-bound lens does not exist")
    if lens_path.stat().st_size != lens.get("bytes") or _sha256_file(lens_path) != lens.get(
        "sha256"
    ):
        raise WorkspaceEvalContractError("supplied lens bytes do not match fit manifest")
    expected_model = {"id": MODEL_ID, "revision": MODEL_REVISION}
    expected_geometry = {
        "source_layers": list(ALL_LAYERS),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
        "d_model": D_MODEL,
        "model_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "artifact_dtype": "float16",
        "attention_backend": "eager",
    }
    fit = config.get("fit", {})
    if config.get("model") != expected_model or config.get("tokenizer") != expected_model:
        raise WorkspaceEvalContractError("fit model/tokenizer identity is not frozen")
    if config.get("jlens") != {"revision": JLENS_REVISION}:
        raise WorkspaceEvalContractError("fit JLens revision is not frozen")
    if config.get("prompt_policy") != {
        "kind": "raw_text",
        "force_bos": True,
        "chat_template": False,
    }:
        raise WorkspaceEvalContractError("fit raw-text/BOS policy is not frozen")
    if any(fit.get(key) != value for key, value in expected_geometry.items()):
        raise WorkspaceEvalContractError("fit estimator geometry is not frozen")
    if config.get("dataset", {}).get("requested_count") != 1_000:
        raise WorkspaceEvalContractError("workspace evaluation requires the 1000-prompt fit")
    if (
        lens.get("dtype") != "float16"
        or lens.get("n_prompts") != 1_000
        or lens.get("d_model") != D_MODEL
        or lens.get("shape") != [D_MODEL, D_MODEL]
        or lens.get("source_layers") != list(ALL_LAYERS)
        or lens.get("target_layer") != 34
        or lens.get("skip_first") != 16
        or lens.get("max_seq_len") != 128
        or lens.get("dim_batch") != 128
    ):
        raise WorkspaceEvalContractError("fit lens metadata is not frozen")
    return {
        "fit_config": config,
        "fit_manifest_relative_path": paths["manifest"],
        "fit_config_sha256": manifest["fit_config_sha256"],
        "lens_relative_path": lens["relative_path"],
        "lens_sha256": lens["sha256"],
        "lens_bytes": lens["bytes"],
        "lens_dtype": lens["dtype"],
        "lens_n_prompts": lens["n_prompts"],
        "lens_d_model": lens["d_model"],
        "lens_shape": lens["shape"],
        "source_layers": lens["source_layers"],
        "target_layer": lens["target_layer"],
        "skip_first": lens["skip_first"],
        "max_seq_len": lens["max_seq_len"],
        "dim_batch": lens["dim_batch"],
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "dataset": config["dataset"],
        "jlens": config["jlens"],
        "prompt_policy": config["prompt_policy"],
        "fit_geometry": config["fit"],
        "fit_lock": config["lock"],
        "fit_source": config["source"],
        "fit_runtime": config["runtime"],
    }


def _control_permutations(d_model: int = D_MODEL) -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(CONTROL_SEED)
    output = torch.randperm(d_model, generator=generator).tolist()
    source = {str(layer): (layer + 17) % len(ALL_LAYERS) for layer in ALL_LAYERS}
    return {
        "source_layers": source,
        "output_basis": output,
        "output_basis_sha256": _digest(output),
        "seed": CONTROL_SEED,
    }


def _transport_variants(
    residual: Any,
    layer: int,
    jacobians: Mapping[int, Any],
    permuted_jacobians: Mapping[int, Any],
):
    jacobian = jacobians[layer]
    return {
        "candidate": residual @ jacobian.T,
        "logit": residual,
        "transposed": residual @ jacobian,
        "permuted": residual @ permuted_jacobians[layer].T,
    }


def _js_divergence_nats(left_logits: Any, right_logits: Any):
    import torch

    left = torch.log_softmax(left_logits.float(), dim=-1)
    right = torch.log_softmax(right_logits.float(), dim=-1)
    mixture = torch.logaddexp(left, right) - math.log(2.0)
    result = 0.5 * (
        (left.exp() * (left - mixture)).sum(dim=-1)
        + (right.exp() * (right - mixture)).sum(dim=-1)
    )
    if not bool(torch.isfinite(result).all()):
        raise WorkspaceEvalContractError("J/logit divergence is nonfinite")
    return result.clamp(min=0.0, max=JS_DIVERGENCE_MAX_NATS)


def _score_items(
    model: Any,
    hf: Any,
    lens: Any,
    prepared: list[dict[str, Any]],
    controls: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import torch
    from jlens.hooks import ActivationRecorder

    jacobians = {
        layer: lens.jacobians[layer].to(device="cuda", dtype=torch.float32)
        for layer in ALL_LAYERS
    }
    output_permutation = torch.tensor(
        controls["output_basis"], dtype=torch.long, device="cuda"
    )
    permuted_jacobians = {
        layer: jacobians[int(controls["source_layers"][str(layer)])]
        .index_select(0, output_permutation)
        .contiguous()
        for layer in ALL_LAYERS
    }
    distribution_pools: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        distribution_pools.setdefault(item["distribution"], []).extend(item["concepts"])
    pool_runtime: dict[str, dict[str, Any]] = {}
    for slug, concepts in distribution_pools.items():
        max_forms = max(len(concept["allowed_token_ids"]) for concept in concepts)
        group_ids = torch.zeros(
            (len(concepts), max_forms), dtype=torch.long, device="cuda"
        )
        group_mask = torch.zeros_like(group_ids, dtype=torch.bool)
        for group, concept in enumerate(concepts):
            token_ids = concept["allowed_token_ids"]
            group_ids[group, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long, device="cuda"
            )
            group_mask[group, : len(token_ids)] = True
        concept_ids = [concept["concept_id"] for concept in concepts]
        pool_runtime[slug] = {
            "concept_ids": concept_ids,
            "index": {
                concept_id: index for index, concept_id in enumerate(concept_ids)
            },
            "group_ids": group_ids,
            "group_mask": group_mask,
        }

    scored: list[dict[str, Any]] = []
    for item in prepared:
        scored.append(
            {
                "distribution": item["distribution"],
                "name": item["name"],
                "included_in_metrics": item["included_in_metrics"],
                "eligible_concept_ids": [
                    concept["concept_id"] for concept in item["concepts"]
                ],
                "boundary": item["boundary"],
                "base_model_target_competence": None,
                "actual_final_top1_id": None,
                "actual_final_top1_token": None,
                "layers": {},
            }
        )
    pad_id = getattr(model.tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, int) or pad_id < 0:
        pad_id = getattr(model.tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, int) or pad_id < 0:
        raise WorkspaceEvalContractError("tokenizer has no pad/eos token")
    for batch_start in range(0, len(prepared), BATCH_SIZE):
        batch_items = prepared[batch_start : batch_start + BATCH_SIZE]
        batch_records = scored[batch_start : batch_start + BATCH_SIZE]
        lengths = [len(item["boundary"]["scored_prefix_input_ids"]) for item in batch_items]
        max_length = max(lengths)
        input_ids = torch.full(
            (len(batch_items), max_length), pad_id, dtype=torch.long, device="cuda"
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, item in enumerate(batch_items):
            ids = torch.tensor(
                item["boundary"]["scored_prefix_input_ids"],
                dtype=torch.long,
                device="cuda",
            )
            input_ids[row, : ids.numel()] = ids
            attention_mask[row, : ids.numel()] = 1
        positions = torch.tensor([length - 1 for length in lengths], device="cuda")
        rows = torch.arange(len(batch_items), device="cuda")
        with torch.no_grad(), ActivationRecorder(
            model.layers, at=[*ALL_LAYERS, 34]
        ) as recorder:
            hf.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        final_residual = recorder.activations[34][rows, positions].float()
        final_logits = model.unembed(final_residual).float()
        if not bool(torch.isfinite(final_logits).all()):
            raise WorkspaceEvalContractError("final model logits contain nonfinite values")
        actual_ids = final_logits.argmax(dim=-1)
        actual_values = [int(token) for token in actual_ids.to("cpu").tolist()]
        final_targets = torch.tensor(
            [
                (
                    item["boundary"]["target_token_ids"][0]
                    if item["boundary"]["target_token_ids"]
                    else actual_values[row]
                )
                for row, item in enumerate(batch_items)
            ],
            dtype=torch.long,
            device="cuda",
        ).unsqueeze(1)
        final_target_mask = torch.ones_like(final_targets, dtype=torch.bool)
        final_rank_matrix = _batched_group_ranks(
            final_logits,
            final_logits.sort(dim=-1).values,
            final_targets,
            final_target_mask,
            validated=True,
        )
        final_target_ranks = [
            int(rank)
            for rank in final_rank_matrix.diagonal().to("cpu").tolist()
        ]
        for row, record in enumerate(batch_records):
            actual = actual_values[row]
            record["actual_final_top1_id"] = actual
            record["actual_final_top1_token"] = _decoded_token(model.tokenizer, actual)
            target_ids = record["boundary"]["target_token_ids"]
            if target_ids:
                record["base_model_target_competence"] = {
                    "first_target_token_id": int(target_ids[0]),
                    "rank": final_target_ranks[row],
                    "top1_match": actual == int(target_ids[0]),
                    "used_as_filter": False,
                }
        batch_rows_by_distribution: dict[str, list[int]] = {}
        for row, item in enumerate(batch_items):
            batch_rows_by_distribution.setdefault(item["distribution"], []).append(row)
        for layer in ALL_LAYERS:
            residual = recorder.activations[layer][rows, positions].float()
            transports = _transport_variants(
                residual, layer, jacobians, permuted_jacobians
            )
            logits_by_variant = {
                variant: model.unembed(transports[variant]).float() for variant in VARIANTS
            }
            if any(
                not bool(torch.isfinite(logits).all())
                for logits in logits_by_variant.values()
            ):
                raise WorkspaceEvalContractError(
                    "candidate/control logits contain nonfinite values"
                )
            candidate_logits = logits_by_variant["candidate"]
            logit_logits = logits_by_variant["logit"]
            divergence_values = [
                float(value)
                for value in _js_divergence_nats(
                    candidate_logits, logit_logits
                ).to("cpu").tolist()
            ]
            top1 = {
                variant: logits_by_variant[variant].argmax(dim=-1).to("cpu").tolist()
                for variant in ("candidate", "logit")
            }
            sorted_by_variant = {
                variant: logits.sort(dim=-1).values
                for variant, logits in logits_by_variant.items()
            }
            rank_tables: dict[str, list[list[int] | None]] = {
                variant: [None] * len(batch_items) for variant in VARIANTS
            }
            for slug, row_numbers in batch_rows_by_distribution.items():
                selected_rows = torch.tensor(
                    row_numbers, dtype=torch.long, device="cuda"
                )
                pool = pool_runtime[slug]
                for variant in VARIANTS:
                    ranks = _batched_group_ranks(
                        logits_by_variant[variant].index_select(0, selected_rows),
                        sorted_by_variant[variant].index_select(0, selected_rows),
                        pool["group_ids"],
                        pool["group_mask"],
                        validated=True,
                    ).to("cpu").tolist()
                    for local_row, batch_row in enumerate(row_numbers):
                        rank_tables[variant][batch_row] = [
                            int(rank) for rank in ranks[local_row]
                        ]
            actual_group_ids = actual_ids.unsqueeze(1)
            actual_group_mask = torch.ones_like(actual_group_ids, dtype=torch.bool)
            motor_ranks = {
                variant: [
                    int(rank)
                    for rank in _batched_group_ranks(
                        logits_by_variant[variant],
                        sorted_by_variant[variant],
                        actual_group_ids,
                        actual_group_mask,
                        validated=True,
                    )
                    .diagonal()
                    .to("cpu")
                    .tolist()
                ]
                for variant in ("candidate", "logit")
            }
            for row, (item, record) in enumerate(
                zip(batch_items, batch_records, strict=True)
            ):
                pool = pool_runtime[item["distribution"]]
                own_ids = [concept["concept_id"] for concept in item["concepts"]]
                own_ranks = {
                    variant: {
                        concept_id: rank_tables[variant][row][
                            pool["index"][concept_id]
                        ]
                        for concept_id in own_ids
                    }
                    for variant in VARIANTS
                }
                candidate_pool_values = rank_tables["candidate"][row]
                pool_ranks = dict(
                    zip(pool["concept_ids"], candidate_pool_values, strict=True)
                )
                record["layers"][str(layer)] = {
                    "concept_ranks": own_ranks,
                    "candidate_label_pool_ranks": pool_ranks,
                    "motor": {
                        "actual_final_rank": {
                            variant: motor_ranks[variant][row]
                            for variant in ("candidate", "logit")
                        },
                        "candidate_logit_top1_agreement": top1["candidate"][row]
                        == top1["logit"][row],
                        "candidate_logit_js_nats": divergence_values[row],
                    },
                }
            del sorted_by_variant, logits_by_variant, transports
        del recorder, final_logits, final_residual
    return scored


def _evaluation_config(
    fit_identity: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    controls: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": WORKSPACE_REPORT_SCHEMA_VERSION,
        "fit": dict(fit_identity),
        "fixtures": [_fixture_identity(spec) for spec in FIXTURES],
        "eligibility_sha256": eligibility["eligibility_sha256"],
        "tokenizer_preflight": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "add_special_tokens": False,
            "force_bos_for_model_input": True,
            "offsets_from_complete_prompt_plus_target": True,
            "target_tokens_excluded_from_scored_prefix": True,
        },
        "controls": {
            "source_layers": controls["source_layers"],
            "output_basis_sha256": controls["output_basis_sha256"],
            "seed": controls["seed"],
        },
        "metrics": _metric_config(),
        "source": {"git_revision": GIT_REVISION, "digest": SOURCE_DIGEST},
        "runtime": dict(runtime),
    }


def _complete_report(report: dict[str, Any]) -> dict[str, Any]:
    body = dict(report)
    body.pop("workspace_report_sha256", None)
    return {**body, "workspace_report_sha256": _digest(body)}


def _require_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkspaceEvalContractError(f"{path} contains nonfinite value")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value: Any, prefix: str) -> bool:
    if not isinstance(value, str):
        return False
    path = pathlib.PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and len(path.parts) >= 2
        and path.parts[0] == prefix
    )


def _validate_runtime_identity(runtime: Any, *, fit_side: bool) -> None:
    expected_keys = (
        {
            "packages",
            "python",
            "cuda",
            "device",
            "torch_cuda_alloc_conf",
            "modal_image_id",
            "modal_function_timeout_seconds",
        }
        if fit_side
        else {
            "packages",
            "python",
            "cuda",
            "torch_cuda_alloc_conf",
            "device",
            "modal_environment",
        }
    )
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != expected_keys
        or runtime["packages"] != EXPECTED_RUNTIME_PACKAGES
        or not isinstance(runtime["python"], str)
        or not runtime["python"].startswith("3.12")
        or not isinstance(runtime["cuda"], str)
        or not runtime["cuda"]
        or not isinstance(runtime["device"], str)
        or "H100" not in runtime["device"]
        or runtime["torch_cuda_alloc_conf"] != "expandable_segments:True"
    ):
        raise WorkspaceEvalContractError(
            f"workspace {'fit' if fit_side else 'scorer'} runtime identity changed"
        )
    if fit_side:
        if runtime["modal_function_timeout_seconds"] != 86_400:
            raise WorkspaceEvalContractError(
                "workspace fit Modal timeout identity changed"
            )
        image_id = runtime["modal_image_id"]
        if image_id is not None and (
            not isinstance(image_id, str) or not image_id
        ):
            raise WorkspaceEvalContractError("workspace fit Modal image identity changed")
    else:
        environment = runtime["modal_environment"]
        if (
            not isinstance(environment, Mapping)
            or not set(environment) <= {"MODAL_IMAGE_ID"}
            or any(not isinstance(value, str) or not value for value in environment.values())
        ):
            raise WorkspaceEvalContractError(
                "workspace scorer Modal image identity changed"
            )


def _validate_report_fit_identity(fit: Any) -> None:
    required = {
        "fit_config",
        "fit_manifest_relative_path",
        "fit_config_sha256",
        "lens_relative_path",
        "lens_sha256",
        "lens_bytes",
        "lens_dtype",
        "lens_n_prompts",
        "lens_d_model",
        "lens_shape",
        "source_layers",
        "target_layer",
        "skip_first",
        "max_seq_len",
        "dim_batch",
        "model",
        "tokenizer",
        "dataset",
        "jlens",
        "prompt_policy",
        "fit_geometry",
        "fit_lock",
        "fit_source",
        "fit_runtime",
    }
    if not isinstance(fit, Mapping) or set(fit) != required:
        raise WorkspaceEvalContractError("workspace fit identity schema is incomplete")
    frozen_model = {"id": MODEL_ID, "revision": MODEL_REVISION}
    if fit["model"] != frozen_model or fit["tokenizer"] != frozen_model:
        raise WorkspaceEvalContractError("workspace fit model/tokenizer identity changed")
    if fit["jlens"] != {"revision": JLENS_REVISION}:
        raise WorkspaceEvalContractError("workspace fit JLens identity changed")
    if fit["prompt_policy"] != {
        "kind": "raw_text",
        "force_bos": True,
        "chat_template": False,
    }:
        raise WorkspaceEvalContractError("workspace fit raw-text/BOS policy changed")
    dataset = fit["dataset"]
    expected_dataset_keys = {
        "id",
        "config",
        "split",
        "text_field",
        "revision",
        "streaming",
        "trust_remote_code",
        "chunking",
        "requested_count",
        "ordered_prompt_sha256",
    }
    if (
        not isinstance(dataset, Mapping)
        or set(dataset) != expected_dataset_keys
        or dataset["id"] != "Salesforce/wikitext"
        or dataset["config"] != "wikitext-103-raw-v1"
        or dataset["split"] != "train"
        or dataset["text_field"] != "text"
        or dataset["revision"] != "b08601e04326c79dfdd32d625aee71d232d685c3"
        or dataset["streaming"] is not True
        or dataset["trust_remote_code"] is not False
        or dataset["chunking"]
        != {
            "algorithm": "neuronpedia_concat_space_strip_emit_strict_gt_v1",
            "max_chars": 2_000,
            "min_tail_chars": 200,
        }
        or dataset["requested_count"] != 1_000
        or not _is_lower_hex(dataset["ordered_prompt_sha256"], 64)
    ):
        raise WorkspaceEvalContractError("workspace fit corpus identity changed")
    expected_geometry = {
        "source_layers": list(ALL_LAYERS),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
        "checkpoint_every": 5,
        "resume": True,
        "compile": False,
        "d_model": D_MODEL,
        "model_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "artifact_dtype": "float16",
        "attention_backend": "eager",
    }
    if fit["fit_geometry"] != expected_geometry:
        raise WorkspaceEvalContractError("workspace fit estimator geometry changed")
    fit_lock = fit["fit_lock"]
    if (
        not isinstance(fit_lock, Mapping)
        or set(fit_lock) != {"uv_lock_sha256", "frozen", "dependency_group"}
        or not _is_lower_hex(fit_lock["uv_lock_sha256"], 64)
        or fit_lock["frozen"] is not True
        or fit_lock["dependency_group"] != "fit"
    ):
        raise WorkspaceEvalContractError("workspace fit lock identity changed")
    fit_source = fit["fit_source"]
    if (
        not isinstance(fit_source, Mapping)
        or set(fit_source) != {"git_revision", "digest"}
        or not _is_lower_hex(fit_source["git_revision"], 40)
        or not _is_lower_hex(fit_source["digest"], 64)
        or not isinstance(fit["fit_runtime"], Mapping)
    ):
        raise WorkspaceEvalContractError("workspace fit source/runtime identity changed")
    _validate_runtime_identity(fit["fit_runtime"], fit_side=True)
    fit_config = fit["fit_config"]
    expected_fit_config = {
        "schema_version": 1,
        "model": fit["model"],
        "tokenizer": fit["tokenizer"],
        "dataset": fit["dataset"],
        "jlens": fit["jlens"],
        "prompt_policy": fit["prompt_policy"],
        "fit": fit["fit_geometry"],
        "lock": fit["fit_lock"],
        "source": fit["fit_source"],
        "runtime": fit["fit_runtime"],
    }
    if (
        fit_config != expected_fit_config
        or _digest(fit_config) != fit["fit_config_sha256"]
    ):
        raise WorkspaceEvalContractError(
            "workspace fit config digest does not match embedded config"
        )
    fit_tag = f"gemma-4-E2B-it-jlens-{fit['fit_config_sha256']}"
    if (
        fit["fit_manifest_relative_path"] != f"runs/{fit_tag}.json"
        or fit["lens_relative_path"] != f"lenses/{fit_tag}.pt"
    ):
        raise WorkspaceEvalContractError(
            "workspace fit config digest does not match bound paths"
        )
    if (
        not _safe_relative_path(fit["fit_manifest_relative_path"], "runs")
        or not _safe_relative_path(fit["lens_relative_path"], "lenses")
        or not _is_lower_hex(fit["fit_config_sha256"], 64)
        or not _is_lower_hex(fit["lens_sha256"], 64)
        or not isinstance(fit["lens_bytes"], int)
        or isinstance(fit["lens_bytes"], bool)
        or fit["lens_bytes"] <= 0
        or fit["lens_dtype"] != "float16"
        or fit["lens_n_prompts"] != 1_000
        or fit["lens_d_model"] != D_MODEL
        or fit["lens_shape"] != [D_MODEL, D_MODEL]
        or fit["source_layers"] != list(ALL_LAYERS)
        or fit["target_layer"] != 34
        or fit["skip_first"] != 16
        or fit["max_seq_len"] != 128
        or fit["dim_batch"] != 128
    ):
        raise WorkspaceEvalContractError("workspace fit/lens artifact identity changed")


def _validate_boundary_record(
    boundary: Any,
    *,
    expected_context_kind: str,
    target_boundary: bool,
) -> None:
    required = {
        "context_kind",
        "full_context_sha256",
        "full_context_input_ids",
        "scored_prefix_input_ids",
        "target_span",
        "target_token_ids",
        "scored_position",
        "predecessor_token_id",
        "decoded_predecessor",
    }
    if not isinstance(boundary, Mapping) or set(boundary) != required:
        raise WorkspaceEvalContractError("workspace boundary schema is invalid")
    if boundary["context_kind"] != expected_context_kind:
        raise WorkspaceEvalContractError("workspace boundary context kind changed")
    full = boundary["full_context_input_ids"]
    prefix = boundary["scored_prefix_input_ids"]
    if (
        not _is_lower_hex(boundary["full_context_sha256"], 64)
        or not isinstance(full, list)
        or not isinstance(prefix, list)
        or not full
        or not prefix
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in full)
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in prefix)
        or boundary["scored_position"] != len(prefix) - 1
        or boundary["predecessor_token_id"] != prefix[-1]
        or not isinstance(boundary["decoded_predecessor"], str)
    ):
        raise WorkspaceEvalContractError("workspace boundary token identity is invalid")
    span = boundary["target_span"]
    target_ids = boundary["target_token_ids"]
    if target_boundary:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(index, int) for index in span)
            or span[0] != len(prefix)
            or not span[0] < span[1] == len(full)
            or not isinstance(target_ids, list)
            or not target_ids
            or target_ids != full[span[0] : span[1]]
            or prefix != full[: span[0]]
        ):
            raise WorkspaceEvalContractError("workspace target boundary leaked or changed")
    elif (
        span is not None
        or target_ids != []
        or prefix != full[: len(prefix)]
    ):
        raise WorkspaceEvalContractError(
            "workspace nontarget boundary is not a full-context prefix"
        )


def _validate_eligibility_manifest(eligibility: Any) -> dict[str, list[str]]:
    if not isinstance(eligibility, Mapping):
        raise WorkspaceEvalContractError("workspace eligibility is missing")
    body = dict(eligibility)
    claimed = body.pop("eligibility_sha256", None)
    if claimed != _digest(body):
        raise WorkspaceEvalContractError("workspace eligibility content digest mismatch")
    if (
        set(body) != {"schema_version", "rule", "distributions"}
        or body["schema_version"] != 1
        or body["rule"] != _eligibility_rule()
        or not isinstance(body["distributions"], Mapping)
        or set(body["distributions"]) != set(FIXTURE_BY_SLUG)
    ):
        raise WorkspaceEvalContractError("workspace eligibility schema/rule changed")
    pools: dict[str, list[str]] = {}
    for spec in FIXTURES:
        distribution = body["distributions"][spec.slug]
        if (
            not isinstance(distribution, Mapping)
            or set(distribution)
            != {
                "selected_items",
                "eligible_items",
                "eligible_concepts",
                "minimum_eligible_items",
                "minimum_eligible_concepts",
                "items",
            }
            or distribution["selected_items"] != spec.publication_count
            or distribution["minimum_eligible_items"] != spec.minimum_eligible_items
            or distribution["minimum_eligible_concepts"]
            != spec.minimum_eligible_concepts
            or not isinstance(distribution["items"], list)
            or len(distribution["items"]) != spec.publication_count
        ):
            raise WorkspaceEvalContractError(
                f"workspace {spec.slug} eligibility coverage schema changed"
            )
        names: list[str] = []
        pool: list[str] = []
        counted_items = 0
        counted_concepts = 0
        for item in distribution["items"]:
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {
                    "name",
                    "concepts",
                    "eligible_concept_ids",
                    "included_in_metrics",
                    "item_exclusion_reason",
                    "boundary",
                }
                or not isinstance(item["name"], str)
                or not isinstance(item["concepts"], list)
                or not isinstance(item["eligible_concept_ids"], list)
            ):
                raise WorkspaceEvalContractError(
                    f"workspace {spec.slug} eligibility item schema changed"
                )
            names.append(item["name"])
            eligible_ids: list[str] = []
            for index, concept in enumerate(item["concepts"]):
                expected_concept_id = f"{item['name']}:{index}"
                if (
                    not isinstance(concept, Mapping)
                    or set(concept)
                    != {
                        "concept_id",
                        "authored",
                        "forms",
                        "accepted",
                        "allowed_token_ids",
                        "exclusions",
                        "eligible",
                        "exclusion_reason",
                    }
                    or concept["concept_id"] != expected_concept_id
                    or not isinstance(concept["authored"], str)
                    or concept["forms"]
                    != list(_forms_for_concept(spec.slug, concept["authored"]))
                    or not isinstance(concept["accepted"], list)
                    or not isinstance(concept["exclusions"], list)
                    or not isinstance(concept["allowed_token_ids"], list)
                    or len(concept["allowed_token_ids"])
                    != len(set(concept["allowed_token_ids"]))
                    or not all(
                        isinstance(token, int) and not isinstance(token, bool) and token >= 0
                        for token in concept["allowed_token_ids"]
                    )
                    or concept["eligible"] is not bool(concept["allowed_token_ids"])
                    or concept["exclusion_reason"]
                    != (
                        None
                        if concept["allowed_token_ids"]
                        else "zero_single_token_forms"
                    )
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']} concept identity changed"
                    )
                attempts = {
                    (entry.get("form"), entry.get("boundary"))
                    for entry in [*concept["accepted"], *concept["exclusions"]]
                    if isinstance(entry, Mapping)
                }
                expected_attempts = {
                    (form, boundary)
                    for form in concept["forms"]
                    for boundary in ("exact", "leading")
                }
                accepted_ids = [
                    entry.get("token_id")
                    for entry in concept["accepted"]
                    if isinstance(entry, Mapping)
                ]
                if (
                    attempts != expected_attempts
                    or set(accepted_ids) != set(concept["allowed_token_ids"])
                    or any(
                        not isinstance(entry, Mapping)
                        or set(entry)
                        != {
                            "form",
                            "boundary",
                            "rendered",
                            "token_id",
                            "decoded",
                        }
                        for entry in concept["accepted"]
                    )
                    or any(
                        not isinstance(entry, Mapping)
                        or set(entry)
                        != {
                            "form",
                            "boundary",
                            "rendered",
                            "token_ids",
                            "reason",
                        }
                        or entry["reason"] != "not_single_token"
                        for entry in concept["exclusions"]
                    )
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']} token eligibility changed"
                    )
                if concept["eligible"]:
                    eligible_ids.append(expected_concept_id)
            included = bool(eligible_ids)
            if (
                item["eligible_concept_ids"] != eligible_ids
                or item["included_in_metrics"] is not included
                or item["item_exclusion_reason"]
                != (None if included else "zero_eligible_intermediates")
            ):
                raise WorkspaceEvalContractError(
                    f"workspace {spec.slug}/{item['name']} denominator changed"
                )
            _validate_boundary_record(
                item["boundary"],
                expected_context_kind=spec.slug,
                target_boundary=spec.target_boundary,
            )
            counted_items += int(included)
            counted_concepts += len(eligible_ids)
            pool.extend(eligible_ids)
        if (
            _digest(names) != spec.selected_name_sha256
            or distribution["eligible_items"] != counted_items
            or distribution["eligible_concepts"] != counted_concepts
            or counted_items < spec.minimum_eligible_items
            or counted_concepts < spec.minimum_eligible_concepts
        ):
            raise WorkspaceEvalContractError(
                f"workspace {spec.slug} eligibility denominator changed"
            )
        pools[spec.slug] = pool
    return pools


def _validate_scored_items(
    items: Any,
    eligibility: Mapping[str, Any],
    pools: Mapping[str, list[str]],
) -> dict[str, list[Mapping[str, Any]]]:
    if not isinstance(items, list) or len(items) != sum(
        spec.publication_count for spec in FIXTURES
    ):
        raise WorkspaceEvalContractError("workspace scored item count changed")
    by_distribution: dict[str, list[Mapping[str, Any]]] = {
        slug: [] for slug in FIXTURE_BY_SLUG
    }
    for item in items:
        if (
            not isinstance(item, Mapping)
            or set(item)
            != {
                "distribution",
                "name",
                "included_in_metrics",
                "eligible_concept_ids",
                "boundary",
                "base_model_target_competence",
                "actual_final_top1_id",
                "actual_final_top1_token",
                "layers",
            }
            or item["distribution"] not in by_distribution
        ):
            raise WorkspaceEvalContractError("workspace scored item schema changed")
        by_distribution[item["distribution"]].append(item)
    expected_layer_keys = {str(layer) for layer in ALL_LAYERS}
    for spec in FIXTURES:
        distribution_items = by_distribution[spec.slug]
        eligibility_items = eligibility["distributions"][spec.slug]["items"]
        if [item["name"] for item in distribution_items] != [
            item["name"] for item in eligibility_items
        ]:
            raise WorkspaceEvalContractError(
                f"workspace {spec.slug} scored item order changed"
            )
        for item, eligible in zip(distribution_items, eligibility_items, strict=True):
            own_ids = eligible["eligible_concept_ids"]
            if (
                item["included_in_metrics"] is not eligible["included_in_metrics"]
                or item["eligible_concept_ids"] != own_ids
                or item["boundary"] != eligible["boundary"]
                or not isinstance(item["actual_final_top1_id"], int)
                or isinstance(item["actual_final_top1_id"], bool)
                or item["actual_final_top1_id"] < 0
                or not isinstance(item["actual_final_top1_token"], str)
                or not isinstance(item["layers"], Mapping)
                or set(item["layers"]) != expected_layer_keys
            ):
                raise WorkspaceEvalContractError(
                    f"workspace {spec.slug}/{item['name']} scored identity changed"
                )
            competence = item["base_model_target_competence"]
            if spec.target_boundary:
                if (
                    not isinstance(competence, Mapping)
                    or set(competence)
                    != {
                        "first_target_token_id",
                        "rank",
                        "top1_match",
                        "used_as_filter",
                    }
                    or competence["first_target_token_id"]
                    != eligible["boundary"]["target_token_ids"][0]
                    or not isinstance(competence["rank"], int)
                    or competence["rank"] < 1
                    or not isinstance(competence["top1_match"], bool)
                    or competence["used_as_filter"] is not False
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']} competence changed"
                    )
            elif competence is not None:
                raise WorkspaceEvalContractError(
                    f"workspace {spec.slug}/{item['name']} has spurious competence"
                )
            for layer in ALL_LAYERS:
                layer_record = item["layers"][str(layer)]
                if (
                    not isinstance(layer_record, Mapping)
                    or set(layer_record)
                    != {
                        "concept_ranks",
                        "candidate_label_pool_ranks",
                        "motor",
                    }
                    or not isinstance(layer_record["concept_ranks"], Mapping)
                    or set(layer_record["concept_ranks"]) != set(VARIANTS)
                    or any(
                        not isinstance(layer_record["concept_ranks"][variant], Mapping)
                        or set(layer_record["concept_ranks"][variant]) != set(own_ids)
                        or any(
                            not isinstance(rank, int)
                            or isinstance(rank, bool)
                            or rank < 1
                            for rank in layer_record["concept_ranks"][variant].values()
                        )
                        for variant in VARIANTS
                    )
                    or not isinstance(layer_record["candidate_label_pool_ranks"], Mapping)
                    or set(layer_record["candidate_label_pool_ranks"])
                    != set(pools[spec.slug])
                    or any(
                        not isinstance(rank, int)
                        or isinstance(rank, bool)
                        or rank < 1
                        for rank in layer_record[
                            "candidate_label_pool_ranks"
                        ].values()
                    )
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']}/L{layer} ranks changed"
                    )
                if any(
                    layer_record["candidate_label_pool_ranks"][concept_id]
                    != layer_record["concept_ranks"]["candidate"][concept_id]
                    for concept_id in own_ids
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']}/L{layer} "
                        "candidate pool rank disagrees with own rank"
                    )
                motor = layer_record["motor"]
                if (
                    not isinstance(motor, Mapping)
                    or set(motor)
                    != {
                        "actual_final_rank",
                        "candidate_logit_top1_agreement",
                        "candidate_logit_js_nats",
                    }
                    or not isinstance(motor["actual_final_rank"], Mapping)
                    or set(motor["actual_final_rank"]) != {"candidate", "logit"}
                    or any(
                        not isinstance(rank, int)
                        or isinstance(rank, bool)
                        or rank < 1
                        for rank in motor["actual_final_rank"].values()
                    )
                    or not isinstance(motor["candidate_logit_top1_agreement"], bool)
                    or not isinstance(motor["candidate_logit_js_nats"], (int, float))
                    or isinstance(motor["candidate_logit_js_nats"], bool)
                    or not math.isfinite(motor["candidate_logit_js_nats"])
                    or motor["candidate_logit_js_nats"] < 0.0
                    or motor["candidate_logit_js_nats"]
                    > JS_DIVERGENCE_MAX_NATS + JS_DIVERGENCE_FLOAT32_TOLERANCE
                ):
                    raise WorkspaceEvalContractError(
                        f"workspace {spec.slug}/{item['name']}/L{layer} motor evidence changed"
                    )
    return by_distribution


def _validate_workspace_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(report, Mapping):
        raise WorkspaceEvalContractError("workspace report must be an object")
    required = {
        "schema_version",
        "kind",
        "status",
        "evaluation_config_sha256",
        "config",
        "fit_identity",
        "fixtures",
        "eligibility",
        "controls",
        "items",
        "summaries",
        "statistics",
        "adjudication",
        "workspace_report_sha256",
    }
    if set(report) != required:
        raise WorkspaceEvalContractError("workspace report schema mismatch")
    if (
        report["schema_version"] != WORKSPACE_REPORT_SCHEMA_VERSION
        or report["kind"] != WORKSPACE_REPORT_KIND
        or report["status"] != "complete"
    ):
        raise WorkspaceEvalContractError("workspace report is incomplete or wrong kind")
    body = dict(report)
    claimed_report_sha = body.pop("workspace_report_sha256")
    if claimed_report_sha != _digest(body):
        raise WorkspaceEvalContractError("workspace report content digest mismatch")
    _require_finite(report)
    config = report["config"]
    if (
        not isinstance(config, Mapping)
        or set(config)
        != {
            "schema_version",
            "fit",
            "fixtures",
            "eligibility_sha256",
            "tokenizer_preflight",
            "controls",
            "metrics",
            "source",
            "runtime",
        }
        or config["schema_version"] != WORKSPACE_REPORT_SCHEMA_VERSION
        or report["evaluation_config_sha256"] != _digest(config)
    ):
        raise WorkspaceEvalContractError("workspace evaluation config digest/schema mismatch")
    if config["metrics"] != _metric_config():
        raise WorkspaceEvalContractError("workspace metric/layer config is not frozen")
    if report["fit_identity"] != config["fit"]:
        raise WorkspaceEvalContractError("workspace fit/lens identity chain mismatch")
    _validate_report_fit_identity(report["fit_identity"])
    expected_preflight = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "add_special_tokens": False,
        "force_bos_for_model_input": True,
        "offsets_from_complete_prompt_plus_target": True,
        "target_tokens_excluded_from_scored_prefix": True,
    }
    if config["tokenizer_preflight"] != expected_preflight:
        raise WorkspaceEvalContractError("workspace tokenizer preflight identity changed")
    source = config["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"git_revision", "digest"}
        or not _is_lower_hex(source["git_revision"], 40)
        or not _is_lower_hex(source["digest"], 64)
        or not isinstance(config["runtime"], Mapping)
        or not config["runtime"]
    ):
        raise WorkspaceEvalContractError("workspace scorer source/runtime identity changed")
    _validate_runtime_identity(config["runtime"], fit_side=False)
    expected_fixtures = [_fixture_identity(spec) for spec in FIXTURES]
    if config["fixtures"] != expected_fixtures or report["fixtures"] != expected_fixtures:
        raise WorkspaceEvalContractError("workspace fixture identity mismatch")
    eligibility = report["eligibility"]
    pools = _validate_eligibility_manifest(eligibility)
    if eligibility["eligibility_sha256"] != config["eligibility_sha256"]:
        raise WorkspaceEvalContractError("workspace eligibility identity mismatch")
    controls = report["controls"]
    expected_controls = _control_permutations()
    expected_source_permutation = {
        str(layer): (layer + 17) % len(ALL_LAYERS) for layer in ALL_LAYERS
    }
    if (
        not isinstance(controls, Mapping)
        or set(controls)
        != {
            "source_layers",
            "output_basis",
            "output_basis_sha256",
            "seed",
        }
        or controls != expected_controls
        or controls["source_layers"] != expected_source_permutation
        or controls["seed"] != CONTROL_SEED
        or not isinstance(controls["output_basis"], list)
        or sorted(controls["output_basis"]) != list(range(D_MODEL))
        or controls["output_basis_sha256"] != _digest(controls["output_basis"])
        or config["controls"]
        != {
            "source_layers": expected_source_permutation,
            "output_basis_sha256": controls["output_basis_sha256"],
            "seed": CONTROL_SEED,
        }
    ):
        raise WorkspaceEvalContractError("workspace control identity mismatch")
    by_distribution = _validate_scored_items(report["items"], eligibility, pools)
    expected_summaries = _build_summaries(report["items"])
    if report["summaries"] != expected_summaries:
        raise WorkspaceEvalContractError("workspace summaries do not match raw ranks")
    statistics = report["statistics"]
    expected_bootstrap = _bootstrap_deltas(by_distribution)
    expected_permutation = _label_permutation_scores(by_distribution)
    if statistics != {
        "bootstrap": expected_bootstrap,
        "label_permutation": expected_permutation,
    }:
        raise WorkspaceEvalContractError(
            "workspace deterministic statistics do not match raw ranks"
        )
    expected_evidence = _adjudication_evidence(
        expected_summaries, expected_bootstrap, expected_permutation
    )
    expected_adjudication = _adjudicate(expected_evidence)
    if report["adjudication"] != expected_adjudication:
        raise WorkspaceEvalContractError(
            "workspace verdict does not match preregistered raw evidence"
        )
    return dict(report)


def _read_json_object(
    path: str | pathlib.Path,
    *,
    label: str,
) -> dict[str, Any]:
    path = pathlib.Path(path)
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkspaceEvalContractError(f"invalid {label} at {path}") from exc
    if not isinstance(payload, Mapping):
        raise WorkspaceEvalContractError(f"invalid {label} object at {path}")
    return dict(payload)


def load_completed_workspace_report(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    if path.suffix == ".pt":
        raise WorkspaceEvalContractError("expected explicit workspace report, not bare .pt")
    payload = _read_json_object(path, label="workspace report")
    return _validate_workspace_report(payload)


@_modal_eval_function
def evaluate_workspace(fit_manifest: str, lens: str) -> str:
    import sys

    import torch
    import transformers

    import jlens

    sys.path.insert(0, "/root/scripts")
    from modal_fit_lens import load_completed_fit_manifest

    manifest_path, fit_record = _load_source_bound_fit_manifest(
        fit_manifest,
        VOL_MOUNT,
        load_completed_fit_manifest,
    )
    fit_identity = _bind_fit_identity(fit_record, manifest_path, lens, VOL_MOUNT)
    fixtures = _fetch_fixtures(f"{VOL_MOUNT}/eval-fixtures/{JLENS_REVISION}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    eligibility, prepared = _preflight(tokenizer, fixtures)
    controls = _control_permutations()
    runtime = _runtime_identity()
    config = _evaluation_config(fit_identity, eligibility, controls, runtime)
    config_sha = _digest(config)
    tag = f"gemma-4-E2B-it-workspace-{config_sha}"
    report_path = pathlib.Path(f"{VOL_MOUNT}/eval/{tag}.json")
    if report_path.exists():
        existing = _read_json_object(
            report_path, label="workspace evaluation resume record"
        )
        if existing.get("status") == "complete":
            _validate_workspace_report(existing)
            if (
                existing.get("evaluation_config_sha256") != config_sha
                or existing.get("config") != config
            ):
                raise WorkspaceEvalContractError(
                    f"completed workspace evaluation mismatch at {report_path}"
                )
            return str(report_path)
        if (
            existing.get("evaluation_config_sha256") != config_sha
            or existing.get("config") != config
        ):
            raise WorkspaceEvalContractError(
                f"workspace evaluation config mismatch at {report_path}"
            )
    pending = {
        "schema_version": WORKSPACE_REPORT_SCHEMA_VERSION,
        "kind": WORKSPACE_REPORT_KIND,
        "status": "pending",
        "evaluation_config_sha256": config_sha,
        "config": config,
    }
    _atomic_write_json(report_path, pending)
    _commit_volume()

    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    model = jlens.from_hf(hf, tokenizer, force_bos=True)
    candidate_lens = jlens.JacobianLens.load(lens)
    if (
        candidate_lens.n_prompts != 1_000
        or candidate_lens.d_model != D_MODEL
        or candidate_lens.source_layers != list(ALL_LAYERS)
    ):
        raise WorkspaceEvalContractError("loaded lens tensor metadata changed")
    scored_items = _score_items(model, hf, candidate_lens, prepared, controls)
    summaries = _build_summaries(scored_items)
    by_distribution = {
        slug: [item for item in scored_items if item["distribution"] == slug]
        for slug in FIXTURE_BY_SLUG
    }
    bootstrap = _bootstrap_deltas(by_distribution)
    permutation = _label_permutation_scores(by_distribution)
    evidence = _adjudication_evidence(summaries, bootstrap, permutation)
    adjudication = _adjudicate(evidence)
    report = _complete_report(
        {
            "schema_version": WORKSPACE_REPORT_SCHEMA_VERSION,
            "kind": WORKSPACE_REPORT_KIND,
            "status": "complete",
            "evaluation_config_sha256": config_sha,
            "config": config,
            "fit_identity": fit_identity,
            "fixtures": [_fixture_identity(spec) for spec in FIXTURES],
            "eligibility": eligibility,
            "controls": controls,
            "items": scored_items,
            "summaries": summaries,
            "statistics": {"bootstrap": bootstrap, "label_permutation": permutation},
            "adjudication": adjudication,
        }
    )
    _validate_workspace_report(report)
    _atomic_write_json(report_path, report)
    _commit_volume()
    return str(report_path)


@_modal_local_entrypoint
def main(fit_manifest: str, lens: str):
    print(evaluate_workspace.remote(fit_manifest=fit_manifest, lens=lens))
