"""Pure contracts and statistics for the frozen audio workspace evaluation.

This module deliberately contains no Modal, model, processor, audio decoder, or
final-lens imports.  Deployment code supplies bytes and raw ranks; the
functions here fail closed over their identities, coordinates, reductions, and
adjudication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class AudioWorkspaceEvalContractError(RuntimeError):
    """A frozen audio-evaluation identity or scientific contract was violated."""


SCHEMA_VERSION = 1
STIMULUS_MANIFEST_KIND = "audio_workspace_stimulus_manifest"
CALIBRATION_KIND = "audio_workspace_synthetic_speech_calibration"
PREREGISTRATION_KIND = "audio_workspace_preregistration"
REPORT_KIND = "audio_workspace_fixed_band_report"

VALIDATED_STATUS = "validated_fixed_band_synthetic_speech_readout"
NO_READOUT_STATUS = "no_fixed_band_synthetic_speech_readout"
INCONCLUSIVE_STIMULUS_STATUS = "inconclusive_synthetic_stimulus"
INVALID_PROTOCOL_STATUS = "invalid_protocol_or_artifact"
SCIENTIFIC_STATUSES = (
    VALIDATED_STATUS,
    NO_READOUT_STATUS,
    INCONCLUSIVE_STIMULUS_STATUS,
    INVALID_PROTOCOL_STATUS,
)

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "70af34e20bd4b7a91f0de6b22675850c43922a03"
D_MODEL = 1_536
JLENS_REVISION = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
WHISPER_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
AUDIO_FIT_CONFIG_SHA256 = "ee7cd4e42991fec5a00b4256ba466ff163ebd64fa22963334033066e7d531275"
AUDIO_LENS_SHA256 = "da0ccabf1ee14e4df060f97f31cf0132a0d3f6ed2cb45b6c77738693bc8f1aa9"
CONTROL_OUTPUT_BASIS_SHA256 = "57b908355f62e17de36979d52d3b7a60bc7556ed3f87e95b6a2928f29c083b2d"

TTS_ENGINE = "openai-tts"
TTS_ENDPOINT = "https://api.openai.com/v1/audio/speech"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_RESPONSE_FORMAT = "wav"
TTS_INPUT_POLICY = "strip_double_quotes_collapse_whitespace"
TTS_SYNTHESIS_POLICY = "sealed_source_bytes_nonreproducible_generation"
TTS_VARIANTS = ("onyx", "nova")
TTS_SAMPLE_RATE = 24_000
NORMALIZED_SAMPLE_RATE = 16_000
RESAMPLE_UP = 2
RESAMPLE_DOWN = 3
AUDIO_POSITION = "last_processor_valid_audio_position"
RESPONSE_POSITION = "response_position"
POSITIONS = (AUDIO_POSITION, RESPONSE_POSITION)

KS = (1, 2, 5, 10, 20, 50, 100)
LENS_LAYERS = tuple(range(34))
RESIDUAL_LAYERS = tuple(range(35))
EARLY_LAYERS = tuple(range(13))
CANDIDATE_LAYERS = tuple(range(13, 32))
MOTOR_LAYERS = (32, 33)
REGIONS = {
    "early_l0_l12": EARLY_LAYERS,
    "candidate_l13_l31": CANDIDATE_LAYERS,
    "motor_l32_l33": MOTOR_LAYERS,
}
RANK_VARIANTS = ("candidate", "logit", "transposed", "permuted")
DISTRIBUTIONS = ("association", "multihop", "multilingual", "order-ops", "poetry")
NON_MULTILINGUAL_DISTRIBUTIONS = ("association", "multihop", "order-ops", "poetry")
PUBLICATION_COUNTS = {
    "association": 50,
    "multihop": 50,
    "multilingual": 52,
    "order-ops": 55,
    "poetry": 52,
}
EXPECTED_ITEM_COUNT = 259
EXPECTED_OBSERVATION_COUNT = 518
EXPECTED_CALIBRATION_CELL_COUNT = 68
EXCLUDED_DISTRIBUTIONS = ("typo",)
EXCLUDED_COORDINATES = (
    ("multilingual", "filipino-opposite-up"),
    ("multilingual", "irish-opposite-big"),
)

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026070902
PERMUTATION_REPLICATES = 10_000
PERMUTATION_SEED = 2026070901
CONTROL_SEED = 2026070903
CONTROL_SOURCE_ROTATION = 17
CONTENT_DELTA_THRESHOLD = 0.02
LABEL_MAX_STAT_P_THRESHOLD = 0.01
CALIBRATION_MACRO_CER_MAX = 0.35
CALIBRATION_CELL_CER_MAX = 0.80
JS_DIVERGENCE_MAX_NATS = math.log(2.0)
JS_DIVERGENCE_TOLERANCE = 1e-6

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


def allowed_forms(distribution: str, authored: str) -> tuple[str, ...]:
    """Return the canonical evaluator's exact authored-token form policy."""
    if distribution not in DISTRIBUTIONS or not isinstance(authored, str) or not authored:
        raise AudioWorkspaceEvalContractError("concept form request is invalid")
    if distribution != "order-ops":
        return (authored,)
    try:
        return ORDER_OP_SYNONYMS[authored]
    except KeyError as exc:
        raise AudioWorkspaceEvalContractError(
            f"order-ops has unregistered intermediate {authored!r}"
        ) from exc


_COORDINATE_NAMES = {
    "association": (
        "grief",
        "pregnant",
        "lie",
        "poverty",
        "proposal",
        "fired",
        "divorce",
        "relief",
        "pennsylvania",
        "prohibition",
        "nasa",
        "einstein",
        "darwin",
        "beatles",
        "lincoln",
        "obama",
        "olympics",
        "churchill",
        "newton",
        "mozart",
        "berlin",
        "armstrong",
        "wright",
        "gandhi",
        "edison",
        "cia",
        "vatican",
        "hollywood",
        "pentagon",
        "unesco",
        "anger",
        "frustrated",
        "lonely",
        "shame",
        "noir",
        "horror",
        "fairy",
        "siblings",
        "rivals",
        "mother",
        "boss",
        "neighbors",
        "teacher",
        "underwater",
        "desert",
        "chess",
        "poker",
        "interview",
        "dawn",
        "winter",
    ),
    "multihop": (
        "carnival-ocean",
        "amazon-language",
        "mars-color",
        "spider-legs",
        "basketball-players",
        "paper-continent",
        "christmas-season",
        "osu-rival-mascot",
        "topeka-west",
        "atomic-79-symbol",
        "atomic-26-symbol",
        "atomic-29-symbol",
        "atomic-80-state",
        "atomic-29-flame",
        "planet-3-moons",
        "month-3-godof",
        "spaceneedle-border",
        "rhyme-rain-neighbor",
        "rhyme-door-doubled",
        "rhyme-chair-flag",
        "rhyme-spoon-orbit",
        "etym-wargod-month",
        "etym-frigg-position",
        "etym-saturn-position",
        "chem-photosynthesis-Z",
        "chem-atmosphere-Z",
        "chem-organic-Z",
        "chem-bones-Z",
        "func-pumps-chambers",
        "func-filters-count",
        "birthstone-emerald-month",
        "violin-strings",
        "nhop-primary-planet",
        "nhop-rings-planet",
        "nhop-guitar-planet",
        "nhop-compass-planet",
        "nhop-volleyball-planet",
        "nhop-alphabet-element",
        "nhop-fortnight-element",
        "nhop-cube-element",
        "nhop-rainbow-element",
        "nhop-spider-element",
        "nhop-insect-element",
        "nhop-week-element",
        "super-largest-country-capital",
        "super-smallest-continent",
        "super-populous-capital",
        "super-largest-island-nation",
        "colosseum-currency",
        "greatwall-ocean",
    ),
    "multilingual": (
        "spanish-opposite-big",
        "french-season-summer",
        "german-opposite-loud",
        "chinese-color-banana",
        "japanese-day-monday",
        "russian-opposite-heavy",
        "italian-color-sky",
        "portuguese-season-spring",
        "arabic-color-blood",
        "korean-opposite-hot",
        "dutch-opposite-big",
        "greek-color-sea",
        "hindi-color-snow",
        "polish-opposite-wet",
        "turkish-opposite-fast",
        "swedish-opposite-hard",
        "hebrew-color-moon",
        "finnish-season-summer",
        "danish-color-ocean",
        "norwegian-season-spring",
        "czech-opposite-tall",
        "thai-color-blood",
        "vietnamese-opposite-heavy",
        "indonesian-color-grass",
        "romanian-opposite-young",
        "hungarian-color-sea",
        "spanish-number-three",
        "german-number-five",
        "french-number-seven",
        "italian-number-two",
        "russian-number-four",
        "chinese-number-eight",
        "portuguese-direction-up",
        "dutch-direction-left",
        "polish-direction-left",
        "spanish-body-see",
        "german-body-hear",
        "french-body-walk",
        "spanish-family-uncle",
        "german-family-aunt",
        "french-family-nephew",
        "italian-family-grandmother",
        "spanish-phase-water",
        "german-phase-water",
        "ukrainian-opposite-day",
        "bulgarian-color-grass",
        "serbian-opposite-day",
        "croatian-opposite-big",
        "slovak-opposite-young",
        "estonian-opposite-open",
        "persian-color-blood",
        "bengali-opposite-up",
    ),
    "order-ops": (
        "parens-add-mult",
        "parens-sub-mult",
        "mult-parens-add",
        "mult-parens-sub",
        "add-mult-right",
        "mult-add-left",
        "sub-mult-right",
        "mult-sub-left",
        "add-sub-left-right",
        "sub-add-left-right",
        "div-parens-add",
        "parens-add-div",
        "parens-sub-div",
        "div-sub-left",
        "chain-add-mult-add",
        "chain-mult-sub-add",
        "chain-sub-mult-sub",
        "nested-add-mult-sub",
        "nested-mult-add-div",
        "nested-sub-add-mult",
        "word-add-mult",
        "word-mult-sub",
        "word-parens",
        "mod-add",
        "mod-mult",
        "square-sub",
        "mult-mult-left",
        "div-div-left",
        "mult-div-left",
        "div-mult-left",
        "mult-div-mult",
        "add-add-add",
        "sub-sub-sub",
        "add-sub-add-sub",
        "chain-mult-mult-add",
        "nested-add-add-mult",
        "nested-mult-sub-add",
        "nested-div-add-sub",
        "nested-sub-div-mult",
        "nested4-add-mult-add-div",
        "nested4-sub-add-mult-sub",
        "redundant-parens-mult",
        "redundant-parens-div",
        "word-sub-mult",
        "word-add-add",
        "word-div-sub",
        "mixed-mult-add",
        "mixed-parens-mult",
        "mod-sub",
        "square-div",
        "square-mult",
        "pystar-add",
        "pystar-sub",
        "floordiv-add",
        "floordiv-mult",
    ),
    "poetry": (
        "couplet-breath-death",
        "couplet-stone-bone",
        "couplet-divine-wine",
        "couplet-flame-name",
        "couplet-sleep-deep",
        "couplet-gold-cold",
        "couplet-flight-night",
        "couplet-wave-grave",
        "couplet-true-through",
        "couplet-throne-alone",
        "couplet-divine-wine-to-sign",
        "couplet-dust-trust",
        "couplet-ground-sound",
        "couplet-bed-dead",
        "couplet-fight-light",
        "couplet-sea-free",
        "couplet-moon-soon",
        "couplet-art-heart",
        "couplet-rain-pain",
        "couplet-snow-grow",
        "couplet-shore-more",
        "couplet-heart-apart",
        "couplet-miss-kiss",
        "couplet-eyes-lies",
        "couplet-hold-old",
        "couplet-fall-wall",
        "couplet-chase-face",
        "couplet-trap-map",
        "couplet-gate-fate",
        "couplet-gun-run",
        "couplet-told-gold",
        "couplet-mind-find",
        "couplet-dream-seem",
        "couplet-time-climb",
        "couplet-known-grown",
        "couplet-war-door",
        "couplet-smoke-spoke",
        "couplet-head-bed",
        "couplet-deep-keep",
        "couplet-cat-mat",
        "couplet-cake-mistake",
        "couplet-shoe-blue",
        "couplet-foam-home",
        "couplet-song-long",
        "couplet-near-fear",
        "couplet-clear-year",
        "couplet-sky-fly",
        "couplet-high-cry",
        "couplet-star-far",
        "couplet-bend-end",
        "couplet-bright-sight",
        "couplet-white-write",
    ),
}
EXPECTED_COORDINATES = tuple(
    (distribution, name)
    for distribution in DISTRIBUTIONS
    for name in _COORDINATE_NAMES[distribution]
)
EXPECTED_COORDINATES_SHA256 = "12a160e85aa877dfa3e20a7aecb92e0ad9d2e0d63ed2290406a0af2a0697fd18"


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
    ),
)
FIXTURE_BY_SLUG = {fixture.slug: fixture for fixture in FIXTURES}


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact sorted UTF-8 JSON, rejecting non-finite numbers."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AudioWorkspaceEvalContractError("value is not canonical JSON") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise AudioWorkspaceEvalContractError(f"{label} schema mismatch")
    return value


def _require_finite_tree(value: Any, label: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AudioWorkspaceEvalContractError(f"{label} contains a nonfinite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_tree(child, f"{label}[{index}]")


def _detached_json(value: Any, label: str) -> Any:
    """Return a recursively detached, canonical-JSON-compatible value."""
    canonical_json_bytes(value)
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise AudioWorkspaceEvalContractError(f"{label} cannot be detached") from exc


def seal_mapping(value: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AudioWorkspaceEvalContractError("sealed value must be a mapping")
    body = _detached_json(dict(value), "sealed value")
    body.pop(hash_field, None)
    return {**body, hash_field: canonical_sha256(body)}


def validate_seal(value: Any, hash_field: str, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or hash_field not in value:
        raise AudioWorkspaceEvalContractError(f"{label} is unsealed")
    body = dict(value)
    claimed = body.pop(hash_field)
    if not _is_sha256(claimed) or canonical_sha256(body) != claimed:
        raise AudioWorkspaceEvalContractError(f"{label} content digest mismatch")
    _require_finite_tree(value, label)
    return _detached_json(dict(value), label)


def validate_audio_artifact_chain(
    completed_run: Mapping[str, Any],
    *,
    completed_run_manifest_bytes: bytes,
    volume_root: str | pathlib.Path,
    completed_run_manifest_sha256: str,
) -> dict[str, Any]:
    """Physically validate the completed run and its exact serialized identity."""
    if (
        not isinstance(completed_run_manifest_bytes, bytes)
        or not completed_run_manifest_bytes
        or not _is_sha256(completed_run_manifest_sha256)
        or sha256_bytes(completed_run_manifest_bytes) != completed_run_manifest_sha256
    ):
        raise AudioWorkspaceEvalContractError("completed-run manifest bytes or SHA-256 are invalid")
    try:
        serialized_run = json.loads(completed_run_manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioWorkspaceEvalContractError(
            "completed-run manifest bytes are invalid JSON"
        ) from exc
    if not isinstance(completed_run, Mapping) or canonical_json_bytes(
        serialized_run
    ) != canonical_json_bytes(dict(completed_run)):
        raise AudioWorkspaceEvalContractError(
            "completed-run mapping does not match its exact manifest bytes"
        )
    try:
        from .audio_fitting import validate_completed_run

        validated = validate_completed_run(completed_run, volume_root=volume_root)
    except Exception as exc:
        raise AudioWorkspaceEvalContractError(
            "completed audio run or physical lens chain is invalid"
        ) from exc
    lens = validated["lens"]
    if (
        validated["fit_config_sha256"] != AUDIO_FIT_CONFIG_SHA256
        or lens["fit_config_sha256"] != AUDIO_FIT_CONFIG_SHA256
        or lens["sha256"] != AUDIO_LENS_SHA256
        or lens["dtype"] != "float16"
        or lens["source_layers"] != list(LENS_LAYERS)
        or lens["d_model"] != D_MODEL
    ):
        raise AudioWorkspaceEvalContractError(
            "completed audio run is not the exact final audio lens"
        )
    return {
        "fit_config_sha256": AUDIO_FIT_CONFIG_SHA256,
        "lens_sha256": AUDIO_LENS_SHA256,
        "lens_dtype": "float16",
        "lens_layers": list(LENS_LAYERS),
        "completed_run_manifest_sha256": completed_run_manifest_sha256,
    }


def fixture_identity(spec: FixtureSpec) -> dict[str, Any]:
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
    }


def _valid_prompt(prompt: Any) -> bool:
    return isinstance(prompt, str) and bool(prompt)


def decode_publication_fixture(spec: FixtureSpec, raw: bytes) -> list[dict[str, Any]]:
    """Validate canonical fixture bytes and return its frozen publication prefix."""
    if spec.slug not in FIXTURE_BY_SLUG or spec != FIXTURE_BY_SLUG[spec.slug]:
        raise AudioWorkspaceEvalContractError("fixture spec is not frozen")
    if len(raw) != spec.n_bytes or sha256_bytes(raw) != spec.sha256:
        raise AudioWorkspaceEvalContractError(f"{spec.slug}: fixture bytes or SHA-256 changed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AudioWorkspaceEvalContractError(f"{spec.slug}: invalid fixture JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"items"}:
        raise AudioWorkspaceEvalContractError(f"{spec.slug}: fixture root schema mismatch")
    items = payload["items"]
    if not isinstance(items, list) or len(items) != spec.raw_count:
        raise AudioWorkspaceEvalContractError(f"{spec.slug}: fixture raw count changed")
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != spec.item_keys:
            raise AudioWorkspaceEvalContractError(f"{spec.slug}[{index}]: item schema changed")
        name = item.get("name")
        concepts = item.get("intermediates")
        if not isinstance(name, str) or not name or name in seen:
            raise AudioWorkspaceEvalContractError(f"{spec.slug}[{index}]: invalid name")
        seen.add(name)
        if not _valid_prompt(item.get("prompt")):
            raise AudioWorkspaceEvalContractError(f"{spec.slug}/{name}: prompt is not a string")
        if (
            not isinstance(concepts, list)
            or not concepts
            or not all(isinstance(c, str) and c for c in concepts)
        ):
            raise AudioWorkspaceEvalContractError(f"{spec.slug}/{name}: invalid intermediates")
        if spec.target_boundary and (not isinstance(item.get("target"), str) or not item["target"]):
            raise AudioWorkspaceEvalContractError(f"{spec.slug}/{name}: invalid target")
    selected = items[: spec.publication_count]
    if canonical_sha256([item["name"] for item in selected]) != spec.selected_name_sha256:
        raise AudioWorkspaceEvalContractError(f"{spec.slug}: selected coordinate identity changed")
    return [dict(item) for item in selected]


def decode_publication_fixtures(raw_by_distribution: Mapping[str, bytes]) -> list[dict[str, Any]]:
    """Decode the five sources and apply only the frozen engine exclusions."""
    if set(raw_by_distribution) != set(DISTRIBUTIONS):
        raise AudioWorkspaceEvalContractError(
            "fixture set must be exactly the five spoken distributions"
        )
    selected: list[dict[str, Any]] = []
    for spec in FIXTURES:
        rows = decode_publication_fixture(spec, raw_by_distribution[spec.slug])
        if spec.slug == "multilingual":
            rows = [
                row
                for row in rows
                if row["name"] not in {"filipino-opposite-up", "irish-opposite-big"}
            ]
        selected.extend({"distribution": spec.slug, **row} for row in rows)
    coordinates = [(row["distribution"], row["name"]) for row in selected]
    if (
        coordinates != list(EXPECTED_COORDINATES)
        or canonical_sha256(coordinates) != EXPECTED_COORDINATES_SHA256
    ):
        raise AudioWorkspaceEvalContractError("spoken fixture coordinates changed")
    return selected


def language_for_coordinate(distribution: str, name: str) -> str:
    if distribution != "multilingual":
        if distribution not in DISTRIBUTIONS:
            raise AudioWorkspaceEvalContractError("unknown distribution")
        return "en-us"
    language = name.split("-", 1)[0]
    try:
        return LANGUAGE_CODES[language]
    except KeyError as exc:
        raise AudioWorkspaceEvalContractError(f"no frozen language code for {name}") from exc


def spoken_script(distribution: str, item: Mapping[str, Any]) -> str:
    """Mechanically stop at the canonical target-excluding spoken boundary."""
    prompt = item.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise AudioWorkspaceEvalContractError("spoken prompt must be a nonempty string")
    if distribution in {"multihop", "multilingual", "order-ops"}:
        if not isinstance(item.get("target"), str) or not item["target"]:
            raise AudioWorkspaceEvalContractError("target-boundary item has no target")
        result = prompt
    elif distribution == "association":
        if "target" in item:
            raise AudioWorkspaceEvalContractError("association unexpectedly contains a target")
        result = prompt
    elif distribution == "poetry":
        if "target" in item:
            raise AudioWorkspaceEvalContractError("poetry unexpectedly contains a target")
        newline = prompt.rfind("\n")
        if newline < 0:
            raise AudioWorkspaceEvalContractError("poetry prompt has no canonical newline")
        result = prompt[: newline + 1]
    else:
        raise AudioWorkspaceEvalContractError("distribution has no spoken-script policy")
    if not result:
        raise AudioWorkspaceEvalContractError("spoken script is empty")
    return result


def build_spoken_items(selected_items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if [(row.get("distribution"), row.get("name")) for row in selected_items] != list(
        EXPECTED_COORDINATES
    ):
        raise AudioWorkspaceEvalContractError(
            f"selected items are not the exact {EXPECTED_ITEM_COUNT} coordinates"
        )
    result = []
    for index, row in enumerate(selected_items):
        distribution = str(row["distribution"])
        script = spoken_script(distribution, row)
        item = {key: value for key, value in row.items() if key != "distribution"}
        result.append(
            {
                "coordinate_index": index,
                "distribution": distribution,
                "name": str(row["name"]),
                "language": language_for_coordinate(distribution, str(row["name"])),
                "script": script,
                "script_sha256": sha256_bytes(script.encode("utf-8")),
                "source_item_sha256": canonical_sha256(item),
                "intermediates": list(row["intermediates"]),
                "target_excluded": True,
            }
        )
    return result


def tts_input(script: str) -> str:
    """Spoken synthesis input: dangling prompt quotes are removed.

    The publication prompts end in an opening double quote that has no spoken
    form and silences the pinned synthesis engine. CER normalization already
    discards punctuation, so removing quotes never changes a calibration
    reference.
    """
    if not isinstance(script, str) or not script:
        raise AudioWorkspaceEvalContractError("TTS script must be a nonempty string")
    cleaned = " ".join(script.replace('"', " ").split())
    if not cleaned:
        raise AudioWorkspaceEvalContractError("TTS input is empty after quote removal")
    return cleaned


def normalize_transcript(value: str) -> str:
    """NFKC/casefold, remove Unicode punctuation, and collapse whitespace."""
    if not isinstance(value, str):
        raise AudioWorkspaceEvalContractError("transcript must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in normalized
    )
    return " ".join(normalized.split())


# Vuk-Gaj digraphic correspondence over casefolded text. Serbian is written in
# both scripts; the pinned ASR prefers Latin while the publication scripts are
# Cyrillic, so calibration compares both sides in the Latin form.
SERBIAN_CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "ђ": "đ",
    "е": "e",
    "ж": "ž",
    "з": "z",
    "и": "i",
    "ј": "j",
    "к": "k",
    "л": "l",
    "љ": "lj",
    "м": "m",
    "н": "n",
    "њ": "nj",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ћ": "ć",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "č",
    "џ": "dž",
    "ш": "š",
}


def transliterate_serbian(value: str) -> str:
    if not isinstance(value, str):
        raise AudioWorkspaceEvalContractError("transliteration input must be text")
    return "".join(SERBIAN_CYRILLIC_TO_LATIN.get(char, char) for char in value)


def normalize_transcript_for_language(value: str, language: str) -> str:
    """Frozen calibration normalization; Serbian maps to its Latin script."""
    normalized = normalize_transcript(value)
    if language == "sr":
        normalized = transliterate_serbian(normalized)
    return normalized


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, 1):
        current = [row_index]
        for column_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str, language: str = "") -> float:
    left = normalize_transcript_for_language(reference, language)
    right = normalize_transcript_for_language(hypothesis, language)
    if not left:
        raise AudioWorkspaceEvalContractError("CER reference is empty after normalization")
    return _levenshtein(left, right) / len(left)


def calibration_status(cers: Sequence[float]) -> dict[str, Any]:
    """Apply the frozen macro/per-cell intelligibility thresholds."""
    values = list(cers)
    if len(values) != EXPECTED_CALIBRATION_CELL_COUNT or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise AudioWorkspaceEvalContractError("calibration CER array is invalid")
    macro = sum(float(value) for value in values) / len(values)
    maximum = max(float(value) for value in values)
    return {
        "macro_cer": macro,
        "max_cell_cer": maximum,
        "status": (
            "passed"
            if macro <= CALIBRATION_MACRO_CER_MAX and maximum <= CALIBRATION_CELL_CER_MAX
            else "failed"
        ),
    }


def calibration_coordinates(spoken_items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    multilingual = [item for item in spoken_items if item.get("distribution") == "multilingual"]
    by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in multilingual:
        by_language[str(item.get("language"))].append(item)
    if set(by_language) != set(LANGUAGE_CODES.values()):
        raise AudioWorkspaceEvalContractError("calibration does not cover the exact 34 languages")
    cells = []
    for language in LANGUAGE_CODES.values():
        # Number prompts invite digit-vs-word ASR spellings that measure the
        # metric rather than intelligibility, so non-number items are
        # preferred whenever the language offers one.
        pool = by_language[language]
        non_number = [item for item in pool if "-number-" not in str(item["name"])]
        shortest = min(
            non_number or pool,
            key=lambda item: (len(str(item["script"])), int(item["coordinate_index"])),
        )
        for variant in TTS_VARIANTS:
            cells.append(
                {
                    "distribution": "multilingual",
                    "name": shortest["name"],
                    "language": language,
                    "variant": variant,
                    "script_sha256": shortest["script_sha256"],
                }
            )
    if len(cells) != EXPECTED_CALIBRATION_CELL_COUNT:
        raise AudioWorkspaceEvalContractError("calibration cell count changed")
    return cells


def audit_fit_overlap(
    observations: Sequence[Mapping[str, Any]],
    spoken_items: Sequence[Mapping[str, Any]],
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    fit_manifest_sha256: str,
) -> dict[str, Any]:
    """Reject waveform, decoded-PCM, or normalized-transcript overlap."""
    if (
        not _is_sha256(fit_manifest_sha256)
        or not isinstance(observations, list)
        or len(observations) != EXPECTED_OBSERVATION_COUNT
        or not isinstance(spoken_items, list)
        or len(spoken_items) != EXPECTED_ITEM_COUNT
        or not isinstance(fit_rows, list)
        or len(fit_rows) != 1_000
    ):
        raise AudioWorkspaceEvalContractError(
            "overlap audit inputs have invalid identities or cardinalities"
        )
    stimulus_wave: set[str] = set()
    stimulus_pcm: set[str] = set()
    for row in observations:
        if not isinstance(row, Mapping):
            raise AudioWorkspaceEvalContractError("overlap stimulus observation is invalid")
        for key in ("source_wav_sha256", "normalized_wav_sha256"):
            if not _is_sha256(row.get(key)):
                raise AudioWorkspaceEvalContractError(
                    f"overlap stimulus observation has invalid {key}"
                )
            stimulus_wave.add(str(row[key]))
        for key in ("source_pcm_sha256", "decoded_pcm_sha256"):
            if not _is_sha256(row.get(key)):
                raise AudioWorkspaceEvalContractError(
                    f"overlap stimulus observation has invalid {key}"
                )
            stimulus_pcm.add(str(row[key]))
    if [
        (item.get("distribution"), item.get("name")) if isinstance(item, Mapping) else (None, None)
        for item in spoken_items
    ] != list(EXPECTED_COORDINATES):
        raise AudioWorkspaceEvalContractError("overlap spoken-item coordinates changed")
    stimulus_text: set[str] = set()
    for item in spoken_items:
        script = item.get("script")
        normalized = normalize_transcript(script)
        if not normalized:
            raise AudioWorkspaceEvalContractError("overlap spoken transcript is empty")
        stimulus_text.add(normalized)
    fit_wave: set[str] = set()
    fit_pcm: set[str] = set()
    fit_text: set[str] = set()
    for row in fit_rows:
        if (
            not isinstance(row, Mapping)
            or not _is_sha256(row.get("audio_sha256"))
            or not _is_sha256(row.get("decoded_pcm_sha256"))
            or not isinstance(row.get("transcript"), str)
        ):
            raise AudioWorkspaceEvalContractError(
                "overlap fit row lacks waveform, PCM, or transcript identity"
            )
        normalized = normalize_transcript(row["transcript"])
        if not normalized:
            raise AudioWorkspaceEvalContractError("overlap fit transcript is empty")
        fit_wave.add(str(row["audio_sha256"]))
        fit_pcm.add(str(row["decoded_pcm_sha256"]))
        fit_text.add(normalized)
    overlaps = {
        "waveform": sorted(stimulus_wave & fit_wave),
        "decoded_pcm": sorted(stimulus_pcm & fit_pcm),
        "normalized_transcript": sorted(stimulus_text & fit_text),
    }
    if any(overlaps.values()):
        raise AudioWorkspaceEvalContractError("fit/stimulus overlap detected")
    return {
        "policy": "reject_waveform_pcm_or_nfkc_casefold_punctuation_whitespace_transcript_overlap",
        "fit_manifest_sha256": fit_manifest_sha256,
        "stimulus_observations": len(observations),
        "fit_rows": len(fit_rows),
        "waveform_overlap_count": 0,
        "decoded_pcm_overlap_count": 0,
        "normalized_transcript_overlap_count": 0,
        "stimulus_waveform_set_sha256": canonical_sha256(sorted(stimulus_wave)),
        "fit_waveform_set_sha256": canonical_sha256(sorted(fit_wave)),
        "stimulus_pcm_set_sha256": canonical_sha256(sorted(stimulus_pcm)),
        "fit_pcm_set_sha256": canonical_sha256(sorted(fit_pcm)),
        "stimulus_transcript_set_sha256": canonical_sha256(sorted(stimulus_text)),
        "fit_transcript_set_sha256": canonical_sha256(sorted(fit_text)),
    }


def frozen_protocol() -> dict[str, Any]:
    """The complete no-choice scientific and inference policy bound by preregistration."""
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinates": {
            "distributions": list(DISTRIBUTIONS),
            "publication_counts": dict(PUBLICATION_COUNTS),
            "expected_items": EXPECTED_ITEM_COUNT,
            "expected_observations": EXPECTED_OBSERVATION_COUNT,
            "coordinates_sha256": EXPECTED_COORDINATES_SHA256,
            "excluded_distributions": list(EXCLUDED_DISTRIBUTIONS),
            "excluded_coordinates": [list(value) for value in EXCLUDED_COORDINATES],
        },
        "tts": {
            "engine": TTS_ENGINE,
            "endpoint": TTS_ENDPOINT,
            "model": TTS_MODEL,
            "response_format": TTS_RESPONSE_FORMAT,
            "input_policy": TTS_INPUT_POLICY,
            "synthesis_policy": TTS_SYNTHESIS_POLICY,
            "variants": list(TTS_VARIANTS),
            "language_mapping": dict(LANGUAGE_CODES),
            "non_multilingual_language": "en-us",
            "variants_are_same_engine_robustness_checks": True,
        },
        "audio_normalization": {
            "source": "mono_pcm16",
            "source_sample_rate": TTS_SAMPLE_RATE,
            "method": "scipy.signal.resample_poly",
            "up": RESAMPLE_UP,
            "down": RESAMPLE_DOWN,
            "finite": "reject",
            "clip": [-1.0, 1.0],
            "target_sample_rate": NORMALIZED_SAMPLE_RATE,
            "target": "mono_pcm16",
        },
        "calibration": {
            "model": WHISPER_MODEL_ID,
            "revision": WHISPER_REVISION,
            "selection": "shortest_non_number_script_by_unicode_codepoints_then_coordinate_index_per_multilingual_language",
            "decoding": {"do_sample": False, "num_beams": 1},
            "normalization": "unicode_nfkc_casefold_remove_unicode_punctuation_collapse_whitespace_serbian_cyrillic_to_gaj_latin",
            "cells": EXPECTED_CALIBRATION_CELL_COUNT,
            "macro_cer_max": CALIBRATION_MACRO_CER_MAX,
            "cell_cer_max": CALIBRATION_CELL_CER_MAX,
        },
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "attention": "eager"},
        "positions": list(POSITIONS),
        "layers": {
            "residual": list(RESIDUAL_LAYERS),
            "lens": list(LENS_LAYERS),
            "actual_output_layer": 34,
        },
        "regions": {key: list(value) for key, value in REGIONS.items()},
        "rank": {
            "definition": "1_plus_count_strictly_greater_than_best_allowed_target",
            "ties": "optimistic",
            "vocabulary": "full",
            "nonfinite": "reject",
            "variants": list(RANK_VARIANTS),
            "allowed_forms": {key: list(value) for key, value in ORDER_OP_SYNONYMS.items()},
        },
        "reducer": {
            "primary": [
                "minimum_allowed_form_rank_at_item_layer",
                "per_item_per_layer_pass_at_k_and_log_k_auc",
                "equal_mean_layers_in_fixed_region",
                "equal_mean_two_tts_variants_within_item",
                "equal_mean_items_within_distribution",
                "equal_mean_distributions",
            ],
            "historical_secondary": "minimum_rank_over_region_non_adjudicating",
            "ks": list(KS),
            "auc": "normalized_trapezoid_against_natural_log_k",
        },
        "controls": {
            "candidate": "residual_row_vector_matmul_J_transpose",
            "logit": "untransported_residual",
            "transposed": "residual_row_vector_matmul_J",
            "permuted": {
                "source_layers": {
                    str(layer): (layer + CONTROL_SOURCE_ROTATION) % len(LENS_LAYERS)
                    for layer in LENS_LAYERS
                },
                "output_rows": "single_torch_cpu_randperm",
                "output_basis_sha256": CONTROL_OUTPUT_BASIS_SHA256,
                "seed": CONTROL_SEED,
            },
        },
        "statistics": {
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "unit": "original_item_with_both_variants_bundled",
                "interval": [0.025, 0.975],
                "quantile": "linear_type7",
            },
            "label_permutation": {
                "replicates": PERMUTATION_REPLICATES,
                "seed": PERMUTATION_SEED,
                "unit": "same_distribution_item_label_bundle_across_variants_positions_layers",
                "self_assignments": True,
                "p_value": "plus_one_max_stat",
            },
            "label_max_stat_cells": [
                {"scope": "all_five", "position": AUDIO_POSITION},
                {"scope": "all_five", "position": RESPONSE_POSITION},
                {"scope": "non_multilingual", "position": AUDIO_POSITION},
                {"scope": "non_multilingual", "position": RESPONSE_POSITION},
            ],
        },
        "thresholds": {
            "content_delta_min": CONTENT_DELTA_THRESHOLD,
            "label_max_stat_p_max": LABEL_MAX_STAT_P_THRESHOLD,
            "confidence": "paired_95_percent_strict",
        },
        "inference": {
            "batch_size": 1,
            "max_sequence_length": 512,
            "cublas_workspace_config": ":4096:8",
            "tf32": False,
            "deterministic_algorithms": True,
            "dropout": False,
            "sampling": False,
            "item_order": "EXPECTED_COORDINATES_then_m1_f1",
            "partial_scores": False,
            "restart_from_item_zero": True,
        },
        "claims": {
            "formal_j_space": False,
            "causal_mediation": False,
            "natural_speech_generalization": False,
            "global_workspace": False,
            "selectivity": False,
            "consciousness": False,
            "universal_layer_boundary": False,
            "alternate_band_search": False,
        },
        "allowed_statuses": list(SCIENTIFIC_STATUSES),
    }


def control_source_layer(layer: int) -> int:
    if isinstance(layer, bool) or not isinstance(layer, int) or layer not in LENS_LAYERS:
        raise AudioWorkspaceEvalContractError("control source layer is invalid")
    return (layer + CONTROL_SOURCE_ROTATION) % len(LENS_LAYERS)


def control_output_basis(d_model: int = D_MODEL) -> list[int]:
    """Reproduce the single seeded CPU torch.randperm output-row control."""
    if d_model != D_MODEL:
        raise AudioWorkspaceEvalContractError("control dimension is not frozen")
    try:
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(CONTROL_SEED)
        return [int(value) for value in torch.randperm(d_model, generator=generator).tolist()]
    except Exception as exc:
        raise AudioWorkspaceEvalContractError(
            "cannot construct frozen control permutation"
        ) from exc


def control_identity() -> dict[str, Any]:
    output_basis = control_output_basis()
    identity = {
        "source_layers": {str(layer): control_source_layer(layer) for layer in LENS_LAYERS},
        "output_basis": output_basis,
        "output_basis_sha256": canonical_sha256(output_basis),
        "seed": CONTROL_SEED,
    }
    if identity["output_basis_sha256"] != CONTROL_OUTPUT_BASIS_SHA256:
        raise AudioWorkspaceEvalContractError("frozen torch control permutation changed")
    return identity


def full_vocabulary_rank(logits: Sequence[float], allowed_token_ids: Sequence[int]) -> int:
    """Exact optimistic full-vocabulary rank of the best allowed form."""
    if (
        isinstance(logits, (str, bytes))
        or isinstance(allowed_token_ids, (str, bytes))
        or any(isinstance(value, bool) for value in logits)
    ):
        raise AudioWorkspaceEvalContractError("full-vocabulary rank inputs are invalid")
    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError) as exc:
        raise AudioWorkspaceEvalContractError("full-vocabulary logits are invalid") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise AudioWorkspaceEvalContractError("full-vocabulary logits are empty or nonfinite")
    ids = list(allowed_token_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= len(values)
            for token in ids
        )
    ):
        raise AudioWorkspaceEvalContractError("allowed token IDs are invalid")
    best = max(values[token] for token in ids)
    return 1 + sum(value > best for value in values)


def log_k_auc(pass_at_k: Mapping[int | str, float]) -> float:
    if not isinstance(pass_at_k, Mapping) or set(pass_at_k) not in (
        set(KS),
        {str(k) for k in KS},
    ):
        raise AudioWorkspaceEvalContractError("pass@k curve must use exactly the frozen k grid")
    try:
        values = [float(pass_at_k[k] if k in pass_at_k else pass_at_k[str(k)]) for k in KS]
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioWorkspaceEvalContractError("pass@k curve is incomplete") from exc
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values) or any(
        left > right for left, right in zip(values, values[1:])
    ):
        raise AudioWorkspaceEvalContractError("pass@k values must be finite monotone probabilities")
    xs = [math.log(k) for k in KS]
    area = sum(
        (xs[index + 1] - xs[index]) * (values[index] + values[index + 1]) / 2.0
        for index in range(len(xs) - 1)
    )
    return area / (xs[-1] - xs[0])


def _curve_from_ranks(ranks: Sequence[int]) -> dict[str, float]:
    if not ranks:
        raise AudioWorkspaceEvalContractError("a pass@k curve needs at least one rank")
    return {str(k): sum(rank <= k for rank in ranks) / len(ranks) for k in KS}


def _mean_curves(curves: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not curves:
        raise AudioWorkspaceEvalContractError("cannot average zero curves")
    return {str(k): sum(float(curve[str(k)]) for curve in curves) / len(curves) for k in KS}


def _curve_summary(curve: Mapping[str, float]) -> dict[str, Any]:
    canonical_curve = {str(k): float(curve[str(k)]) for k in KS}
    return {"pass_at_k": canonical_curve, "log_k_auc": log_k_auc(canonical_curve)}


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if (
        not ordered
        or not 0.0 <= probability <= 1.0
        or any(not math.isfinite(value) for value in ordered)
    ):
        raise AudioWorkspaceEvalContractError("invalid quantile values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rank_map(
    value: Any, expected_keys: set[str], vocabulary_size: int, label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise AudioWorkspaceEvalContractError(f"{label} concept identity changed")
    for concept_id, rank in value.items():
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= vocabulary_size:
            raise AudioWorkspaceEvalContractError(f"{label}.{concept_id} rank is invalid")
    return value


def validate_score_records(
    records: list[Mapping[str, Any]],
    *,
    eligibility: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Validate every item-variant record and L0--L34 evidence coordinate."""
    if not isinstance(records, list) or len(records) != EXPECTED_OBSERVATION_COUNT:
        raise AudioWorkspaceEvalContractError(
            f"score records must contain exactly {EXPECTED_OBSERVATION_COUNT} observations"
        )
    preparation_records = None
    if runtime_identity is not None:
        runtime = _validate_preregistration_runtime(runtime_identity)
        preparation_records = runtime["processor_preparation"]["observations"]
    expected_order = [
        (distribution, name, variant)
        for distribution, name in EXPECTED_COORDINATES
        for variant in TTS_VARIANTS
    ]
    actual_order = [
        (record.get("distribution"), record.get("name"), record.get("variant"))
        if isinstance(record, Mapping)
        else (None, None, None)
        for record in records
    ]
    if actual_order != expected_order:
        raise AudioWorkspaceEvalContractError(
            "score records are not in the frozen item-major m1/f1 order"
        )
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    concept_ids_by_item: dict[tuple[str, str], tuple[str, ...]] = {}
    included_by_item: dict[tuple[str, str], bool] = {}
    concept_owner: dict[str, tuple[str, str]] = {}
    vocabularies: set[int] = set()
    for record_index, record in enumerate(records):
        _require_exact_keys(
            record,
            {
                "distribution",
                "name",
                "variant",
                "included_in_metrics",
                "eligible_concept_ids",
                "vocabulary_size",
                "positions",
                "actual_output",
            },
            "score record",
        )
        distribution = record["distribution"]
        name = record["name"]
        variant = record["variant"]
        key = (distribution, name, variant)
        if key in by_key:
            raise AudioWorkspaceEvalContractError("duplicate score-record coordinate")
        by_key[key] = record
        if (distribution, name) not in EXPECTED_COORDINATES or variant not in TTS_VARIANTS:
            raise AudioWorkspaceEvalContractError("extra score-record coordinate")
        included = record["included_in_metrics"]
        concept_ids = record["eligible_concept_ids"]
        if (
            not isinstance(included, bool)
            or not isinstance(concept_ids, list)
            or len(set(concept_ids)) != len(concept_ids)
            or not all(isinstance(value, str) and value for value in concept_ids)
        ):
            raise AudioWorkspaceEvalContractError("eligible concept IDs are invalid")
        if included != bool(concept_ids):
            raise AudioWorkspaceEvalContractError(
                "included-in-metrics disagrees with eligible concepts"
            )
        item_key = (distribution, name)
        prior_ids = concept_ids_by_item.setdefault(item_key, tuple(concept_ids))
        prior_included = included_by_item.setdefault(item_key, included)
        if prior_ids != tuple(concept_ids) or prior_included != included:
            raise AudioWorkspaceEvalContractError("TTS variants disagree on item eligibility")
        for concept_id in concept_ids:
            owner = concept_owner.setdefault(concept_id, item_key)
            if owner != item_key:
                raise AudioWorkspaceEvalContractError("concept ID is reused by different items")
        vocabulary_size = record["vocabulary_size"]
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= max(KS)
        ):
            raise AudioWorkspaceEvalContractError("vocabulary size is invalid")
        vocabularies.add(vocabulary_size)
        positions = _require_exact_keys(record["positions"], set(POSITIONS), "record positions")
        position_indices: dict[str, int] = {}
        for position_name in POSITIONS:
            position = _require_exact_keys(
                positions[position_name], {"index", "layers"}, f"{position_name} record"
            )
            position_index = position["index"]
            if (
                isinstance(position_index, bool)
                or not isinstance(position_index, int)
                or position_index < 0
            ):
                raise AudioWorkspaceEvalContractError("position index is invalid")
            position_indices[position_name] = position_index
            layers = position["layers"]
            if not isinstance(layers, Mapping) or set(layers) != {
                str(layer) for layer in LENS_LAYERS
            }:
                raise AudioWorkspaceEvalContractError(f"{position_name} must contain L0-L33")
        if position_indices[AUDIO_POSITION] >= position_indices[RESPONSE_POSITION]:
            raise AudioWorkspaceEvalContractError(
                "audio position must precede the response position"
            )
        if preparation_records is not None:
            prepared = preparation_records[record_index]
            if (
                position_indices[AUDIO_POSITION] != prepared[AUDIO_POSITION]
                or position_indices[RESPONSE_POSITION] != prepared[RESPONSE_POSITION]
            ):
                raise AudioWorkspaceEvalContractError(
                    "score positions differ from preregistered processor framing"
                )
        actual = _require_exact_keys(
            record["actual_output"],
            {"layer", "position", "position_index", "token_id"},
            "L34 actual-output evidence",
        )
        if (
            actual["layer"] != 34
            or actual["position"] != RESPONSE_POSITION
            or actual["position_index"] != position_indices[RESPONSE_POSITION]
            or isinstance(actual["token_id"], bool)
            or not isinstance(actual["token_id"], int)
            or not 0 <= actual["token_id"] < vocabulary_size
        ):
            raise AudioWorkspaceEvalContractError("L34 actual-output evidence is invalid")
    expected_keys = {
        (distribution, name, variant)
        for distribution, name in EXPECTED_COORDINATES
        for variant in TTS_VARIANTS
    }
    if set(by_key) != expected_keys:
        raise AudioWorkspaceEvalContractError("score records have missing or extra coordinates")
    if len(vocabularies) != 1:
        raise AudioWorkspaceEvalContractError("score records disagree on vocabulary size")
    vocabulary_size = next(iter(vocabularies))
    pools: dict[str, set[str]] = {distribution: set() for distribution in DISTRIBUTIONS}
    for (distribution, _), concept_ids in concept_ids_by_item.items():
        pools[distribution].update(concept_ids)
    if any(not pool for pool in pools.values()):
        raise AudioWorkspaceEvalContractError("a distribution has no eligible label pool")
    for record in records:
        own = set(record["eligible_concept_ids"])
        pool = pools[str(record["distribution"])]
        for position_name in POSITIONS:
            for layer_name, layer in record["positions"][position_name]["layers"].items():
                expected_layer_keys = {"concept_ranks", "candidate_label_pool_ranks"}
                if position_name == RESPONSE_POSITION:
                    expected_layer_keys.add("motor")
                _require_exact_keys(layer, expected_layer_keys, f"{position_name}/L{layer_name}")
                variants = _require_exact_keys(
                    layer["concept_ranks"], set(RANK_VARIANTS), "rank variants"
                )
                for rank_variant in RANK_VARIANTS:
                    _rank_map(variants[rank_variant], own, vocabulary_size, f"{rank_variant} ranks")
                pool_ranks = _rank_map(
                    layer["candidate_label_pool_ranks"],
                    pool,
                    vocabulary_size,
                    "candidate label-pool ranks",
                )
                for concept_id in own:
                    if pool_ranks[concept_id] != variants["candidate"][concept_id]:
                        raise AudioWorkspaceEvalContractError(
                            "candidate own-label pool rank identity changed"
                        )
                if position_name == RESPONSE_POSITION:
                    motor = _require_exact_keys(
                        layer["motor"],
                        {
                            "actual_token_id",
                            "actual_token_ranks",
                            "candidate_logit_top1_agreement",
                            "candidate_logit_js_nats",
                        },
                        "motor evidence",
                    )
                    if motor["actual_token_id"] != record["actual_output"]["token_id"]:
                        raise AudioWorkspaceEvalContractError(
                            "motor ranks are not bound to the L34 actual token"
                        )
                    _rank_map(
                        motor["actual_token_ranks"],
                        {"candidate", "logit"},
                        vocabulary_size,
                        "actual-token",
                    )
                    if not isinstance(motor["candidate_logit_top1_agreement"], bool):
                        raise AudioWorkspaceEvalContractError("motor top-1 agreement is invalid")
                    divergence = motor["candidate_logit_js_nats"]
                    if (
                        isinstance(divergence, bool)
                        or not isinstance(divergence, (int, float))
                        or not math.isfinite(float(divergence))
                        or not 0.0
                        <= float(divergence)
                        <= JS_DIVERGENCE_MAX_NATS + JS_DIVERGENCE_TOLERANCE
                    ):
                        raise AudioWorkspaceEvalContractError("motor JS evidence is invalid")
    if eligibility is not None:
        frozen_eligibility = _validate_eligibility(eligibility)
        eligibility_by_item = {
            (item["distribution"], item["name"]): item for item in frozen_eligibility["items"]
        }
        for item_key, concept_ids in concept_ids_by_item.items():
            expected = eligibility_by_item[item_key]
            token_ids = [
                token_id
                for concept in expected["concepts"]
                for token_id in concept["allowed_token_ids"]
            ]
            if (
                tuple(expected["eligible_concept_ids"]) != concept_ids
                or expected["included_in_metrics"] != included_by_item[item_key]
                or any(token_id >= vocabulary_size for token_id in token_ids)
            ):
                raise AudioWorkspaceEvalContractError(
                    "score records disagree with preregistered eligibility"
                )
    return _detached_json(list(records), "score records")


def _semantic_item_curve(
    record: Mapping[str, Any],
    position: str,
    rank_variant: str,
    layers: Sequence[int],
    *,
    assigned_concept_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    if not record["included_in_metrics"]:
        raise AudioWorkspaceEvalContractError("excluded item entered a semantic metric")
    concept_ids = (
        list(assigned_concept_ids)
        if assigned_concept_ids is not None
        else list(record["eligible_concept_ids"])
    )
    if not concept_ids:
        raise AudioWorkspaceEvalContractError("semantic item has no labels")
    curves = []
    for layer in layers:
        layer_record = record["positions"][position]["layers"][str(layer)]
        ranks = (
            layer_record["candidate_label_pool_ranks"]
            if assigned_concept_ids is not None
            else layer_record["concept_ranks"][rank_variant]
        )
        try:
            curves.append(_curve_from_ranks([int(ranks[concept_id]) for concept_id in concept_ids]))
        except KeyError as exc:
            raise AudioWorkspaceEvalContractError(
                "assigned label is absent from the rank pool"
            ) from exc
    return _mean_curves(curves)


def fair_item_curve(
    record: Mapping[str, Any],
    *,
    position: str,
    rank_variant: str,
    layers: Sequence[int],
) -> dict[str, float]:
    """Expose the layer-count-fair leaf reducer for scoring and golden tests."""
    declared_layers = tuple(layers)
    if (
        position not in POSITIONS
        or rank_variant not in RANK_VARIANTS
        or not declared_layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer not in LENS_LAYERS
            for layer in declared_layers
        )
    ):
        raise AudioWorkspaceEvalContractError("fair item curve requested invalid coordinates")
    return _semantic_item_curve(
        record,
        position,
        rank_variant,
        declared_layers,
    )


def _historical_item_curve(
    record: Mapping[str, Any], position: str, rank_variant: str, layers: Sequence[int]
) -> dict[str, float]:
    concept_ids = list(record["eligible_concept_ids"])
    minima = [
        min(
            int(
                record["positions"][position]["layers"][str(layer)]["concept_ranks"][rank_variant][
                    concept_id
                ]
            )
            for layer in layers
        )
        for concept_id in concept_ids
    ]
    return _curve_from_ranks(minima)


def _group_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        grouped[(str(record["distribution"]), str(record["name"]))][str(record["variant"])] = record
    return grouped


def _semantic_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    position: str,
    region: str,
    rank_variant: str,
    distributions: Sequence[str],
    historical: bool,
) -> dict[str, Any]:
    if (
        position not in POSITIONS
        or region not in REGIONS
        or rank_variant not in RANK_VARIANTS
        or not distributions
        or any(value not in DISTRIBUTIONS for value in distributions)
    ):
        raise AudioWorkspaceEvalContractError("semantic summary requested an unfrozen choice")
    grouped = _group_records(records)
    distribution_results: dict[str, Any] = {}
    global_variant_curves: dict[str, list[Mapping[str, float]]] = {
        variant: [] for variant in TTS_VARIANTS
    }
    global_item_curves: list[Mapping[str, float]] = []
    for distribution in distributions:
        item_variant_curves: dict[str, list[Mapping[str, float]]] = {
            variant: [] for variant in TTS_VARIANTS
        }
        bundled_curves = []
        for (item_distribution, _), variants in grouped.items():
            if (
                item_distribution != distribution
                or not variants[TTS_VARIANTS[0]]["included_in_metrics"]
            ):
                continue
            curves = {}
            for variant in TTS_VARIANTS:
                record = variants[variant]
                curves[variant] = (
                    _historical_item_curve(record, position, rank_variant, REGIONS[region])
                    if historical
                    else _semantic_item_curve(record, position, rank_variant, REGIONS[region])
                )
                item_variant_curves[variant].append(curves[variant])
            bundled_curves.append(_mean_curves([curves[variant] for variant in TTS_VARIANTS]))
        if not bundled_curves:
            raise AudioWorkspaceEvalContractError(f"{distribution}: no eligible semantic items")
        by_variant = {
            variant: _curve_summary(_mean_curves(item_variant_curves[variant]))
            for variant in TTS_VARIANTS
        }
        aggregate = _curve_summary(_mean_curves(bundled_curves))
        distribution_results[distribution] = {
            "n_items": len(bundled_curves),
            "by_variant": by_variant,
            "aggregate": aggregate,
        }
        for variant in TTS_VARIANTS:
            global_variant_curves[variant].append(by_variant[variant]["pass_at_k"])
        global_item_curves.append(aggregate["pass_at_k"])
    return {
        "position": position,
        "region": region,
        "rank_variant": rank_variant,
        "layer_reducer": "minimum_over_region" if historical else "equal_mean_per_layer",
        "variant_reducer": "equal_mean_within_item",
        "item_reducer": "equal_mean_within_distribution",
        "distribution_reducer": "equal_mean",
        "distributions": distribution_results,
        "by_variant": {
            variant: _curve_summary(_mean_curves(global_variant_curves[variant]))
            for variant in TTS_VARIANTS
        },
        "aggregate": _curve_summary(_mean_curves(global_item_curves)),
    }


def semantic_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    position: str,
    region: str,
    rank_variant: str = "candidate",
) -> dict[str, Any]:
    return _semantic_summary(
        records,
        position=position,
        region=region,
        rank_variant=rank_variant,
        distributions=DISTRIBUTIONS,
        historical=False,
    )


def historical_semantic_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    position: str,
    region: str,
    rank_variant: str = "candidate",
) -> dict[str, Any]:
    return _semantic_summary(
        records,
        position=position,
        region=region,
        rank_variant=rank_variant,
        distributions=DISTRIBUTIONS,
        historical=True,
    )


def _motor_item_values(
    record: Mapping[str, Any], region: str
) -> tuple[dict[str, dict[str, float]], float]:
    agreement_curves = {variant: [] for variant in ("candidate", "logit")}
    js_values = []
    for layer in REGIONS[region]:
        motor = record["positions"][RESPONSE_POSITION]["layers"][str(layer)]["motor"]
        for variant in agreement_curves:
            agreement_curves[variant].append(
                _curve_from_ranks([int(motor["actual_token_ranks"][variant])])
            )
        js_values.append(float(motor["candidate_logit_js_nats"]))
    return (
        {variant: _mean_curves(curves) for variant, curves in agreement_curves.items()},
        sum(js_values) / len(js_values),
    )


def motor_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    region: str,
) -> dict[str, Any]:
    if region not in REGIONS:
        raise AudioWorkspaceEvalContractError("motor summary requested an unfrozen region")
    grouped = _group_records(records)
    distribution_results = {}
    aggregate_curves = {variant: [] for variant in ("candidate", "logit")}
    aggregate_js = []
    for distribution in DISTRIBUTIONS:
        bundled_curves = {variant: [] for variant in ("candidate", "logit")}
        bundled_js = []
        n_items = 0
        for (item_distribution, _), variants in grouped.items():
            if item_distribution != distribution:
                continue
            per_voice = [_motor_item_values(variants[voice], region) for voice in TTS_VARIANTS]
            for rank_variant in bundled_curves:
                bundled_curves[rank_variant].append(
                    _mean_curves([value[0][rank_variant] for value in per_voice])
                )
            bundled_js.append(sum(value[1] for value in per_voice) / len(per_voice))
            n_items += 1
        if n_items == 0:
            raise AudioWorkspaceEvalContractError(f"{distribution}: no motor items")
        agreement = {
            variant: _curve_summary(_mean_curves(curves))
            for variant, curves in bundled_curves.items()
        }
        mean_js = sum(bundled_js) / len(bundled_js)
        distribution_results[distribution] = {
            "n_items": n_items,
            "item_filter": "none",
            "actual_token_agreement": agreement,
            "candidate_logit_js_nats": mean_js,
        }
        for variant in aggregate_curves:
            aggregate_curves[variant].append(agreement[variant]["pass_at_k"])
        aggregate_js.append(mean_js)
    return {
        "region": region,
        "position": RESPONSE_POSITION,
        "reference": "L34_unmodified_actual_next_token_argmax",
        "item_filter": "none",
        "distributions": distribution_results,
        "actual_token_agreement": {
            variant: _curve_summary(_mean_curves(curves))
            for variant, curves in aggregate_curves.items()
        },
        "candidate_logit_js_nats": sum(aggregate_js) / len(aggregate_js),
    }


def build_summaries(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validated = validate_score_records(list(records))
    primary = {
        position: {
            region: {
                rank_variant: semantic_summary(
                    validated, position=position, region=region, rank_variant=rank_variant
                )
                for rank_variant in RANK_VARIANTS
            }
            for region in REGIONS
        }
        for position in POSITIONS
    }
    historical = {
        position: {
            region: {
                rank_variant: historical_semantic_summary(
                    validated, position=position, region=region, rank_variant=rank_variant
                )
                for rank_variant in RANK_VARIANTS
            }
            for region in REGIONS
        }
        for position in POSITIONS
    }
    return {
        "primary": primary,
        "historical_secondary_non_adjudicating": historical,
        "motor": {region: motor_summary(validated, region=region) for region in REGIONS},
    }


def _item_scalar(record: Mapping[str, Any], position: str, region: str, rank_variant: str) -> float:
    return log_k_auc(_semantic_item_curve(record, position, rank_variant, REGIONS[region]))


def _motor_item_scalar(record: Mapping[str, Any], region: str, metric: str) -> float:
    curves, js = _motor_item_values(record, region)
    if metric == "agreement":
        return log_k_auc(curves["candidate"])
    if metric == "js":
        return js
    raise AudioWorkspaceEvalContractError("unknown motor item metric")


type SemanticMetricKey = tuple[str, str, str]
type MotorMetricKey = tuple[str, str]
type ContrastPair = (
    tuple[SemanticMetricKey, SemanticMetricKey] | tuple[MotorMetricKey, MotorMetricKey]
)


def _contrast_specs() -> dict[str, ContrastPair]:
    specs: dict[str, ContrastPair] = {}
    for scope, positions in (("all_five", POSITIONS), ("non_multilingual", (RESPONSE_POSITION,))):
        for position in positions:
            prefix = f"{scope}.{position}."
            specs[prefix + "candidate_minus_logit"] = (
                (position, "candidate_l13_l31", "candidate"),
                (position, "candidate_l13_l31", "logit"),
            )
            specs[prefix + "candidate_minus_early"] = (
                (position, "candidate_l13_l31", "candidate"),
                (position, "early_l0_l12", "candidate"),
            )
            specs[prefix + "candidate_minus_motor"] = (
                (position, "candidate_l13_l31", "candidate"),
                (position, "motor_l32_l33", "candidate"),
            )
            if scope == "all_five":
                specs[prefix + "candidate_minus_transposed"] = (
                    (position, "candidate_l13_l31", "candidate"),
                    (position, "candidate_l13_l31", "transposed"),
                )
                specs[prefix + "candidate_minus_permuted"] = (
                    (position, "candidate_l13_l31", "candidate"),
                    (position, "candidate_l13_l31", "permuted"),
                )
    specs[f"all_five.{RESPONSE_POSITION}.motor_agreement_motor_minus_candidate"] = (
        ("motor_l32_l33", "agreement"),
        ("candidate_l13_l31", "agreement"),
    )
    specs[f"all_five.{RESPONSE_POSITION}.motor_js_motor_minus_candidate"] = (
        ("motor_l32_l33", "js"),
        ("candidate_l13_l31", "js"),
    )
    return specs


def bundled_item_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired original-item bootstrap with both TTS variants kept together."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise AudioWorkspaceEvalContractError("bootstrap replicates must be positive")
    validated_records = validate_score_records(list(records))
    grouped = _group_records(validated_records)
    metrics_by_item: dict[tuple[str, str], dict[tuple[str, ...], float]] = {}
    for coordinate, voices in grouped.items():
        values: dict[tuple[str, ...], float] = {}
        if voices[TTS_VARIANTS[0]]["included_in_metrics"]:
            for position in POSITIONS:
                for region in REGIONS:
                    for rank_variant in RANK_VARIANTS:
                        values[(position, region, rank_variant)] = sum(
                            _item_scalar(voices[voice], position, region, rank_variant)
                            for voice in TTS_VARIANTS
                        ) / len(TTS_VARIANTS)
        for region in REGIONS:
            for metric in ("agreement", "js"):
                values[(region, metric)] = sum(
                    _motor_item_scalar(voices[voice], region, metric) for voice in TTS_VARIANTS
                ) / len(TTS_VARIANTS)
        metrics_by_item[coordinate] = values
    eligible_by_distribution = {
        distribution: [
            coordinate
            for coordinate in EXPECTED_COORDINATES
            if coordinate[0] == distribution
            and grouped[coordinate][TTS_VARIANTS[0]]["included_in_metrics"]
        ]
        for distribution in DISTRIBUTIONS
    }
    all_by_distribution = {
        distribution: [
            coordinate for coordinate in EXPECTED_COORDINATES if coordinate[0] == distribution
        ]
        for distribution in DISTRIBUTIONS
    }
    specs = _contrast_specs()

    def contrast_value(
        spec: Any, sample: Mapping[str, Sequence[tuple[str, str]]], scope: str
    ) -> float:
        distributions = DISTRIBUTIONS if scope == "all_five" else NON_MULTILINGUAL_DISTRIBUTIONS
        left, right = spec
        per_distribution = []
        for distribution in distributions:
            coordinates = sample[distribution]
            per_distribution.append(
                sum(
                    metrics_by_item[coordinate][left] - metrics_by_item[coordinate][right]
                    for coordinate in coordinates
                )
                / len(coordinates)
            )
        return sum(per_distribution) / len(per_distribution)

    point_samples = {
        distribution: eligible_by_distribution[distribution] for distribution in DISTRIBUTIONS
    }
    motor_point_samples = {
        distribution: all_by_distribution[distribution] for distribution in DISTRIBUTIONS
    }
    arrays = {name: [] for name in specs}
    rng = random.Random(seed)
    for _ in range(replicates):
        semantic_sample = {
            distribution: [values[rng.randrange(len(values))] for _ in values]
            for distribution, values in eligible_by_distribution.items()
        }
        motor_sample = {
            distribution: [values[rng.randrange(len(values))] for _ in values]
            for distribution, values in all_by_distribution.items()
        }
        for name, spec in specs.items():
            scope = name.split(".", 1)[0]
            sample = motor_sample if ".motor_" in name else semantic_sample
            arrays[name].append(contrast_value(spec, sample, scope))
    contrasts = {}
    for name, values in arrays.items():
        scope = name.split(".", 1)[0]
        point_sample = motor_point_samples if ".motor_" in name else point_samples
        point = contrast_value(specs[name], point_sample, scope)
        contrasts[name] = {
            "point": point,
            "lower_95": _linear_quantile(values, 0.025),
            "median": _linear_quantile(values, 0.5),
            "upper_95": _linear_quantile(values, 0.975),
            "replicates": values,
            "replicate_sha256": canonical_sha256(values),
        }
    return {
        "method": "paired_original_item_percentile_bootstrap_both_variants_bundled_full_reducer",
        "seed": seed,
        "replicates": replicates,
        "quantile": "linear_type7",
        "contrasts": contrasts,
    }


LABEL_MAX_STAT_CELLS = (
    f"all_five.{AUDIO_POSITION}",
    f"all_five.{RESPONSE_POSITION}",
    f"non_multilingual.{AUDIO_POSITION}",
    f"non_multilingual.{RESPONSE_POSITION}",
)


def label_permutation_max_stat(
    records: Sequence[Mapping[str, Any]],
    *,
    replicates: int = PERMUTATION_REPLICATES,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Same-distribution item-label permutations and the exact four-cell max null."""
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise AudioWorkspaceEvalContractError("permutation replicates must be positive")
    validated_records = validate_score_records(list(records))
    grouped = _group_records(validated_records)
    active = {
        distribution: [
            coordinate
            for coordinate in EXPECTED_COORDINATES
            if coordinate[0] == distribution
            and grouped[coordinate][TTS_VARIANTS[0]]["included_in_metrics"]
        ]
        for distribution in DISTRIBUTIONS
    }
    if any(len(values) < 2 for values in active.values()):
        raise AudioWorkspaceEvalContractError(
            "label permutation needs two eligible items per distribution"
        )

    # The complete reducer is linear outside the assigned-label item score.
    # Precompute every target-item/source-bundle score once, then each replicate
    # only applies its same-distribution bundled permutation to these matrices.
    matrices: dict[str, dict[str, list[list[float]]]] = {}
    for distribution in DISTRIBUTIONS:
        coordinates = active[distribution]
        bundles = [
            list(grouped[coordinate][TTS_VARIANTS[0]]["eligible_concept_ids"])
            for coordinate in coordinates
        ]
        matrices[distribution] = {}
        for position in POSITIONS:
            target_rows = []
            for target_coordinate in coordinates:
                voices = grouped[target_coordinate]
                target_rows.append(
                    [
                        sum(
                            log_k_auc(
                                _semantic_item_curve(
                                    voices[voice],
                                    position,
                                    "candidate",
                                    CANDIDATE_LAYERS,
                                    assigned_concept_ids=bundle,
                                )
                            )
                            for voice in TTS_VARIANTS
                        )
                        / len(TTS_VARIANTS)
                        for bundle in bundles
                    ]
                )
            matrices[distribution][position] = target_rows

    def aggregate_cells(
        assignments: Mapping[str, Sequence[int]],
    ) -> dict[str, float]:
        per_distribution = {
            position: {
                distribution: sum(
                    matrices[distribution][position][target_index][source_index]
                    for target_index, source_index in enumerate(assignments[distribution])
                )
                / len(active[distribution])
                for distribution in DISTRIBUTIONS
            }
            for position in POSITIONS
        }
        values = {
            f"all_five.{position}": sum(
                per_distribution[position][distribution] for distribution in DISTRIBUTIONS
            )
            / len(DISTRIBUTIONS)
            for position in POSITIONS
        }
        values.update(
            {
                f"non_multilingual.{position}": sum(
                    per_distribution[position][distribution]
                    for distribution in NON_MULTILINGUAL_DISTRIBUTIONS
                )
                / len(NON_MULTILINGUAL_DISTRIBUTIONS)
                for position in POSITIONS
            }
        )
        if tuple(values) != LABEL_MAX_STAT_CELLS:
            raise AudioWorkspaceEvalContractError("label max-stat family changed")
        return values

    identity_assignments = {
        distribution: list(range(len(active[distribution]))) for distribution in DISTRIBUTIONS
    }
    observed = aggregate_cells(identity_assignments)
    cell_null = {cell: [] for cell in LABEL_MAX_STAT_CELLS}
    max_null = []
    effective = 0
    rng = random.Random(seed)
    for _ in range(replicates):
        assignments = {}
        changed = False
        for distribution in DISTRIBUTIONS:
            assigned = list(identity_assignments[distribution])
            rng.shuffle(assigned)
            changed = changed or assigned != identity_assignments[distribution]
            assignments[distribution] = assigned
        effective += int(changed)
        values = aggregate_cells(assignments)
        for cell, value in values.items():
            cell_null[cell].append(value)
        max_null.append(max(values.values()))
    return {
        "method": "same_distribution_label_bundle_permutation_both_variants_positions_layers",
        "seed": seed,
        "replicates": replicates,
        "self_assignments_allowed": True,
        "cells": list(LABEL_MAX_STAT_CELLS),
        "observed": observed,
        "cell_null": cell_null,
        "cell_null_sha256": {cell: canonical_sha256(values) for cell, values in cell_null.items()},
        "max_null": max_null,
        "max_null_sha256": canonical_sha256(max_null),
        "effective_replicates": effective,
        "plus_one_p_values": {
            cell: (1 + sum(null >= observed[cell] for null in max_null)) / (replicates + 1)
            for cell in LABEL_MAX_STAT_CELLS
        },
    }


def _auc(
    summary: Mapping[str, Any],
    position: str,
    region: str,
    rank_variant: str,
    *,
    scope: str = "all_five",
) -> float:
    if scope == "all_five":
        return float(summary["primary"][position][region][rank_variant]["aggregate"]["log_k_auc"])
    distributions = summary["primary"][position][region][rank_variant]["distributions"]
    return sum(
        float(distributions[value]["aggregate"]["log_k_auc"])
        for value in NON_MULTILINGUAL_DISTRIBUTIONS
    ) / len(NON_MULTILINGUAL_DISTRIBUTIONS)


def build_evidence(
    summaries: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    permutation: Mapping[str, Any],
    common_item_text_summary: Mapping[str, Any],
) -> dict[str, Any]:
    contrasts = bootstrap["contrasts"]
    content_key = f"all_five.{AUDIO_POSITION}.candidate_minus_logit"
    candidate = summaries["primary"][AUDIO_POSITION]["candidate_l13_l31"]["candidate"]
    logit = summaries["primary"][AUDIO_POSITION]["candidate_l13_l31"]["logit"]
    distribution_deltas = {
        distribution: float(candidate["distributions"][distribution]["aggregate"]["log_k_auc"])
        - float(logit["distributions"][distribution]["aggregate"]["log_k_auc"])
        for distribution in DISTRIBUTIONS
    }
    variant_deltas = {
        variant: float(candidate["by_variant"][variant]["log_k_auc"])
        - float(logit["by_variant"][variant]["log_k_auc"])
        for variant in TTS_VARIANTS
    }
    return {
        "semantic_vs_logit": {
            "cell": f"all_five.{AUDIO_POSITION}",
            "candidate_minus_logit": contrasts[content_key],
            "label_max_stat_plus_one_p": permutation["plus_one_p_values"][
                f"all_five.{AUDIO_POSITION}"
            ],
            "distribution_point_deltas": distribution_deltas,
            "variant_point_deltas": variant_deltas,
        },
        "structural_controls": {
            "candidate_minus_transposed": contrasts[
                f"all_five.{AUDIO_POSITION}.candidate_minus_transposed"
            ],
            "candidate_minus_permuted": contrasts[
                f"all_five.{AUDIO_POSITION}.candidate_minus_permuted"
            ],
        },
        "fixed_region_localization": {
            "candidate_minus_early": contrasts[f"all_five.{AUDIO_POSITION}.candidate_minus_early"],
            "candidate_minus_motor": contrasts[f"all_five.{AUDIO_POSITION}.candidate_minus_motor"],
            "historical_secondary_non_adjudicating": summaries[
                "historical_secondary_non_adjudicating"
            ][AUDIO_POSITION],
        },
        "response_boundary_corroboration": {
            "scope": "four_non_multilingual_distributions",
            "candidate_minus_logit": contrasts[
                f"non_multilingual.{RESPONSE_POSITION}.candidate_minus_logit"
            ],
            "candidate_minus_early": contrasts[
                f"non_multilingual.{RESPONSE_POSITION}.candidate_minus_early"
            ],
            "candidate_minus_motor": contrasts[
                f"non_multilingual.{RESPONSE_POSITION}.candidate_minus_motor"
            ],
            "label_max_stat_plus_one_p": permutation["plus_one_p_values"][
                f"non_multilingual.{RESPONSE_POSITION}"
            ],
        },
        "motor_transition": {
            "reference": "L34_unmodified_actual_next_token_argmax",
            "motor_agreement_motor_minus_candidate": contrasts[
                f"all_five.{RESPONSE_POSITION}.motor_agreement_motor_minus_candidate"
            ],
            "motor_js_motor_minus_candidate": contrasts[
                f"all_five.{RESPONSE_POSITION}.motor_js_motor_minus_candidate"
            ],
        },
        "common_item_text_comparison": dict(common_item_text_summary),
    }


CRITERIA = (
    "content_candidate_minus_logit_point_at_least_0_02",
    "content_candidate_minus_logit_lower_95_above_zero",
    "content_label_max_stat_p_at_most_0_01",
    "every_distribution_content_point_delta_nonnegative",
    "both_tts_variants_content_point_delta_nonnegative",
    "content_candidate_minus_transposed_lower_95_above_zero",
    "content_candidate_minus_permuted_lower_95_above_zero",
    "content_candidate_minus_early_lower_95_above_zero",
    "content_candidate_minus_motor_lower_95_above_zero",
    "non_multilingual_response_candidate_minus_logit_lower_95_above_zero",
    "non_multilingual_response_candidate_minus_early_lower_95_above_zero",
    "non_multilingual_response_candidate_minus_motor_lower_95_above_zero",
    "response_motor_agreement_lower_95_above_zero",
    "response_motor_js_upper_95_below_zero",
)


def _validate_interval(value: Any, label: str) -> None:
    interval = _require_exact_keys(
        value,
        {
            "point",
            "lower_95",
            "median",
            "upper_95",
            "replicates",
            "replicate_sha256",
        },
        label,
    )
    scalars = [
        interval["point"],
        interval["lower_95"],
        interval["median"],
        interval["upper_95"],
    ]
    replicates = interval["replicates"]
    if (
        any(
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            for scalar in scalars
        )
        or not isinstance(replicates, list)
        or not replicates
        or any(
            isinstance(replicate, bool)
            or not isinstance(replicate, (int, float))
            or not math.isfinite(float(replicate))
            for replicate in replicates
        )
        or not _is_sha256(interval["replicate_sha256"])
        or interval["replicate_sha256"] != canonical_sha256(replicates)
        or float(interval["lower_95"]) > float(interval["median"]) + 1e-15
        or float(interval["median"]) > float(interval["upper_95"]) + 1e-15
    ):
        raise AudioWorkspaceEvalContractError(f"{label} is invalid")


def _validate_evidence(evidence: Any) -> None:
    value = _require_exact_keys(
        evidence,
        {
            "semantic_vs_logit",
            "structural_controls",
            "fixed_region_localization",
            "response_boundary_corroboration",
            "motor_transition",
            "common_item_text_comparison",
        },
        "evidence",
    )
    semantic = _require_exact_keys(
        value["semantic_vs_logit"],
        {
            "cell",
            "candidate_minus_logit",
            "label_max_stat_plus_one_p",
            "distribution_point_deltas",
            "variant_point_deltas",
        },
        "semantic-vs-logit evidence",
    )
    structural = _require_exact_keys(
        value["structural_controls"],
        {"candidate_minus_transposed", "candidate_minus_permuted"},
        "structural-control evidence",
    )
    localization = _require_exact_keys(
        value["fixed_region_localization"],
        {
            "candidate_minus_early",
            "candidate_minus_motor",
            "historical_secondary_non_adjudicating",
        },
        "fixed-region evidence",
    )
    response = _require_exact_keys(
        value["response_boundary_corroboration"],
        {
            "scope",
            "candidate_minus_logit",
            "candidate_minus_early",
            "candidate_minus_motor",
            "label_max_stat_plus_one_p",
        },
        "response-boundary evidence",
    )
    motor = _require_exact_keys(
        value["motor_transition"],
        {
            "reference",
            "motor_agreement_motor_minus_candidate",
            "motor_js_motor_minus_candidate",
        },
        "motor-transition evidence",
    )
    interval_values = (
        (semantic["candidate_minus_logit"], "content candidate-minus-logit"),
        (
            structural["candidate_minus_transposed"],
            "candidate-minus-transposed",
        ),
        (
            structural["candidate_minus_permuted"],
            "candidate-minus-permuted",
        ),
        (localization["candidate_minus_early"], "candidate-minus-early"),
        (localization["candidate_minus_motor"], "candidate-minus-motor"),
        (
            response["candidate_minus_logit"],
            "response candidate-minus-logit",
        ),
        (
            response["candidate_minus_early"],
            "response candidate-minus-early",
        ),
        (
            response["candidate_minus_motor"],
            "response candidate-minus-motor",
        ),
        (
            motor["motor_agreement_motor_minus_candidate"],
            "motor agreement interval",
        ),
        (
            motor["motor_js_motor_minus_candidate"],
            "motor JS interval",
        ),
    )
    for interval, label in interval_values:
        _validate_interval(interval, label)
    distributions = semantic["distribution_point_deltas"]
    variants = semantic["variant_point_deltas"]
    probabilities = (
        semantic["label_max_stat_plus_one_p"],
        response["label_max_stat_plus_one_p"],
    )
    if (
        semantic["cell"] != f"all_five.{AUDIO_POSITION}"
        or response["scope"] != "four_non_multilingual_distributions"
        or motor["reference"] != "L34_unmodified_actual_next_token_argmax"
        or not isinstance(localization["historical_secondary_non_adjudicating"], Mapping)
        or not isinstance(value["common_item_text_comparison"], Mapping)
        or not isinstance(distributions, Mapping)
        or set(distributions) != set(DISTRIBUTIONS)
        or not isinstance(variants, Mapping)
        or set(variants) != set(TTS_VARIANTS)
        or any(
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
            for delta in (*distributions.values(), *variants.values())
        )
        or any(
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
            for probability in probabilities
        )
    ):
        raise AudioWorkspaceEvalContractError("evidence identities or scalar values are invalid")


def adjudicate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _validate_evidence(evidence)
    semantic = evidence["semantic_vs_logit"]
    structural = evidence["structural_controls"]
    localization = evidence["fixed_region_localization"]
    response = evidence["response_boundary_corroboration"]
    motor = evidence["motor_transition"]
    checks = {
        CRITERIA[0]: float(semantic["candidate_minus_logit"]["point"]) >= CONTENT_DELTA_THRESHOLD,
        CRITERIA[1]: float(semantic["candidate_minus_logit"]["lower_95"]) > 0.0,
        CRITERIA[2]: float(semantic["label_max_stat_plus_one_p"]) <= LABEL_MAX_STAT_P_THRESHOLD,
        CRITERIA[3]: all(
            float(value) >= 0.0 for value in semantic["distribution_point_deltas"].values()
        )
        and set(semantic["distribution_point_deltas"]) == set(DISTRIBUTIONS),
        CRITERIA[4]: all(float(value) >= 0.0 for value in semantic["variant_point_deltas"].values())
        and set(semantic["variant_point_deltas"]) == set(TTS_VARIANTS),
        CRITERIA[5]: float(structural["candidate_minus_transposed"]["lower_95"]) > 0.0,
        CRITERIA[6]: float(structural["candidate_minus_permuted"]["lower_95"]) > 0.0,
        CRITERIA[7]: float(localization["candidate_minus_early"]["lower_95"]) > 0.0,
        CRITERIA[8]: float(localization["candidate_minus_motor"]["lower_95"]) > 0.0,
        CRITERIA[9]: float(response["candidate_minus_logit"]["lower_95"]) > 0.0,
        CRITERIA[10]: float(response["candidate_minus_early"]["lower_95"]) > 0.0,
        CRITERIA[11]: float(response["candidate_minus_motor"]["lower_95"]) > 0.0,
        CRITERIA[12]: float(motor["motor_agreement_motor_minus_candidate"]["lower_95"]) > 0.0,
        CRITERIA[13]: float(motor["motor_js_motor_minus_candidate"]["upper_95"]) < 0.0,
    }
    failed = [criterion for criterion in CRITERIA if not checks[criterion]]
    return {
        "status": VALIDATED_STATUS if not failed else NO_READOUT_STATUS,
        "criteria": checks,
        "failed_criteria": failed,
        "searched_alternate_bands": False,
    }


def determine_status(
    *,
    protocol_valid: bool,
    calibration_passed: bool,
    complete: bool,
    evidence: Mapping[str, Any] | None,
) -> str:
    """Apply frozen status precedence; incomplete execution has no scientific status."""
    if (
        not isinstance(protocol_valid, bool)
        or not isinstance(calibration_passed, bool)
        or not isinstance(complete, bool)
    ):
        raise AudioWorkspaceEvalContractError("status gates must be booleans")
    if not complete:
        raise AudioWorkspaceEvalContractError(
            "an interrupted job remains pending and has no scientific status"
        )
    if not protocol_valid:
        return INVALID_PROTOCOL_STATUS
    if not calibration_passed:
        return INCONCLUSIVE_STIMULUS_STATUS
    if evidence is None:
        raise AudioWorkspaceEvalContractError("complete calibrated execution has no evidence")
    return adjudicate(evidence)["status"]


RUNTIME_PACKAGES = (
    "accelerate",
    "jlens",
    "modal",
    "numpy",
    "scipy",
    "soundfile",
    "torch",
    "transformers",
)


def _validate_source_identity(value: Any) -> dict[str, Any]:
    identity = _require_exact_keys(
        value,
        {"git_revision", "source_sha256", "lock_sha256"},
        "source identity",
    )
    if (
        not isinstance(identity["git_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity["git_revision"]) is None
        or not _is_sha256(identity["source_sha256"])
        or not _is_sha256(identity["lock_sha256"])
    ):
        raise AudioWorkspaceEvalContractError("source identity is invalid")
    return _detached_json(dict(identity), "source identity")


def _validate_runtime_environment(value: Any, *, include_cuda: bool) -> dict[str, Any]:
    keys = {"python", "platform", "packages", "modal_image_id"}
    if include_cuda:
        keys.update({"cuda", "device"})
    identity = _require_exact_keys(value, keys, "runtime environment")
    packages = _require_exact_keys(identity["packages"], set(RUNTIME_PACKAGES), "runtime packages")
    scalar_keys = keys - {"packages"}
    if any(not isinstance(identity[key], str) or not identity[key] for key in scalar_keys) or any(
        not isinstance(packages[package], str) or not packages[package]
        for package in RUNTIME_PACKAGES
    ):
        raise AudioWorkspaceEvalContractError("runtime environment values are invalid")
    return _detached_json(dict(identity), "runtime environment")


def _validate_model_input_identities(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise AudioWorkspaceEvalContractError("processor model-input identities changed")
    validated: list[dict[str, Any]] = []
    for identity in value:
        _require_exact_keys(
            identity,
            {"name", "dtype", "shape", "sha256"},
            "processor model-input identity",
        )
        if (
            not isinstance(identity["name"], str)
            or not identity["name"]
            or not isinstance(identity["dtype"], str)
            or not identity["dtype"].startswith("torch.")
            or not isinstance(identity["shape"], list)
            or not identity["shape"]
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in identity["shape"]
            )
            or not _is_sha256(identity["sha256"])
        ):
            raise AudioWorkspaceEvalContractError("processor model-input identity changed")
        validated.append(_detached_json(dict(identity), "processor model-input identity"))
    names = [identity["name"] for identity in validated]
    if names != sorted(set(names)) or not {
        "input_ids",
        "input_features",
        "input_features_mask",
    }.issubset(names):
        raise AudioWorkspaceEvalContractError("processor model-input identities changed")
    return validated


def _validate_processor_preparation(
    value: Any,
    *,
    stimulus_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .audio_eval_model import (
        EVALUATION_PREFIX_FRAMING_IDS,
        EVALUATION_SUFFIX_FRAMING_IDS,
    )

    preparation = validate_seal(value, "preparation_sha256", "processor preparation")
    _require_exact_keys(
        preparation,
        {
            "kind",
            "model",
            "max_sequence_length",
            "observations",
            "preparation_sha256",
        },
        "processor preparation",
    )
    if (
        preparation["kind"] != "audio_workspace_processor_preparation"
        or preparation["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}
        or preparation["max_sequence_length"] != 512
        or not isinstance(preparation["observations"], list)
        or len(preparation["observations"]) != EXPECTED_OBSERVATION_COUNT
    ):
        raise AudioWorkspaceEvalContractError("processor preparation identity changed")
    stimulus_observations = (
        stimulus_manifest["observations"] if stimulus_manifest is not None else None
    )
    expected_keys = {
        "observation_index",
        "distribution",
        "name",
        "variant",
        "normalized_wav_sha256",
        "model_inputs",
        "audio_start",
        "n_audio_tokens",
        "audio_stop",
        "sequence_length",
        "max_sequence_length",
        AUDIO_POSITION,
        RESPONSE_POSITION,
        "prefix_framing_ids",
        "suffix_framing_ids",
    }
    for index, record in enumerate(preparation["observations"]):
        _require_exact_keys(record, expected_keys, "processor preparation observation")
        distribution, name = EXPECTED_COORDINATES[index // len(TTS_VARIANTS)]
        variant = TTS_VARIANTS[index % len(TTS_VARIANTS)]
        integer_keys = (
            "observation_index",
            "audio_start",
            "n_audio_tokens",
            "audio_stop",
            "sequence_length",
            "max_sequence_length",
            AUDIO_POSITION,
            RESPONSE_POSITION,
        )
        model_inputs = _validate_model_input_identities(record["model_inputs"])
        if model_inputs != record["model_inputs"]:
            raise AudioWorkspaceEvalContractError("processor model-input identities changed")
        if (
            record["observation_index"] != index
            or (record["distribution"], record["name"], record["variant"])
            != (distribution, name, variant)
            or not _is_sha256(record["normalized_wav_sha256"])
            or any(
                isinstance(record[key], bool) or not isinstance(record[key], int)
                for key in integer_keys
            )
            or record["audio_start"] < 0
            or record["n_audio_tokens"] <= 0
            or record["audio_stop"] != record["audio_start"] + record["n_audio_tokens"]
            or record[AUDIO_POSITION] != record["audio_stop"] - 1
            or record[RESPONSE_POSITION] != record["sequence_length"] - 1
            or record["audio_stop"] > record[RESPONSE_POSITION]
            or record["audio_start"] != len(EVALUATION_PREFIX_FRAMING_IDS)
            or record["sequence_length"] - record["audio_stop"]
            != len(EVALUATION_SUFFIX_FRAMING_IDS)
            or record["prefix_framing_ids"] != list(EVALUATION_PREFIX_FRAMING_IDS)
            or record["suffix_framing_ids"] != list(EVALUATION_SUFFIX_FRAMING_IDS)
            or record["max_sequence_length"] != 512
            or not 0 < record["sequence_length"] <= 512
            or any(
                not isinstance(record[key], list)
                or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in record[key]
                )
                for key in ("prefix_framing_ids", "suffix_framing_ids")
            )
        ):
            raise AudioWorkspaceEvalContractError(
                "processor preparation observation geometry changed"
            )
        if stimulus_observations is not None:
            stimulus = stimulus_observations[index]
            if record["normalized_wav_sha256"] != stimulus["normalized_wav_sha256"] or (
                record["distribution"],
                record["name"],
                record["variant"],
            ) != (
                stimulus["distribution"],
                stimulus["name"],
                stimulus["variant"],
            ):
                raise AudioWorkspaceEvalContractError(
                    "processor preparation differs from sealed stimulus"
                )
    return preparation


def _validate_preregistration_runtime(
    value: Any,
    *,
    stimulus_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _require_exact_keys(
        value,
        {"environment", "processor_preparation"},
        "preregistration runtime identity",
    )
    _validate_runtime_environment(runtime["environment"], include_cuda=True)
    _validate_processor_preparation(
        runtime["processor_preparation"],
        stimulus_manifest=stimulus_manifest,
    )
    return _detached_json(dict(runtime), "preregistration runtime identity")


def build_stimulus_manifest(
    *,
    items: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    overlap_audit: Mapping[str, Any],
    tts_recipe_sha256: str,
    source_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": STIMULUS_MANIFEST_KIND,
        "status": "complete",
        "protocol_sha256": canonical_sha256(frozen_protocol()),
        "fixtures": [fixture_identity(spec) for spec in FIXTURES],
        "items": list(items),
        "observations": list(observations),
        "overlap_audit": dict(overlap_audit),
        "tts_recipe_sha256": tts_recipe_sha256,
        "source_identity": dict(source_identity),
        "runtime_identity": dict(runtime_identity),
    }
    sealed = seal_mapping(body, "stimulus_manifest_sha256")
    return validate_stimulus_manifest(sealed)


def validate_stimulus_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    value = validate_seal(manifest, "stimulus_manifest_sha256", "stimulus manifest")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "status",
            "protocol_sha256",
            "fixtures",
            "items",
            "observations",
            "overlap_audit",
            "tts_recipe_sha256",
            "source_identity",
            "runtime_identity",
            "stimulus_manifest_sha256",
        },
        "stimulus manifest",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != STIMULUS_MANIFEST_KIND
        or value["status"] != "complete"
        or value["protocol_sha256"] != canonical_sha256(frozen_protocol())
        or value["fixtures"] != [fixture_identity(spec) for spec in FIXTURES]
    ):
        raise AudioWorkspaceEvalContractError("stimulus manifest frozen identity changed")
    if not _is_sha256(value["tts_recipe_sha256"]):
        raise AudioWorkspaceEvalContractError("stimulus engine identity is invalid")
    _validate_source_identity(value["source_identity"])
    _validate_runtime_environment(value["runtime_identity"], include_cuda=False)
    items = value["items"]
    if (
        not isinstance(items, list)
        or len(items) != EXPECTED_ITEM_COUNT
        or [
            (item.get("distribution"), item.get("name"))
            if isinstance(item, Mapping)
            else (None, None)
            for item in items
        ]
        != list(EXPECTED_COORDINATES)
    ):
        raise AudioWorkspaceEvalContractError(
            f"stimulus items are not the exact {EXPECTED_ITEM_COUNT} coordinates"
        )
    expected_item_keys = {
        "coordinate_index",
        "distribution",
        "name",
        "language",
        "script",
        "script_sha256",
        "source_item_sha256",
        "intermediates",
        "target_excluded",
    }
    for index, item in enumerate(items):
        _require_exact_keys(item, expected_item_keys, "stimulus item")
        script = item["script"]
        intermediates = item["intermediates"]
        if (
            item["coordinate_index"] != index
            or item["language"] != language_for_coordinate(item["distribution"], item["name"])
            or not isinstance(script, str)
            or not script
            or item["script_sha256"] != sha256_bytes(script.encode("utf-8"))
            or not _is_sha256(item["source_item_sha256"])
            or item["target_excluded"] is not True
            or not isinstance(intermediates, list)
            or not intermediates
            or any(not isinstance(authored, str) or not authored for authored in intermediates)
        ):
            raise AudioWorkspaceEvalContractError("stimulus item content changed")
    if expected_items is not None and items != _detached_json(
        list(expected_items), "expected canonical spoken items"
    ):
        raise AudioWorkspaceEvalContractError(
            "stimulus scripts do not match the decoded canonical fixture boundaries"
        )
    observations = value["observations"]
    if not isinstance(observations, list) or len(observations) != EXPECTED_OBSERVATION_COUNT:
        raise AudioWorkspaceEvalContractError(
            f"stimulus manifest must contain exactly {EXPECTED_OBSERVATION_COUNT} observations"
        )
    expected_observation_keys = {
        "observation_index",
        "coordinate_index",
        "distribution",
        "name",
        "variant",
        "language",
        "script_sha256",
        "tts_input",
        "tts_input_sha256",
        "wav_relative_path",
        "source_wav_sha256",
        "source_pcm_sha256",
        "normalized_wav_sha256",
        "decoded_pcm_sha256",
        "source_sample_rate",
        "sample_rate",
        "sample_count",
        "duration_seconds",
    }
    paths: set[str] = set()
    stimulus_wave: set[str] = set()
    stimulus_pcm: set[str] = set()
    for index, observation in enumerate(observations):
        _require_exact_keys(observation, expected_observation_keys, "stimulus observation")
        item_index = index // len(TTS_VARIANTS)
        variant = TTS_VARIANTS[index % len(TTS_VARIANTS)]
        item = items[item_index]
        if (
            observation["observation_index"] != index
            or observation["coordinate_index"] != item_index
            or (
                observation["distribution"],
                observation["name"],
                observation["variant"],
            )
            != (item["distribution"], item["name"], variant)
            or observation["language"] != item["language"]
            or observation["script_sha256"] != item["script_sha256"]
            or observation["tts_input"] != tts_input(item["script"])
            or observation["tts_input_sha256"]
            != sha256_bytes(str(observation["tts_input"]).encode("utf-8"))
        ):
            raise AudioWorkspaceEvalContractError("stimulus observation order or TTS input changed")
        duration = observation["duration_seconds"]
        if (
            observation["source_sample_rate"] != TTS_SAMPLE_RATE
            or observation["sample_rate"] != NORMALIZED_SAMPLE_RATE
            or isinstance(observation["sample_count"], bool)
            or not isinstance(observation["sample_count"], int)
            or observation["sample_count"] <= 0
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not math.isclose(
                float(duration),
                observation["sample_count"] / NORMALIZED_SAMPLE_RATE,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise AudioWorkspaceEvalContractError("stimulus observation audio geometry changed")
        if any(
            not _is_sha256(observation[key])
            for key in (
                "source_wav_sha256",
                "source_pcm_sha256",
                "normalized_wav_sha256",
                "decoded_pcm_sha256",
            )
        ):
            raise AudioWorkspaceEvalContractError("stimulus observation hashes changed")
        path = observation["wav_relative_path"]
        expected_path = f"wavs/{index:03d}-{variant}.wav"
        if path != expected_path or path in paths:
            raise AudioWorkspaceEvalContractError("stimulus WAV path identity changed")
        paths.add(path)
        stimulus_wave.update(
            (
                observation["source_wav_sha256"],
                observation["normalized_wav_sha256"],
            )
        )
        stimulus_pcm.update(
            (
                observation["source_pcm_sha256"],
                observation["decoded_pcm_sha256"],
            )
        )
    overlap = value["overlap_audit"]
    _require_exact_keys(
        overlap,
        {
            "policy",
            "fit_manifest_sha256",
            "stimulus_observations",
            "fit_rows",
            "waveform_overlap_count",
            "decoded_pcm_overlap_count",
            "normalized_transcript_overlap_count",
            "stimulus_waveform_set_sha256",
            "fit_waveform_set_sha256",
            "stimulus_pcm_set_sha256",
            "fit_pcm_set_sha256",
            "stimulus_transcript_set_sha256",
            "fit_transcript_set_sha256",
        },
        "stimulus overlap audit",
    )
    expected_stimulus_digests = {
        "stimulus_waveform_set_sha256": canonical_sha256(sorted(stimulus_wave)),
        "stimulus_pcm_set_sha256": canonical_sha256(sorted(stimulus_pcm)),
        "stimulus_transcript_set_sha256": canonical_sha256(
            sorted({normalize_transcript(item["script"]) for item in items})
        ),
    }
    if (
        overlap["policy"]
        != "reject_waveform_pcm_or_nfkc_casefold_punctuation_whitespace_transcript_overlap"
        or not _is_sha256(overlap["fit_manifest_sha256"])
        or overlap["stimulus_observations"] != EXPECTED_OBSERVATION_COUNT
        or overlap["fit_rows"] != 1_000
        or any(
            overlap[key] != 0
            for key in (
                "waveform_overlap_count",
                "decoded_pcm_overlap_count",
                "normalized_transcript_overlap_count",
            )
        )
        or any(overlap[key] != expected for key, expected in expected_stimulus_digests.items())
        or any(
            not _is_sha256(overlap[key])
            for key in (
                "fit_waveform_set_sha256",
                "fit_pcm_set_sha256",
                "fit_transcript_set_sha256",
            )
        )
    ):
        raise AudioWorkspaceEvalContractError(
            "stimulus overlap audit is absent, inconsistent, or failed"
        )
    return value


def build_calibration(
    *, stimulus_manifest: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    stimulus = validate_stimulus_manifest(stimulus_manifest)
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": CALIBRATION_KIND,
        "stimulus_manifest_sha256": stimulus["stimulus_manifest_sha256"],
        "config": frozen_protocol()["calibration"],
        "cells": list(cells),
    }
    normalized_cells = _validate_calibration_cells(body["cells"], stimulus["items"])
    threshold_result = calibration_status([float(cell["cer"]) for cell in normalized_cells])
    body.update(threshold_result)
    return seal_mapping(body, "calibration_sha256")


def _validate_calibration_cells(
    cells: Any, spoken_items: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    if not isinstance(cells, list) or len(cells) != EXPECTED_CALIBRATION_CELL_COUNT:
        raise AudioWorkspaceEvalContractError(
            f"calibration must contain exactly {EXPECTED_CALIBRATION_CELL_COUNT} cells"
        )
    expected_coordinates = calibration_coordinates(spoken_items)
    item_by_coordinate = {(item["distribution"], item["name"]): item for item in spoken_items}
    expected_keys = {
        "distribution",
        "name",
        "language",
        "variant",
        "script_sha256",
        "reference",
        "transcript",
        "normalized_reference",
        "normalized_transcript",
        "cer",
    }
    for cell, expected in zip(cells, expected_coordinates, strict=True):
        _require_exact_keys(cell, expected_keys, "calibration cell")
        if {key: cell[key] for key in expected} != expected:
            raise AudioWorkspaceEvalContractError("calibration coordinate order changed")
        item = item_by_coordinate[(cell["distribution"], cell["name"])]
        language = str(cell["language"])
        if (
            cell["reference"] != item["script"]
            or cell["normalized_reference"]
            != normalize_transcript_for_language(cell["reference"], language)
            or cell["normalized_transcript"]
            != normalize_transcript_for_language(cell["transcript"], language)
        ):
            raise AudioWorkspaceEvalContractError("calibration transcript normalization changed")
        expected_cer = character_error_rate(cell["reference"], cell["transcript"], language)
        if (
            isinstance(cell["cer"], bool)
            or not isinstance(cell["cer"], (int, float))
            or not math.isclose(float(cell["cer"]), expected_cer, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise AudioWorkspaceEvalContractError("calibration CER does not recompute")
    return cells


def validate_calibration(
    calibration: Mapping[str, Any], stimulus_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    stimulus = validate_stimulus_manifest(stimulus_manifest)
    value = validate_seal(calibration, "calibration_sha256", "calibration")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "stimulus_manifest_sha256",
            "config",
            "cells",
            "macro_cer",
            "max_cell_cer",
            "status",
            "calibration_sha256",
        },
        "calibration",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != CALIBRATION_KIND
        or value["stimulus_manifest_sha256"] != stimulus["stimulus_manifest_sha256"]
        or value["config"] != frozen_protocol()["calibration"]
    ):
        raise AudioWorkspaceEvalContractError("calibration identity changed")
    cells = _validate_calibration_cells(value["cells"], stimulus["items"])
    threshold_result = calibration_status([float(cell["cer"]) for cell in cells])
    macro = threshold_result["macro_cer"]
    maximum = threshold_result["max_cell_cer"]
    expected_status = threshold_result["status"]
    if (
        isinstance(value["macro_cer"], bool)
        or not isinstance(value["macro_cer"], (int, float))
        or isinstance(value["max_cell_cer"], bool)
        or not isinstance(value["max_cell_cer"], (int, float))
        or not math.isclose(float(value["macro_cer"]), macro, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(
            float(value["max_cell_cer"]),
            maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or value["status"] != expected_status
    ):
        raise AudioWorkspaceEvalContractError("calibration result or threshold status changed")
    return value


def _validate_eligibility(
    eligibility: Any,
    *,
    stimulus_items: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    value = validate_seal(eligibility, "eligibility_sha256", "eligibility")
    _require_exact_keys(value, {"items", "eligibility_sha256"}, "eligibility")
    items = value["items"]
    if not isinstance(items, list) or [
        (item.get("distribution"), item.get("name")) if isinstance(item, Mapping) else (None, None)
        for item in items
    ] != list(EXPECTED_COORDINATES):
        raise AudioWorkspaceEvalContractError("eligibility coordinates changed")
    if stimulus_items is not None and (
        not isinstance(stimulus_items, list) or len(stimulus_items) != EXPECTED_ITEM_COUNT
    ):
        raise AudioWorkspaceEvalContractError("eligibility needs the exact stimulus items")
    global_concept_ids: set[str] = set()
    for item_index, item in enumerate(items):
        _require_exact_keys(
            item,
            {
                "distribution",
                "name",
                "included_in_metrics",
                "eligible_concept_ids",
                "concepts",
            },
            "eligibility item",
        )
        concept_ids = item["eligible_concept_ids"]
        concepts = item["concepts"]
        if (
            not isinstance(item["included_in_metrics"], bool)
            or not isinstance(concept_ids, list)
            or len(set(concept_ids)) != len(concept_ids)
            or not isinstance(concepts, list)
        ):
            raise AudioWorkspaceEvalContractError("eligibility item concepts are invalid")
        expected_prefix = f"{item['distribution']}/{item['name']}/"
        concept_indices: list[int] = []
        by_id: dict[str, list[int]] = {}
        for concept in concepts:
            _require_exact_keys(
                concept,
                {"concept_id", "authored", "forms", "allowed_token_ids"},
                "eligibility concept",
            )
            concept_id = concept["concept_id"]
            authored = concept["authored"]
            forms = concept["forms"]
            ids = concept["allowed_token_ids"]
            if (
                not isinstance(concept_id, str)
                or not concept_id.startswith(expected_prefix)
                or re.fullmatch(
                    re.escape(expected_prefix) + r"(0|[1-9][0-9]*)",
                    concept_id,
                )
                is None
                or concept_id in global_concept_ids
                or not isinstance(authored, str)
                or not authored
                or not isinstance(forms, list)
                or forms != list(allowed_forms(item["distribution"], authored))
                or not isinstance(ids, list)
                or not ids
                or len(set(ids)) != len(ids)
                or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in ids
                )
            ):
                raise AudioWorkspaceEvalContractError(
                    "eligibility authored forms, concept identity, or allowed-token IDs are invalid"
                )
            concept_index = int(concept_id.removeprefix(expected_prefix))
            concept_indices.append(concept_index)
            global_concept_ids.add(concept_id)
            by_id[concept_id] = list(ids)
            if stimulus_items is not None:
                stimulus = stimulus_items[item_index]
                intermediates = stimulus["intermediates"]
                if (
                    (stimulus["distribution"], stimulus["name"])
                    != (item["distribution"], item["name"])
                    or concept_index >= len(intermediates)
                    or authored != intermediates[concept_index]
                ):
                    raise AudioWorkspaceEvalContractError(
                        "eligibility authored concept differs from the "
                        "canonical stimulus intermediate"
                    )
        if (
            concept_indices != sorted(concept_indices)
            or len(set(concept_indices)) != len(concept_indices)
            or list(by_id) != concept_ids
            or item["included_in_metrics"] != bool(concept_ids)
        ):
            raise AudioWorkspaceEvalContractError("eligibility concept identity changed")
    return value


def build_preregistration(
    *,
    stimulus_manifest: Mapping[str, Any],
    expected_items: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    artifact_identity: Mapping[str, Any],
    historical_text_report: Mapping[str, Any],
    eligibility: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    stimulus = validate_stimulus_manifest(stimulus_manifest, expected_items=expected_items)
    calibrated = validate_calibration(calibration, stimulus)
    eligibility_value = _validate_eligibility(eligibility, stimulus_items=stimulus["items"])
    source = _validate_source_identity(source_identity)
    if source != stimulus["source_identity"]:
        raise AudioWorkspaceEvalContractError("preregistration source differs from stimulus source")
    runtime = _validate_preregistration_runtime(runtime_identity, stimulus_manifest=stimulus)
    common_text = recompute_common_item_text_summary(historical_text_report)
    historical_text_report_identity = {
        "report_sha256": common_text["text_report_sha256"],
        "status": common_text["text_status"],
        "kind": "workspace_jlens_evaluation",
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": PREREGISTRATION_KIND,
        "status": "sealed",
        "protocol": frozen_protocol(),
        "stimulus_manifest_sha256": stimulus["stimulus_manifest_sha256"],
        "calibration_sha256": calibrated["calibration_sha256"],
        "calibration_status": calibrated["status"],
        "artifact_identity": dict(artifact_identity),
        "historical_text_report_identity": dict(historical_text_report_identity),
        "eligibility": eligibility_value,
        "source_identity": source,
        "runtime_identity": runtime,
        "expected_coordinates": [list(value) for value in EXPECTED_COORDINATES],
        "expected_coordinates_sha256": EXPECTED_COORDINATES_SHA256,
    }
    sealed = seal_mapping(body, "preregistration_sha256")
    return validate_preregistration(sealed)


def validate_preregistration(
    preregistration: Mapping[str, Any],
    *,
    stimulus_manifest: Mapping[str, Any] | None = None,
    expected_items: Sequence[Mapping[str, Any]] | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = validate_seal(preregistration, "preregistration_sha256", "preregistration")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "status",
            "protocol",
            "stimulus_manifest_sha256",
            "calibration_sha256",
            "calibration_status",
            "artifact_identity",
            "historical_text_report_identity",
            "eligibility",
            "source_identity",
            "runtime_identity",
            "expected_coordinates",
            "expected_coordinates_sha256",
            "preregistration_sha256",
        },
        "preregistration",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != PREREGISTRATION_KIND
        or value["status"] != "sealed"
        or value["protocol"] != frozen_protocol()
        or value["expected_coordinates"] != [list(item) for item in EXPECTED_COORDINATES]
        or value["expected_coordinates_sha256"] != EXPECTED_COORDINATES_SHA256
    ):
        raise AudioWorkspaceEvalContractError(
            "preregistration frozen protocol or coordinates changed"
        )
    artifact = _require_exact_keys(
        value["artifact_identity"],
        {
            "fit_config_sha256",
            "lens_sha256",
            "lens_dtype",
            "lens_layers",
            "completed_run_manifest_sha256",
        },
        "audio fit artifact identity",
    )
    if (
        artifact["fit_config_sha256"] != AUDIO_FIT_CONFIG_SHA256
        or artifact["lens_sha256"] != AUDIO_LENS_SHA256
        or artifact["lens_dtype"] != "float16"
        or artifact["lens_layers"] != list(LENS_LAYERS)
        or not _is_sha256(artifact["completed_run_manifest_sha256"])
    ):
        raise AudioWorkspaceEvalContractError("final audio lens/completed-run identity changed")
    text_identity = _require_exact_keys(
        value["historical_text_report_identity"],
        {"report_sha256", "status", "kind"},
        "historical text report identity",
    )
    if (
        not _is_sha256(text_identity["report_sha256"])
        or text_identity["status"] != "no_band"
        or text_identity["kind"] != "workspace_jlens_evaluation"
    ):
        raise AudioWorkspaceEvalContractError("historical text report identity changed")
    _validate_eligibility(value["eligibility"])
    _validate_source_identity(value["source_identity"])
    _validate_preregistration_runtime(value["runtime_identity"])
    if (
        not _is_sha256(value["stimulus_manifest_sha256"])
        or not _is_sha256(value["calibration_sha256"])
        or value["calibration_status"] not in {"passed", "failed"}
    ):
        raise AudioWorkspaceEvalContractError("preregistration dynamic identity is invalid")
    if stimulus_manifest is not None:
        if expected_items is None:
            raise AudioWorkspaceEvalContractError(
                "canonical expected items are required with a stimulus"
            )
        stimulus = validate_stimulus_manifest(stimulus_manifest, expected_items=expected_items)
        if (
            value["stimulus_manifest_sha256"] != stimulus["stimulus_manifest_sha256"]
            or value["source_identity"] != stimulus["source_identity"]
        ):
            raise AudioWorkspaceEvalContractError("preregistration stimulus identity mismatch")
        _validate_eligibility(value["eligibility"], stimulus_items=stimulus["items"])
        _validate_preregistration_runtime(value["runtime_identity"], stimulus_manifest=stimulus)
        if calibration is None:
            raise AudioWorkspaceEvalContractError(
                "preregistration validation needs its calibration"
            )
        calibrated = validate_calibration(calibration, stimulus)
        if (
            value["calibration_sha256"] != calibrated["calibration_sha256"]
            or value["calibration_status"] != calibrated["status"]
        ):
            raise AudioWorkspaceEvalContractError("preregistration calibration identity mismatch")
    elif expected_items is not None or calibration is not None:
        raise AudioWorkspaceEvalContractError(
            "stimulus is required for canonical preregistration validation"
        )
    return value


def recompute_common_item_text_summary(text_report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute both reducers on the frozen common identities from a validated text report."""
    if (
        not isinstance(text_report, Mapping)
        or text_report.get("kind") != "workspace_jlens_evaluation"
        or text_report.get("status") != "complete"
        or not isinstance(text_report.get("adjudication"), Mapping)
        or text_report["adjudication"].get("status") != "no_band"
    ):
        raise AudioWorkspaceEvalContractError(
            "historical text report is not the immutable complete no_band report"
        )
    report_sha = text_report.get("workspace_report_sha256")
    report_body = dict(text_report)
    report_body.pop("workspace_report_sha256", None)
    if not _is_sha256(report_sha) or canonical_sha256(report_body) != report_sha:
        raise AudioWorkspaceEvalContractError("historical text report content digest mismatch")
    items = text_report.get("items")
    if not isinstance(items, list):
        raise AudioWorkspaceEvalContractError("historical text report has no raw items")
    indexed = {
        (item.get("distribution"), item.get("name")): item
        for item in items
        if isinstance(item, Mapping)
    }
    if len(indexed) != len(items) or any(
        coordinate not in indexed for coordinate in EXPECTED_COORDINATES
    ):
        raise AudioWorkspaceEvalContractError(
            "historical text report is missing common-item coordinates"
        )
    selected = [indexed[coordinate] for coordinate in EXPECTED_COORDINATES]
    expected_layer_names = {str(layer) for layer in LENS_LAYERS}
    for item in selected:
        concepts = item.get("eligible_concept_ids")
        included = item.get("included_in_metrics")
        layers = item.get("layers")
        if (
            not isinstance(included, bool)
            or not isinstance(concepts, list)
            or len(set(concepts)) != len(concepts)
            or any(not isinstance(concept, str) or not concept for concept in concepts)
            or included != bool(concepts)
            or not isinstance(layers, Mapping)
            or set(layers) != expected_layer_names
        ):
            raise AudioWorkspaceEvalContractError("historical text common-item identity changed")
        concept_set = set(concepts)
        for layer_name in expected_layer_names:
            layer = layers[layer_name]
            if not isinstance(layer, Mapping):
                raise AudioWorkspaceEvalContractError(
                    "historical text common-item layer is invalid"
                )
            rank_variants = layer.get("concept_ranks")
            if (
                not isinstance(rank_variants, Mapping)
                or set(rank_variants) != set(RANK_VARIANTS)
                or any(
                    not isinstance(rank_variants[variant], Mapping)
                    or set(rank_variants[variant]) != concept_set
                    or any(
                        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
                        for rank in rank_variants[variant].values()
                    )
                    for variant in RANK_VARIANTS
                )
            ):
                raise AudioWorkspaceEvalContractError(
                    "historical text common-item ranks are invalid"
                )

    def text_summary(
        position: str, region: str, rank_variant: str, historical: bool
    ) -> dict[str, Any]:
        distribution_curves = []
        per_distribution = {}
        for distribution in DISTRIBUTIONS:
            curves = []
            for item in selected:
                if item["distribution"] != distribution or not item.get("included_in_metrics"):
                    continue
                concepts = item["eligible_concept_ids"]
                if historical:
                    ranks = [
                        min(
                            int(item["layers"][str(layer)]["concept_ranks"][rank_variant][concept])
                            for layer in REGIONS[region]
                        )
                        for concept in concepts
                    ]
                    curve = _curve_from_ranks(ranks)
                else:
                    curve = _mean_curves(
                        [
                            _curve_from_ranks(
                                [
                                    int(
                                        item["layers"][str(layer)]["concept_ranks"][rank_variant][
                                            concept
                                        ]
                                    )
                                    for concept in concepts
                                ]
                            )
                            for layer in REGIONS[region]
                        ]
                    )
                curves.append(curve)
            if not curves:
                raise AudioWorkspaceEvalContractError(
                    f"text common-item {distribution} has no eligible items"
                )
            per_distribution[distribution] = {
                "n_items": len(curves),
                **_curve_summary(_mean_curves(curves)),
            }
            distribution_curves.append(per_distribution[distribution]["pass_at_k"])
        return {
            "position": position,
            "region": region,
            "rank_variant": rank_variant,
            "layer_reducer": "minimum_over_region" if historical else "equal_mean_per_layer",
            "distributions": per_distribution,
            "aggregate": _curve_summary(_mean_curves(distribution_curves)),
        }

    primary = {
        region: {
            variant: text_summary("canonical_text_position", region, variant, False)
            for variant in RANK_VARIANTS
        }
        for region in REGIONS
    }
    historical = {
        region: {
            variant: text_summary("canonical_text_position", region, variant, True)
            for variant in RANK_VARIANTS
        }
        for region in REGIONS
    }
    body = {
        "text_report_sha256": report_sha,
        "text_status": "no_band",
        "n_common_items": EXPECTED_ITEM_COUNT,
        "coordinates_sha256": EXPECTED_COORDINATES_SHA256,
        "primary_fair_region": primary,
        "historical_secondary": historical,
        "canonical_text_verdict_unchanged": True,
    }
    return {**body, "common_item_text_summary_sha256": canonical_sha256(body)}


def validate_common_item_text_summary_identity(
    summary: Mapping[str, Any],
    historical_text_report_identity: Mapping[str, Any],
) -> dict[str, Any]:
    value = _require_exact_keys(
        summary,
        {
            "text_report_sha256",
            "text_status",
            "n_common_items",
            "coordinates_sha256",
            "primary_fair_region",
            "historical_secondary",
            "canonical_text_verdict_unchanged",
            "common_item_text_summary_sha256",
        },
        "common-item text summary",
    )
    body = dict(value)
    claimed = body.pop("common_item_text_summary_sha256")
    if (
        not _is_sha256(claimed)
        or canonical_sha256(body) != claimed
        or value["text_report_sha256"] != historical_text_report_identity.get("report_sha256")
        or value["text_status"] != "no_band"
        or value["n_common_items"] != EXPECTED_ITEM_COUNT
        or value["coordinates_sha256"] != EXPECTED_COORDINATES_SHA256
        or value["canonical_text_verdict_unchanged"] is not True
        or not isinstance(value["primary_fair_region"], Mapping)
        or set(value["primary_fair_region"]) != set(REGIONS)
        or not isinstance(value["historical_secondary"], Mapping)
        or set(value["historical_secondary"]) != set(REGIONS)
        or any(
            not isinstance(value[section][region], Mapping)
            or set(value[section][region]) != set(RANK_VARIANTS)
            for section in ("primary_fair_region", "historical_secondary")
            for region in REGIONS
        )
    ):
        raise AudioWorkspaceEvalContractError("common-item text summary identity changed")
    return dict(value)


def validate_common_item_text_summary(
    summary: Mapping[str, Any], text_report: Mapping[str, Any]
) -> dict[str, Any]:
    expected = recompute_common_item_text_summary(text_report)
    if summary != expected:
        raise AudioWorkspaceEvalContractError("common-item text summary does not recompute")
    return dict(summary)


def build_report(
    *,
    preregistration: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    historical_text_report: Mapping[str, Any],
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    if prereg["calibration_status"] != "passed":
        raise AudioWorkspaceEvalContractError(
            "confirmatory scoring cannot follow failed calibration"
        )
    common_text = recompute_common_item_text_summary(historical_text_report)
    validate_common_item_text_summary_identity(
        common_text,
        prereg["historical_text_report_identity"],
    )
    validated_records = validate_score_records(
        list(records),
        eligibility=prereg["eligibility"],
        runtime_identity=prereg["runtime_identity"],
    )
    summaries = build_summaries(validated_records)
    bootstrap_value = bundled_item_bootstrap(validated_records)
    permutation_value = label_permutation_max_stat(validated_records)
    evidence = build_evidence(
        summaries,
        bootstrap_value,
        permutation_value,
        common_text,
    )
    adjudication = adjudicate(evidence)
    body = {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "status": adjudication["status"],
        "completion": "complete",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "protocol_sha256": canonical_sha256(frozen_protocol()),
        "records": validated_records,
        "summaries": summaries,
        "statistics": {
            "bootstrap": bootstrap_value,
            "label_permutation_max_stat": permutation_value,
        },
        "evidence": evidence,
        "adjudication": adjudication,
        "common_item_text_summary": common_text,
    }
    return seal_mapping(body, "report_sha256")


def validate_report(
    report: Mapping[str, Any],
    *,
    preregistration: Mapping[str, Any],
    historical_text_report: Mapping[str, Any],
) -> dict[str, Any]:
    prereg = validate_preregistration(preregistration)
    if prereg["calibration_status"] != "passed":
        raise AudioWorkspaceEvalContractError(
            "score-bearing report cannot follow failed calibration"
        )
    value = validate_seal(report, "report_sha256", "audio workspace report")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "kind",
            "status",
            "completion",
            "preregistration_sha256",
            "protocol_sha256",
            "records",
            "summaries",
            "statistics",
            "evidence",
            "adjudication",
            "common_item_text_summary",
            "report_sha256",
        },
        "audio workspace report",
    )
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["kind"] != REPORT_KIND
        or value["completion"] != "complete"
        or value["preregistration_sha256"] != prereg["preregistration_sha256"]
        or value["protocol_sha256"] != canonical_sha256(frozen_protocol())
    ):
        raise AudioWorkspaceEvalContractError("audio workspace report identity changed")
    common_text = validate_common_item_text_summary_identity(
        value["common_item_text_summary"],
        prereg["historical_text_report_identity"],
    )
    validate_common_item_text_summary(common_text, historical_text_report)
    records = validate_score_records(
        value["records"],
        eligibility=prereg["eligibility"],
        runtime_identity=prereg["runtime_identity"],
    )
    summaries = build_summaries(records)
    if value["summaries"] != summaries:
        raise AudioWorkspaceEvalContractError("report summaries do not match raw ranks")
    expected_bootstrap = bundled_item_bootstrap(records)
    expected_permutation = label_permutation_max_stat(records)
    expected_statistics = {
        "bootstrap": expected_bootstrap,
        "label_permutation_max_stat": expected_permutation,
    }
    if value["statistics"] != expected_statistics:
        raise AudioWorkspaceEvalContractError(
            "report deterministic statistics do not match raw ranks"
        )
    evidence = build_evidence(
        summaries,
        expected_bootstrap,
        expected_permutation,
        common_text,
    )
    if value["evidence"] != evidence:
        raise AudioWorkspaceEvalContractError("report evidence does not match raw ranks")
    adjudication = adjudicate(evidence)
    if value["adjudication"] != adjudication or value["status"] != adjudication["status"]:
        raise AudioWorkspaceEvalContractError("report status does not match frozen adjudication")
    return value
