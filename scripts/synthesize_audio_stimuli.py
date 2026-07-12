"""Synthesize the sealed OpenAI TTS stimulus set for the audio workspace eval.

This runs locally: it needs network access and OPENAI_API_KEY, and it never
imports Modal, Torch, or model weights. Generation through the OpenAI speech
endpoint is not reproducible, so the scientific object this script produces is
the sealed byte set itself: every WAV is rewritten into a canonical RIFF
container (the API emits a streaming header with a bogus data size), hashed,
and bound into one content-addressed recipe that Modal staging verifies
byte-for-byte before normalization.

    uv run python scripts/synthesize_audio_stimuli.py --output stimuli-build
    uv run modal volume put audiolens-vol \
        stimuli-build/<recipe_sha256> \
        audio-workspace-eval/source-stimuli/<recipe_sha256>
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
import pathlib
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from audiolens.audio_workspace_eval import (
    EXPECTED_ITEM_COUNT,
    EXPECTED_OBSERVATION_COUNT,
    FIXTURES,
    TTS_ENDPOINT,
    TTS_ENGINE,
    TTS_INPUT_POLICY,
    TTS_MODEL,
    TTS_RESPONSE_FORMAT,
    TTS_SAMPLE_RATE,
    TTS_SYNTHESIS_POLICY,
    TTS_VARIANTS,
    build_spoken_items,
    decode_publication_fixtures,
    seal_mapping,
    sha256_bytes,
    tts_input,
)

TTS_RECIPE_KIND = "audio_workspace_tts_recipe"
SMOKE_ROW_RANGE = (50, 52)
MIN_DURATION_SECONDS = 0.3
MIN_PEAK_AMPLITUDE = 200
SYNTHESIS_ATTEMPTS = 4
MAX_WAV_BYTES = 64 * 1024 * 1024
WORKERS = 8


class SynthesisError(RuntimeError):
    """A stimulus could not be synthesized to a valid sealed WAV."""


def _fixture_url(spec: Any) -> str:
    return spec.url


def fetch_fixture(spec: Any, cache_root: pathlib.Path) -> bytes:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / spec.filename
    if path.is_file():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(_fixture_url(spec), timeout=60) as response:
            raw = response.read(spec.n_bytes + 1)
        path.write_bytes(raw)
    if len(raw) != spec.n_bytes or sha256_bytes(raw) != spec.sha256:
        raise SynthesisError(f"{spec.slug}: pinned fixture bytes changed")
    return raw


def canonical_wav(raw: bytes) -> tuple[bytes, int]:
    """Rewrite API WAV bytes into a well-formed canonical mono PCM16 RIFF file.

    The speech endpoint streams a RIFF header whose data-chunk size is a
    placeholder, so sizes are recovered from the actual payload and the
    container is rewritten with exact lengths.
    """
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise SynthesisError("API response is not a RIFF WAVE file")
    offset = 12
    fmt: tuple[int, int, int, int] | None = None
    data: bytes | None = None
    while offset + 8 <= len(raw):
        chunk_id = raw[offset : offset + 4]
        (declared,) = struct.unpack("<I", raw[offset + 4 : offset + 8])
        body_start = offset + 8
        available = len(raw) - body_start
        size = min(declared, available)
        if chunk_id == b"fmt ":
            audio_format, channels, rate, _, _, bits = struct.unpack(
                "<HHIIHH", raw[body_start : body_start + 16]
            )
            fmt = (audio_format, channels, rate, bits)
        elif chunk_id == b"data":
            data = raw[body_start : body_start + size]
            if declared > available:
                # Streaming placeholder size: the payload runs to end of file.
                data = raw[body_start:]
        offset = body_start + size + (size % 2)
    if fmt is None or data is None:
        raise SynthesisError("API WAV lacks fmt or data chunks")
    if fmt != (1, 1, TTS_SAMPLE_RATE, 16):
        raise SynthesisError(f"API WAV format changed: {fmt}")
    if len(data) % 2:
        data = data[:-1]
    if not data:
        raise SynthesisError("API WAV has no PCM payload")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(TTS_SAMPLE_RATE)
        writer.writeframes(data)
    frames = len(data) // 2
    return output.getvalue(), frames


def _validate_pcm(payload: bytes, frames: int, label: str) -> None:
    duration = frames / TTS_SAMPLE_RATE
    if duration < MIN_DURATION_SECONDS:
        raise SynthesisError(f"{label}: synthesized audio is only {duration:.3f}s")
    with wave.open(io.BytesIO(payload)) as reader:
        pcm = reader.readframes(reader.getnframes())
    peak = max(abs(value) for (value,) in struct.iter_unpack("<h", pcm))
    if peak < MIN_PEAK_AMPLITUDE:
        raise SynthesisError(f"{label}: synthesized audio is effectively silent")


def synthesize(text: str, voice: str, *, api_key: str, label: str) -> bytes:
    body = json.dumps(
        {
            "model": TTS_MODEL,
            "voice": voice,
            "input": text,
            "response_format": TTS_RESPONSE_FORMAT,
        }
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(SYNTHESIS_ATTEMPTS):
        try:
            request = urllib.request.Request(
                TTS_ENDPOINT,
                data=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read(MAX_WAV_BYTES + 1)
            if not raw or len(raw) > MAX_WAV_BYTES:
                raise SynthesisError(f"{label}: API returned invalid byte size")
            payload, frames = canonical_wav(raw)
            _validate_pcm(payload, frames, label)
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, SynthesisError, OSError) as exc:
            last_error = exc
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise SynthesisError(f"{label}: synthesis failed after retries") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="build directory for the sealed set")
    arguments = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    output_root = pathlib.Path(arguments.output)
    cache_root = output_root / "fixture-cache"

    raw_by_distribution = {spec.slug: fetch_fixture(spec, cache_root) for spec in FIXTURES}
    items = build_spoken_items(decode_publication_fixtures(raw_by_distribution))
    if len(items) != EXPECTED_ITEM_COUNT:
        raise SynthesisError(f"derived {len(items)} items, expected {EXPECTED_ITEM_COUNT}")

    association_raw = json.loads(raw_by_distribution["association"])["items"]
    smoke_rows = association_raw[SMOKE_ROW_RANGE[0] : SMOKE_ROW_RANGE[1]]
    if len(smoke_rows) != 2:
        raise SynthesisError("nonpublication smoke rows are unavailable")

    jobs: list[dict[str, Any]] = []
    for item in items:
        for variant_index, variant in enumerate(TTS_VARIANTS):
            observation_index = item["coordinate_index"] * len(TTS_VARIANTS) + variant_index
            spoken = tts_input(item["script"])
            jobs.append(
                {
                    "entry": {
                        "observation_index": observation_index,
                        "coordinate_index": item["coordinate_index"],
                        "distribution": item["distribution"],
                        "name": item["name"],
                        "variant": variant,
                        "language": item["language"],
                        "script_sha256": item["script_sha256"],
                        "tts_input": spoken,
                        "tts_input_sha256": sha256_bytes(spoken.encode("utf-8")),
                        "wav_relative_path": f"wavs/{observation_index:03d}-{variant}.wav",
                    },
                    "voice": variant,
                    "text": spoken,
                    "kind": "observation",
                }
            )
    for index, row in enumerate(smoke_rows):
        prompt = str(row["prompt"])
        spoken = tts_input(prompt)
        publication_index = SMOKE_ROW_RANGE[0] + index
        jobs.append(
            {
                "entry": {
                    "publication_index": publication_index,
                    "name": f"nonconfirmatory/{row['name']}",
                    "variant": TTS_VARIANTS[0],
                    "language": "en-us",
                    "script_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "tts_input": spoken,
                    "tts_input_sha256": sha256_bytes(spoken.encode("utf-8")),
                    "wav_relative_path": (f"smoke/{publication_index:03d}-{TTS_VARIANTS[0]}.wav"),
                },
                "voice": TTS_VARIANTS[0],
                "text": spoken,
                "kind": "smoke",
            }
        )

    def run_job(job: dict[str, Any]) -> dict[str, Any]:
        label = job["entry"]["wav_relative_path"]
        payload = synthesize(job["text"], job["voice"], api_key=api_key, label=label)
        return {
            "kind": job["kind"],
            "entry": {
                **job["entry"],
                "source_wav_sha256": sha256_bytes(payload),
                "n_bytes": len(payload),
            },
            "payload": payload,
        }

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for index, result in enumerate(pool.map(run_job, jobs), 1):
            completed.append(result)
            if index % 50 == 0 or index == len(jobs):
                print(f"synthesized {index}/{len(jobs)}", flush=True)

    observations = [item["entry"] for item in completed if item["kind"] == "observation"]
    smoke_observations = [item["entry"] for item in completed if item["kind"] == "smoke"]
    observations.sort(key=lambda entry: entry["observation_index"])
    smoke_observations.sort(key=lambda entry: entry["publication_index"])
    if len(observations) != EXPECTED_OBSERVATION_COUNT or len(smoke_observations) != 2:
        raise SynthesisError("synthesis did not produce the exact sealed cardinalities")

    recipe = seal_mapping(
        {
            "schema_version": 1,
            "kind": TTS_RECIPE_KIND,
            "engine": {
                "engine": TTS_ENGINE,
                "endpoint": TTS_ENDPOINT,
                "model": TTS_MODEL,
                "response_format": TTS_RESPONSE_FORMAT,
                "input_policy": TTS_INPUT_POLICY,
                "synthesis_policy": TTS_SYNTHESIS_POLICY,
                "voices": list(TTS_VARIANTS),
            },
            "synthesized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "observations": observations,
            "smoke_observations": smoke_observations,
        },
        "recipe_sha256",
    )
    recipe_root = output_root / recipe["recipe_sha256"]
    payload_by_path = {item["entry"]["wav_relative_path"]: item["payload"] for item in completed}
    for relative, payload in payload_by_path.items():
        destination = recipe_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (recipe_root / "recipe.json").write_text(
        json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    )
    total_bytes = sum(len(payload) for payload in payload_by_path.values())
    print(f"recipe_sha256: {recipe['recipe_sha256']}")
    print(f"observations: {len(observations)}  smoke: {len(smoke_observations)}")
    print(f"total audio bytes: {total_bytes}")
    print(
        "upload: uv run modal volume put audiolens-vol "
        f"{recipe_root} audio-workspace-eval/source-stimuli/{recipe['recipe_sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
