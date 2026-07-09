"""Full RAVDESS speech eval of the gemma-4-E2B-it text-fit lens, on Modal.

Stages RAVDESS speech (Zenodo, 1440 clips: 24 actors x 2 statements x 8 acted
emotions x intensities x 2 reps) on the audiolens-vol volume, runs every clip
through the model, and appends one JSON record per clip — mood-anchor masses
and span-mean top-k tokens over the audio positions, at several layers.
Resumable: clips already recorded are skipped on re-run, and a partial
trailing line (container death between flushes; Modal volumes background-
commit) is truncated away before appending.

    modal run scripts/modal_audio_eval.py --limit 3     # cheap smoke run
    modal run scripts/modal_audio_eval.py               # full corpus
    modal volume get audiolens-vol eval/ravdess_gemma-4-E2B-it.jsonl eval/
    uv run python scripts/analyze_audio_eval.py eval/ravdess_gemma-4-E2B-it.jsonl
"""

from __future__ import annotations

import modal

VOL_MOUNT = "/vol"
JLENS_PIN = "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"
RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
RAVDESS_SHA256 = "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657"
RAVDESS_N_CLIPS = 1440
MODEL_ID = "google/gemma-4-E2B-it"
READ_LAYERS = [23, 29, 33]  # L29 is the primary (sanity_check.py); neighbors for depth check
TOPK = 10

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=5.5",
        "accelerate",
        "huggingface_hub",
        "librosa",
        "soundfile",
        "pillow",
        "torchvision",
        f"jlens @ {JLENS_PIN}",
    )
    .env({"HF_HOME": f"{VOL_MOUNT}/hf"})
    .add_local_python_source("audiolens")
)

app = modal.App("audiolens-audio-eval", image=image)
vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)


def _stage_ravdess() -> str:
    """Download, verify, and extract RAVDESS speech onto the volume once.

    Atomic at each step: download to a temp file and rename only after the
    checksum matches; extract to a temp dir and rename at the end — so the
    `root` sentinel implies a complete corpus even across container deaths.
    """
    import hashlib
    import pathlib
    import shutil
    import urllib.request
    import zipfile

    root = pathlib.Path(f"{VOL_MOUNT}/ravdess")
    if root.is_dir():
        return str(root)
    print("staging RAVDESS speech corpus...")

    zip_path = pathlib.Path(f"{VOL_MOUNT}/ravdess_speech.zip")
    if not zip_path.exists():
        tmp_zip = zip_path.with_suffix(".part")
        with urllib.request.urlopen(RAVDESS_URL) as resp, open(tmp_zip, "wb") as f:
            shutil.copyfileobj(resp, f)
        digest = hashlib.sha256(tmp_zip.read_bytes()).hexdigest()
        if digest != RAVDESS_SHA256:
            tmp_zip.unlink()
            raise RuntimeError(f"RAVDESS zip sha256 mismatch: {digest}")
        tmp_zip.replace(zip_path)

    tmp_root = root.with_suffix(".extracting")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_root)
    tmp_root.replace(root)
    zip_path.unlink()
    vol.commit()
    return str(root)


def _resume_done_set(results_path) -> set[str]:
    """Clip names already recorded, truncating any partial trailing line
    (buffered writes + background volume commits can persist one)."""
    import json

    done: set[str] = set()
    if not results_path.exists():
        return done
    valid_bytes = 0
    with open(results_path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break
            try:
                done.add(json.loads(line)["clip"])
            except (json.JSONDecodeError, KeyError):
                break
            valid_bytes += len(line)
    if valid_bytes < results_path.stat().st_size:
        print(f"truncating partial tail of {results_path} to {valid_bytes} bytes")
        with open(results_path, "r+b") as f:
            f.truncate(valid_bytes)
    return done


@app.function(
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_eval(limit: int = 0) -> str:
    import json
    import pathlib

    import torch
    import transformers

    import jlens
    from jlens.hooks import ActivationRecorder

    from audiolens import anchor_token_ids, mood_readout, parse_ravdess_name, resolve_audio_token_id

    slug = MODEL_ID.split("/")[-1]
    ravdess_root = _stage_ravdess()
    results_path = pathlib.Path(f"{VOL_MOUNT}/eval/ravdess_{slug}.jsonl")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    clips = sorted(pathlib.Path(ravdess_root).rglob("*.wav"))
    if len(clips) != RAVDESS_N_CLIPS:
        raise RuntimeError(
            f"staged corpus has {len(clips)} clips, expected {RAVDESS_N_CLIPS}; "
            f"delete {ravdess_root} on the volume and re-run to re-stage"
        )

    done = _resume_done_set(results_path)
    todo = [c for c in clips if c.name not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(clips)} clips staged, {len(done)} done, {len(todo)} to run")
    if not todo:
        return f"{results_path}: already complete ({len(done)} clips)"

    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    tok = processor.tokenizer
    hf = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        f"{VOL_MOUNT}/lenses/{slug}_jacobian_lens.pt"
    )
    print(f"lens: {lens}  read layers: {READ_LAYERS}")

    audio_id = resolve_audio_token_id(hf.config, tok)
    anchors = anchor_token_ids(tok)

    with open(results_path, "a") as out:
        for i, wav in enumerate(todo):
            messages = [
                {"role": "user", "content": [{"type": "audio", "audio": str(wav)}]}
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            ).to("cuda")
            positions = (inputs["input_ids"][0] == audio_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise RuntimeError(f"{wav.name}: no audio soft tokens in the prefill")

            with torch.no_grad(), ActivationRecorder(model.layers, at=READ_LAYERS) as rec:
                hf(**inputs, use_cache=False)

            record = {
                "clip": wav.name,
                "meta": parse_ravdess_name(wav.stem),
                "n_audio_tokens": positions.numel(),
                "seq_len": inputs["input_ids"].shape[1],
                "layers": {},
            }
            for layer in READ_LAYERS:
                residual = rec.activations[layer][0][positions].float()
                logits = model.unembed(lens.transport(residual, layer)).float()
                mass, top_ids = mood_readout(logits, anchors, topk=TOPK)
                record["layers"][str(layer)] = {
                    "anchor_mass": mass,
                    "topk_ids": top_ids,
                    "topk_toks": [tok.decode([t]) for t in top_ids],
                }
            out.write(json.dumps(record) + "\n")

            if (i + 1) % 100 == 0 or i == len(todo) - 1:
                out.flush()
                vol.commit()
                print(f"{i + 1}/{len(todo)}  ({wav.name})")

    return f"{results_path}: {len(done) + len(todo)} clips recorded"


@app.local_entrypoint()
def main(limit: int = 0):
    print(run_eval.remote(limit=limit))
    slug = MODEL_ID.split("/")[-1]
    print(f"download: modal volume get audiolens-vol eval/ravdess_{slug}.jsonl eval/")
