"""Fit a Jacobian lens for an audio-input LLM's text decoder on a Modal H100.

Mirrors the Neuronpedia recipe (wikitext-103 raw text, max_chars 2000,
max_seq_len 128) so our instruct-model lens is comparable to their base-model
gemma-4-e2b reference. Fitting is text-only prefill through the multimodal
model's decoder — the lens then reads any position, including audio.

    modal run scripts/modal_fit_lens.py                      # gemma-4-E2B-it
    modal run scripts/modal_fit_lens.py --n-prompts 1000
    modal volume get audiolens-vol lenses/<file> lenses/     # download result

Checkpoints land on the audiolens-vol Volume; the fit resumes from its own
checkpoint if re-run, so preemptions and timeouts only cost the current prompt.
"""

from __future__ import annotations

import modal

VOL_MOUNT = "/vol"

JLENS_PIN = "git+https://github.com/anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=5.5",
        "datasets>=2.20",
        "huggingface_hub",
        f"jlens @ {JLENS_PIN}",
    )
    .env({"HF_HOME": f"{VOL_MOUNT}/hf"})
)

app = modal.App("audiolens-fit", image=image)
vol = modal.Volume.from_name("audiolens-vol", create_if_missing=True)


def _load_decoder(model_id: str):
    """Load a (possibly multimodal) HF model for text-only prefill."""
    import torch
    import transformers

    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
            print(f"loaded with {cls_name}: {type(model).__name__}")
            return model
        except (ValueError, OSError) as e:
            print(f"{cls_name} failed: {str(e)[:120]}")
    raise RuntimeError(f"no Auto class could load {model_id}")


def _wikitext_prompts(n_prompts: int, max_chars: int = 2000) -> list[str]:
    """Neuronpedia's prompt recipe: wikitext-103 train, non-trivial rows,
    truncated to max_chars."""
    from datasets import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    prompts: list[str] = []
    for row in rows:
        text = row["text"].strip()
        if len(text) >= 200:
            prompts.append(text[:max_chars])
        if len(prompts) >= n_prompts:
            break
    return prompts


@app.function(
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={VOL_MOUNT: vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def fit_lens(
    model_id: str = "google/gemma-4-E2B-it",
    n_prompts: int = 400,
    dim_batch: int = 128,
    max_seq_len: int = 128,
) -> str:
    import os

    import transformers

    import jlens

    slug = model_id.split("/")[-1]
    os.makedirs(f"{VOL_MOUNT}/lenses", exist_ok=True)
    os.makedirs(f"{VOL_MOUNT}/ckpt", exist_ok=True)

    hf = _load_decoder(model_id)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    print(f"wrapped: {model}")

    prompts = _wikitext_prompts(n_prompts)
    print(f"{len(prompts)} prompts")

    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        checkpoint_path=f"{VOL_MOUNT}/ckpt/{slug}.pt",
        checkpoint_every=5,
        resume=True,
    )
    out = f"{VOL_MOUNT}/lenses/{slug}_jacobian_lens.pt"
    lens.save(out)
    vol.commit()
    return f"{out} :: {lens!r}"


@app.local_entrypoint()
def main(
    model_id: str = "google/gemma-4-E2B-it",
    n_prompts: int = 400,
    dim_batch: int = 128,
    max_seq_len: int = 128,
):
    result = fit_lens.remote(
        model_id=model_id,
        n_prompts=n_prompts,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
    )
    print(result)
    print("download: modal volume get audiolens-vol "
          f"lenses/{model_id.split('/')[-1]}_jacobian_lens.pt lenses/")
