from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import pathlib
import sys
from typing import Any

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

import modal_workspace_eval as workspace  # noqa: E402
import sanity_check  # noqa: E402


class CharacterTokenizer:
    bos_token_id = 900
    pad_token_id = 0
    eos_token_id = 901

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        assert add_special_tokens is False
        ids = [100 + ord(character) for character in text]
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, ids: list[int]) -> str:
        token = ids[0]
        if token in {self.bos_token_id, self.eos_token_id, self.pad_token_id}:
            return f"<{token}>"
        return chr(token - 100)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        continue_final_message: bool,
    ) -> str:
        assert tokenize is False
        assert not (add_generation_prompt and continue_final_message)
        rendered = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        return rendered + ("<assistant>" if add_generation_prompt else "")


class FormTokenizer(CharacterTokenizer):
    def __init__(self, forms: dict[str, list[int]]):
        self.forms = forms

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        if return_offsets_mapping:
            return super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                return_offsets_mapping=True,
            )
        assert add_special_tokens is False
        return {"input_ids": list(self.forms.get(text, [10, 11]))}

    def decode(self, ids: list[int]) -> str:
        return f"tok-{ids[0]}"


class ZeroTargetTokenizer(CharacterTokenizer):
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        result = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        if return_offsets_mapping and text.endswith("TARGET"):
            result["input_ids"] = result["input_ids"][:-6]
            result["offset_mapping"] = result["offset_mapping"][:-6]
        return result


def _tiny_spec(raw: bytes, **changes: Any) -> workspace.FixtureSpec:
    payload = json.loads(raw)
    names = [item["name"] for item in payload["items"]]
    values = {
        "slug": "tiny",
        "filename": "tiny.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_bytes": len(raw),
        "raw_count": len(names),
        "publication_count": len(names),
        "selected_name_sha256": workspace._digest(names),
        "item_keys": frozenset({"name", "prompt", "intermediates"}),
        "target_boundary": False,
        "minimum_eligible_items": len(names),
        "minimum_eligible_concepts": len(names),
    }
    values.update(changes)
    return workspace.FixtureSpec(**values)


def _tiny_fixture_bytes() -> bytes:
    return json.dumps(
        {
            "items": [
                {"name": "one", "prompt": "A.", "intermediates": ["alpha"]},
                {"name": "two", "prompt": "B.", "intermediates": ["beta"]},
            ]
        },
        indent=2,
    ).encode()


def test_evaluator_module_is_deployable_locally_and_bundle_import_safe(
    monkeypatch,
):
    assert workspace.image is not None
    assert workspace.app is not None
    assert workspace.vol is not None

    monkeypatch.setenv("AUDIOLENS_GIT_REVISION", workspace.GIT_REVISION)
    monkeypatch.setenv(
        "AUDIOLENS_WORKSPACE_EVAL_SOURCE_DIGEST", workspace.SOURCE_DIGEST
    )
    monkeypatch.setenv("AUDIOLENS_SOURCE_DIGEST", workspace.FIT_SOURCE_DIGEST)
    monkeypatch.setenv("AUDIOLENS_LOCK_SHA256", workspace.LOCK_SHA256)
    monkeypatch.delenv("AUDIOLENS_REPORT_INSPECTOR_ONLY", raising=False)
    source_path = pathlib.Path(__file__).parent.parent / "scripts" / (
        "modal_workspace_eval.py"
    )
    source = source_path.read_text(encoding="utf-8")
    bundled_module = type(sys)("modal_workspace_eval_bundle_regression")
    bundled_module.__file__ = (
        "/tmp/audiolens-eval-bundle/scripts/modal_workspace_eval.py"
    )
    monkeypatch.setitem(sys.modules, bundled_module.__name__, bundled_module)
    bundled_namespace = bundled_module.__dict__
    exec(compile(source, bundled_module.__file__, "exec"), bundled_namespace)
    assert bundled_namespace["image"] is None
    assert bundled_namespace["app"] is None
    assert bundled_namespace["vol"] is None
    assert callable(bundled_namespace["evaluate_workspace"])
    assert callable(bundled_namespace["load_completed_workspace_report"])
    assert bundled_namespace["FIT_SOURCE_DIGEST"] == workspace.FIT_SOURCE_DIGEST
    assert bundled_namespace["LOCK_SHA256"] == workspace.LOCK_SHA256

    monkeypatch.setenv("AUDIOLENS_REPORT_INSPECTOR_ONLY", "1")
    inspector_module = type(sys)("modal_workspace_eval_inspector_regression")
    inspector_module.__file__ = str(source_path)
    monkeypatch.setitem(sys.modules, inspector_module.__name__, inspector_module)
    inspector_namespace = inspector_module.__dict__
    exec(compile(source, str(source_path), "exec"), inspector_namespace)
    assert inspector_namespace["image"] is None
    assert inspector_namespace["app"] is None
    assert inspector_namespace["vol"] is None
    assert callable(inspector_namespace["load_completed_workspace_report"])


def test_frozen_fixture_constants_are_complete_and_nonplaceholder():
    assert [fixture.publication_count for fixture in workspace.FIXTURES] == [
        50,
        50,
        54,
        55,
        52,
        96,
    ]
    assert [fixture.raw_count for fixture in workspace.FIXTURES] == [
        102,
        93,
        107,
        55,
        98,
        96,
    ]
    for fixture in workspace.FIXTURES:
        assert len(fixture.sha256) == 64
        assert len(fixture.selected_name_sha256) == 64
        assert set(fixture.sha256) <= set("0123456789abcdef")
        assert set(fixture.selected_name_sha256) <= set("0123456789abcdef")
        assert fixture.n_bytes > 9_000
        assert workspace.JLENS_REVISION in fixture.url


def test_fixture_bytes_schema_and_selected_name_manifest_fail_closed():
    raw = _tiny_fixture_bytes()
    spec = _tiny_spec(raw)
    selected = workspace._decode_fixture(spec, raw)
    assert [item["name"] for item in selected] == ["one", "two"]

    with pytest.raises(workspace.WorkspaceEvalContractError, match="bytes"):
        workspace._decode_fixture(spec, raw + b"\n")
    corrupt = raw.replace(b'"alpha"', b'"alphx"')
    with pytest.raises(workspace.WorkspaceEvalContractError, match="SHA-256"):
        workspace._decode_fixture(spec, corrupt)
    wrong_schema = _tiny_spec(raw, item_keys=frozenset({"name", "prompt"}))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="schema"):
        workspace._decode_fixture(wrong_schema, raw)
    wrong_names = _tiny_spec(raw, selected_name_sha256="0" * 64)
    with pytest.raises(workspace.WorkspaceEvalContractError, match="selected-name"):
        workspace._decode_fixture(wrong_names, raw)

def test_fixture_stream_is_bounded_before_buffering_excess_bytes():
    raw = _tiny_fixture_bytes()
    spec = _tiny_spec(raw)
    exact = io.BytesIO(raw)
    assert workspace._read_bounded_fixture(exact, spec) == raw
    oversized = io.BytesIO(raw + b"hostile trailing bytes")
    with pytest.raises(workspace.WorkspaceEvalContractError, match="exceeds pinned"):
        workspace._read_bounded_fixture(oversized, spec)
    assert oversized.tell() == spec.n_bytes + 1


def test_fit_manifest_input_rejects_escape_symlink_nonfile_and_oversize(
    tmp_path: pathlib.Path,
):
    runs = tmp_path / "runs"
    runs.mkdir()
    valid = runs / "valid.json"
    valid.write_text("{}")
    assert workspace._validate_fit_manifest_input(valid, tmp_path) == valid

    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(workspace.WorkspaceEvalContractError, match="under /vol/runs"):
        workspace._validate_fit_manifest_input(outside, tmp_path)

    directory = runs / "directory.json"
    directory.mkdir()
    with pytest.raises(workspace.WorkspaceEvalContractError, match="regular file"):
        workspace._validate_fit_manifest_input(directory, tmp_path)

    symlink = runs / "symlink.json"
    symlink.symlink_to(outside)
    with pytest.raises(workspace.WorkspaceEvalContractError, match="nonsymlink"):
        workspace._validate_fit_manifest_input(symlink, tmp_path)

    oversized = runs / "oversized.json"
    oversized.write_bytes(b"x" * (workspace.FIT_MANIFEST_MAX_BYTES + 1))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="size"):
        workspace._validate_fit_manifest_input(oversized, tmp_path)


def test_source_mismatch_rejects_before_canonical_loader_or_lens_validation(
    tmp_path: pathlib.Path,
):
    runs = tmp_path / "runs"
    runs.mkdir()
    manifest = runs / "fit.json"
    manifest.write_text(json.dumps({"config": {"source": {"digest": "0" * 64}}}))
    calls = []

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("canonical loader must not run for mismatched source")

    with pytest.raises(workspace.WorkspaceEvalContractError, match="source digest"):
        workspace._load_source_bound_fit_manifest(manifest, tmp_path, loader)
    assert calls == []

    manifest.write_text(
        json.dumps(
            {"config": {"source": {"digest": workspace.FIT_SOURCE_DIGEST}}}
        )
    )

    def valid_loader(path, *, volume_root):
        calls.append((path, volume_root))
        return {"loaded": True}

    path, record = workspace._load_source_bound_fit_manifest(
        manifest, tmp_path, valid_loader
    )
    assert path == manifest
    assert record == {"loaded": True}
    assert calls == [(manifest, tmp_path)]


def test_eligibility_records_exact_and_leading_forms_and_stable_hash():
    tokenizer = FormTokenizer(
        {
            "Brazil": [1],
            " Brazil": [2],
            "26": [3, 4],
            " 26": [5, 6, 7],
        }
    )
    brazil = workspace._eligible_concept(tokenizer, "multihop", "item", 0, "Brazil")
    unsupported = workspace._eligible_concept(tokenizer, "multihop", "item", 1, "26")
    assert brazil["allowed_token_ids"] == [1, 2]
    assert [(entry["boundary"], entry["token_id"]) for entry in brazil["accepted"]] == [
        ("exact", 1),
        ("leading", 2),
    ]
    assert unsupported["eligible"] is False
    assert unsupported["exclusion_reason"] == "zero_single_token_forms"
    assert {entry["reason"] for entry in unsupported["exclusions"]} == {
        "not_single_token"
    }
    assert workspace._digest([brazil, unsupported]) == workspace._digest(
        copy.deepcopy([brazil, unsupported])
    )
    assert workspace._digest([brazil, unsupported]) == (
        "235c20bec88a6aca7dc7eb38ca1c86e13c2a920bd69850e2de23cdfeb1c57fb1"
    )
    changed = copy.deepcopy(brazil)
    changed["accepted"][0]["token_id"] = 9
    assert workspace._digest([changed, unsupported]) != workspace._digest(
        [brazil, unsupported]
    )


def test_order_ops_policy_expands_frozen_forms_and_rejects_unknown_key():
    assert workspace._forms_for_concept("order-ops", "multiplication") == (
        "multiplication",
        "*",
        "×",
        "times",
        "multiply",
    )
    assert workspace._forms_for_concept("multihop", "three") == ("three",)
    with pytest.raises(workspace.WorkspaceEvalContractError, match="unregistered"):
        workspace._forms_for_concept("order-ops", "power")


@pytest.mark.parametrize("distribution", ["multihop", "multilingual", "order-ops"])
def test_target_boundary_uses_complete_context_and_excludes_target(distribution: str):
    tokenizer = CharacterTokenizer()
    item = {
        "name": "boundary",
        "prompt": "answer is ",
        "target": "TARGET",
        "intermediates": ["answer"],
    }
    record = workspace._build_readout(tokenizer, distribution, item)
    target_ids = [100 + ord(character) for character in "TARGET"]
    assert record["target_token_ids"] == target_ids
    assert record["target_span"][0] == len(record["scored_prefix_input_ids"])
    assert record["scored_prefix_input_ids"][-1] == 100 + ord(" ")
    assert record["decoded_predecessor"] == " "
    assert record["scored_prefix_input_ids"] == record["full_context_input_ids"][
        : record["target_span"][0]
    ]
    assert record["context_kind"] == distribution


def test_zero_token_target_is_rejected():
    item = {
        "name": "empty-target-span",
        "prompt": "answer is ",
        "target": "TARGET",
        "intermediates": ["answer"],
    }
    with pytest.raises(workspace.WorkspaceEvalContractError, match="zero tokenizer tokens"):
        workspace._build_readout(ZeroTargetTokenizer(), "multihop", item)


def test_association_typo_poetry_and_multiturn_readout_positions():
    tokenizer = CharacterTokenizer()
    association = workspace._build_readout(
        tokenizer,
        "association",
        {"name": "a", "prompt": "evoked.", "intermediates": ["idea"]},
    )
    assert association["decoded_predecessor"] == "."
    assert association["scored_position"] == len("evoked.")

    typo = workspace._build_readout(
        tokenizer,
        "typo",
        {"name": "t", "prompt": "bad speling", "intermediates": ["spelling"]},
    )
    assert typo["decoded_predecessor"] == "g"
    assert typo["scored_position"] == len("bad speling")

    poetry = workspace._build_readout(
        tokenizer,
        "poetry",
        {
            "name": "p",
            "prompt": "first line\nsecond line\ncontinuation",
            "intermediates": ["rhyme"],
        },
    )
    assert poetry["decoded_predecessor"] == "\n"
    assert poetry["scored_position"] == len("first line\nsecond line\n")
    assert len(poetry["scored_prefix_input_ids"]) < len(poetry["full_context_input_ids"])

    multiturn = workspace._build_readout(
        tokenizer,
        "multihop",
        {
            "name": "m",
            "prompt": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer is "},
            ],
            "target": "X",
            "intermediates": ["bridge"],
        },
    )
    assert multiturn["decoded_predecessor"] == " "
    assert multiturn["target_token_ids"] == [100 + ord("X")]
    assert association["context_kind"] == "association"
    assert typo["context_kind"] == "typo"
    assert poetry["context_kind"] == "poetry"
    assert multiturn["context_kind"] == "multihop"


def test_boundary_validator_requires_exact_context_suffix_and_prefix_contracts():
    tokenizer = CharacterTokenizer()
    target = workspace._build_readout(
        tokenizer,
        "multihop",
        {
            "name": "target",
            "prompt": "answer is ",
            "target": "TARGET",
            "intermediates": ["bridge"],
        },
    )
    workspace._validate_boundary_record(
        target,
        expected_context_kind="multihop",
        target_boundary=True,
    )
    truncated_target = copy.deepcopy(target)
    truncated_target["target_span"][1] -= 1
    truncated_target["target_token_ids"] = truncated_target[
        "full_context_input_ids"
    ][slice(*truncated_target["target_span"])]
    with pytest.raises(workspace.WorkspaceEvalContractError, match="target boundary"):
        workspace._validate_boundary_record(
            truncated_target,
            expected_context_kind="multihop",
            target_boundary=True,
        )
    wrong_kind = copy.deepcopy(target)
    wrong_kind["context_kind"] = "order-ops"
    with pytest.raises(workspace.WorkspaceEvalContractError, match="context kind"):
        workspace._validate_boundary_record(
            wrong_kind,
            expected_context_kind="multihop",
            target_boundary=True,
        )

    nontarget = workspace._build_readout(
        tokenizer,
        "association",
        {"name": "association", "prompt": "evoked.", "intermediates": ["idea"]},
    )
    nontarget["scored_prefix_input_ids"][0] += 1
    with pytest.raises(workspace.WorkspaceEvalContractError, match="full-context prefix"):
        workspace._validate_boundary_record(
            nontarget,
            expected_context_kind="association",
            target_boundary=False,
        )


def test_real_gemma_trailing_space_and_numeric_hazards():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            workspace.MODEL_ID,
            revision=workspace.MODEL_REVISION,
            use_fast=True,
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"pinned Gemma tokenizer is absent from the local cache: {exc}")
    prompt = (
        "Fact: The ocean on the coast of the country where Carnival is most "
        "famously celebrated is the "
    )
    item = {
        "name": "carnival-ocean",
        "prompt": prompt,
        "target": "Atlantic",
        "intermediates": ["Brazil"],
    }
    prompt_only = workspace._token_ids(tokenizer, prompt)
    boundary = workspace._build_readout(tokenizer, "multihop", item)
    assert boundary["scored_prefix_input_ids"][1:] != prompt_only
    assert len(boundary["scored_prefix_input_ids"]) == boundary["target_span"][0]
    chat_boundary = workspace._build_readout(
        tokenizer,
        "multihop",
        {
            "name": "chat-boundary",
            "prompt": [{"role": "user", "content": "Answer briefly."}],
            "target": "Atlantic",
            "intermediates": ["Brazil"],
        },
    )
    assert chat_boundary["full_context_input_ids"][0] == tokenizer.bos_token_id
    assert chat_boundary["scored_prefix_input_ids"][0] == tokenizer.bos_token_id
    assert len(chat_boundary["scored_prefix_input_ids"]) == chat_boundary["target_span"][0]
    unsupported = workspace._eligible_concept(
        tokenizer, "multihop", "nhop-alphabet-element", 0, "26"
    )
    assert unsupported["eligible"] is False
    assert workspace._token_ids(tokenizer, "26") == [236778, 236825]


def test_batched_competition_rank_uses_variant_min_optimistic_ties_and_rejects_nonfinite():
    logits = torch.tensor([[0.0, 4.0, 4.0, 3.0, -1.0]])
    group_ids = torch.tensor([[0, 0], [0, 3], [1, 1], [2, 2]])
    group_mask = torch.tensor(
        [[True, False], [True, True], [True, False], [True, False]]
    )
    ranks = workspace._batched_group_ranks(
        logits, logits.sort(dim=-1).values, group_ids, group_mask
    )
    assert ranks.tolist() == [[4, 3, 1, 1]]
    nonfinite = torch.tensor([[0.0, float("nan")]])
    with pytest.raises(workspace.WorkspaceEvalContractError, match="nonfinite"):
        workspace._batched_group_ranks(
            nonfinite,
            nonfinite.sort(dim=-1).values,
            torch.tensor([[0]]),
            torch.tensor([[True]]),
        )
    with pytest.raises(workspace.WorkspaceEvalContractError, match="invalid"):
        workspace._batched_group_ranks(
            logits,
            logits.sort(dim=-1).values,
            torch.empty((0, 1), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.bool),
        )


def test_batched_production_ranks_match_naive_definition():
    logits = torch.tensor([[2.0, 2.0, 0.0, 4.0], [1.0, 3.0, 2.0, 0.0]])
    group_ids = torch.tensor([[0, 1], [2, 0], [3, 0]])
    group_mask = torch.tensor(
        [[True, True], [True, False], [True, False]]
    )
    actual = workspace._batched_group_ranks(
        logits, logits.sort(dim=-1).values, group_ids, group_mask
    )
    expected = []
    for row in logits:
        row_ranks = []
        for ids, mask in zip(group_ids, group_mask, strict=True):
            allowed = ids[mask]
            best = row[allowed].max()
            row_ranks.append(1 + int((row > best).sum()))
        expected.append(row_ranks)
    assert actual.tolist() == expected


def test_js_divergence_is_clamped_to_the_binary_information_bound():
    identical = torch.tensor([[2.0, 0.0]])
    assert workspace._js_divergence_nats(identical, identical).item() == 0.0
    separated_left = torch.tensor([[100.0, -100.0]])
    separated_right = torch.tensor([[-100.0, 100.0]])
    divergence = workspace._js_divergence_nats(separated_left, separated_right).item()
    assert (
        0.0
        <= divergence
        <= workspace.JS_DIVERGENCE_MAX_NATS
        + workspace.JS_DIVERGENCE_FLOAT32_TOLERANCE
    )
    assert divergence == pytest.approx(math.log(2.0), abs=1e-6)


def _rank_item(
    name: str,
    candidate: list[dict[int, int]],
    logit: list[dict[int, int]] | None = None,
) -> dict[str, Any]:
    concept_ids = [f"{name}:{index}" for index in range(len(candidate))]
    layers: dict[str, Any] = {}
    layer_ids = sorted(candidate[0])
    for layer in layer_ids:
        candidate_ranks = {
            concept_id: candidate[index][layer]
            for index, concept_id in enumerate(concept_ids)
        }
        logit_ranks = {
            concept_id: (logit or candidate)[index][layer]
            for index, concept_id in enumerate(concept_ids)
        }
        layers[str(layer)] = {
            "concept_ranks": {
                "candidate": candidate_ranks,
                "logit": logit_ranks,
                "transposed": dict(candidate_ranks),
                "permuted": dict(candidate_ranks),
            },
            "candidate_label_pool_ranks": dict(candidate_ranks),
            "motor": {
                "actual_final_rank": {"candidate": 1, "logit": 1},
                "candidate_logit_top1_agreement": True,
                "candidate_logit_js_nats": 0.1,
            },
        }
    return {
        "name": name,
        "included_in_metrics": True,
        "eligible_concept_ids": concept_ids,
        "layers": layers,
    }


def test_reducer_order_is_variant_min_then_layer_min_then_item_fraction_then_item_mean():
    first = _rank_item("first", [{0: 100, 1: 1}, {0: 2, 1: 2}])
    second = _rank_item("second", [{0: 1, 1: 100}])
    summary = workspace._summarize_variant([first, second], "candidate", (0, 1))
    assert summary["pass_at_k"]["1"] == 0.75
    assert summary["pass_at_k"]["2"] == 1.0
    assert summary["n_items"] == 2
    # Pooling concepts would be 2/3, so 3/4 proves equal item weighting.
    assert summary["pass_at_k"]["1"] != pytest.approx(2 / 3)

def test_motor_metrics_include_items_with_ineligible_intermediates():
    included = _rank_item("included", [{0: 1}])
    excluded = _rank_item("excluded", [{0: 1}])
    excluded["included_in_metrics"] = False
    excluded["eligible_concept_ids"] = []
    excluded["layers"]["0"]["motor"]["actual_final_rank"] = {
        "candidate": 100,
        "logit": 100,
    }
    summary = workspace._motor_region_summary([included, excluded], (0,))
    assert summary["n_items"] == 2
    assert summary["item_filter"] == "none"
    assert summary["next_token_agreement_at_k"]["candidate"]["1"] == 0.5
    assert summary["next_token_agreement_log_k_auc"]["candidate"] < 1.0


def test_log_k_auc_exact_identities_and_fixed_layer_partition():
    assert workspace._log_k_auc({k: 0.0 for k in workspace.KS}) == 0.0
    assert workspace._log_k_auc({k: 1.0 for k in workspace.KS}) == 1.0
    curve = {1: 0.0, 2: 1.0, 5: 1.0, 10: 1.0, 20: 1.0, 50: 1.0, 100: 1.0}
    expected = 1.0 - math.log(2.0) / (2.0 * math.log(100.0))
    assert workspace._log_k_auc(curve) == pytest.approx(expected)
    assert workspace.EARLY_LAYERS == tuple(range(13))
    assert workspace.CANDIDATE_LAYERS == tuple(range(13, 32))
    assert workspace.MOTOR_LAYERS == (32, 33)
    assert (*workspace.EARLY_LAYERS, *workspace.CANDIDATE_LAYERS, *workspace.MOTOR_LAYERS) == workspace.ALL_LAYERS


class TinyLens:
    def __init__(self):
        self.jacobians = {
            0: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            1: torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        }


def test_candidate_transposed_and_permuted_controls_are_distinct_and_exact():
    residual = torch.tensor([[2.0, 3.0]])
    jacobians = TinyLens().jacobians
    permuted_jacobians = {0: jacobians[1][[1, 0], :].contiguous()}
    variants = workspace._transport_variants(
        residual, 0, jacobians, permuted_jacobians
    )
    assert torch.equal(variants["candidate"], residual @ TinyLens().jacobians[0].T)
    assert torch.equal(variants["logit"], residual)
    assert torch.equal(variants["transposed"], residual @ TinyLens().jacobians[0])
    expected_permuted = residual @ TinyLens().jacobians[1][[1, 0], :].T
    assert torch.equal(variants["permuted"], expected_permuted)
    assert not torch.equal(variants["candidate"], variants["transposed"])
    assert not torch.equal(variants["candidate"], variants["permuted"])


def _stat_item(name: str, own_rank: int, other_rank: int) -> dict[str, Any]:
    item = _rank_item(name, [{0: own_rank}], [{0: own_rank + 1}])
    own = item["eligible_concept_ids"][0]
    other = "b:0" if own == "a:0" else "a:0"
    item["layers"]["0"]["candidate_label_pool_ranks"] = {
        own: own_rank,
        other: other_rank,
    }
    return item


def test_fixed_bootstrap_and_label_permutation_are_deterministic():
    by_distribution = {
        slug: [_stat_item("a", 1, 100), _stat_item("b", 1, 100)]
        for slug in workspace.FIXTURE_BY_SLUG
    }
    boot_one = workspace._bootstrap_deltas(
        by_distribution, layers=(0,), replicates=25, seed=17
    )
    boot_two = workspace._bootstrap_deltas(
        copy.deepcopy(by_distribution), layers=(0,), replicates=25, seed=17
    )
    assert boot_one == boot_two
    assert boot_one["replicate_sha256"] == boot_two["replicate_sha256"]

    perm_one = workspace._label_permutation_scores(
        by_distribution, layers=(0,), replicates=25, seed=19
    )
    perm_two = workspace._label_permutation_scores(
        copy.deepcopy(by_distribution), layers=(0,), replicates=25, seed=19
    )
    assert perm_one == perm_two
    assert perm_one["replicate_sha256"] == perm_two["replicate_sha256"]
    assert 0.0 <= perm_one["p99"] <= 1.0


def _passing_evidence() -> dict[str, Any]:
    return {
        "bootstrap_candidate_minus_logit_auc": {"lower_95": 0.01},
        "distribution_candidate_minus_logit_auc": {
            slug: 0.01 for slug in workspace.FIXTURE_BY_SLUG
        },
        "aggregate_band_auc": {
            "candidate": 0.8,
            "logit": 0.6,
            "transposed": 0.4,
            "permuted": 0.3,
        },
        "label_permutation": {"observed_candidate_auc": 0.8, "p99": 0.5},
        "candidate_intermediate_auc_by_region": {
            "early_l0_l12": 0.4,
            "candidate_l13_l31": 0.8,
            "motor_l32_l33": 0.3,
        },
        "candidate_next_token_agreement_at_k_by_region": {
            "early_l0_l12": {str(k): 0.2 for k in workspace.KS},
            "candidate_l13_l31": {str(k): 0.4 for k in workspace.KS},
            "motor_l32_l33": {str(k): 0.6 for k in workspace.KS},
        },
        "candidate_next_token_agreement_log_k_auc_by_region": {
            "early_l0_l12": 0.2,
            "candidate_l13_l31": 0.4,
            "motor_l32_l33": 0.6,
        },
        "candidate_logit_js_nats_by_region": {
            "early_l0_l12": 0.3,
            "candidate_l13_l31": 0.2,
            "motor_l32_l33": 0.1,
        },
    }


def test_adjudication_validated_and_no_band_paths_do_not_search():
    validated = workspace._adjudicate(_passing_evidence())
    assert validated["status"] == "validated"
    assert validated["failed_criteria"] == []
    assert validated["searched_alternate_bands"] is False

    failed_evidence = _passing_evidence()
    failed_evidence["aggregate_band_auc"]["candidate"] = 0.2
    no_band = workspace._adjudicate(failed_evidence)
    assert no_band["status"] == "no_band"
    assert "beats_transposed_control" in no_band["failed_criteria"]
    assert "beats_permuted_control" in no_band["failed_criteria"]
    assert no_band["searched_alternate_bands"] is False


def _fit_manifest(tmp_path: pathlib.Path) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path]:
    lens = tmp_path / "lenses" / "fit.pt"
    lens.parent.mkdir()
    lens.write_bytes(b"bound lens")
    manifest_path = tmp_path / "runs" / "fit.json"
    manifest_path.parent.mkdir()
    fit_config = {
        "model": {"id": workspace.MODEL_ID, "revision": workspace.MODEL_REVISION},
        "tokenizer": {"id": workspace.MODEL_ID, "revision": workspace.MODEL_REVISION},
        "dataset": {
            "requested_count": 1_000,
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "ordered_prompt_sha256": "1" * 64,
        },
        "jlens": {"revision": workspace.JLENS_REVISION},
        "prompt_policy": {"kind": "raw_text", "force_bos": True, "chat_template": False},
        "fit": {
            "source_layers": list(workspace.ALL_LAYERS),
            "target_layer": 34,
            "skip_first": 16,
            "max_seq_len": 128,
            "dim_batch": 128,
            "d_model": workspace.D_MODEL,
            "model_dtype": "bfloat16",
            "accumulation_dtype": "float32",
            "artifact_dtype": "float16",
            "attention_backend": "eager",
            "checkpoint_every": 5,
            "resume": True,
            "compile": False,
        },
        "lock": {
            "uv_lock_sha256": "5" * 64,
            "frozen": True,
            "dependency_group": "fit",
        },
        "source": {
            "git_revision": "2" * 40,
            "digest": workspace.FIT_SOURCE_DIGEST,
        },
        "runtime": {"packages": {}},
    }
    lens_meta = {
        "relative_path": "lenses/fit.pt",
        "sha256": workspace._sha256_file(lens),
        "bytes": lens.stat().st_size,
        "dtype": "float16",
        "n_prompts": 1_000,
        "d_model": workspace.D_MODEL,
        "shape": [workspace.D_MODEL, workspace.D_MODEL],
        "source_layers": list(workspace.ALL_LAYERS),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
    }
    manifest = {
        "kind": "canonical_text_jlens_fit",
        "status": "complete",
        "canonical": True,
        "fit_config_sha256": "4" * 64,
        "config": fit_config,
        "paths": {"manifest": "runs/fit.json", "lens": "lenses/fit.pt"},
        "lens": lens_meta,
    }
    manifest_path.write_text(json.dumps(manifest))
    return manifest, manifest_path, lens


def test_fit_manifest_lens_chain_binds_exact_paths_bytes_and_geometry(tmp_path: pathlib.Path):
    manifest, manifest_path, lens = _fit_manifest(tmp_path)
    identity = workspace._bind_fit_identity(manifest, manifest_path, lens, tmp_path)
    assert identity["fit_config_sha256"] == "4" * 64
    assert identity["lens_sha256"] == workspace._sha256_file(lens)
    assert identity["dataset"]["ordered_prompt_sha256"] == "1" * 64
    mismatched_source = copy.deepcopy(manifest)
    mismatched_source["config"]["source"]["digest"] = "0" * 64
    with pytest.raises(workspace.WorkspaceEvalContractError, match="source digest"):
        workspace._bind_fit_identity(
            mismatched_source, manifest_path, lens, tmp_path
        )
    with pytest.raises(workspace.WorkspaceEvalContractError, match="bound lens"):
        workspace._bind_fit_identity(manifest, manifest_path, tmp_path / "other.pt", tmp_path)
    lens.write_bytes(b"mutated lens")
    with pytest.raises(workspace.WorkspaceEvalContractError, match="bytes"):
        workspace._bind_fit_identity(manifest, manifest_path, lens, tmp_path)


@pytest.fixture
def tiny_report_benchmark(monkeypatch):
    specs = tuple(
        workspace.FixtureSpec(
            slug=fixture.slug,
            filename=fixture.filename,
            sha256=fixture.sha256,
            n_bytes=fixture.n_bytes,
            raw_count=1,
            publication_count=1,
            selected_name_sha256=workspace._digest([f"{fixture.slug}-item"]),
            item_keys=fixture.item_keys,
            target_boundary=fixture.target_boundary,
            minimum_eligible_items=1,
            minimum_eligible_concepts=1,
        )
        for fixture in workspace.FIXTURES
    )
    monkeypatch.setattr(workspace, "FIXTURES", specs)
    monkeypatch.setattr(
        workspace, "FIXTURE_BY_SLUG", {fixture.slug: fixture for fixture in specs}
    )


def _report_boundary(fixture: workspace.FixtureSpec) -> dict[str, Any]:
    if fixture.target_boundary:
        full = [2, 10, 11]
        prefix = [2, 10]
        span = [2, 3]
        target_ids = [11]
    else:
        full = prefix = [2, 10]
        span = None
        target_ids = []
    kind = fixture.slug
    return {
        "context_kind": kind,
        "full_context_sha256": "f" * 64,
        "full_context_input_ids": full,
        "scored_prefix_input_ids": prefix,
        "target_span": span,
        "target_token_ids": target_ids,
        "scored_position": len(prefix) - 1,
        "predecessor_token_id": prefix[-1],
        "decoded_predecessor": " token",
    }


def _report_concept(fixture: workspace.FixtureSpec) -> dict[str, Any]:
    authored = "3" if fixture.slug == "order-ops" else "concept"
    forms = list(workspace._forms_for_concept(fixture.slug, authored))
    accepted = []
    token_ids = []
    for form_index, form in enumerate(forms):
        for boundary_index, boundary in enumerate(("exact", "leading")):
            token_id = 100 + form_index * 2 + boundary_index
            token_ids.append(token_id)
            accepted.append(
                {
                    "form": form,
                    "boundary": boundary,
                    "rendered": form if boundary == "exact" else " " + form,
                    "token_id": token_id,
                    "decoded": f"tok-{token_id}",
                }
            )
    return {
        "concept_id": f"{fixture.slug}-item:0",
        "authored": authored,
        "forms": forms,
        "accepted": accepted,
        "allowed_token_ids": token_ids,
        "exclusions": [],
        "eligible": True,
        "exclusion_reason": None,
    }


def _report_runtime(*, fit_side: bool) -> dict[str, Any]:
    shared = {
        "packages": dict(workspace.EXPECTED_RUNTIME_PACKAGES),
        "python": "3.12.10",
        "cuda": "13.0",
        "device": "NVIDIA H100 80GB HBM3",
        "torch_cuda_alloc_conf": "expandable_segments:True",
    }
    if fit_side:
        return {
            **shared,
            "modal_image_id": "im-test-fit",
            "modal_function_timeout_seconds": 86_400,
        }
    return {**shared, "modal_environment": {"MODAL_IMAGE_ID": "im-test-eval"}}


def _report_fit_identity() -> dict[str, Any]:
    model = {"id": workspace.MODEL_ID, "revision": workspace.MODEL_REVISION}
    dataset = {
        "id": "Salesforce/wikitext",
        "config": "wikitext-103-raw-v1",
        "split": "train",
        "text_field": "text",
        "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "streaming": True,
        "trust_remote_code": False,
        "chunking": {
            "algorithm": "neuronpedia_concat_space_strip_emit_strict_gt_v1",
            "max_chars": 2_000,
            "min_tail_chars": 200,
        },
        "requested_count": 1_000,
        "ordered_prompt_sha256": "1" * 64,
    }
    geometry = {
        "source_layers": list(workspace.ALL_LAYERS),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
        "checkpoint_every": 5,
        "resume": True,
        "compile": False,
        "d_model": workspace.D_MODEL,
        "model_dtype": "bfloat16",
        "accumulation_dtype": "float32",
        "artifact_dtype": "float16",
        "attention_backend": "eager",
    }
    fit_lock = {
        "uv_lock_sha256": "5" * 64,
        "frozen": True,
        "dependency_group": "fit",
    }
    fit_source = {"git_revision": "2" * 40, "digest": "3" * 64}
    fit_runtime = _report_runtime(fit_side=True)
    prompt_policy = {
        "kind": "raw_text",
        "force_bos": True,
        "chat_template": False,
    }
    fit_config = {
        "schema_version": 1,
        "model": model,
        "tokenizer": model,
        "dataset": dataset,
        "jlens": {"revision": workspace.JLENS_REVISION},
        "prompt_policy": prompt_policy,
        "fit": geometry,
        "lock": fit_lock,
        "source": fit_source,
        "runtime": fit_runtime,
    }
    fit_digest = workspace._digest(fit_config)
    tag = f"gemma-4-E2B-it-jlens-{fit_digest}"
    return {
        "fit_config": fit_config,
        "fit_manifest_relative_path": f"runs/{tag}.json",
        "fit_config_sha256": fit_digest,
        "lens_relative_path": f"lenses/{tag}.pt",
        "lens_sha256": "b" * 64,
        "lens_bytes": 123,
        "lens_dtype": "float16",
        "lens_n_prompts": 1_000,
        "lens_d_model": workspace.D_MODEL,
        "lens_shape": [workspace.D_MODEL, workspace.D_MODEL],
        "source_layers": list(workspace.ALL_LAYERS),
        "target_layer": 34,
        "skip_first": 16,
        "max_seq_len": 128,
        "dim_batch": 128,
        "model": model,
        "tokenizer": model,
        "dataset": dataset,
        "jlens": {"revision": workspace.JLENS_REVISION},
        "prompt_policy": prompt_policy,
        "fit_geometry": geometry,
        "fit_lock": fit_lock,
        "fit_source": fit_source,
        "fit_runtime": fit_runtime,
    }


def _completed_report() -> dict[str, Any]:
    eligibility_distributions = {}
    scored_items = []
    for fixture in workspace.FIXTURES:
        name = f"{fixture.slug}-item"
        concept = _report_concept(fixture)
        boundary = _report_boundary(fixture)
        eligibility_item = {
            "name": name,
            "concepts": [concept],
            "eligible_concept_ids": [concept["concept_id"]],
            "included_in_metrics": True,
            "item_exclusion_reason": None,
            "boundary": boundary,
        }
        eligibility_distributions[fixture.slug] = {
            "selected_items": 1,
            "eligible_items": 1,
            "eligible_concepts": 1,
            "minimum_eligible_items": 1,
            "minimum_eligible_concepts": 1,
            "items": [eligibility_item],
        }
        layers = {}
        for layer in workspace.ALL_LAYERS:
            if layer in workspace.CANDIDATE_LAYERS:
                candidate_rank, actual_rank, divergence = 1, 20, 0.2
            elif layer in workspace.EARLY_LAYERS:
                candidate_rank, actual_rank, divergence = 2, 50, 0.3
            else:
                candidate_rank, actual_rank, divergence = 3, 1, 0.1
            concept_id = concept["concept_id"]
            layers[str(layer)] = {
                "concept_ranks": {
                    "candidate": {concept_id: candidate_rank},
                    "logit": {concept_id: candidate_rank + 1},
                    "transposed": {concept_id: candidate_rank + 2},
                    "permuted": {concept_id: candidate_rank + 3},
                },
                "candidate_label_pool_ranks": {concept_id: candidate_rank},
                "motor": {
                    "actual_final_rank": {
                        "candidate": actual_rank,
                        "logit": actual_rank + 1,
                    },
                    "candidate_logit_top1_agreement": False,
                    "candidate_logit_js_nats": divergence,
                },
            }
        scored_items.append(
            {
                "distribution": fixture.slug,
                "name": name,
                "included_in_metrics": True,
                "eligible_concept_ids": [concept["concept_id"]],
                "boundary": boundary,
                "base_model_target_competence": (
                    {
                        "first_target_token_id": 11,
                        "rank": 1,
                        "top1_match": False,
                        "used_as_filter": False,
                    }
                    if fixture.target_boundary
                    else None
                ),
                "actual_final_top1_id": 7,
                "actual_final_top1_token": "actual",
                "layers": layers,
            }
        )
    eligibility_body = {
        "schema_version": 1,
        "rule": workspace._eligibility_rule(),
        "distributions": eligibility_distributions,
    }
    eligibility = {
        **eligibility_body,
        "eligibility_sha256": workspace._digest(eligibility_body),
    }
    controls = workspace._control_permutations()
    fit_identity = _report_fit_identity()
    fixture_identity = [
        workspace._fixture_identity(fixture) for fixture in workspace.FIXTURES
    ]
    config = {
        "schema_version": workspace.WORKSPACE_REPORT_SCHEMA_VERSION,
        "fit": copy.deepcopy(fit_identity),
        "fixtures": fixture_identity,
        "eligibility_sha256": eligibility["eligibility_sha256"],
        "tokenizer_preflight": {
            "model": workspace.MODEL_ID,
            "revision": workspace.MODEL_REVISION,
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
        "metrics": workspace._metric_config(),
        "source": {"git_revision": "6" * 40, "digest": "7" * 64},
        "runtime": _report_runtime(fit_side=False),
    }
    summaries = workspace._build_summaries(scored_items)
    by_distribution = {
        slug: [item for item in scored_items if item["distribution"] == slug]
        for slug in workspace.FIXTURE_BY_SLUG
    }
    bootstrap = workspace._bootstrap_deltas(by_distribution)
    permutation = workspace._label_permutation_scores(by_distribution)
    evidence = workspace._adjudication_evidence(summaries, bootstrap, permutation)
    adjudication = workspace._adjudicate(evidence)
    assert adjudication["status"] == "no_band"
    report = {
        "schema_version": workspace.WORKSPACE_REPORT_SCHEMA_VERSION,
        "kind": workspace.WORKSPACE_REPORT_KIND,
        "status": "complete",
        "evaluation_config_sha256": workspace._digest(config),
        "config": config,
        "fit_identity": fit_identity,
        "fixtures": fixture_identity,
        "eligibility": eligibility,
        "controls": controls,
        "items": scored_items,
        "summaries": summaries,
        "statistics": {
            "bootstrap": bootstrap,
            "label_permutation": permutation,
        },
        "adjudication": adjudication,
    }
    return workspace._complete_report(report)


def test_completed_report_loads_and_thin_sanity_renderer_prints_frozen_results(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
):
    report = _completed_report()
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(report))
    loaded = workspace.load_completed_workspace_report(path)
    rendered = sanity_check.render_workspace_report(loaded)
    assert "status: no_band" in rendered
    assert "fit config: " + report["fit_identity"]["fit_config_sha256"] in rendered
    assert (
        f"lens: {report['fit_identity']['lens_relative_path']} "
        f"sha256={'b' * 64}"
    ) in rendered
    assert "association:" in rendered
    assert "all L0-L33" in rendered
    assert "candidate_l13_l31" in rendered
    assert "failed criteria:" in rendered
    assert "no workspace-like band validated" in rendered


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.update(status="pending"), "incomplete"),
        (
            lambda report: report["fit_identity"].update(fit_config_sha256="d" * 64),
            "identity chain",
        ),
        (
            lambda report: report["fit_identity"].update(lens_sha256="e" * 64),
            "identity chain",
        ),
        (
            lambda report: report["config"]["metrics"]["regions"].update(
                candidate_l13_l31=[14, 15]
            ),
            "metric/layer",
        ),
    ],
)
def test_incomplete_or_identity_mutated_reports_reject_even_with_resealed_content(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
    mutation,
    match: str,
):
    report = _completed_report()
    mutation(report)
    report["evaluation_config_sha256"] = workspace._digest(report["config"])
    report = workspace._complete_report(report)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match=match):
        workspace.load_completed_workspace_report(path)


def test_resealed_fit_digest_and_runtime_forgery_reject(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
):
    digest_report = _completed_report()
    forged_digest = "d" * 64
    forged_tag = f"gemma-4-E2B-it-jlens-{forged_digest}"
    for fit in (digest_report["fit_identity"], digest_report["config"]["fit"]):
        fit["fit_config_sha256"] = forged_digest
        fit["fit_manifest_relative_path"] = f"runs/{forged_tag}.json"
        fit["lens_relative_path"] = f"lenses/{forged_tag}.pt"
    digest_report["evaluation_config_sha256"] = workspace._digest(
        digest_report["config"]
    )
    digest_report = workspace._complete_report(digest_report)
    digest_path = tmp_path / "fit-digest.json"
    digest_path.write_text(json.dumps(digest_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="embedded config"):
        workspace.load_completed_workspace_report(digest_path)

    runtime_report = _completed_report()
    for fit in (runtime_report["fit_identity"], runtime_report["config"]["fit"]):
        fit["fit_runtime"]["packages"]["torch"] = "forged"
        fit["fit_config"]["runtime"]["packages"]["torch"] = "forged"
    runtime_report["evaluation_config_sha256"] = workspace._digest(
        runtime_report["config"]
    )
    runtime_report = workspace._complete_report(runtime_report)
    runtime_path = tmp_path / "fit-runtime.json"
    runtime_path.write_text(json.dumps(runtime_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="fit runtime"):
        workspace.load_completed_workspace_report(runtime_path)

    scorer_report = _completed_report()
    scorer_report["config"]["runtime"]["packages"]["torch"] = "forged"
    scorer_report["evaluation_config_sha256"] = workspace._digest(
        scorer_report["config"]
    )
    scorer_report = workspace._complete_report(scorer_report)
    scorer_path = tmp_path / "scorer-runtime.json"
    scorer_path.write_text(json.dumps(scorer_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="scorer runtime"):
        workspace.load_completed_workspace_report(scorer_path)


def test_resealed_eligibility_and_control_mutations_reject(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
):
    eligibility_report = _completed_report()
    eligibility_report["eligibility"]["distributions"]["association"]["items"][0][
        "concepts"
    ][0]["accepted"][0]["token_id"] = 999
    eligibility_report = workspace._complete_report(eligibility_report)
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="eligibility content"):
        workspace.load_completed_workspace_report(eligibility_path)

    control_report = _completed_report()
    control_report["controls"]["output_basis"][0:2] = reversed(
        control_report["controls"]["output_basis"][0:2]
    )
    control_report = workspace._complete_report(control_report)
    control_path = tmp_path / "control.json"
    control_path.write_text(json.dumps(control_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="control identity"):
        workspace.load_completed_workspace_report(control_path)


def test_resealed_rank_identity_and_js_bound_mutations_reject(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
):
    rank_report = _completed_report()
    item = next(
        item
        for item in rank_report["items"]
        if item["distribution"] == "association"
    )
    concept_id = item["eligible_concept_ids"][0]
    item["layers"]["0"]["candidate_label_pool_ranks"][concept_id] += 1
    rank_report = workspace._complete_report(rank_report)
    rank_path = tmp_path / "rank-identity.json"
    rank_path.write_text(json.dumps(rank_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="pool rank"):
        workspace.load_completed_workspace_report(rank_path)

    js_report = _completed_report()
    js_item = next(
        item for item in js_report["items"] if item["distribution"] == "association"
    )
    js_item["layers"]["0"]["motor"]["candidate_logit_js_nats"] = (
        workspace.JS_DIVERGENCE_MAX_NATS
        + 2 * workspace.JS_DIVERGENCE_FLOAT32_TOLERANCE
    )
    js_report = workspace._complete_report(js_report)
    js_path = tmp_path / "js-bound.json"
    js_path.write_text(json.dumps(js_report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="motor evidence"):
        workspace.load_completed_workspace_report(js_path)


@pytest.mark.parametrize("payload", [b"\xff", b"{", b"[]"])
def test_json_object_reader_wraps_encoding_truncation_and_root_shape(
    tmp_path: pathlib.Path,
    payload: bytes,
):
    path = tmp_path / "resume.json"
    path.write_bytes(payload)
    with pytest.raises(workspace.WorkspaceEvalContractError, match="invalid resume"):
        workspace._read_json_object(path, label="resume")
    with pytest.raises(workspace.WorkspaceEvalContractError, match="invalid workspace report"):
        workspace.load_completed_workspace_report(path)

    valid = tmp_path / "valid.json"
    valid.write_text('{"status":"pending"}')
    assert workspace._read_json_object(valid, label="resume") == {
        "status": "pending"
    }


def test_unresealed_mutation_and_bare_lens_reject(
    tmp_path: pathlib.Path,
    tiny_report_benchmark,
):
    report = _completed_report()
    report["summaries"]["equal_distribution_aggregate"]["layer_sets"][
        "candidate_l13_l31"
    ]["variants"]["candidate"]["log_k_auc"] = 0.99
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(report))
    with pytest.raises(workspace.WorkspaceEvalContractError, match="content digest"):
        workspace.load_completed_workspace_report(path)

    lens_path = tmp_path / "candidate.pt"
    lens_path.write_bytes(b"lens")
    with pytest.raises(workspace.WorkspaceEvalContractError, match="bare .pt"):
        workspace.load_completed_workspace_report(lens_path)
