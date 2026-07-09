from __future__ import annotations

import hashlib
import json

import pytest

from audiolens.hub import (
    HubBundleError,
    download_lens,
    prepare_hf_bundle,
    publish_hf_bundle,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_run(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    def output(name, content, kind="runtime_lens"):
        path = artifacts / f"{name}.pt"
        path.write_bytes(content)
        return {
            "kind": kind,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    run = {
        "completed": True,
        "runtime": {
            "python": "3.12",
            "modal_environment": {"MODAL_IMAGE_ID": "im-private-top-level"},
        },
        "config": {
            "schema_version": 1,
            "runtime": {
                "device": "H100",
                "modal_environment": {"MODAL_IMAGE_ID": "im-private-config"},
            },
        },
        "outputs": {
            "text400": output("text400", b"tiny text lens\n"),
            "mixed528": output("mixed528", b"tiny mixed lens\n"),
            "audio32": output("audio32", b"unselected runtime lens\n"),
            "text_checkpoint": output(
                "text_checkpoint",
                b"training checkpoint\n",
                kind="fp32_running_sum_checkpoint",
            ),
            "evaluation": output(
                "evaluation",
                b"evaluation output\n",
                kind="evaluation_records",
            ),
        },
    }
    run_path = tmp_path / "completed-run.json"
    _write_run(run_path, run)
    return run_path, run


def _write_run(path, run):
    path.write_text(json.dumps(run), encoding="utf-8")


class RecordingApi:
    def __init__(self):
        self.calls = []
        self.upload_result = object()

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return self.upload_result


def test_prepare_hf_bundle_copies_only_selected_verified_lenses_and_sanitizes_metadata(
    tmp_path,
):
    run_path, run = _completed_run(tmp_path)
    bundle_dir = tmp_path / "bundle"

    result = prepare_hf_bundle(
        run_path,
        bundle_dir,
        lenses=("mixed528", "text400"),
        artifact_license="cc-by-sa-4.0",
    )

    assert result == bundle_dir
    assert {entry.name for entry in bundle_dir.iterdir()} == {
        "mixed528.pt",
        "text400.pt",
        "README.md",
        "audiolens-run.json",
    }
    assert (bundle_dir / "mixed528.pt").read_bytes() == (
        tmp_path / "artifacts" / "mixed528.pt"
    ).read_bytes()
    assert (bundle_dir / "text400.pt").read_bytes() == (
        tmp_path / "artifacts" / "text400.pt"
    ).read_bytes()

    manifest = json.loads((bundle_dir / "audiolens-run.json").read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {"mixed528", "text400"}
    assert manifest["outputs"]["mixed528"]["path"] == "mixed528.pt"
    assert manifest["outputs"]["text400"]["path"] == "text400.pt"
    assert manifest["outputs"]["mixed528"]["sha256"] == run["outputs"]["mixed528"][
        "sha256"
    ]
    assert manifest["outputs"]["text400"]["bytes"] == run["outputs"]["text400"][
        "bytes"
    ]
    assert manifest["runtime"] == {"python": "3.12"}
    assert manifest["config"]["runtime"] == {"device": "H100"}


def test_prepare_hf_bundle_rejects_incomplete_run(tmp_path):
    run_path, run = _completed_run(tmp_path)
    run["completed"] = False
    _write_run(run_path, run)
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(HubBundleError, match="not complete"):
        prepare_hf_bundle(
            run_path,
            bundle_dir,
            lenses=("text400",),
            artifact_license="cc-by-sa-4.0",
        )

    assert not bundle_dir.exists()


def test_prepare_hf_bundle_rejects_duplicate_selection(tmp_path):
    run_path, _run = _completed_run(tmp_path)
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(HubBundleError, match="must be unique"):
        prepare_hf_bundle(
            run_path,
            bundle_dir,
            lenses=("text400", "text400"),
            artifact_license="cc-by-sa-4.0",
        )

    assert not bundle_dir.exists()


@pytest.mark.parametrize("name", ["text_checkpoint", "evaluation"])
def test_prepare_hf_bundle_rejects_non_runtime_outputs(tmp_path, name):
    run_path, _run = _completed_run(tmp_path)
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(HubBundleError, match="is not a runtime lens"):
        prepare_hf_bundle(
            run_path,
            bundle_dir,
            lenses=(name,),
            artifact_license="cc-by-sa-4.0",
        )

    assert not bundle_dir.exists()


@pytest.mark.parametrize(
    ("defect", "message"),
    [("bytes", "byte-size mismatch"), ("sha256", "SHA-256 mismatch")],
)
def test_prepare_hf_bundle_rejects_lens_with_bad_declared_integrity(
    tmp_path, defect, message
):
    run_path, run = _completed_run(tmp_path)
    if defect == "bytes":
        run["outputs"]["text400"]["bytes"] += 1
    else:
        run["outputs"]["text400"]["sha256"] = "0" * 64
    _write_run(run_path, run)
    bundle_dir = tmp_path / "bundle"

    with pytest.raises(HubBundleError, match=message):
        prepare_hf_bundle(
            run_path,
            bundle_dir,
            lenses=("text400",),
            artifact_license="cc-by-sa-4.0",
        )

    assert not bundle_dir.exists()


def test_publish_hf_bundle_creates_model_repo_before_upload_with_explicit_policy(tmp_path):
    run_path, _run = _completed_run(tmp_path)
    bundle_dir = prepare_hf_bundle(
        run_path,
        tmp_path / "bundle",
        lenses=("text400",),
        artifact_license="cc-by-sa-4.0",
    )
    api = RecordingApi()

    result = publish_hf_bundle(
        bundle_dir,
        "owner/audiolens",
        private=True,
        api=api,
    )

    assert result is api.upload_result
    assert api.calls == [
        (
            "create_repo",
            {
                "repo_id": "owner/audiolens",
                "repo_type": "model",
                "private": True,
                "exist_ok": True,
            },
        ),
        (
            "upload_folder",
            {
                "repo_id": "owner/audiolens",
                "repo_type": "model",
                "folder_path": str(bundle_dir),
                "commit_message": "Publish verified Audiolens runtime lenses",
            },
        ),
    ]


def test_publish_hf_bundle_rejects_extra_files_before_api_calls(tmp_path):
    run_path, _run = _completed_run(tmp_path)
    bundle_dir = prepare_hf_bundle(
        run_path,
        tmp_path / "bundle",
        lenses=("text400",),
        artifact_license="cc-by-sa-4.0",
    )
    (bundle_dir / "unmanifested.pt").write_bytes(b"must not upload\n")
    api = RecordingApi()

    with pytest.raises(HubBundleError, match="extras=.*unmanifested.pt"):
        publish_hf_bundle(
            bundle_dir,
            "owner/audiolens",
            private=False,
            api=api,
        )

    assert api.calls == []


def test_download_lens_forwards_pinned_source_and_verifies_checksum(tmp_path):
    downloaded = tmp_path / "downloaded.pt"
    downloaded.write_bytes(b"verified lens bytes\n")
    expected_sha256 = _sha256(downloaded)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    result = download_lens(
        "owner/audiolens",
        "mixed528.pt",
        revision="0123456789abcdef",
        expected_sha256=expected_sha256,
        verify_checksum=True,
        cache_dir=tmp_path / "hf-cache",
        download_fn=fake_download,
    )

    assert result == downloaded
    assert calls == [
        {
            "repo_id": "owner/audiolens",
            "repo_type": "model",
            "filename": "mixed528.pt",
            "revision": "0123456789abcdef",
            "cache_dir": str(tmp_path / "hf-cache"),
        }
    ]


def test_download_lens_rejects_checksum_mismatch(tmp_path):
    downloaded = tmp_path / "downloaded.pt"
    downloaded.write_bytes(b"tampered lens bytes\n")
    expected_sha256 = hashlib.sha256(b"expected lens bytes\n").hexdigest()

    def fake_download(**_kwargs):
        return str(downloaded)

    with pytest.raises(HubBundleError, match="SHA-256 mismatch"):
        download_lens(
            "owner/audiolens",
            "mixed528.pt",
            revision="0123456789abcdef",
            expected_sha256=expected_sha256,
            verify_checksum=True,
            cache_dir=tmp_path / "hf-cache",
            download_fn=fake_download,
        )


def test_download_lens_requires_checksum_before_download(tmp_path):
    def fail_if_called(**_kwargs):
        raise AssertionError("download must not start without a required checksum")

    with pytest.raises(HubBundleError, match="expected_sha256 is required"):
        download_lens(
            "owner/audiolens",
            "mixed528.pt",
            revision="0123456789abcdef",
            expected_sha256=None,
            verify_checksum=True,
            cache_dir=tmp_path / "hf-cache",
            download_fn=fail_if_called,
        )
