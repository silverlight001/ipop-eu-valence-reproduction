import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from ipop.data import ChecksumMismatch, build_provenance_record, download_artifacts, load_manifest


def test_load_manifest_exposes_pinned_provenance(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "record_id": 24771186,
        "version": 1,
        "doi": "10.6084/m9.figshare.24771186.v1",
        "license": "CC BY 4.0",
        "artifacts": [{"name": "a.csv", "url": "https://example/a", "size": 3,
                       "md5": hashlib.md5(b"abc").hexdigest(),
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.record_id == 24771186
    assert manifest.license == "CC BY 4.0"
    assert manifest.artifacts[0].name == "a.csv"


def test_download_is_atomic_and_checksum_verified(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    digest = hashlib.md5(b"abc").hexdigest()
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3, "md5": digest,
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    def fetcher(url: str, destination: Path) -> None:
        destination.write_bytes(b"abc")

    paths = download_artifacts(load_manifest(manifest_path), tmp_path / "raw", fetcher=fetcher)

    assert paths == [tmp_path / "raw" / "a.csv"]
    assert paths[0].read_bytes() == b"abc"
    assert not (tmp_path / "raw" / "a.csv.part").exists()


def test_download_rejects_checksum_mismatch_and_removes_partial_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3,
                       "md5": hashlib.md5(b"abc").hexdigest(),
                       "sha256": hashlib.sha256(b"abc").hexdigest()}],
    }), encoding="utf-8")

    def corrupt_fetcher(url: str, destination: Path) -> None:
        destination.write_bytes(b"bad")

    with pytest.raises(ChecksumMismatch, match="a.csv"):
        download_artifacts(load_manifest(manifest_path), tmp_path / "raw", fetcher=corrupt_fetcher)

    assert not (tmp_path / "raw" / "a.csv").exists()
    assert not (tmp_path / "raw" / "a.csv.part").exists()


def test_provenance_records_both_hashes_and_retrieval_time(tmp_path: Path) -> None:
    payload = b"abc"
    artifact = tmp_path / "a.csv"
    artifact.write_bytes(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "CC BY 4.0",
        "artifacts": [{"name": "a.csv", "url": "memory://a", "size": 3,
                       "md5": hashlib.md5(payload).hexdigest(),
                       "sha256": hashlib.sha256(payload).hexdigest()}],
    }), encoding="utf-8")
    provenance = build_provenance_record(
        load_manifest(manifest_path), [artifact], "2026-09-14T00:00:00Z"
    )
    assert provenance["retrieved_at"] == "2026-09-14T00:00:00Z"
    assert provenance["artifacts"][0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_download_uses_verified_official_mirror_after_figshare_403(tmp_path: Path) -> None:
    payload = b"abc"
    manifest_path = tmp_path / "manifest.json"
    primary_url = "https://ndownloader.figshare.com/files/43535559"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{
            "name": "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv",
            "url": primary_url,
            "size": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }), encoding="utf-8")
    attempted_urls: list[str] = []

    def fetcher(url: str, destination: Path) -> None:
        attempted_urls.append(url)
        if url == primary_url:
            raise HTTPError(url, 403, "Forbidden", None, None)
        destination.write_bytes(payload)

    resolved_urls: dict[str, str] = {}
    paths = download_artifacts(
        load_manifest(manifest_path), tmp_path / "raw", fetcher=fetcher, resolved_urls=resolved_urls
    )
    provenance = build_provenance_record(
        load_manifest(manifest_path), paths, "2026-09-14T00:00:00Z", resolved_urls
    )

    assert attempted_urls == [
        primary_url,
        (
            "https://raw.githubusercontent.com/KRICT-DATA/IPOP-dataset-ver-3.0/main/"
            "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
        ),
    ]
    assert provenance["artifacts"][0]["retrieved_url"] == attempted_urls[-1]


def test_rerun_preserves_verified_mirror_source_and_enforces_pin(tmp_path: Path) -> None:
    payload = b"abc"
    manifest_path = tmp_path / "manifest.json"
    primary_url = "https://ndownloader.figshare.com/files/43535559"
    artifact_name = "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    manifest_path.write_text(json.dumps({
        "record_id": 1, "version": 1, "doi": "d", "license": "l",
        "artifacts": [{
            "name": artifact_name,
            "url": primary_url,
            "size": len(payload),
            "md5": hashlib.md5(payload).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    mirror_url = (
        "https://raw.githubusercontent.com/KRICT-DATA/IPOP-dataset-ver-3.0/main/"
        f"{artifact_name}"
    )

    def mirror_fetcher(url: str, destination: Path) -> None:
        if url == primary_url:
            raise HTTPError(url, 403, "Forbidden", None, None)
        destination.write_bytes(payload)

    first_sources: dict[str, str] = {}
    first_reused: set[str] = set()
    paths = download_artifacts(
        manifest,
        tmp_path / "raw",
        fetcher=mirror_fetcher,
        resolved_urls=first_sources,
        reused_artifacts=first_reused,
    )
    first = build_provenance_record(
        manifest, paths, "2026-09-14T00:00:00Z", first_sources, first_reused
    )

    def no_fetch(url: str, destination: Path) -> None:
        raise AssertionError(f"rerun should reuse verified file, not fetch {url}")

    second_sources = dict(first_sources)
    second_reused: set[str] = set()
    paths = download_artifacts(
        manifest,
        tmp_path / "raw",
        fetcher=no_fetch,
        resolved_urls=second_sources,
        reused_artifacts=second_reused,
    )
    second = build_provenance_record(
        manifest, paths, "2026-09-15T00:00:00Z", second_sources, second_reused
    )

    assert first["artifacts"][0]["retrieved_url"] == mirror_url
    assert first["artifacts"][0]["retrieval_status"] == "downloaded"
    assert second["artifacts"][0]["retrieved_url"] == mirror_url
    assert second["artifacts"][0]["retrieval_status"] == "reused_verified"
    assert second_reused == {artifact_name}

    paths[0].write_bytes(b"bad")
    with pytest.raises(ChecksumMismatch, match=artifact_name):
        download_artifacts(manifest, tmp_path / "raw", fetcher=no_fetch)
