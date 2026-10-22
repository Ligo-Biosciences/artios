"""Contracts for locating pinned release assets (CCD files, model weights)."""

from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from artios import assets


@pytest.fixture(autouse=True)
def isolated_ccd_resolution(tmp_path, monkeypatch):
    """Point the repo-root lookup at an empty temp dir and clear the cache."""
    monkeypatch.delenv(assets.CCD_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(assets, "REPO_ROOT", tmp_path / "repo")
    assets.resolve_ccd_files.cache_clear()
    yield
    assets.resolve_ccd_files.cache_clear()


def _write_ccd_pair(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / assets.CCD_COMPONENTS.filename).write_text("cif")
    (directory / assets.CCD_RDKIT_MOLS.filename).write_text("pkl")


def test_env_dir_takes_precedence_over_repo_ccd_files(tmp_path, monkeypatch):
    _write_ccd_pair(tmp_path / "repo" / "ccd_files")
    _write_ccd_pair(tmp_path / "env")
    monkeypatch.setenv(assets.CCD_DIR_ENV_VAR, str(tmp_path / "env"))

    files = assets.resolve_ccd_files()

    assert files.components.parent == tmp_path / "env"
    assert files.rdkit_mols.parent == tmp_path / "env"


def test_env_dir_without_files_does_not_fall_back(tmp_path, monkeypatch):
    _write_ccd_pair(tmp_path / "repo" / "ccd_files")
    (tmp_path / "env").mkdir()
    monkeypatch.setenv(assets.CCD_DIR_ENV_VAR, str(tmp_path / "env"))

    with pytest.raises(FileNotFoundError, match=assets.CCD_DIR_ENV_VAR):
        assets.resolve_ccd_files()


@pytest.mark.parametrize(
    "present", [assets.CCD_COMPONENTS.filename, assets.CCD_RDKIT_MOLS.filename]
)
def test_half_present_pair_is_rejected(tmp_path, present):
    ccd_dir = tmp_path / "repo" / "ccd_files"
    ccd_dir.mkdir(parents=True)
    (ccd_dir / present).write_text("x")

    with pytest.raises(FileNotFoundError, match="same CCD snapshot"):
        assets.resolve_ccd_files()


def test_missing_local_files_download_pinned_pair(monkeypatch, tmp_path):
    requested = []

    def fake_download(asset):
        requested.append(asset)
        return tmp_path / asset.filename

    monkeypatch.setattr(assets, "download", fake_download)

    files = assets.resolve_ccd_files()

    assert requested == [assets.CCD_COMPONENTS, assets.CCD_RDKIT_MOLS]
    assert files.components.name == assets.CCD_COMPONENTS.filename


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    corrupt = tmp_path / "weights.pt"
    corrupt.write_bytes(b"not the released weights")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_: str(corrupt))

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        assets.download(assets.POTTS_CHECKPOINT)


def test_download_explains_private_repo_access(monkeypatch):
    def unauthorized(**_):
        raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unauthorized)

    with pytest.raises(RuntimeError, match="hf auth login"):
        assets.download(assets.POTTS_CHECKPOINT)
