"""Pinned release assets hosted on the Hugging Face Hub.

Model weights and the Chemical Component Dictionary (CCD) are downloaded from
immutable Hub revisions and checked against their SHA-256 checksums. Downloads
are cached by ``huggingface_hub``, so later runs reuse them without network
access.
"""

import functools
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directory containing components.cif and components.cif.rdkit_mol.pkl.
CCD_DIR_ENV_VAR = "ARTIOS_CCD_DIR"


@dataclass(frozen=True)
class HubFile:
    """A file pinned to an immutable Hugging Face Hub revision."""

    repo_id: str
    filename: str
    revision: str
    sha256: str
    size: int
    repo_type: str = "model"


POTTS_CHECKPOINT = HubFile(
    repo_id="Ligo-Biosciences/ArtiosPotts",
    filename="35200_ema_0.9999.pt",
    revision="eccf8808a629b5de76167077cc7d70b67718b91b",
    sha256="019de1d4aebd2c8b7c6a80bdadd4f94691c1cd4ad7507c43352a22ac173ba538",
    size=10_434_785,
)

CCD_COMPONENTS = HubFile(
    repo_id="Ligo-Biosciences/Artios-CCD",
    filename="components.cif",
    revision="df1722f8aad777a3a4ec2474bb47fa97bee83f84",
    sha256="fe7c0bf2ca0a51531805f5a5ca7094281ed4d9e9f93b9275d9e6feebe2510892",
    size=427_767_291,
    repo_type="dataset",
)

CCD_RDKIT_MOLS = HubFile(
    repo_id="Ligo-Biosciences/Artios-CCD",
    filename="components.cif.rdkit_mol.pkl",
    revision="df1722f8aad777a3a4ec2474bb47fa97bee83f84",
    sha256="2d6caced2d26c62015115a1d0a50f4106755a300e0c2a2d2b5c101c9038dcbcd",
    size=125_903_802,
    repo_type="dataset",
)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: HubFile) -> Path:
    """Return a local path to ``asset``, downloading it if it is not cached.

    Raises:
        RuntimeError: If the repository cannot be accessed, the file is not
            cached while offline, or the checksum does not match.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError

    try:
        path = Path(
            hf_hub_download(
                repo_id=asset.repo_id,
                filename=asset.filename,
                revision=asset.revision,
                repo_type=asset.repo_type,
            )
        )
    except RepositoryNotFoundError as e:
        raise RuntimeError(
            f"Could not access {asset.repo_id} on the Hugging Face Hub. If the "
            "repository is private, log in with `hf auth login` or set HF_TOKEN "
            "to a token with read access."
        ) from e
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"{asset.filename} from {asset.repo_id} is not in the local cache and "
            "the Hugging Face Hub could not be reached."
        ) from e

    actual = sha256sum(path)
    if actual != asset.sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {asset.sha256}, got {actual}. "
            "Delete the file and download it again."
        )
    return path


@dataclass(frozen=True)
class CCDFiles:
    components: Path
    rdkit_mols: Path


@functools.cache
def resolve_ccd_files() -> CCDFiles:
    """Locate the CCD components file and its matching RDKit molecule cache.

    Locations are checked in order, and both files always come from the same one:

    1. The directory named by the ``ARTIOS_CCD_DIR`` environment variable.
    2. ``ccd_files/`` at the repository root (source checkouts).
    3. The pinned Artios-CCD release on the Hugging Face Hub.
    """
    env_dir = os.environ.get(CCD_DIR_ENV_VAR)
    if env_dir:
        files = _local_ccd_files(Path(env_dir))
        if files is None:
            raise FileNotFoundError(
                f"{CCD_DIR_ENV_VAR}={env_dir} does not contain "
                f"{CCD_COMPONENTS.filename} and {CCD_RDKIT_MOLS.filename}."
            )
        return files

    files = _local_ccd_files(REPO_ROOT / "ccd_files")
    if files is not None:
        return files

    logger.info(
        f"Using CCD files from {CCD_COMPONENTS.repo_id}@{CCD_COMPONENTS.revision[:8]}"
    )
    return CCDFiles(
        components=download(CCD_COMPONENTS), rdkit_mols=download(CCD_RDKIT_MOLS)
    )


def _local_ccd_files(directory: Path) -> CCDFiles | None:
    files = CCDFiles(
        components=directory / CCD_COMPONENTS.filename,
        rdkit_mols=directory / CCD_RDKIT_MOLS.filename,
    )
    present = [files.components.exists(), files.rdkit_mols.exists()]
    if not any(present):
        return None
    if not all(present):
        raise FileNotFoundError(
            f"{directory} must contain both {CCD_COMPONENTS.filename} and "
            f"{CCD_RDKIT_MOLS.filename} from the same CCD snapshot."
        )

    sizes = (files.components.stat().st_size, files.rdkit_mols.stat().st_size)
    if sizes != (CCD_COMPONENTS.size, CCD_RDKIT_MOLS.size):
        logger.warning(
            f"CCD files in {directory} differ from the pinned "
            f"{CCD_COMPONENTS.repo_id} release; ligand featurization may not match "
            "published results."
        )
    logger.info(f"Using CCD files from {directory}")
    return files
