"""Typed schema, validation, and weight calculation for training indices DataFrames.

The indices CSV describes training examples (chains and interfaces) from PDB
structures. This module provides:

- Enums for molecule types, row types, and interaction groups
- Pandera schema for validation with clear error messages
- Weight calculation functions including protein-cluster-balanced sampling
"""

from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from artios.utils.logger import get_logger

logger = get_logger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────


class MolType(str, Enum):
    PROT = "prot"
    NUC = "nuc"
    LIGAND = "ligand"


class RowType(str, Enum):
    CHAIN = "chain"
    INTERFACE = "interface"


class MolTypeGroup(str, Enum):
    """All known mol_type_group values in the indices CSV."""

    INTRA_PROT = "intra_prot"
    INTRA_LIGAND = "intra_ligand"
    INTRA_NUC = "intra_nuc"
    PROT_PROT = "prot_prot"
    LIGAND_PROT = "ligand_prot"
    NUC_PROT = "nuc_prot"
    LIGAND_LIGAND = "ligand_ligand"
    LIGAND_NUC = "ligand_nuc"
    NUC_NUC = "nuc_nuc"


class LigandCategory(str, Enum):
    PROPER = "proper"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


# Default groups to include in protein-cluster balancing
DEFAULT_BALANCED_GROUPS = [
    MolTypeGroup.LIGAND_PROT.value,
    MolTypeGroup.NUC_PROT.value,
]


# ── Pandera Schema ─────────────────────────────────────────────────────────


class IndicesSchema(pa.DataFrameModel):
    """Schema for the training indices DataFrame.

    Validates column presence, dtypes, and allowed values. Use with
    ``IndicesSchema.validate(df, lazy=True)`` to collect all errors at once.
    """

    pdb_id: Series[str]
    assembly_id: Series[int] = pa.Field(coerce=True)
    entity_1_id: Series[float] = pa.Field(coerce=True)
    chain_1_id: Series[str]
    mol_1_type: Series[str] = pa.Field(isin=[m.value for m in MolType])
    cluster_1_id: Series[str] = pa.Field(nullable=True)
    sub_mol_1_type: Series[str] = pa.Field(nullable=True)

    entity_2_id: Series[float] = pa.Field(nullable=True, coerce=True)
    chain_2_id: Series[str] = pa.Field(nullable=True)
    mol_2_type: Series[str] = pa.Field(nullable=True, isin=[m.value for m in MolType])
    cluster_2_id: Series[str] = pa.Field(nullable=True)
    sub_mol_2_type: Series[str] = pa.Field(nullable=True)

    cluster_id: Series[str] = pa.Field(nullable=True)
    type: Series[str] = pa.Field(isin=[r.value for r in RowType])
    mol_type_group: Series[str] = pa.Field(isin=[g.value for g in MolTypeGroup])
    eval_type: Series[str] = pa.Field(nullable=True)
    num_tokens: Series[int] = pa.Field(coerce=True)
    num_prot_chains: Series[int] = pa.Field(coerce=True)
    resolution: Series[float] = pa.Field(nullable=True, coerce=True)
    release_date: Series[str] = pa.Field(nullable=True)

    # Optional PLINDER enrichment columns — use Optional to mark as not required
    ligand_1_category: Optional[Series[str]] = pa.Field(
        nullable=True,
        isin=[c.value for c in LigandCategory],
    )
    ligand_2_category: Optional[Series[str]] = pa.Field(
        nullable=True,
        isin=[c.value for c in LigandCategory],
    )

    class Config:
        coerce = True
        strict = False  # allow extra columns (e.g., weights, pdb_sorted_entity_id)


def validate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an indices DataFrame against the schema.

    Raises pandera.errors.SchemaErrors with all violations collected.
    """
    return IndicesSchema.validate(df, lazy=True)


# ── Filtering ──────────────────────────────────────────────────────────────


def _filter_artifact_ligands(indices_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows involving artifact ligands (crystallization buffer, solvents, etc.).

    Removes rows where any ligand side is categorized as 'artifact' based on
    PLINDER annotations. Requires 'ligand_1_category' and/or 'ligand_2_category'
    columns to be present (added by enrich_indices_with_plinder.py).

    If the columns are not present, returns the dataframe unchanged.
    """
    has_col_1 = "ligand_1_category" in indices_df.columns
    has_col_2 = "ligand_2_category" in indices_df.columns

    if not has_col_1 and not has_col_2:
        logger.info("No ligand category columns found, skipping artifact filtering")
        return indices_df

    n_before = len(indices_df)
    mask = pd.Series(True, index=indices_df.index)

    if has_col_1:
        mask &= indices_df["ligand_1_category"] != LigandCategory.ARTIFACT
    if has_col_2:
        mask &= indices_df["ligand_2_category"] != LigandCategory.ARTIFACT

    indices_df = indices_df[mask].reset_index(drop=True)
    n_removed = n_before - len(indices_df)
    logger.info(
        f"[filter_artifact_ligands] Removed {n_removed}/{n_before} rows "
        f"({100 * n_removed / n_before:.1f}%) containing artifact ligands"
    )
    return indices_df


# ── Weight calculation ─────────────────────────────────────────────────────


def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """Get sample weight for a single chain or interface entry.

    Args:
        data_type: 'chain' or 'interface'.
        cluster_size: Number of unique entity-pairs in this cluster.
        chain_count: Counts per molecule type, e.g. {"prot": 1, "nuc": 0, "ligand": 1}.
        eps: Small value to avoid division by zero.
        beta_dict: Weight multipliers per row type. Default: chain=0.5, interface=1.
        alpha_dict: Weight multipliers per molecule type. Default: prot=3, nuc=3, ligand=1.

    Returns:
        Unnormalized sampling weight for this entry.
    """
    if not beta_dict:
        beta_dict = {RowType.CHAIN: 0.5, RowType.INTERFACE: 1}
    if not alpha_dict:
        alpha_dict = {MolType.PROT: 3, MolType.NUC: 3, MolType.LIGAND: 1}

    assert cluster_size > 0, f"cluster_size must be > 0, got {cluster_size}"
    assert data_type in (
        RowType.CHAIN,
        RowType.INTERFACE,
    ), f"data_type must be 'chain' or 'interface', got {data_type!r}"

    beta = beta_dict[data_type]
    alpha_sum = sum(
        alpha * chain_count.get(mol_type, 0) for mol_type, alpha in alpha_dict.items()
    )
    return beta * alpha_sum / (cluster_size + eps)


def apply_protein_cluster_balancing(
    indices_df: pd.DataFrame,
    percentile: float = 0.9,
    balanced_mol_type_groups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Apply protein-cluster-balanced sampling weights.

    Caps the total interface weight per protein cluster at tau = percentile(W_k),
    scaling down over-budget interface weights. All other rows (monomers,
    prot-prot interfaces, etc.) keep their original weights.

    Only applies to interfaces with exactly one protein side (e.g., ligand_prot,
    nuc_prot). Prot-prot interfaces keep their original weights.

    Args:
        indices_df: DataFrame with 'weights' column already computed.
        percentile: Percentile of W_k to use as cap (default 0.9 = p90).
        balanced_mol_type_groups: Interface mol_type_groups to include in
            balancing. Defaults to ["ligand_prot", "nuc_prot"].

    Returns:
        DataFrame with rebalanced weights.
    """
    if balanced_mol_type_groups is None:
        balanced_mol_type_groups = DEFAULT_BALANCED_GROUPS

    # Extract protein cluster ID for balanced interface rows
    is_interface = indices_df["type"] == RowType.INTERFACE
    prot_cluster = pd.Series(None, index=indices_df.index, dtype=object)

    prot_side_1 = (
        is_interface
        & (indices_df["mol_1_type"] == MolType.PROT)
        & (indices_df["mol_2_type"] != MolType.PROT)
    )
    prot_side_2 = (
        is_interface
        & (indices_df["mol_2_type"] == MolType.PROT)
        & (indices_df["mol_1_type"] != MolType.PROT)
    )
    prot_cluster[prot_side_1] = indices_df.loc[prot_side_1, "cluster_1_id"]
    prot_cluster[prot_side_2] = indices_df.loc[prot_side_2, "cluster_2_id"]

    # Balanced interface rows: one-protein-side interfaces in the target groups
    is_balanced_interface = (
        is_interface
        & indices_df["mol_type_group"].isin(balanced_mol_type_groups)
        & prot_cluster.notna()
    )

    # W_k: total interface weight per protein cluster
    interface_prot_clusters = prot_cluster[is_balanced_interface]
    W_k = (
        indices_df.loc[is_balanced_interface]
        .groupby(interface_prot_clusters)["weights"]
        .sum()
    )

    if len(W_k) == 0:
        logger.info(
            "[protein_cluster_balancing] No balanced interfaces found, skipping"
        )
        return indices_df

    tau = float(W_k.quantile(percentile))
    n_capped = int((W_k > tau).sum())
    total_weight_before = W_k.sum()

    logger.info(
        f"[protein_cluster_balancing] {len(W_k)} protein clusters, "
        f"tau={tau:.1f} (p{int(percentile * 100)}), "
        f"{n_capped} clusters capped ({100 * n_capped / len(W_k):.1f}%), "
        f"total interface weight: {total_weight_before:.0f}"
    )

    # Scale factor per protein cluster: min(1, tau / W_k)
    scale_k = (tau / W_k).clip(upper=1.0)

    # Apply scaling to balanced interface weights
    interface_scales = interface_prot_clusters.map(scale_k)
    indices_df.loc[is_balanced_interface, "weights"] *= interface_scales.values

    total_weight_after = indices_df.loc[is_balanced_interface, "weights"].sum()
    logger.info(
        f"[protein_cluster_balancing] Interface weight: "
        f"{total_weight_before:.0f} -> {total_weight_after:.0f} "
        f"({100 * (1 - total_weight_after / total_weight_before):.1f}% reduction)"
    )

    return indices_df


def apply_monomer_pdb_boost(
    indices_df: pd.DataFrame,
    target_fraction: float = 0.10,
) -> pd.DataFrame:
    """Boost weights of rows from PDBs that have no interface rows.

    Monomer-only PDBs (structures with only chain rows, no interfaces) are
    underrepresented in sampling because interface PDBs generate many more
    rows per structure. This multiplies all weights from monomer-only PDBs
    by a factor that achieves the target sampling fraction.

    Args:
        indices_df: DataFrame with 'weights' column already computed.
        target_fraction: Desired fraction of total sampling weight from
            monomer-only PDBs. Default 0.10 (10%).

    Returns:
        DataFrame with boosted monomer-only PDB weights.
    """
    pdb_has_interface = set(
        indices_df.loc[indices_df["type"] == RowType.INTERFACE, "pdb_id"].unique()
    )
    is_mono_only_pdb = ~indices_df["pdb_id"].isin(pdb_has_interface)

    n_mono_pdbs = is_mono_only_pdb.sum()
    if n_mono_pdbs == 0:
        return indices_df

    mono_w = indices_df.loc[is_mono_only_pdb, "weights"].sum()
    non_mono_w = indices_df.loc[~is_mono_only_pdb, "weights"].sum()

    if mono_w == 0 or target_fraction <= 0 or target_fraction >= 1:
        return indices_df

    boost = target_fraction * non_mono_w / (mono_w * (1 - target_fraction))
    indices_df.loc[is_mono_only_pdb, "weights"] *= boost

    n_mono_pdbs_unique = indices_df.loc[is_mono_only_pdb, "pdb_id"].nunique()
    n_total_pdbs = indices_df["pdb_id"].nunique()
    logger.info(
        f"[monomer_pdb_boost] {n_mono_pdbs_unique}/{n_total_pdbs} monomer-only PDBs "
        f"boosted {boost:.1f}x to reach {100 * target_fraction:.0f}% of sampling weight"
    )

    return indices_df


def calc_weights_for_df(
    indices_df: pd.DataFrame,
    beta_dict: dict[str, Any],
    alpha_dict: dict[str, Any],
    balance_protein_clusters: bool = False,
    balance_percentile: float = 0.9,
    balance_mol_type_groups: Optional[list[str]] = None,
    monomer_pdb_target_fraction: float = 0.0,
    should_filter_artifact_ligands: bool = False,
) -> pd.DataFrame:
    """Calculate sampling weights for each row in the indices DataFrame.

    Steps:
      1. Optionally filter artifact ligands (if PLINDER columns present)
      2. Compute base weight per row: beta * alpha_sum / cluster_size / member_num
      3. Optionally apply protein-cluster-balanced sampling
      4. Optionally boost monomer-only PDB weights

    Args:
        indices_df: Training indices DataFrame (validated against IndicesSchema).
        beta_dict: Weight multipliers per row type (chain/interface).
        alpha_dict: Weight multipliers per molecule type (prot/nuc/ligand).
        balance_protein_clusters: Whether to apply protein-cluster-balanced sampling.
        balance_percentile: Percentile for the protein cluster weight cap.
        balance_mol_type_groups: Interface mol_type_groups to include in balancing.
        monomer_pdb_target_fraction: Target sampling fraction for monomer-only PDBs.
            Set to 0 to disable (default).
        should_filter_artifact_ligands: Whether to filter out artifact ligands. Default False.

    Returns:
        DataFrame with 'weights' column added.
    """
    if should_filter_artifact_ligands:
        indices_df = _filter_artifact_ligands(indices_df)

    # Build entity key: pdb_assembly_sortedEntities
    # .to_numpy() avoids Arrow-backed Series which break np.minimum
    e1 = indices_df["entity_1_id"].astype(str).fillna("nan").to_numpy()
    e2 = indices_df["entity_2_id"].astype(str).fillna("nan").to_numpy()
    lo = np.minimum(e1, e2)
    hi = np.maximum(e1, e2)
    indices_df["pdb_sorted_entity_id"] = (
        indices_df["pdb_id"]
        + "_"
        + indices_df["assembly_id"].astype(str)
        + "_"
        + lo
        + "_"
        + hi
    )

    # Rows sharing an entity key (member count)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.groupby(
        "pdb_sorted_entity_id"
    )["pdb_sorted_entity_id"].transform("count")

    # Unique entity keys per cluster
    cluster_sizes = indices_df.groupby("cluster_id")["pdb_sorted_entity_id"].transform(
        "nunique"
    )

    # Vectorized weight: beta * alpha_sum / cluster_size / member_num
    # All arithmetic on numpy arrays to avoid Arrow dtype issues on Python 3.12
    _beta = {(k.value if hasattr(k, "value") else k): v for k, v in beta_dict.items()}
    _alpha = {(k.value if hasattr(k, "value") else k): v for k, v in alpha_dict.items()}
    beta = (
        pd.Series(indices_df["type"].to_numpy(), index=indices_df.index)
        .map(_beta)
        .to_numpy(dtype="float64")
    )
    alpha_sum = pd.Series(
        indices_df["mol_1_type"].to_numpy(), index=indices_df.index
    ).map(_alpha).fillna(0).to_numpy(dtype="float64") + pd.Series(
        indices_df["mol_2_type"].to_numpy(), index=indices_df.index
    ).map(_alpha).fillna(0).to_numpy(dtype="float64")
    eps = 1e-9
    indices_df["weights"] = (
        beta
        * alpha_sum
        / (cluster_sizes.to_numpy(dtype="float64") + eps)
        / indices_df["pdb_sorted_entity_id_member_num"].to_numpy(dtype="float64")
    )

    if balance_protein_clusters:
        indices_df = apply_protein_cluster_balancing(
            indices_df,
            percentile=balance_percentile,
            balanced_mol_type_groups=balance_mol_type_groups,
        )

    if monomer_pdb_target_fraction > 0:
        indices_df = apply_monomer_pdb_boost(
            indices_df,
            target_fraction=monomer_pdb_target_fraction,
        )

    return indices_df


def get_sample_weights(
    sampler_type: str,
    indices_df: Optional[pd.DataFrame] = None,
    beta_dict: dict = {
        RowType.CHAIN: 0.5,
        RowType.INTERFACE: 1,
    },
    alpha_dict: dict = {
        MolType.PROT: 3,
        MolType.NUC: 3,
        MolType.LIGAND: 1,
    },
    force_recompute_weight: bool = False,
    balance_protein_clusters: bool = False,
    balance_percentile: float = 0.9,
    balance_mol_type_groups: Optional[list[str]] = None,
    monomer_pdb_target_fraction: float = 0.0,
    should_filter_artifact_ligands: bool = False,
    length_reweight_gamma: float = 0.0,
    cluster_size_exponent: float = 0.75,
) -> Union[pd.Series, list[float]]:
    """Compute sample weights based on the specified sampler type.

    Args:
        sampler_type: 'weighted' or 'uniform'.
        indices_df: Training indices DataFrame.
        beta_dict: Weight multipliers per row type.
        alpha_dict: Weight multipliers per molecule type.
        force_recompute_weight: Recompute even if 'weights' column exists.
        balance_protein_clusters: Apply protein-cluster-balanced sampling.
        balance_percentile: Percentile for the protein cluster weight cap.
        balance_mol_type_groups: Interface groups to include in balancing.
        monomer_pdb_target_fraction: Target sampling fraction for monomer-only
            PDBs. Set to 0 to disable.
        should_filter_artifact_ligands: Whether to filter out artifact ligands. Default False.
        length_reweight_gamma: Power exponent for length-based reweighting.
            Weight per sample is num_tokens^gamma. 0 = uniform, 0.5 = sqrt,
            1.0 = linear (token-proportional). Only applies to uniform sampler.
        cluster_size_exponent: Power exponent for the 'cluster_balanced' sampler.
            Per-cluster aggregate sampling weight scales as cluster_size^exponent.
            0.0 = fully cluster-balanced (every cluster equally likely),
            0.5 = sqrt, 0.75 = default (mild down-weighting),
            1.0 = uniform per row (no balancing). Only applies to the
            'cluster_balanced' sampler.

    Returns:
        Sample weights as a Series (weighted) or list (uniform).

    Raises:
        ValueError: If sampler_type is not 'weighted' or 'uniform'.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
                balance_protein_clusters=balance_protein_clusters,
                balance_percentile=balance_percentile,
                balance_mol_type_groups=balance_mol_type_groups,
                monomer_pdb_target_fraction=monomer_pdb_target_fraction,
                should_filter_artifact_ligands=should_filter_artifact_ligands,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "cluster_balanced":
        assert indices_df is not None
        cluster_sizes = indices_df.groupby("cluster_id")["pdb_id"].transform("nunique")
        # Per-row weight = N^(exp-1)  →  per-cluster aggregate = N * N^(exp-1) = N^exp
        weights = cluster_sizes.astype("float32").pow(cluster_size_exponent - 1.0)
        n_clusters = indices_df["cluster_id"].nunique()
        logger.info(
            f"[cluster_balanced] {len(indices_df)} rows, {n_clusters} clusters, "
            f"exponent={cluster_size_exponent} "
            f"(per-cluster aggregate weight ∝ N^{cluster_size_exponent}), "
            f"largest cluster aggregate weight (vs singleton): "
            f"{cluster_sizes.max() ** cluster_size_exponent:.1f}x"
        )
        return weights / weights.sum()
    elif sampler_type == "uniform":
        assert indices_df is not None
        if length_reweight_gamma > 0:
            weights = (
                indices_df["num_tokens"].astype("float32") ** length_reweight_gamma
            )
            return weights / weights.sum()
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
