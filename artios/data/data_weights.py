"""Typed schema, validation, and weight calculation for training indices DataFrames.

The indices CSV describes training examples (chains and interfaces) from PDB
structures. This module provides:

- Enums for molecule types, row types, and interaction groups
- Pandera schema for validation with clear error messages
- Weight calculation functions including protein-cluster-balanced sampling
"""

from enum import Enum
from typing import Any, Optional, Union

import pandas as pd
import pandera as pa
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


# Interaction groups that involve exactly one protein side
PROTEIN_PARTNER_GROUPS = frozenset({MolTypeGroup.LIGAND_PROT, MolTypeGroup.NUC_PROT})

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


# ── Helper functions ───────────────────────────────────────────────────────


def get_protein_cluster_id(row: pd.Series) -> Optional[str]:
    """Extract the protein-side cluster ID from an indices row.

    Returns the protein cluster ID for:
      - intra_prot chains (cluster_1_id)
      - Interfaces with exactly one protein side (the protein side's cluster_id)

    Returns None for prot_prot interfaces, non-protein chains, or rows without
    a protein side.
    """
    mol_type_group = row["mol_type_group"]

    if mol_type_group == MolTypeGroup.INTRA_PROT:
        return row["cluster_1_id"]

    if row["type"] != RowType.INTERFACE:
        return None

    mol_1 = row["mol_1_type"]
    mol_2 = row["mol_2_type"]

    if mol_1 == MolType.PROT and mol_2 != MolType.PROT:
        return row["cluster_1_id"]
    if mol_2 == MolType.PROT and mol_1 != MolType.PROT:
        return row["cluster_2_id"]

    return None


# ── Filtering ──────────────────────────────────────────────────────────────


def filter_artifact_ligands(indices_df: pd.DataFrame) -> pd.DataFrame:
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
    scales down over-budget interface weights, and sets monomer (intra_prot)
    weights to fill the gap so each protein cluster's total weight ≈ tau.

    Only applies to interfaces with exactly one protein side (e.g., ligand_prot,
    nuc_prot). Prot-prot interfaces keep their original weights.
    Monomer entries from protein clusters with no interfaces keep their original
    weights (they have nothing to balance against).

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

    # Extract protein cluster ID for relevant rows
    prot_cluster = pd.Series(None, index=indices_df.index, dtype=object)

    # For intra_prot chains: the protein cluster is cluster_1_id
    is_intra_prot = indices_df["mol_type_group"] == MolTypeGroup.INTRA_PROT
    prot_cluster[is_intra_prot] = indices_df.loc[is_intra_prot, "cluster_1_id"]

    # For interfaces with exactly one protein side
    is_interface = indices_df["type"] == RowType.INTERFACE
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
    is_monomer = is_intra_prot & prot_cluster.notna()

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

    # Capped W_k
    W_k_capped = W_k.clip(upper=tau)

    # Monomer weights: (tau - W_k_capped) / |M_k|
    # Only for clusters that have interfaces — interface-free clusters keep original weights
    monomer_prot_clusters = prot_cluster[is_monomer]
    has_interfaces = monomer_prot_clusters.isin(W_k.index)
    is_rebalanced_monomer = is_monomer & prot_cluster.isin(W_k.index)

    rebal_prot_clusters = prot_cluster[is_rebalanced_monomer]
    M_k_size = rebal_prot_clusters.value_counts()

    monomer_W_capped = rebal_prot_clusters.map(W_k_capped)
    monomer_M_size = rebal_prot_clusters.map(M_k_size)

    new_monomer_weights = ((tau - monomer_W_capped) / monomer_M_size).clip(lower=0.0)

    old_monomer_weight = indices_df.loc[is_monomer, "weights"].sum()
    indices_df.loc[is_rebalanced_monomer, "weights"] = new_monomer_weights.values
    new_monomer_weight = indices_df.loc[is_monomer, "weights"].sum()

    n_skipped = (~has_interfaces).sum()
    logger.info(
        f"[protein_cluster_balancing] {n_skipped} monomer rows from "
        f"interface-free clusters kept original weights"
    )

    logger.info(
        f"[protein_cluster_balancing] Monomer weight: "
        f"{old_monomer_weight:.0f} -> {new_monomer_weight:.0f}"
    )

    return indices_df


def calc_weights_for_df(
    indices_df: pd.DataFrame,
    beta_dict: dict[str, Any],
    alpha_dict: dict[str, Any],
    balance_protein_clusters: bool = False,
    balance_percentile: float = 0.9,
    balance_mol_type_groups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Calculate sampling weights for each row in the indices DataFrame.

    Steps:
      1. Filter artifact ligands (if PLINDER columns present)
      2. Compute base weight per row: beta * alpha_sum / cluster_size / member_num
      3. Optionally apply protein-cluster-balanced sampling

    Args:
        indices_df: Training indices DataFrame (validated against IndicesSchema).
        beta_dict: Weight multipliers per row type (chain/interface).
        alpha_dict: Weight multipliers per molecule type (prot/nuc/ligand).
        balance_protein_clusters: Whether to apply protein-cluster-balanced sampling.
        balance_percentile: Percentile for the protein cluster weight cap.
        balance_mol_type_groups: Interface mol_type_groups to include in balancing.

    Returns:
        DataFrame with 'weights' column added.
    """
    indices_df = filter_artifact_ligands(indices_df)

    # Specific to assembly, and entities (chain or interface)
    indices_df["pdb_sorted_entity_id"] = indices_df.apply(
        lambda x: f"{x['pdb_id']}_{x['assembly_id']}_{'_'.join(sorted([str(x['entity_1_id']), str(x['entity_2_id'])]))}",
        axis=1,
    )

    entity_member_num_dict = {}
    for pdb_sorted_entity_id, sub_df in indices_df.groupby("pdb_sorted_entity_id"):
        entity_member_num_dict[pdb_sorted_entity_id] = len(sub_df)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.apply(
        lambda x: entity_member_num_dict[x["pdb_sorted_entity_id"]], axis=1
    )

    cluster_size_record = {}
    for cluster_id, sub_df in indices_df.groupby("cluster_id"):
        cluster_size_record[cluster_id] = len(set(sub_df["pdb_sorted_entity_id"]))

    weights = []
    for _, row in indices_df.iterrows():
        data_type = row["type"]
        cluster_size = cluster_size_record[row["cluster_id"]]
        chain_count = {MolType.PROT: 0, MolType.NUC: 0, MolType.LIGAND: 0}
        for mol_type in [row["mol_1_type"], row["mol_2_type"]]:
            if mol_type in chain_count:
                chain_count[mol_type] += 1
        weight = get_weighted_pdb_weight(
            data_type=data_type,
            cluster_size=cluster_size,
            chain_count=chain_count,
            beta_dict=beta_dict,
            alpha_dict=alpha_dict,
        )
        weights.append(weight)
    indices_df["weights"] = weights / indices_df["pdb_sorted_entity_id_member_num"]

    if balance_protein_clusters:
        indices_df = apply_protein_cluster_balancing(
            indices_df,
            percentile=balance_percentile,
            balanced_mol_type_groups=balance_mol_type_groups,
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
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
