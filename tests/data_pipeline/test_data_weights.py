"""Tests for data_weights and dataset loading optimizations — verifies vectorized
implementations match original iterrows-based versions."""

import numpy as np
import pandas as pd
import pytest

from artios.data.data_weights import (
    DataWeightsConfig,
    MolType,
    RowType,
    calc_weights_for_df,
    get_sample_weights,
    get_weighted_pdb_weight,
)


def _calc_weights_reference(
    indices_df: pd.DataFrame,
    beta_dict: dict,
    alpha_dict: dict,
) -> pd.DataFrame:
    """Original iterrows-based implementation, kept as ground truth."""
    indices_df = indices_df.copy()

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
    return indices_df


def _make_test_df() -> pd.DataFrame:
    """Build a DataFrame covering chains, interfaces, NaN entity_2, duplicate
    entity keys, multiple clusters, and all three mol types."""
    rows = [
        # Cluster clA: 2 structures (1abc, 2def)
        # 1abc has a prot chain + 2 ligand-prot interfaces (same sorted entity pair)
        dict(
            pdb_id="1abc",
            assembly_id=1,
            entity_1_id=1.0,
            entity_2_id=float("nan"),
            chain_1_id="A",
            chain_2_id=None,
            mol_1_type="prot",
            mol_2_type=None,
            cluster_1_id="clA",
            cluster_2_id=None,
            cluster_id="clA",
            type="chain",
            mol_type_group="intra_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=100,
            num_prot_chains=1,
            resolution=2.0,
            release_date="2020-01-01",
        ),
        dict(
            pdb_id="1abc",
            assembly_id=1,
            entity_1_id=1.0,
            entity_2_id=2.0,
            chain_1_id="A",
            chain_2_id="B",
            mol_1_type="prot",
            mol_2_type="ligand",
            cluster_1_id="clA",
            cluster_2_id="clX",
            cluster_id="clA_clX",
            type="interface",
            mol_type_group="ligand_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=120,
            num_prot_chains=1,
            resolution=2.0,
            release_date="2020-01-01",
        ),
        # Same sorted entity pair (1.0, 2.0) but different chain — tests member_num > 1
        dict(
            pdb_id="1abc",
            assembly_id=1,
            entity_1_id=2.0,
            entity_2_id=1.0,
            chain_1_id="C",
            chain_2_id="A",
            mol_1_type="ligand",
            mol_2_type="prot",
            cluster_1_id="clX",
            cluster_2_id="clA",
            cluster_id="clA_clX",
            type="interface",
            mol_type_group="ligand_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=120,
            num_prot_chains=1,
            resolution=2.0,
            release_date="2020-01-01",
        ),
        # 2def: prot chain in same cluster clA
        dict(
            pdb_id="2def",
            assembly_id=1,
            entity_1_id=1.0,
            entity_2_id=float("nan"),
            chain_1_id="A",
            chain_2_id=None,
            mol_1_type="prot",
            mol_2_type=None,
            cluster_1_id="clA",
            cluster_2_id=None,
            cluster_id="clA",
            type="chain",
            mol_type_group="intra_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=200,
            num_prot_chains=1,
            resolution=1.5,
            release_date="2020-06-01",
        ),
        # Cluster clB_clB: prot-prot interface (single structure)
        dict(
            pdb_id="3ghi",
            assembly_id=1,
            entity_1_id=1.0,
            entity_2_id=2.0,
            chain_1_id="A",
            chain_2_id="B",
            mol_1_type="prot",
            mol_2_type="prot",
            cluster_1_id="clB",
            cluster_2_id="clB",
            cluster_id="clB_clB",
            type="interface",
            mol_type_group="prot_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=300,
            num_prot_chains=2,
            resolution=3.0,
            release_date="2021-01-01",
        ),
        # Cluster clC: nuc chain (single structure)
        dict(
            pdb_id="4jkl",
            assembly_id=1,
            entity_1_id=1.0,
            entity_2_id=float("nan"),
            chain_1_id="A",
            chain_2_id=None,
            mol_1_type="nuc",
            mol_2_type=None,
            cluster_1_id="clC",
            cluster_2_id=None,
            cluster_id="clC",
            type="chain",
            mol_type_group="intra_nuc",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=50,
            num_prot_chains=0,
            resolution=4.0,
            release_date="2019-01-01",
        ),
        # Cluster clD: nuc-prot interface (entity IDs are high to test lexicographic sort)
        dict(
            pdb_id="5mno",
            assembly_id=2,
            entity_1_id=10.0,
            entity_2_id=3.0,
            chain_1_id="A",
            chain_2_id="B",
            mol_1_type="nuc",
            mol_2_type="prot",
            cluster_1_id="clD",
            cluster_2_id="clE",
            cluster_id="clD_clE",
            type="interface",
            mol_type_group="nuc_prot",
            sub_mol_1_type=None,
            sub_mol_2_type=None,
            eval_type=None,
            num_tokens=250,
            num_prot_chains=1,
            resolution=2.5,
            release_date="2020-03-01",
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def weights_config():
    return DataWeightsConfig()


@pytest.fixture
def beta_dict(weights_config):
    return weights_config.beta_dict


@pytest.fixture
def alpha_dict(weights_config):
    return weights_config.alpha_dict


@pytest.fixture
def all_types_weighted_config():
    """Default ``alpha_nuc`` is 0.0, which zeroes the nuc rows in the test
    DataFrame. The cluster-size and positivity invariants below are about the
    weighting mechanics, so give every mol type a positive alpha."""
    return DataWeightsConfig(alpha_prot=3.0, alpha_nuc=3.0, alpha_ligand=1.0)


class TestCalcWeightsForDf:
    def test_matches_reference(self, beta_dict, alpha_dict, weights_config):
        """Vectorized implementation must produce identical weights to the
        original iterrows-based reference."""
        df = _make_test_df()

        ref = _calc_weights_reference(df.copy(), beta_dict, alpha_dict)
        new = calc_weights_for_df(df.copy(), weights_config)

        np.testing.assert_allclose(
            new["weights"].values,
            ref["weights"].values,
            rtol=1e-6,
        )
        assert list(new["pdb_sorted_entity_id"]) == list(ref["pdb_sorted_entity_id"])
        assert list(new["pdb_sorted_entity_id_member_num"]) == list(
            ref["pdb_sorted_entity_id_member_num"]
        )

    def test_member_num_divides_weight(self, weights_config):
        """Rows sharing a pdb_sorted_entity_id should have lower weight than
        singletons, all else equal."""
        df = _make_test_df()
        result = calc_weights_for_df(df.copy(), weights_config)

        # Rows 1 and 2 share entity key 1abc_1_1.0_2.0 (member_num=2)
        # Row 4 (3ghi interface) has member_num=1
        assert result.loc[1, "pdb_sorted_entity_id_member_num"] == 2
        assert result.loc[2, "pdb_sorted_entity_id_member_num"] == 2
        assert result.loc[4, "pdb_sorted_entity_id_member_num"] == 1

    def test_larger_cluster_lower_weight(self, all_types_weighted_config):
        """A chain in a 2-member cluster should weigh less than an identical
        chain in a singleton cluster (weight ~ 1/cluster_size)."""
        df = _make_test_df()
        result = calc_weights_for_df(df.copy(), all_types_weighted_config)

        # Row 0: prot chain in cluster clA (size 2 entities: 1abc and 2def)
        # Row 5: nuc chain in cluster clC (size 1 entity: 4jkl)
        # Both are chains with alpha=3, but clA has cluster_size=2 vs clC=1
        w_clA_chain = result.loc[0, "weights"]
        w_clC_chain = result.loc[5, "weights"]
        assert w_clA_chain < w_clC_chain

    def test_no_zero_weights(self, all_types_weighted_config):
        df = _make_test_df()
        result = calc_weights_for_df(df.copy(), all_types_weighted_config)
        assert (result["weights"] > 0).all()


class TestExclusionDict:
    """Vectorized exclusion filtering matches the original apply-based version."""

    def _apply_exclusion_reference(self, df, exclusion_dict):
        """Original apply-based implementation."""
        for col_name, exclusion_list in exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = df.apply(is_valid, axis=1)
            df = df[valid_mask].reset_index(drop=True)
        return df

    def _apply_exclusion_new(self, df, exclusion_dict):
        """Vectorized version (mirrors dataset.py)."""
        for col_name, exclusion_list in exclusion_dict.items():
            cols = col_name.split("|")
            if len(cols) == 1:
                valid_mask = ~df[cols[0]].isin(exclusion_list)
            else:
                combined = df[cols].astype(str).agg("|".join, axis=1)
                valid_mask = ~combined.isin(exclusion_list)
            df = df[valid_mask].reset_index(drop=True)
        return df

    def test_single_column_exclusion(self):
        df = _make_test_df()
        exclusion_dict = {"mol_1_type": ["nuc"]}
        ref = self._apply_exclusion_reference(df.copy(), exclusion_dict)
        new = self._apply_exclusion_new(df.copy(), exclusion_dict)
        assert len(ref) == len(new)
        assert list(ref["pdb_id"]) == list(new["pdb_id"])

    def test_multi_value_exclusion(self):
        df = _make_test_df()
        exclusion_dict = {"sub_mol_1_type": ["excluded_ligand"]}
        # No rows have sub_mol_1_type="excluded_ligand" in test data, so all pass
        ref = self._apply_exclusion_reference(df.copy(), exclusion_dict)
        new = self._apply_exclusion_new(df.copy(), exclusion_dict)
        assert len(ref) == len(new)

    def test_multi_column_pipe_exclusion(self):
        df = _make_test_df()
        exclusion_dict = {"mol_1_type|type": ["prot|chain"]}
        ref = self._apply_exclusion_reference(df.copy(), exclusion_dict)
        new = self._apply_exclusion_new(df.copy(), exclusion_dict)
        assert len(ref) == len(new)
        assert list(ref["pdb_id"]) == list(new["pdb_id"])


class TestGetSampleWeights:
    def test_uniform_honors_negative_length_gamma(self):
        df = pd.DataFrame({"num_tokens": [100, 400]})
        weights = get_sample_weights(
            sampler_type="uniform",
            indices_df=df,
            config=DataWeightsConfig(length_reweight_gamma=-1.0),
        )

        np.testing.assert_allclose(weights.to_numpy(), [0.8, 0.2], rtol=1e-6)

    def test_cluster_balanced_honors_negative_length_gamma(self):
        df = pd.DataFrame(
            {
                "cluster_id": ["cluster_a", "cluster_a", "cluster_b", "cluster_c"],
                "pdb_id": ["pdb_1", "pdb_2", "pdb_3", "pdb_4"],
                "num_tokens": [100, 400, 100, 200],
            }
        )
        config = DataWeightsConfig(
            cluster_size_exponent=0.5,
            length_reweight_gamma=-1.0,
        )

        weights = get_sample_weights(
            sampler_type="cluster_balanced",
            indices_df=df,
            config=config,
        )

        cluster_sizes = np.array([2, 2, 1, 1], dtype=np.float32)
        lengths = np.array([100, 400, 100, 200], dtype=np.float32)
        expected = (cluster_sizes ** (0.5 - 1.0)) * (lengths**-1.0)
        expected = expected / expected.sum()
        np.testing.assert_allclose(weights.to_numpy(), expected, rtol=1e-6)
        assert weights.iloc[0] > weights.iloc[1]


class TestDataWeightsConfig:
    def test_default_short_sequence_sampling_dial(self):
        config = DataWeightsConfig()

        assert config.cluster_size_exponent == 0.5
        assert config.length_reweight_gamma == -1.0

    def test_custom_config_changes_weights(self):
        """Non-default alpha/beta values produce different weights."""
        df = _make_test_df()
        default_result = calc_weights_for_df(df.copy())
        custom = DataWeightsConfig(alpha_ligand=10.0)
        custom_result = calc_weights_for_df(df.copy(), custom)

        # Rows with ligand mol_type should have higher weights
        ligand_rows = df["mol_1_type"].eq("ligand") | df["mol_2_type"].eq("ligand")
        assert (
            custom_result.loc[ligand_rows, "weights"]
            > default_result.loc[ligand_rows, "weights"]
        ).all()

    def test_rejects_invalid_cluster_size_exponent(self):
        with pytest.raises(ValueError, match="cluster_size_exponent"):
            DataWeightsConfig(cluster_size_exponent=1.5)

    def test_rejects_invalid_balance_percentile(self):
        with pytest.raises(ValueError, match="balance_percentile"):
            DataWeightsConfig(balance_percentile=-0.1)

    def test_accepts_negative_length_reweight_gamma(self):
        assert DataWeightsConfig(length_reweight_gamma=-1.0).length_reweight_gamma == -1

    def test_unexpected_kwargs_rejected(self):
        df = _make_test_df()
        with pytest.raises(TypeError):
            get_sample_weights(
                sampler_type="weighted",
                indices_df=df,
                beta_dict={RowType.CHAIN: 0.5, RowType.INTERFACE: 1},
            )
