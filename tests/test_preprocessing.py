from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genospatial.preprocessing import (
    PreprocessingError,
    aggregate_subclusters,
    annotate_lr_with_datasplits,
    build_gene_chromosome_map,
    calculate_gene_weights_from_marker_tables,
    extract_ccc_feature_vectors,
    filter_spatial_groups,
    match_sequence_genes,
    prepare_training_data,
    split_genes_by_chromosome,
)


class TinyAdata:
    def __init__(self, obs, x, var_names, obsm):
        self.obs = obs.copy()
        self.X = np.asarray(x)
        self.var_names = pd.Index(var_names)
        self.obsm = {key: np.asarray(value) for key, value in obsm.items()}

    @property
    def n_obs(self):
        return len(self.obs)

    def __getitem__(self, index):
        rows = index[0] if isinstance(index, tuple) else index
        if isinstance(rows, pd.Series):
            rows = rows.to_numpy()
        positions = np.arange(self.n_obs)[rows]
        return TinyAdata(
            self.obs.iloc[positions],
            self.X[positions],
            self.var_names,
            {key: value[positions] for key, value in self.obsm.items()},
        )

    def copy(self):
        return TinyAdata(self.obs, self.X, self.var_names, self.obsm)


def test_gene_matching_uses_set_intersection_membership():
    result = match_sequence_genes(["g1", "g2", "g3"], ["g2", "g3", "g4"])
    assert set(result) == {"g2", "g3"}
    assert len(result) == 2


def test_gtf_first_gene_record_and_chromosome_split(tmp_path):
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        "chr1\tsrc\tgene\t1\t2\t.\t+\t.\tgene_id \"a\"; gene_name \"g1\";\n"
        "chr9\tsrc\tgene\t3\t4\t.\t+\t.\tgene_id \"a2\"; gene_name \"g1\";\n"
        "chr13\tsrc\tgene\t5\t6\t.\t+\t.\tgene_id \"b\"; gene_name \"g2\";\n"
        "chr16\tsrc\tgene\t7\t8\t.\t+\t.\tgene_id \"c\"; gene_name \"g3\";\n"
        "chr1\tsrc\texon\t1\t2\t.\t+\t.\tgene_id \"x\"; gene_name \"ignored\";\n",
        encoding="utf-8",
    )
    mapping = build_gene_chromosome_map(gtf, ["g1", "g2", "g3"])
    assert mapping == {"g1": "chr1", "g2": "chr13", "g3": "chr16"}

    train, validation, test = split_genes_by_chromosome(
        ["g3", "g1", "g2"],
        mapping,
        {
            "train_chromosomes": ["chr1"],
            "validation_chromosomes": ["chr13"],
            "test_chromosomes": ["chr16"],
        },
    )
    assert train == ["g1"]
    assert validation == ["g2"]
    assert test == ["g3"]


def test_lr_complex_expansion_and_test_priority():
    lr_info = {
        "lr_train": {"ligand": "g_train", "receptor": "r_other"},
        "lr_complex": {"ligand": "complex_a", "receptor": "g_train"},
        "lr_unknown": {"ligand": "x", "receptor": "y"},
    }
    db_complex = pd.DataFrame(
        [["g_validation", "g_test", ""], ["", "", ""]],
        index=["complex_a", "unused"],
    )
    adata = SimpleNamespace(uns={"LR_pair_information": lr_info})

    result = annotate_lr_with_datasplits(
        adata,
        db_complex,
        ["g_train"],
        ["g_validation"],
        ["g_test"],
        lr_information_key="LR_pair_information",
    )

    assert result.loc["lr_train", "Data_Split"] == "Train"
    assert result.loc["lr_complex", "has_val_gene"]
    assert result.loc["lr_complex", "has_test_gene"]
    assert result.loc["lr_complex", "Data_Split"] == "Test"
    assert result.loc["lr_unknown", "Data_Split"] == "Unknown"


def test_ccc_features_preserve_key_order_and_sum_direction():
    matrices = OrderedDict(
        [
            ("lr_a", np.array([[1, 2], [3, 4]], dtype=float)),
            ("lr_test", np.array([[9, 9], [9, 9]], dtype=float)),
            ("lr_b", np.array([[5, 6], [7, 8]], dtype=float)),
        ]
    )
    adata = SimpleNamespace(uns={"LR_cell_weight": matrices}, n_obs=2)

    sender, receiver, names = extract_ccc_feature_vectors(
        adata,
        excluded_lr_pairs=["lr_test"],
        lr_weight_key="LR_cell_weight",
    )

    assert names == ["lr_a", "lr_b"]
    np.testing.assert_array_equal(sender, [[3, 11], [7, 15]])
    np.testing.assert_array_equal(receiver, [[4, 12], [6, 14]])


def test_filtering_occurs_niche_then_subcluster():
    obs = pd.DataFrame(
        {
            "cell_type_sub": ["A", "A", "B", "B", "C"],
            "X_hg_final_cluster": ["n1", "n1", "n1", "n2", "n3"],
        },
        index=["c1", "c2", "c3", "c4", "c5"],
    )
    adata = TinyAdata(obs, np.ones((5, 2)), ["g1", "g2"], {"env": np.ones((5, 2)), "ccc": np.ones((5, 1))})

    filtered = filter_spatial_groups(
        adata,
        cell_type_column="cell_type_sub",
        spatial_cluster_column="X_hg_final_cluster",
        subcluster_column="spatial_sub_cluster",
        minimum_niche_cells=2,
        minimum_subcluster_cells=2,
    )

    assert list(filtered.obs.index) == ["c1", "c2"]
    assert filtered.obs["spatial_sub_cluster"].tolist() == ["A_n1", "A_n1"]


def test_aggregation_uses_receiver_ccc_means():
    obs = pd.DataFrame(
        {
            "cell_type_sub": ["A", "A", "B"],
            "spatial_sub_cluster": ["A_n1", "A_n1", "B_n2"],
        },
        index=["c1", "c2", "c3"],
    )
    adata = TinyAdata(
        obs,
        [[1, 3], [3, 5], [9, 7]],
        ["g1", "g2"],
        {
            "env": [[1, 2], [3, 4], [8, 6]],
            "receiver": [[10], [14], [20]],
        },
    )
    mapping = {"A": 0, "B": 1}

    names, major_ids, env, receiver, expression = aggregate_subclusters(
        adata,
        cell_type_to_id=mapping,
        cell_type_column="cell_type_sub",
        subcluster_column="spatial_sub_cluster",
        environment_key="env",
        ccc_receiver_key="receiver",
    )

    assert names == ["A_n1", "B_n2"]
    np.testing.assert_array_equal(major_ids, [0, 1])
    np.testing.assert_array_equal(env, [[2, 3], [8, 6]])
    np.testing.assert_array_equal(receiver, [[12], [20]])
    np.testing.assert_array_equal(expression, [[2, 4], [9, 7]])


def test_gene_weight_formula_and_missing_gene_failure():
    cell_markers = pd.DataFrame(
        {"names": ["g1", "g2"], "scores": [2.0, -1.0], "logfoldchanges": [1.0, -1.0]}
    )
    niche_markers = pd.DataFrame(
        {"names": ["g1", "g2"], "scores": [4.0, 1.0], "logfoldchanges": [2.0, 0.5]}
    )
    weights = calculate_gene_weights_from_marker_tables(
        cell_markers, niche_markers, ["g1"], ["g2"]
    )
    # Median of positive maxima [4, 1] is 2.5.
    assert weights == {"g1": 2.6, "g2": 1.4}

    with pytest.raises(PreprocessingError, match="g3"):
        calculate_gene_weights_from_marker_tables(
            cell_markers, niche_markers, ["g3"], []
        )


def test_full_authoritative_input_pipeline_with_synthetic_files(tmp_path):
    ad = pytest.importorskip("anndata")
    h5py = pytest.importorskip("h5py")

    cell_ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
    genes = ["g_train", "g_val", "g_test", "not_embedded"]
    expression = np.array(
        [
            [10.0, 1.0, 2.0, 0.0],
            [9.0, 1.0, 2.0, 0.0],
            [5.0, 5.0, 2.0, 0.0],
            [1.0, 10.0, 2.0, 0.0],
            [1.0, 9.0, 2.0, 0.0],
            [1.0, 8.0, 2.0, 0.0],
        ]
    )
    expression_adata = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"X_hg_final_cluster": ["n1", "n1", "n1", "n2", "n2", "n2"]},
            index=cell_ids,
        ),
        var=pd.DataFrame(index=genes),
    )
    expression_adata.obsm["X_hg_final_emb"] = np.array(
        [[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [8.0, 8.0], [10.0, 10.0], [12.0, 12.0]]
    )
    expression_path = tmp_path / "expression.h5ad"
    expression_adata.write_h5ad(expression_path)

    stcase_order = list(reversed(cell_ids))
    stcase = ad.AnnData(
        X=np.zeros((6, 1)),
        obs=pd.DataFrame(index=stcase_order),
        var=pd.DataFrame(index=["placeholder"]),
    )
    stcase.uns["LR_pair_information"] = {
        "lr_train": {"ligand": "g_train", "receptor": "x"},
        "lr_test": {"ligand": "g_test", "receptor": "x"},
        "lr_complex": {"ligand": "complex_a", "receptor": "x"},
    }
    stcase.uns["DB_complex"] = pd.DataFrame(
        [["g_val", "g_test"]], index=["complex_a"], columns=["subunit_1", "subunit_2"]
    )
    stcase.uns["LR_cell_weight"] = {
        "lr_train": np.arange(36, dtype=float).reshape(6, 6),
        "lr_test": np.ones((6, 6), dtype=float),
        "lr_complex": np.full((6, 6), 2.0),
    }
    stcase_path = tmp_path / "stcase.h5ad"
    stcase.write_h5ad(stcase_path)

    cell_type_path = tmp_path / "cell_types.csv"
    pd.DataFrame(
        {"label": ["A", "A", "B", "B", "B", "B"]}, index=cell_ids
    ).to_csv(cell_type_path)

    sequence_path = tmp_path / "sequence.h5"
    with h5py.File(sequence_path, "w") as handle:
        for offset, gene in enumerate(["g_train", "g_val", "g_test"]):
            handle.create_dataset(gene, data=np.full((3, 4), offset, dtype=np.float32))

    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "chr1\tsrc\tgene\t1\t2\t.\t+\t.\tgene_id \"a\"; gene_name \"g_train\";\n"
        "chr13\tsrc\tgene\t3\t4\t.\t+\t.\tgene_id \"b\"; gene_name \"g_val\";\n"
        "chr16\tsrc\tgene\t5\t6\t.\t+\t.\tgene_id \"c\"; gene_name \"g_test\";\n",
        encoding="utf-8",
    )
    config = {
        "inputs": {
            "expression_h5ad": str(expression_path),
            "stcase_h5ad": str(stcase_path),
            "sequence_hdf5": str(sequence_path),
            "gtf": str(gtf_path),
            "cell_type_csv": str(cell_type_path),
        },
        "data": {
            "cell_type_column": "cell_type_sub",
            "spatial_cluster_column": "X_hg_final_cluster",
            "environment_key": "X_hg_final_emb",
            "ccc_sender_key": "X_ccc_sender",
            "ccc_receiver_key": "X_ccc_receiver",
            "subcluster_column": "spatial_sub_cluster",
            "lr_information_key": "LR_pair_information",
            "lr_complex_key": "DB_complex",
            "lr_weight_key": "LR_cell_weight",
            "minimum_niche_cells": 2,
            "minimum_subcluster_cells": 2,
        },
        "split": {
            "train_chromosomes": ["chr1"],
            "validation_chromosomes": ["chr13"],
            "test_chromosomes": ["chr16"],
        },
        "model": {"seq_len": 3, "seq_channels": 4, "environment_dim": 2},
    }

    prepared = prepare_training_data(config)

    assert set(prepared.expression_genes) == {"g_train", "g_val", "g_test"}
    assert prepared.train_genes == ["g_train"]
    assert prepared.validation_genes == ["g_val"]
    assert prepared.test_genes == ["g_test"]
    assert prepared.lr_feature_names == ["lr_train"]
    assert set(prepared.subcluster_names) == {"A_n1", "B_n2"}
    assert set(prepared.cell_type_to_id) == {"A", "B"}
    assert prepared.mean_environment.shape == (2, 2)
    assert prepared.mean_receiver_ccc.shape == (2, 1)
    assert prepared.mean_expression.shape == (2, 3)
    assert set(prepared.gene_weights) == {"g_train", "g_val"}
