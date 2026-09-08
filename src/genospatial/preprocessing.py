"""Faithful reconstruction of the v12_r mouse-brain preprocessing path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class PreprocessingError(RuntimeError):
    """Raised when authoritative inputs cannot support frozen preprocessing."""


@dataclass
class PreparedTrainingData:
    """Arrays and realized orders needed by the frozen training workflow."""

    subcluster_names: list[str]
    major_ids: np.ndarray
    mean_environment: np.ndarray
    mean_receiver_ccc: np.ndarray
    mean_expression: np.ndarray
    expression_genes: list[str]
    train_genes: list[str]
    validation_genes: list[str]
    test_genes: list[str]
    gene_weights: dict[str, float]
    cell_type_to_id: dict[str, int]
    lr_feature_names: list[str]
    sequence_hdf5_path: str


def match_sequence_genes(
    expression_genes: Sequence[str], embedding_genes: Sequence[str]
) -> list[str]:
    """Match genes using the notebook's order-sensitive set intersection."""

    matched = list(set(embedding_genes) & set(expression_genes))
    if not matched:
        raise PreprocessingError(
            "Sequence HDF5 keys and expression H5AD variable names do not intersect"
        )
    return matched


def build_gene_chromosome_map(
    gtf_path: str | Path, target_genes: Sequence[str]
) -> dict[str, str]:
    """Map gene names to chromosomes using the first matching GTF gene record."""

    columns = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]
    try:
        gtf = pd.read_csv(
            gtf_path,
            sep="\t",
            comment="#",
            header=None,
            names=columns,
            low_memory=False,
        )
    except Exception as exc:
        raise PreprocessingError(f"Unable to read GTF: {gtf_path}") from exc
    genes = gtf[gtf["feature"] == "gene"].copy()
    genes["gene_name"] = genes["attribute"].str.extract(r'gene_name "([^"]+)"')
    target = genes[genes["gene_name"].isin(target_genes)].drop_duplicates(
        subset=["gene_name"], keep="first"
    )
    return dict(zip(target["gene_name"], target["chrom"]))


def split_genes_by_chromosome(
    sequence_names: Sequence[str],
    gene_chromosome_map: Mapping[str, str],
    split_config: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], list[str]]:
    """Apply the notebook's chromosome-held-out gene split."""

    train_chromosomes = set(split_config["train_chromosomes"])
    validation_chromosomes = set(split_config["validation_chromosomes"])
    test_chromosomes = set(split_config["test_chromosomes"])
    train = [g for g in sequence_names if gene_chromosome_map.get(g) in train_chromosomes]
    validation = [
        g for g in sequence_names if gene_chromosome_map.get(g) in validation_chromosomes
    ]
    test = [g for g in sequence_names if gene_chromosome_map.get(g) in test_chromosomes]
    if not train or not validation or not test:
        raise PreprocessingError(
            "Chromosome assignment produced an empty train, validation or test gene split"
        )
    return train, validation, test


def annotate_lr_with_datasplits(
    adata: Any,
    db_complex: pd.DataFrame,
    train_genes: Sequence[str],
    validation_genes: Sequence[str],
    test_genes: Sequence[str],
    *,
    lr_information_key: str = "LR_pair_information",
) -> pd.DataFrame:
    """Annotate LR pairs after expanding multi-subunit complexes."""

    if lr_information_key not in adata.uns:
        raise PreprocessingError(f"STCase H5AD is missing uns[{lr_information_key!r}]")
    lr_frame = pd.DataFrame(adata.uns[lr_information_key]).T
    required_columns = {"ligand", "receptor"}
    missing_columns = required_columns - set(lr_frame.columns)
    if missing_columns:
        raise PreprocessingError(
            "LR information is missing columns: " + ", ".join(sorted(missing_columns))
        )

    train_set = set(train_genes)
    validation_set = set(validation_genes)
    test_set = set(test_genes)

    def actual_genes(entity: Any) -> list[str]:
        entity_text = str(entity).strip()
        if entity_text in db_complex.index:
            subunits = db_complex.loc[entity_text].tolist()
            return [
                str(gene).strip()
                for gene in subunits
                if str(gene).strip() not in {"", "nan", "NaN", "None"}
            ]
        return [entity_text]

    has_test: list[bool] = []
    has_train: list[bool] = []
    has_validation: list[bool] = []
    for lr_name in lr_frame.index:
        genes = set(
            actual_genes(lr_frame.loc[lr_name, "ligand"])
            + actual_genes(lr_frame.loc[lr_name, "receptor"])
        )
        has_test.append(bool(genes & test_set))
        has_train.append(bool(genes & train_set))
        has_validation.append(bool(genes & validation_set))

    lr_frame["has_test_gene"] = has_test
    lr_frame["has_train_gene"] = has_train
    lr_frame["has_val_gene"] = has_validation

    def assign_split(row: pd.Series) -> str:
        if row["has_test_gene"]:
            return "Test"
        if row["has_val_gene"]:
            return "Val"
        if row["has_train_gene"]:
            return "Train"
        return "Unknown"

    lr_frame["Data_Split"] = lr_frame.apply(assign_split, axis=1)
    return lr_frame


def extract_ccc_feature_vectors(
    adata: Any,
    excluded_lr_pairs: Sequence[str] | None = None,
    *,
    lr_weight_key: str = "LR_cell_weight",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Calculate sender row sums and receiver column sums for retained LR pairs."""

    if lr_weight_key not in adata.uns:
        raise PreprocessingError(f"STCase H5AD is missing uns[{lr_weight_key!r}]")
    lr_matrices = adata.uns[lr_weight_key]
    lr_keys = list(lr_matrices.keys())
    if excluded_lr_pairs is not None:
        excluded = set(excluded_lr_pairs)
        lr_keys = [name for name in lr_keys if name not in excluded]
    if not lr_keys:
        raise PreprocessingError("No LR pairs remain after test-gene exclusion")

    sender_features: dict[str, np.ndarray] = {}
    receiver_features: dict[str, np.ndarray] = {}
    for lr_name in lr_keys:
        matrix = lr_matrices[lr_name]
        if matrix.shape != (adata.n_obs, adata.n_obs):
            raise PreprocessingError(
                f"LR matrix {lr_name!r} has shape {matrix.shape}; expected "
                f"({adata.n_obs}, {adata.n_obs})"
            )
        sender_features[f"{lr_name}_sender"] = np.array(matrix.sum(axis=1)).flatten()
        receiver_features[f"{lr_name}_receiver"] = np.array(matrix.sum(axis=0)).flatten()

    sender = pd.DataFrame(sender_features).values
    receiver = pd.DataFrame(receiver_features).values
    return sender, receiver, lr_keys


def filter_spatial_groups(
    adata: Any,
    *,
    cell_type_column: str,
    spatial_cluster_column: str,
    subcluster_column: str,
    minimum_niche_cells: int,
    minimum_subcluster_cells: int,
) -> Any:
    """Apply niche filtering followed by cell-type × niche filtering."""

    for column in (cell_type_column, spatial_cluster_column):
        if column not in adata.obs:
            raise PreprocessingError(f"Expression H5AD is missing obs[{column!r}]")
    adata.obs[subcluster_column] = [
        left + "_" + right
        for left, right in zip(
            adata.obs[cell_type_column], adata.obs[spatial_cluster_column]
        )
    ]

    niche_counts = adata.obs[spatial_cluster_column].value_counts()
    small_niches = list(niche_counts[niche_counts < minimum_niche_cells].index)
    filtered = adata[~adata.obs[spatial_cluster_column].isin(small_niches)].copy()

    subcluster_counts = filtered.obs[subcluster_column].value_counts()
    retained_subclusters = list(
        subcluster_counts[subcluster_counts >= minimum_subcluster_cells].index
    )
    filtered = filtered[
        filtered.obs[subcluster_column].isin(retained_subclusters)
    ].copy()
    if filtered.n_obs == 0:
        raise PreprocessingError("Spatial filtering removed all observations")
    return filtered


def aggregate_subclusters(
    adata: Any,
    *,
    cell_type_to_id: Mapping[str, int],
    cell_type_column: str,
    subcluster_column: str,
    environment_key: str,
    ccc_receiver_key: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate expression, environment and receiver CCC by spatial subcluster."""

    for key in (environment_key, ccc_receiver_key):
        if key not in adata.obsm:
            raise PreprocessingError(f"Expression H5AD is missing obsm[{key!r}]")
    subcluster_names = adata.obs[subcluster_column].unique().tolist()
    major_ids: list[int] = []
    mean_environment: list[np.ndarray] = []
    mean_receiver_ccc: list[np.ndarray] = []
    mean_expression: list[np.ndarray] = []

    for subcluster in subcluster_names:
        mask = adata.obs[subcluster_column] == subcluster
        mask_array = mask.to_numpy()
        cell_type = adata.obs.loc[mask, cell_type_column].iloc[0]
        major_ids.append(cell_type_to_id[cell_type])
        mean_expression.append(np.asarray(adata[mask].X.mean(axis=0)).squeeze())
        mean_environment.append(np.asarray(adata.obsm[environment_key][mask_array]).mean(axis=0))
        mean_receiver_ccc.append(np.asarray(adata.obsm[ccc_receiver_key][mask_array]).mean(axis=0))

    return (
        subcluster_names,
        np.asarray(major_ids, dtype=np.int64),
        np.asarray(mean_environment),
        np.asarray(mean_receiver_ccc),
        np.asarray(mean_expression),
    )


def calculate_gene_weights_from_marker_tables(
    cell_type_markers: pd.DataFrame,
    niche_markers: pd.DataFrame,
    train_genes: Sequence[str],
    validation_genes: Sequence[str],
) -> dict[str, float]:
    """Apply the exact positive-marker maximum and capped weighting formula."""

    required = {"names", "scores", "logfoldchanges"}
    for name, frame in (
        ("cell-type marker table", cell_type_markers),
        ("niche marker table", niche_markers),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise PreprocessingError(
                f"{name} is missing columns: {', '.join(sorted(missing))}"
            )
    positive_cell = cell_type_markers[cell_type_markers["scores"] > 0]
    positive_niche = niche_markers[niche_markers["scores"] > 0]
    combined = pd.concat([positive_cell, positive_niche])
    grouped = combined[["names", "scores", "logfoldchanges"]].groupby("names").max()

    genes = list(train_genes) + list(validation_genes)
    missing_genes = [gene for gene in genes if gene not in grouped.index]
    if missing_genes:
        preview = ", ".join(missing_genes[:10])
        raise PreprocessingError(
            "Marker-score table has no positive row for required gene(s): " + preview
        )

    scores = np.asarray([grouped.loc[gene, "scores"] for gene in genes])
    positive_scores = scores[scores > 0]
    if positive_scores.size == 0:
        raise PreprocessingError("The positive marker-score median is undefined")
    median_score = float(np.median(positive_scores))

    weights: dict[str, float] = {}
    for gene in genes:
        score = float(grouped.loc[gene, "scores"])
        if score == 0:
            weight = 1.0
        else:
            weight = min(1.0 + score / median_score, 10.0)
        weights[gene] = float(weight)
    return weights


def _require_adata_fields(adata: Any, config: Mapping[str, Any]) -> None:
    data_config = config["data"]
    for column in (data_config["spatial_cluster_column"],):
        if column not in adata.obs:
            raise PreprocessingError(f"Expression H5AD is missing obs[{column!r}]")
    if data_config["environment_key"] not in adata.obsm:
        raise PreprocessingError(
            f"Expression H5AD is missing obsm[{data_config['environment_key']!r}]"
        )


def prepare_training_data(config: Mapping[str, Any]) -> PreparedTrainingData:
    """Run the authoritative notebook preprocessing from the five upstream inputs."""

    try:
        import h5py
        import scanpy as sc
    except ImportError as exc:
        raise PreprocessingError(
            "Full preprocessing requires h5py and scanpy; install requirements.txt"
        ) from exc

    inputs = config["inputs"]
    data_config = config["data"]

    with h5py.File(inputs["sequence_hdf5"], "r") as handle:
        embedding_names = list(handle.keys())

    adata = sc.read_h5ad(inputs["expression_h5ad"])
    adata_stcase = sc.read_h5ad(inputs["stcase_h5ad"])
    _require_adata_fields(adata, config)

    sequence_names = match_sequence_genes(list(adata.var.index), embedding_names)
    adata = adata[:, sequence_names].copy()

    cell_types = pd.read_csv(inputs["cell_type_csv"], index_col=0)
    if cell_types.shape[1] != 1:
        raise PreprocessingError(
            "Cell-type CSV must contain exactly one data column after the index"
        )
    if not adata.obs.index.isin(cell_types.index).all():
        raise PreprocessingError("Cell-type CSV cannot be aligned to all expression observations")
    adata.obs[data_config["cell_type_column"]] = cell_types

    missing_observations = adata_stcase.obs.index.difference(adata.obs.index)
    if len(missing_observations):
        raise PreprocessingError(
            "STCase observation IDs are not all present in the expression H5AD"
        )
    adata = adata[adata_stcase.obs.index, :].copy()

    gene_chromosome_map = build_gene_chromosome_map(inputs["gtf"], sequence_names)
    train_genes, validation_genes, test_genes = split_genes_by_chromosome(
        sequence_names, gene_chromosome_map, config["split"]
    )

    complex_key = data_config["lr_complex_key"]
    if complex_key not in adata_stcase.uns:
        raise PreprocessingError(f"STCase H5AD is missing uns[{complex_key!r}]")
    lr_frame = annotate_lr_with_datasplits(
        adata_stcase,
        adata_stcase.uns[complex_key],
        train_genes,
        validation_genes,
        test_genes,
        lr_information_key=data_config["lr_information_key"],
    )
    excluded_lr_pairs = lr_frame[lr_frame["has_test_gene"]].index.tolist()
    sender, receiver, lr_feature_names = extract_ccc_feature_vectors(
        adata_stcase,
        excluded_lr_pairs,
        lr_weight_key=data_config["lr_weight_key"],
    )
    if sender.shape[0] != adata.n_obs or receiver.shape[0] != adata.n_obs:
        raise PreprocessingError("CCC feature rows do not match aligned expression observations")
    adata.obsm[data_config["ccc_sender_key"]] = sender
    adata.obsm[data_config["ccc_receiver_key"]] = receiver

    adata_sub = filter_spatial_groups(
        adata,
        cell_type_column=data_config["cell_type_column"],
        spatial_cluster_column=data_config["spatial_cluster_column"],
        subcluster_column=data_config["subcluster_column"],
        minimum_niche_cells=data_config["minimum_niche_cells"],
        minimum_subcluster_cells=data_config["minimum_subcluster_cells"],
    )

    top_cell_types = list(set(adata_sub.obs[data_config["cell_type_column"]]))
    cell_type_to_id = {cell_type: index for index, cell_type in enumerate(top_cell_types)}
    (
        subcluster_names,
        major_ids,
        mean_environment,
        mean_receiver_ccc,
        mean_expression,
    ) = aggregate_subclusters(
        adata_sub,
        cell_type_to_id=cell_type_to_id,
        cell_type_column=data_config["cell_type_column"],
        subcluster_column=data_config["subcluster_column"],
        environment_key=data_config["environment_key"],
        ccc_receiver_key=data_config["ccc_receiver_key"],
    )

    environment_dim = config["model"]["environment_dim"]
    if mean_environment.shape[1] != environment_dim:
        raise PreprocessingError(
            f"Environment dimension is {mean_environment.shape[1]}; expected {environment_dim}"
        )

    sc.tl.rank_genes_groups(
        adata_sub,
        groupby=data_config["cell_type_column"],
        method="wilcoxon",
        use_raw=False,
        key_added="cell_type_rank_genes_groups",
    )
    cell_type_markers = sc.get.rank_genes_groups_df(
        adata_sub, group=None, key="cell_type_rank_genes_groups"
    )
    sc.tl.rank_genes_groups(
        adata_sub,
        groupby=data_config["spatial_cluster_column"],
        method="wilcoxon",
        use_raw=False,
        key_added="niche_rank_genes_groups",
    )
    niche_markers = sc.get.rank_genes_groups_df(
        adata_sub, group=None, key="niche_rank_genes_groups"
    )
    gene_weights = calculate_gene_weights_from_marker_tables(
        cell_type_markers,
        niche_markers,
        train_genes,
        validation_genes,
    )

    expected_sequence_shape = (
        config["model"]["seq_len"],
        config["model"]["seq_channels"],
    )
    with h5py.File(inputs["sequence_hdf5"], "r") as handle:
        first_shape = tuple(handle[sequence_names[0]].shape)
    if first_shape != expected_sequence_shape:
        raise PreprocessingError(
            f"Sequence embedding shape is {first_shape}; expected {expected_sequence_shape}"
        )

    return PreparedTrainingData(
        subcluster_names=subcluster_names,
        major_ids=major_ids,
        mean_environment=mean_environment,
        mean_receiver_ccc=mean_receiver_ccc,
        mean_expression=mean_expression,
        expression_genes=list(adata_sub.var_names),
        train_genes=train_genes,
        validation_genes=validation_genes,
        test_genes=test_genes,
        gene_weights=gene_weights,
        cell_type_to_id=cell_type_to_id,
        lr_feature_names=lr_feature_names,
        sequence_hdf5_path=str(inputs["sequence_hdf5"]),
    )
