# Main-model training provenance

## Authoritative implementation

The public code is reconstructed from:

`code/fig1/model training/Model_training_v12_r.ipynb`

The adjacent `readme.txt` identifies this notebook as the main script. Frozen prediction and motif notebooks load `best_Advtrimodal_model_v12_r.pth` with the same parameter hierarchy, supporting checkpoint compatibility.

## Public mapping

| Public implementation | Notebook source | Preserved behavior |
| --- | --- | --- |
| `preprocessing.match_sequence_genes` | cell 2 | Set intersection of HDF5 and expression genes. |
| `preprocessing.build_gene_chromosome_map` | cell 4 | GTF `gene` filter, `gene_name` extraction and first duplicate. |
| `preprocessing.split_genes_by_chromosome` | cell 5 | Train chr1–chr12+chrX, validation chr13–chr15, test chr16–chr19. |
| `preprocessing.annotate_lr_with_datasplits` | cells 6–7 | Complex expansion and test > validation > train priority. |
| `preprocessing.extract_ccc_feature_vectors` | cells 8–10 | Test-linked LR exclusion; sender row sums and receiver column sums. |
| `preprocessing.filter_spatial_groups` | cells 11–17 | Niche threshold 500 followed by subcluster threshold 150. |
| `preprocessing.aggregate_subclusters` | cells 23–24 | Set-derived cell IDs and subcluster mean targets/context. |
| `preprocessing.calculate_gene_weights_from_marker_tables` | cells 30–36 | Positive Wilcoxon marker maxima and capped relative weight. |
| `dataset.GenoSpatialDataset` | cell 38 | Subcluster-major Cartesian sampling and lazy sequence HDF5 access. |
| `dataset.build_dataloaders` | cell 39 | Frozen batch sizes, shuffle settings and worker counts. |
| `model.GenoSpatialModel` | cell 41 | Convolutions, FiLM, dual attention, factor sums, branch dropout, fusion and prediction head. |
| `training.weighted_huber_loss` | cells 43–44 | Huber delta 0.1, no reduction, gene weighting, mean. |
| `training.validate` | cell 44 | Sample-count-normalized validation loss and global Pearson correlation. |
| `training.EarlyStopping` | cells 40 and 44 | Minimum-loss checkpoint, delta 1e-4, patience ten validation events. |
| `training.train` | cell 44 | AdamW step and validation every 250 global optimizer steps. |

## Non-scientific public additions

- YAML path and parameter configuration;
- explicit schema/dimension errors instead of downstream exceptions;
- optional versioned preprocessing cache;
- checkpoint metadata sidecar;
- JSON validation history;
- accurate weighted-Huber and validation-event log labels;
- portable device selection that prefers `cuda:1` when multiple CUDA devices are available.

These additions do not change valid-input tensor construction, model operations, optimized loss or checkpoint criterion.

## Dependency provenance

The public runtime imports require PyTorch, Scanpy/AnnData, pandas, NumPy, h5py, SciPy and PyYAML. The authoritative notebook also imports plotting and evaluation packages that are not required by this training-only public path.

Exact package versions are unavailable from notebook metadata and are therefore not pinned in `requirements.txt`. The notebook's stored rendered output contains a Matplotlib version string, but that indirect rendering metadata is not evidence for the complete training environment and is not used as a dependency pin.

## Reproducibility boundary

No explicit seed is present in the authoritative notebook. The saved run therefore cannot be reproduced bit-for-bit from currently available provenance. Scientific procedure, tensor operations, split rules, loss and optimization settings are reproduced; stochastic trajectory is not claimed.
