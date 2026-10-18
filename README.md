# CCMST-POI: Confidence-Calibrated Multi-Scale Spatio-Temporal POI Recommendation

PyTorch implementation of a multi-scale next-point-of-interest (next-POI) recommendation model with information-quality-aware fusion, branch-level reliability supervision, and post-hoc confidence calibration.

## Features

- Multi-context check-in features: POI identity, category, timestamp, geographical coordinates and position.
- Three causal Transformer branches for short-, medium- and long-range mobility history; default windows: **10 / 30 / 100**.
- Pairwise relation-aware attention using log time intervals, log Haversine distances and category-match features.
- Five information-quality descriptors: history availability, temporal regularity, spatial regularity, category consistency and branch predictive entropy.
- Reliability fusion with auxiliary branch predictions, detached instance-level performance targets and KL-divergence supervision.
- Fusion choices: `reliability`, `uniform`, `attention`, `gating`, `quality`.
- Ranking metrics: Recall@5/10/20, NDCG@5/10/20 and MRR.
- Validation-fitted scalar temperature scaling; top-1 ECE, full-distribution NLL and multiclass Brier score.
- Paired user-cluster bootstrap confidence intervals and history/noise perturbation evaluation.

## Requirements

- Python with the packages listed in `requirements.txt`.
- PyTorch; CPU and CUDA execution are supported via `--device`.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Input data

The `prepare` command accepts a UTF-8 CSV with the following columns:

```csv
user_id,poi_id,timestamp,latitude,longitude,category
```

- `timestamp`: Unix timestamp in seconds (UTC).
- `latitude`, `longitude`: decimal degrees.
- `user_id`, `poi_id`, `category`: identifiers or category labels.

The preprocessing pipeline applies iterative minimum user/POI frequency filters and creates per-user chronological training, validation and test instances at a 70/10/20 ratio. The POI vocabulary is constructed from training interactions; evaluation instances with out-of-vocabulary target POIs are excluded and counted in `meta.json`.

Prepare a dataset:

```bash
python run.py prepare --csv data/checkins.csv --out data/processed --min-user 20 --min-poi 10
```

Raw Foursquare and Gowalla releases must first be converted to the six-column CSV format. Dataset use and redistribution are governed by their respective licenses.

## Training

Train with the default architecture:

```bash
python run.py train --data data/processed --out results/ccmst_seed42 --seed 42 --device cuda
```

Explicit architecture and optimization settings:

```bash
python run.py train \
  --data data/processed \
  --out results/ccmst_seed42 \
  --epochs 100 --patience 10 --batch 256 \
  --windows 10 30 100 --dim 64 --heads 4 --layers 2 \
  --fusion reliability \
  --aux-weight 0.2 --rel-weight 0.1 --target-temperature 1.0 \
  --lr 0.001 --weight-decay 0.00001 \
  --seed 42 --device cuda
```

On Windows PowerShell, put the command on a single line or use PowerShell backticks instead of the Bash line-continuation character `\`.

Training outputs:

- `best.pt`: selected model checkpoint and model configuration.
- `history.json`: per-epoch training loss and validation metrics.

## Evaluation

Test-set predictions:

```bash
python run.py evaluate --data data/processed --checkpoint results/ccmst_seed42/best.pt --split test --out results/ccmst_seed42/test --device cuda
```

Validation-set predictions:

```bash
python run.py evaluate --data data/processed --checkpoint results/ccmst_seed42/best.pt --split val --out results/ccmst_seed42/val --device cuda
```

Each evaluation produces:

- `<split>_predictions.csv`: prediction-level ranking metrics and identifiers.
- `<split>_logits.npz`: uncalibrated candidate logits and labels.
- `<split>_metrics.json`: aggregated ranking and calibration metrics.

## Confidence calibration

Fit a scalar temperature on validation logits and evaluate calibration on test logits:

```bash
python run.py calibrate \
  --val results/ccmst_seed42/val/val_logits.npz \
  --test results/ccmst_seed42/test/test_logits.npz \
  --out results/ccmst_seed42/calibration.json
```

The command reports calibration metrics before and after temperature scaling. It requires the validation and test logits to use the same candidate vocabulary. Positive scalar temperature scaling does not change the candidate ranking.

## Fusion ablation experiments

Train separate runs using the `--fusion` parameter:

| Value | Fusion mechanism |
|---|---|
| `reliability` | Quality-aware reliability fusion with KL supervision |
| `uniform` | Uniform three-branch fusion |
| `attention` | Attention-based fusion |
| `gating` | Learned gating fusion |
| `quality` | Information-quality-based fusion |

For non-reliability fusion controls, specify `--rel-weight 0`. For `quality`, this removes the KL reliability objective while retaining information-quality inputs. Use matched splits, random seeds, and architecture settings for controlled comparisons.

## Paired statistical analysis

```bash
python run.py stats \
  --model-a results/ccmst_seed42/test/test_predictions.csv \
  --model-b results/baseline_seed42/test/test_predictions.csv \
  --out results/paired_stats.json \
  --reps 3000 --seed 42
```

The script computes paired differences for Recall@10, NDCG@10 and reciprocal rank, using user-level cluster bootstrap resampling with instance-weighted metric aggregation. Matching prediction instances are required for a valid paired comparison. The output includes confidence intervals; statistical p-values and multiple-comparison corrections are separate analyses.

## Robustness evaluation

```bash
python run.py robustness \
  --data data/processed \
  --checkpoint results/ccmst_seed42/best.pt \
  --out results/ccmst_seed42/robustness \
  --modes random block recent time geo \
  --repeats 5 --seed 42 --device cuda
```

Available perturbation modes:

- `random`: randomly omitted historical interactions.
- `block`: consecutive-block history deletion.
- `recent`: deletion of recent historical interactions.
- `time`: timestamp perturbation.
- `geo`: coordinate perturbation.

The command saves aggregate perturbation metrics in `robustness.csv`. Review the perturbation functions in `ccmst/perturb.py` for the exact sampling and timestamp-handling rules before changing the evaluation protocol.

## Project structure

```text
CCMST_POI_Research_Code/
├── ccmst/
│   ├── data.py          # Data preparation and dataset loader
│   ├── model.py         # Encoders, relation attention, fusion, objectives
│   ├── evaluation.py    # Metrics, calibration and bootstrap analysis
│   └── perturb.py       # Missing-history and noise perturbations
├── examples/
├── tests/
├── run.py               # Command-line entry point
├── requirements.txt
└── README.md
```

## Implementation notes

- Dataset cohort filtering currently precedes per-user chronological splitting.
- Coordinate features use a linear projection; training-statistic coordinate normalization and location-specific timezone conversion are not part of this version.
- Temporal perturbations do not enforce monotonic perturbed timestamps.
- Full-candidate logits are materialized, so large POI vocabularies may require candidate batching or memory optimization.
- The command-line interface currently covers training, evaluation, calibration, paired bootstrap and perturbation experiments. Strict unseen-user cold start, global-time OOD splitting, external baseline implementations, multi-seed orchestration and corrected significance testing are not exposed as complete standalone workflows.

## Tests

```bash
python -m pytest tests -q
```

## License and data usage

Follow the license terms of the source datasets and any external baseline implementations included or used alongside this project.
