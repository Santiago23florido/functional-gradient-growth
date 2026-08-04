# Fixed-architecture grid search

This experiment trains the four final architectures from the headline result
from a fresh random initialization, with growth and FGD disabled. It preserves
the original `smooth_sin` split: 1024 train points, 1024 validation points,
8192 fresh test points, `train_seed=0`, and 70 epochs.

The grid contains 672 trials: four architectures x four model seeds x 42
optimizer/scheduler combinations. Hyperparameters are ranked by mean
validation accuracy across seeds. Test accuracy is recorded but never used to
select a trial.

From the repository root:

```bash
PYTHONPATH=src python -m grid_search.run list
PYTHONPATH=src python -m grid_search.run run --trial-index 0
PYTHONPATH=src python -m grid_search.run summarize
```

For a job array, pass its zero-based task ID to `--trial-index`; the valid
range is printed by `list`. For fewer, longer jobs, shard the deterministic
trial list:

```bash
PYTHONPATH=src python -m grid_search.run run --shard-index 0 --num-shards 8
```

Completed trials are skipped automatically, so rerunning the same command is
safe. Individual records go to `results/grid_search_fixed_architectures/trials`,
full histories to `.../histories`, and `summarize` writes `summary.json`.

## Stage 2: follow the boundary

The completed first search put every architecture's best validation epoch at
61-69 of 70 and every winner on AdamW's largest searched learning rate, 0.01.
`fixed_architectures_stage2.yaml` follows both boundaries: 400 epochs, AdamW
rates from 0.001 through 0.04 with extra resolution around 0.01, and a fresh
`stage2_lr_sweep` results directory. It contains 960 trials (indices 0-959):

```bash
PYTHONPATH=src python -m grid_search.run list \
  --config grid_search/fixed_architectures_stage2.yaml
PYTHONPATH=src python -m grid_search.run run \
  --config grid_search/fixed_architectures_stage2.yaml --trial-index 0
PYTHONPATH=src python -m grid_search.run summarize \
  --config grid_search/fixed_architectures_stage2.yaml
```

## Primary comparison: paired same-seed retraining

The largest individual test score in a grid is useful for exploration, but it
is not a fair comparison against train-and-grow's four prespecified runs. It
selects a favorable random initialization after seeing the outcomes. The
traditional `summary.json` is therefore retained as an exploratory report,
not used as the primary fixed-vs-growth result.

Stage 2 declares the architecture and test accuracy produced by each growth
seed under `paired_same_seed_evaluation`. For seed `s`, the comparison:

1. takes the architecture produced by growth seed `s`;
2. discards all learned weights and constructs a fresh model initialized with
   the same `model_seed=s`;
3. trains that fixed architecture with ordinary AdamW, with growth disabled;
4. selects LR, weight decay and scheduler by validation accuracy among trials
   whose architecture and seed both match `(A_s, s)`, breaking ties by
   validation loss and then canonical override representation;
5. reads test only after selection and compares it with train-and-grow seed
   `s`.

### Same-seed results

The table reports test metrics only; LR, weight decay and scheduler were
selected by validation before test was read.

| seed | architecture | LR | weight decay | scheduler | fixed test | growth test | fixed - growth |
|---:|:---:|---:|---:|:---|---:|---:|---:|
| 0 | `11-19-16` | 0.04 | 0.01 | cosine annealing | 0.9426 | 0.9490 | -0.0064 |
| 1 | `12-17-18` | 0.015 | 0.0 | cosine annealing | 0.9332 | 0.9400 | -0.0068 |
| 2 | `13-19-14` | 0.04 | 0.001 | cosine annealing | 0.9386 | 0.9270 | +0.0116 |
| 3 | `9-16-22` | 0.04 | 0.001 | cosine annealing | 0.9253 | 0.9580 | -0.0327 |

Across the four prescribed pairs, fixed retraining wins once and growth wins
three times. Fixed retraining averages 0.9349 test accuracy versus 0.9435 for
train-and-grow; the mean paired difference is -0.0086 (fixed minus growth).

No favorable seed is selected per architecture. The epoch inside each run is
still selected by maximum validation accuracy, and test never participates in
either epoch or hyperparameter selection. `summarize` reuses the existing
trial JSON files and writes both:

- `summary.json`: the unchanged exploratory ranking;
- `paired_same_seed_retraining_summary.json`: the primary paired comparison.

The primary statistic is `aggregate.mean_paired_difference`, defined as fixed
test accuracy minus growth test accuracy. The report also includes its sample
standard deviation and standard error, plus per-seed wins. A positive value
means the architecture found by growth had more potential than the growth
trajectory itself extracted when the architecture was retrained from scratch.
This is therefore a two-stage method: architecture search by growth, then a
fresh final fit with AdamW.

The paired report is strict: every candidate configuration must have a
completed result for its exact reference architecture and seed. Missing or
failed trials stop the summary with the seed, expected architecture, candidate
count and incomplete overrides. Grids without `paired_same_seed_evaluation`
continue to generate only the traditional summary.

This protocol is deliberately **not** leave-one-seed-out. It answers whether
fresh AdamW training can extract more performance from each architecture under
the same initialization seed that produced it, rather than estimating a
hyperparameter choice transferred across seeds.

## Search for strictly smaller architectures (scratch + SLURM)

This independent experiment asks, for each growth seed, whether a freshly
initialized three-hidden-layer MLP with at least 400 parameters and strictly
fewer parameters than that seed's growth architecture can do better. The
admissible space for seed `s` is

```text
400 <= parameter_count(A) < parameter_count(growth_architecture_s)
```

The parameter count includes weights and biases from 4 inputs through the
three hidden layers to 1 output. Width 2 is the default minimum because it is
the current initial structure and the space reachable by growth. The reference
budgets are:

| seed | growth architecture | parameter budget | growth test accuracy |
|---:|:---:|---:|---:|
| 0 | `11-19-16` | 620 | 0.949 |
| 1 | `12-17-18` | 624 | 0.940 |
| 2 | `13-19-14` | 626 | 0.927 |
| 3 | `9-16-22` | 602 | 0.958 |

Architectures are enumerated separately for each seed, ordered by parameter
count and then lexicographically. Each candidate uses that same seed as its
`model_seed`, starts from new weights, trains normally with growth disabled,
and keeps the original `smooth_sin` data protocol.

Stage 1 screens every valid architecture once with the common AdamW recipe in
`smaller_architectures_400_stage1.yaml`, then retains the best 50 per seed.
Stage 2 evaluates the 60 AdamW configurations in
`smaller_architectures_400_stage2.yaml` only on those candidates and, by
default, on the corresponding growth architecture as an ineligible control.
Both stages rank by validation accuracy, then validation loss, parameter count,
architecture and canonical overrides as applicable. Test metrics are recorded
but are not consulted until the winner has been fixed by validation.

The two-stage search does not tune all 60 configurations for every
architecture. Every architecture is screened with one common recipe and only
the top 50 proceeds to fine-tuning. Consequently, the result establishes the
absence of a better candidate within this evaluated protocol, not the
mathematical non-existence of a better architecture. Setting
`search_mode: exhaustive` applies all 60 configurations to every architecture;
the preparation command prints a prominent warning and the resulting trial
count before writing its manifest.

### Inspect and launch

On the cluster, first inspect the scratch link and capacity:

```bash
readlink -f ~/datasets
df -h "$(readlink -f ~/datasets)"
```

The inspection command enumerates and reports counts, bounds and five examples
per seed without training:

```bash
cd "$HOME/dev/functional-gradient-growth"
conda activate cct
PYTHONPATH=src python -m grid_search.budget_search inspect \
  --config grid_search/smaller_architectures_400_stage1.yaml
```

A dry run performs preflight, creates or verifies the deterministic stage 1
manifest in scratch and prints the planned `sbatch` chain, but submits nothing:

```bash
REPO_ROOT="$HOME/dev/functional-gradient-growth" \
FGG_SCRATCH_ROOT="$HOME/datasets" \
RUN_NAME="smaller_architectures_400_v1" \
MAX_CONCURRENT=30 CONDA_ENV=cct DRY_RUN=1 \
bash cluster/slurm/submit_smaller_architecture_search.sh
```

Remove `DRY_RUN=1` to submit. The script asks for confirmation; use
`CONFIRM_SUBMIT=1` only for a non-interactive submission. Optional cluster
settings are `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`,
`SLURM_TIME_STAGE1`, `SLURM_TIME_STAGE2`, `SLURM_MEM` and
`SBATCH_EXTRA_ARGS`. No GPU is requested by default.

```bash
REPO_ROOT="$HOME/dev/functional-gradient-growth" \
FGG_SCRATCH_ROOT="$HOME/datasets" \
RUN_NAME="smaller_architectures_400_v1" \
MAX_CONCURRENT=30 CONDA_ENV=cct \
bash cluster/slurm/submit_smaller_architecture_search.sh
```

Operational commands use the same `RUN_NAME` and scratch variables:

```bash
bash cluster/slurm/status_smaller_architecture_search.sh
RESUME_STAGE=stage1 bash cluster/slurm/resume_smaller_architecture_search.sh
RESUME_STAGE=stage2 bash cluster/slurm/resume_smaller_architecture_search.sh
bash cluster/slurm/cancel_smaller_architecture_search.sh

# Rebuild/verify the final summary after all shards are complete:
PYTHONPATH=src python -m grid_search.budget_search finalize \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --run-root "$(readlink -f ~/datasets)/functional-gradient-growth/smaller_architectures_400_v1"

# Retrain and save full outputs for only the four selected winners:
PYTHONPATH=src python -m grid_search.budget_search materialize-winners \
  --config grid_search/smaller_architectures_400_stage1.yaml \
  --run-root "$(readlink -f ~/datasets)/functional-gradient-growth/smaller_architectures_400_v1"
```

Stage 1 uses 1,000 shards and stage 2 uses 600 by default. Array element `i`
processes entries satisfying `trial_index % num_shards == i`; each shard is
the sole appender to its JSONL file. Completed trial IDs are skipped on rerun,
while the resume script submits only incomplete or failed shard indices with
failure retry enabled. Change `STAGE1_NUM_SHARDS` and `STAGE2_NUM_SHARDS` if
the cluster's `MaxArraySize` requires it. `MAX_CONCURRENT` controls the `%30`
array throttle and defaults to 30.

All generated artifacts and SLURM logs live under the resolved scratch path:

```text
$SCRATCH_ROOT/functional-gradient-growth/$RUN_NAME/
  manifests/  shards/  summaries/  selected/
  slurm_logs/ metadata/ failures/
```

Mass trials write one compact JSONL row and never call the full output writer.
Thus no large result, history, model or SLURM log is stored in the checkout or
under its `results/` directory. Full histories are produced only when
`materialize-winners` explicitly retrains the four final winners.

The default plan contains 76,646 stage 1 trials and 12,240 planned stage 2
trials. At one minute per trial and a perfectly utilized concurrency of 30,
the idealized times are about 42.6 hours and 6.8 hours respectively (49.4
hours total). Real runtime can be longer; inspect the first completed shards
and adjust time limits based on measured throughput before relying on this
estimate.
