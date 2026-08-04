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
