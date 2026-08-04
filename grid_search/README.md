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

## Primary comparison: paired leave-one-seed-out

The largest individual test score in a grid is useful for exploration, but it
is not a fair comparison against train-and-grow's four prespecified runs. It
selects a favorable random initialization after seeing the outcomes. The
traditional `summary.json` is therefore retained as an exploratory report,
not used as the primary fixed-vs-growth result.

Stage 2 declares the architecture and test accuracy produced by each growth
seed under `paired_evaluation`. For fold `s`, the comparison:

1. takes the architecture produced by growth seed `s`;
2. selects its AdamW hyperparameters by mean validation accuracy on the other
   three model seeds, breaking ties by mean validation loss and then canonical
   override representation;
3. reads the selected configuration's result on seed `s` only after selection;
4. compares that fixed-model test accuracy with train-and-grow on the same
   seed.

Neither test accuracy nor the held-out seed participates in hyperparameter
selection. The epoch inside each individual run remains the epoch with maximum
validation accuracy. `summarize` reuses the existing trial JSON files and
writes both:

- `summary.json`: the unchanged exploratory ranking;
- `paired_leave_one_seed_out_summary.json`: the primary paired comparison.

The primary statistic is `aggregate.mean_paired_difference`, defined as fixed
test accuracy minus growth test accuracy. The report also includes its sample
standard deviation and standard error, plus per-fold wins. There are only four
folds, so this uncertainty must be reported and a small observed difference is
not enough for a strong general claim.

The paired report is strict: every candidate configuration must have a
completed result for all four seeds. Missing or failed trials stop the summary
with the architecture, held-out seed, missing seeds and incomplete overrides;
partial folds are never accepted silently. Grids without `paired_evaluation`
continue to generate only the traditional summary.
