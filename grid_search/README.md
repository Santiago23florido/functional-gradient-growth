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
