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
rates from 0.003 through 0.03, and a fresh results directory. It contains 384
trials (indices 0-383):

```bash
PYTHONPATH=src python -m grid_search.run list \
  --config grid_search/fixed_architectures_stage2.yaml
PYTHONPATH=src python -m grid_search.run run \
  --config grid_search/fixed_architectures_stage2.yaml --trial-index 0
PYTHONPATH=src python -m grid_search.run summarize \
  --config grid_search/fixed_architectures_stage2.yaml
```
