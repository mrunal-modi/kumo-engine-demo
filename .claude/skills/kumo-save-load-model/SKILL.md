---
name: kumo-save-load-model
description: Save a trained Kumo Engine model to local disk as a model bundle and reload it later to score customers without retraining, with a check that the rebuilt graph matches the one the model was trained on. Use whenever the user wants to keep, save, export, persist, version, archive, reuse, reload or "load back" a Kumo model or its trained weights, score new data with an already-trained model, move a model between sessions or machines, or asks where the weights went after the notebook's cleanup cell — even if they don't say "bundle". Also use when someone suggests committing .ckpt files or weights to git. Not for publishing to the Snowflake model registry (use Kumo's own kumo-model-persistence skill for that).
---

# Save and reload a Kumo model

## What gets saved, and why it's a folder

When `Trainer.fit` finishes, Kumo writes a **model bundle**: a folder at
`training_result.bundle_dir`. It holds everything needed to reuse the model:

| In the bundle | What it is |
| --- | --- |
| `model/checkpoint` | The trained weights (about 1–20 MB for the shop demo, depending on which FAST candidate wins) |
| `model/encoders.json`, `model/config.json`, `model/resolved-model-plan.json` | How columns are turned into numbers, and the model's shape |
| `training-spec.json` | The graph it was trained on (tables, keys, types, links) and the PQL query |
| `metrics.json`, `holdout.parquet` | Validation and test scores; the test rows include customer keys |
| `scoring-defaults.json` | Optional: the threshold `predict` applies by default |
| `_provenance.json`, `publication-manifest.json` | When and with what versions it was built; file hashes Kumo checks on load |

The bundle holds **no source data and no graph snapshot**. A reloaded model
always scores against a snapshot rebuilt from the tables, which is the point:
the same model can score next month's data.

The bundle lives under Kumo's scratch directory, which notebooks often delete at
the end and which is never in git. Saving means copying the bundle
somewhere durable. Kumo's `trainer.export(ModelOutputConfig(...))` publishes only
through the Snowflake connector; with a `FileConnector` it raises
`NotImplementedError`, so don't suggest it for local saving.

## The helper

`scripts/kumo_bundle.py` holds tested helpers. Import it from a notebook or
script run at the repo root:

```python
import sys
sys.path.insert(0, ".claude/skills/kumo-save-load-model/scripts")
from kumo_bundle import save_bundle, load_bundle, check_compatible, describe
```

Read the module's docstrings if you need more detail than below. The
kumo-engine-demo repo keeps a copy at `./kumo_bundle.py` for its notebooks, so
they don't depend on this folder; when you change the helper, update both. From a shell,
`python .claude/skills/kumo-save-load-model/scripts/kumo_bundle.py describe <bundle>`
prints what a saved model answers, what it was trained on and its metrics.

## Save

After `fit`, in the same session:

```python
MODEL_DIR = Path("models") / "orders-next-30d"        # outside scratch/, ignored by git
save_bundle(training_result, MODEL_DIR, threshold=0.5)  # threshold only for binary classification
```

- **Keep the folder out of git.** Add `models/` to `.gitignore` if it isn't
  there. Weights are binary, change on every training run, can't be reviewed
  in a diff and can't be used without Kumo anyway. They're also derived from
  customer data (`holdout.parquet` lists customer keys), so treat a bundle like
  the data it came from. For sharing, use object storage or a model registry.
- **Don't overwrite a model someone might rely on.** `save_bundle` refuses an
  existing folder unless `overwrite=True`. For versions, use a new folder name,
  such as a date suffix.
- **Store the threshold in the bundle.** `threshold=` writes Kumo's own
  `scoring-defaults.json` (via `TrainingResult.with_scoring_defaults`), so a
  reloaded classifier makes the same yes/no calls without being told again.
- **Don't count on a seed to reproduce a model; save it.** Setting
  `model_plan.training_job.manual_seed = <int>` before `Trainer(model_plan)`
  narrows the variation, but on this stack (CPU, kumo-engine
  1.0.0.dev0+5d17fb3) it doesn't make separate runs repeatable. Two seeded fits
  in one Python process came out within 0.006 of each other in probability.
  Two separate executions of the same seeded notebook moved individual
  probabilities by up to about 0.09 and test AUROC from 0.81 to 0.80. A saved
  bundle is the only way to keep exactly the same model.

## Reload and score

In a new session, with no training:

1. **Rebuild the graph with the training code**: the same tables, the same
   column lists, `with_column` type fixes, keys, time column and edges. Reuse
   the code itself rather than retyping it. Then take a fresh snapshot of the
   current data with `graph.snapshot()`.
2. **Load with the check:**
   ```python
   model = load_bundle(MODEL_DIR, snapshot=snapshot, query=QUERY)
   ```
   This compares the rebuilt graph with `training-spec.json` before loading.
   It raises a plain-language `ValueError` for anything the model was trained
   on that's missing or typed differently (a table, key, time column, link,
   column or stype), or for a different query. Without this check, Kumo fails
   deep inside `predict` with a bare `KeyError: 'categorical'` when a trained
   column is missing, which tells the user nothing. It prints notes, without
   raising, for new columns (the model ignores them; tested, the scores were
   unchanged), changed dtypes and a different kumo-engine version.
3. **Predict** exactly as with a freshly trained trainer:
   ```python
   batch = model.predict(snapshot, entities=rows)          # rows: ENTITY1 (+ TIMESTAMP) table
   # or: model.predict(snapshot, prediction_time=datetime(2026, 12, 15))
   ```
   The saved threshold applies when `binary_classification_threshold` is
   omitted. Pass it explicitly to override it.

Load only bundles you or your team created. Kumo's own documentation warns that
checkpoint deserialization can run code, and its hashes detect corruption, not
who made the file. Use the same kumo-engine build that trained the model where
possible; the check notes a mismatch.

## Prove it worked

A save is only proven by a reload. When the original predictions are still
available, score the same rows on the same data with the reloaded model and
compare `True_PROB` per entity. In testing on the shop demo, a fresh process
reproduced all 10 probabilities to within 3e-8 (CSV rounding), with identical
`TARGET_PRED`. A larger difference means the graph or the data differs from
the original run. On new data there is nothing to compare against, so say so
rather than claiming the scores match.

## Tell the user

- Where the bundle is, its size, and that it's excluded from git.
- What it answers (the PQL), when it was trained, and its test metrics
  (`describe` prints all three).
- Whether a reload was verified, and how: rows compared and the largest
  difference.
- Anything the check noted.

Write for the reader's level. For a non-technical reader: the bundle is "the
trained model saved as a folder: its learned numbers plus the instructions to
use them". It needs fresh data to score, the same way a trained analyst needs
this month's numbers.
