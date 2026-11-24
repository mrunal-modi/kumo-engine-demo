# Kumo Engine demo: shop tables → index → model → call list

Three short notebooks take synthetic shop tables (customers, products and orders) through [Kumo Engine](https://github.com/NVIDIA/kumo-engine) and answer one business question: which customers will order in the next 30 days?
They are written for business readers in plain language, and each is saved with its outputs, so you can read them without running anything.
This is personal learning and demo material.

| Notebook | What it does | Produces |
| --- | --- | --- |
| `1_ingest.ipynb` | Loads the three tables, links them, builds Kumo's index and checks its health | The index in `scratch/kumo_work/` |
| `2_training.ipynb` | Turns history into practice questions, trains the model, grades it on the newest dates and saves it | The model in `models/orders-next-30d/` |
| `3_inference.ipynb` | Loads the saved model without retraining, scores all 500 customers and shows the records behind a score | `output/call_list.csv` |

**Last run with:** kumo-engine `1.0.0.dev0+5d17fb3`, Python 3.14.6, torch 2.15.0, on macOS with Apple silicon, CPU only, in about 30 seconds for all three.
That version is a wheel built from NVIDIA/kumo-engine `main` at commit `5d17fb3`.

## What's where

| Path | Contents | In git |
| --- | --- | --- |
| `data/` | Input: `customers.csv`, `products.csv`, `orders.csv` (synthetic sample data) | yes |
| `output/call_list.csv` | Output of `3_inference.ipynb`: all 500 customers ranked by their chance of ordering | yes |
| `shop_demo.py` | Shared setup for the notebooks: the table and link definitions, and readable summaries | yes |
| `kumo_bundle.py` | Saves a trained model, checks new data against it and reloads it | yes |
| `models/` | The model saved by `2_training.ipynb`; rebuilt on every run and derived from the data | no |
| `scratch/` | Kumo's working files (`kumo_work/`), the built wheel (`wheels/`) and `build.log` | no |
| `.claude/skills/kumo-save-load-model/` | A Claude Code skill for saving and reloading Kumo models | yes |

## Set up

PyPI's kumo-engine 1.0.1 has only a Linux x86_64 wheel, so build the macOS wheel from `main` first.
The build needs Rust; it uses toolchain 1.99.1, pinned in `packages/diskgraph/rust-toolchain.toml`.

```bash
brew install rustup && export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
git clone https://github.com/NVIDIA/kumo-engine.git /tmp/kumo-src && git -C /tmp/kumo-src checkout 5d17fb3
(cd /tmp/kumo-src/packages/diskgraph && rustup toolchain install --profile minimal)
python3 -m venv /tmp/kumo-build && /tmp/kumo-build/bin/pip install wheel
(cd /tmp/kumo-src && PYTHON=/tmp/kumo-build/bin/python OUTPUT_DIR="$OLDPWD/scratch/wheels" \
  ./packages/kumo-engine/scripts/build_wheel.sh)    # run from this folder; writes scratch/wheels/*.whl
```

Then create the notebooks' environment from this folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

Open each notebook in VS Code, select the `.venv` kernel and choose **Run All**, in order: 1, 2, 3.

- `2_training.ipynb` retrains and replaces the saved model. It sets a training seed, but separate runs still differ: a customer's score can move by several points.
- `3_inference.ipynb` scores from the saved model, so it gives the same call list every time until notebook 2 runs again.
- `scratch/kumo_work/` grows a little with each training run. Delete it whenever you like; the notebooks rebuild the index in a second or two. Keep `scratch/wheels/`, which `requirements.txt` installs from.

When PyPI publishes a macOS wheel (the `main` README describes 1.1.0), pin that version in `requirements.txt` and drop the `--find-links` line.
