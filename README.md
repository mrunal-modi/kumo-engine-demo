# Kumo Engine demo: shop tables → graph → context → predictions

`kumo_engine_shop_workbook_graph.ipynb` follows three synthetic shop tables (customers, products and orders) through [Kumo Engine](https://github.com/NVIDIA/kumo-engine).
It answers one business question: which accounts will order in the next 30 days?
Along the way it shows the tables, the graph, the DiskGraph snapshot, the context Kumo samples for each customer, the predictive query, FAST training on the CPU and batch predictions.
The notebook is saved with its outputs, so you can read it as an explanation without running it.
This is personal learning and demo material.

**Last run with:** kumo-engine `1.0.0.dev0+5d17fb3`, Python 3.14.6, torch 2.15.0, on macOS with Apple silicon, CPU only, in under a minute.
That version is a wheel built from NVIDIA/kumo-engine `main` at commit `5d17fb3`.

## What's where

| Path | Contents | In git |
| --- | --- | --- |
| `data/` | Input: `customers.csv`, `products.csv`, `orders.csv`, written by the notebook's seeded generator | yes |
| `output/predictions.csv` | Output: the ranked 30-day order probabilities for 10 customers | yes |
| `scratch/` | Temporary: Kumo's working files during a run, the built wheel and `build.log` | no |

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

Then create the notebook's environment from this folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

Open the notebook in VS Code, select the `.venv` kernel and choose **Run All**.
It rewrites `data/` and `output/predictions.csv`, and its last cell removes Kumo's temporary files from `scratch/`.
Training is not seeded, so the probabilities change a little from run to run.

When PyPI publishes a macOS wheel (the `main` README describes 1.1.0), pin that version in `requirements.txt` and drop the `--find-links` line.
