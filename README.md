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
| `Makefile` | `make install` and the clean-up commands; `make` lists them | yes |
| `scripts/build_kumo_wheel.sh` | Builds the macOS kumo-engine wheel; `make install` runs it | yes |
| `models/` | The model saved by `2_training.ipynb`; rebuilt on every run and derived from the data | no |
| `scratch/` | Kumo's working files (`kumo_work/`), plus the kumo-engine source, its build and the built wheel (`kumo-src/`, `build-venv/`, `wheels/`, `build.log`) | no |
| `.claude/skills/kumo-save-load-model/` | A Claude Code skill for saving and reloading Kumo models | yes |

## Set up

```bash
make install
```

PyPI's kumo-engine 1.0.1 has only a Linux x86_64 wheel, so `make install` first builds the macOS wheel from NVIDIA/kumo-engine at the commit `requirements.txt` pins, then creates the notebooks' environment.
The first run takes several minutes, mostly compiling Kumo's Rust index (DiskGraph) and downloading torch. After that it does nothing until `requirements.txt` changes.
In order, it:

1. installs rustup with Homebrew if there's no Rust, then the toolchain kumo-engine pins (1.99.1, in `packages/diskgraph/rust-toolchain.toml`);
2. clones kumo-engine into `scratch/kumo-src/` and builds the wheel into `scratch/wheels/`, logging to `scratch/build.log`;
3. creates `.venv/` and installs `requirements.txt` into it.

It uses `python3`. For another interpreter (3.11 or newer), run `make install PYTHON=python3.14`.
The build steps are in `scripts/build_kumo_wheel.sh`.

## Run

Open each notebook in VS Code, select the `.venv` kernel and choose **Run All**, in order: 1, 2, 3.

- `2_training.ipynb` retrains and replaces the saved model. It sets a training seed, but separate runs still differ: a customer's score can move by several points.
- `3_inference.ipynb` scores from the saved model, so it gives the same call list every time until notebook 2 runs again.
- `scratch/kumo_work/` grows a little with each training run. Delete it whenever you like (`make clean` also deletes the saved model); the notebooks rebuild the index in a second or two. Keep `scratch/wheels/`, which `requirements.txt` installs from.

When PyPI publishes a macOS wheel (the `main` README describes 1.1.0), pin that version in `requirements.txt` and drop the `--find-links` line; `make install` then skips the build.

## License

MIT; see `LICENSE`.
Kumo Engine itself is licensed separately under Apache-2.0 and isn't included in this repo: you install it as described in Set up.
