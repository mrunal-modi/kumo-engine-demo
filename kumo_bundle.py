"""Save, check, reload and describe Kumo Engine model bundles on local disk.

A bundle is the folder Kumo writes for a fitted model (``TrainingResult.bundle_dir``):
the selected checkpoint (the trained weights), the column encoders, the resolved model
plan, the training spec (graph definition and predictive query), metrics and the test
split's scores. It holds no source tables and no graph snapshot, so scoring with a
reloaded model always needs a snapshot rebuilt from the data.

Only Kumo's Snowflake connector can *publish* a model; locally, saving a model means
copying its bundle folder somewhere that outlives Kumo's scratch directory.

From Python (a notebook or script):

    import sys
    sys.path.insert(0, ".claude/skills/kumo-save-load-model/scripts")
    from kumo_bundle import save_bundle, load_bundle

    save_bundle(training_result, "models/orders-next-30d", threshold=0.5)
    ...
    model = load_bundle("models/orders-next-30d", snapshot=snapshot, query=QUERY)
    scores = model.predict(snapshot, entities=rows)   # the saved threshold applies

From a shell:

    python kumo_bundle.py describe models/orders-next-30d
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_bundle(training_result: Any, dest: str | Path, *, threshold: Optional[float] = None,
                overwrite: bool = False) -> Path:
    """Copy a fitted model's bundle to ``dest``.

    ``threshold`` is stored in the bundle as Kumo's own scoring default
    (``scoring-defaults.json``), so a reloaded binary classifier applies it when
    ``predict`` is called without ``binary_classification_threshold``.
    An existing ``dest`` is refused unless ``overwrite=True``.
    """
    import kumo.engine as kumo

    dest = Path(dest)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} already exists. Choose a new folder (for example a dated "
                              f"version) or pass overwrite=True to replace it.")
    result = training_result
    if threshold is not None:
        result = training_result.with_scoring_defaults(
            kumo.ScoringDefaults(binary_classification_threshold=threshold))
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(result.bundle_dir, dest)
    return dest


def _tables(graph_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tables = {}
    for table in graph_config["tables"]:
        time_col = table.get("time_col")
        tables[table["table_name"]] = {
            "pkey": (table.get("pkey") or {}).get("name"),
            "time": time_col.get("name") if isinstance(time_col, dict) else time_col,
            "cols": {c["name"]: (c.get("dtype"), c.get("stype")) for c in table["cols"]},
            "fkeys": {(f["name"], f["dst_table"]) for f in table.get("fkeys", [])},
        }
    return tables


def check_compatible(bundle_dir: str | Path, snapshot: Any, query: Optional[str] = None) -> List[str]:
    """Compare a rebuilt snapshot's graph with the graph the model was trained on.

    Raises ``ValueError`` naming every table, key, time column, link or column the
    model was trained on that the rebuilt graph lacks, or declares with a different
    semantic type (stype), and a different predictive query. Kumo itself does not
    catch a missing column cleanly: it fails deep inside ``predict`` with a bare
    ``KeyError``. Returns notes for differences that don't block scoring: new
    columns (the model ignores them), changed physical dtypes and a different
    kumo-engine version.
    """
    bundle_dir = Path(bundle_dir)
    spec = json.loads((bundle_dir / "training-spec.json").read_text())
    trained, live = _tables(spec["graph_config"]), _tables(snapshot.graph_config)
    problems: List[str] = []
    notes: List[str] = []
    for name, want in trained.items():
        have = live.get(name)
        if have is None:
            problems.append(f"table '{name}' is missing")
            continue
        for role in ("pkey", "time"):
            if have[role] != want[role]:
                label = "primary key" if role == "pkey" else "time column"
                problems.append(f"{name}: {label} is {have[role]!r}, but the model was trained with {want[role]!r}")
        for col, (dtype, stype) in want["cols"].items():
            if col not in have["cols"]:
                problems.append(f"{name}.{col} was used in training but is missing")
                continue
            new_dtype, new_stype = have["cols"][col]
            if new_stype != stype:
                problems.append(f"{name}.{col} is stype {new_stype!r}, but the model was trained on {stype!r}")
            elif new_dtype != dtype:
                notes.append(f"{name}.{col} dtype changed from {dtype!r} to {new_dtype!r}")
        for col in sorted(have["cols"].keys() - want["cols"].keys()):
            notes.append(f"{name}.{col} is new; the model ignores it")
        for fkey, dst in sorted(want["fkeys"] - have["fkeys"]):
            problems.append(f"link {name}.{fkey} -> {dst} is missing")
    if query is not None and " ".join(query.split()) != " ".join(spec["predictive_query"].split()):
        problems.append(f"query differs: the model answers {spec['predictive_query']!r}")
    try:
        import kumo.engine as kumo
        trained_version = json.loads((bundle_dir / "_provenance.json").read_text())["environment"]["kumo_engine"]
        if trained_version != kumo.__version__:
            notes.append(f"trained with kumo-engine {trained_version}; this environment has {kumo.__version__}")
    except (OSError, KeyError, ValueError):
        pass
    if problems:
        raise ValueError("The rebuilt graph does not match the graph this model was trained on:\n- "
                         + "\n- ".join(problems)
                         + "\nRebuild the tables exactly as the training code declared them.")
    return notes


def load_bundle(bundle_dir: str | Path, snapshot: Any = None, *, query: Optional[str] = None,
                accelerator: str = "cpu") -> Any:
    """Reload a saved bundle as a ``kumo.Trainer`` ready to ``predict``.

    With ``snapshot`` (and optionally ``query``), runs :func:`check_compatible`
    first and prints its notes. Load only bundles you or your team created:
    Kumo's checkpoint files can run code when they are read.
    """
    import kumo.engine as kumo

    bundle_dir = Path(bundle_dir)
    if not (bundle_dir / "training-spec.json").is_file():
        raise FileNotFoundError(f"{bundle_dir} is not a Kumo model bundle (no training-spec.json).")
    if snapshot is not None:
        for note in check_compatible(bundle_dir, snapshot, query):
            print("note:", note)
    return kumo.Trainer.load(bundle_dir, accelerator=accelerator)


def describe(bundle_dir: str | Path) -> str:
    """A plain-text summary of a bundle: question, training graph, metrics, files."""
    bundle_dir = Path(bundle_dir)
    spec = json.loads((bundle_dir / "training-spec.json").read_text())
    provenance = json.loads((bundle_dir / "_provenance.json").read_text())
    env = provenance.get("environment", {})
    created = str(provenance.get("created_at", ""))
    lines = [f"Model bundle: {bundle_dir}",
             f"Question (PQL): {spec['predictive_query']}",
             f"Trained: {created[:16].replace('T', ' ')} UTC with kumo-engine {env.get('kumo_engine')} "
             f"(torch {env.get('torch')}, Python {env.get('python')})"]
    task = bundle_dir / "prediction-schema.json"
    if task.is_file():
        lines.append(f"Task type: {json.loads(task.read_text()).get('task_type')}")
    defaults = bundle_dir / "scoring-defaults.json"
    saved = {k: v for k, v in json.loads(defaults.read_text()).items() if v is not None} if defaults.is_file() else {}
    lines.append("Scoring defaults: " + (", ".join(f"{k} {v}" for k, v in saved.items()) or "none saved"))
    metrics = bundle_dir / "metrics.json"
    if metrics.is_file():
        values = json.loads(metrics.read_text())
        for split, label in (("test", "Test results"), ("val", "Validation results")):
            shown = [f"{k[len(split) + 1:]} {v:.3f}" for k, v in values.items()
                     if k.startswith(f"{split}_") and not k.endswith("_loss") and isinstance(v, (int, float))]
            if shown:
                lines.append(f"{label}: " + ", ".join(shown))
    lines.append("Trained on:")
    for name, table in _tables(spec["graph_config"]).items():
        links = ", ".join(f"{fkey}->{dst}" for fkey, dst in sorted(table["fkeys"]))
        lines.append(f"  {name}: key={table['pkey']} time={table['time']} "
                     f"columns={list(table['cols'])}" + (f" links={links}" if links else ""))
    files = sorted(p for p in bundle_dir.rglob("*") if p.is_file())
    lines.append(f"Files: {len(files)}, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB "
                 f"(weights: {sum(p.stat().st_size for p in files if 'checkpoint' in p.parts) / 1e6:.1f} MB)")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "describe":
        sys.exit("usage: python kumo_bundle.py describe <bundle_dir>")
    print(describe(sys.argv[2]))
