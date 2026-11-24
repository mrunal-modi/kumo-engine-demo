"""Shared setup for the three short notebooks: 1_ingest, 2_training and 3_inference.

The notebooks keep only the steps worth reading; the plumbing lives here,
including the one definition of the tables, their keys and types, and their links.
"""
import contextlib
import io
import json
import logging
import time
import warnings
from pathlib import Path

# Known, harmless messages on this stack (Python 3.14, pandas 2.3 with NumPy 2.5, CPU-only Lightning).
for _message in [r".*torch\.jit\.script", r"IProgress not found", r"The 'generic' unit for NumPy timedelta",
                 r".*does not have many workers", r"GPU available but not used",
                 r"Converting a tensor with requires_grad=True", r"Be aware that when using `ckpt_path`",
                 r"The dirpath has changed"]:
    warnings.filterwarnings("ignore", message=_message)

import pandas as pd
import kumo.engine as kumo
from kumoapi.model_plan import RunMode
from kumoapi.typing import Dtype, Stype

DATA_DIR = Path("data")                        # input: customers.csv, products.csv, orders.csv
OUTPUT_DIR = Path("output")                    # output: call_list.csv
MODEL_DIR = Path("models") / "orders-next-30d" # the saved model (ignored by git)
KUMO_WORK = Path("scratch") / "kumo_work"      # Kumo's index, practice tables and training runs (ignored by git)

QUESTION = "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH customers.customer_id"
AS_OF = pd.Timestamp("2026-11-15")             # last day of order history
RECENT_ORDERS = 10                             # orders per customer the model reads
SEED = 7

# (table, foreign key, linked table, linked table's key)
LINKS = [("orders", "customer_id", "customers", "customer_id"),
         ("orders", "product_id", "products", "product_id")]


def start():
    """Point Kumo's working files at scratch/kumo_work and hide routine log lines."""
    for folder in (OUTPUT_DIR, KUMO_WORK):
        folder.mkdir(parents=True, exist_ok=True)
    kumo.configure(scratch_dir=KUMO_WORK.resolve())
    pd.set_option("display.max_colwidth", None)
    for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric",
                 "kumo", "kumo_core", "diskgraph", "optuna"):
        logging.getLogger(name).setLevel(logging.WARNING)


@contextlib.contextmanager
def quiet():
    """Hide progress bars and model summaries that Lightning prints during training and scoring."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def build_graph():
    """The three tables, their keys and types, and the two links between them."""
    files = kumo.FileConnector(root_dir=str(DATA_DIR.resolve()))
    customers = (kumo.Table.from_source_table(files["customers"], ["customer_id", "region", "signup_date"],
                                              primary_key="customer_id")        # name left out: one value per customer
                 .infer_metadata()
                 .with_column("signup_date", dtype=Dtype.date, stype=Stype.timestamp))
    products = kumo.Table.from_source_table(files["products"], primary_key="product_id").infer_metadata()
    orders = (kumo.Table.from_source_table(files["orders"], primary_key="order_id", time_column="order_date")
              .infer_metadata()
              .with_column("order_date", dtype=Dtype.date)
              .with_column("customer_id", stype=Stype.ID)
              .with_column("product_id", stype=Stype.ID))
    return kumo.Graph(tables=[customers, products, orders],
                      edges=[kumo.Edge(src_table=src, fkey=fkey, dst_table=dst) for src, fkey, dst, _ in LINKS])


def _ingest_stats(snapshot):
    """Kumo's own ingest statistics, written next to the index when it was built."""
    folder = Path(snapshot.index_path)
    return (json.loads((folder / "ingest_stats.json").read_text())["stats"],
            json.loads((folder / "edge_stats.json").read_text()))


def index_report(snapshot):
    """What the index holds: records per table and links per relationship (Kumo's counts)."""
    stats, _ = _ingest_stats(snapshot)
    rows = [{"In the index": f"{table} records", "Count": f"{stats['node_counts'][table]:,}"}
            for table in ("customers", "products", "orders")]
    rows += [{"In the index": f"links {src} → {dst}",
              "Count": f"{stats['edge_counts'][f'{src}_{fkey}_to_{dst}_{pkey}']:,}"}
             for src, fkey, dst, pkey in LINKS]
    rows.append({"In the index": "total records / links (each link is stored both ways)",
                 "Count": f"{stats['total_nodes']:,} / {stats['total_edges']:,}"})
    return pd.DataFrame(rows).set_index("In the index")


def health_check(snapshot):
    """The index's link statistics, in business terms."""
    stats, edges = _ingest_stats(snapshot)
    orders = stats["node_counts"]["orders"]
    customers = stats["node_counts"]["customers"]
    to_customer = stats["edge_counts"]["orders_customer_id_to_customers_customer_id"]
    to_product = stats["edge_counts"]["orders_product_id_to_products_product_id"]
    share_with_orders = next(v["value"] for k, v in edges.items()
                             if k.startswith("orders__") and "__customers__" in k
                             and k.endswith("customer_id__customer_id__FRAC_DST_NONZERO_DEGREE"))
    with_orders = round(share_with_orders * customers)

    def unmatched(count, what):
        if count == 0:
            return f"every order's {what} is known"
        return f"{count} order{'s' if count > 1 else ''} name{'' if count > 1 else 's'} a {what} that isn't in the {what} table"

    return pd.DataFrame([
        {"Check": "Orders linked to a known customer", "Result": f"{to_customer:,} of {orders:,}",
         "What it means": unmatched(orders - to_customer, "customer")},
        {"Check": "Orders linked to a known product", "Result": f"{to_product:,} of {orders:,}",
         "What it means": unmatched(orders - to_product, "product")},
        {"Check": "Customers with at least one order", "Result": f"{with_orders:,} of {customers:,}",
         "What it means": f"the other {customers - with_orders} have no history; the model knows only "
                          f"their region and signup date"},
    ]).set_index("Check")


def practice_summary(training_table):
    """One line on the practice questions Kumo generated from history."""
    rows = pd.read_parquet(training_table.train_table_path)
    dates = sorted(rows.TIMESTAMP.unique())
    print(f"{len(rows):,} practice questions = {rows.ENTITY1.nunique()} customers × {len(dates)} dates, "
          f"every {training_table.generation_plan.timeframe_step} days from "
          f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}")
    print(f"Answered 'yes, ordered within 30 days': {rows.TARGET1.mean():.1%}")
    return rows


def flashcards(practice_rows, customer_id="C001", last=3):
    """A customer's last practice questions: Kumo's date and answer, plus what the order file shows (pandas)."""
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    mine = orders[orders.customer_id == customer_id]
    cards = practice_rows[practice_rows.ENTITY1 == customer_id].sort_values("TIMESTAMP").tail(last)
    window = pd.Timedelta(days=30)
    return pd.DataFrame({
        "As of": [t.date() for t in cards.TIMESTAMP],
        "Orders the model may look at": [mine.order_id[mine.order_date <= t].tolist() or "none yet"
                                         for t in cards.TIMESTAMP],
        "Answer (Kumo): ordered in the next 30 days?": ["yes" if v else "no" for v in cards.TARGET1],
        "The orders behind the answer": [mine.order_id[(mine.order_date > t) & (mine.order_date <= t + window)]
                                         .tolist() or "none" for t in cards.TIMESTAMP],
    }).set_index("As of")


def train(question, snapshot, training_table):
    """Kumo's FAST preset, reading each customer's 10 newest orders, with a fixed seed. Returns (trainer, result)."""
    plan = question.suggest_model_plan(snapshot, run_mode=RunMode.FAST)
    hops = plan.neighbor_sampling.num_neighbors[0]
    hops.hop1["default"] = RECENT_ORDERS
    hops.hop1["customers.customer_id->orders.customer_id"] = RECENT_ORDERS
    hops.hop2["default"] = RECENT_ORDERS
    plan.training_job.manual_seed = SEED
    trainer = kumo.Trainer(plan, accelerator="cpu")
    started = time.time()
    with quiet():
        result = trainer.fit(training_table, snapshot=snapshot)
    print(f"Trained in {time.time() - started:.0f} seconds on the laptop CPU")
    return trainer, result


def exam_results(result):
    """The model's scores on the held-back newest dates, in plain words."""
    exam = result.holdout_df()
    auroc = result.metrics["test_auroc"]
    top = exam.nlargest(len(exam) // 10, "True_PROB")
    base_rate, top_rate = exam.TARGET.mean(), top.TARGET.mean()
    predicted, actual = exam.True_PROB.sum(), int(exam.TARGET.sum())
    dates = ", ".join(str(pd.Timestamp(t).date()) for t in sorted(exam.TIMESTAMP.unique()))
    print(f"Exam: {len(exam):,} practice questions from the newest dates ({dates}), "
          f"never seen in training; {base_rate:.1%} answered yes")
    return pd.DataFrame([
        {"Measure": "Ranking (AUROC)", "Result": f"{auroc:.2f}",
         "In plain words": f"given one customer who ordered and one who didn't, the model ranks the buyer "
                           f"higher {auroc:.0%} of the time (a coin flip would be 50%)"},
        {"Measure": "Top of the list", "Result": f"{top_rate:.0%} vs {base_rate:.1%}",
         "In plain words": f"customers in the model's top 10% ordered at {top_rate / base_rate:.1f}× the "
                           f"average rate"},
        {"Measure": "Forecast", "Result": f"{predicted:.0f} predicted vs {actual} actual",
         "In plain words": f"over the exam dates, the model's probabilities added up to {predicted:.0f} customers "
                           f"ordering, and {actual} did: {predicted / actual - 1:+.0%}, so treat its forecast as "
                           f"a rough guide"},
    ]).set_index("Measure")


def score_everyone(model, snapshot):
    """Every customer's chance of ordering in the 30 days after AS_OF."""
    with quiet():
        batch = model.predict(snapshot, prediction_time=AS_OF.to_pydatetime())
    return batch.frame


def customer_facts():
    """Name, region and order history per customer, from the files (pandas, for readability)."""
    customers = pd.read_csv(DATA_DIR / "customers.csv").set_index("customer_id")
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    orders = orders[orders.order_date <= AS_OF]
    by_customer = orders.groupby("customer_id").order_date
    return customers.assign(**{
        "orders so far": by_customer.count().reindex(customers.index).fillna(0).astype(int),
        "last order": by_customer.max().reindex(customers.index).dt.date,
    })[["name", "region", "orders so far", "last order"]]


def headline(scores):
    """The month's forecast from the model, next to last month's actual (from the order file)."""
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["order_date"])
    last_month = orders[(orders.order_date > AS_OF - pd.Timedelta(days=30)) & (orders.order_date <= AS_OF)]
    known = last_month.customer_id.isin(scores.ENTITY)
    print(f"Expected to order between {AS_OF.date()} and {(AS_OF + pd.Timedelta(days=30)).date()}: "
          f"about {scores.True_PROB.sum():.0f} of {len(scores)} customers "
          f"(the sum of every customer's probability)")
    print(f"For comparison, {last_month.customer_id[known].nunique()} customers ordered in the 30 days "
          f"up to {AS_OF.date()} (from the order file)")
    print(f"Customers with a better than 50% chance: {(scores.True_PROB > 0.5).sum()}")


def call_list(scores):
    """All customers ranked by their chance of ordering, with readable details."""
    # Ties (often customers with no orders, who share one score) are ordered by customer ID.
    ranked = scores.sort_values(["True_PROB", "ENTITY"], ascending=[False, True]).reset_index(drop=True)
    facts = customer_facts().reindex(ranked.ENTITY)
    out = pd.DataFrame({
        "rank": range(1, len(ranked) + 1),
        "customer_id": ranked.ENTITY.to_numpy(),
        **{column: facts[column].to_numpy() for column in facts.columns},
        "chance of ordering in next 30 days": ranked.True_PROB.round(3).to_numpy(),
    })
    out["orders so far"] = out["orders so far"].fillna(0).astype(int)
    out["last order"] = out["last order"].fillna("none")
    return out.set_index("rank")


def _readable(value):
    if isinstance(value, float) and value.is_integer():
        return f"{value:,.0f}"
    text = str(value)
    return text[:10] if text.endswith("T00:00:00") else text


def sources(model, snapshot, customer_id, as_of=AS_OF):
    """Every record Kumo retrieved for one customer as of a date: what the model read to score it."""
    probes = KUMO_WORK / "probes"
    probes.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ENTITY1": [customer_id], "TIMESTAMP": [as_of]}).to_parquet(probes / "probe.parquet", index=False)
    with quiet():
        run = model.predict(snapshot, entities=kumo.FileConnector(root_dir=str(probes.resolve()))["probe"],
                            explanations=kumo.PredictionExplanationOptions(sample_rows=1))
    context = run.explanations.sample(run.explanations.samples()[0].artifact_id)
    store, features = context.subgraph_store, context.feature_store.features
    rows = []
    for table in ("customers", "orders", "products"):
        for node_id in store.tables.get(table, []):
            values = features.get(node_id, {})
            rows.append({"record": table.rstrip("s"), "key": store.nodes[node_id].pkey,
                         "what the model saw": ", ".join(f"{name} {_readable(f.value)}"
                                                         for name, f in values.items())})
    name = customer_facts().name.get(customer_id, customer_id)
    print(f"{name} ({customer_id}) as of {as_of.date()}: {run.frame.True_PROB.iloc[0]:.0%} chance of ordering "
          f"in the next 30 days, from {len(rows)} records")
    return pd.DataFrame(rows).set_index(["record", "key"])
