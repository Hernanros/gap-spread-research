"""Estimate Databento data costs before any fetch."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import os
import databento as db

client = db.Historical()

START = "2019-01-01"
END   = "2026-06-22"

# Candidate instruments for universe expansion
queries = [
    # (label, dataset, symbols, stype_in)
    ("NKD futures (Nikkei CME)",      "GLBX.MDP3",  ["NKD.c.0"],  "continuous"),
    ("NIY futures (Nikkei JPY CME)",  "GLBX.MDP3",  ["NIY.c.0"],  "continuous"),
    ("FDAX (DAX Eurex)",              "XEUR.EOBI",  ["FDAX.c.0"], "continuous"),
    # For sanity-check: what we already have
    ("ES (for reference, already have)", "GLBX.MDP3", ["ES.c.0"],  "continuous"),
]

schemas = ["ohlcv-1d", "ohlcv-1h", "ohlcv-1m"]

print(f"Cost estimate for {START} → {END} (7.5 years)")
print(f"{'instrument':45s}  {'schema':10s}  {'cost USD':>10s}")
print("-" * 75)

results = []
for label, dataset, symbols, stype_in in queries:
    for schema in schemas:
        try:
            cost = client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                stype_in=stype_in,
                schema=schema,
                start=START,
                end=END,
            )
            print(f"{label:45s}  {schema:10s}  ${cost:>9.4f}")
            results.append((label, schema, cost, None))
        except Exception as e:
            err = str(e).split("\n")[0][:120]
            print(f"{label:45s}  {schema:10s}  ERR: {err}")
            results.append((label, schema, None, err))
    print()

# Also try a "fits in monthly free credit" check
print("\nSummary table (cost in USD):")
print(f"{'instrument':45s}  {'1-day':>10s}  {'1-hour':>10s}  {'1-min':>10s}")
print("-" * 80)
by_inst = {}
for label, schema, cost, err in results:
    by_inst.setdefault(label, {})[schema] = cost if cost is not None else "—"
for label, costs in by_inst.items():
    row = f"{label:45s}  "
    for schema in schemas:
        v = costs.get(schema, "—")
        row += f"{'$'+format(v, '.4f') if isinstance(v, float) else str(v):>10s}  "
    print(row)
