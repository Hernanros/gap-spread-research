"""Find FDAX availability + cost across Databento Eurex datasets."""
import os as _os
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_REPO_ROOT)

import os
import databento as db

client = db.Historical()

# List datasets the user has access to (or that exist)
print("Listing Databento datasets...")
ds_list = client.metadata.list_datasets()
print(ds_list)
print()

# Try various potential DAX datasets / symbols / start dates
candidates = [
    # (dataset, symbol, stype_in, start)
    ("XEUR.EOBI", "FDAX.c.0", "continuous", "2019-01-01"),
    ("XEUR.EOBI", "FDAX.c.0", "continuous", "2021-01-01"),
    ("XEUR.EOBI", "FDAX.c.0", "continuous", "2023-01-01"),
    # other DAX symbols / micro
    ("XEUR.EOBI", "FDXM.c.0", "continuous", "2023-01-01"),  # mini DAX
    ("XEUR.EOBI", "FDXS.c.0", "continuous", "2023-01-01"),  # micro DAX
]

END = "2026-06-22"
schemas = ["ohlcv-1d", "ohlcv-1h", "ohlcv-1m"]

print(f"{'dataset':12s} {'symbol':12s} {'start':12s} {'schema':10s} {'USD':>10s}")
print("-"*65)
for dataset, sym, stype_in, start in candidates:
    for schema in schemas:
        try:
            cost = client.metadata.get_cost(
                dataset=dataset, symbols=[sym], stype_in=stype_in,
                schema=schema, start=start, end=END,
            )
            print(f"{dataset:12s} {sym:12s} {start:12s} {schema:10s} ${cost:>9.4f}")
        except Exception as e:
            err = str(e).split("\n")[0][:80]
            print(f"{dataset:12s} {sym:12s} {start:12s} {schema:10s} ERR: {err}")
