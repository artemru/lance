"""FM-index vs no-index scan on the same copies and queries as sa_bench.py.

    python fm_bench.py build URI     # create the FM-index (name fm_text)
    python fm_bench.py query URI     # contains() filter per query: rows vs ground truth, cold/warm ms
"""

import json
import os
import time

import fire

from sa_bench import ALLOWED, HERE, _check, _emit, _index, _index_bytes, _open, _rss_gb, _timed


def build(uri, name="fm_text", column="text"):
    _check(uri)
    from lance.indices import IndexConfig

    ds = _open(uri)
    t = time.time()
    ds.create_scalar_index(column, IndexConfig(index_type="fm", parameters={}), name=name)
    build_s = time.time() - t
    ds = _open(uri)
    idx = _index(ds, name)
    size, files = _index_bytes(uri, idx["uuid"])
    _emit("fm_build", {"uri": uri, "name": name, "rows": ds.count_rows(), "build_s": round(build_s, 1),
                       "peak_rss_gb": _rss_gb(), "index_bytes": size, "n_files": len(files),
                       "index_type": idx.get("type"), "stats": ds.stats.index_stats(name)})


def _contains(ds, s, use_index=True):
    lit = s.replace("'", "''")
    return ds.to_table(columns=["id"], filter=f"contains(text, '{lit}')", use_scalar_index=use_index)


def query(uri, queries=os.path.join(HERE, "queries.json"), warm_runs=3, flat_sample=3, skip=""):
    assert uri.startswith(ALLOWED)
    q = json.load(open(queries))
    gt_valid = uri.rstrip("/").endswith(q["dataset"].rstrip("/").rsplit("/", 1)[-1])
    ds = _open(uri)
    plan = ds.scanner(columns=["id"], filter="contains(text, 'the purple quokka')").explain_plan()
    results = []
    skipped = set(skip.split("|")) if skip else set()
    for i, item in enumerate(q["queries"]):
        s = item["query"]
        if s in skipped:
            continue
        tbl, cold, warm = _timed(lambda: _contains(ds, s), warm_runs)
        rec = {"query": s, "kind": item["kind"], "rows": tbl.num_rows, "ms": [cold, warm]}
        if gt_valid:
            rec.update(gt_rows=item["rows"], ok=tbl.num_rows == item["rows"])
        if i < flat_sample or item["kind"] == "absent" and i == len(q["queries"]) - 1:
            flat, fcold, fwarm = _timed(lambda: _contains(ds, s, use_index=False), 1)
            rec.update(flat_rows=flat.num_rows, flat_ms=[fcold, fwarm])
        results.append(rec)
        print(json.dumps(rec), flush=True)
    _emit("fm_query", {"uri": uri, "plan_uses_index": "ScalarIndexQuery" in plan or "MaterializeIndex" in plan
                       or "Index" in plan, "plan": plan[:2000], "n": len(results),
                       "n_ok": sum(bool(r.get("ok")) for r in results) if gt_valid else None,
                       "peak_rss_gb": _rss_gb(), "results": results})


if __name__ == "__main__":
    fire.Fire({"build": build, "query": query})
