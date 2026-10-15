"""SUFFIX_ARRAY index benchmark on a copy of a text dataset (PR lance-format/lance#6655).

Run with a pylance build of the PR branch:

    python sa_bench.py build  URI [--case_insensitive=False] [--name=sa_text]
    python sa_bench.py query  URI [--queries=queries.json] [--name=sa_text] [--warm_runs=5]
    python sa_bench.py append URI [--rows=1000] [--name=sa_text] [--restore=True]

Every command appends one JSON line to results/<command>.jsonl. Only URIs under $SA_BENCH_PREFIX are
accepted, because building, appending and optimizing write into the dataset directory: point it at
the directory holding your test copies. $SA_BENCH_REGION sets the S3 region if needed.

queries.json: {"dataset": <URI of the copy the ground truth was computed on>, "queries": [{"query",
"kind" (frequent | phrase | absent), "rows", "occurrences", "rows_lower", "occurrences_lower"}]}.
Results for other copies are only checked for internal consistency (count vs search vs scanner)
and that search hits really contain the query.
"""

import json
import os
import resource
import statistics
import subprocess
import time

import fire

ALLOWED = os.environ.get("SA_BENCH_PREFIX", "")
OPTS = {"region": os.environ["SA_BENCH_REGION"]} if os.environ.get("SA_BENCH_REGION") else {}
HERE = os.path.dirname(os.path.abspath(__file__))


def _check(uri):
    assert ALLOWED and uri.startswith(ALLOWED) and uri.rstrip("/").endswith(".lance"), f"refusing to touch {uri}"


def _open(uri):
    import lance

    # SA_INDEX_CACHE_GB overrides the default 6 GiB index cache.
    cache_gb = os.environ.get("SA_INDEX_CACHE_GB")
    extra = {"index_cache_size_bytes": int(float(cache_gb) * 2**30)} if cache_gb else {}
    return lance.dataset(uri, storage_options=OPTS, **extra)


def _rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def _emit(cmd, rec):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    rec.update(ts=time.time(), host=os.uname().nodename)
    with open(os.path.join(HERE, "results", f"{cmd}.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def _index(ds, name):
    for idx in ds.list_indices():
        if idx["name"] == name:
            return idx
    raise KeyError(f"no index named {name}; have {[i['name'] for i in ds.list_indices()]}")


def _index_bytes(uri, uuid):
    out = subprocess.run(["aws", "s3", "ls", "--recursive", "--summarize", f"{uri.rstrip('/')}/_indices/{uuid}/"],
                         capture_output=True, text=True).stdout
    files = [ln.split()[-1].rsplit("/", 1)[-1] for ln in out.splitlines() if ln[:4].isdigit()]
    size = next((int(ln.split()[-1]) for ln in out.splitlines() if "Total Size" in ln), None)
    return size, files


def build(uri, case_insensitive=False, name="sa_text", column="text"):
    """Create the index; record build time, peak RSS, index size and file list."""
    _check(uri)
    ds = _open(uri)
    t = time.time()
    ds.create_scalar_index(column, index_type="SUFFIX_ARRAY", name=name, case_insensitive=case_insensitive)
    build_s = time.time() - t
    ds = _open(uri)
    idx = _index(ds, name)
    size, files = _index_bytes(uri, idx["uuid"])
    _emit("build", {"uri": uri, "name": name, "case_insensitive": case_insensitive, "rows": ds.count_rows(),
                    "build_s": round(build_s, 1), "peak_rss_gb": _rss_gb(), "index_bytes": size,
                    "n_files": len(files), "files_sample": sorted(files)[:12],
                    "stats": ds.stats.index_stats(name)})


def _timed(fn, runs):
    t = time.time()
    first = fn()
    cold = time.time() - t
    warm = []
    for _ in range(runs):
        t = time.time()
        fn()
        warm.append(time.time() - t)
    return first, round(cold * 1e3, 2), round(statistics.median(warm) * 1e3, 2) if warm else None


def query(uri, queries=os.path.join(HERE, "queries.json"), name="sa_text", warm_runs=5, case_insensitive=False,
          check_hits=50):
    """count / search / scanner per query vs ground truth; cold = first call in this process, warm = median."""
    import pyarrow.compute as pc

    q = json.load(open(queries))
    gt_valid = uri.rstrip("/").endswith(q["dataset"].rstrip("/").rsplit("/", 1)[-1])
    t = time.time()
    ds = _open(uri)
    open_s = time.time() - t
    rows_key, occ_key = ("rows_lower", "occurrences_lower") if case_insensitive else ("rows", "occurrences")
    results = []
    for item in q["queries"]:
        s = item["query"]
        count, count_cold, count_warm = _timed(lambda: ds._suffix_array_info(name, s)["count"], warm_runs)
        addrs, search_cold, search_warm = _timed(
            lambda: ds._suffix_array_info(name, s, operation="search", max_results=2**31)["row_addrs"], warm_runs)
        tbl, scan_cold, scan_warm = _timed(
            lambda: ds.scanner(columns=["id"], infgram_query=s, with_row_address=True).to_table(), max(1, warm_runs // 2))
        bad_hits = 0
        if addrs and check_hits:
            sample = sorted(addrs)[:check_hits]
            text = ds._take_rows(sample, columns=["text"])["text"]
            if case_insensitive:
                text, s_cmp = pc.utf8_lower(text), s.lower()
            else:
                s_cmp = s
            bad_hits = len(sample) - int(pc.sum(pc.match_substring(text, s_cmp)).as_py() or 0)
        rec = {"query": s, "kind": item["kind"], "count": count, "search_rows": len(addrs),
               "scan_rows": tbl.num_rows, "scan_count_sum": int(pc.sum(tbl["_count"]).as_py() or 0)
               if "_count" in tbl.column_names else None,
               "bad_hits_in_sample": bad_hits, "count_ms": [count_cold, count_warm],
               "search_ms": [search_cold, search_warm], "scan_ms": [scan_cold, scan_warm]}
        if gt_valid:
            rec.update(gt_rows=item[rows_key], gt_occurrences=item[occ_key],
                       ok=(count == item[occ_key] and len(addrs) == item[rows_key] and tbl.num_rows == item[rows_key]))
        results.append(rec)
        print(json.dumps(rec), flush=True)
    phrases = [r for r in results if r["kind"] == "phrase"]
    probe = phrases[0]["query"].split(" ") if phrases else ["of", "the"]
    prob, prob_cold, prob_warm = _timed(
        lambda: ds._suffix_array_info(name, " ".join(probe[:-1]) + " ", operation="infgram_prob",
                                      continuation=probe[-1]), warm_runs)
    _emit("query", {"uri": uri, "name": name, "case_insensitive": case_insensitive, "open_s": round(open_s, 2),
                    "gt_checked": gt_valid, "n": len(results),
                    "n_ok": sum(bool(r.get("ok")) for r in results) if gt_valid else None,
                    "n_bad_hits": sum(r["bad_hits_in_sample"] for r in results),
                    "infgram_prob_example": {"result": prob, "ms": [prob_cold, prob_warm]},
                    "peak_rss_gb": _rss_gb(), "results": results})


def append(uri, rows=1000, name="sa_text", restore=True):
    """Append rows with a unique marker, optimize_indices, check they are found; optionally restore the version."""
    import pyarrow as pa

    _check(uri)
    ds = _open(uri)
    base_version = ds.version
    before = ds._suffix_array_info(name, "the")["count"]
    start = ds.count_rows()
    marker = "quokkamarker"
    tbl = pa.table({"id": pa.array(range(start, start + rows), pa.int64()),
                    "src_row_addr": pa.array([-1] * rows, pa.int64()),
                    "book_id": pa.array(["sa-append-test"] * rows),
                    "char_offset": pa.array([0] * rows, pa.int64()),
                    "text": pa.array([f"appended row {i} with {marker} and the end" for i in range(rows)])})
    import lance

    lance.write_dataset(tbl, uri, mode="append", storage_options=OPTS)
    ds = _open(uri)
    unindexed_before = ds.stats.index_stats(name).get("num_unindexed_rows")
    scan_before_opt = ds.scanner(columns=["id"], infgram_query=marker).to_table().num_rows  # unindexed rows via flat search?
    t = time.time()
    ds.optimize.optimize_indices(index_names=[name])
    opt_s = time.time() - t
    ds = _open(uri)
    rec = {"uri": uri, "rows_appended": rows, "unindexed_before_opt": unindexed_before,
           "unindexed_after_opt": ds.stats.index_stats(name).get("num_unindexed_rows"),
           "optimize_s": round(opt_s, 1), "marker_scan_rows_before_opt": scan_before_opt,
           "marker_count_after": ds._suffix_array_info(name, marker)["count"],
           "marker_search_rows_after": len(ds._suffix_array_info(name, marker, operation="search", max_results=2**31)["row_addrs"]),
           "marker_scan_rows_after": ds.scanner(columns=["id"], infgram_query=marker).to_table().num_rows,
           "the_count_before": before, "the_count_after": ds._suffix_array_info(name, "the")["count"],
           "expected_the_delta": rows, "peak_rss_gb": _rss_gb()}
    if restore:
        ds.checkout_version(base_version).restore()
        rec["restored_to_version"] = base_version
        rec["rows_after_restore"] = _open(uri).count_rows()
    _emit("append", rec)


if __name__ == "__main__":
    fire.Fire({"build": build, "query": query, "append": append})
