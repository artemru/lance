"""Find rows where the suffix array index disagrees with a brute-force scan."""
import json
import fire
import lance
import pyarrow as pa
import pyarrow.compute as pc


def main(uri, queries="Chapter,function,and", name="sa_text", version=None, region=None):
    # Pin `version` if another benchmark may append to the same copy meanwhile.
    opts = {"region": region} if region else {}
    ds = lance.dataset(uri, version=version, storage_options=opts)
    tbl = ds.to_table(columns=["text"], with_row_address=True)
    text = tbl["text"]
    addrs = tbl["_rowaddr"].to_numpy()
    frag_rows = [(f.fragment_id, f.count_rows()) for f in ds.get_fragments()]
    print("fragments", frag_rows)
    for q in queries.split(","):
        hits = pc.match_substring(text, q).to_numpy(zero_copy_only=False)
        occ = pc.count_substring(text, q).to_numpy(zero_copy_only=False)
        truth = {int(a): int(o) for a, h, o in zip(addrs, hits, occ) if h}
        sa = ds.scanner(columns=["text"], infgram_query=q, with_row_address=True).to_table()
        sa_counts = dict(zip(sa["_rowaddr"].to_pylist(), sa["_count"].to_pylist()))
        missing = sorted(set(truth) - set(sa_counts))
        extra = sorted(set(sa_counts) - set(truth))
        diff = sorted(a for a in truth if a in sa_counts and truth[a] != sa_counts[a])
        print(json.dumps({"query": q, "truth_rows": len(truth), "sa_rows": len(sa_counts), "missing": missing[:10],
                          "extra": extra[:10], "n_count_diff": len(diff),
                          "count_diff_sample": [(a, truth[a], sa_counts[a]) for a in diff[:10]]}))
        for a in missing[:3] + diff[:3]:
            frag, off = a >> 32, a & 0xFFFFFFFF
            t = ds._take_rows([a], columns=["text"])["text"][0].as_py()
            pos = [i for i in range(len(t)) if t.startswith(q, i)]
            print(f"  row addr={a} frag={frag} off={off} len={len(t)} bytes={len(t.encode())} positions={pos[:10]} "
                  f"head={t[:80]!r} tail={t[-80:]!r}")


if __name__ == "__main__":
    fire.Fire(main)
