"""Inspect per-segment row_ids / doc_offsets of a suffix array index around given rows."""
import numpy as np
import fire
import lance
from lance.file import LanceFileReader


def blob_u64(path, opts):
    tbl = LanceFileReader(path, storage_options=opts).read_all().to_table()
    return np.frombuffer(tbl.column(0)[0].as_py(), dtype="<u8")


def main(uri, missing, name="sa_text", version=None, region=None):
    """`missing`: comma-separated row addresses that search failed to return."""
    opts = {"region": region} if region else {}
    ds = lance.dataset(uri, version=version, storage_options=opts)
    idx = [i for i in ds.list_indices() if i["name"] == name][0]
    num_segments = ds.stats.index_stats(name)["indices"][0]["num_segments"]
    d = f"{uri}/_indices/{idx['uuid']}"
    missing = [int(m) for m in str(missing).split(",")]
    total = 0
    for s in range(num_segments):
        rids = blob_u64(f"{d}/segment_{s}_row_ids.bin", opts)
        offs = blob_u64(f"{d}/segment_{s}_doc_offsets.bin", opts)
        total += len(rids)
        info = f"seg {s}: row_ids={len(rids)} doc_offsets={len(offs)} first={rids[0]} last={rids[-1]} corpus_end={offs[-1]}"
        for m in missing:
            hit = np.nonzero(rids == m)[0]
            if len(hit):
                k = int(hit[0])
                info += f"\n   missing row {m} at local_doc {k}/{len(rids)}; doc span [{offs[k-1] if k else 0}, {offs[k]})"
        print(info, flush=True)
    print("total row_ids", total)


if __name__ == "__main__":
    fire.Fire(main)
