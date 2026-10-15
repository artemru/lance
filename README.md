# Notes: SUFFIX_ARRAY index (lance-format/lance#6655)

This branch holds only notes; it contains no lance code. The code lives on these branches of this fork:

- **`feat/suffix-array-index`**: the PR branch. It has the fixes for the review comments, a merge with main (2026-10-14), a fix so that segments end on document boundaries, and a switch to `libsais-rs`.
- **`feat/fm-index-python`**: a separate small change that adds `"FM"` to `create_scalar_index` in Python.

## Contents

- `REVIEW_REPLIES_DRAFT.md`: replies to the 5 review comments, and a "Relation to the FM-index (#7026)" section for the PR description.
- `BENCHMARK.md`: a comparison of SUFFIX_ARRAY, the FM-index and a full scan on 2M and 10M rows of real English text.
- `harness/`: the benchmark and diagnosis scripts. They are python-fire CLIs.
  - Set `SA_BENCH_PREFIX` to the directory holding your test copies; the scripts refuse to write anywhere else.
  - `queries.json` is not included because it quotes the private corpus. Its format is described in the `sa_bench.py` docstring.

```
SA_BENCH_PREFIX=s3://bucket/path/ python harness/sa_bench.py build s3://bucket/path/copy.lance
python harness/sa_bench.py query s3://bucket/path/copy.lance --queries=queries.json
python harness/fm_bench.py build s3://bucket/path/copy.lance
python harness/sa_diag.py s3://bucket/path/copy.lance --queries="word1,word2"
```

## Open follow-ups

1. Report each segment's real size to the index cache instead of `file_size / 10`. At 52 segments the default 6 GiB cache thrashes: warm counts take 600 ms instead of 2 ms.
2. Build segments in parallel, or use smaller segments. A 2M-row build takes 453 s, against 14 s for the FM-index.
3. Optionally, let SUFFIX_ARRAY accelerate SQL `contains()` filters.
4. Reopen the PR, post the replies and add the FM-index section to the description.
