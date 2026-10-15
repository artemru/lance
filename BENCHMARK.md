# SUFFIX_ARRAY (PR #6655) vs FM-index (main) on real text

Measured 2026-10-14. Each run used one dedicated 192-vCPU cloud node, with data on S3.
Code: PR branch `d11dd2ed0` (merged with main `b0eb9f85e`), release build, pylance 14.0.0-beta.15.

## Data

Text-only copies of an English book-text corpus. The 200 fragments were sampled across the source, and the 2M copy is a subset of the 10M one. The corpus itself is not public, so the queries and raw results are not included here.

| Copy | Rows | Text |
|---|---|---|
| 2M copy | 2,000,000 | 5.56 GB |
| 10M copy | 9,983,491 | 27.4 GB |

There are 30 queries:
- 10 frequent words or bigrams;
- 16 phrases copied from random rows;
- 4 strings that never occur.

Ground truth comes from a brute-force pyarrow scan of the 2M copy: matching rows and occurrences, case-sensitive.

## Correctness (2M)

| Index | Matches ground truth |
|---|---|
| SUFFIX_ARRAY | 30/30 for occurrence counts, matching rows and per-row `_count`. No false positives. |
| FM-index (`contains()` filter) | 28/28 for matching rows. "the" and "and" were not completed (see below). |

Before the segment-boundary fix (`aba68ab9c`), SUFFIX_ARRAY scored 25/30. Segments were cut at exactly 512 MiB, splitting one document per cut, so up to one row or one occurrence per segment was missing.

## Build (2M)

| | SUFFIX_ARRAY | FM-index |
|---|---|---|
| Build time | 453 s (libsais-rs). It was 394 s with the vendored C libsais. | **14 s** |
| Peak RSS | 21 GB | 36 GB |
| Index size | 27.9 GB (5.0× text), 11 segments | **4.8 GB (0.86× text)**, 333 partitions |

At 10M, SUFFIX_ARRAY took 1,696–1,711 s, peak 33 GB, and produced a 137 GB index in 52 segments.

## Query latency (2M, warm, milliseconds)

All three columns return the matching rows' `id` column. The SUFFIX_ARRAY figure is `scanner(infgram_query=...)`. The FM figure is `to_table(filter="contains(text, ...)")`, and `explain_plan` shows `ScalarIndexQuery ... @fm_text(Fm)`.

| Query | Matching rows | SUFFIX_ARRAY | FM-index | No index (full scan) |
|---|---|---|---|---|
| the | 1,677,182 | 13,270 | not completed (stopped after more than 75 min) | |
| of the | 1,128,260 | 2,630 | 1,423,527 | |
| in the | 979,643 | 1,820 | 771,204 | |
| data | 127,807 | 159 | 59,680 | 3,578 |
| function | 143,463 | 170 | 53,825 | |
| However, | 142,308 | 143 | 33,395 | |
| model | 113,531 | 142 | 50,397 | |
| Chapter | 76,962 | 84 | 22,725 | |
| learning | 63,398 | 78 | 23,400 | |
| phrases, 135–450 rows | | 10–11 | 109–205 | |
| phrases, 1–18 rows | | 5–10 | 28–109 | |
| absent strings | 0 | 0.6–0.8 | 19–34 | 3,146 |

SUFFIX_ARRAY also answers occurrence counts in about 0.4 ms warm (about 0.5 s cold, which loads segments from S3). Search returning row addresses only takes 0.3–0.8 ms for rare phrases.

The FM-index runs each query on a single core. Its cost grows with the number of matches, because every match is located by walking the sampled suffix array.

## 10M: index cache size

With Lance's default 6 GiB index cache, SUFFIX_ARRAY warm counts took about 600 ms and rare phrases 300–600 ms. Segments were evicted and reloaded on every call. With `index_cache_size_bytes` set to 48 GiB, warm counts took about 2 ms and rare-phrase searches 1.4–3.9 ms.

Cause: each segment reports an estimated cache size of `(text file + suffix-array file) / 10`. With 52 segments the estimates total about 13.7 GB, which overflows the cache. This is not fixed yet.

## Append (2M)

- 1,000 rows were appended. Until `optimize_indices` ran, they were unindexed and invisible to infgram search.
- `optimize_indices` (a full rebuild) took 392 s.
- Afterwards: 0 unindexed rows, the marker string found in 1,000 rows, and the "the" count up by exactly 1,000.

## Files

The harness is in `harness/`: `sa_bench.py`, `fm_bench.py`, `sa_diag.py` and `sa_diag2.py` (see README.md).
