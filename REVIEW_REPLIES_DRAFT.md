# Draft replies to Xuanwo's review on lance-format/lance#6655 (not posted)

Commits on `artemru/lance:feat/suffix-array-index` (pushed 2026-10-14, head `d11dd2ed0`; PR still closed):
- `db343dee0` fix: address review comments on the suffix array index
- `39347506f` chore: remove dead code and deprecated API use in suffix array index
- `9f4b61877` style: apply cargo fmt and ruff format to the suffix array index code
- `1852bf872` Merge upstream main into feat/suffix-array-index
- `d7fe34b47` style: format and lint the suffix array Python bindings
- `aba68ab9c` fix: end suffix array segments on document boundaries
- `d11dd2ed0` refactor: build suffix arrays with libsais-rs instead of vendored C

---

**index.rs:667: query path writes a raw sidecar**
> Fixed. The build now writes the tokenized text as a raw file (`segment_N_tokenized.raw`), like the suffix array, so the query path never writes. It writes it instead of the Lance `.bin` file, so build I/O is unchanged. Indices built before the raw format only have the `.bin` file; they are now loaded into memory without writing anything. Test: `test_query_legacy_index_never_writes` loads a legacy-layout index through a read-only store wrapper that counts write attempts, queries it, and asserts zero writes and an unchanged file list.

**index.rs:333: statistics returned as row addresses**
> Fixed. `SuffixArrayQuery` now only carries the row-selection `Search` query, and `search()` returns the matching documents' row addresses. Corpus statistics stay on the index's dedicated methods (`total_count`, `compute_prob`, `compute_ntd`, `compute_infgram_prob`). The Python `suffix_array_count` binding now calls `total_count` directly, like the other statistic bindings already did. While moving it, I made the index normalize queries itself for case-insensitive indices, so every entry point lowercases consistently: before, only `search()` and the scanner did.

**index.rs:414: `update` unsupported**
> Fixed. `update` rebuilds the index from the stream of all live rows that `update_criteria` requests (`requires_old_data`), now including `_rowid`, which the rebuild needs. `derive_index_params` now serializes the build options (case insensitivity, separator token id), so the optimizer's from-scratch rebuild path no longer silently drops them. Tests: `test_optimize_suffix_array_index_after_append` (create, append, `optimize_indices`, appended rows searchable, 0 unindexed rows) and `test_update_rebuilds_with_original_params`.

**plugin.rs:818: empty index writes legacy file names**
> Fixed. `write_segment` always uses the segmented `segment_N_*` names, matching the `segments` list recorded in the details; the legacy-name branch was only reachable from the empty path. Test: `test_empty_index_roundtrip` (train on empty input, check the file names, load, query).

**scanner.rs:1887: `_count` type / `_positions` missing**
> Fixed. The planning schema now takes both result columns from the infgram exec's `INFGRAM_SCHEMA` (`_count` UInt32, `_positions` List<UInt64>), so they can't drift from what execution emits. Test: `test_infgram_search_planning_schema_matches_output` compares the planning schema with the executed output for both columns.

---

Also in this update:
- Clippy (`-D warnings`) and `cargo fmt --check` are clean for `lance-index` and `lance`; `ruff format --check` and `ruff check` are clean for the touched Python files. The `chore`/`style` commits fix pre-existing lint and format issues.
- Each new test was checked to fail when its fix is reverted.
- Merged current main (the index APIs moved to `lance-index-core`; `SuffixArray` is now IndexType 13, since 11 and 12 went to Fm and MinHashLsh).
- Testing on a real corpus (2M rows, 5.6 GB of English text, 11 segments) found that the builder cut segments at exactly 512 MiB, splitting a document: its matches before the cut were counted but not returned by search, and n-grams across the cut were lost. Segments now end on document boundaries (`split_segment`, unit-tested). After the fix, 30 queries match a brute-force scan exactly (counts, rows, per-row `_count`).
- Suffix arrays are now built with the `libsais-rs` crate the FM-index already uses; the vendored libsais C sources, their `cc` build (which used `-march=native`) and the hand-written FFI are gone. If #8015 replaces the crate, both indices move together.

---

**Relation to the FM-index (#7026)** (for the PR description)

Both are suffix-array indices; they serve different workloads. The FM-index is a compact filter that accelerates `contains()`; SUFFIX_ARRAY keeps the full suffix array and the text, about 5x larger, so it can count occurrences in O(log n) and read every match position directly, which the infini-gram statistics (`count`, `prob`, `ntd`, `infgram_prob`) and per-row `_count` need. Measured on the same 2M-row, 5.6 GB English corpus (details in BENCHMARK.md):

| | SUFFIX_ARRAY | FM-index | Full scan |
|---|---|---|---|
| Index size | 27.9 GB | 4.8 GB | none |
| Build | 453 s | 14 s | none |
| Rare phrase (1-18 rows) | 5-10 ms | 28-109 ms | ~3.5 s |
| Common word (63k-143k rows) | 78-170 ms | 23-60 s | ~3.6 s |
| "of the" (1.1M rows) | 2.6 s | 24 min | |
| Occurrence count | 0.4 ms | not available | |

