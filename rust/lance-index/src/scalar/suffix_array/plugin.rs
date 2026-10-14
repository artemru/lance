// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! SuffixArrayIndexPlugin implementing ScalarIndexPlugin.
//!
//! Handles training (building) and loading of suffix array indices.
//! For corpora that exceed `MAX_SEGMENT_BYTES` (~512 MB) the build
//! automatically splits the data into multiple segments, each with
//! its own suffix array.  This keeps individual suffix arrays within
//! the i32 addressing limit while providing transparent aggregation
//! at query time.
//!
//! Segment SA construction is parallelized using rayon to saturate
//! available CPU cores, providing near-linear speedup on multi-core
//! machines.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, Field};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::StreamExt;
use lance_core::{Error, ROW_ID, Result};
use prost::Message;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::pb;
use crate::progress::IndexBuildProgress;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::{CreatedIndex, IndexStore, RowIdRemapper, ScalarIndex};

use super::bloom::BloomFilter;
use super::builder::{
    build_suffix_array, compact_suffix_array, compute_pointer_width, release_memory_to_os,
};
use super::index::SuffixArrayIndex;

const SUFFIX_ARRAY_INDEX_VERSION: u32 = 0;

/// Maximum text bytes per segment (~512 MB).
/// Each segment needs its own suffix array with i32 pointers, so must be < 2 GB.
/// 512 MB balances SA working set size vs. parallelism granularity.
const MAX_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

/// Maximum number of parallel SA builder threads.
/// Each builder uses ~5×N memory (N = segment size), so 4 threads × 512MB
/// = ~10 GB for SA construction. Combined with segment text buffers (~19 GB
/// for large datasets), total stays within typical 96 GB cgroup limits.
/// Configurable via LANCE_SA_PARALLEL_BUILDERS env var.
fn max_parallel_builders() -> usize {
    std::env::var("LANCE_SA_PARALLEL_BUILDERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4)
}

/// Result of building a suffix array for a single segment.
struct BuiltSegment {
    seg_idx: usize,
    text: Vec<u8>,
    compact_sa: Vec<u8>,
    pointer_width: usize,
    corpus_bytes: u64,
    /// Cumulative byte offsets marking the end of each document.
    /// `doc_offsets[i]` is the byte offset in `text` where document `i` ends.
    /// Document `i` spans `[doc_offsets[i-1], doc_offsets[i])` (with doc_offsets[-1] = 0).
    doc_offsets: Vec<u64>,
    /// Lance row IDs for each document.
    /// `row_ids[i]` is the `_rowid` (fragment-encoded) for document `i`.
    /// This allows `search_rows` to return correct row addresses.
    row_ids: Vec<u64>,
    /// Serialized bloom filter bytes for this segment's n-grams.
    /// Used to quickly skip segments that definitely don't contain a query.
    bloom_bytes: Vec<u8>,
}

/// Plugin for creating and loading suffix array indices.
#[derive(Debug, Default)]
pub struct SuffixArrayIndexPlugin;

/// Parameters parsed from the JSON params string for SA index creation.
///
/// Serialized back by [`SuffixArrayIndex::derive_index_params`](crate::scalar::ScalarIndex::derive_index_params)
/// so that a rebuild keeps the original build options.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SuffixArrayBuildParams {
    /// When true, lowercase all text at build time for case-insensitive matching.
    #[serde(default)]
    pub case_insensitive: bool,
    /// Separator token ID placed between documents in token-level indices.
    /// Defaults to 0xFFFF for u16 tokens and 0xFFFFFFFF for u32 tokens.
    #[serde(default)]
    pub separator_token_id: Option<u32>,
}

/// Training request that carries SA build parameters.
pub struct SuffixArrayTrainingRequest {
    criteria: TrainingCriteria,
    params: SuffixArrayBuildParams,
}

impl SuffixArrayTrainingRequest {
    /// Build requests read the indexed column together with `_rowid`, which
    /// maps each document back to its row.
    pub fn new(params: SuffixArrayBuildParams) -> Self {
        Self {
            criteria: TrainingCriteria::new(TrainingOrdering::None).with_row_id(),
            params,
        }
    }
}

impl TrainingRequest for SuffixArrayTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

/// Write one pre-built segment's (text, compact_sa, doc_offsets, row_ids) to the index store.
///
/// Files always use the segmented `segment_{N}_*` names: the index details
/// always record a `segments` list, so readers load through
/// [`SuffixArrayIndex::load_multi`], which expects those names.
async fn write_segment(
    index_store: &dyn IndexStore,
    built: &BuiltSegment,
) -> Result<pb::SuffixArraySegmentInfo> {
    let seg_idx = built.seg_idx;
    let offsets_name = format!("segment_{seg_idx}_doc_offsets.bin");
    let row_ids_name = format!("segment_{seg_idx}_row_ids.bin");
    let bloom_name = format!("segment_{seg_idx}_bloom.bin");

    // The tokenized text and the suffix array are written as raw bytes so
    // queries can fetch 64 KB blocks with range reads instead of downloading
    // whole (multi-GB) files. The reader derives these `.raw` names from the
    // `.bin` names kept in the segment metadata, and falls back to the Lance
    // `.bin` files of indices built before the raw format. Both files are
    // written here, at build time: the query path never writes to the store.
    index_store
        .write_raw_file(&format!("segment_{seg_idx}_tokenized.raw"), &built.text)
        .await?;
    index_store
        .write_raw_file(
            &format!("segment_{seg_idx}_suffix_array.raw"),
            &built.compact_sa,
        )
        .await?;

    // Write document offsets (u64 array, one per document)
    let offsets_bytes: Vec<u8> = built
        .doc_offsets
        .iter()
        .flat_map(|o| o.to_le_bytes())
        .collect();
    let offsets_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "data",
        arrow_schema::DataType::LargeBinary,
        false,
    )]));
    let offsets_batch = arrow_array::RecordBatch::try_new(
        offsets_schema.clone(),
        vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
            offsets_bytes.as_slice(),
        ]))],
    )?;
    let mut offsets_writer = index_store
        .new_index_file(&offsets_name, offsets_schema)
        .await?;
    offsets_writer.write_record_batch(offsets_batch).await?;
    offsets_writer.finish().await?;

    // Write row IDs (u64 array, one per document — Lance row addresses)
    let row_ids_bytes: Vec<u8> = built.row_ids.iter().flat_map(|r| r.to_le_bytes()).collect();
    let row_ids_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "data",
        arrow_schema::DataType::LargeBinary,
        false,
    )]));
    let row_ids_batch = arrow_array::RecordBatch::try_new(
        row_ids_schema.clone(),
        vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
            row_ids_bytes.as_slice(),
        ]))],
    )?;
    let mut row_ids_writer = index_store
        .new_index_file(&row_ids_name, row_ids_schema)
        .await?;
    row_ids_writer.write_record_batch(row_ids_batch).await?;
    row_ids_writer.finish().await?;

    // Write bloom filter (serialized bytes for n-gram existence checks)
    if !built.bloom_bytes.is_empty() {
        let bloom_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "data",
            arrow_schema::DataType::LargeBinary,
            false,
        )]));
        let bloom_batch = arrow_array::RecordBatch::try_new(
            bloom_schema.clone(),
            vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
                built.bloom_bytes.as_slice(),
            ]))],
        )?;
        let mut bloom_writer = index_store
            .new_index_file(&bloom_name, bloom_schema)
            .await?;
        bloom_writer.write_record_batch(bloom_batch).await?;
        bloom_writer.finish().await?;
    }

    // Return freed memory to the OS. Without this, glibc retains freed pages
    // in the process address space, causing RSS to grow monotonically across
    // segments and eventually OOM during the commit phase.
    release_memory_to_os();

    Ok(pb::SuffixArraySegmentInfo {
        pointer_width: built.pointer_width as u32,
        total_entries: built.corpus_bytes,
        corpus_bytes: built.corpus_bytes,
        num_documents: built.doc_offsets.len() as u64,
    })
}

/// Sentinel byte placed between documents in text-level (byte) SA indices.
/// Prevents cross-document boundary matches. 0x00 cannot appear in valid
/// UTF-8 text, so it's safe to use as a separator.
const TEXT_SEPARATOR_BYTE: u8 = 0x00;

/// Extract text bytes from a column array, appending to `out`.
/// Also records the cumulative byte offset after each document in `doc_offsets`.
/// When `case_insensitive` is true, text is lowercased before appending.
/// For list<int16/int32> columns, each element's tokens are serialized as
/// fixed-width little-endian integers with a separator token between documents.
/// A sentinel byte (0x00) is inserted between text documents to prevent
/// cross-document boundary matches.
/// Returns the number of non-null rows processed.
fn extract_bytes(
    col: &dyn arrow_array::Array,
    out: &mut Vec<u8>,
    doc_offsets: &mut Vec<u64>,
    case_insensitive: bool,
    separator_token_id: Option<u32>,
) -> Result<u64> {
    let mut docs = 0u64;
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| Error::invalid_input("Failed to downcast to StringArray"))?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    if case_insensitive {
                        out.extend_from_slice(arr.value(i).to_lowercase().as_bytes());
                    } else {
                        out.extend_from_slice(arr.value(i).as_bytes());
                    }
                    out.push(TEXT_SEPARATOR_BYTE);
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| Error::invalid_input("Failed to downcast to LargeStringArray"))?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    if case_insensitive {
                        out.extend_from_slice(arr.value(i).to_lowercase().as_bytes());
                    } else {
                        out.extend_from_slice(arr.value(i).as_bytes());
                    }
                    out.push(TEXT_SEPARATOR_BYTE);
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::Binary => {
            let arr = col
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| Error::invalid_input("Failed to downcast to BinaryArray"))?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.extend_from_slice(arr.value(i));
                    out.push(TEXT_SEPARATOR_BYTE);
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::LargeBinary => {
            let arr = col
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| Error::invalid_input("Failed to downcast to LargeBinaryArray"))?;
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    out.extend_from_slice(arr.value(i));
                    out.push(TEXT_SEPARATOR_BYTE);
                    doc_offsets.push(out.len() as u64);
                    docs += 1;
                }
            }
        }
        DataType::List(_) | DataType::LargeList(_) => {
            docs = extract_token_list_bytes(col, out, doc_offsets, separator_token_id)?;
        }
        other => {
            return Err(Error::invalid_input(format!(
                "Unsupported column type for suffix array index: {other:?}"
            )));
        }
    }
    Ok(docs)
}

/// Extract token IDs from a list column and serialize them as fixed-width
/// little-endian bytes. Inserts a separator token between documents.
///
/// Supports List<Int16>, List<UInt16>, List<Int32>, List<UInt32> and
/// their LargeList variants. Returns the number of non-null rows.
fn extract_token_list_bytes(
    col: &dyn arrow_array::Array,
    out: &mut Vec<u8>,
    doc_offsets: &mut Vec<u64>,
    separator_token_id: Option<u32>,
) -> Result<u64> {
    use arrow_array::{Int16Array, Int32Array, Int64Array, UInt16Array, UInt32Array, UInt64Array};

    // Determine element type and token width from the list's value type.
    let (elem_type, values_ref) =
        if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
            (
                list_arr.values().data_type().clone(),
                list_arr.values().clone(),
            )
        } else if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            (
                list_arr.values().data_type().clone(),
                list_arr.values().clone(),
            )
        } else {
            return Err(Error::invalid_input("Expected List or LargeList array"));
        };

    // Int64/UInt64 are downcast to 4-byte tokens (validated below).
    let token_width: usize = match &elem_type {
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Int64 | DataType::UInt64 => 4,
        other => {
            return Err(Error::invalid_input(format!(
                "Unsupported list element type for token SA: {other:?}"
            )));
        }
    };

    let sep = separator_token_id.unwrap_or({
        if token_width == 2 {
            0xFFFF
        } else {
            0xFFFF_FFFF
        }
    });
    let sep_bytes: Vec<u8> = sep.to_le_bytes()[..token_width].to_vec();

    let num_rows = col.len();
    let mut docs = 0u64;

    // Helper: get (start, end) offsets for row i, or None if null.
    let get_offsets = |i: usize| -> Option<(usize, usize)> {
        if col.is_null(i) {
            return None;
        }
        if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::ListArray>() {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            Some((start, end))
        } else if let Some(list_arr) = col.as_any().downcast_ref::<arrow_array::LargeListArray>() {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            Some((start, end))
        } else {
            None
        }
    };

    for i in 0..num_rows {
        let Some((start, end)) = get_offsets(i) else {
            continue;
        };

        // Insert separator between documents (not before the first)
        if docs > 0 {
            out.extend_from_slice(&sep_bytes);
        }

        // Serialize each token as little-endian bytes
        match &elem_type {
            DataType::Int16 => {
                let arr = values_ref.as_any().downcast_ref::<Int16Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::UInt16 => {
                let arr = values_ref.as_any().downcast_ref::<UInt16Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::Int32 => {
                let arr = values_ref.as_any().downcast_ref::<Int32Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::UInt32 => {
                let arr = values_ref.as_any().downcast_ref::<UInt32Array>().unwrap();
                for j in start..end {
                    out.extend_from_slice(&arr.value(j).to_le_bytes());
                }
            }
            DataType::Int64 => {
                let arr = values_ref.as_any().downcast_ref::<Int64Array>().unwrap();
                for j in start..end {
                    let v = arr.value(j);
                    if v < i32::MIN as i64 || v > i32::MAX as i64 {
                        return Err(Error::invalid_input(format!(
                            "Token value {v} at row {i} exceeds i32 range. \
                             Cast your column to list<int32> or use smaller token IDs."
                        )));
                    }
                    out.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            DataType::UInt64 => {
                let arr = values_ref.as_any().downcast_ref::<UInt64Array>().unwrap();
                for j in start..end {
                    let v = arr.value(j);
                    if v > u32::MAX as u64 {
                        return Err(Error::invalid_input(format!(
                            "Token value {v} at row {i} exceeds u32 range. \
                             Cast your column to list<uint32> or use smaller token IDs."
                        )));
                    }
                    out.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
            _ => unreachable!(),
        }

        doc_offsets.push(out.len() as u64);
        docs += 1;
    }

    Ok(docs)
}

/// Splits the leading segment off an accumulated corpus.
///
/// `doc_offsets[i]` is the end of document `i` in `bytes` and `row_ids[i]` its
/// row. The segment ends at the last document boundary at or below `max_bytes`,
/// or after the first document if that one alone is larger, so no document
/// spans two segments: a match inside a split document could not be mapped back
/// to its row, and an n-gram across the split would be lost. The remaining
/// offsets are rebased to the start of what is left in `bytes`.
fn split_segment(
    bytes: &mut Vec<u8>,
    doc_offsets: &mut Vec<u64>,
    row_ids: &mut Vec<u64>,
    max_bytes: usize,
) -> (Vec<u8>, Vec<u64>, Vec<u64>) {
    let num_docs = doc_offsets
        .partition_point(|&o| o <= max_bytes as u64)
        .max(1);
    let split_point = doc_offsets[num_docs - 1];
    let rest = bytes.split_off(split_point as usize);
    let remaining_offsets = doc_offsets[num_docs..]
        .iter()
        .map(|&o| o - split_point)
        .collect();
    let mut segment_offsets = std::mem::replace(doc_offsets, remaining_offsets);
    segment_offsets.truncate(num_docs);
    let remaining_row_ids = row_ids.split_off(num_docs);
    (
        std::mem::replace(bytes, rest),
        segment_offsets,
        std::mem::replace(row_ids, remaining_row_ids),
    )
}

#[async_trait]
impl BasicTrainer for SuffixArrayIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        // Validate field type: accept text/binary or list of integer tokens
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {}
            DataType::List(inner) | DataType::LargeList(inner) => {
                match inner.data_type() {
                    DataType::Int16 | DataType::UInt16 | DataType::Int32 | DataType::UInt32 => {}
                    // Int64/UInt64 accepted — values validated at build time and
                    // stored as 4-byte (i32) tokens.
                    DataType::Int64 | DataType::UInt64 => {}
                    other => {
                        return Err(Error::invalid_input(format!(
                            "Suffix array index on list columns requires Int16, UInt16, Int32, UInt32, \
                             Int64, or UInt64 elements, got {other:?}"
                        )));
                    }
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Suffix array index requires Utf8, LargeUtf8, Binary, LargeBinary, or \
                     List<Int16/Int32/Int64> column, got {other:?}"
                )));
            }
        }
        let build_params: SuffixArrayBuildParams = if params.is_empty() {
            SuffixArrayBuildParams::default()
        } else {
            serde_json::from_str(params).map_err(|e| {
                Error::invalid_input(format!("Failed to parse suffix array index params: {e}"))
            })?
        };
        Ok(Box::new(SuffixArrayTrainingRequest::new(build_params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        // Extract build params from the training request
        let build_params = request
            .as_any()
            .downcast_ref::<SuffixArrayTrainingRequest>()
            .map(|r| r.params.clone())
            .unwrap_or_default();
        let case_insensitive = build_params.case_insensitive;
        let separator_token_id = build_params.separator_token_id;

        if case_insensitive {
            info!("Building case-insensitive suffix array index (lowercasing all text)");
        }

        let schema = data.schema();
        let value_col_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "value")
            .ok_or_else(|| Error::invalid_input("Training data stream missing 'value' column"))?;

        // Detect token_width from the column type. For text/binary columns it's 1 (byte-level).
        // For list<int16/uint16> columns it's 2, for list<int32/uint32> it's 4.
        let value_field = &schema.fields()[value_col_idx];
        let token_width: u32 = match value_field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => 1,
            DataType::List(inner) | DataType::LargeList(inner) => {
                match inner.data_type() {
                    DataType::Int16 | DataType::UInt16 => 2,
                    DataType::Int32 | DataType::UInt32 => 4,
                    // Int64/UInt64 values are validated and stored as 4-byte tokens
                    DataType::Int64 | DataType::UInt64 => 4,
                    other => {
                        return Err(Error::invalid_input(format!(
                            "Unsupported list element type for suffix array: {other:?}"
                        )));
                    }
                }
            }
            other => {
                return Err(Error::invalid_input(format!(
                    "Unsupported column type for suffix array index: {other:?}"
                )));
            }
        };

        if token_width > 1 {
            let sep = separator_token_id.unwrap_or({
                if token_width == 2 {
                    0xFFFF
                } else {
                    0xFFFF_FFFF
                }
            });
            info!(
                token_width,
                separator_token_id = sep,
                "Building token-level suffix array index"
            );
        }

        // ── Streaming pipeline: read → batch → build SA → write → repeat ──
        //
        // Instead of reading ALL text into memory first (which can OOM on large
        // datasets, especially when reading from S3), we interleave reading with
        // building and writing. We accumulate just enough text segments for one
        // batch, then build their suffix arrays, write them, free memory, and
        // continue reading from the stream.
        //
        // Peak memory ≈ batch_size × (512MB text + 2.5GB SA working set) ≈ 10-20GB
        // instead of ≈ 19GB text + batch × SA working set ≈ 40-80GB.

        let parallel_threads = max_parallel_builders();
        let mut stream = data;
        let mut current_bytes: Vec<u8> = Vec::new();
        let mut current_doc_offsets: Vec<u64> = Vec::new();
        let mut current_row_ids: Vec<u64> = Vec::new();
        let mut pending_segments: Vec<(Vec<u8>, Vec<u64>, Vec<u64>)> = Vec::new();
        let mut total_documents: u64 = 0;
        let mut total_corpus_bytes: u64 = 0;
        let mut segment_infos: Vec<pb::SuffixArraySegmentInfo> = Vec::new();
        let mut seg_counter: usize = 0;
        let mut batch_idx: usize = 0;
        let build_start = std::time::Instant::now();

        // Helper closure to process a batch of segments
        async fn process_batch(
            pending: &mut Vec<(Vec<u8>, Vec<u64>, Vec<u64>)>,
            parallel_threads: usize,
            seg_counter: &mut usize,
            batch_idx: &mut usize,
            segment_infos: &mut Vec<pb::SuffixArraySegmentInfo>,
            index_store: &dyn IndexStore,
            token_width: u32,
        ) -> Result<()> {
            if pending.is_empty() {
                return Ok(());
            }

            let batch_size = parallel_threads.min(pending.len());
            let batch: Vec<(Vec<u8>, Vec<u64>, Vec<u64>)> = pending.drain(..batch_size).collect();
            let seg_offset = *seg_counter;
            *seg_counter += batch_size;

            info!(
                batch = *batch_idx,
                batch_size,
                seg_offset,
                remaining = pending.len(),
                "Building batch"
            );

            // Build this batch's SAs in parallel
            let built_batch = tokio::task::spawn_blocking(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(batch_size)
                    .build()
                    .map_err(|e| Error::io(format!("Failed to create rayon thread pool: {e}")))?;

                pool.install(|| {
                    batch
                        .into_par_iter()
                        .enumerate()
                        .map(|(i, (text, doc_offsets, row_ids))| {
                            let seg_idx = seg_offset + i;
                            let seg_start = std::time::Instant::now();
                            let corpus_bytes = text.len() as u64;
                            let pointer_width = compute_pointer_width(corpus_bytes);

                            let sa = build_suffix_array(&text);
                            let compact_sa = compact_suffix_array(&sa, pointer_width)?;
                            drop(sa);
                            release_memory_to_os();

                            let seg_elapsed = seg_start.elapsed();
                            info!(
                                segment = seg_idx,
                                bytes = corpus_bytes,
                                docs = doc_offsets.len(),
                                secs = seg_elapsed.as_secs_f64(),
                                mb_per_sec = corpus_bytes as f64 / seg_elapsed.as_secs_f64() / 1e6,
                                "Segment SA built"
                            );

                            // Build bloom filter for fast segment skipping
                            let bloom_bytes = if token_width > 1 {
                                BloomFilter::from_text_token_aligned(&text, token_width as usize)
                                    .to_bytes()
                            } else {
                                BloomFilter::from_text(&text).to_bytes()
                            };

                            Ok(BuiltSegment {
                                seg_idx,
                                text,
                                compact_sa,
                                pointer_width,
                                corpus_bytes,
                                doc_offsets,
                                row_ids,
                                bloom_bytes,
                            })
                        })
                        .collect::<Result<Vec<_>>>()
                })
            })
            .await
            .map_err(|e| Error::io(format!("SA builder task panicked: {e}")))??;

            for built in &built_batch {
                let info = write_segment(index_store, built).await?;
                segment_infos.push(info);
            }
            drop(built_batch);
            release_memory_to_os();

            *batch_idx += 1;
            Ok(())
        }

        // Read text from stream, accumulating segments. When we have enough
        // for a full batch, process it immediately.
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result?;
            let col = batch.column(value_col_idx);

            // Extract row IDs for non-null rows (same order as extract_bytes processes them)
            let row_id_col = batch[ROW_ID]
                .as_any()
                .downcast_ref::<arrow_array::UInt64Array>()
                .ok_or_else(|| {
                    Error::invalid_input("Training data stream missing '_rowid' column (UInt64)")
                })?;
            for i in 0..col.len() {
                if !col.is_null(i) {
                    current_row_ids.push(row_id_col.value(i));
                }
            }

            let docs = extract_bytes(
                col.as_ref(),
                &mut current_bytes,
                &mut current_doc_offsets,
                case_insensitive,
                separator_token_id,
            )?;
            total_documents += docs;

            // Split off complete segments as we accumulate data.
            while current_bytes.len() >= MAX_SEGMENT_BYTES {
                let segment = split_segment(
                    &mut current_bytes,
                    &mut current_doc_offsets,
                    &mut current_row_ids,
                    MAX_SEGMENT_BYTES,
                );
                total_corpus_bytes += segment.0.len() as u64;
                pending_segments.push(segment);

                // When we have a full batch, process it immediately
                if pending_segments.len() >= parallel_threads {
                    process_batch(
                        &mut pending_segments,
                        parallel_threads,
                        &mut seg_counter,
                        &mut batch_idx,
                        &mut segment_infos,
                        index_store,
                        token_width,
                    )
                    .await?;
                }
            }
        }

        // Push remaining bytes as the final segment
        if !current_bytes.is_empty() {
            total_corpus_bytes += current_bytes.len() as u64;
            pending_segments.push((current_bytes, current_doc_offsets, current_row_ids));
        }

        // Process any remaining segments
        while !pending_segments.is_empty() {
            process_batch(
                &mut pending_segments,
                parallel_threads,
                &mut seg_counter,
                &mut batch_idx,
                &mut segment_infos,
                index_store,
                token_width,
            )
            .await?;
        }

        let num_segments = segment_infos.len();

        // Handle empty dataset
        if num_segments == 0 {
            // Write a single empty segment
            let empty_segment = BuiltSegment {
                seg_idx: 0,
                text: Vec::new(),
                compact_sa: Vec::new(),
                pointer_width: 1,
                corpus_bytes: 0,
                doc_offsets: Vec::new(),
                row_ids: Vec::new(),
                bloom_bytes: Vec::new(),
            };
            let info = write_segment(index_store, &empty_segment).await?;
            segment_infos.push(info);
        }

        let num_segments = segment_infos.len();
        let build_elapsed = build_start.elapsed();
        info!(
            total_corpus_bytes,
            total_documents,
            num_segments,
            build_secs = build_elapsed.as_secs_f64(),
            parallel_threads,
            batches = batch_idx,
            "Streaming build complete"
        );

        // Build protobuf details.
        // Since streaming always uses multi-segment file naming (segment_N_*),
        // we always populate the segments array, even for single-segment indices.
        // This ensures load_multi is used at read time, which matches the
        // file naming convention.
        let details = pb::SuffixArrayIndexDetails {
            token_width,
            pointer_width: 0,
            total_tokens: 0,
            total_documents,
            corpus_bytes: total_corpus_bytes,
            tokenizer_name: None,
            vocab_size: None,
            separator_token_id: if token_width > 1 {
                let sep = separator_token_id.unwrap_or({
                    if token_width == 2 {
                        0xFFFF
                    } else {
                        0xFFFF_FFFF
                    }
                });
                Some(sep)
            } else {
                None
            },
            num_segments: num_segments as u32,
            segments: segment_infos,
            case_insensitive,
        };

        let index_details = prost_types::Any::from_msg(&details)?;

        Ok(CreatedIndex {
            index_details,
            index_version: SUFFIX_ARRAY_INDEX_VERSION,
            files: index_store.list_files_with_sizes().await?,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for SuffixArrayIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "SuffixArray"
    }

    fn provides_exact_answer(&self) -> bool {
        true
    }

    fn version(&self) -> u32 {
        SUFFIX_ARRAY_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        _index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &lance_core::cache::LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let details = pb::SuffixArrayIndexDetails::decode(&*index_details.value)?;
        let build_params = SuffixArrayBuildParams {
            case_insensitive: details.case_insensitive,
            separator_token_id: details.separator_token_id,
        };
        let case_insensitive = details.case_insensitive;
        let token_width = details.token_width.max(1) as u8;

        if !details.segments.is_empty() {
            // Multi-segment index (v1+) — also used for single-segment indices
            // built with streaming pipeline that always uses segment_N_ naming.
            info!(
                num_segments = details.segments.len(),
                case_insensitive, token_width, "Loading multi-segment suffix array index"
            );
            Ok(SuffixArrayIndex::load_multi(
                index_store,
                &details.segments,
                build_params,
                token_width,
                cache,
            )
            .await? as Arc<dyn ScalarIndex>)
        } else {
            // Single-segment index (v0 / backwards compat)
            let pointer_width = details.pointer_width as u8;
            let total_entries = details.total_tokens;
            Ok(SuffixArrayIndex::load(
                index_store,
                frag_reuse_index,
                cache,
                pointer_width,
                total_entries,
                build_params,
                token_width,
            )
            .await? as Arc<dyn ScalarIndex>)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{RecordBatch, StringArray, UInt64Array};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use lance_core::Result;
    use lance_core::cache::LanceCache;
    use lance_core::deepsize::DeepSizeOf;
    use lance_core::utils::tempfile::TempDir;
    use lance_io::object_store::ObjectStore;

    use super::{SuffixArrayBuildParams, SuffixArrayIndexPlugin, split_segment};
    use crate::metrics::NoOpMetricsCollector;
    use crate::progress::NoopIndexBuildProgress;
    use crate::scalar::lance_format::LanceIndexStore;
    use crate::scalar::registry::{BasicTrainer, ScalarIndexPlugin, VALUE_COLUMN_NAME};
    use crate::scalar::suffix_array::builder::{
        build_suffix_array, compact_suffix_array, compute_pointer_width,
    };
    use crate::scalar::suffix_array::query;
    use crate::scalar::suffix_array::{SuffixArrayIndex, SuffixArrayQuery};
    use crate::scalar::{
        CreatedIndex, IndexFile, IndexReader, IndexStore, IndexWriter, ScalarIndex, SearchResult,
    };

    /// One batch of (`value`, `_rowid`) training data; row ids start at `first_row_id`.
    fn text_stream(texts: &[&str], first_row_id: u64) -> SendableRecordBatchStream {
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
            Field::new("_rowid", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(texts.to_vec())),
                Arc::new(UInt64Array::from_iter_values(
                    first_row_id..first_row_id + texts.len() as u64,
                )),
            ],
        )
        .unwrap();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::once(std::future::ready(Ok(batch))),
        ))
    }

    fn local_store(dir: &TempDir) -> Arc<LanceIndexStore> {
        Arc::new(LanceIndexStore::new(
            Arc::new(ObjectStore::local()),
            dir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ))
    }

    async fn train(texts: &[&str], params: &str, store: &dyn IndexStore) -> CreatedIndex {
        let plugin = SuffixArrayIndexPlugin;
        let field = Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false);
        let request = plugin.new_training_request(params, &field).unwrap();
        plugin
            .train_index(
                text_stream(texts, 0),
                store,
                request,
                None,
                Arc::new(NoopIndexBuildProgress),
            )
            .await
            .unwrap()
    }

    async fn load(created: &CreatedIndex, store: Arc<dyn IndexStore>) -> Arc<dyn ScalarIndex> {
        SuffixArrayIndexPlugin
            .load_index(
                store,
                &created.index_details,
                created.index_version,
                None,
                &LanceCache::no_cache(),
            )
            .await
            .unwrap()
    }

    fn as_suffix_array(index: &Arc<dyn ScalarIndex>) -> &SuffixArrayIndex {
        index.as_any().downcast_ref::<SuffixArrayIndex>().unwrap()
    }

    /// Row addresses returned by `ScalarIndex::search` for `pattern`.
    async fn search_rows(index: &dyn ScalarIndex, pattern: &[u8]) -> Vec<u64> {
        let query = SuffixArrayQuery::Search {
            query_bytes: pattern.to_vec(),
            max_results: 100,
        };
        match index.search(&query, &NoOpMetricsCollector).await.unwrap() {
            SearchResult::Exact(rows) => rows
                .true_rows()
                .row_addrs()
                .unwrap()
                .map(u64::from)
                .collect(),
            other => panic!("expected an exact result, got {other:?}"),
        }
    }

    async fn file_names(store: &dyn IndexStore) -> Vec<String> {
        let mut names: Vec<String> = store
            .list_files_with_sizes()
            .await
            .unwrap()
            .into_iter()
            .map(|f| f.path)
            .collect();
        names.sort();
        names
    }

    /// Store that serves reads from `inner` and rejects every write, counting
    /// the attempts — a read-only object store as seen by the index.
    #[derive(Debug)]
    struct ReadOnlyStore {
        inner: Arc<dyn IndexStore>,
        write_attempts: Arc<AtomicUsize>,
    }

    impl DeepSizeOf for ReadOnlyStore {
        fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
            0
        }
    }

    impl ReadOnlyStore {
        fn reject_write(&self, name: &str) -> lance_core::Error {
            self.write_attempts.fetch_add(1, Ordering::SeqCst);
            lance_core::Error::not_supported(format!("read-only store: refused to write {name}"))
        }
    }

    #[async_trait]
    impl IndexStore for ReadOnlyStore {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn clone_arc(&self) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.clone(),
                write_attempts: self.write_attempts.clone(),
            })
        }

        fn io_parallelism(&self) -> usize {
            self.inner.io_parallelism()
        }

        fn with_io_priority(&self, io_priority: u64) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.with_io_priority(io_priority),
                write_attempts: self.write_attempts.clone(),
            })
        }

        fn with_io_buffer_size(&self, bytes: u64) -> Arc<dyn IndexStore> {
            Arc::new(Self {
                inner: self.inner.with_io_buffer_size(bytes),
                write_attempts: self.write_attempts.clone(),
            })
        }

        async fn new_index_file(
            &self,
            name: &str,
            _schema: Arc<Schema>,
        ) -> Result<Box<dyn IndexWriter>> {
            Err(self.reject_write(name))
        }

        async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
            self.inner.open_index_file(name).await
        }

        async fn copy_index_file(
            &self,
            name: &str,
            _dest_store: &dyn IndexStore,
        ) -> Result<IndexFile> {
            Err(self.reject_write(name))
        }

        async fn rename_index_file(&self, name: &str, _new_name: &str) -> Result<IndexFile> {
            Err(self.reject_write(name))
        }

        async fn delete_index_file(&self, name: &str) -> Result<()> {
            Err(self.reject_write(name))
        }

        async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>> {
            self.inner.list_files_with_sizes().await
        }

        async fn write_raw_file(&self, name: &str, _data: &[u8]) -> Result<()> {
            Err(self.reject_write(name))
        }

        async fn read_raw_range(&self, name: &str, range: std::ops::Range<usize>) -> Result<Bytes> {
            self.inner.read_raw_range(name, range).await
        }

        async fn raw_file_size(&self, name: &str) -> Result<usize> {
            self.inner.raw_file_size(name).await
        }
    }

    #[test]
    fn test_sa_construction_small() {
        let data = b"abcabc";
        let sa = build_suffix_array(data);
        assert_eq!(sa.len(), data.len());

        for i in 1..sa.len() {
            let s1 = &data[sa[i - 1] as usize..];
            let s2 = &data[sa[i] as usize..];
            assert!(s1 < s2, "SA not sorted at position {i}: {s1:?} >= {s2:?}");
        }
    }

    #[test]
    fn test_binary_search_correctness() {
        let data = b"mississippi";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"issi"), 2);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"ss"), 2);
        assert_eq!(
            query::count(data, &compact, ptr_width, n, b"mississippi"),
            1
        );
        assert_eq!(query::count(data, &compact, ptr_width, n, b"z"), 0);
    }

    #[test]
    fn test_count_returns_correct_counts() {
        let data = b"the cat sat on the mat";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"the"), 2);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"at"), 3);
        assert_eq!(query::count(data, &compact, ptr_width, n, b" "), 5);
    }

    #[test]
    fn test_empty_data() {
        let data = b"";
        let sa = build_suffix_array(data);
        assert!(sa.is_empty());

        let ptr_width = compute_pointer_width(0);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        assert_eq!(query::count(data, &compact, ptr_width, 0, b"anything"), 0);
    }

    #[test]
    fn test_single_character_data() {
        let data = b"x";
        let sa = build_suffix_array(data);
        let ptr_width = compute_pointer_width(data.len() as u64);
        let compact = compact_suffix_array(&sa, ptr_width).unwrap();
        let n = sa.len() as u64;

        assert_eq!(query::count(data, &compact, ptr_width, n, b"x"), 1);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"y"), 0);
        assert_eq!(query::count(data, &compact, ptr_width, n, b"xx"), 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_train_load_query_roundtrip() {
        let tmpdir = TempDir::default();
        let store = local_store(&tmpdir);
        let created = train(
            &[
                "the quick brown fox",
                "jumps over the lazy dog",
                "the cat sat on the mat",
            ],
            "",
            store.as_ref(),
        )
        .await;

        // Both large blobs are raw files written at build time (block-cached
        // reads); no tokenized Lance file is written.
        let names = file_names(store.as_ref()).await;
        assert!(
            names.contains(&"segment_0_tokenized.raw".to_string()),
            "{names:?}"
        );
        assert!(
            names.contains(&"segment_0_suffix_array.raw".to_string()),
            "{names:?}"
        );
        assert!(
            !names.iter().any(|n| n.ends_with("tokenized.bin")),
            "{names:?}"
        );

        let index = load(&created, store.clone()).await;
        let sa = as_suffix_array(&index);
        assert_eq!(sa.total_count(b"the").await.unwrap(), 4);
        assert_eq!(sa.total_count(b"xyz").await.unwrap(), 0);

        // `search` returns the row addresses of the matching documents.
        assert_eq!(search_rows(index.as_ref(), b"fox").await, vec![0]);
        assert_eq!(search_rows(index.as_ref(), b"the").await, vec![0, 1, 2]);
        assert!(search_rows(index.as_ref(), b"xyz").await.is_empty());
    }

    #[test_log::test(tokio::test)]
    async fn test_empty_index_roundtrip() {
        let tmpdir = TempDir::default();
        let store = local_store(&tmpdir);
        let created = train(&[], "", store.as_ref()).await;

        // An empty build still writes segment 0 under the segmented names
        // that the loader expects.
        let names = file_names(store.as_ref()).await;
        for expected in [
            "segment_0_tokenized.raw",
            "segment_0_suffix_array.raw",
            "segment_0_doc_offsets.bin",
            "segment_0_row_ids.bin",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing {expected}: {names:?}"
            );
        }

        let index = load(&created, store.clone()).await;
        let sa = as_suffix_array(&index);
        assert_eq!(sa.total_count(b"anything").await.unwrap(), 0);
        assert!(search_rows(index.as_ref(), b"anything").await.is_empty());
        assert_eq!(sa.compute_prob(b"a", b"b").await.unwrap().cont_cnt, 0);
    }

    #[test_log::test(tokio::test)]
    async fn test_query_legacy_index_never_writes() {
        // Indices built before the raw format store the tokenized text only as
        // a Lance `.bin` file. Querying them must not write anything to the
        // store (e.g. a raw sidecar): reads must work on read-only stores.
        let tmpdir = TempDir::default();
        let store = local_store(&tmpdir);
        let created = train(&["the quick brown fox", "the lazy dog"], "", store.as_ref()).await;

        let raw_name = "segment_0_tokenized.raw";
        let size = store.raw_file_size(raw_name).await.unwrap();
        let text = store.read_raw_range(raw_name, 0..size).await.unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::LargeBinary,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::LargeBinaryArray::from(vec![
                text.as_ref(),
            ]))],
        )
        .unwrap();
        let mut writer = store
            .new_index_file("segment_0_tokenized.bin", schema)
            .await
            .unwrap();
        writer.write_record_batch(batch).await.unwrap();
        writer.finish().await.unwrap();
        store.delete_index_file(raw_name).await.unwrap();
        let files_before = file_names(store.as_ref()).await;

        let write_attempts = Arc::new(AtomicUsize::new(0));
        let read_only = Arc::new(ReadOnlyStore {
            inner: store.clone(),
            write_attempts: write_attempts.clone(),
        });
        let index = load(&created, read_only).await;
        let sa = as_suffix_array(&index);
        assert_eq!(sa.total_count(b"the").await.unwrap(), 2);
        assert_eq!(search_rows(index.as_ref(), b"fox").await, vec![0]);

        // Let any task spawned by the query path run before checking.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(write_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(file_names(store.as_ref()).await, files_before);
    }

    #[test_log::test(tokio::test)]
    async fn test_update_rebuilds_with_original_params() {
        let old_texts = ["The quick brown Fox", "a lazy dog"];
        let tmpdir = TempDir::default();
        let store = local_store(&tmpdir);
        let created = train(&old_texts, r#"{"case_insensitive": true}"#, store.as_ref()).await;
        let index = load(&created, store.clone()).await;

        // The derived params keep the build options, so a rebuild through
        // `build_scalar_index` stays case-insensitive.
        let derived = index.derive_index_params().unwrap();
        assert_eq!(derived.index_type, "suffixarray");
        let params: SuffixArrayBuildParams =
            serde_json::from_str(derived.params.as_deref().unwrap()).unwrap();
        assert!(params.case_insensitive);

        // `update` receives every live row (old and appended) with `_rowid`,
        // as requested by `update_criteria`, and rebuilds the index from it.
        let criteria = index.update_criteria();
        assert!(criteria.requires_old_data);
        assert!(criteria.data_criteria.needs_row_ids);
        let all_texts = [old_texts[0], old_texts[1], "ANOTHER FOX, a new one"];
        let new_dir = TempDir::default();
        let new_store = local_store(&new_dir);
        let updated = index
            .update(text_stream(&all_texts, 0), new_store.as_ref(), None)
            .await
            .unwrap();
        let updated = load(&updated, new_store.clone()).await;
        let sa = as_suffix_array(&updated);
        assert!(sa.case_insensitive());
        // Queries are normalized by the index itself, whatever their case.
        assert_eq!(sa.total_count(b"FOX").await.unwrap(), 2);
        assert_eq!(sa.total_count(b"fox").await.unwrap(), 2);
        assert_eq!(
            sa.compute_prob(b"A NEW ", b"ONE").await.unwrap().cont_cnt,
            1
        );
        assert_eq!(search_rows(updated.as_ref(), b"New").await, vec![2]);
    }

    #[test_log::test(tokio::test)]
    async fn test_multi_segment_roundtrip() {
        // Verify that a multi-segment index produces the same counts as a
        // single-segment index over the same corpus.
        use crate::scalar::suffix_array::SuffixArraySegment;

        // Build a corpus split across two segments
        let text1 = b"the cat sat on the mat";
        let text2 = b"the dog ran in the park";

        let sa1 = build_suffix_array(text1);
        let pw1 = compute_pointer_width(text1.len() as u64);
        let csa1 = compact_suffix_array(&sa1, pw1).unwrap();

        let sa2 = build_suffix_array(text2);
        let pw2 = compute_pointer_width(text2.len() as u64);
        let csa2 = compact_suffix_array(&sa2, pw2).unwrap();

        let index = SuffixArrayIndex::from_segments(vec![
            SuffixArraySegment {
                tokenized: Some(Bytes::from(text1.to_vec())),
                tokenized_cache: None,
                suffix_array: Some(Bytes::from(csa1)),
                sa_cache: None,
                pointer_width: pw1 as u8,
                total_entries: text1.len() as u64,
                doc_offsets: vec![text1.len() as u64],
                row_ids: vec![0],
            },
            SuffixArraySegment {
                tokenized: Some(Bytes::from(text2.to_vec())),
                tokenized_cache: None,
                suffix_array: Some(Bytes::from(csa2)),
                sa_cache: None,
                pointer_width: pw2 as u8,
                total_entries: text2.len() as u64,
                doc_offsets: vec![text2.len() as u64],
                row_ids: vec![1],
            },
        ]);

        // "the" appears 2 times in text1 + 2 times in text2 = 4
        assert_eq!(index.total_count(b"the").await.unwrap(), 4);
        // "cat" appears once in text1 only
        assert_eq!(index.total_count(b"cat").await.unwrap(), 1);
        // "dog" appears once in text2 only
        assert_eq!(index.total_count(b"dog").await.unwrap(), 1);
        // "xyz" appears nowhere
        assert_eq!(index.total_count(b"xyz").await.unwrap(), 0);

        // Test prob across segments
        let prob_result = index.compute_prob(b"the ", b"cat").await.unwrap();
        // "the " appears: text1 has "the cat" and "the mat" = 2, text2 has "the dog" and "the park" = 2, total = 4
        assert_eq!(prob_result.prompt_cnt, 4);
        // "the cat" appears once (text1 only)
        assert_eq!(prob_result.cont_cnt, 1);
        assert!((prob_result.prob - 0.25).abs() < 1e-10);

        // Test ntd across segments
        let ntd_result = index.compute_ntd(b"the ", None).await.unwrap();
        assert_eq!(ntd_result.prompt_cnt, 4);
        // Should have 4 distinct next bytes: 'c', 'm', 'd', 'p'
        assert_eq!(ntd_result.distribution.len(), 4);

        // Test infgram_prob across segments
        let igp = index.compute_infgram_prob(b"the ", b"cat").await.unwrap();
        assert_eq!(igp.prob_result.prompt_cnt, 4);
        assert_eq!(igp.prob_result.cont_cnt, 1);
        assert_eq!(igp.effective_suffix_len, 4);

        // `search` returns the matching documents' row addresses across segments.
        assert_eq!(search_rows(&index, b"the").await, vec![0, 1]);
        assert_eq!(search_rows(&index, b"park").await, vec![1]);
    }

    #[test]
    fn test_split_segment_keeps_documents_whole() {
        // Documents of 4, 6, 3 and 9 bytes (each ends with its separator).
        let docs: [&[u8]; 4] = [b"abc\0", b"defgh\0", b"ij\0", b"klmnopqr\0"];
        let mut bytes: Vec<u8> = docs.concat();
        let mut doc_offsets = vec![4, 10, 13, 22];
        let mut row_ids = vec![100, 101, 102, 103];

        // The 8-byte limit falls inside "defgh": the segment ends after "abc".
        let (seg, offsets, rows) = split_segment(&mut bytes, &mut doc_offsets, &mut row_ids, 8);
        assert_eq!(seg, b"abc\0");
        assert_eq!((offsets, rows), (vec![4], vec![100]));

        // A limit at a document end includes that document.
        let (seg, offsets, rows) = split_segment(&mut bytes, &mut doc_offsets, &mut row_ids, 9);
        assert_eq!(seg, b"defgh\0ij\0");
        assert_eq!((offsets, rows), (vec![6, 9], vec![101, 102]));

        // A document larger than the limit gets a segment of its own.
        let (seg, offsets, rows) = split_segment(&mut bytes, &mut doc_offsets, &mut row_ids, 4);
        assert_eq!(seg, b"klmnopqr\0");
        assert_eq!((offsets, rows), (vec![9], vec![103]));
        assert!(bytes.is_empty() && doc_offsets.is_empty() && row_ids.is_empty());
    }

    #[test]
    fn test_plugin_rejects_invalid_field_type() {
        use crate::scalar::registry::BasicTrainer;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        let field = Field::new("value", DataType::Int32, false);
        let result = plugin.new_training_request("", &field);
        assert!(result.is_err(), "Int32 field should be rejected");
    }

    #[test]
    fn test_plugin_accepts_valid_field_types() {
        use crate::scalar::registry::BasicTrainer;
        use arrow_schema::{DataType, Field};

        let plugin = super::SuffixArrayIndexPlugin;
        for dt in &[
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
        ] {
            let field = Field::new("value", dt.clone(), false);
            assert!(
                plugin.new_training_request("", &field).is_ok(),
                "Should accept {:?}",
                dt
            );
        }
    }
}
