//! Block-based KV cache with paged-attention support.
//!
//! This module implements the vLLM-style paged attention algorithm:
//!
//! * The KV cache is divided into fixed-size **blocks** (pages).  Each block
//!   holds `block_size` token slots for every (layer, head).
//! * A per-sequence **block table** maps logical block indices to physical
//!   block IDs allocated from a shared pool.
//! * A free list (LRU ordering) tracks available blocks so that memory can be
//!   reclaimed from finished sequences and reused.
//!
//! The public API used by the attention kernel is:
//! ```
//! slot_id = block_table[logical_block] * block_size + (position % block_size)
//! ```
//! The physical KV tensors are indexed by `slot_id` instead of the raw
//! sequence position, breaking the requirement for contiguous per-sequence
//! memory.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Paged attention: BlockPool
// ---------------------------------------------------------------------------

/// Configuration for the paged KV cache.
#[derive(Debug, Clone)]
pub struct PagedCacheConfig {
    /// Number of token slots per block (vLLM default: 16).
    pub block_size: usize,
    /// Total number of physical blocks in the pool.
    pub num_blocks: usize,
    /// Number of KV heads per attention layer.
    pub num_kv_heads: usize,
    /// Dimension of each KV head.
    pub head_dim: usize,
    /// Number of full-attention layers whose KV cache is managed here.
    pub num_layers: usize,
}

impl PagedCacheConfig {
    /// Compute the number of physical blocks that fit in `memory_fraction` of
    /// the given total memory (in bytes).
    pub fn from_memory_fraction(
        memory_bytes: usize,
        memory_fraction: f64,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_layers: usize,
        bytes_per_element: usize, // 2 for bf16/f16, 4 for f32
    ) -> Self {
        let available = (memory_bytes as f64 * memory_fraction) as usize;
        // Each block holds block_size tokens × num_kv_heads × head_dim × 2 (K+V) per layer.
        let bytes_per_block =
            block_size * num_kv_heads * head_dim * 2 * num_layers * bytes_per_element;
        let num_blocks = available.checked_div(bytes_per_block).unwrap_or(0);
        Self {
            block_size,
            num_blocks,
            num_kv_heads,
            head_dim,
            num_layers,
        }
    }
}

/// A physical block in the KV cache pool.
#[derive(Debug)]
struct Block {
    /// Reference count: number of sequences currently using this block.
    ref_cnt: u32,
}

/// Shared physical block pool — analogous to vLLM's `BlockPool`.
///
/// Blocks are allocated from a free list (FIFO/LRU head) and returned to the
/// tail when freed so that recently freed blocks (which may still be warm in
/// cache) are evicted last.
pub struct BlockPool {
    blocks: Vec<Block>,
    /// Indices of free blocks in LRU order (front = oldest / evict first).
    free_list: VecDeque<u32>,
    pub block_size: usize,
}

impl BlockPool {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let mut blocks = Vec::with_capacity(num_blocks);
        let mut free_list = VecDeque::with_capacity(num_blocks);
        for id in 0..num_blocks as u32 {
            blocks.push(Block { ref_cnt: 0 });
            free_list.push_back(id);
        }
        Self {
            blocks,
            free_list,
            block_size,
        }
    }

    /// Number of blocks currently available for allocation.
    #[allow(dead_code)]
    pub fn num_free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Allocate `n` blocks.  Returns `None` if there are not enough free blocks.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.free_list.len() < n {
            return None;
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free_list.pop_front().unwrap();
            self.blocks[id as usize].ref_cnt = 1;
            ids.push(id);
        }
        Some(ids)
    }

    /// Allocate a single block.
    pub fn allocate_one(&mut self) -> Option<u32> {
        self.allocate(1).map(|v| v[0])
    }

    /// Decrement reference counts and return freed blocks to the tail of the
    /// free list (MRU position — evicted last).
    pub fn free_blocks(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            let block = &mut self.blocks[id as usize];
            if block.ref_cnt > 0 {
                block.ref_cnt -= 1;
            }
            if block.ref_cnt == 0 {
                self.free_list.push_back(id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Paged attention: per-sequence BlockTable
// ---------------------------------------------------------------------------

/// Maps logical block indices → physical block IDs for a single sequence.
///
/// The block table grows on demand as the sequence generates more tokens.
/// The slot for token at position `pos` is:
/// ```
/// slot = block_table[pos / block_size] * block_size + (pos % block_size)
/// ```
#[derive(Debug, Default)]
pub struct BlockTable {
    /// Physical block IDs, indexed by logical block number.
    physical_blocks: Vec<u32>,
    /// Total number of tokens whose slots have been reserved.
    num_tokens: usize,
    block_size: usize,
}

impl BlockTable {
    pub fn new(block_size: usize) -> Self {
        Self {
            physical_blocks: Vec::new(),
            num_tokens: 0,
            block_size,
        }
    }

    /// Number of tokens already mapped in the block table.
    #[allow(dead_code)]
    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    /// Number of physical blocks currently allocated.
    #[allow(dead_code)]
    pub fn num_blocks(&self) -> usize {
        self.physical_blocks.len()
    }

    /// Resolve a token position to a flat cache slot ID.
    /// Returns `None` if the position hasn't been allocated yet.
    pub fn slot_for(&self, position: usize) -> Option<u32> {
        let block_idx = position / self.block_size;
        if block_idx >= self.physical_blocks.len() {
            return None;
        }
        let block_offset = (position % self.block_size) as u32;
        Some(self.physical_blocks[block_idx] * self.block_size as u32 + block_offset)
    }

    /// Ensure that `position + 1` token slots are allocated.  Allocates new
    /// blocks from `pool` as needed.  Returns `false` if the pool is full.
    pub fn ensure_allocated(&mut self, position: usize, pool: &mut BlockPool) -> bool {
        let needed_blocks = position / self.block_size + 1;
        while self.physical_blocks.len() < needed_blocks {
            match pool.allocate_one() {
                Some(id) => self.physical_blocks.push(id),
                None => return false,
            }
        }
        if position + 1 > self.num_tokens {
            self.num_tokens = position + 1;
        }
        true
    }

    /// Return all physical blocks to the pool.
    pub fn free_all(&mut self, pool: &mut BlockPool) {
        pool.free_blocks(&self.physical_blocks);
        self.physical_blocks.clear();
        self.num_tokens = 0;
    }

    /// Raw slice of physical block IDs (for passing to the attention kernel).
    #[allow(dead_code)]
    pub fn physical_blocks(&self) -> &[u32] {
        &self.physical_blocks
    }
}

// ---------------------------------------------------------------------------
// Paged KV store: the physical K/V tensors indexed by slot_id
// ---------------------------------------------------------------------------

use candle_core::{DType, Device, Tensor};

/// Physical KV cache storage for all full-attention layers.
///
/// Memory layout (per layer):
///   key_cache[layer]:   [num_blocks * block_size, num_kv_heads, head_dim]
///   value_cache[layer]: [num_blocks * block_size, num_kv_heads, head_dim]
///
/// Access: `key_cache[layer].narrow(0, slot_id, 1)` gives the key for one
/// token slot.
pub struct PagedKvStore {
    /// Key caches, one tensor per full-attention layer.
    pub key_caches: Vec<Tensor>,
    /// Value caches, one tensor per full-attention layer.
    pub value_caches: Vec<Tensor>,
    #[allow(dead_code)]
    pub cfg: PagedCacheConfig,
}

impl PagedKvStore {
    /// Allocate zeroed KV cache tensors.
    pub fn new(cfg: PagedCacheConfig, dtype: DType, device: &Device) -> candle_core::Result<Self> {
        let total_slots = cfg.num_blocks * cfg.block_size;
        let mut key_caches = Vec::with_capacity(cfg.num_layers);
        let mut value_caches = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            key_caches.push(Tensor::zeros(
                (total_slots, cfg.num_kv_heads, cfg.head_dim),
                dtype,
                device,
            )?);
            value_caches.push(Tensor::zeros(
                (total_slots, cfg.num_kv_heads, cfg.head_dim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            key_caches,
            value_caches,
            cfg,
        })
    }

    /// Gather key and value tensors for all slots in `slot_ids`.
    ///
    /// Returns `(k, v)` each of shape `[seq_len, num_kv_heads, head_dim]`.
    pub fn gather_slots(
        &self,
        layer_idx: usize,
        slot_ids: &[u32],
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let device = self.key_caches[layer_idx].device();
        let idx = Tensor::new(slot_ids, device)?;
        let k = self.key_caches[layer_idx].index_select(&idx, 0)?;
        let v = self.value_caches[layer_idx].index_select(&idx, 0)?;
        Ok((k, v))
    }

    /// Zero out the given slots in all layers so that writes don't accumulate
    /// stale values from a previous sequence.
    ///
    /// This must be called once per new sequence (at `seqlen_offset == 0`)
    /// after blocks have been allocated for the prompt.  Without it, reused
    /// physical blocks carry K/V data from the previous occupant, and
    /// `scatter_add_set` writes in `forward_returning_kv_paged` would add the
    /// new values on top of the old ones instead of replacing them.
    ///
    /// Implementation: uses `scatter_set` (direct assign) with a zeros source
    /// tensor, which requires only **2 GPU kernels per layer** (vs the previous
    /// 6-kernel read-negate-index_add approach).  A single expanded index
    /// tensor is built once and reused across all layers.
    pub fn zero_slots(&mut self, slot_ids: &[u32]) -> candle_core::Result<()> {
        if slot_ids.is_empty() {
            return Ok(());
        }
        let n_slots = slot_ids.len();
        let num_kv_heads = self.cfg.num_kv_heads;
        let head_dim = self.cfg.head_dim;
        // All layers share the same device; build the index and zeros tensors once.
        let device = self.key_caches[0].device().clone();
        let dtype = self.key_caches[0].dtype();

        // Build expanded index: [n_slots, num_kv_heads, head_dim]
        // indexes[i, j, k] = slot_ids[i] for all j, k.
        // scatter_set requires indexes.shape() == source.shape().
        let idx_1d = Tensor::new(slot_ids, &device)?.to_dtype(candle_core::DType::U32)?;
        let idx_expanded = idx_1d
            .reshape((n_slots, 1, 1))?
            .expand((n_slots, num_kv_heads, head_dim))?
            .contiguous()?;

        // Zeros source to scatter into the slots.
        let zeros = Tensor::zeros((n_slots, num_kv_heads, head_dim), dtype, &device)?;

        for layer_idx in 0..self.key_caches.len() {
            // Direct scatter-assign: cache[slot_ids[i], j, k] = 0
            // This is 2 GPU kernels per layer instead of the previous 6
            // (gather + neg + index_add for each of K and V).
            self.key_caches[layer_idx].scatter_set(&idx_expanded, &zeros, 0)?;
            self.value_caches[layer_idx].scatter_set(&idx_expanded, &zeros, 0)?;
        }
        Ok(())
    }
}
