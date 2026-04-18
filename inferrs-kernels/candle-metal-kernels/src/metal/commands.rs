use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 64;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

struct EntryState {
    current: CommandBuffer,
    /// Persistent compute encoder kept open across multiple dispatches.
    ///
    /// Metal allows a single `MTLComputeCommandEncoder` to encode arbitrarily
    /// many `dispatchThreadgroups` calls before `endEncoding`.  By keeping one
    /// encoder alive across all kernel dispatches that share the same command
    /// buffer we eliminate the `endEncoding()` + `computeCommandEncoder()` pair
    /// that was previously called for every single kernel — the dominant
    /// per-dispatch CPU overhead on Apple Silicon.
    ///
    /// The encoder is ended (and `None`-d out) only when the command buffer is
    /// committed via `commit_swap_locked` or `flush_and_wait`.
    active_encoder: Option<ComputeCommandEncoder>,
    in_flight: Vec<CommandBuffer>,
}

impl EntryState {
    /// Return a reference to the persistent encoder, creating it if not yet open.
    /// Return a reference to the persistent encoder, creating it if not yet open.
    ///
    /// The stored encoder is non-owning (owned=false) so it does NOT call
    /// `endEncoding` on drop.  `end_encoder()` calls `end_encoding()` explicitly.
    fn get_or_create_encoder(&mut self) -> Result<&ComputeCommandEncoder, MetalKernelError> {
        if self.active_encoder.is_none() {
            self.active_encoder = Some(self.current.compute_command_encoder_persistent());
        }
        Ok(self.active_encoder.as_ref().unwrap())
    }

    /// End the active encoder (if any) before committing the command buffer.
    fn end_encoder(&mut self) {
        if let Some(enc) = self.active_encoder.take() {
            enc.end_encoding();
        }
    }
}

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The `state` mutex guards the current buffer and the in-flight list for coherent updates.
/// `compute_count` and `semaphore` remain accessible without locking for selection/coordination.
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
}

pub struct Commands {
    /// Maintains a pool of command buffers, allowing
    /// the pool to balance load across multiple buffers and improve GPU utilization.
    /// Can be shared across threads safely.
    pool: Vec<Arc<CommandBufferEntry>>,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
        })
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(command_queue, Arc::clone(&semaphore))?;

        // SAFETY: Commands is used from a single device thread; the unsafe Send/Sync
        // impls on Commands ensure correct access. The Arc here is for interior sharing
        // within the pool, not cross-thread transfer.
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: cb,
                active_encoder: None,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        let flush = self.maybe_flush_entry(&entry)?;

        // Return a non-owning alias of the persistent encoder.
        // The underlying MTLComputeCommandEncoder stays open; `endEncoding` is
        // only called at commit time (in `commit_swap_locked`).
        //
        // Signal Available immediately so subsequent calls to `select_entry`
        // can reuse this pool entry without waiting.  The entry's command buffer
        // is still being built; only `compute_count` tracks that.
        //
        // SAFETY: `Commands` is accessed exclusively from the single device thread
        // (enforced by `unsafe impl Send/Sync for Commands`).  No other thread can
        // acquire or use this pool entry concurrently, so signalling Available here
        // is safe: the persistent encoder is accessed only from this thread.
        entry.semaphore.set_status(CommandStatus::Available);
        let mut state = entry.state.lock()?;
        let encoder = state.get_or_create_encoder()?.non_owning_alias();
        Ok((flush, encoder))
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        let entry = self.select_entry()?;
        let flush = self.maybe_flush_entry(&entry)?;

        // Blit encoders cannot coexist with an active compute encoder.
        // End the compute encoder first, then create a blit encoder.
        // The blit encoder manages its own semaphore via Drop.
        let mut state = entry.state.lock()?;
        state.end_encoder();
        let blit = state.current.blit_command_encoder();
        // Don't signal Available here — the blit encoder's Drop will do it.
        Ok((flush, blit))
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    // Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the most work and wait for it
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Command buffer pool is empty".to_string(),
            ))?;

        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        Ok(entry)
    }

    /// Increments the dispatch counter and flushes the command buffer if the limit is reached.
    /// Returns `true` when a flush occurred.
    fn maybe_flush_entry(&self, entry: &Arc<CommandBufferEntry>) -> Result<bool, MetalKernelError> {
        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;
        if flush {
            let mut state = entry.state.lock()?;
            self.commit_swap_locked(entry, &mut state, 1)?;
        }
        Ok(flush)
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                // Ensure no active encoder is still encoding on this entry.
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(entry, &mut state, 0)?;
                }

                // Drain `in_flight` into a local vec to wait without holding the lock.
                // Replaces `state.in_flight` with an empty vec and returns its previous contents.
                std::mem::take(&mut state.in_flight)
            };

            for cb in to_wait {
                Self::ensure_completed(&cb)?;
            }
        }

        Ok(())
    }

    /// Flushes all buffers without waiting for completion.
    /// Commits any pending work and moves current buffers to in-flight.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = entry.state.lock()?;

            if entry.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(entry, &mut state, 0)?;
            }
        }

        Ok(())
    }

    /// Commit the current command buffer, swap in a fresh one, push the old into `in_flight`,
    /// and reset `compute_count` to `reset_to`.
    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        // End the persistent encoder before committing.
        state.end_encoder();
        state.current.commit();
        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        entry.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}
