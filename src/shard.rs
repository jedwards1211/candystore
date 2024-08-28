use anyhow::{bail, ensure};
use bytemuck::{bytes_of_mut, Pod, Zeroable};
use parking_lot::{Mutex, RwLock};
use simd_itertools::PositionSimd;
use std::{
    ffi::c_void,
    fs::{File, OpenOptions},
    io::Read,
    ops::Range,
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::PathBuf,
    ptr::null_mut,
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::Instant,
};

use crate::{
    hashing::{PartedHash, INVALID_SIG},
    stats::InternalStats,
    store::InternalConfig,
};
use crate::{CandyError, Result};

//
// these numbers were chosen according to the simulation, as they allow for 90% utilization of the shard with
// virtually zero chance of in-row collisions and "smallish" shard size: shards start at 384KB and
// can hold 32K entries, and since we're limited at 4GB file sizes, we can key-value pairs of up to 128KB
// (keys and values are limited to 64KB each anyway)
//
// other good combinations are 32/512, 32/1024, 64/256, 64/1024, 128/512, 256/256
//
pub(crate) const NUM_ROWS: usize = 64;
pub(crate) const ROW_WIDTH: usize = 512;

#[repr(C)]
struct ShardRow {
    signatures: [u32; ROW_WIDTH],
    offsets_and_sizes: [u64; ROW_WIDTH], // | key_size: 16 | val_size: 16 | file_offset: 32 |
}

impl ShardRow {
    #[inline]
    fn lookup(&self, sig: u32, start_idx: &mut usize) -> Option<usize> {
        if let Some(rel_idx) = self.signatures[*start_idx..].iter().position_simd(sig) {
            let abs_idx = rel_idx + *start_idx;
            *start_idx = abs_idx + 1;
            Some(abs_idx)
        } else {
            None
        }
    }
}

#[test]
fn test_row_lookup() -> Result<()> {
    let mut row = ShardRow {
        signatures: [0; ROW_WIDTH],
        offsets_and_sizes: [0; ROW_WIDTH],
    };

    row.signatures[7] = 123;
    row.signatures[8] = 123;
    row.signatures[9] = 123;
    row.signatures[90] = 123;
    row.signatures[ROW_WIDTH - 1] = 999;

    let mut start = 0;
    assert_eq!(row.lookup(123, &mut start), Some(7));
    assert_eq!(start, 8);
    assert_eq!(row.lookup(123, &mut start), Some(8));
    assert_eq!(start, 9);
    assert_eq!(row.lookup(123, &mut start), Some(9));
    assert_eq!(start, 10);
    assert_eq!(row.lookup(123, &mut start), Some(90));
    assert_eq!(start, 91);
    assert_eq!(row.lookup(123, &mut start), None);
    assert_eq!(start, 91);

    start = 0;
    assert_eq!(row.lookup(0, &mut start), Some(0));
    assert_eq!(start, 1);

    start = 0;
    assert_eq!(row.lookup(999, &mut start), Some(ROW_WIDTH - 1));
    assert_eq!(start, ROW_WIDTH);

    assert_eq!(row.lookup(999, &mut start), None);
    assert_eq!(start, ROW_WIDTH);

    Ok(())
}

#[repr(C, align(4096))]
struct PageAligned<T>(T);

pub(crate) const SHARD_FILE_MAGIC: [u8; 8] = *b"CandyStr";
pub(crate) const SHARD_FILE_VERSION: u64 = 0x08;

#[derive(Clone, Copy, Default, Debug, Pod, Zeroable)]
#[repr(C)]
struct MetaHeader {
    magic: [u8; 8],
    version: u64,
}

#[repr(C)]
struct ShardHeader {
    metadata: MetaHeader,
    wasted_bytes: AtomicU64,
    write_offset: AtomicU64,
    num_entries: AtomicU64,
    rows: PageAligned<[ShardRow; NUM_ROWS]>,
}

pub(crate) const HEADER_SIZE: u64 = size_of::<ShardHeader>() as u64;
const _: () = assert!(HEADER_SIZE % 4096 == 0);

#[derive(Debug)]
pub(crate) enum InsertStatus {
    Added,
    Replaced(Vec<u8>),
    KeyDoesNotExist,
    SplitNeeded,
    AlreadyExists(Vec<u8>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum InsertMode<'a> {
    Set,
    Replace(Option<&'a [u8]>),
    GetOrCreate,
    MustCreate,
}

enum TryReplaceStatus {
    KeyDoesNotExist(bool),
    KeyExistsNotReplaced(Vec<u8>),
    KeyExistsReplaced(Vec<u8>, bool),
}

pub(crate) type KVPair = (Vec<u8>, Vec<u8>);

struct MmapHeaderFile {
    file: File,
    ptr: *mut c_void,
    stats: Arc<InternalStats>,
}

unsafe impl Sync for MmapHeaderFile {}
unsafe impl Send for MmapHeaderFile {}

impl MmapHeaderFile {
    fn new(file: File, stats: Arc<InternalStats>, mlock: bool) -> Result<Self> {
        let ptr = unsafe {
            libc::mmap(
                null_mut(),
                HEADER_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                if mlock { libc::MAP_LOCKED } else { 0 } | libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        ensure!(ptr != libc::MAP_FAILED, std::io::Error::last_os_error());
        Ok(Self { file, ptr, stats })
    }

    fn header(&self) -> &ShardHeader {
        unsafe { &*(self.ptr as *const ShardHeader) }
    }
    fn header_mut(&self) -> &mut ShardHeader {
        unsafe { &mut *(self.ptr as *mut ShardHeader) }
    }

    // reading doesn't require holding any locks - we only ever extend the file, never overwrite data
    fn read_kv(&self, offset_and_size: u64) -> Result<KVPair> {
        let klen = (offset_and_size >> 48) as usize;
        debug_assert_eq!(klen >> 14, 0, "attempting to read a special key");
        let vlen = ((offset_and_size >> 32) & 0xffff) as usize;
        let offset = (offset_and_size as u32) as u64;
        let mut buf = vec![0u8; klen + vlen];
        self.file.read_exact_at(&mut buf, HEADER_SIZE + offset)?;

        self.stats
            .num_read_bytes
            .fetch_add(buf.len(), Ordering::Relaxed);
        self.stats.num_read_ops.fetch_add(1, Ordering::Relaxed);

        let val = buf[klen..klen + vlen].to_owned();
        buf.truncate(klen);

        Ok((buf, val))
    }

    // writing doesn't require holding any locks since we write with an offset
    fn write_kv(&self, key: &[u8], val: &[u8]) -> Result<u64> {
        let entry_size = key.len() + val.len();
        let mut buf = vec![0u8; entry_size];
        buf[..key.len()].copy_from_slice(key);
        buf[key.len()..].copy_from_slice(val);

        // atomically allocate some area. it may leak if the IO below fails or if we crash before updating the
        // offsets_and_size array, but we're okay with leaks
        let write_offset = self
            .header()
            .write_offset
            .fetch_add(buf.len() as u64, Ordering::SeqCst) as u64;

        // now writing can be non-atomic (pwrite)
        self.file.write_all_at(&buf, HEADER_SIZE + write_offset)?;
        self.stats.add_entry(entry_size);

        Ok(((key.len() as u64) << 48) | ((val.len() as u64) << 32) | write_offset)
    }
}

impl Drop for MmapHeaderFile {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr, HEADER_SIZE as usize) };
        self.ptr = libc::MAP_FAILED;
    }
}

const COMPACTION_NOT_RUNNING: u8 = 0;
const COMPACTION_RUNNING: u8 = 1;
const COMPACTION_FINISHED: u8 = 2;

struct InnerShard {
    files: RwLock<(MmapHeaderFile, Option<MmapHeaderFile>)>,
    compacted_up_to_idx: AtomicUsize,
    compaction_state: AtomicU8,
    row_locks: Vec<RwLock<()>>,
}

pub(crate) struct Shard {
    pub(crate) span: Range<u32>,
    pub(crate) config: Arc<InternalConfig>,
    inner: Arc<InnerShard>,
    stats: Arc<InternalStats>,
    compaction_handle: Mutex<Option<JoinHandle<Result<()>>>>,
    #[cfg(feature = "flush_aggregation")]
    sync_agg_mutex: parking_lot::Mutex<()>,
    #[cfg(feature = "flush_aggregation")]
    in_sync_agg_delay: std::sync::atomic::AtomicBool,
}

impl Shard {
    pub(crate) const EXPECTED_CAPACITY: usize = (NUM_ROWS * ROW_WIDTH * 9) / 10; // ~ 29,500

    pub(crate) fn open(
        filename: PathBuf,
        span: Range<u32>,
        truncate: bool,
        config: Arc<InternalConfig>,
        stats: Arc<InternalStats>,
    ) -> Result<Self> {
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(truncate)
            .open(&filename)?;

        let mut file_size = file.metadata()?.len();
        if file_size != 0 {
            let mut meta_header = MetaHeader::default();
            let sz = file.read(bytes_of_mut(&mut meta_header))?;
            if sz != size_of::<MetaHeader>()
                || meta_header.magic != SHARD_FILE_MAGIC
                || meta_header.version != SHARD_FILE_VERSION
            {
                if config.clear_on_unsupported_version {
                    file.set_len(0)?;
                    file_size = 0;
                } else {
                    bail!(
                        "{filename:?} unsupported magic={:?} version={} size={}",
                        meta_header.magic,
                        meta_header.version,
                        file_size,
                    );
                }
            }

            if file_size != 0 && file_size < HEADER_SIZE {
                if config.clear_on_unsupported_version {
                    file.set_len(0)?;
                    file_size = 0;
                } else {
                    bail!("corrupt shard file (size={})", file_size);
                }
            }
        }

        if file_size == 0 {
            if config.truncate_up {
                // when creating, set the file's length so that we won't need to extend it every time we write
                // (saves on file metadata updates)
                file.set_len(HEADER_SIZE + config.max_shard_size as u64)?;
            } else {
                file.set_len(HEADER_SIZE)?;
            }
        }

        let mhf = MmapHeaderFile::new(file, stats.clone(), config.mlock_headers)?;
        mhf.header_mut().metadata.magic = SHARD_FILE_MAGIC;
        mhf.header_mut().metadata.version = SHARD_FILE_VERSION;

        if file_size > 0 {
            // if the shard existed before, update the stats
            stats.num_inserts.fetch_add(
                mhf.header().num_entries.load(Ordering::Relaxed) as usize,
                Ordering::Relaxed,
            );
        }

        let mut row_locks = Vec::with_capacity(NUM_ROWS);
        for _ in 0..NUM_ROWS {
            row_locks.push(RwLock::new(()));
        }

        let inner = InnerShard {
            compacted_up_to_idx: AtomicUsize::new(0),
            compaction_state: AtomicU8::new(COMPACTION_NOT_RUNNING),
            files: RwLock::new((mhf, None)),
            row_locks,
        };

        Ok(Self {
            span,
            config,
            inner: Arc::new(inner),
            stats,
            compaction_handle: Default::default(),
            #[cfg(feature = "flush_aggregation")]
            sync_agg_mutex: parking_lot::Mutex::new(()),
            #[cfg(feature = "flush_aggregation")]
            in_sync_agg_delay: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn flush(&self) -> Result<()> {
        let guard = self.inner.files.read();
        guard.0.file.sync_data()?;
        // no need to flush the currently-compacting file, if any
        Ok(())
    }

    pub(crate) fn read_at(&self, row_idx: usize, entry_idx: usize) -> Result<Option<KVPair>> {
        self.operate_on_row(row_idx, |mhf, row| {
            if row.signatures[entry_idx] != INVALID_SIG {
                Ok(Some(mhf.read_kv(row.offsets_and_sizes[entry_idx])?))
            } else {
                Ok(None)
            }
        })
    }

    pub(crate) fn split_into(&self, bottom_shard: &Shard, top_shard: &Shard) -> Result<()> {
        // once we have the write lock, no compaction can be ongoing
        let guard = self.inner.files.write();
        assert!(
            guard.1.is_none(),
            "[{:?}] guard.1 = {:?}",
            self.span,
            guard.1.as_ref().unwrap().ptr
        );

        for row in guard.0.header().rows.0.iter() {
            for (i, &sig) in row.signatures.iter().enumerate() {
                if sig != INVALID_SIG {
                    let (k, v) = guard.0.read_kv(row.offsets_and_sizes[i])?;
                    let ph = PartedHash::new(&self.config.hash_seed, &k);
                    println!(
                        "[{:?}] SPLIT => {k:?} {v:?} sh={}",
                        self.span,
                        ph.shard_selector()
                    );
                    if ph.shard_selector() < bottom_shard.span.end {
                        bottom_shard.insert(ph, &k, &v, InsertMode::MustCreate, false)?;
                    } else {
                        top_shard.insert(ph, &k, &v, InsertMode::MustCreate, false)?;
                    }
                }
            }
        }

        Ok(())
    }

    fn check_compaction_res(&self) -> Result<()> {
        match self.inner.compaction_state.load(Ordering::Relaxed) {
            COMPACTION_NOT_RUNNING => {}
            COMPACTION_RUNNING => {
                let mut handle_guard = self.compaction_handle.lock();
                if let Some(ref handle) = &*handle_guard {
                    if !handle.is_finished() {
                        return Ok(());
                    }
                }
                // this means the thread crashed before clearing the state
                if let Some(handle) = handle_guard.take() {
                    let res = handle.join().unwrap();
                    self.inner
                        .compaction_state
                        .store(COMPACTION_NOT_RUNNING, Ordering::Relaxed);
                    res?;
                }
            }
            COMPACTION_FINISHED => {
                let mut handle_guard = self.compaction_handle.lock();
                if let Some(handle) = handle_guard.take() {
                    handle.join().unwrap()?;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    fn operate_on_row<T>(
        &self,
        row_idx: usize,
        func: impl FnOnce(&MmapHeaderFile, &ShardRow) -> Result<T>,
    ) -> Result<T> {
        let file_guard = self.inner.files.read();
        self.check_compaction_res()?;
        let _row_guard = self.inner.row_locks[row_idx].read();

        if let Some(ref mhf) = file_guard.1 {
            if row_idx < self.inner.compacted_up_to_idx.load(Ordering::Relaxed) {
                return func(mhf, &mhf.header().rows.0[row_idx]);
            }
        }

        func(&file_guard.0, &file_guard.0.header().rows.0[row_idx])
    }

    fn operate_on_row_mut<T>(
        &self,
        row_idx: usize,
        func: impl FnOnce(&MmapHeaderFile, &mut ShardRow) -> Result<T>,
    ) -> Result<T> {
        let file_guard = self.inner.files.read();
        self.check_compaction_res()?;
        let _row_guard = self.inner.row_locks[row_idx].write();

        if let Some(ref mhf) = file_guard.1 {
            if row_idx < self.inner.compacted_up_to_idx.load(Ordering::Relaxed) {
                return func(mhf, &mut mhf.header_mut().rows.0[row_idx]);
            }
        }

        func(
            &file_guard.0,
            &mut file_guard.0.header_mut().rows.0[row_idx],
        )
    }

    pub(crate) fn get_by_hash(&self, ph: PartedHash) -> Result<Vec<KVPair>> {
        let mut kvs = Vec::with_capacity(1);

        self.operate_on_row(ph.row_selector(), |mhf, row| {
            let mut start = 0;
            let mut first_time = true;
            while let Some(idx) = row.lookup(ph.signature(), &mut start) {
                kvs.push(mhf.read_kv(row.offsets_and_sizes[idx])?);
                if first_time {
                    self.stats
                        .num_positive_lookups
                        .fetch_add(1, Ordering::Relaxed);
                    first_time = false;
                }
            }
            if kvs.is_empty() {
                self.stats
                    .num_negative_lookups
                    .fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })?;

        Ok(kvs)
    }

    pub(crate) fn get(&self, ph: PartedHash, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.operate_on_row(ph.row_selector(), |mhf, row| {
            let mut start = 0;
            while let Some(idx) = row.lookup(ph.signature(), &mut start) {
                let (k, v) = mhf.read_kv(row.offsets_and_sizes[idx])?;
                if key == k {
                    self.stats
                        .num_positive_lookups
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(v));
                }
            }
            self.stats
                .num_negative_lookups
                .fetch_add(1, Ordering::Relaxed);
            Ok(None)
        })
    }

    #[cfg(feature = "flush_aggregation")]
    fn flush_aggregation(&self) -> Result<()> {
        let Some(delay) = self.config.flush_aggregation_delay else {
            return Ok(());
        };

        let do_sync = || {
            self.in_sync_agg_delay.store(true, Ordering::SeqCst);
            std::thread::sleep(delay);
            self.in_sync_agg_delay.store(false, Ordering::SeqCst);
            self.file.sync_data()
        };

        if let Some(_guard) = self.sync_agg_mutex.try_lock() {
            // we're the first ones here. wait for the aggregation duration and sync the file
            do_sync()?;
        } else {
            // another thread is currently sync'ing, we're waiting in line. if the holder of the lock is in the
            // sleep (aggregation) phase, we can just wait for it to finish and return -- the other thread will
            // have sync'ed us by the time we got the lock. otherwise, we'll need to sync as well
            let was_in_delay = self.in_sync_agg_delay.load(Ordering::Relaxed);
            let _guard = self.sync_agg_mutex.lock();
            if !was_in_delay {
                do_sync()?;
            }
        }
        Ok(())
    }

    fn try_replace(
        &self,
        mhf: &MmapHeaderFile,
        row: &mut ShardRow,
        ph: PartedHash,
        key: &[u8],
        val: &[u8],
        mode: InsertMode,
        inc_stats: bool,
    ) -> Result<TryReplaceStatus> {
        let mut start = 0;
        let mut had_collision = false;
        while let Some(idx) = row.lookup(ph.signature(), &mut start) {
            let (k, existing_val) = mhf.read_kv(row.offsets_and_sizes[idx])?;
            if key != k {
                had_collision = true;
                continue;
            }
            match mode {
                InsertMode::MustCreate => {
                    panic!("[{:?}] key already exists {:?} {}", self.span, key, ph);
                    bail!(CandyError::KeyAlreadyExists(
                        self.span.clone(),
                        key.into(),
                        ph.as_u64()
                    ))
                }
                InsertMode::GetOrCreate => {
                    // no-op, key already exists
                    if inc_stats {
                        self.stats
                            .num_positive_lookups
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(TryReplaceStatus::KeyExistsNotReplaced(existing_val));
                }
                InsertMode::Set => {
                    // fall through
                }
                InsertMode::Replace(expected_val) => {
                    if expected_val.is_some_and(|expected_val| expected_val != existing_val) {
                        return Ok(TryReplaceStatus::KeyExistsNotReplaced(existing_val));
                    }
                }
            }

            let mut should_compact = false;
            // optimization
            if val != existing_val {
                row.offsets_and_sizes[idx] = mhf.write_kv(key, val)?;
                mhf.header()
                    .wasted_bytes
                    .fetch_add((k.len() + existing_val.len()) as u64, Ordering::Relaxed);
                if inc_stats {
                    self.stats.num_updates.fetch_add(1, Ordering::Relaxed);
                }
                should_compact = self.should_compact(mhf);

                #[cfg(feature = "flush_aggregation")]
                {
                    //drop(guard);
                    self.flush_aggregation()?;
                }
            }
            return Ok(TryReplaceStatus::KeyExistsReplaced(
                existing_val,
                should_compact,
            ));
        }

        Ok(TryReplaceStatus::KeyDoesNotExist(had_collision))
    }

    fn should_compact(&self, mhf: &MmapHeaderFile) -> bool {
        let state = self.inner.compaction_state.load(Ordering::Relaxed);
        let waste = mhf.header().wasted_bytes.load(Ordering::Relaxed);
        state == COMPACTION_NOT_RUNNING && waste > self.config.min_compaction_threashold as u64
    }

    pub(crate) fn insert(
        &self,
        ph: PartedHash,
        full_key: &[u8],
        val: &[u8],
        mode: InsertMode,
        inc_stats: bool,
    ) -> Result<InsertStatus> {
        let mut should_compact = false;

        let res = self.operate_on_row_mut(ph.row_selector(), |mhf, row| {
            if self.inner.compaction_state.load(Ordering::Relaxed) != COMPACTION_RUNNING {
                if mhf.header().write_offset.load(Ordering::Relaxed) as u64
                    + (full_key.len() + val.len()) as u64
                    > self.config.max_shard_size as u64
                {
                    return Ok(InsertStatus::SplitNeeded);
                }
            }

            let status = self.try_replace(mhf, row, ph, &full_key, val, mode, inc_stats)?;

            match status {
                TryReplaceStatus::KeyDoesNotExist(had_collision) => {
                    if matches!(mode, InsertMode::Replace(_)) {
                        return Ok(InsertStatus::KeyDoesNotExist);
                    }

                    // find an empty slot
                    let mut start = 0;
                    if let Some(idx) = row.lookup(INVALID_SIG, &mut start) {
                        row.offsets_and_sizes[idx] = mhf.write_kv(&full_key, val)?;
                        // we don't want a reorder to happen here - first write the offset, then the signature
                        std::sync::atomic::fence(Ordering::SeqCst);
                        row.signatures[idx] = ph.signature();
                        if inc_stats {
                            if had_collision {
                                self.stats.num_collisions.fetch_add(1, Ordering::Relaxed);
                            }
                            self.stats.num_inserts.fetch_add(1, Ordering::Relaxed);
                        }
                        mhf.header().num_entries.fetch_add(1, Ordering::Relaxed);
                        #[cfg(feature = "flush_aggregation")]
                        {
                            //drop(_guard);
                            self.flush_aggregation()?;
                        }
                        Ok(InsertStatus::Added)
                    } else {
                        // no room in this row, must split
                        Ok(InsertStatus::SplitNeeded)
                    }
                }
                TryReplaceStatus::KeyExistsNotReplaced(existing) => {
                    Ok(InsertStatus::AlreadyExists(existing))
                }
                TryReplaceStatus::KeyExistsReplaced(existing, should_compact_) => {
                    should_compact = should_compact_;
                    Ok(InsertStatus::Replaced(existing))
                }
            }
        })?;

        if should_compact {
            assert!(!matches!(mode, InsertMode::MustCreate));
            self.begin_compaction()?;
        }

        Ok(res)
    }

    pub(crate) fn remove(&self, ph: PartedHash, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let (res, should_compact) = self.operate_on_row_mut(ph.row_selector(), |mhf, row| {
            let mut start = 0;
            while let Some(idx) = row.lookup(ph.signature(), &mut start) {
                let (k, v) = mhf.read_kv(row.offsets_and_sizes[idx])?;
                if key == k {
                    row.signatures[idx] = INVALID_SIG;
                    // we managed to remove this key
                    self.stats.num_removals.fetch_add(1, Ordering::Relaxed);
                    mhf.header().num_entries.fetch_sub(1, Ordering::Relaxed);
                    mhf.header()
                        .wasted_bytes
                        .fetch_add((k.len() + v.len()) as u64, Ordering::Relaxed);
                    #[cfg(feature = "flush_aggregation")]
                    {
                        drop(_guard);
                        self.flush_aggregation()?;
                    }
                    let should_compact = self.should_compact(mhf);

                    return Ok((Some(v), should_compact));
                }
            }

            Ok((None, false))
        })?;

        if should_compact {
            self.begin_compaction()?;
        }
        Ok(res)
    }

    fn begin_compaction(&self) -> Result<()> {
        println!("[{:?}] beginning compaction", self.span);
        // take a write lock, which would prevent any competing compaction/split/IO
        let _files_guard = self.inner.files.write();

        match self.inner.compaction_state.load(Ordering::Relaxed) {
            COMPACTION_NOT_RUNNING | COMPACTION_FINISHED => {
                // continue
            }
            COMPACTION_RUNNING => {
                // already running, just skip
                println!("[{:?}] compaction skips", self.span);
                return Ok(());
            }
            _ => unreachable!(),
        }

        let mut handle_guard = self.compaction_handle.lock();
        if let Some(handle) = handle_guard.take() {
            handle.join().unwrap()?;
        }
        self.inner
            .compaction_state
            .store(COMPACTION_RUNNING, Ordering::SeqCst);

        let inner = self.inner.clone();
        let span = self.span.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();
        let t0 = Instant::now();

        let handle = std::thread::spawn(move || {
            let res = (|| {
                let mut files_guard = inner.files.upgradable_read();
                assert!(files_guard.1.is_none());

                let orig_filename = config
                    .dir_path
                    .join(format!("shard_{:04x}-{:04x}", span.start, span.end));
                let tmp_filename = config
                    .dir_path
                    .join(format!("compact_{:04x}-{:04x}", span.start, span.end));
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .read(true)
                    .write(true)
                    .open(&tmp_filename)?;
                if config.truncate_up {
                    file.set_len(HEADER_SIZE + config.max_shard_size as u64)?;
                } else {
                    file.set_len(HEADER_SIZE)?;
                }
                let dst = MmapHeaderFile::new(file, stats.clone(), config.mlock_headers)?;
                files_guard.with_upgraded(|files| {
                    files.1 = Some(dst);
                });

                let src = &files_guard.0;
                let dst = files_guard.1.as_ref().unwrap();

                println!(
                    "[{span:?}] COMPACTION THREAD running {orig_filename:?} -> {tmp_filename:?}",
                );

                for row_idx in 0..NUM_ROWS {
                    inner.compacted_up_to_idx.store(row_idx, Ordering::SeqCst);
                    let _row_guard = inner.row_locks[row_idx].write();

                    let src_row = &src.header().rows.0[row_idx];
                    let dst_row = &mut dst.header_mut().rows.0[row_idx];

                    println!("[{span:?}] COMPACTION THREAD row_idx={row_idx}");
                    for (i, &sig) in src_row.signatures.iter().enumerate() {
                        if sig == INVALID_SIG {
                            continue;
                        }
                        let (k, v) = src.read_kv(src_row.offsets_and_sizes[i])?;

                        let ph = PartedHash::new(&config.hash_seed, &k);
                        assert!(ph.row_selector() == row_idx, "{ph} row_idx={row_idx}");

                        {
                            let mut start_idx = 0;
                            while let Some(idx) = dst_row.lookup(ph.signature(), &mut start_idx) {
                                let (k2, _) = dst.read_kv(dst_row.offsets_and_sizes[idx])?;
                                assert!(
                                    k != k2,
                                    "key {k:?} already exists at idx={idx} sigs={:?}",
                                    dst_row.signatures
                                );
                            }
                        }

                        let mut start_idx = 0;
                        if let Some(idx) = dst_row.lookup(INVALID_SIG, &mut start_idx) {
                            dst_row.offsets_and_sizes[idx] = dst.write_kv(&k, &v)?;
                            std::sync::atomic::fence(Ordering::SeqCst);
                            dst_row.signatures[idx] = ph.signature();
                        } else {
                            panic!("row full");
                        }
                    }
                }

                println!("[{span:?}] COMPACTION THREAD renaming");
                std::fs::rename(tmp_filename, orig_filename)?;

                files_guard.with_upgraded(|files| {
                    let src_woff = files.0.header().write_offset.load(Ordering::Relaxed);
                    files.0 = files.1.take().unwrap();
                    let dst_woff = files.0.header().write_offset.load(Ordering::Relaxed);
                    stats.report_compaction(t0, src_woff, dst_woff);
                });
                println!("[{span:?}] COMPACTION THREAD done");
                assert!(files_guard.1.is_none());

                Ok(())
            })();

            println!("[{span:?}] COMPACTION THREAD res={res:?}");

            inner
                .compaction_state
                .store(COMPACTION_FINISHED, Ordering::SeqCst);
            res
        });

        //println!("started compaction");
        *handle_guard = Some(handle);

        Ok(())
    }

    pub(crate) fn wait_for_compaction(&self) -> Result<()> {
        let _guard = self.inner.files.read();
        let mut handle_guard = self.compaction_handle.lock();
        if let Some(handle) = handle_guard.take() {
            handle.join().unwrap()?;
        }
        Ok(())
    }

    pub(crate) fn get_write_offset(&self) -> u64 {
        self.inner
            .files
            .read()
            .0
            .header()
            .write_offset
            .load(Ordering::Relaxed)
    }
    pub(crate) fn get_wasted_bytes(&self) -> u64 {
        self.inner
            .files
            .read()
            .0
            .header()
            .wasted_bytes
            .load(Ordering::Relaxed)
    }
}
