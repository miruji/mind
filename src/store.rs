//! The long term memory of the mind.
//!
//! The rows of the mind live in the window of the RAM while they are in use.
//! A row that leaves the window is not lost: it is written into the file and
//! the index remembers where it lies, so that the row can be raised again
//! when its feature comes up.
//!
//! The file holds one area with the records of both tables of the mind. The
//! area only grows, so the offsets of the records stay valid for ever. The
//! index (the hash of a feature -> the offset of its newest record) is
//! rewritten at every save, it is small: 16 bytes per feature.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::rc::Rc;

/// The size of the header of the checkpoint file.
pub const HEADER: u64 = 4096;

/// The kind of a record: a row of the subconscious.
pub const KIND_ROW: u8 = 0;
/// The kind of a record: the edges of one feature of the consciousness.
pub const KIND_EDGE: u8 = 1;

/// The biggest record of the long term memory that is read back. A record
/// that is bigger is broken and is skipped.
pub const MAXRECORD: usize = 1 << 22;

/// The hash of a feature name. The index of the memory uses it, so a feature
/// can be found without reading its name from the disk.
pub fn hashFeature(feature: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in feature.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The records of the long term memory: the rows of the tables of the mind
/// that do not fit into the window of the RAM.
pub struct Records {
    /// The offset of the area in the file.
    start: u64,
    /// The bytes of the area that are already on the disk.
    len: u64,
    /// The records that were not written to the disk yet.
    pub pending: Vec<u8>,
    /// The index of the subconscious: the hash of a feature -> the offset of
    /// its newest record.
    pub rows: HashMap<u64, u64>,
    /// The index of the consciousness: the same for its features.
    pub edges: HashMap<u64, u64>,
    /// How many records were written to the disk.
    pub written: u64,
}

impl Records {
    pub fn new(start: u64) -> Records {
        Records {
            start,
            len: 0,
            pending: Vec::new(),
            rows: HashMap::new(),
            edges: HashMap::new(),
            written: 0,
        }
    }

    /// The number of bytes the area takes on the disk.
    pub fn len(&self) -> u64 { self.len }

    /// Sets the length of the area that is already on the disk. It is read
    /// back from the header of the checkpoint.
    pub fn setLen(&mut self, len: u64) { self.len = len; }

    /// Adds a record and returns the offset it will have on the disk. The
    /// index of the feature points to it from now on: it is the newest
    /// version of the row.
    pub fn append(&mut self, kind: u8, hash: u64, bytes: &[u8]) -> u64 {
        let offset = self.start + self.len + self.pending.len() as u64;
        self.pending.push(kind);
        self.pending.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.pending.extend_from_slice(bytes);
        match kind {
            KIND_ROW => { self.rows.insert(hash, offset); }
            _ => { self.edges.insert(hash, offset); }
        }
        offset
    }

    /// Writes the records that are waiting in the memory into the file.
    pub fn flush(&mut self, file: &mut File) -> std::io::Result<()> {
        if self.pending.is_empty() { return Ok(()); }
        file.seek(SeekFrom::Start(self.start + self.len))?;
        file.write_all(&self.pending)?;
        self.len += self.pending.len() as u64;
        self.written += 1;
        self.pending.clear();
        Ok(())
    }

    /// Reads one record back.
    pub fn read(&self, file: &mut File, offset: u64) -> std::io::Result<Vec<u8>> {
        let mut head = [0u8; 5];
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut head)?;
        let len = u32::from_le_bytes([head[1], head[2], head[3], head[4]]) as usize;
        if len > MAXRECORD {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "the record of the memory is broken"));
        }
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

/// The long term memory: the file of the checkpoint and the records in it.
pub struct Store {
    pub file: File,
    pub records: Records,
    /// When the memory is read only the rows leave the window and are lost.
    writable: bool,
}

/// The long term memory, shared between the tables of the mind.
pub type SharedStore = Rc<RefCell<Store>>;

impl Store {
    /// Opens the memory of a checkpoint. The file is created when it is not
    /// there yet.
    pub fn open(path: &Path) -> std::io::Result<Store> {
        let file = OpenOptions::new().read(true).write(true).create(true).open(path)?;
        Ok(Store { file, records: Records::new(HEADER), writable: true })
    }

    /// A read only memory keeps the rows in the window only: they are lost
    /// when they leave it.
    pub fn setWritable(&mut self, writable: bool) { self.writable = writable; }

    /// Whether the rows that leave the window are written into the file.
    pub fn writable(&self) -> bool { self.writable }

    /// The number of records that are waiting in the memory.
    pub fn pending(&self) -> usize { self.records.pending.len() }
}

/// Copies a checkpoint to another file, so that a session can read one
/// memory and write another one.
pub fn copyMemory(from: &Path, to: &Path) -> std::io::Result<u64> {
    std::fs::copy(from, to)?;
    Ok(std::fs::metadata(to)?.len())
}
