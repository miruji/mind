//! Checkpoints of the mind: the whole state (subconscious rows, conscious
//! edges, the act classifier) plus the position in the streamed corpus, so a
//! long run can be split into chunks and continued later.
//!
//! Version 4 is the window of the mind: the `.bin` holds the long term
//! memory (the rows that left the window of the RAM) with an index that says
//! where every feature lies, the RAM holds the window itself. A row that
//! leaves the window is written into the file, a row that comes up again is
//! raised from the file. Nothing is lost.
//!
//! The file:
//!
//! ```text
//! [0, 4096)          the header: the position, the counters, the lengths
//! [4096, records)    the long term records, they are only appended
//! [records, ...)     the index of the rows, the index of the features
//! [...]              the window of the subconscious
//! [...]              the window of the consciousness
//! ```
//!
//! Version 3 (the old dump of the whole mind in a row) is still read.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::binio::{readU64, writeU64};
use crate::dialogue::ReplyStats;
use crate::mind::{Consciousness, Subconscious};
use crate::store::{SharedStore, Store, HEADER};

const MAGIC: &[u8; 8] = b"MINDCKPT";
const VERSION: u64 = 4;
const VERSION3: u64 = 3;

/// Everything that has to survive between two chunks.
pub struct Checkpoint {
    /// Dialogues of the streamed corpus that were already read.
    pub dialogsDone: u64,
    /// Dialogues of the streamed corpus that were already seen.
    pub dialogsSeen: u64,
    /// Training statistics of the corpus so far.
    pub train: ReplyStats,
    /// Statistics of the held out part so far.
    pub holdout: ReplyStats,
}

impl Default for Checkpoint {
    fn default() -> Checkpoint {
        Checkpoint {
            dialogsDone: 0,
            dialogsSeen: 0,
            train: ReplyStats::default(),
            holdout: ReplyStats::default(),
        }
    }
}

fn saveStats(out: &mut dyn Write, st: &ReplyStats) -> std::io::Result<()> {
    let values = [
        st.steps, st.top1, st.top3, st.surprises, st.wills,
        st.turns, st.turnTop1, st.turnTop3, st.cand, st.candSize,
    ];
    for v in values { writeU64(out, v)?; }
    Ok(())
}

fn loadStats(input: &mut dyn Read) -> std::io::Result<ReplyStats> {
    let mut st = ReplyStats::default();
    st.steps = readU64(input)?;
    st.top1 = readU64(input)?;
    st.top3 = readU64(input)?;
    st.surprises = readU64(input)?;
    st.wills = readU64(input)?;
    st.turns = readU64(input)?;
    st.turnTop1 = readU64(input)?;
    st.turnTop3 = readU64(input)?;
    st.cand = readU64(input)?;
    st.candSize = readU64(input)?;
    Ok(st)
}

/// The header of the file: the position in the corpus, the counters of the
/// mind and the lengths of the parts of the file.
struct Header {
    version: u64,
    dialogsDone: u64,
    dialogsSeen: u64,
    train: ReplyStats,
    holdout: ReplyStats,
    recordsLen: u64,
    rowsIndexLen: u64,
    edgesIndexLen: u64,
    subWindowLen: u64,
    consWindowLen: u64,
    passiveWrites: u64,
    consciousWrites: u64,
    consActions: u64,
    consFailed: u64,
    subEvicted: u64,
    consEvicted: u64,
    subRaised: u64,
    consRaised: u64,
}

fn putU64(buf: &mut [u8], off: usize, value: u64) {
    buf[off..off + 8].copy_from_slice(&value.to_le_bytes());
}

fn getU64(buf: &[u8], off: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[off..off + 8]);
    u64::from_le_bytes(b)
}

impl Header {
    fn write(&self) -> Vec<u8> {
        let mut buf = vec![0u8; HEADER as usize];
        buf[0..8].copy_from_slice(MAGIC);
        putU64(&mut buf, 8, self.version);
        putU64(&mut buf, 16, self.dialogsDone);
        putU64(&mut buf, 24, self.dialogsSeen);
        let mut off = 32usize;
        for v in [self.train.steps, self.train.top1, self.train.top3, self.train.surprises, self.train.wills,
                  self.train.turns, self.train.turnTop1, self.train.turnTop3, self.train.cand, self.train.candSize] {
            putU64(&mut buf, off, v); off += 8;
        }
        for v in [self.holdout.steps, self.holdout.top1, self.holdout.top3, self.holdout.surprises, self.holdout.wills,
                  self.holdout.turns, self.holdout.turnTop1, self.holdout.turnTop3, self.holdout.cand, self.holdout.candSize] {
            putU64(&mut buf, off, v); off += 8;
        }
        putU64(&mut buf, off, self.recordsLen); off += 8;
        putU64(&mut buf, off, self.rowsIndexLen); off += 8;
        putU64(&mut buf, off, self.edgesIndexLen); off += 8;
        putU64(&mut buf, off, self.subWindowLen); off += 8;
        putU64(&mut buf, off, self.consWindowLen); off += 8;
        putU64(&mut buf, off, self.passiveWrites); off += 8;
        putU64(&mut buf, off, self.consciousWrites); off += 8;
        putU64(&mut buf, off, self.consActions); off += 8;
        putU64(&mut buf, off, self.consFailed); off += 8;
        putU64(&mut buf, off, self.subEvicted); off += 8;
        putU64(&mut buf, off, self.consEvicted); off += 8;
        putU64(&mut buf, off, self.subRaised); off += 8;
        putU64(&mut buf, off, self.consRaised);
        buf
    }

    fn read(buf: &[u8]) -> Header {
        let mut off = 32usize;
        let mut take = |off: &mut usize| -> u64 { let v = getU64(buf, *off); *off += 8; v };
        let mut train = ReplyStats::default();
        train.steps = take(&mut off);
        train.top1 = take(&mut off);
        train.top3 = take(&mut off);
        train.surprises = take(&mut off);
        train.wills = take(&mut off);
        train.turns = take(&mut off);
        train.turnTop1 = take(&mut off);
        train.turnTop3 = take(&mut off);
        train.cand = take(&mut off);
        train.candSize = take(&mut off);
        let mut holdout = ReplyStats::default();
        holdout.steps = take(&mut off);
        holdout.top1 = take(&mut off);
        holdout.top3 = take(&mut off);
        holdout.surprises = take(&mut off);
        holdout.wills = take(&mut off);
        holdout.turns = take(&mut off);
        holdout.turnTop1 = take(&mut off);
        holdout.turnTop3 = take(&mut off);
        holdout.cand = take(&mut off);
        holdout.candSize = take(&mut off);
        Header {
            version: getU64(buf, 8),
            dialogsDone: getU64(buf, 16),
            dialogsSeen: getU64(buf, 24),
            train,
            holdout,
            recordsLen: take(&mut off),
            rowsIndexLen: take(&mut off),
            edgesIndexLen: take(&mut off),
            subWindowLen: take(&mut off),
            consWindowLen: take(&mut off),
            passiveWrites: take(&mut off),
            consciousWrites: take(&mut off),
            consActions: take(&mut off),
            consFailed: take(&mut off),
            subEvicted: take(&mut off),
            consEvicted: take(&mut off),
            subRaised: take(&mut off),
            consRaised: take(&mut off),
        }
    }
}

/// Counts the bytes that are written, so that the length of a part of the
/// file is known.
struct Counting<'a> { file: &'a mut File, count: u64 }

impl Write for Counting<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.file.write(buf)?;
        self.count += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.file.flush() }
}

/// Writes the whole mind to disk: the long term memory with its index and
/// the window of the RAM.
pub fn save(
    path: &Path,
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    cp: &Checkpoint,
    store: &SharedStore,
) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() { std::fs::create_dir_all(dir)?; }
    }
    let mut store = store.borrow_mut();
    // the records that are waiting in the RAM go into the file first: after
    // this the area of the records is complete
    {
        let Store { file, records, .. } = &mut *store;
        records.flush(file)?;
    }

    // the parts of the file follow the area of the records: it is only
    // appended, so the offsets of the records stay valid
    let mut pos = HEADER + store.records.len();
    // the index of the rows of the subconscious
    let rowsIndexLen = {
        let Store { file, records, .. } = &mut *store;
        writeIndex(file, pos, &records.rows)?
    };
    pos += rowsIndexLen;
    // the index of the features of the consciousness
    let edgesIndexLen = {
        let Store { file, records, .. } = &mut *store;
        writeIndex(file, pos, &records.edges)?
    };
    pos += edgesIndexLen;
    // the window of the subconscious
    let subLen = {
        let Store { file, .. } = &mut *store;
        writeSection(file, pos, |out| sub.save(out))?
    };
    pos += subLen;
    // the window of the consciousness
    let consLen = {
        let Store { file, .. } = &mut *store;
        writeSection(file, pos, |out| cons.save(out))?
    };
    pos += consLen;

    let (passive, conscious, subEvicted, subRaised) = sub.counters();
    let (actions, failed, consEvicted, consRaised) = cons.counters();
    let header = Header {
        version: VERSION,
        dialogsDone: cp.dialogsDone,
        dialogsSeen: cp.dialogsSeen,
        train: cp.train.clone(),
        holdout: cp.holdout.clone(),
        recordsLen: store.records.len(),
        rowsIndexLen,
        edgesIndexLen,
        subWindowLen: subLen,
        consWindowLen: consLen,
        passiveWrites: passive,
        consciousWrites: conscious,
        consActions: actions,
        consFailed: failed,
        subEvicted,
        consEvicted,
        subRaised,
        consRaised,
    };
    let bytes = header.write();
    store.file.seek(SeekFrom::Start(0))?;
    store.file.write_all(&bytes)?;
    store.file.flush()?;
    Ok(())
}

/// Writes one part of the file and returns its length.
fn writeSection(
    file: &mut File,
    pos: u64,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
) -> std::io::Result<u64> {
    file.seek(SeekFrom::Start(pos))?;
    let mut counter = Counting { file, count: 0 };
    write(&mut counter)?;
    Ok(counter.count)
}

/// Writes the index of the long term memory: the hash of a feature and the
/// offset of its newest record.
fn writeIndex(file: &mut File, pos: u64, index: &std::collections::HashMap<u64, u64>) -> std::io::Result<u64> {
    let mut list: Vec<(u64, u64)> = index.iter().map(|(h, o)| (*h, *o)).collect();
    list.sort_by(|a, b| a.0.cmp(&b.0));
    let len = (list.len() as u64) * 16;
    file.seek(SeekFrom::Start(pos))?;
    {
        let mut out = BufWriter::with_capacity(1 << 20, &mut *file);
        for (hash, offset) in list {
            writeU64(&mut out, hash)?;
            writeU64(&mut out, offset)?;
        }
        out.flush()?;
    }
    Ok(len)
}

/// Reads the mind back from disk.
pub fn load(
    path: &Path,
    sub: &mut Subconscious,
    cons: &mut Consciousness,
    store: &SharedStore,
) -> std::io::Result<Checkpoint> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "not a mind checkpoint"));
    }
    let version = readU64(&mut file)?;
    if version == VERSION3 {
        return loadV3(path, sub, cons);
    }
    if version != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("unsupported checkpoint version {}", version)));
    }
    // the header
    let mut head = vec![0u8; HEADER as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut head)?;
    let h = Header::read(&head);

    let mut store = store.borrow_mut();
    store.records.setLen(h.recordsLen);
    store.records.rows.clear();
    store.records.edges.clear();

    // the parts of the file follow the area of the records
    let mut pos = HEADER + h.recordsLen;
    readIndex(&mut file, pos, h.rowsIndexLen, &mut store.records.rows)?;
    pos += h.rowsIndexLen;
    readIndex(&mut file, pos, h.edgesIndexLen, &mut store.records.edges)?;
    pos += h.edgesIndexLen;

    file.seek(SeekFrom::Start(pos))?;
    sub.load(&mut file)?;
    pos += h.subWindowLen;
    file.seek(SeekFrom::Start(pos))?;
    cons.load(&mut file)?;

    sub.setCounters(h.passiveWrites, h.consciousWrites, h.subEvicted, h.subRaised);
    cons.setCounters(h.consActions, h.consFailed, h.consEvicted, h.consRaised);

    let mut cp = Checkpoint::default();
    cp.dialogsDone = h.dialogsDone;
    cp.dialogsSeen = h.dialogsSeen;
    cp.train = h.train;
    cp.holdout = h.holdout;
    Ok(cp)
}

/// Reads the index of the long term memory.
fn readIndex(
    file: &mut File,
    pos: u64,
    len: u64,
    index: &mut std::collections::HashMap<u64, u64>,
) -> std::io::Result<()> {
    if len == 0 { return Ok(()); }
    file.seek(SeekFrom::Start(pos))?;
    let mut input = BufReader::with_capacity(1 << 20, &mut *file);
    for _ in 0..len / 16 {
        let hash = readU64(&mut input)?;
        let offset = readU64(&mut input)?;
        index.insert(hash, offset);
    }
    Ok(())
}

/// Reads the old checkpoint format: the whole mind in a row.
fn loadV3(path: &Path, sub: &mut Subconscious, cons: &mut Consciousness) -> std::io::Result<Checkpoint> {
    let file = File::open(path)?;
    let mut input = BufReader::with_capacity(1 << 20, file);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    let version = readU64(&mut input)?;
    if version != VERSION3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("unsupported checkpoint version {}", version)));
    }
    let mut cp = Checkpoint::default();
    cp.dialogsDone = readU64(&mut input)?;
    cp.dialogsSeen = readU64(&mut input)?;
    cp.train = loadStats(&mut input)?;
    cp.holdout = loadStats(&mut input)?;
    sub.load(&mut input)?;
    cons.load(&mut input)?;
    Ok(cp)
}
