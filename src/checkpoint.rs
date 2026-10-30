//! Checkpoints of the mind: the whole state (subconscious rows, conscious
//! edges, the act classifier) plus the position in the streamed corpus, so a
//! long run can be split into chunks and continued later.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::binio::{readU64, writeU64};
use crate::dialogue::ReplyStats;
use crate::mind::{Consciousness, Subconscious};

const MAGIC: &[u8; 8] = b"MINDCKPT";
const VERSION: u64 = 2;

/// Everything that has to survive between two chunks.
pub struct Checkpoint {
    /// Files of the streamed corpus that were already read.
    pub filesDone: u64,
    /// Dialogues of the streamed corpus that were already seen.
    pub dialoguesDone: u64,
    /// Training statistics of the corpus so far.
    pub train: ReplyStats,
    /// Statistics of the held out part so far.
    pub holdout: ReplyStats,
}

impl Default for Checkpoint {
    fn default() -> Checkpoint {
        Checkpoint {
            filesDone: 0,
            dialoguesDone: 0,
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

/// Writes the whole mind to disk.
pub fn save(
    path: &Path,
    sub: &Subconscious,
    cons: &Consciousness,
    cp: &Checkpoint,
) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() { std::fs::create_dir_all(dir)?; }
    }
    let file = File::create(path)?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(MAGIC)?;
    writeU64(&mut out, VERSION)?;
    writeU64(&mut out, cp.filesDone)?;
    writeU64(&mut out, cp.dialoguesDone)?;
    saveStats(&mut out, &cp.train)?;
    saveStats(&mut out, &cp.holdout)?;
    sub.save(&mut out)?;
    cons.save(&mut out)?;
    out.flush()
}

/// Reads the mind back from disk.
pub fn load(
    path: &Path,
    sub: &mut Subconscious,
    cons: &mut Consciousness,
) -> std::io::Result<Checkpoint> {
    let file = File::open(path)?;
    let mut input = BufReader::with_capacity(1 << 20, file);
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, "not a mind checkpoint"));
    }
    let version = readU64(&mut input)?;
    if version != VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("unsupported checkpoint version {}", version)));
    }
    let mut cp = Checkpoint::default();
    cp.filesDone = readU64(&mut input)?;
    cp.dialoguesDone = readU64(&mut input)?;
    cp.train = loadStats(&mut input)?;
    cp.holdout = loadStats(&mut input)?;
    sub.load(&mut input)?;
    cons.load(&mut input)?;
    Ok(cp)
}
