use std::borrow::Borrow;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub const MAXBAG: usize = 6;
/// Longest turn kept from the noisy support logs.
pub const MAXTURNTOKENS: usize = 40;

const STOP: &[&str] = &[
    "the", "and", "you", "that", "have", "for", "not", "with", "are", "but", "was", "were", "this",
    "they", "them", "their", "there", "here", "what", "when", "where", "which", "who", "why", "how",
    "all", "any", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "did",
    "does", "doing", "has", "had", "been", "being", "into", "onto", "from", "than", "then", "too",
    "just", "get", "got", "goes", "going", "went", "say", "says", "said", "know", "knows", "think",
    "thinks", "thought", "want", "wants", "wanted", "need", "needs", "needed", "make", "makes",
    "made", "take", "takes", "took", "come", "comes", "came", "give", "gives", "gave", "really",
    "something", "someone", "thing", "things", "lot", "much", "more", "most", "other", "others",
    "such", "only", "also", "because", "about", "after", "before", "while", "both", "each", "few",
    "many", "some", "own", "same", "down", "over", "under", "again", "cannot", "let's", "gonna",
    "gotta", "yeah", "okay", "yes", "hello", "hi", "like", "well", "right", "sure", "never", "ever",
];

/// Dialogue tokenizer: keeps punctuation as separate tokens so the model can
/// learn where a reply ends (".", "?", "!").
pub fn tokenizeDialogue(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.to_lowercase().chars() {
        if c.is_alphanumeric() || c == '\'' || c == '-' { cur.push(c); }
        else {
            if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            if c == '.' || c == ',' || c == '?' || c == '!' || c == ';' || c == ':' { out.push(c.to_string()); }
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

pub struct Dialogue { pub turns: Vec<Vec<String>> }

/// Document frequency over prompts; used to keep only discriminative words of
/// the user message when it is used as conditioning context.
pub struct PromptModel { df: HashMap<String, u32>, n: u32 }

impl PromptModel {
    pub fn build<I, D>(dialogs: I) -> PromptModel
    where I: Iterator<Item = D>, D: Borrow<Dialogue> {
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut n = 0u32;
        for d in dialogs {
            let d = d.borrow();
            for i in 1..d.turns.len() {
                let mut seen = HashSet::new();
                for w in contentWords(&d.turns[i - 1]) {
                    if seen.insert(w.clone()) { *df.entry(w).or_insert(0) += 1; }
                }
                n += 1;
            }
        }
        PromptModel { df, n }
    }

    pub fn bag(&self, tokens: &[String]) -> Vec<String> {
        let mut scored: Vec<(f64, String)> = Vec::new();
        for w in contentWords(tokens) {
            let d = *self.df.get(&w).unwrap_or(&0);
            if d == 0 { continue; }
            let idf = (self.n as f64 + 1.0).ln() / (d as f64 + 1.0).ln();
            scored.push((idf, w));
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        scored.into_iter().take(MAXBAG).map(|(_, w)| w).collect()
    }
}

pub fn contentWords(tokens: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in tokens {
        if t.len() < 3 { continue; }
        if !t.chars().all(|c| c.is_alphabetic()) { continue; }
        if STOP.contains(&t.as_str()) { continue; }
        if seen.insert(t.clone()) { out.push(t.clone()); }
    }
    out
}

/// One Ubuntu IRC support log: `timestamp \t nick \t target \t message`.
/// The files live in `dialogs/<group>/<file>.tsv`, there are about 1.8
/// million of them, so they are read one by one and never kept in memory.
pub fn readUbuntuFile(path: &Path) -> Option<Dialogue> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::with_capacity(1 << 16, file);
    let mut turns: Vec<Vec<String>> = Vec::new();
    let mut currentKey: Option<(String, String)> = None;
    let mut current: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.is_empty() { continue; }
        let mut parts = line.splitn(4, '\t');
        let _stamp = parts.next().unwrap_or("");
        let nick = parts.next().unwrap_or("").trim().to_string();
        let target = parts.next().unwrap_or("").trim().to_string();
        let message = parts.next().unwrap_or("");
        if nick.is_empty() && target.is_empty() { continue; }
        let key = (nick, target);
        if currentKey.as_ref() != Some(&key) {
            if !current.is_empty() { turns.push(std::mem::take(&mut current)); }
            currentKey = Some(key);
        }
        let tokens = tokenizeDialogue(message);
        if tokens.is_empty() { continue; }
        for t in tokens {
            if current.len() >= MAXTURNTOKENS { break; }
            current.push(t);
        }
    }
    if !current.is_empty() { turns.push(current); }
    if turns.len() < 2 { return None; }
    Some(Dialogue { turns })
}

/// Shared progress counters of a streamed corpus.
#[derive(Clone)]
pub struct CorpusStatus {
    pub files: Rc<Cell<u64>>,
    pub totalFiles: Rc<Cell<u64>>,
    pub turns: Rc<Cell<u64>>,
    /// Files that were already done when this chunk started.
    pub chunkStart: Rc<Cell<u64>>,
}

impl CorpusStatus {
    pub fn new() -> CorpusStatus {
        CorpusStatus {
            files: Rc::new(Cell::new(0)),
            totalFiles: Rc::new(Cell::new(0)),
            turns: Rc::new(Cell::new(0)),
            chunkStart: Rc::new(Cell::new(0)),
        }
    }
}

/// Counts the files of the whole tree without reading them (fast enough for
/// 1.8 million entries and it gives an exact ETA).
pub fn countFiles(root: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }
            if let Ok(files) = fs::read_dir(&path) {
                for file in files.flatten() {
                    if file.path().extension().map(|e| e == "tsv").unwrap_or(false) { total += 1; }
                }
            }
        }
    }
    total
}

/// Streaming reader over the whole Ubuntu support log tree.
pub struct UbuntuDialogs {
    dirs: Vec<PathBuf>,
    dirIdx: usize,
    files: Vec<PathBuf>,
    fileIdx: usize,
    pub read: u64,
    pub turns: u64,
    pub status: CorpusStatus,
    limit: u64,
    skip: u64,
}

impl UbuntuDialogs {
    pub fn new(root: &Path) -> UbuntuDialogs {
        UbuntuDialogs::withLimit(root, 0, 0)
    }

    /// `limit` caps the number of files (0 = the whole corpus),
    /// `skip` ignores the files that were already read in an earlier chunk.
    pub fn withLimit(root: &Path, limit: u64, skip: u64) -> UbuntuDialogs {
        let mut dirs: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() { dirs.push(path); }
            }
        }
        dirs.sort();
        let status = CorpusStatus::new();
        let total = countFiles(root);
        status.totalFiles.set(if limit > 0 { limit } else { total });
        UbuntuDialogs { dirs, dirIdx: 0, files: Vec::new(), fileIdx: 0, read: 0, turns: 0, status, limit, skip }
    }

    fn loadDir(&mut self) -> bool {
        while self.dirIdx < self.dirs.len() {
            let dir = &self.dirs[self.dirIdx];
            self.dirIdx += 1;
            let mut files: Vec<PathBuf> = Vec::new();
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map(|e| e == "tsv").unwrap_or(false) { files.push(path); }
                }
            }
            if files.is_empty() { continue; }
            files.sort();
            self.files = files;
            self.fileIdx = 0;
            return true;
        }
        self.files.clear();
        self.fileIdx = 0;
        false
    }
}

impl Iterator for UbuntuDialogs {
    type Item = Dialogue;
    fn next(&mut self) -> Option<Dialogue> {
        loop {
            if self.limit > 0 && self.read >= self.limit { return None; }
            if self.fileIdx >= self.files.len() {
                if !self.loadDir() { return None; }
            }
            let path = self.files[self.fileIdx].clone();
            self.fileIdx += 1;
            self.read += 1;
            if self.skip > 0 {
                self.skip -= 1;
                self.status.files.set(self.read);
                continue;
            }
            if let Some(d) = readUbuntuFile(&path) {
                self.turns += d.turns.len() as u64;
                self.status.files.set(self.read);
                self.status.turns.set(self.turns);
                return Some(d);
            }
        }
    }
}

/// Splits a stream into a training part and a small held out part.
pub struct SplitDialogs<I> {
    pub inner: I,
    pub counter: u64,
    holdEvery: u64,
    holdCap: usize,
    pub held: Vec<Dialogue>,
}

impl<I> SplitDialogs<I> {
    /// Continues an interrupted run: the holdout pattern stays the same.
    pub fn at(inner: I, counter: u64, holdEvery: u64, holdCap: usize) -> SplitDialogs<I> {
        SplitDialogs { inner, counter, holdEvery: holdEvery.max(1), holdCap, held: Vec::new() }
    }
}

impl<I: Iterator<Item = Dialogue>> Iterator for SplitDialogs<I> {
    type Item = Dialogue;
    fn next(&mut self) -> Option<Dialogue> {
        loop {
            let d = self.inner.next()?;
            self.counter += 1;
            if self.counter % self.holdEvery == 0 {
                if self.held.len() < self.holdCap { self.held.push(d); }
                continue;
            }
            return Some(d);
        }
    }
}
